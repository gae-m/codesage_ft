import inspect
import unittest
import torch
import torch.nn as nn
from torch.utils.checkpoint import checkpoint
from transformers import AutoConfig, AutoModel
from peft import LoraConfig, get_peft_model


def enable_codesage_block_checkpointing(model: nn.Module) -> int:
    """Safely enables non-reentrant gradient checkpointing on all `CodeSageBlock` layers."""
    # Disable use_cache on underlying model configs (handles both raw AutoModel and PeftModel)
    for mod in model.modules():
        cfg = getattr(mod, "config", None)
        if cfg is not None and hasattr(cfg, "use_cache"):
            cfg.use_cache = False

    wrapped_count = 0
    for module in model.modules():
        if module.__class__.__name__ != "CodeSageBlock":
            continue
        if getattr(module, "_manual_gc_enabled", False):
            wrapped_count += 1
            continue

        orig_forward = module.forward
        sig = inspect.signature(orig_forward)

        def make_checkpointed_forward(fn, signature, blk):
            def checkpointed_forward(*args, **kwargs):
                if not blk.training or not torch.is_grad_enabled():
                    return fn(*args, **kwargs)
                bound = signature.bind(*args, **kwargs)
                bound.apply_defaults()
                return checkpoint(fn, *bound.args, use_reentrant=False)

            return checkpointed_forward

        module._orig_forward = orig_forward
        module.forward = make_checkpointed_forward(orig_forward, sig, module)
        module._manual_gc_enabled = True
        wrapped_count += 1

    if wrapped_count == 0:
        raise RuntimeError("No `CodeSageBlock` modules found in model.")
    return wrapped_count


def disable_codesage_block_checkpointing(model: nn.Module) -> None:
    """Restores original uncheckpointed `forward` methods on all wrapped `CodeSageBlock` layers."""
    for module in model.modules():
        if module.__class__.__name__ == "CodeSageBlock" and getattr(module, "_manual_gc_enabled", False):
            module.forward = module._orig_forward
            delattr(module, "_orig_forward")
            module._manual_gc_enabled = False


@unittest.skipUnless(torch.cuda.is_available(), "CUDA GPU is required for bfloat16 VRAM smoke test.")
class TestCodeSageBlockCheckpointing(unittest.TestCase):
    MODEL_ID = "codesage/codesage-large-v2"
    DEVICE = "cuda"

    @classmethod
    def setUpClass(cls):
        config = AutoConfig.from_pretrained(cls.MODEL_ID, trust_remote_code=True)
        config.attention_dropout_prob = 0.0
        config.residual_dropout_prob = 0.0
        config.embedding_dropout_prob = 0.0
        config.use_cache = True  # Intentionally True initially to verify the wrapper disables it

        base_encoder = AutoModel.from_pretrained(
            cls.MODEL_ID,
            config=config,
            trust_remote_code=True,
            torch_dtype=torch.bfloat16,
        )

        lora_config = LoraConfig(
            r=16,
            lora_alpha=32,
            lora_dropout=0.05,
            target_modules="all-linear",
            fan_in_fan_out=True,
            bias="none",
        )
        cls.lora_encoder = get_peft_model(base_encoder, lora_config).to(cls.DEVICE)

    def setUp(self):
        # Guarantee a clean, uncheckpointed state before every test regardless of execution order
        disable_codesage_block_checkpointing(self.lora_encoder)
        self.lora_encoder.zero_grad(set_to_none=True)
        self.lora_encoder.train()

    def test_01_dropout_configuration(self):
        """Verify base CodeSage dropouts are 0.0 while LoRA dropouts remain 0.05."""
        base_dropouts = []
        lora_dropouts = []

        for name, mod in self.lora_encoder.named_modules():
            if isinstance(mod, nn.Dropout):
                if "lora_dropout" in name:
                    lora_dropouts.append(mod.p)
                else:
                    base_dropouts.append(mod.p)

        self.assertGreater(len(base_dropouts), 0)
        self.assertTrue(all(p == 0.0 for p in base_dropouts), f"Non-zero base dropout: {set(base_dropouts)}")
        self.assertGreater(len(lora_dropouts), 0)
        self.assertTrue(all(p == 0.05 for p in lora_dropouts), f"Unexpected LoRA dropout: {set(lora_dropouts)}")

    def test_02_gradient_equivalence_before_and_after_checkpointing(self):
        """Verify checkpointed gradients match uncheckpointed gradients across all LoRA parameters."""
        dummy_ids = torch.randint(10, 1000, (2, 128), device=self.DEVICE)
        dummy_mask = torch.ones((2, 128), dtype=torch.long, device=self.DEVICE)

        # 1. Baseline uncheckpointed forward/backward
        torch.manual_seed(42)
        torch.cuda.manual_seed_all(42)
        out_unckpt = self.lora_encoder(input_ids=dummy_ids, attention_mask=dummy_mask)
        loss_unckpt = out_unckpt.last_hidden_state[:, 0, :].float().sum()
        loss_unckpt.backward()

        unckpt_grads = {
            name: param.grad.detach().clone()
            for name, param in self.lora_encoder.named_parameters()
            if "lora_" in name and param.grad is not None
        }
        self.assertGreater(len(unckpt_grads), 0)
        self.lora_encoder.zero_grad(set_to_none=True)

        # 2. Enable block checkpointing and verify underlying config.use_cache was disabled
        num_wrapped = enable_codesage_block_checkpointing(self.lora_encoder)
        self.assertEqual(num_wrapped, 24)
        self.assertFalse(self.lora_encoder.get_base_model().config.use_cache)

        torch.manual_seed(42)
        torch.cuda.manual_seed_all(42)
        out_ckpt = self.lora_encoder(input_ids=dummy_ids, attention_mask=dummy_mask)
        loss_ckpt = out_ckpt.last_hidden_state[:, 0, :].float().sum()
        loss_ckpt.backward()

        ckpt_params = dict(self.lora_encoder.named_parameters())
        for name, expected_grad in unckpt_grads.items():
            actual_grad = ckpt_params[name].grad
            self.assertIsNotNone(actual_grad, f"Missing gradient after checkpointing on {name}")
            self.assertFalse(torch.isnan(actual_grad).any(), f"NaN gradient on {name}")
            self.assertTrue(
                torch.allclose(actual_grad, expected_grad, rtol=1e-2, atol=1e-2),
                f"Gradient mismatch on {name}: max diff = {(actual_grad - expected_grad).abs().max().item()}",
            )

    def test_03_full_bag_gradient_flow_and_vram(self):
        """Simulate a full 8-chunk x 512-token bag and verify gradient flow and activation VRAM delta."""
        enable_codesage_block_checkpointing(self.lora_encoder)

        torch.cuda.empty_cache()
        torch.cuda.reset_peak_memory_stats(self.DEVICE)
        baseline_vram_gb = torch.cuda.memory_allocated(self.DEVICE) / (1024 ** 3)

        dummy_ids = torch.randint(10, 1000, (8, 512), device=self.DEVICE)
        dummy_mask = torch.ones((8, 512), dtype=torch.long, device=self.DEVICE)

        outputs = self.lora_encoder(input_ids=dummy_ids, attention_mask=dummy_mask)
        loss = outputs.last_hidden_state[:, 0, :].float().sum()
        loss.backward()

        peak_vram_gb = torch.cuda.max_memory_allocated(self.DEVICE) / (1024 ** 3)

        block0_grads = [
            p.grad.abs().sum().item()
            for n, p in self.lora_encoder.named_parameters()
            if "h.0." in n and "lora_" in n and p.grad is not None
        ]
        block23_grads = [
            p.grad.abs().sum().item()
            for n, p in self.lora_encoder.named_parameters()
            if "h.23." in n and "lora_" in n and p.grad is not None
        ]

        self.assertGreater(len(block0_grads), 0)
        self.assertGreater(sum(block0_grads), 0.0)
        self.assertFalse(torch.isnan(torch.tensor(block0_grads)).any())

        self.assertGreater(len(block23_grads), 0)
        self.assertGreater(sum(block23_grads), 0.0)
        self.assertFalse(torch.isnan(torch.tensor(block23_grads)).any())

        print(
            f"\n[Smoke Test] Total Peak VRAM: {peak_vram_gb:.2f} GB "
            f"(Weights: {baseline_vram_gb:.2f} GB + Activations/Grads: {peak_vram_gb - baseline_vram_gb:.2f} GB)"
        )

    def test_04_eval_mode_bypass(self):
        """Ensure eval() + no_grad() executes cleanly without checkpointing overhead."""
        enable_codesage_block_checkpointing(self.lora_encoder)
        self.lora_encoder.eval()

        dummy_ids = torch.randint(10, 1000, (2, 128), device=self.DEVICE)
        dummy_mask = torch.ones((2, 128), dtype=torch.long, device=self.DEVICE)

        with torch.no_grad():
            outputs = self.lora_encoder(input_ids=dummy_ids, attention_mask=dummy_mask)

        self.assertEqual(tuple(outputs.last_hidden_state.shape), (2, 128, 2048))
        self.assertFalse(torch.isnan(outputs.last_hidden_state).any())


if __name__ == "__main__":
    unittest.main(verbosity=2)