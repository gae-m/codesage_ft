import argparse
import os
import torch
from safetensors import safe_open
from safetensors.torch import load_file


def inspect_mil_checkpoint(ckpt_dir: str, prev_ckpt_dir: str | None = None):
    safetensors_path = os.path.join(ckpt_dir, "model.safetensors")
    if not os.path.exists(safetensors_path):
        raise FileNotFoundError(
            f"`{safetensors_path}` not found. Check for sharded `.safetensors` or `pytorch_model.bin`."
        )

    prev_f = (
        safe_open(os.path.join(prev_ckpt_dir, "model.safetensors"), framework="pt", device="cpu")
        if prev_ckpt_dir
        else None
    )

    with safe_open(safetensors_path, framework="pt", device="cpu") as f:
        keys = list(f.keys())
        print(f"Total tensors: {len(keys)} | `_orig_mod.` prefix: {any(k.startswith('_orig_mod.') for k in keys)}")

        # 1. Native-dtype NaN/Inf audit (no `.float()` copy needed)
        corrupted = [k for k in keys if not torch.isfinite(f.get_tensor(k)).all()]
        print(f"Corrupted (NaN/Inf) tensors: {len(corrupted)}")
        if corrupted:
            print("  Corrupted keys:", corrupted[:10])

        # 2. Trainable layer stats (uses `correction=0` so 1-element bias tensors don't print `nan`)
        print("\n--- Trainable Layer Diagnostics ---")
        for k in keys:
            if any(tag in k for tag in ("lora_A", "lora_B", "attention.", "classifier.")):
                t = f.get_tensor(k)
                tf = t.float()
                delta_str = ""
                if prev_f is not None and k in prev_f.keys():
                    delta = torch.norm(tf - prev_f.get_tensor(k).float()).item()
                    delta_str = f" | l2_delta={delta:.6f}"

                print(
                    f"{k:<68} | shape={str(tuple(t.shape)):<14} | dtype={str(t.dtype):<14} "
                    f"| abs_max={tf.abs().max().item():.6f} | std={tf.std().item():.6f}{delta_str}"
                )


def load_mil_checkpoint_for_debug(uncompiled_model: torch.nn.Module, ckpt_dir: str, device: str = "cuda"):
    raw_sd = load_file(os.path.join(ckpt_dir, "model.safetensors"), device="cpu")
    clean_sd = {k.removeprefix("_orig_mod."): v for k, v in raw_sd.items()}
    uncompiled_model.load_state_dict(clean_sd, strict=True)
    uncompiled_model.to(device).eval()
    return uncompiled_model


def main():
    parser = argparse.ArgumentParser(description="Inspect a CodeSage MIL checkpoint.")
    parser.add_argument("ckpt_dir", type=str, help="Path to the checkpoint directory.")

    agrs = parser.parse_args()

    inspect_mil_checkpoint(ckpt_dir=agrs.ckpt_dir)


if __name__ == "__main__":
    main()