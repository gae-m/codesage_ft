import unittest
from types import SimpleNamespace
from tempfile import TemporaryDirectory
import json
from pathlib import Path

import torch

from classifier_wrapper import AttentionMILCodeSage
from collator import AttentionMILCollator
from finetune import compute_auroc_metrics, load_data


class FakeEncoder(torch.nn.Module):
    def __init__(self, hidden_size=4):
        super().__init__()
        self.hidden_size = hidden_size
        self.gradient_checkpointing_calls = []

    def forward(self, input_ids, attention_mask):
        embeddings = input_ids.float().unsqueeze(-1).expand(
            -1, -1, self.hidden_size
        )
        return SimpleNamespace(last_hidden_state=embeddings)

    def gradient_checkpointing_enable(self, gradient_checkpointing_kwargs=None):
        self.gradient_checkpointing_calls.append(
            ("enable", gradient_checkpointing_kwargs)
        )

    def gradient_checkpointing_disable(self):
        self.gradient_checkpointing_calls.append(("disable", None))


class FakeTokenizer:
    pad_token_id = 0

    def __call__(
        self,
        code,
        truncation,
        max_length,
        stride,
        return_overflowing_tokens,
        return_offsets_mapping,
        padding,
    ):
        return {
            "input_ids": [[1] * max_length, [2] * max_length],
            "attention_mask": [[1] * max_length, [1] * max_length],
        }


class AttentionMILSmokeTest(unittest.TestCase):
    def test_dataset_mapping(self):
        with TemporaryDirectory() as directory:
            data_file = Path(directory) / "sample.jsonl"
            data_file.write_text(
                json.dumps({"func": "int f() {}", "target": 1, "idx": 1}) + "\n",
                encoding="utf-8",
            )

            dataset = load_data(
                {"train": str(data_file)},
                FakeTokenizer(),
            )

        self.assertEqual(dataset["train"][0]["label"], 1)
        self.assertEqual(len(dataset["train"][0]["input_ids"]), 2)

    def test_collator_forward_loss_and_metrics(self):
        collator = AttentionMILCollator(pad_token_id=0)
        batch = collator(
            [
                {
                    "input_ids": [[1, 2, 3], [4, 5, 6]],
                    "attention_mask": [[1, 1, 1], [1, 1, 1]],
                    "label": 0,
                },
                {
                    "input_ids": [[7, 8, 9]],
                    "attention_mask": [[1, 1, 1]],
                    "label": 1,
                },
            ]
        )

        model = AttentionMILCodeSage(FakeEncoder(), hidden_size=4)
        output = model(**batch)

        self.assertEqual(tuple(output["logits"].shape), (2, 1))
        self.assertEqual(tuple(output["attention_weights"].shape), (2, 2))
        self.assertTrue(torch.isfinite(output["loss"]))
        self.assertAlmostEqual(
            output["attention_weights"][1, 1].item(), 0.0, places=6
        )

        metrics = compute_auroc_metrics(
            (torch.tensor([[-2.0], [2.0]]), torch.tensor([0, 1]))
        )
        self.assertAlmostEqual(metrics["val/auroc"], 1.0)

        metrics = compute_auroc_metrics(
            (
                (
                    torch.tensor([[-2.0], [2.0]]),
                    torch.zeros(2, 2),
                ),
                torch.tensor([0, 1]),
            )
        )
        self.assertAlmostEqual(metrics["val/auroc"], 1.0)

    def test_gradient_checkpointing_is_delegated(self):
        encoder = FakeEncoder()
        model = AttentionMILCodeSage(encoder, hidden_size=4)

        model.gradient_checkpointing_enable({"use_reentrant": False})
        model.gradient_checkpointing_disable()

        self.assertEqual(
            encoder.gradient_checkpointing_calls,
            [
                ("enable", {"use_reentrant": False}),
                ("disable", None),
            ],
        )


if __name__ == "__main__":
    unittest.main()
