import random
import tempfile
import unittest

import torch

from CSDNet.model.edit_scheduler import (
    EditScheduleNet,
    load_edit_scheduler_checkpoint,
)
from CSDNet.model.edit_scheduler_data import (
    EditScheduleCorruptionCollator,
)
from CSDNet.model.edit_scheduler_lightning import (
    EditSchedulerLightningModule,
)
from CSDNet.util.edit_schedule_sampling import (
    sample_de_novo_lengths,
    schedule_replacement_lengths,
)


class _Tokenizer:
    pad_id = 0
    bos_id = 1
    eos_id = 2
    mask_id = 3


class _Teacher(torch.nn.Module):
    def __init__(self, vocab_size):
        super().__init__()
        self.embedding = torch.nn.Embedding(vocab_size, vocab_size)

    def forward(self, input_ids, attention_mask):
        del attention_mask
        return self.embedding(input_ids)


class EditSchedulerTest(unittest.TestCase):
    def setUp(self):
        torch.manual_seed(7)
        random.seed(7)
        self.tokenizer = _Tokenizer()
        self.model = EditScheduleNet(
            vocab_size=24,
            pad_token_id=0,
            mask_token_id=3,
            hidden_size=32,
            num_layers=2,
            num_heads=4,
            intermediate=64,
            max_position_embeddings=32,
            max_replacement_length=8,
            dropout=0.0,
        )

    def _batch(self):
        collator = EditScheduleCorruptionCollator(
            tokenizer=self.tokenizer,
            max_len=32,
            max_gaps=3,
            max_span_length=8,
            max_replacement_length=8,
            zero_gap_probability=1.0,
        )
        rows = [
            {
                "input_ids": torch.tensor(
                    [1, 4, 5, 6, 7, 8, 9, 10, 2]
                    + [0] * 23,
                )
            },
            {
                "input_ids": torch.tensor(
                    [1, 11, 12, 13, 14, 15, 16, 2]
                    + [0] * 24,
                )
            },
        ]
        return collator(rows)

    def test_corruption_has_positive_and_zero_gaps(self):
        batch = self._batch()
        labels = batch["length_labels"][batch["gap_mask"]]
        self.assertTrue(labels.gt(0).any())
        self.assertTrue(labels.eq(0).any())
        self.assertTrue(
            batch["teacher_labels"].ne(-100).any()
        )
        self.assertLessEqual(batch["input_ids"].size(1), 32)

    def test_forward_shapes_and_positive_rates(self):
        batch = self._batch()
        output = self.model(
            input_ids=batch["input_ids"],
            attention_mask=batch["attention_mask"],
            gap_mask=batch["gap_mask"],
            removed_lengths=batch["removed_lengths"],
            corruption_fraction=batch["corruption_fraction"],
        )
        self.assertEqual(
            output["length_logits"].shape,
            (*batch["input_ids"].shape, 9),
        )
        self.assertTrue(
            output["insertion_rate"][batch["gap_mask"]].gt(0).all()
        )
        self.assertTrue(
            torch.isneginf(
                output["order_logits"][~batch["gap_mask"]]
            ).all()
        )

    def test_training_loss_is_finite_and_teacher_is_frozen(self):
        batch = self._batch()
        teacher = _Teacher(vocab_size=24)
        teacher.requires_grad_(False)
        module = EditSchedulerLightningModule(
            scheduler=self.model,
            teacher=teacher,
            teacher_checkpoint="dummy.ckpt",
        )
        metrics = module._step(batch)
        self.assertTrue(torch.isfinite(metrics["loss"]))
        metrics["loss"].backward()
        self.assertTrue(
            any(
                parameter.grad is not None
                for parameter in self.model.parameters()
            )
        )
        self.assertTrue(
            all(
                parameter.grad is None
                for parameter in teacher.parameters()
            )
        )

    def test_checkpoint_loader_ignores_teacher(self):
        checkpoint = {
            "hyper_parameters": {
                "scheduler_config": self.model.config_dict(),
            },
            "state_dict": {
                f"scheduler.{name}": value
                for name, value in self.model.state_dict().items()
            },
        }
        with tempfile.NamedTemporaryFile(suffix=".ckpt") as handle:
            torch.save(checkpoint, handle.name)
            restored = load_edit_scheduler_checkpoint(handle.name)
        for expected, actual in zip(
            self.model.parameters(),
            restored.parameters(),
        ):
            self.assertTrue(torch.equal(expected, actual))

    def test_unconditional_gap_teaches_full_de_novo_body_length(self):
        collator = EditScheduleCorruptionCollator(
            tokenizer=self.tokenizer,
            max_len=32,
            max_gaps=1,
            max_span_length=8,
            max_replacement_length=16,
            zero_gap_probability=0.0,
            unconditional_gap_probability=1.0,
        )
        batch = collator(
            [{"input_ids": torch.tensor([1, 4, 5, 6, 7, 2] + [0] * 26)}]
        )
        self.assertEqual(batch["input_ids"][0, :3].tolist(), [1, 3, 2])
        self.assertEqual(batch["length_labels"][0, 1].item(), 4)
        self.assertEqual(batch["removed_lengths"][0, 1].item(), 0)

    def test_learned_lengths_replace_random_de_novo_and_gap_lengths(self):
        tokenizer = _Tokenizer()
        tokenizer.encode = lambda _smiles, max_len: [1, 4, 5, 6, 2] + [0] * (
            max_len - 5
        )
        with torch.no_grad():
            self.model.length_head[-1].weight.zero_()
            self.model.length_head[-1].bias.fill_(-20.0)
            self.model.length_head[-1].bias[4] = 20.0

        lengths = sample_de_novo_lengths(
            self.model,
            tokenizer,
            3,
            max_len=10,
            device=torch.device("cpu"),
            temperature=1.0,
            top_k=1,
        )
        self.assertEqual(lengths, [6, 6, 6])

        plans, diagnostics = schedule_replacement_lengths(
            self.model,
            tokenizer,
            ["CCC"],
            [[{"start": 1, "stop": 2, "replacement_len": 1}]],
            max_len=10,
            device=torch.device("cpu"),
            temperature=1.0,
            top_k=1,
        )
        self.assertEqual(plans[0][0]["replacement_len"], 4)
        self.assertEqual(diagnostics[0]["target_body_length"], 6)


if __name__ == "__main__":
    unittest.main()
