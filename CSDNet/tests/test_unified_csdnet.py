import random

import torch

from CSDNet.model.unified_backbone import UnifiedCSDNetBackbone
from CSDNet.model.unified_corruption import (
    MODE_FRAGMENT,
    MODE_MDM,
    MODE_REFINE,
    MODE_VARIABLE,
    UnifiedCorruptionCollator,
)
from CSDNet.model.unified_lightning_module import UnifiedCSDNetLightningModule
from CSDNet.util.unified_sampling import UnifiedDynamicSampler
from CSDNet.util.checkpoint import load_backbone_from_checkpoint


class _Tokenizer:
    pad_id = 0
    mask_id = 1
    bos_id = 2
    eos_id = 3
    unk_id = 4
    vocab_size = 20
    aromatic_ids = {8, 9}


def _item(body):
    ids = [_Tokenizer.bos_id] + list(body) + [_Tokenizer.eos_id]
    ids += [_Tokenizer.pad_id] * (16 - len(ids))
    aromatic = [False] * 16
    for index, token_id in enumerate(ids):
        aromatic[index] = token_id in _Tokenizer.aromatic_ids
    return {
        "input_ids": torch.tensor(ids, dtype=torch.long),
        "aromatic_mask": torch.tensor(aromatic, dtype=torch.bool),
    }


def test_balanced_collator_contains_all_training_modes():
    collator = UnifiedCorruptionCollator(
        _Tokenizer(), max_len=16, pad_to_multiple_of=1, seed=7
    )
    batch = collator([_item([5, 6, 7, 8, 9, 10])] * 8)
    assert set(batch["mode_ids"].tolist()) == {
        MODE_MDM,
        MODE_VARIABLE,
        MODE_FRAGMENT,
        MODE_REFINE,
    }
    assert batch["input_ids"].shape == batch["token_labels"].shape
    assert batch["input_ids"].shape == batch["gap_labels"].shape
    assert batch["input_ids"].shape == batch["delete_labels"].shape


def test_unknown_fragment_gap_target_uses_atomic_token_count():
    collator = UnifiedCorruptionCollator(
        _Tokenizer(),
        max_len=16,
        max_gap_count=8,
        fragment_unknown_length_probability=1.0,
        fragment_multi_span_probability=0.0,
        fragment_long_span_probability=0.0,
    )
    rng = random.Random(2)
    clean = [2, 5, 6, 7, 8, 9, 10, 3]
    aromatic = [False] * len(clean)
    row = collator._make_fragment(clean, aromatic, rng)
    positive = [label for label in row["gap_labels"] if label > 0]
    assert positive
    assert sum(positive) <= len(clean) - 2
    assert all(label <= collator.max_gap_count for label in positive)


def test_refinement_contains_deletion_and_substitution_supervision():
    collator = UnifiedCorruptionCollator(
        _Tokenizer(), max_len=16, refine_clean_weight=0.1
    )
    rng = random.Random(11)
    clean = [2, 5, 6, 7, 8, 9, 10, 11, 12, 13, 3]
    aromatic = [False] * len(clean)
    rows = [collator._make_refine(clean, aromatic, rng) for _ in range(20)]
    assert any(
        any(state.delete_label == 1.0 for state in row["states"])
        for row in rows
    )
    assert all(
        any(state.target_id != -100 for state in row["states"])
        for row in rows
    )


def test_unified_backbone_shapes():
    model = UnifiedCSDNetBackbone(
        vocab_size=20,
        hidden_size=32,
        num_layers=2,
        num_heads=4,
        intermediate=64,
        pad_token_id=0,
        mask_token_id=1,
        max_position_embeddings=32,
        max_gap_count=4,
        position_embedding_type="rotary",
        gradient_checkpointing=False,
        hidden_dropout_prob=0.0,
        attention_probs_dropout_prob=0.0,
    )
    input_ids = torch.tensor([[2, 1, 8, 3], [2, 6, 3, 0]])
    attention = input_ids.ne(0).long()
    output = model(
        input_ids,
        attention,
        corruption_level=torch.tensor([0.8, 0.2]),
        return_aux=True,
    )
    assert output["logits"].shape == (2, 4, 20)
    assert output["gap_logits"].shape == (2, 4, 5)
    assert output["delete_logits"].shape == (2, 4)
    assert output["confidence_logits"].shape == (2, 4)


def test_joint_loss_backward_uses_auxiliary_heads():
    collator = UnifiedCorruptionCollator(
        _Tokenizer(),
        max_len=16,
        max_gap_count=4,
        pad_to_multiple_of=1,
        seed=13,
    )
    batch = collator([_item([5, 6, 7, 8, 9, 10])] * 8)
    module = UnifiedCSDNetLightningModule(
        vocab_size=20,
        pad_id=0,
        mask_id=1,
        bos_id=2,
        eos_id=3,
        unk_id=4,
        aromatic_ids={8, 9},
        hidden_size=32,
        num_layers=2,
        num_heads=4,
        intermediate=64,
        max_position_embeddings=32,
        max_gap_count=4,
        position_embedding_type="rotary",
        gradient_checkpointing=False,
        hidden_dropout_prob=0.0,
        attention_probs_dropout_prob=0.0,
        use_ema=False,
    )
    loss, metrics = module._compute_losses(batch)
    assert torch.isfinite(loss)
    loss.backward()
    assert module.backbone.gap_head[-1].weight.grad is not None
    assert module.backbone.delete_head[-1].weight.grad is not None
    assert module.backbone.confidence_head[-1].weight.grad is not None
    assert set(name + "_loss" for name in ("mdm", "variable", "fragment", "refine")) <= set(metrics)


class _ScriptedBackbone(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.anchor = torch.nn.Parameter(torch.zeros(()))
        self.calls = 0

    def forward(self, input_ids, attention_mask, corruption_level, return_aux):
        self.calls += 1
        batch, length = input_ids.shape
        logits = torch.full((batch, length, 20), -8.0, device=input_ids.device)
        logits[..., 5] = 8.0
        gap_logits = torch.full((batch, length, 3), -8.0, device=input_ids.device)
        gap_logits[..., 0] = 8.0
        if self.calls == 1:
            gap_logits[:, 0, 0] = -8.0
            gap_logits[:, 0, 2] = 8.0
        return {
            "logits": logits,
            "gap_logits": gap_logits,
            "delete_logits": torch.full((batch, length), -8.0, device=input_ids.device),
            "confidence_logits": torch.full((batch, length), 8.0, device=input_ids.device),
        }


def test_dynamic_sampler_can_insert_and_fill_tokens():
    sampler = UnifiedDynamicSampler(
        _ScriptedBackbone(),
        pad_id=0,
        mask_id=1,
        bos_id=2,
        eos_id=3,
        valid_token_ids=range(5, 20),
        max_len=12,
        max_gap_count=2,
    )
    result = sampler.sample_de_novo(
        batch_size=1, num_steps=4, stochastic=False
    )[0]
    assert result.tolist() == [2, 5, 5, 3]


def test_checkpoint_loader_recovers_unified_architecture(tmp_path):
    module = UnifiedCSDNetLightningModule(
        vocab_size=20,
        pad_id=0,
        mask_id=1,
        bos_id=2,
        eos_id=3,
        unk_id=4,
        aromatic_ids={8, 9},
        hidden_size=32,
        num_layers=2,
        num_heads=4,
        intermediate=64,
        max_position_embeddings=32,
        max_gap_count=4,
        position_embedding_type="rotary",
        gradient_checkpointing=False,
        use_ema=False,
    )
    path = tmp_path / "unified.ckpt"
    torch.save(
        {
            "state_dict": module.state_dict(),
            "hyper_parameters": dict(module.hparams),
        },
        path,
    )
    loaded = load_backbone_from_checkpoint(
        str(path), _Tokenizer(), torch.device("cpu"), use_ema=False
    )
    assert isinstance(loaded, UnifiedCSDNetBackbone)
    assert loaded.max_gap_count == 4
