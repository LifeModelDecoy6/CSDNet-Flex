import torch

from CSDNet.exp.denovo.sampler_profiles import SAMPLER_PROFILES
from CSDNet.exp.denovo.train_refinement import trajectory_profile_kwargs
from CSDNet.model.backbone import CSDNetBackbone
from CSDNet.model.lightning_module import CSDNetLightningModule


def test_zero_initialized_corruption_condition_preserves_initial_logits():
    model = CSDNetBackbone(
        vocab_size=8,
        cond_dim=0,
        hidden_size=32,
        num_layers=1,
        num_heads=4,
        intermediate=64,
        pad_token_id=0,
        mask_token_id=1,
        max_position_embeddings=16,
        hidden_dropout_prob=0.0,
        attention_probs_dropout_prob=0.0,
        corruption_level_conditioning=True,
    ).eval()
    tokens = torch.tensor([[2, 1, 5, 3]])
    attention = torch.ones_like(tokens)

    low = model(tokens, attention, corruption_level=torch.tensor([0.1]))
    high = model(tokens, attention, corruption_level=torch.tensor([0.9]))

    assert torch.equal(low, high)


def test_mixed_refinement_objective_is_finite_and_differentiable(monkeypatch):
    class DummyMDLM:
        def sample_time(self, batch_size):
            return torch.full((batch_size,), 0.5)

        def forward_process(self, input_ids, _time):
            corrupted = input_ids.clone()
            corrupted[:, 1] = 1
            return corrupted

    monkeypatch.setattr(
        "CSDNet.model.lightning_module.build_mdlm",
        lambda _vocab_size, _mask_id: DummyMDLM(),
    )
    model = CSDNetLightningModule(
        vocab_size=8,
        pad_id=0,
        mask_id=1,
        bos_id=2,
        eos_id=3,
        unk_id=4,
        scaffold_ids=frozenset(),
        aromatic_ids=frozenset(),
        cond_dim=0,
        use_ema=False,
        use_cbi=False,
        hidden_size=32,
        num_layers=1,
        num_heads=4,
        intermediate=64,
        max_position_embeddings=16,
        hidden_dropout_prob=0.0,
        attention_probs_dropout_prob=0.0,
        corruption_level_conditioning=True,
        refinement_loss_weight=0.5,
        refinement_corruption_min=0.25,
        refinement_corruption_max=0.50,
        refinement_clean_weight=0.2,
    )
    batch = {
        "input_ids": torch.tensor(
            [
                [2, 5, 6, 3, 0],
                [2, 6, 7, 3, 0],
                [2, 5, 7, 6, 3],
                [2, 7, 5, 3, 0],
            ]
        ),
        "attention_mask": torch.tensor(
            [
                [1, 1, 1, 1, 0],
                [1, 1, 1, 1, 0],
                [1, 1, 1, 1, 1],
                [1, 1, 1, 1, 0],
            ]
        ),
    }

    loss = model._step(batch, is_train=True)
    loss.backward()

    assert torch.isfinite(loss)
    assert model._last_refinement_metrics["refinement_loss"].item() > 0.0
    assert model.backbone.corruption_level_embedding.proj[-1].weight.grad is not None


def test_three_way_objective_is_equal_and_fragment_conditioned(monkeypatch):
    class DummyMDLM:
        def sample_time(self, batch_size):
            return torch.full((batch_size,), 0.5)

        def forward_process(self, input_ids, _time):
            corrupted = input_ids.clone()
            corrupted[:, 1] = 1
            return corrupted

    monkeypatch.setattr(
        "CSDNet.model.lightning_module.build_mdlm",
        lambda _vocab_size, _mask_id: DummyMDLM(),
    )
    model = CSDNetLightningModule(
        vocab_size=9,
        pad_id=0,
        mask_id=1,
        bos_id=2,
        eos_id=3,
        unk_id=4,
        scaffold_ids=frozenset(),
        aromatic_ids=frozenset(),
        cond_dim=0,
        use_ema=False,
        use_cbi=False,
        hidden_size=32,
        num_layers=1,
        num_heads=4,
        intermediate=64,
        max_position_embeddings=16,
        hidden_dropout_prob=0.0,
        attention_probs_dropout_prob=0.0,
        corruption_level_conditioning=True,
        refinement_loss_weight=1.0 / 3.0,
        refinement_corruption_min=0.25,
        refinement_corruption_max=0.50,
        refinement_mask_fraction=1.0,
        refinement_corruption_mode="uniform",
        training_objective_mode="three_way_equal",
        fragment_span_min=1,
        fragment_span_max=3,
    )
    batch = {
        "input_ids": torch.tensor(
            [
                [2, 5, 6, 7, 3, 0],
                [2, 6, 7, 8, 3, 0],
                [2, 5, 8, 6, 3, 0],
                [2, 7, 5, 8, 3, 0],
                [2, 8, 6, 5, 3, 0],
                [2, 6, 5, 7, 3, 0],
            ]
        ),
        "attention_mask": torch.tensor([[1, 1, 1, 1, 1, 0]] * 6),
    }

    loss = model._step(batch, is_train=True)
    loss.backward()
    metrics = model._last_refinement_metrics

    assert torch.isfinite(loss)
    assert metrics["mask_loss"].item() > 0.0
    assert metrics["refinement_loss"].item() > 0.0
    assert metrics["fragment_loss"].item() > 0.0
    expected = (
        metrics["mask_loss"]
        + metrics["refinement_loss"]
        + metrics["fragment_loss"]
    ) / 3.0
    assert torch.allclose(loss.detach(), expected)
    assert metrics["fragment_corruption_rate"].item() > 0.0


def test_trajectory_refinement_uses_model_generated_drafts(monkeypatch):
    class DummyMDLM:
        pass

    monkeypatch.setattr(
        "CSDNet.model.lightning_module.build_mdlm",
        lambda _vocab_size, _mask_id: DummyMDLM(),
    )
    model = CSDNetLightningModule(
        vocab_size=8,
        pad_id=0,
        mask_id=1,
        bos_id=2,
        eos_id=3,
        unk_id=4,
        scaffold_ids=frozenset(),
        aromatic_ids=frozenset(),
        cond_dim=0,
        use_ema=False,
        use_cbi=False,
        hidden_size=32,
        num_layers=1,
        num_heads=4,
        intermediate=64,
        max_position_embeddings=16,
        hidden_dropout_prob=0.0,
        attention_probs_dropout_prob=0.0,
        corruption_level_conditioning=True,
        refinement_loss_weight=0.5,
        refinement_corruption_min=1.0,
        refinement_corruption_max=1.0,
        refinement_mask_fraction=0.0,
        refinement_corruption_mode="trajectory",
    )

    def proposal_logits(input_ids, attention_mask, **_kwargs):
        logits = torch.full(
            (*input_ids.shape, 8),
            -100.0,
            device=input_ids.device,
        )
        logits[..., 7] = 100.0
        return logits

    monkeypatch.setattr(model, "_logits", proposal_logits)
    clean = torch.tensor([[2, 5, 6, 3, 0]])
    attention = torch.tensor([[1, 1, 1, 1, 0]])
    draft, corrupt, valid, level = model._sample_refinement_corruption(
        clean,
        attention,
    )

    assert draft.tolist() == [[2, 7, 7, 3, 0]]
    assert corrupt.tolist() == [[False, True, True, False, False]]
    assert valid.tolist() == [[False, True, True, False, False]]
    assert torch.allclose(level, torch.tensor([0.0]))


def test_two_step_trajectory_returns_the_next_sampler_mask_state(monkeypatch):
    monkeypatch.setattr(
        "CSDNet.model.lightning_module.build_mdlm",
        lambda _vocab_size, _mask_id: object(),
    )
    model = CSDNetLightningModule(
        vocab_size=8,
        pad_id=0,
        mask_id=1,
        bos_id=2,
        eos_id=3,
        unk_id=4,
        scaffold_ids=frozenset(),
        aromatic_ids=frozenset(),
        cond_dim=0,
        use_ema=False,
        use_cbi=False,
        hidden_size=32,
        num_layers=1,
        num_heads=4,
        intermediate=64,
        max_position_embeddings=16,
        hidden_dropout_prob=0.0,
        attention_probs_dropout_prob=0.0,
        corruption_level_conditioning=True,
        refinement_loss_weight=0.5,
        refinement_corruption_min=1.0,
        refinement_corruption_max=1.0,
        refinement_mask_fraction=0.0,
        refinement_corruption_mode="trajectory",
        trajectory_rollout_steps=2,
        trajectory_rollout_decay=0.5,
    )

    def proposal_logits(input_ids, attention_mask, **_kwargs):
        logits = torch.full(
            (*input_ids.shape, 8),
            -100.0,
            device=input_ids.device,
        )
        logits[..., 7] = 100.0
        return logits

    monkeypatch.setattr(model, "_logits", proposal_logits)
    clean = torch.tensor([[2, 5, 5, 5, 5, 5, 5, 5, 5, 3]])
    attention = torch.ones_like(clean)
    draft, _corrupt, valid, level = model._sample_refinement_corruption(
        clean,
        attention,
    )

    assert int((draft == model.mask_id).sum().item()) == 2
    assert int(valid.sum().item()) == 8
    assert torch.allclose(level, torch.tensor([0.25]))


def test_trajectory_temperature_matches_short_and_long_endpoints(monkeypatch):
    monkeypatch.setattr(
        "CSDNet.model.lightning_module.build_mdlm",
        lambda _vocab_size, _mask_id: object(),
    )
    model = CSDNetLightningModule(
        vocab_size=8,
        pad_id=0,
        mask_id=1,
        bos_id=2,
        eos_id=3,
        unk_id=4,
        scaffold_ids=frozenset(),
        cond_dim=0,
        use_ema=False,
        use_cbi=False,
        hidden_size=32,
        num_layers=1,
        num_heads=4,
        intermediate=64,
        max_position_embeddings=64,
        corruption_level_conditioning=True,
        refinement_loss_weight=0.5,
    )
    valid = torch.zeros((2, 48), dtype=torch.bool)
    valid[0, :20] = True
    valid[1, :44] = True
    ratios = torch.ones(2)
    temperatures = model._trajectory_temperatures(valid, ratios)
    assert torch.allclose(
        temperatures,
        torch.tensor([
            model.trajectory_temperature_start_short,
            model.trajectory_temperature_start,
        ]),
    )


def test_trajectory_profile_is_copied_from_progressive_length_coupled():
    profile = SAMPLER_PROFILES["promax_progressive_length_coupled"]
    values = trajectory_profile_kwargs("promax_progressive_length_coupled")

    assert values["trajectory_length_low"] == profile["adaptive_length_low"]
    assert values["trajectory_length_high"] == profile["adaptive_length_high"]
    assert values["trajectory_temperature_start"] == profile["temperature_start"]
    assert values["trajectory_remask_power"] == profile["remask_power"]
    assert values["trajectory_confidence_length_adaptive"] is True
    assert values["trajectory_confidence_length_low"] == 28.0
    assert values["trajectory_confidence_length_high"] == 34.0
    assert values["trajectory_confidence_temperature_short"] == 1.0


def test_trajectory_confidence_matches_sampler_length_calibration(monkeypatch):
    monkeypatch.setattr(
        "CSDNet.model.lightning_module.build_mdlm",
        lambda _vocab_size, _mask_id: object(),
    )
    profile = trajectory_profile_kwargs("promax_progressive_length_coupled")
    model = CSDNetLightningModule(
        vocab_size=8,
        pad_id=0,
        mask_id=1,
        bos_id=2,
        eos_id=3,
        unk_id=4,
        scaffold_ids=frozenset(),
        cond_dim=0,
        use_ema=False,
        use_cbi=False,
        hidden_size=32,
        num_layers=1,
        num_heads=4,
        intermediate=64,
        max_position_embeddings=64,
        corruption_level_conditioning=True,
        refinement_loss_weight=0.5,
        **profile,
    )
    valid = torch.zeros((3, 48), dtype=torch.bool)
    valid[0, :20] = True  # Full sequence length 22: short endpoint.
    valid[1, :29] = True  # Full sequence length 31: midpoint of 28--34.
    valid[2, :40] = True  # Full sequence length 42: long endpoint.
    sampling_temperatures = torch.tensor([1.8, 0.7, 0.15])

    confidence = model._trajectory_confidence_temperatures(
        valid,
        sampling_temperatures,
    )

    assert torch.allclose(confidence, torch.tensor([1.0, 0.85, 0.15]))
