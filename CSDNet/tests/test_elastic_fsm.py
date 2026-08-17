import inspect
import pickle
from pathlib import Path
from types import SimpleNamespace

import pytest
import torch

from CSDNet.model.elastic_backbone import ElasticCSDNetBackbone
from CSDNet.model.elastic_lightning_module import (
    ElasticCSDNetLightningModule,
)
from CSDNet.model.elastic_schedule import (
    ElasticKumaSchedule,
    apply_structured_span_mask,
    bregman_poisson,
)
from CSDNet.util.checkpoint import load_backbone_from_checkpoint
from CSDNet.util.elastic_sampling import (
    _build_local_infill_state,
    _repair_final_sequences,
    _select_unmask_events,
)
from CSDNet.util.fsm import (
    ValenceFSMTracker,
    compute_rdkit_sanitization_penalties,
    prepare_rdkit_kekulize_checker,
)
from CSDNet.util.sampling import sample_csdnet
from csdnet_tokenizer import SMILESTokenizer
from train_csdnet_hf_streaming import make_streaming_collate_fn


ROOT = Path(__file__).resolve().parents[2]


def load_tokenizer():
    with (ROOT / "csdnet_vocab.pkl").open("rb") as handle:
        return SMILESTokenizer(pickle.load(handle))


def trim_padding(token_ids, pad_id):
    return [token_id for token_id in token_ids if token_id != pad_id]


def test_elastic_schedule_is_a_normalized_three_state_process():
    schedule = ElasticKumaSchedule(shape_a=1.0)
    time = torch.linspace(0.01, 0.99, 41).unsqueeze(-1)
    insertion_rate = torch.tensor([[0.05, 0.5, 1.0, 1.01, 5.0, 20.0]])
    unmask_rate = torch.tensor([[0.05, 0.5, 1.0, 2.0, 5.0, 20.0]])

    log_probabilities = schedule.state_log_probabilities(
        time,
        insertion_rate,
        unmask_rate,
    )
    probabilities = torch.stack(log_probabilities, dim=-1).exp()

    assert torch.isfinite(probabilities).all()
    assert (probabilities >= 0.0).all()
    assert torch.allclose(
        probabilities.sum(dim=-1),
        torch.ones_like(probabilities[..., 0]),
        atol=2e-5,
        rtol=2e-5,
    )


def test_elastic_schedule_inverse_cdf_and_rate_loss_are_consistent():
    schedule = ElasticKumaSchedule(shape_a=1.0)
    probabilities = torch.linspace(0.01, 0.99, 51)
    rates = torch.linspace(0.05, 20.0, 51)

    event_time = schedule.inverse_cdf(probabilities, rates)
    recovered = schedule.cdf(event_time, rates)
    rate_loss = bregman_poisson(rates, rates)

    assert torch.allclose(recovered, probabilities, atol=2e-5, rtol=2e-5)
    assert torch.allclose(rate_loss, torch.zeros_like(rate_loss), atol=1e-7)


def test_fsm_penalties_are_stable_for_variable_length_batches():
    tokenizer = load_tokenizer()
    tracker = ValenceFSMTracker(tokenizer)
    rows = [
        tokenizer.encode("CCO", max_len=16),
        tokenizer.encode("C#C#C", max_len=16),
        tokenizer.encode("c1ccccc1", max_len=16),
    ]
    batch = torch.tensor(rows, dtype=torch.long)
    batched = tracker.compute_penalties(batch)

    for index, row in enumerate(rows):
        single = tracker.compute_penalties(
            torch.tensor([row], dtype=torch.long)
        )
        assert torch.equal(batched[index], single[0])


def test_rdkit_checker_marks_parser_failures_not_only_kekulization():
    tokenizer = load_tokenizer()
    tracker = ValenceFSMTracker(tokenizer)
    checker = prepare_rdkit_kekulize_checker(tokenizer, tracker)
    if checker is None:
        return
    chem, focus_ids = checker
    invalid = "c1ccccc11"
    batch = torch.tensor(
        [
            tokenizer.encode("c1ccccc1", max_len=32),
            tokenizer.encode(invalid, max_len=32),
        ],
        dtype=torch.long,
    )
    penalties = compute_rdkit_sanitization_penalties(
        batch,
        tokenizer,
        chem,
        focus_ids,
    )

    assert not penalties[0].lt(0).any()
    assert penalties[1].lt(0).any()


def test_elastic_backbone_supports_fsm_repair_api():
    tokenizer = load_tokenizer()
    model = ElasticCSDNetBackbone(
        vocab_size=tokenizer.vocab_size,
        cond_dim=0,
        hidden_size=32,
        num_layers=1,
        num_heads=4,
        intermediate=64,
        pad_token_id=tokenizer.pad_id,
        mask_token_id=tokenizer.mask_id,
        max_position_embeddings=32,
        gradient_checkpointing=False,
    )
    sequences = [
        trim_padding(
            tokenizer.encode("CCO", max_len=16),
            tokenizer.pad_id,
        ),
        trim_padding(
            tokenizer.encode("C#C#C", max_len=16),
            tokenizer.pad_id,
        ),
    ]
    repaired = _repair_final_sequences(
        model=model,
        tk=tokenizer,
        sequences=sequences,
        device="cpu",
        use_fsm_check=True,
        use_rdkit_kekulize_check=False,
        max_sample_retries=1,
        violation_neighborhood=2,
        temperature=0.5,
        top_k=0,
        top_p=1.0,
    )
    assert len(repaired) == len(sequences)
    assert all(row[0] == tokenizer.bos_id for row in repaired)
    assert all(tokenizer.eos_id in row for row in repaired)


def test_fixed_unmask_rate_removes_heads_and_masks_padding():
    tokenizer = load_tokenizer()
    model = ElasticCSDNetBackbone(
        vocab_size=tokenizer.vocab_size,
        cond_dim=0,
        hidden_size=32,
        num_layers=1,
        num_heads=4,
        intermediate=64,
        pad_token_id=tokenizer.pad_id,
        mask_token_id=tokenizer.mask_id,
        max_position_embeddings=32,
        fixed_unmask_rate=1.0,
        gradient_checkpointing=False,
    )
    input_ids = torch.tensor(
        [tokenizer.encode("CCO", max_len=10)],
        dtype=torch.long,
    )
    attention = input_ids.ne(tokenizer.pad_id).long()
    output = model(
        input_ids,
        attention,
        t=torch.tensor([0.5]),
        return_aux=True,
    )

    assert not hasattr(model, "theta_unmask_head")
    assert not hasattr(model, "phi_unmask_head")
    assert torch.equal(
        output["b_unmask"],
        attention.to(dtype=output["b_unmask"].dtype),
    )


def test_phi_rate_only_forward_skips_unused_vocabulary_logits():
    tokenizer = load_tokenizer()
    model = ElasticCSDNetBackbone(
        vocab_size=tokenizer.vocab_size,
        cond_dim=0,
        hidden_size=32,
        num_layers=1,
        num_heads=4,
        intermediate=64,
        pad_token_id=tokenizer.pad_id,
        mask_token_id=tokenizer.mask_id,
        max_position_embeddings=32,
        fixed_unmask_rate=1.0,
        gradient_checkpointing=False,
    )
    input_ids = torch.tensor(
        [tokenizer.encode("CCO", max_len=10)],
        dtype=torch.long,
    )
    output = model(
        input_ids,
        input_ids.ne(tokenizer.pad_id).long(),
        t=torch.ones(1),
        return_aux=True,
        rate_family="phi",
        compute_logits=False,
    )

    assert "logits" not in output
    assert output["hidden_states"].shape[:2] == input_ids.shape
    assert output["b_ins"].shape == input_ids.shape


def test_loflex_rate_heads_use_distinct_theta_and_phi_supports():
    model = ElasticCSDNetBackbone(
        vocab_size=32,
        cond_dim=0,
        hidden_size=32,
        num_layers=1,
        num_heads=4,
        intermediate=64,
        pad_token_id=0,
        mask_token_id=1,
        max_position_embeddings=32,
        rate_max=100.0,
        rate_initial=1.0,
        rate_parameterization="exp",
        theta_rate_min=0.0,
        phi_rate_min=1.01,
        rate_output_bias=-4.0,
        fixed_unmask_rate=1.0,
        kuma_shape_a=1.0,
        gradient_checkpointing=False,
    )
    input_ids = torch.tensor([[2, 4, 5, 3, 0]])
    attention = input_ids.ne(0).long()
    time = torch.tensor([0.5])
    theta = model(
        input_ids,
        attention,
        t=time,
        return_aux=True,
        rate_family="theta",
    )
    phi = model(
        input_ids,
        attention,
        t=time,
        return_aux=True,
        rate_family="phi",
    )

    active = attention.bool()
    assert theta["b_ins"][active].min() >= 0.0
    assert theta["b_ins"][active].max() < 0.1
    assert phi["b_ins"][active].min() >= 1.01
    assert torch.equal(phi["b_unmask"], attention.to(phi["b_unmask"].dtype))


def test_gradient_checkpointing_is_non_reentrant_for_shared_trajectories():
    model = ElasticCSDNetBackbone(
        vocab_size=32,
        cond_dim=0,
        hidden_size=32,
        num_layers=1,
        num_heads=4,
        intermediate=64,
        pad_token_id=0,
        mask_token_id=1,
        max_position_embeddings=32,
        fixed_unmask_rate=1.0,
        gradient_checkpointing=True,
    )
    checkpoint_function = (
        model.esm.esm.encoder._gradient_checkpointing_func
    )

    assert checkpoint_function.keywords["use_reentrant"] is False

    input_ids = torch.tensor(
        [
            [2, 4, 1, 5, 3, 0],
            [2, 1, 6, 7, 3, 0],
        ],
        dtype=torch.long,
    )
    attention = input_ids.ne(0).long()
    time = torch.full((input_ids.size(0),), 0.5)
    phi = model(
        input_ids,
        attention,
        t=time,
        return_aux=True,
        rate_family="phi",
    )
    theta_one = model(
        input_ids,
        attention,
        t=time,
        return_aux=True,
        rate_family="theta",
    )
    theta_two = model(
        input_ids,
        attention,
        t=time,
        return_aux=True,
        rate_family="theta",
    )
    loss = (
        phi["b_ins"].mean()
        + theta_one["logits"].mean()
        + theta_one["b_ins"].mean()
        + theta_two["logits"].mean()
        + theta_two["b_ins"].mean()
    )
    loss.backward()

    final_layer = model.esm.esm.encoder.layer[-1].output.dense.weight
    assert final_layer.grad is not None
    assert torch.isfinite(final_layer.grad).all()


def test_complete_elastic_training_graph_backpropagates():
    module = ElasticCSDNetLightningModule(
        vocab_size=32,
        pad_id=0,
        mask_id=1,
        bos_id=2,
        eos_id=3,
        unk_id=31,
        scaffold_ids=frozenset(),
        aromatic_ids=frozenset(),
        cond_dim=0,
        hidden_size=32,
        num_layers=1,
        num_heads=4,
        intermediate=64,
        max_position_embeddings=32,
        fixed_unmask_rate=1.0,
        structured_corruption_prob=0.10,
        gradient_checkpointing=False,
    )
    input_ids = torch.tensor(
        [
            [2, 4, 5, 6, 3, 0, 0, 0],
            [2, 7, 8, 9, 10, 3, 0, 0],
        ],
        dtype=torch.long,
    )
    batch = {
        "input_ids": input_ids,
        "attention_mask": input_ids.ne(0).long(),
        "cond": torch.empty(input_ids.size(0), 0),
        "aromatic_mask": torch.zeros_like(input_ids, dtype=torch.bool),
    }

    module.train()
    losses = module._step(batch, is_train=True)
    assert torch.isfinite(losses["loss"])
    losses["loss"].backward()

    final_layer = (
        module.backbone.esm.esm.encoder.layer[-1].output.dense.weight
    )
    assert final_layer.grad is not None
    assert torch.isfinite(final_layer.grad).all()
    for rate_head in (
        module.backbone.phi_insertion_head,
        module.backbone.theta_insertion_head,
    ):
        assert rate_head.out.weight.grad is not None
        assert torch.isfinite(rate_head.out.weight.grad).all()


def test_loflex_aligned_multitask_training_graph_backpropagates():
    module = ElasticCSDNetLightningModule(
        vocab_size=32,
        pad_id=0,
        mask_id=1,
        bos_id=2,
        eos_id=3,
        unk_id=31,
        scaffold_ids=frozenset(),
        aromatic_ids=frozenset(),
        cond_dim=0,
        hidden_size=32,
        num_layers=1,
        num_heads=4,
        intermediate=64,
        max_position_embeddings=32,
        rate_max=100.0,
        rate_parameterization="exp",
        theta_rate_min=0.0,
        phi_rate_min=1.01,
        rate_output_bias=-4.0,
        fixed_unmask_rate=1.0,
        kuma_shape_a=1.0,
        loflex_objective=True,
        policy_time_conditioning=True,
        fragment_corruption_prob=0.25,
        mdm_corruption_prob=0.25,
        refine_corruption_prob=0.25,
        length_loss_normalizer=32,
        structured_span_min=2,
        structured_span_max=4,
        gradient_checkpointing=False,
    )
    input_ids = torch.tensor(
        [
            [2, 4, 5, 6, 7, 3, 0, 0],
            [2, 8, 9, 10, 11, 3, 0, 0],
            [2, 12, 13, 14, 15, 3, 0, 0],
            [2, 16, 17, 18, 19, 3, 0, 0],
        ],
        dtype=torch.long,
    )
    batch = {
        "input_ids": input_ids,
        "attention_mask": input_ids.ne(0).long(),
        "cond": torch.empty(input_ids.size(0), 0),
        "aromatic_mask": torch.zeros_like(input_ids, dtype=torch.bool),
    }

    torch.manual_seed(7)
    module.train()
    losses = module._step(batch, is_train=True)
    assert torch.isfinite(losses["loss"])
    losses["loss"].backward()
    assert module.backbone.theta_insertion_head.out.weight.grad is not None
    assert module.backbone.phi_insertion_head.out.weight.grad is not None
    assert torch.isfinite(
        module.backbone.theta_insertion_head.out.weight.grad
    ).all()
    assert torch.isfinite(
        module.backbone.phi_insertion_head.out.weight.grad
    ).all()


def test_fragment_geometry_probabilities_must_form_distribution():
    with pytest.raises(ValueError, match="geometry probabilities"):
        ElasticCSDNetLightningModule(
            vocab_size=32,
            pad_id=0,
            mask_id=1,
            bos_id=2,
            eos_id=3,
            unk_id=31,
            scaffold_ids=frozenset(),
            aromatic_ids=frozenset(),
            fragment_internal_probability=0.5,
            fragment_terminal_probability=0.5,
            fragment_dual_probability=0.5,
            fragment_multi_probability=0.5,
            hidden_size=32,
            num_layers=1,
            num_heads=4,
            intermediate=64,
            max_position_embeddings=32,
            gradient_checkpointing=False,
        )


@pytest.mark.parametrize("geometry_id", range(4))
def test_fragment_geometry_course_produces_requested_edit_shape(geometry_id):
    probabilities = [0.0, 0.0, 0.0, 0.0]
    probabilities[geometry_id] = 1.0
    module = ElasticCSDNetLightningModule(
        vocab_size=32,
        pad_id=0,
        mask_id=1,
        bos_id=2,
        eos_id=3,
        unk_id=31,
        scaffold_ids=frozenset(),
        aromatic_ids=frozenset(),
        fragment_internal_probability=probabilities[0],
        fragment_terminal_probability=probabilities[1],
        fragment_dual_probability=probabilities[2],
        fragment_multi_probability=probabilities[3],
        structured_span_min=2,
        structured_span_max=4,
        hidden_size=32,
        num_layers=1,
        num_heads=4,
        intermediate=64,
        max_position_embeddings=32,
        gradient_checkpointing=False,
    )
    active = torch.zeros(1, 24, dtype=torch.bool)
    active[:, 1:23] = True
    selected, geometry = module._sample_fragment_active_mask(
        active,
        torch.tensor([True]),
    )
    selected_positions = selected[0].nonzero(as_tuple=False).flatten()

    assert geometry.item() == geometry_id
    assert selected_positions.numel() > 0
    if geometry_id == 0:
        assert selected_positions.min() > 1
        assert selected_positions.max() < 22
    elif geometry_id == 1:
        assert selected_positions.min() == 1 or selected_positions.max() == 22
    elif geometry_id in {2, 3}:
        gaps = selected_positions[1:] - selected_positions[:-1]
        assert gaps.gt(1).any()


def test_rotary_backbone_accepts_full_256_token_sequence():
    model = ElasticCSDNetBackbone(
        vocab_size=32,
        cond_dim=0,
        hidden_size=32,
        num_layers=1,
        num_heads=4,
        intermediate=64,
        pad_token_id=0,
        mask_token_id=1,
        max_position_embeddings=256,
        position_embedding_type="rotary",
        gradient_checkpointing=False,
    )
    input_ids = torch.full((1, 256), 4, dtype=torch.long)
    input_ids[:, 0] = 2
    input_ids[:, -1] = 3
    attention = torch.ones_like(input_ids)

    logits = model(input_ids, attention, t=torch.tensor([0.5]))

    assert logits.shape == (1, 256, 32)
    assert torch.isfinite(logits).all()


def test_elastic_step_uses_one_phi_and_one_merged_theta_forward():
    module = ElasticCSDNetLightningModule(
        vocab_size=32,
        pad_id=0,
        mask_id=1,
        bos_id=2,
        eos_id=3,
        unk_id=31,
        scaffold_ids=frozenset(),
        aromatic_ids=frozenset(),
        cond_dim=0,
        hidden_size=32,
        num_layers=1,
        num_heads=4,
        intermediate=64,
        max_position_embeddings=32,
        fixed_unmask_rate=1.0,
        structured_corruption_prob=0.0,
        gradient_checkpointing=False,
    )
    input_ids = torch.tensor(
        [
            [2, 4, 5, 6, 3, 0, 0, 0],
            [2, 7, 8, 9, 10, 3, 0, 0],
        ],
        dtype=torch.long,
    )
    batch = {
        "input_ids": input_ids,
        "attention_mask": input_ids.ne(0).long(),
        "cond": torch.empty(input_ids.size(0), 0),
        "aromatic_mask": torch.zeros_like(input_ids, dtype=torch.bool),
    }
    forward_calls = []
    hook = module.backbone.esm.esm.register_forward_hook(
        lambda *_: forward_calls.append(1)
    )
    try:
        module._step(batch, is_train=True)
    finally:
        hook.remove()

    assert len(forward_calls) == 2


def test_merged_theta_matches_two_separate_eval_forwards():
    module = ElasticCSDNetLightningModule(
        vocab_size=32,
        pad_id=0,
        mask_id=1,
        bos_id=2,
        eos_id=3,
        unk_id=31,
        scaffold_ids=frozenset(),
        aromatic_ids=frozenset(),
        cond_dim=0,
        hidden_size=32,
        num_layers=1,
        num_heads=4,
        intermediate=64,
        max_position_embeddings=32,
        fixed_unmask_rate=1.0,
        gradient_checkpointing=False,
    )
    module.eval()
    states = [
        {
            "input_ids": torch.tensor(
                [[2, 1, 5, 3, 0, 0], [2, 7, 1, 3, 0, 0]]
            )
        },
        {
            "input_ids": torch.tensor(
                [[2, 4, 1, 3, 0, 0], [2, 1, 8, 3, 0, 0]]
            )
        },
    ]
    time = torch.tensor([0.25, 0.75])
    with torch.no_grad():
        merged = module._merged_theta_outputs(
            states=states,
            t=time,
            cond=None,
            drop_cond=False,
        )
        separate = [
            module.backbone(
                state["input_ids"],
                state["input_ids"].ne(0).long(),
                t=time,
                return_aux=True,
                rate_family="theta",
            )
            for state in states
        ]

    for merged_output, separate_output in zip(merged, separate):
        for name in ("logits", "b_ins", "b_unmask"):
            assert torch.allclose(
                merged_output[name],
                separate_output[name],
                atol=1e-6,
                rtol=1e-6,
            )


def test_dynamic_streaming_collate_crops_padding_and_aromatic_mask():
    tokenizer = SimpleNamespace(pad_id=0)
    collate = make_streaming_collate_fn(
        tokenizer,
        use_aromatic_cbi=True,
        dynamic_padding=True,
        pad_to_multiple_of=4,
    )
    batch = [
        {
            "input_ids": torch.tensor([2, 4, 3, 0, 0, 0, 0, 0, 0, 0]),
            "aromatic_mask": torch.tensor(
                [False, True, False, False, False, False, False, False, False, False]
            ),
        },
        {
            "input_ids": torch.tensor([2, 5, 6, 7, 3, 0, 0, 0, 0, 0]),
            "aromatic_mask": torch.tensor(
                [False, False, True, False, False, False, False, False, False, False]
            ),
        },
    ]

    output = collate(batch)

    assert output["input_ids"].shape == (2, 8)
    assert output["attention_mask"].shape == (2, 8)
    assert output["aromatic_mask"].shape == (2, 8)
    assert output["attention_mask"].sum(dim=1).tolist() == [3, 5]


def test_top_prob_unmask_selection_preserves_event_count():
    current_mask = torch.tensor([[True, True, True, True]])
    probability = torch.tensor([[1.0, 1.0, 0.0, 0.0]])
    confidence = torch.tensor([[0.1, 0.9, 0.8, 0.2]])

    selected = _select_unmask_events(
        current_mask=current_mask,
        unmask_probability=probability,
        confidence=confidence,
        selection="top_prob",
    )

    assert selected.sum().item() == 2
    assert selected.tolist() == [[False, True, True, False]]


def test_top_level_sampler_exposes_unmask_selection():
    signature = inspect.signature(sample_csdnet)

    assert signature.parameters["unmask_selection"].default == "top_prob"


def test_local_infill_can_delegate_length_entirely_to_learned_rates():
    tokenizer = load_tokenizer()
    state = _build_local_infill_state(
        "CCO",
        {"start": 1, "stop": 2},
        tk=tokenizer,
        max_len=16,
    )

    assert state is not None
    assert state["gaps"][0]["minimum"] == 0
    assert state["gaps"][0]["maximum"] == 12


def test_explicit_task_length_bounds_override_unbounded_default():
    tokenizer = load_tokenizer()
    state = _build_local_infill_state(
        "CCO",
        {
            "start": 1,
            "stop": 2,
            "min_replacement_len": 2,
            "max_replacement_len": 5,
        },
        tk=tokenizer,
        max_len=16,
    )

    assert state is not None
    assert state["gaps"][0]["minimum"] == 2
    assert state["gaps"][0]["maximum"] == 5


def test_structured_span_mask_only_changes_selected_observed_tokens():
    tokenizer = load_tokenizer()
    clean_ids = torch.tensor(
        [
            tokenizer.encode("CCOCC", max_len=12),
            tokenizer.encode("c1ccccc1", max_len=12),
        ],
        dtype=torch.long,
    )
    batch_size, max_length = clean_ids.shape
    source_positions = torch.arange(max_length).repeat(batch_size, 1)
    fixed = (
        clean_ids.eq(tokenizer.bos_id)
        | clean_ids.eq(tokenizer.eos_id)
        | clean_ids.eq(tokenizer.pad_id)
    )
    state = {
        "input_ids": clean_ids.clone(),
        "source_positions": source_positions,
        "deleted": torch.zeros_like(clean_ids, dtype=torch.bool),
        "masked": torch.zeros_like(clean_ids, dtype=torch.bool),
    }

    output, applied = apply_structured_span_mask(
        state=state,
        fixed=fixed,
        mask_id=tokenizer.mask_id,
        pad_id=tokenizer.pad_id,
        selected_samples=torch.tensor([True, False]),
        min_span=2,
        max_span=2,
    )

    assert applied.tolist() == [True, False]
    assert output["input_ids"][0].eq(tokenizer.mask_id).sum().item() == 2
    assert torch.equal(output["input_ids"][1], clean_ids[1])
    assert not output["input_ids"][fixed].eq(tokenizer.mask_id).any()
    assert output["masked"][0].sum().item() == 2
    assert not output["deleted"].any()
    assert torch.equal(state["input_ids"], clean_ids)


def test_aromatic_weight_uses_hold_cosine_anneal_and_unbiased_tail():
    schedule = SimpleNamespace(
        use_aromatic_cbi=True,
        aromatic_cbi_weight=1.2,
        aromatic_cbi_final_weight=1.0,
        aromatic_cbi_anneal_steps=100000,
        aromatic_cbi_anneal_start_fraction=0.10,
        aromatic_cbi_anneal_end_fraction=0.90,
    )
    weight = ElasticCSDNetLightningModule._aromatic_weight

    assert weight(schedule, step=0) == 1.2
    assert weight(schedule, step=10000) == 1.2
    assert abs(weight(schedule, step=50000) - 1.1) < 1e-7
    assert weight(schedule, step=90000) == 1.0
    assert weight(schedule, step=100000) == 1.0


def test_aromatic_weighting_preserves_per_molecule_loss_mass():
    fixture = SimpleNamespace(
        alpha=3.0,
        tau=1.0,
        use_cbi=False,
        use_aromatic_cbi=True,
        normalized_cbi=True,
        _aromatic_weight=lambda: 1.2,
        _scaffold_buf=torch.zeros(8, dtype=torch.bool),
        _aromatic_buf=torch.zeros(8, dtype=torch.bool),
    )
    target_ids = torch.tensor([[1, 2, 3, 0], [2, 3, 4, 5]])
    valid = torch.tensor(
        [[True, True, True, False], [True, True, True, True]]
    )
    log_target = torch.tensor(
        [[-0.2, -1.0, -2.0, 0.0], [-2.0, -1.0, -0.5, -0.1]]
    )
    aromatic_mask = torch.tensor(
        [[True, False, True, False], [False, True, False, True]]
    )

    multiplier = ElasticCSDNetLightningModule._token_multiplier(
        fixture,
        target_ids,
        valid,
        log_target,
        aromatic_mask,
    )
    valid_mean = (
        (multiplier * valid.float()).sum(dim=-1)
        / valid.sum(dim=-1).float()
    )

    assert torch.allclose(valid_mean, torch.full_like(valid_mean, 4.0))


def test_unnormalized_aromatic_weighting_increases_loss_mass():
    fixture = SimpleNamespace(
        alpha=3.0,
        tau=1.0,
        use_cbi=False,
        use_aromatic_cbi=True,
        normalized_cbi=False,
        _aromatic_weight=lambda: 1.2,
        _scaffold_buf=torch.zeros(8, dtype=torch.bool),
        _aromatic_buf=torch.zeros(8, dtype=torch.bool),
    )
    target_ids = torch.tensor([[1, 2, 3]])
    valid = torch.ones_like(target_ids, dtype=torch.bool)
    log_target = torch.tensor([[-0.2, -1.0, -2.0]])
    aromatic_mask = torch.tensor([[True, False, True]])

    multiplier = ElasticCSDNetLightningModule._token_multiplier(
        fixture,
        target_ids,
        valid,
        log_target,
        aromatic_mask,
    )

    assert multiplier.mean() > 4.0


def test_phi_schedule_is_independent_of_sampled_observation_time():
    sampled_time = torch.tensor([0.01, 0.25, 0.75, 0.99])
    phi_time = ElasticCSDNetLightningModule._phi_schedule_time(sampled_time)

    assert torch.equal(phi_time, torch.ones_like(sampled_time))


def test_loflex_policy_uses_sampled_observation_time():
    sampled_time = torch.tensor([0.01, 0.25, 0.75, 0.99])
    phi_time = ElasticCSDNetLightningModule._phi_schedule_time(
        sampled_time,
        use_observation_time=True,
    )

    assert torch.equal(phi_time, sampled_time)


def test_masked_row_mean_is_invariant_to_padding_width():
    short_values = torch.tensor([[2.0, 4.0]])
    short_mask = torch.tensor([[True, True]])
    padded_values = torch.tensor([[2.0, 4.0, 99.0, 99.0]])
    padded_mask = torch.tensor([[True, True, False, False]])

    short_mean = ElasticCSDNetLightningModule._masked_row_mean(
        short_values,
        short_mask,
    )
    padded_mean = ElasticCSDNetLightningModule._masked_row_mean(
        padded_values,
        padded_mask,
    )

    assert torch.equal(short_mean, padded_mean)
    assert short_mean.item() == 3.0


def test_elastic_checkpoint_restores_architecture_and_ema(tmp_path):
    tokenizer = load_tokenizer()
    module = ElasticCSDNetLightningModule(
        vocab_size=tokenizer.vocab_size,
        pad_id=tokenizer.pad_id,
        mask_id=tokenizer.mask_id,
        bos_id=tokenizer.bos_id,
        eos_id=tokenizer.eos_id,
        unk_id=tokenizer.unk_id,
        scaffold_ids=tokenizer.scaffold_ids,
        aromatic_ids=tokenizer.aromatic_ids,
        cond_dim=0,
        hidden_size=32,
        num_layers=1,
        num_heads=4,
        intermediate=64,
        max_position_embeddings=32,
        fixed_unmask_rate=1.0,
        gradient_checkpointing=False,
    )
    parameter_name, parameter = next(module.backbone.named_parameters())
    parameter.data.fill_(0.125)
    ema_name = parameter_name.replace(".", "___")
    getattr(module.ema, ema_name).fill_(0.375)
    checkpoint_path = tmp_path / "elastic.ckpt"
    torch.save(
        {
            "state_dict": module.state_dict(),
            "hyper_parameters": dict(module.hparams),
        },
        checkpoint_path,
    )

    loaded = load_backbone_from_checkpoint(
        checkpoint_path,
        tokenizer,
        device="cpu",
        use_ema=True,
    )

    assert loaded.is_elastic
    assert loaded.fixed_unmask_rate == 1.0
    assert not hasattr(loaded, "theta_unmask_head")
    assert loaded.position_embedding_type == "rotary"
    loaded_parameter = dict(loaded.named_parameters())[parameter_name]
    assert torch.allclose(
        loaded_parameter,
        torch.full_like(loaded_parameter, 0.375),
    )
