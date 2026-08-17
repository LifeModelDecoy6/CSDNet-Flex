from argparse import ArgumentParser, Namespace

import pandas as pd
import pytest
import torch

from CSDNet.exp.denovo.aggregate_promax import _mark_pareto_front
from CSDNet.exp.denovo.sampler_profiles import (
    SAMPLER_PROFILES,
    add_all_position_refinement_arguments,
    add_confidence_planning_arguments,
    add_length_adaptive_arguments,
    apply_sampler_profile,
    all_position_refinement_kwargs,
    confidence_planning_kwargs,
    length_adaptive_kwargs,
)
from CSDNet.util.sampling import (
    _cosine_remask_rate,
    _cosine_remask_rates,
    _length_conditioned_confidence_temperatures,
    _refresh_progressive_scores,
    _smooth_length_mix,
)


def make_args(profile):
    return Namespace(
        sampler_profile=profile,
        temperature_start=9.0,
        temperature_end=9.0,
        temperature_power=9.0,
        gumbel_scale=9.0,
        top_k=9,
        top_p=0.1,
        length_min=99,
        length_max=100,
        length_explore_fraction=0.9,
        length_batching="random",
        remask_power=9.0,
    )


def test_custom_profile_preserves_explicit_parameters():
    args = make_args("custom")
    apply_sampler_profile(args)
    assert args.temperature_start == 9.0
    assert args.length_min == 99


def test_shared_entrypoint_arguments_forward_an_adaptive_profile():
    parser = ArgumentParser()
    parser.add_argument("--sampler_profile", default="promax_length_entropy")
    add_length_adaptive_arguments(parser)
    args = parser.parse_args([])
    apply_sampler_profile(args)
    forwarded = length_adaptive_kwargs(args)
    assert forwarded["length_adaptive"] is True
    assert forwarded["adaptive_length_low"] == 30.0
    assert forwarded["adaptive_length_high"] == 38.0
    assert forwarded["adaptive_gumbel_scale_short"] == 1.35


def test_shared_entrypoint_arguments_forward_confidence_planning():
    parser = ArgumentParser()
    parser.add_argument(
        "--sampler_profile",
        default="promax_confidence_progressive",
    )
    add_length_adaptive_arguments(parser)
    add_confidence_planning_arguments(parser)
    args = parser.parse_args([])
    apply_sampler_profile(args)
    forwarded = confidence_planning_kwargs(args)
    assert forwarded == {
        "confidence_temperature": 1.0,
        "confidence_length_adaptive": False,
        "adaptive_confidence_length_low": 28.0,
        "adaptive_confidence_length_high": 34.0,
        "adaptive_confidence_temperature_short": 1.0,
        "progressive_commit": True,
        "progressive_refresh_confidence": False,
        "progressive_refresh_start": 0.5,
        "progressive_refresh_gain_weight": 0.0,
    }


def test_shared_entrypoint_arguments_forward_all_position_refinement():
    parser = ArgumentParser()
    parser.add_argument(
        "--sampler_profile",
        default="promax_all_position_refine",
    )
    add_all_position_refinement_arguments(parser)
    args = parser.parse_args([])
    apply_sampler_profile(args)
    forwarded = all_position_refinement_kwargs(args)
    assert forwarded == {
        "all_position_refine_steps": 24,
        "all_position_corruption_start": 0.25,
        "all_position_corruption_end": 0.05,
        "all_position_corruption_power": 1.5,
        "all_position_max_edits": 4,
        "all_position_max_total_edits": 0,
        "all_position_min_logprob_gain": 0.05,
        "all_position_verify_masked": False,
        "all_position_verify_min_logprob_gain": 0.25,
        "all_position_prevent_revisit": False,
        "all_position_patience": 0,
        "all_position_rdkit_each_step": False,
    }


def test_named_profiles_override_the_complete_sampler_surface():
    expected = set(next(iter(SAMPLER_PROFILES.values())))
    for name, values in SAMPLER_PROFILES.items():
        assert set(values) == expected
        args = make_args(name)
        apply_sampler_profile(args)
        for key, value in values.items():
            assert getattr(args, key) == value


def test_historical_genmol_profile_uses_fixed_low_temperature_and_randomness():
    values = SAMPLER_PROFILES["genmol_quality"]
    assert values["temperature_start"] == values["temperature_end"] == 0.5
    assert values["gumbel_scale"] == 0.5
    assert values["top_p"] == 1.0


def test_promax_balanced_preserves_the_complete_length_prior():
    baseline = SAMPLER_PROFILES["baseline"]
    empirical = SAMPLER_PROFILES["promax_balanced"]
    for key in (
        "temperature_start",
        "temperature_end",
        "temperature_power",
        "gumbel_scale",
        "top_k",
        "top_p",
        "remask_power",
    ):
        assert empirical[key] == baseline[key]
    assert (empirical["length_min"], empirical["length_max"]) == (0, 0)
    assert empirical["length_explore_fraction"] == 0.0
    assert empirical["length_batching"] == "sorted"


def test_promax_profiles_share_length_protocol_and_order_entropy():
    names = ("promax_balanced", "promax_quality", "promax_diversity")
    length_keys = (
        "length_min",
        "length_max",
        "length_explore_fraction",
        "length_batching",
    )
    for key in length_keys:
        assert len({SAMPLER_PROFILES[name][key] for name in names}) == 1
    quality = SAMPLER_PROFILES["promax_quality"]
    balanced = SAMPLER_PROFILES["promax_balanced"]
    diversity = SAMPLER_PROFILES["promax_diversity"]
    assert quality["temperature_start"] < balanced["temperature_start"]
    assert balanced["temperature_start"] < diversity["temperature_start"]
    assert quality["gumbel_scale"] < balanced["gumbel_scale"]
    assert balanced["gumbel_scale"] < diversity["gumbel_scale"]
    assert quality["remask_power"] > balanced["remask_power"]
    assert balanced["remask_power"] > diversity["remask_power"]


def test_length_adaptive_profiles_preserve_the_full_length_protocol():
    names = ("promax_length_temperature", "promax_length_entropy")
    for name in names:
        values = SAMPLER_PROFILES[name]
        assert values["length_adaptive"] is True
        assert (values["length_min"], values["length_max"]) == (0, 0)
        assert values["length_explore_fraction"] == 0.0
        assert values["length_batching"] == "sorted"
        assert values["adaptive_length_low"] < values["adaptive_length_high"]


def test_temperature_ablation_changes_only_temperature_across_lengths():
    values = SAMPLER_PROFILES["promax_length_temperature"]
    assert values["adaptive_temperature_start_short"] > values["temperature_start"]
    assert values["adaptive_temperature_end_short"] > values["temperature_end"]
    assert values["adaptive_gumbel_scale_short"] == values["gumbel_scale"]
    assert values["adaptive_remask_power_short"] == values["remask_power"]


def test_length_entropy_profile_interpolates_between_observed_endpoints():
    values = SAMPLER_PROFILES["promax_length_entropy"]
    diversity = SAMPLER_PROFILES["promax_diversity"]
    quality = SAMPLER_PROFILES["promax_quality"]
    for key in (
        "temperature_start",
        "temperature_end",
        "temperature_power",
        "gumbel_scale",
        "remask_power",
    ):
        assert values[key] == quality[key]
    assert values["adaptive_temperature_start_short"] == diversity["temperature_start"]
    assert values["adaptive_temperature_end_short"] == diversity["temperature_end"]
    assert values["adaptive_temperature_power_short"] == diversity["temperature_power"]
    assert values["adaptive_gumbel_scale_short"] == diversity["gumbel_scale"]
    assert values["adaptive_remask_power_short"] == diversity["remask_power"]


def test_confidence_profiles_are_clean_length_entropy_ablations():
    baseline = SAMPLER_PROFILES["promax_length_entropy"]
    decoupled = SAMPLER_PROFILES["promax_confidence_decoupled"]
    progressive = SAMPLER_PROFILES["promax_confidence_progressive"]
    progressive_coupled = SAMPLER_PROFILES["promax_progressive_coupled"]
    excluded = {"confidence_temperature", "progressive_commit"}
    for key in set(baseline) - excluded:
        assert decoupled[key] == baseline[key]
        assert progressive[key] == baseline[key]
        assert progressive_coupled[key] == baseline[key]
    assert decoupled["confidence_temperature"] == 1.0
    assert decoupled["progressive_commit"] is False
    assert progressive["confidence_temperature"] == 1.0
    assert progressive["progressive_commit"] is True
    assert progressive_coupled["confidence_temperature"] == 0.0
    assert progressive_coupled["progressive_commit"] is True


def test_all_position_profile_changes_only_the_refinement_stage():
    baseline = SAMPLER_PROFILES["promax_length_entropy"]
    refined = SAMPLER_PROFILES["promax_all_position_refine"]
    keys = {
        "all_position_refine_steps",
        "all_position_corruption_start",
        "all_position_corruption_end",
        "all_position_corruption_power",
        "all_position_max_edits",
        "all_position_max_total_edits",
        "all_position_min_logprob_gain",
        "all_position_verify_masked",
        "all_position_verify_min_logprob_gain",
        "all_position_prevent_revisit",
        "all_position_patience",
        "all_position_rdkit_each_step",
    }
    for key in set(baseline) - keys:
        assert refined[key] == baseline[key]
    assert baseline["all_position_refine_steps"] == 0
    assert refined["all_position_refine_steps"] == 24


def test_verified_refine_profile_is_a_conservative_coordinate_sampler():
    values = SAMPLER_PROFILES["promax_verified_refine"]
    assert values["all_position_refine_steps"] == 8
    assert values["all_position_max_edits"] == 1
    assert values["all_position_max_total_edits"] == 0
    assert values["all_position_verify_masked"] is True
    assert values["all_position_verify_min_logprob_gain"] > 0.0
    assert values["all_position_prevent_revisit"] is True
    assert values["all_position_patience"] == 2
    assert values["all_position_rdkit_each_step"] is True


def test_progressive_consistency_profiles_are_clean_factorial_ablations():
    baseline = SAMPLER_PROFILES["promax_progressive_coupled"]
    refreshed = SAMPLER_PROFILES["promax_progressive_refresh"]
    verified = SAMPLER_PROFILES["promax_progressive_verified"]
    combined = SAMPLER_PROFILES["promax_progressive_consistency"]

    assert refreshed["progressive_commit"] is True
    assert refreshed["confidence_temperature"] == 0.0
    assert refreshed["progressive_refresh_confidence"] is True
    assert refreshed["progressive_refresh_start"] == 0.5
    assert refreshed["all_position_refine_steps"] == 0

    assert verified["progressive_refresh_confidence"] is False
    assert verified["all_position_refine_steps"] == 8
    assert verified["all_position_verify_masked"] is True
    assert verified["all_position_max_total_edits"] == 2

    assert combined["progressive_refresh_confidence"] is True
    assert combined["all_position_refine_steps"] == 8
    assert combined["all_position_verify_masked"] is True
    assert combined["all_position_max_total_edits"] == 2

    changed = {
        "progressive_refresh_confidence",
        "progressive_refresh_start",
        "progressive_refresh_gain_weight",
        "all_position_refine_steps",
        "all_position_corruption_start",
        "all_position_corruption_end",
        "all_position_corruption_power",
        "all_position_max_edits",
        "all_position_max_total_edits",
        "all_position_min_logprob_gain",
        "all_position_verify_masked",
        "all_position_verify_min_logprob_gain",
        "all_position_prevent_revisit",
        "all_position_patience",
        "all_position_rdkit_each_step",
    }
    for key in set(baseline) - changed:
        assert refreshed[key] == baseline[key]
        assert verified[key] == baseline[key]
        assert combined[key] == baseline[key]


def test_length_conditioned_progressive_profile_changes_only_confidence_scale():
    baseline = SAMPLER_PROFILES["promax_progressive_coupled"]
    adaptive = SAMPLER_PROFILES["promax_progressive_length_coupled"]
    changed = {
        "confidence_length_adaptive",
        "adaptive_confidence_length_low",
        "adaptive_confidence_length_high",
        "adaptive_confidence_temperature_short",
    }
    for key in set(baseline) - changed:
        assert adaptive[key] == baseline[key]
    assert adaptive["confidence_length_adaptive"] is True
    assert adaptive["adaptive_confidence_length_low"] == 28.0
    assert adaptive["adaptive_confidence_length_high"] == 34.0


def test_task_adaptive_profiles_preserve_operator_controls_and_isolate_refinement():
    local = SAMPLER_PROFILES["promax_task_adaptive_local"]
    refined = SAMPLER_PROFILES["promax_task_adaptive_refine"]

    assert local["local_confidence_uses_editable_length"] is True
    assert local["local_temperature_mode"] == "operator_scaled"
    assert local["all_position_refine_steps"] == 0

    assert refined["local_confidence_uses_editable_length"] is True
    assert refined["local_temperature_mode"] == "operator_scaled"
    assert refined["all_position_refine_steps"] == 8
    assert refined["all_position_max_total_edits"] == 2
    assert refined["all_position_verify_masked"] is True
    assert refined["all_position_prevent_revisit"] is True
    assert refined["all_position_rdkit_each_step"] is True

    refinement_keys = {
        key for key in refined if key.startswith("all_position_")
    }
    for key in set(local) - refinement_keys:
        assert refined[key] == local[key]


def test_fragment_profiles_isolate_editable_schedule_and_masked_proposal():
    baseline = SAMPLER_PROFILES["promax_fragment_conditional_refine"]
    editable = SAMPLER_PROFILES["promax_fragment_editable_refine"]
    masked = SAMPLER_PROFILES["promax_fragment_masked_refine"]

    assert baseline.get("local_sampling_uses_editable_length", False) is False
    assert editable["local_sampling_uses_editable_length"] is True
    assert editable["local_adaptive_length_low"] < editable[
        "local_adaptive_length_high"
    ]
    assert editable.get("all_position_proposal_masked", False) is False
    assert masked["all_position_proposal_masked"] is True
    assert masked["all_position_max_edits"] == 1
    assert masked["all_position_max_total_edits"] == 4


def test_length_conditioned_confidence_temperature_preserves_endpoints():
    sampling = torch.tensor([1.8, 0.7, 0.15])
    mix = torch.tensor([0.0, 0.5, 1.0])
    result = _length_conditioned_confidence_temperatures(
        sampling,
        mix,
        short_temperature=1.0,
    )
    assert torch.allclose(result, torch.tensor([1.0, 0.85, 0.15]))


def test_progressive_refresh_penalizes_contextual_contradictions_only():
    tokens = torch.tensor([[0, 1, 2]])
    output_scores = torch.tensor([[-0.2, -0.3, -0.4]])
    logits = torch.tensor(
        [[[0.0, 2.0, -1.0], [0.0, 2.0, -1.0], [0.0, -1.0, 2.0]]]
    )
    log_probs = torch.log_softmax(logits, dim=-1)
    committed = torch.tensor([[True, True, False]])
    refresh_rows = torch.tensor([True])

    refreshed, stats = _refresh_progressive_scores(
        tokens=tokens,
        output_scores=output_scores,
        log_probs=log_probs,
        committed_positions=committed,
        refresh_rows=refresh_rows,
        gain_weight=0.5,
    )

    current = log_probs.gather(2, tokens.unsqueeze(-1)).squeeze(-1)
    gain = (log_probs.max(dim=-1).values - current).clamp(min=0.0)
    expected = current - 0.5 * gain
    assert refreshed[0, 0].item() == pytest.approx(expected[0, 0].item())
    assert refreshed[0, 1].item() == pytest.approx(expected[0, 1].item())
    assert refreshed[0, 2].item() == pytest.approx(output_scores[0, 2].item())
    assert stats == {"positions": 2, "contradictions": 1}


def test_smooth_length_mix_has_exact_endpoints_and_monotonic_transition():
    lengths = torch.tensor([20.0, 30.0, 34.0, 38.0, 50.0])
    mix = _smooth_length_mix(lengths, 30.0, 38.0)
    assert torch.allclose(mix, torch.tensor([0.0, 0.0, 0.5, 1.0, 1.0]))
    assert torch.all(mix[1:] >= mix[:-1])


def test_remask_power_preserves_endpoints_and_changes_only_interior():
    assert _cosine_remask_rate(0, 100, power=1.0) == 1.0
    assert _cosine_remask_rate(100, 100, power=1.0) < 1e-12

    core_midpoint = _cosine_remask_rate(50, 100, power=1.0)
    refine_midpoint = _cosine_remask_rate(50, 100, power=1.5)
    assert 0.0 < refine_midpoint < core_midpoint < 1.0

    powers = torch.tensor([[0.8], [1.0], [1.35]])
    vector_midpoint = _cosine_remask_rates(50, 100, powers).squeeze(1)
    assert vector_midpoint[0] > vector_midpoint[1] > vector_midpoint[2]
    assert vector_midpoint[1].item() == pytest.approx(core_midpoint)


def test_promax_summary_marks_quality_diversity_frontier():
    frame = pd.DataFrame(
        {
            "Quality_mean": [0.80, 0.75, 0.82],
            "Diversity_mean": [0.87, 0.86, 0.84],
        }
    )
    marked = _mark_pareto_front(frame)
    assert marked["pareto_nondominated"].tolist() == [True, False, True]
