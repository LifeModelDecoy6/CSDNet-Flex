#!/usr/bin/env python
"""Run geometry-aware direct infilling for fragment-constrained generation."""

from __future__ import annotations

import argparse
import csv
import hashlib
import random
from collections import defaultdict
from dataclasses import replace
from pathlib import Path

import numpy as np
import pandas as pd
import torch
from rdkit import RDLogger

from CSDNet.exp.denovo.sampler_profiles import SAMPLER_PROFILES
from CSDNet.exp.frag.direct_infill import build_masked_template, load_length_prior
from CSDNet.exp.frag.run_direct_infill import (
    _read_rows,
    _validate_fixed_tokens,
    _write_rows,
)
from CSDNet.exp.frag.run_linker_design import evaluate_samples
from CSDNet.exp.frag.task_head import (
    CANONICAL_TASKS,
    assess_candidate,
    build_constraint_spec,
    normalize_task,
)
from CSDNet.exp.pmo.optimizer import (
    load_csdnet_model,
    sample_csdnet_local_remask,
)
from CSDNet.model.edit_scheduler import load_edit_scheduler_checkpoint
from CSDNet.optim.length_policy import ProtectedLengthAllocator
from CSDNet.util.edit_schedule_sampling import schedule_replacement_lengths


RDLogger.DisableLog("rdApp.*")


# These ranges depend only on the structural prompt geometry.  They are broad
# residual-length bands rather than per-drug or target-molecule settings.
GEOMETRY_LENGTH_PROFILES = {
    "multi_anchor": {
        "quality": (9, 24),
        "explore": (25, 48),
    },
    "single_attachment": {
        "quality": (17, 32),
        "explore": (33, 52),
    },
    "multi_attachment": {
        "quality": (5, 12),
        "explore": (13, 36),
    },
    "substructure_expand": {
        "quality": (5, 12),
        "explore": (13, 36),
    },
}


# Geometry-only bands inferred from the normalized-CBI 90M V2 attempts. The
# short arm targets the high-yield region while the wide arm preserves an
# explicit diversity floor. No task name, target molecule, QED, or SA is read
# by this policy during generation.
QUALITY_FRONTIER_LENGTH_PROFILES = {
    "multi_anchor": {
        "quality": (12, 22),
        "explore": (25, 48),
    },
    "single_attachment": {
        "quality": (21, 26),
        "explore": (33, 52),
    },
    "multi_attachment": {
        "quality": (7, 10),
        "explore": (13, 36),
    },
    "substructure_expand": {
        "quality": (5, 8),
        "explore": (13, 36),
    },
}


LENGTH_PROFILE_PRESETS = {
    "v2": GEOMETRY_LENGTH_PROFILES,
    "quality_frontier": QUALITY_FRONTIER_LENGTH_PROFILES,
}


def stable_case_seed(seed: int, task: str, name: str) -> int:
    random_task = "linker_design" if task == "scaffold_morphing" else task
    payload = f"direct-infill-v2:{seed}:{random_task}:{name}".encode()
    return int.from_bytes(hashlib.sha256(payload).digest()[:4], "big")


def allocate_profiles(num_samples: int, quality_fraction: float, rng: random.Random):
    quality_count = int(round(num_samples * quality_fraction))
    quality_count = max(0, min(num_samples, quality_count))
    profiles = ["quality"] * quality_count
    profiles.extend(["explore"] * (num_samples - quality_count))
    rng.shuffle(profiles)
    return profiles


def allocate_stratified_quantiles(num_samples: int, rng: random.Random):
    """Cover a conditional empirical CDF without changing the output budget."""
    if num_samples <= 0:
        return []
    quantiles = [(index + 0.5) / num_samples for index in range(num_samples)]
    rng.shuffle(quantiles)
    return quantiles


def allocate_confidence_profiles(
    num_samples: int,
    confidence_fraction: float,
    baseline_fraction: float,
    rng: random.Random,
):
    confidence_count = int(round(num_samples * confidence_fraction))
    baseline_count = int(round(num_samples * baseline_fraction))
    confidence_count = max(0, min(num_samples, confidence_count))
    baseline_count = max(
        0,
        min(num_samples - confidence_count, baseline_count),
    )
    profiles = ["confidence_quality"] * confidence_count
    profiles.extend(["quality"] * baseline_count)
    profiles.extend(["explore"] * (num_samples - len(profiles)))
    rng.shuffle(profiles)
    return profiles


def _temperature_kwargs(args, profile: str):
    if profile == "confidence_quality":
        return {
            "temperature_start": args.confidence_temperature_start,
            "temperature_end": args.confidence_temperature_end,
            "temperature_power": args.confidence_temperature_power,
        }
    if profile in {"conditional", "quality", "global_quality", "local_quality"}:
        return {
            "temperature_start": args.quality_temperature_start,
            "temperature_end": args.quality_temperature_end,
            "temperature_power": args.quality_temperature_power,
        }
    return {
        "temperature_start": args.explore_temperature_start,
        "temperature_end": args.explore_temperature_end,
        "temperature_power": args.explore_temperature_power,
    }


def _sampling_kwargs(args, profile: str):
    if profile == "confidence_quality":
        return {
            "top_p": args.confidence_top_p,
            "gumbel_scale": args.confidence_gumbel_scale,
            "remask_power": args.confidence_remask_power,
        }
    if profile in {"conditional", "quality", "global_quality", "local_quality"}:
        return {
            "top_p": args.quality_top_p,
            "gumbel_scale": args.quality_gumbel_scale,
            "remask_power": args.quality_remask_power,
        }
    return {
        "top_p": args.explore_top_p,
        "gumbel_scale": args.explore_gumbel_scale,
        "remask_power": args.explore_remask_power,
    }


def _effective_sampling_kwargs(args, proposal_profile: str):
    """Return the controls that remain after shared-profile resolution."""
    if args.local_sampler_profile in {
        "progressive_length_coupled",
        "conditional_progressive_refine",
        "conditional_editable_refine",
        "conditional_masked_refine",
    }:
        source_name = {
            "conditional_progressive_refine": "promax_fragment_conditional_refine",
            "conditional_editable_refine": "promax_fragment_editable_refine",
            "conditional_masked_refine": "promax_fragment_masked_refine",
        }.get(
            args.local_sampler_profile,
            "promax_progressive_length_coupled",
        )
        source = SAMPLER_PROFILES[source_name]
        return {
            "temperature_start": source["temperature_start"],
            "temperature_end": source["temperature_end"],
            "temperature_power": source["temperature_power"],
            "top_p": source["top_p"],
            "gumbel_scale": source["gumbel_scale"],
            "remask_power": source["remask_power"],
        }
    return {
        **_temperature_kwargs(args, proposal_profile),
        **_sampling_kwargs(args, proposal_profile),
    }


def _method_name(args):
    if args.length_allocation_policy == "learned_scheduler":
        return "direct_infill_learned_length"
    if args.length_allocation_policy == "empirical_atomic":
        if args.local_sampler_profile == "conditional_editable_refine":
            return "direct_infill_v8_editable_refine"
        if args.local_sampler_profile == "conditional_masked_refine":
            return "direct_infill_v8_masked_refine"
        return "direct_infill_v7_conditional_refine"
    if args.length_allocation_policy == "protected_total":
        return "direct_infill_v5_protected_total"
    if args.sampling_policy == "confidence_floor":
        return "direct_infill_v6_confidence_floor"
    return f"direct_infill_v2_{args.length_profile_preset}"


def _sample_indices(
    args,
    model_bundle,
    templates,
    profiles,
    indices,
):
    model, tokenizer, device = model_bundle
    grouped = defaultdict(list)
    for index in indices:
        grouped[profiles[index]].append(index)

    outputs = {}
    for profile in (
        "conditional",
        "confidence_quality",
        "quality",
        "global_quality",
        "local_quality",
        "explore",
    ):
        group = grouped.get(profile, [])
        if not group:
            continue
        selected = [templates[index] for index in group]
        want_diagnostics = (
            args.local_sampler_profile
            in {
                "conditional_progressive_refine",
                "conditional_editable_refine",
                "conditional_masked_refine",
            }
        )
        generated = sample_csdnet_local_remask(
            model=model,
            tk=tokenizer,
            seed_smiles=[template.seed_smiles for template in selected],
            max_len=args.max_len,
            device=device,
            batch_size=args.batch_size,
            n_steps=args.n_steps,
            use_fsm_check=not args.disable_fsm_check,
            use_rdkit_kekulize_check=not args.disable_rdkit_kekulize_check,
            rdkit_check_interval=args.rdkit_check_interval,
            max_sample_retries=args.max_sample_retries,
            violation_neighborhood=args.violation_neighborhood,
            edit_plans=[list(template.edit_plans) for template in selected],
            local_sampler_profile=args.local_sampler_profile,
            return_seed_indices=True,
            return_diagnostics=want_diagnostics,
            **_temperature_kwargs(args, profile),
            **_sampling_kwargs(args, profile),
        )
        for item in generated:
            if want_diagnostics:
                smiles, local_index, diagnostics = item
            else:
                smiles, local_index = item
                diagnostics = {}
            outputs[group[int(local_index)]] = (smiles, diagnostics)
    return outputs


def run_case(
    args,
    model_bundle,
    row,
    length_prior,
    length_scheduler=None,
):
    spec = build_constraint_spec(args.task, row)
    length_profiles = LENGTH_PROFILE_PRESETS[args.length_profile_preset]
    if spec.geometry not in length_profiles:
        raise ValueError(f"No direct-infill profile for geometry={spec.geometry}")

    case_seed = stable_case_seed(args.seed, spec.task, spec.name)
    random.seed(case_seed)
    np.random.seed(case_seed)
    torch.manual_seed(case_seed)
    rng = random.Random(case_seed)
    _, tokenizer, _ = model_bundle

    proposal_metadata = []
    if args.length_allocation_policy == "learned_scheduler":
        if length_scheduler is None:
            raise ValueError("learned_scheduler policy requires a scheduler")
        profiles = ["conditional"] * args.num_samples
        # Construct exactly one placeholder per required gap. The learned
        # scheduler, rather than a random delta or empirical prior, determines
        # every replacement length from the visible atomic-token context.
        base_templates = [
            build_masked_template(
                spec,
                max_len=args.max_len,
                length_prior=[3],
                min_added_tokens=1,
                rng=rng,
                added_token_range=(1, 1),
            )
            for _ in range(args.num_samples)
        ]
        scheduled_plans, length_diagnostics = schedule_replacement_lengths(
            length_scheduler,
            tokenizer,
            [template.seed_smiles for template in base_templates],
            [template.edit_plans for template in base_templates],
            max_len=args.max_len,
            device=next(length_scheduler.parameters()).device,
            temperature=args.length_scheduler_temperature,
            top_k=args.length_scheduler_top_k,
            minimum_replacement=1,
        )
        templates = []
        proposal_metadata = []
        for template, plans, diagnostics in zip(
            base_templates,
            scheduled_plans,
            length_diagnostics,
        ):
            target_body = int(diagnostics["target_body_length"])
            added = sum(int(plan["replacement_len"]) for plan in plans)
            templates.append(
                replace(
                    template,
                    edit_plans=tuple(plans),
                    target_length=target_body,
                    added_tokens=added,
                )
            )
            proposal_metadata.append(
                {
                    "length_space": "learned_conditional_atomic",
                    "profile_length_min": min(
                        diagnostics["replacement_lengths"]
                    ),
                    "profile_length_max": max(
                        diagnostics["replacement_lengths"]
                    ),
                    "length_quantile": None,
                }
            )
    elif args.length_allocation_policy == "empirical_atomic":
        quantiles = allocate_stratified_quantiles(args.num_samples, rng)
        profiles = ["conditional"] * args.num_samples
        templates = [
            build_masked_template(
                spec,
                max_len=args.max_len,
                length_prior=length_prior,
                min_added_tokens=args.min_added_tokens,
                rng=rng,
                length_quantile=quantile,
            )
            for quantile in quantiles
        ]
        proposal_metadata = [
            {
                "length_space": "atomic_total_empirical",
                "profile_length_min": min(length_prior),
                "profile_length_max": max(length_prior),
                "length_quantile": quantile,
            }
            for quantile in quantiles
        ]
    elif args.length_allocation_policy == "protected_total":
        allocator = ProtectedLengthAllocator(
            total_quality_range=(
                args.total_quality_length_min,
                args.total_quality_length_max,
            ),
            global_fraction=args.global_quality_fraction,
            local_fraction=args.local_quality_fraction,
            explore_fraction=args.explore_fraction,
        )
        geometry_profiles = GEOMETRY_LENGTH_PROFILES[spec.geometry]
        proposals = allocator.allocate(
            args.num_samples,
            local_added_range=geometry_profiles["quality"],
            explore_added_range=geometry_profiles["explore"],
            rng=rng,
        )
        profiles = [proposal.arm for proposal in proposals]
        templates = []
        for proposal in proposals:
            common = {
                "spec": spec,
                "max_len": args.max_len,
                "length_prior": length_prior,
                "min_added_tokens": args.min_added_tokens,
                "rng": rng,
                "length_quantile": proposal.quantile,
            }
            support = (proposal.support_min, proposal.support_max)
            if proposal.length_space == "total":
                template = build_masked_template(
                    **common,
                    target_length_range=support,
                )
            else:
                template = build_masked_template(
                    **common,
                    added_token_range=support,
                )
            templates.append(template)
            proposal_metadata.append(
                {
                    "length_space": proposal.length_space,
                    "profile_length_min": proposal.support_min,
                    "profile_length_max": proposal.support_max,
                    "length_quantile": proposal.quantile,
                }
            )
    else:
        if args.sampling_policy == "confidence_floor":
            profiles = allocate_confidence_profiles(
                args.num_samples,
                args.confidence_quality_fraction,
                args.baseline_quality_fraction,
                rng,
            )
        else:
            profiles = allocate_profiles(
                args.num_samples,
                args.quality_fraction,
                rng,
            )
        templates = [
            build_masked_template(
                spec,
                max_len=args.max_len,
                length_prior=length_prior,
                min_added_tokens=args.min_added_tokens,
                rng=rng,
                added_token_range=length_profiles[spec.geometry][
                    "quality" if profile == "confidence_quality" else profile
                ],
            )
            for profile in profiles
        ]
        proposal_metadata = [
            {
                "length_space": "added",
                "profile_length_min": length_profiles[spec.geometry][
                    "quality" if profile == "confidence_quality" else profile
                ][0],
                "profile_length_max": length_profiles[spec.geometry][
                    "quality" if profile == "confidence_quality" else profile
                ][1],
                "length_quantile": None,
            }
            for profile in profiles
        ]
    for template in templates:
        _validate_fixed_tokens(template, tokenizer)

    all_indices = list(range(args.num_samples))
    outputs = _sample_indices(
        args,
        model_bundle,
        templates,
        profiles,
        all_indices,
    )
    draw_counts = [1] * args.num_samples
    initial_assessments = [
        assess_candidate(
            outputs[index][0] if index in outputs else None,
            spec,
        )
        for index in all_indices
    ]
    final_diagnostics = [
        outputs.get(index, (None, {}))[1] for index in all_indices
    ]
    final_assessments = list(initial_assessments)

    pending = [
        index
        for index, assessment in enumerate(final_assessments)
        if not assessment.structural_success
    ]
    for _ in range(args.constraint_retries):
        if not pending:
            break
        retry_outputs = _sample_indices(
            args,
            model_bundle,
            templates,
            profiles,
            pending,
        )
        next_pending = []
        for index in pending:
            draw_counts[index] += 1
            retry_smiles, retry_diagnostics = retry_outputs.get(
                index,
                (None, {}),
            )
            assessment = assess_candidate(retry_smiles, spec)
            final_assessments[index] = assessment
            final_diagnostics[index] = retry_diagnostics
            if not assessment.structural_success:
                next_pending.append(index)
        pending = next_pending

    accepted = [
        assessment.smiles
        for assessment in final_assessments
        if assessment.structural_success
    ]
    attempts = []
    for index, (template, profile, proposal, initial, final) in enumerate(
        zip(
            templates,
            profiles,
            proposal_metadata,
            initial_assessments,
            final_assessments,
        )
    ):
        effective_sampling = _effective_sampling_kwargs(args, profile)
        attempts.append(
            {
                "task": spec.task,
                "name": spec.name,
                "seed": args.seed,
                "case_seed": case_seed,
                "attempt": index,
                "geometry": spec.geometry,
                "profile": profile,
                "length_profile_preset": args.length_profile_preset,
                "length_allocation_policy": args.length_allocation_policy,
                "length_space": proposal["length_space"],
                "profile_length_min": proposal["profile_length_min"],
                "profile_length_max": proposal["profile_length_max"],
                "length_quantile": proposal["length_quantile"],
                "temperature_start": effective_sampling["temperature_start"],
                "temperature_end": effective_sampling["temperature_end"],
                "temperature_power": effective_sampling["temperature_power"],
                "top_p": effective_sampling["top_p"],
                "gumbel_scale": effective_sampling["gumbel_scale"],
                "remask_power": effective_sampling["remask_power"],
                "target_length": template.target_length,
                "added_tokens": template.added_tokens,
                "attachment_count": template.attachment_count,
                "draw_count": draw_counts[index],
                "initial_model_output": initial.valid,
                "initial_structural_success": initial.structural_success,
                "recovered_structural": (
                    not initial.structural_success and final.structural_success
                ),
                "model_output": final.valid,
                "smiles": final.smiles,
                "connected": final.connected,
                "no_dummies": final.no_dummies,
                "preserved": final.preserved,
                "required": final.required,
                "structural_success": final.structural_success,
                "mean_log_prob": final_diagnostics[index].get(
                    "mean_log_prob",
                    float("nan"),
                ),
                "refinement_edits": final_diagnostics[index].get(
                    "refinement_edits",
                    0,
                ),
            }
        )
    return spec, accepted, attempts


def run_task(args):
    from tdc import Evaluator, Oracle

    args.task = normalize_task(args.task)
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    data = pd.read_csv(args.fragments_csv)
    if (
        args.length_allocation_policy == "empirical_atomic"
        and Path(args.length_prior).suffix.lower() != ".json"
    ):
        raise ValueError(
            "empirical_atomic length allocation requires the validated CSDNet "
            "atomic JSON prior, not GenMol data/len.pk"
        )
    if args.length_allocation_policy == "learned_scheduler":
        if not args.length_scheduler_ckpt:
            raise ValueError(
                "--length_scheduler_ckpt is required for learned_scheduler"
            )
        length_prior = [3]
    else:
        if args.length_scheduler_ckpt:
            raise ValueError(
                "A learned scheduler cannot be mixed with random/empirical "
                "fragment length allocation."
            )
        length_prior = load_length_prior(args.length_prior, max_len=args.max_len)
    model_bundle = load_csdnet_model(args)
    length_scheduler = None
    if args.length_scheduler_ckpt:
        length_scheduler = load_edit_scheduler_checkpoint(
            args.length_scheduler_ckpt,
            device=model_bundle[2],
        )
    model = model_bundle[0]
    if (
        args.local_sampler_profile
        in {
            "conditional_progressive_refine",
            "conditional_editable_refine",
            "conditional_masked_refine",
        }
        and not bool(getattr(model, "corruption_level_conditioning", False))
    ):
        raise RuntimeError(
            "conditional fragment refinement requires a trajectory-refinement "
            "checkpoint with corruption-level conditioning"
        )
    oracle_qed = Oracle("qed")
    oracle_sa = Oracle("sa")
    diversity_evaluator = Evaluator("diversity")

    stem = f"{args.task}_seed{args.seed}"
    metrics_path = output_dir / f"metrics_{stem}.csv"
    samples_path = output_dir / f"samples_{stem}.csv"
    attempts_path = output_dir / f"attempts_{stem}.csv"
    summary_path = output_dir / f"summary_{stem}.csv"
    metric_rows = _read_rows(metrics_path) if args.resume else []
    sample_rows = _read_rows(samples_path) if args.resume else []
    attempt_rows = _read_rows(attempts_path) if args.resume else []
    completed = {str(record["name"]) for record in metric_rows}

    for _, row in data.iterrows():
        name = str(row["name"])
        if name in completed:
            print(f"[{args.task}] {name}: already complete, skipping")
            continue
        print("=" * 72)
        proposal_description = (
            "single_conditional_learned_length"
            if args.length_allocation_policy == "learned_scheduler"
            else (
                "single_conditional_empirical"
                if args.length_allocation_policy == "empirical_atomic"
                else f"quality_fraction={args.quality_fraction:.2f}"
            )
        )
        print(
            f"[{args.task}] {name}: direct attempts={args.num_samples} "
            f"proposal={proposal_description} "
            f"length_profile={args.length_profile_preset} "
            f"length_policy={args.length_allocation_policy}"
        )
        spec, accepted, attempts = run_case(
            args,
            model_bundle,
            row,
            length_prior,
            length_scheduler=length_scheduler,
        )
        metrics, unique_records = evaluate_samples(
            spec.original,
            accepted,
            args.num_samples,
            oracle_qed,
            oracle_sa,
            diversity_evaluator,
        )
        metrics.update(
            {
                "task": spec.task,
                "name": spec.name,
                "seed": args.seed,
                "method": _method_name(args),
                "original": spec.original,
                "fragment": spec.fragment,
                "geometry": spec.geometry,
                "raw_attempts": args.num_samples,
                "model_draws": sum(item["draw_count"] for item in attempts),
                "constraint_retry_attempts": sum(
                    item["draw_count"] - 1 for item in attempts
                ),
                "recovered_structural": sum(
                    item["recovered_structural"] for item in attempts
                ),
                "model_outputs": sum(item["model_output"] for item in attempts),
                "structural_successes": len(accepted),
                "mean_added_tokens": float(
                    np.mean([item["added_tokens"] for item in attempts])
                ),
                "mean_refinement_edits": float(
                    np.mean([item["refinement_edits"] for item in attempts])
                ),
                "refined_attempts": sum(
                    item["refinement_edits"] > 0 for item in attempts
                ),
            }
        )
        metric_rows.append(metrics)
        attempt_rows.extend(attempts)
        sample_rows.extend(
            {
                "task": spec.task,
                "name": spec.name,
                "seed": args.seed,
                **record,
            }
            for record in unique_records
        )
        print(
            f"[{args.task}] {name}: validity={metrics['validity']:.3f} "
            f"uniqueness={metrics['uniqueness']:.3f} "
            f"quality={metrics['quality']:.3f} "
            f"recovered={metrics['recovered_structural']} "
            f"draws={metrics['model_draws']}"
        )
        _write_rows(metrics_path, metric_rows)
        _write_rows(samples_path, sample_rows)
        _write_rows(attempts_path, attempt_rows)

    frame = pd.DataFrame(metric_rows)
    summary = {
        "task": args.task,
        "seed": args.seed,
        "method": _method_name(args),
        "n_cases": len(frame),
        "quality_fraction": (
            1.0
            if args.length_allocation_policy
            in {"empirical_atomic", "learned_scheduler"}
            else args.quality_fraction
        ),
        "proposal_mode": (
            "single_conditional"
            if args.length_allocation_policy
            in {"empirical_atomic", "learned_scheduler"}
            else args.sampling_policy
        ),
        "length_profile_preset": args.length_profile_preset,
        "length_allocation_policy": args.length_allocation_policy,
        "sampling_policy": args.sampling_policy,
        "local_sampler_profile": args.local_sampler_profile,
        "constraint_retries": args.constraint_retries,
    }
    for metric in (
        "validity",
        "uniqueness",
        "quality",
        "diversity",
        "distance",
        "mean_qed",
        "mean_sa",
        "model_draws",
        "recovered_structural",
        "mean_refinement_edits",
        "refined_attempts",
    ):
        summary[f"{metric}_mean"] = float(frame[metric].mean()) if len(frame) else 0.0
    with summary_path.open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(summary))
        writer.writeheader()
        writer.writerow(summary)
    print(pd.DataFrame([summary]).to_string(index=False))
    print(f"Saved: {metrics_path}")
    print(f"Saved: {samples_path}")
    print(f"Saved: {attempts_path}")
    print(f"Saved: {summary_path}")


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--task", required=True, choices=(*CANONICAL_TASKS, "superstructure_design")
    )
    parser.add_argument("--ckpt_path", required=True)
    parser.add_argument("--vocab", default="csdnet_vocab.pkl")
    parser.add_argument("--fragments_csv", default="data/fragments.csv")
    parser.add_argument("--length_prior", default="data/len.pk")
    parser.add_argument("--length_scheduler_ckpt", default="")
    parser.add_argument("--length_scheduler_temperature", type=float, default=1.0)
    parser.add_argument("--length_scheduler_top_k", type=int, default=8)
    parser.add_argument(
        "--output_dir",
        default="CSDNet/exp/frag/results/direct_infill_v2",
    )
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--cpu", action="store_true")
    parser.add_argument("--resume", action="store_true")
    parser.add_argument("--num_samples", type=int, default=100)
    parser.add_argument("--min_added_tokens", type=int, default=4)
    parser.add_argument("--quality_fraction", type=float, default=0.80)
    parser.add_argument(
        "--sampling_policy",
        choices=("legacy", "confidence_floor"),
        default="legacy",
    )
    parser.add_argument(
        "--local_sampler_profile",
        choices=(
            "legacy",
            "progressive_length_coupled",
            "conditional_progressive_refine",
            "conditional_editable_refine",
            "conditional_masked_refine",
        ),
        default="legacy",
    )
    parser.add_argument("--confidence_quality_fraction", type=float, default=0.60)
    parser.add_argument("--baseline_quality_fraction", type=float, default=0.20)
    parser.add_argument(
        "--length_profile_preset",
        choices=tuple(LENGTH_PROFILE_PRESETS),
        default="v2",
    )
    parser.add_argument(
        "--length_allocation_policy",
        choices=(
            "legacy",
            "protected_total",
            "empirical_atomic",
            "learned_scheduler",
        ),
        default="legacy",
    )
    parser.add_argument("--total_quality_length_min", type=int, default=32)
    parser.add_argument("--total_quality_length_max", type=int, default=47)
    parser.add_argument("--global_quality_fraction", type=float, default=0.40)
    parser.add_argument("--local_quality_fraction", type=float, default=0.40)
    parser.add_argument("--explore_fraction", type=float, default=0.20)
    parser.add_argument("--constraint_retries", type=int, default=1)
    parser.add_argument("--max_len", type=int, default=128)
    parser.add_argument("--batch_size", type=int, default=64)
    parser.add_argument("--n_steps", type=int, default=160)
    parser.add_argument("--quality_temperature_start", type=float, default=1.15)
    parser.add_argument("--quality_temperature_end", type=float, default=0.12)
    parser.add_argument("--quality_temperature_power", type=float, default=1.8)
    parser.add_argument("--quality_top_p", type=float, default=1.0)
    parser.add_argument("--quality_gumbel_scale", type=float, default=1.0)
    parser.add_argument("--quality_remask_power", type=float, default=1.0)
    parser.add_argument("--confidence_temperature_start", type=float, default=1.0)
    parser.add_argument("--confidence_temperature_end", type=float, default=0.10)
    parser.add_argument("--confidence_temperature_power", type=float, default=1.8)
    parser.add_argument("--confidence_top_p", type=float, default=0.97)
    parser.add_argument("--confidence_gumbel_scale", type=float, default=0.25)
    parser.add_argument("--confidence_remask_power", type=float, default=1.0)
    parser.add_argument("--explore_temperature_start", type=float, default=1.55)
    parser.add_argument("--explore_temperature_end", type=float, default=0.22)
    parser.add_argument("--explore_temperature_power", type=float, default=1.35)
    parser.add_argument("--explore_top_p", type=float, default=1.0)
    parser.add_argument("--explore_gumbel_scale", type=float, default=1.0)
    parser.add_argument("--explore_remask_power", type=float, default=1.0)
    parser.add_argument("--disable_fsm_check", action="store_true")
    parser.add_argument("--disable_rdkit_kekulize_check", action="store_true")
    parser.add_argument("--rdkit_check_interval", type=int, default=25)
    parser.add_argument("--max_sample_retries", type=int, default=2)
    parser.add_argument("--violation_neighborhood", type=int, default=2)
    args = parser.parse_args()
    if not 0.0 <= args.quality_fraction <= 1.0:
        parser.error("--quality_fraction must be in [0, 1]")
    confidence_total = (
        args.confidence_quality_fraction + args.baseline_quality_fraction
    )
    if (
        args.confidence_quality_fraction < 0.0
        or args.baseline_quality_fraction < 0.0
        or confidence_total > 1.0
    ):
        parser.error(
            "confidence and baseline quality fractions must be non-negative "
            "and sum to at most 1"
        )
    if args.constraint_retries < 0:
        parser.error("--constraint_retries must be non-negative")
    if args.total_quality_length_min > args.total_quality_length_max:
        parser.error("total quality length bounds must be ordered")
    length_fraction_sum = (
        args.global_quality_fraction
        + args.local_quality_fraction
        + args.explore_fraction
    )
    if any(
        value < 0.0
        for value in (
            args.global_quality_fraction,
            args.local_quality_fraction,
            args.explore_fraction,
        )
    ):
        parser.error("length allocation fractions must be non-negative")
    if abs(length_fraction_sum - 1.0) > 1e-8:
        parser.error("length allocation fractions must sum to 1")
    for name in ("quality_top_p", "confidence_top_p", "explore_top_p"):
        value = float(getattr(args, name))
        if not 0.0 < value <= 1.0:
            parser.error(f"--{name} must be in (0, 1]")
    for name in (
        "quality_gumbel_scale",
        "confidence_gumbel_scale",
        "explore_gumbel_scale",
    ):
        if float(getattr(args, name)) < 0.0:
            parser.error(f"--{name} must be non-negative")
    for name in (
        "quality_remask_power",
        "confidence_remask_power",
        "explore_remask_power",
    ):
        if float(getattr(args, name)) <= 0.0:
            parser.error(f"--{name} must be positive")
    return args


def main():
    args = parse_args()
    required_paths = [args.fragments_csv, args.vocab, args.ckpt_path]
    if args.length_allocation_policy != "learned_scheduler":
        required_paths.append(args.length_prior)
    for path in required_paths:
        if not Path(path).exists():
            raise SystemExit(f"Cannot find required file: {path}")
    run_task(args)


if __name__ == "__main__":
    main()
