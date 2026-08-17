#!/usr/bin/env python
"""Prior-protected non-parametric length search for fragment infilling."""

from __future__ import annotations

import argparse
import csv
import hashlib
import random
from collections import defaultdict
from pathlib import Path

import numpy as np
import pandas as pd
import torch
from rdkit import RDLogger

from CSDNet.exp.frag.direct_infill import build_masked_template, load_length_prior
from CSDNet.exp.frag.length_search import NonParametricLengthController
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


RDLogger.DisableLog("rdApp.*")


def allocate_temperature_profiles(
    num_samples: int,
    conservative_fraction: float,
    rng: random.Random,
):
    conservative_count = int(round(num_samples * conservative_fraction))
    conservative_count = max(0, min(num_samples, conservative_count))
    profiles = ["conservative"] * conservative_count
    profiles.extend(["explore"] * (num_samples - conservative_count))
    rng.shuffle(profiles)
    return profiles


def _temperature_kwargs(args, profile: str):
    if profile == "conservative":
        return {
            "temperature_start": args.conservative_temperature_start,
            "temperature_end": args.conservative_temperature_end,
            "temperature_power": args.conservative_temperature_power,
        }
    return {
        "temperature_start": args.explore_temperature_start,
        "temperature_end": args.explore_temperature_end,
        "temperature_power": args.explore_temperature_power,
    }


def stable_case_seed(seed: int, task: str, name: str) -> int:
    random_task = "linker_design" if task == "scaffold_morphing" else task
    payload = f"direct-infill-v4:{seed}:{random_task}:{name}".encode()
    return int.from_bytes(hashlib.sha256(payload).digest()[:4], "big")


def build_feasible_template_bank(args, spec, length_prior, case_seed):
    """Enumerate executable lengths without consulting the target molecule."""

    bank = {}
    maximum_body_tokens = args.max_len - 2
    for added_tokens in range(args.min_added_tokens, maximum_body_tokens + 1):
        variants = []
        for variant in range(args.length_template_variants):
            variant_seed = (
                case_seed
                ^ (added_tokens * 0x9E3779B1)
                ^ (variant * 0x85EBCA77)
            ) & 0xFFFFFFFF
            try:
                template = build_masked_template(
                    spec,
                    max_len=args.max_len,
                    length_prior=length_prior,
                    min_added_tokens=args.min_added_tokens,
                    rng=random.Random(variant_seed),
                    added_token_range=(added_tokens, added_tokens),
                )
            except ValueError:
                continue
            if (
                template.added_tokens == added_tokens
                and template.target_length <= maximum_body_tokens
            ):
                variants.append(template)
        if variants:
            bank[added_tokens] = tuple(variants)

    if not bank:
        raise RuntimeError(
            f"No feasible infill lengths for task={spec.task}, case={spec.name}."
        )

    fixed_counts = [
        template.target_length - template.added_tokens
        for variants in bank.values()
        for template in variants
    ]
    reference_fixed = int(round(float(np.median(fixed_counts))))
    feasible = tuple(sorted(bank))

    def nearest_feasible(value):
        desired = int(value) - reference_fixed
        return min(feasible, key=lambda length: (abs(length - desired), length))

    added_prior = [nearest_feasible(total_length) for total_length in length_prior]
    return bank, reference_fixed, added_prior


def _sample_batch(args, model_bundle, templates, profiles):
    model, tokenizer, device = model_bundle
    grouped = defaultdict(list)
    for index, profile in enumerate(profiles):
        grouped[profile].append(index)

    outputs = {}
    for profile in ("conservative", "explore"):
        indices = grouped.get(profile, [])
        if not indices:
            continue
        selected = [templates[index] for index in indices]
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
            return_seed_indices=True,
            return_diagnostics=True,
            **_temperature_kwargs(args, profile),
        )
        for smiles, local_index, diagnostics in generated:
            outputs[indices[int(local_index)]] = (smiles, diagnostics)
    return outputs


def run_case(args, model_bundle, row, length_prior):
    spec = build_constraint_spec(args.task, row)
    case_seed = stable_case_seed(args.seed, spec.task, spec.name)
    random.seed(case_seed)
    np.random.seed(case_seed)
    torch.manual_seed(case_seed)
    rng = random.Random(case_seed)
    _, tokenizer, _ = model_bundle

    template_bank, reference_fixed, added_prior = build_feasible_template_bank(
        args,
        spec,
        length_prior,
        case_seed,
    )
    feasible_lengths = tuple(sorted(template_bank))
    controller = NonParametricLengthController(
        added_prior,
        minimum=feasible_lengths[0],
        maximum=feasible_lengths[-1],
        warmup_attempts=args.warmup_samples,
        prior_floor=args.prior_floor,
        exploration_fraction=args.length_exploration_fraction,
        ucb_scale=args.ucb_scale,
        softmax_temperature=args.length_softmax_temperature,
        refinement_radius=args.refinement_radius,
        feasible_lengths=feasible_lengths,
    )

    seen = set()
    accepted = []
    attempts = []
    controller_rows = []
    generated_slots = 0
    batch_index = 0

    while generated_slots < args.num_samples:
        current_batch_size = min(
            args.adaptation_batch_size,
            args.num_samples - generated_slots,
        )
        proposals = controller.allocate(current_batch_size, rng)
        profiles = allocate_temperature_profiles(
            current_batch_size,
            args.conservative_fraction,
            rng,
        )
        templates = [
            rng.choice(template_bank[proposal.added_tokens])
            for proposal in proposals
        ]
        for template in templates:
            _validate_fixed_tokens(template, tokenizer)

        outputs = _sample_batch(args, model_bundle, templates, profiles)
        draw_counts = [1] * current_batch_size
        initial = []
        final = []
        diagnostics = []
        for index in range(current_batch_size):
            smiles, diagnostic = outputs.get(index, (None, {}))
            assessment = assess_candidate(smiles, spec)
            initial.append(assessment)
            final.append(assessment)
            diagnostics.append(diagnostic)

        pending = [
            index
            for index, assessment in enumerate(final)
            if not assessment.structural_success
        ]
        for _ in range(args.constraint_retries):
            if not pending:
                break
            retry_outputs = _sample_batch(
                args,
                model_bundle,
                [templates[index] for index in pending],
                [profiles[index] for index in pending],
            )
            next_pending = []
            for retry_index, original_index in enumerate(pending):
                draw_counts[original_index] += 1
                smiles, diagnostic = retry_outputs.get(retry_index, (None, {}))
                assessment = assess_candidate(smiles, spec)
                final[original_index] = assessment
                diagnostics[original_index] = diagnostic
                if not assessment.structural_success:
                    next_pending.append(original_index)
            pending = next_pending

        observations = []
        batch_structural = 0
        batch_novel = 0
        for local_index, (
            proposal,
            profile,
            template,
            first,
            assessment,
            diagnostic,
        ) in enumerate(
            zip(
                proposals,
                profiles,
                templates,
                initial,
                final,
                diagnostics,
            )
        ):
            is_novel = bool(
                assessment.structural_success
                and assessment.smiles is not None
                and assessment.smiles not in seen
            )
            if assessment.structural_success and assessment.smiles is not None:
                accepted.append(assessment.smiles)
                seen.add(assessment.smiles)
                batch_structural += 1
            batch_novel += int(is_novel)
            observations.append(
                (proposal, assessment.structural_success, is_novel)
            )
            temperatures = _temperature_kwargs(args, profile)
            attempts.append(
                {
                    "task": spec.task,
                    "name": spec.name,
                    "seed": args.seed,
                    "case_seed": case_seed,
                    "attempt": generated_slots + local_index,
                    "batch": batch_index,
                    "length_source": proposal.source,
                    "fixed_tokens": (
                        template.target_length - template.added_tokens
                    ),
                    "reference_fixed_tokens": reference_fixed,
                    "minimum_added_tokens": feasible_lengths[0],
                    "maximum_added_tokens": feasible_lengths[-1],
                    "target_length": template.target_length,
                    "added_tokens": template.added_tokens,
                    "attachment_count": template.attachment_count,
                    "temperature_profile": profile,
                    "temperature_start": temperatures["temperature_start"],
                    "temperature_end": temperatures["temperature_end"],
                    "temperature_power": temperatures["temperature_power"],
                    "draw_count": draw_counts[local_index],
                    "initial_model_output": first.valid,
                    "initial_structural_success": first.structural_success,
                    "recovered_structural": (
                        not first.structural_success
                        and assessment.structural_success
                    ),
                    "model_output": assessment.valid,
                    "smiles": assessment.smiles,
                    "connected": assessment.connected,
                    "no_dummies": assessment.no_dummies,
                    "preserved": assessment.preserved,
                    "required": assessment.required,
                    "structural_success": assessment.structural_success,
                    "novel": is_novel,
                    "mean_log_prob": diagnostic.get("mean_log_prob"),
                    "editable_tokens": diagnostic.get("editable_tokens"),
                }
            )

        controller.update(observations)
        for record in controller.records():
            controller_rows.append(
                {
                    "task": spec.task,
                    "name": spec.name,
                    "seed": args.seed,
                    "case_seed": case_seed,
                    "batch": batch_index,
                    **record,
                }
            )
        generated_slots += current_batch_size
        print(
            f"[length-search:{spec.name}] batch={batch_index} "
            f"structural={batch_structural/current_batch_size:.3f} "
            f"novel={batch_novel/max(1, batch_structural):.3f} "
            f"active_lengths={len(controller.active)}"
        )
        batch_index += 1

    return spec, accepted, attempts, controller_rows


def run_task(args):
    from tdc import Evaluator, Oracle

    args.task = normalize_task(args.task)
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    data = pd.read_csv(args.fragments_csv)
    length_prior = load_length_prior(args.length_prior)
    model_bundle = load_csdnet_model(args)
    oracle_qed = Oracle("qed")
    oracle_sa = Oracle("sa")
    diversity_evaluator = Evaluator("diversity")

    stem = f"{args.task}_seed{args.seed}"
    metrics_path = output_dir / f"metrics_{stem}.csv"
    samples_path = output_dir / f"samples_{stem}.csv"
    attempts_path = output_dir / f"attempts_{stem}.csv"
    controller_path = output_dir / f"length_controller_{stem}.csv"
    summary_path = output_dir / f"summary_{stem}.csv"
    metric_rows = _read_rows(metrics_path) if args.resume else []
    sample_rows = _read_rows(samples_path) if args.resume else []
    attempt_rows = _read_rows(attempts_path) if args.resume else []
    controller_rows = _read_rows(controller_path) if args.resume else []
    completed = {str(record["name"]) for record in metric_rows}

    for _, row in data.iterrows():
        name = str(row["name"])
        if name in completed:
            print(f"[{args.task}] {name}: already complete, skipping")
            continue
        print("=" * 72)
        print(
            f"[{args.task}] {name}: non-parametric length search "
            f"attempts={args.num_samples}"
        )
        spec, accepted, attempts, controller = run_case(
            args,
            model_bundle,
            row,
            length_prior,
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
                "method": "direct_infill_v4_length_search",
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
            }
        )
        metric_rows.append(metrics)
        attempt_rows.extend(attempts)
        controller_rows.extend(controller)
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
            f"draws={metrics['model_draws']}"
        )
        _write_rows(metrics_path, metric_rows)
        _write_rows(samples_path, sample_rows)
        _write_rows(attempts_path, attempt_rows)
        _write_rows(controller_path, controller_rows)

    frame = pd.DataFrame(metric_rows)
    summary = {
        "task": args.task,
        "seed": args.seed,
        "method": "direct_infill_v4_length_search",
        "n_cases": len(frame),
        "prior_floor": args.prior_floor,
        "length_exploration_fraction": args.length_exploration_fraction,
        "conservative_fraction": args.conservative_fraction,
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
    ):
        summary[f"{metric}_mean"] = (
            float(frame[metric].mean()) if len(frame) else 0.0
        )
    with summary_path.open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(summary))
        writer.writeheader()
        writer.writerow(summary)
    print(pd.DataFrame([summary]).to_string(index=False))
    print(f"Saved: {metrics_path}")
    print(f"Saved: {samples_path}")
    print(f"Saved: {attempts_path}")
    print(f"Saved: {controller_path}")
    print(f"Saved: {summary_path}")


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--task",
        required=True,
        choices=(*CANONICAL_TASKS, "superstructure_design"),
    )
    parser.add_argument("--ckpt_path", required=True)
    parser.add_argument("--vocab", default="csdnet_vocab.pkl")
    parser.add_argument("--fragments_csv", default="data/fragments.csv")
    parser.add_argument("--length_prior", default="data/len.pk")
    parser.add_argument(
        "--output_dir",
        default="CSDNet/exp/frag/results/direct_infill_v4_length_search",
    )
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--cpu", action="store_true")
    parser.add_argument("--resume", action="store_true")
    parser.add_argument("--num_samples", type=int, default=100)
    parser.add_argument("--min_added_tokens", type=int, default=4)
    parser.add_argument("--adaptation_batch_size", type=int, default=20)
    parser.add_argument("--warmup_samples", type=int, default=20)
    parser.add_argument("--prior_floor", type=float, default=0.65)
    parser.add_argument(
        "--length_exploration_fraction",
        type=float,
        default=0.10,
    )
    parser.add_argument("--ucb_scale", type=float, default=0.10)
    parser.add_argument(
        "--length_softmax_temperature",
        type=float,
        default=0.18,
    )
    parser.add_argument("--refinement_radius", type=int, default=2)
    parser.add_argument("--length_template_variants", type=int, default=3)
    parser.add_argument("--conservative_fraction", type=float, default=0.80)
    parser.add_argument("--constraint_retries", type=int, default=1)
    parser.add_argument("--max_len", type=int, default=128)
    parser.add_argument("--batch_size", type=int, default=64)
    parser.add_argument("--n_steps", type=int, default=160)
    parser.add_argument(
        "--conservative_temperature_start",
        type=float,
        default=1.15,
    )
    parser.add_argument(
        "--conservative_temperature_end",
        type=float,
        default=0.12,
    )
    parser.add_argument(
        "--conservative_temperature_power",
        type=float,
        default=1.8,
    )
    parser.add_argument("--explore_temperature_start", type=float, default=1.55)
    parser.add_argument("--explore_temperature_end", type=float, default=0.22)
    parser.add_argument("--explore_temperature_power", type=float, default=1.35)
    parser.add_argument("--disable_fsm_check", action="store_true")
    parser.add_argument("--disable_rdkit_kekulize_check", action="store_true")
    parser.add_argument("--rdkit_check_interval", type=int, default=25)
    parser.add_argument("--max_sample_retries", type=int, default=2)
    parser.add_argument("--violation_neighborhood", type=int, default=2)
    args = parser.parse_args()
    if args.adaptation_batch_size < 4:
        parser.error("--adaptation_batch_size must be at least four")
    if not 0 <= args.warmup_samples <= args.num_samples:
        parser.error("--warmup_samples must be in [0, num_samples]")
    if not 0.0 <= args.conservative_fraction <= 1.0:
        parser.error("--conservative_fraction must be in [0, 1]")
    if args.constraint_retries < 0:
        parser.error("--constraint_retries must be non-negative")
    if args.length_template_variants < 1:
        parser.error("--length_template_variants must be positive")
    return args


def main():
    args = parse_args()
    for path in (
        args.fragments_csv,
        args.length_prior,
        args.vocab,
        args.ckpt_path,
    ):
        if not Path(path).exists():
            raise SystemExit(f"Cannot find required file: {path}")
    run_task(args)


if __name__ == "__main__":
    main()
