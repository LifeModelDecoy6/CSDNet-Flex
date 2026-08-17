#!/usr/bin/env python
"""Task-agnostic adaptive direct infilling for fragment constraints.

The sampling controller never reads the benchmark task name, target molecule,
QED, SA, diversity, or distance. It adapts only to prompt-conditioned length
priors, structural success, canonical-SMILES collisions, and model confidence.
Every requested output slot remains in the benchmark denominator.
"""

from __future__ import annotations

import argparse
import csv
import hashlib
import math
import random
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import pandas as pd
import torch
from rdkit import RDLogger

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


RDLogger.DisableLog("rdApp.*")


ARM_QUANTILES = {
    "compact": (0.05, 0.25),
    "balanced": (0.25, 0.50),
    "broad": (0.50, 0.75),
    "explore": (0.75, 0.95),
}
ARM_ORDER = tuple(ARM_QUANTILES)
ARM_TEMPERATURE_TARGETS = {
    "compact": (1.16, 0.11),
    "balanced": (1.28, 0.15),
    "broad": (1.43, 0.20),
    "explore": (1.58, 0.25),
}


@dataclass
class ArmStats:
    attempts: int = 0
    structural: int = 0
    novel: int = 0
    confidence_sum: float = 0.0
    confidence_count: int = 0

    @property
    def validity(self) -> float:
        # A light prior prevents five-sample warm-up batches from producing
        # brittle all-or-nothing decisions.
        return (self.structural + 2.0) / (self.attempts + 3.0)

    @property
    def novelty(self) -> float:
        return (self.novel + 1.0) / (self.structural + 2.0)

    @property
    def confidence(self) -> float:
        if self.confidence_count == 0:
            return float("-inf")
        return self.confidence_sum / self.confidence_count


def stable_case_seed(seed: int, fragment: str) -> int:
    payload = f"direct-infill-v3:{seed}:{fragment}".encode()
    return int.from_bytes(hashlib.sha256(payload).digest()[:4], "big")


def _fixed_token_count(args, spec, length_prior, case_seed):
    probe = build_masked_template(
        spec,
        max_len=args.max_len,
        length_prior=length_prior,
        min_added_tokens=args.min_added_tokens,
        rng=random.Random(case_seed ^ 0x6C8E9CF5),
        added_token_range=(args.min_added_tokens, args.min_added_tokens),
    )
    return max(0, int(probe.target_length) - int(probe.added_tokens))


def derive_length_arms(args, spec, length_prior, case_seed):
    """Derive residual-length arms from the global training length prior."""
    fixed_tokens = _fixed_token_count(args, spec, length_prior, case_seed)
    max_body_tokens = args.max_len - 2
    minimum_total = fixed_tokens + args.min_added_tokens
    feasible_totals = np.asarray(
        [
            int(length)
            for length in length_prior
            if minimum_total <= int(length) <= max_body_tokens
        ],
        dtype=np.int64,
    )
    if feasible_totals.size < 32:
        feasible_totals = np.asarray(
            [
                min(max_body_tokens, max(minimum_total, int(length)))
                for length in length_prior
            ],
            dtype=np.int64,
        )

    ranges = {}
    for arm, (lower_q, upper_q) in ARM_QUANTILES.items():
        lower_total, upper_total = np.quantile(
            feasible_totals,
            [lower_q, upper_q],
            method="nearest",
        )
        lower = max(args.min_added_tokens, int(lower_total) - fixed_tokens)
        upper = max(lower, int(upper_total) - fixed_tokens)
        ranges[arm] = (lower, upper)
    return fixed_tokens, ranges


def allocate_batch_arms(batch_size, selected_arm, warmup, rng):
    if batch_size <= 0:
        return []
    arms = []
    if warmup:
        for index in range(batch_size):
            arms.append(ARM_ORDER[index % len(ARM_ORDER)])
    else:
        floor = min(len(ARM_ORDER), batch_size)
        arms.extend(ARM_ORDER[:floor])
        arms.extend([selected_arm] * (batch_size - floor))
    rng.shuffle(arms)
    return arms


def choose_arm(stats, total_attempts, validity_floor, novelty_floor, ucb_scale):
    eligible = [
        arm
        for arm in ARM_ORDER
        if stats[arm].validity >= validity_floor
        and stats[arm].novelty >= novelty_floor
        and math.isfinite(stats[arm].confidence)
    ]
    if eligible:
        return max(
            eligible,
            key=lambda arm: (
                stats[arm].confidence
                + ucb_scale
                * math.sqrt(
                    math.log(max(2, total_attempts))
                    / max(1, stats[arm].confidence_count)
                ),
                -ARM_ORDER.index(arm),
            ),
        )

    structurally_feasible = [
        arm for arm in ARM_ORDER if stats[arm].validity >= validity_floor
    ]
    if structurally_feasible:
        return max(
            structurally_feasible,
            key=lambda arm: (
                stats[arm].novelty,
                stats[arm].validity,
                -ARM_ORDER.index(arm),
            ),
        )
    return max(
        ARM_ORDER,
        key=lambda arm: (
            stats[arm].validity,
            stats[arm].novelty,
            -ARM_ORDER.index(arm),
        ),
    )


def update_temperature(
    *,
    selected_arm,
    current_start,
    current_end,
    batch_validity,
    batch_novelty,
    validity_floor,
    novelty_floor,
    step,
):
    target_start, target_end = ARM_TEMPERATURE_TARGETS[selected_arm]
    start = 0.70 * current_start + 0.30 * target_start
    end = 0.70 * current_end + 0.30 * target_end
    if batch_validity < validity_floor:
        start -= step
        end -= 0.25 * step
    elif batch_novelty < novelty_floor:
        start += step
        end += 0.25 * step
    return (
        float(np.clip(start, 0.95, 1.70)),
        float(np.clip(end, 0.08, 0.30)),
    )


def _sample_batch(args, model_bundle, templates, temperature_start, temperature_end):
    model, tokenizer, device = model_bundle
    generated = sample_csdnet_local_remask(
        model=model,
        tk=tokenizer,
        seed_smiles=[template.seed_smiles for template in templates],
        max_len=args.max_len,
        device=device,
        batch_size=args.batch_size,
        n_steps=args.n_steps,
        use_fsm_check=not args.disable_fsm_check,
        use_rdkit_kekulize_check=not args.disable_rdkit_kekulize_check,
        rdkit_check_interval=args.rdkit_check_interval,
        max_sample_retries=args.max_sample_retries,
        violation_neighborhood=args.violation_neighborhood,
        edit_plans=[list(template.edit_plans) for template in templates],
        return_seed_indices=True,
        return_diagnostics=True,
        temperature_start=temperature_start,
        temperature_end=temperature_end,
        temperature_power=args.temperature_power,
    )
    return {
        int(local_index): (smiles, diagnostics)
        for smiles, local_index, diagnostics in generated
    }


def _controller_records(
    *,
    spec,
    seed,
    case_seed,
    batch_index,
    phase,
    selected_arm,
    next_selected_arm,
    temperature_start,
    temperature_end,
    batch_validity,
    batch_novelty,
    fixed_tokens,
    length_ranges,
    stats,
):
    rows = []
    for arm in ARM_ORDER:
        item = stats[arm]
        rows.append(
            {
                "task": spec.task,
                "name": spec.name,
                "seed": seed,
                "case_seed": case_seed,
                "batch": batch_index,
                "phase": phase,
                "selected_arm": selected_arm,
                "next_selected_arm": next_selected_arm,
                "temperature_start": temperature_start,
                "temperature_end": temperature_end,
                "batch_validity": batch_validity,
                "batch_novelty": batch_novelty,
                "fixed_tokens": fixed_tokens,
                "arm": arm,
                "length_min": length_ranges[arm][0],
                "length_max": length_ranges[arm][1],
                "attempts": item.attempts,
                "structural": item.structural,
                "novel": item.novel,
                "validity_estimate": item.validity,
                "novelty_estimate": item.novelty,
                "mean_log_prob": item.confidence,
            }
        )
    return rows


def run_case(args, model_bundle, row, length_prior):
    spec = build_constraint_spec(args.task, row)
    case_seed = stable_case_seed(args.seed, spec.fragment)
    random.seed(case_seed)
    np.random.seed(case_seed)
    torch.manual_seed(case_seed)
    rng = random.Random(case_seed)
    _, tokenizer, _ = model_bundle

    fixed_tokens, length_ranges = derive_length_arms(
        args,
        spec,
        length_prior,
        case_seed,
    )
    stats = {arm: ArmStats() for arm in ARM_ORDER}
    seen = set()
    accepted = []
    attempts = []
    controller = []
    selected_arm = "balanced"
    temperature_start = args.temperature_start
    temperature_end = args.temperature_end
    generated_slots = 0
    batch_index = 0

    while generated_slots < args.num_samples:
        current_batch_size = min(args.adaptation_batch_size, args.num_samples - generated_slots)
        warmup = generated_slots < args.warmup_samples
        phase = "warmup" if warmup else "adaptive"
        batch_arms = allocate_batch_arms(
            current_batch_size,
            selected_arm,
            warmup,
            rng,
        )
        templates = [
            build_masked_template(
                spec,
                max_len=args.max_len,
                length_prior=length_prior,
                min_added_tokens=args.min_added_tokens,
                rng=rng,
                added_token_range=length_ranges[arm],
            )
            for arm in batch_arms
        ]
        for template in templates:
            _validate_fixed_tokens(template, tokenizer)

        outputs = _sample_batch(
            args,
            model_bundle,
            templates,
            temperature_start,
            temperature_end,
        )
        draw_counts = [1] * current_batch_size
        initial_assessments = []
        final_assessments = []
        diagnostics = []
        for local_index in range(current_batch_size):
            smiles, diagnostic = outputs.get(local_index, (None, {}))
            assessment = assess_candidate(smiles, spec)
            initial_assessments.append(assessment)
            final_assessments.append(assessment)
            diagnostics.append(diagnostic)

        pending = [
            index
            for index, assessment in enumerate(final_assessments)
            if not assessment.structural_success
        ]
        for _ in range(args.constraint_retries):
            if not pending:
                break
            retry_templates = [templates[index] for index in pending]
            retry_outputs = _sample_batch(
                args,
                model_bundle,
                retry_templates,
                temperature_start,
                temperature_end,
            )
            next_pending = []
            for retry_index, original_index in enumerate(pending):
                draw_counts[original_index] += 1
                smiles, diagnostic = retry_outputs.get(retry_index, (None, {}))
                assessment = assess_candidate(smiles, spec)
                final_assessments[original_index] = assessment
                diagnostics[original_index] = diagnostic
                if not assessment.structural_success:
                    next_pending.append(original_index)
            pending = next_pending

        batch_structural = 0
        batch_novel = 0
        for local_index, (arm, template, initial, final, diagnostic) in enumerate(
            zip(
                batch_arms,
                templates,
                initial_assessments,
                final_assessments,
                diagnostics,
            )
        ):
            absolute_index = generated_slots + local_index
            is_novel = bool(
                final.structural_success
                and final.smiles is not None
                and final.smiles not in seen
            )
            if final.structural_success and final.smiles is not None:
                accepted.append(final.smiles)
                seen.add(final.smiles)
                batch_structural += 1
            if is_novel:
                batch_novel += 1

            item = stats[arm]
            item.attempts += 1
            item.structural += int(final.structural_success)
            item.novel += int(is_novel)
            mean_log_prob = float(diagnostic.get("mean_log_prob", float("-inf")))
            if final.structural_success and math.isfinite(mean_log_prob):
                item.confidence_sum += mean_log_prob
                item.confidence_count += 1

            attempts.append(
                {
                    "task": spec.task,
                    "name": spec.name,
                    "seed": args.seed,
                    "case_seed": case_seed,
                    "attempt": absolute_index,
                    "batch": batch_index,
                    "phase": phase,
                    "arm": arm,
                    "selected_arm": selected_arm,
                    "fixed_tokens": fixed_tokens,
                    "profile_length_min": length_ranges[arm][0],
                    "profile_length_max": length_ranges[arm][1],
                    "temperature_start": temperature_start,
                    "temperature_end": temperature_end,
                    "temperature_power": args.temperature_power,
                    "target_length": template.target_length,
                    "added_tokens": template.added_tokens,
                    "attachment_count": template.attachment_count,
                    "draw_count": draw_counts[local_index],
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
                    "novel": is_novel,
                    "mean_log_prob": mean_log_prob,
                    "editable_tokens": int(diagnostic.get("editable_tokens", 0)),
                }
            )

        generated_slots += current_batch_size
        batch_validity = batch_structural / max(1, current_batch_size)
        batch_novelty = batch_novel / max(1, batch_structural)
        next_selected_arm = choose_arm(
            stats,
            generated_slots,
            args.validity_floor,
            args.novelty_floor,
            args.ucb_scale,
        )
        controller.extend(
            _controller_records(
                spec=spec,
                seed=args.seed,
                case_seed=case_seed,
                batch_index=batch_index,
                phase=phase,
                selected_arm=selected_arm,
                next_selected_arm=next_selected_arm,
                temperature_start=temperature_start,
                temperature_end=temperature_end,
                batch_validity=batch_validity,
                batch_novelty=batch_novelty,
                fixed_tokens=fixed_tokens,
                length_ranges=length_ranges,
                stats=stats,
            )
        )
        selected_arm = next_selected_arm
        temperature_start, temperature_end = update_temperature(
            selected_arm=selected_arm,
            current_start=temperature_start,
            current_end=temperature_end,
            batch_validity=batch_validity,
            batch_novelty=batch_novelty,
            validity_floor=args.validity_floor,
            novelty_floor=args.novelty_floor,
            step=args.temperature_step,
        )
        print(
            f"[adaptive:{spec.name}] batch={batch_index} phase={phase} "
            f"validity={batch_validity:.3f} novelty={batch_novelty:.3f} "
            f"next={selected_arm} temp={temperature_start:.3f}/{temperature_end:.3f}"
        )
        batch_index += 1

    return spec, accepted, attempts, controller


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
    controller_path = output_dir / f"controller_{stem}.csv"
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
            f"[{args.task}] {name}: direct-v3 samples={args.num_samples} "
            f"batch={args.adaptation_batch_size} warmup={args.warmup_samples}"
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
                "method": "direct_infill_v3_adaptive",
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
        "method": "direct_infill_v3_adaptive",
        "n_cases": len(frame),
        "adaptation_batch_size": args.adaptation_batch_size,
        "warmup_samples": args.warmup_samples,
        "validity_floor": args.validity_floor,
        "novelty_floor": args.novelty_floor,
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
        summary[f"{metric}_mean"] = float(frame[metric].mean()) if len(frame) else 0.0
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
        "--task", required=True, choices=(*CANONICAL_TASKS, "superstructure_design")
    )
    parser.add_argument("--ckpt_path", required=True)
    parser.add_argument("--vocab", default="csdnet_vocab.pkl")
    parser.add_argument("--fragments_csv", default="data/fragments.csv")
    parser.add_argument("--length_prior", default="data/len.pk")
    parser.add_argument(
        "--output_dir",
        default="CSDNet/exp/frag/results/direct_infill_v3_adaptive",
    )
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--cpu", action="store_true")
    parser.add_argument("--resume", action="store_true")
    parser.add_argument("--num_samples", type=int, default=100)
    parser.add_argument("--min_added_tokens", type=int, default=4)
    parser.add_argument("--adaptation_batch_size", type=int, default=20)
    parser.add_argument("--warmup_samples", type=int, default=20)
    parser.add_argument("--validity_floor", type=float, default=0.78)
    parser.add_argument("--novelty_floor", type=float, default=0.50)
    parser.add_argument("--ucb_scale", type=float, default=0.08)
    parser.add_argument("--temperature_start", type=float, default=1.30)
    parser.add_argument("--temperature_end", type=float, default=0.16)
    parser.add_argument("--temperature_power", type=float, default=1.65)
    parser.add_argument("--temperature_step", type=float, default=0.08)
    parser.add_argument("--constraint_retries", type=int, default=1)
    parser.add_argument("--max_len", type=int, default=128)
    parser.add_argument("--batch_size", type=int, default=64)
    parser.add_argument("--n_steps", type=int, default=160)
    parser.add_argument("--disable_fsm_check", action="store_true")
    parser.add_argument("--disable_rdkit_kekulize_check", action="store_true")
    parser.add_argument("--rdkit_check_interval", type=int, default=25)
    parser.add_argument("--max_sample_retries", type=int, default=2)
    parser.add_argument("--violation_neighborhood", type=int, default=2)
    args = parser.parse_args()
    if args.adaptation_batch_size < len(ARM_ORDER):
        parser.error("--adaptation_batch_size must be at least four")
    if not 0 <= args.warmup_samples <= args.num_samples:
        parser.error("--warmup_samples must be in [0, num_samples]")
    if not 0.0 < args.validity_floor <= 1.0:
        parser.error("--validity_floor must be in (0, 1]")
    if not 0.0 < args.novelty_floor <= 1.0:
        parser.error("--novelty_floor must be in (0, 1]")
    if args.constraint_retries < 0:
        parser.error("--constraint_retries must be non-negative")
    return args


def main():
    args = parse_args()
    for path in (args.fragments_csv, args.length_prior, args.vocab, args.ckpt_path):
        if not Path(path).exists():
            raise SystemExit(f"Cannot find required file: {path}")
    run_task(args)


if __name__ == "__main__":
    main()
