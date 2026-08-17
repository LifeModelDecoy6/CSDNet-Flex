#!/usr/bin/env python
"""Audit and summarize the raw 2 x 2 FSM/RDKit de novo ablation."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import pandas as pd


CONDITIONS = {
    "none": {
        "fsm": False,
        "rdkit": False,
        "profile": "elastic_loflex_constraint_none",
    },
    "fsm_only": {
        "fsm": True,
        "rdkit": False,
        "profile": "elastic_loflex_constraint_fsm_only",
    },
    "rdkit_only": {
        "fsm": False,
        "rdkit": True,
        "profile": "elastic_loflex_constraint_rdkit_only",
    },
    "full": {
        "fsm": True,
        "rdkit": True,
        "profile": "elastic_loflex_constraint_full",
    },
}

PERFORMANCE_METRICS = (
    "Validity",
    "UniquenessTotal",
    "Quality",
    "Diversity",
    "quality_diversity_product",
)

COST_METRICS = (
    "elapsed_seconds",
    "fsm_checked_rows",
    "rdkit_checked_rows",
    "online_repair_events",
    "online_repair_rows",
    "online_remasked_tokens",
    "final_neural_repair_rows",
    "final_neural_repair_rounds",
    "final_neural_recovered_rows",
    "hard_projection_rows",
    "final_invalid_rows",
)


def _integer(diagnostics, key):
    return int(diagnostics.get(key, 0) or 0)


def load_run(base, condition, seed, steps, expected_checkpoint=None):
    spec = CONDITIONS[condition]
    run_dir = Path(base) / condition / f"steps{steps}" / spec["profile"] / f"seed{seed}"
    metrics_path = run_dir / "genmol_denovo_metrics.json"
    diagnostics_path = run_dir / "sampling_diagnostics.json"
    if not metrics_path.is_file() or not diagnostics_path.is_file():
        raise FileNotFoundError(f"Incomplete factorial run: {run_dir}")

    metrics = json.loads(metrics_path.read_text())
    diagnostics = json.loads(diagnostics_path.read_text())
    observed_checkpoint = str(diagnostics.get("checkpoint", ""))
    checks = {
        "profile": diagnostics.get("sampler_profile") == spec["profile"],
        "seed": int(diagnostics.get("seed", -1)) == int(seed),
        "steps": int(diagnostics.get("n_steps", -1)) == int(steps),
        "fsm": diagnostics.get("fsm_check") is spec["fsm"],
        "rdkit": diagnostics.get("rdkit_kekulize_check") is spec["rdkit"],
        "fsm_initialized": diagnostics.get("fsm_tracker_active") is spec["fsm"],
        "rdkit_initialized": diagnostics.get("rdkit_constraint_active")
        is spec["rdkit"],
        "strict_sanitize_off": diagnostics.get("strict_final_sanitize") is False,
        "fixed_proposals": _integer(diagnostics, "proposals") == 1000,
        "all_returned": _integer(diagnostics, "accepted") == 1000,
        "no_empty_rejection": _integer(diagnostics, "empty_rejections") == 0,
        "no_sanitize_rejection": _integer(diagnostics, "sanitization_rejections") == 0,
        "requested_denominator": int(metrics.get("TotalRequested", -1)) == 1000,
        "generated_denominator": int(metrics.get("TotalGenerated", -1)) == 1000,
    }
    if expected_checkpoint:
        checks["checkpoint"] = Path(observed_checkpoint).resolve() == Path(
            expected_checkpoint
        ).resolve()
    failed = [name for name, passed in checks.items() if not passed]
    if failed:
        raise ValueError(
            f"Protocol audit failed for {condition}/seed{seed}: {', '.join(failed)}"
        )

    constraint_mode = diagnostics.get("fsm_constraint_mode")
    if spec["fsm"] and constraint_mode != "online_scan_progressive_completion_neural_then_projection":
        raise ValueError(f"FSM path not executed for {condition}/seed{seed}")
    if not spec["fsm"] and constraint_mode != "disabled":
        raise ValueError(f"FSM path unexpectedly active for {condition}/seed{seed}")

    row = {
        "condition": condition,
        "seed": int(seed),
        "fsm": spec["fsm"],
        "rdkit": spec["rdkit"],
        "checkpoint": observed_checkpoint,
        "Validity": float(metrics["Validity"]),
        "UniquenessTotal": float(metrics["UniquenessTotal"]),
        "Quality": float(metrics["Quality"]),
        "Diversity": float(metrics["Diversity"]),
    }
    row["quality_diversity_product"] = row["Quality"] * row["Diversity"]
    row.update(
        {
            "elapsed_seconds": float(diagnostics.get("ablation_elapsed_seconds", 0.0)),
            "fsm_checked_rows": _integer(diagnostics, "fsm_online_fsm_checked_rows"),
            "rdkit_checked_rows": _integer(diagnostics, "fsm_online_rdkit_checked_rows"),
            "online_repair_events": _integer(
                diagnostics, "fsm_online_online_repair_events"
            ),
            "online_repair_rows": _integer(
                diagnostics, "fsm_online_online_repair_rows"
            ),
            "online_remasked_tokens": _integer(
                diagnostics, "fsm_online_online_remasked_tokens"
            ),
            "final_neural_repair_rows": _integer(
                diagnostics, "fsm_neural_repair_rows"
            ),
            "final_neural_repair_rounds": _integer(
                diagnostics, "fsm_neural_repair_rounds"
            ),
            "final_neural_recovered_rows": _integer(
                diagnostics, "fsm_neural_recovered_rows"
            ),
            "hard_projection_rows": _integer(
                diagnostics, "fsm_hard_projection_rows"
            ),
            "final_invalid_rows": _integer(diagnostics, "fsm_final_invalid_rows"),
        }
    )
    return row


def factorial_effects(runs, seeds):
    indexed = runs.set_index(["seed", "condition"])
    rows = []
    for seed in seeds:
        values = {condition: indexed.loc[(seed, condition)] for condition in CONDITIONS}
        for metric in PERFORMANCE_METRICS:
            y00 = float(values["none"][metric])
            y10 = float(values["fsm_only"][metric])
            y01 = float(values["rdkit_only"][metric])
            y11 = float(values["full"][metric])
            rows.append(
                {
                    "seed": seed,
                    "metric": metric,
                    "fsm_main_effect": 0.5 * ((y10 - y00) + (y11 - y01)),
                    "rdkit_main_effect": 0.5 * ((y01 - y00) + (y11 - y10)),
                    "interaction": y11 - y10 - y01 + y00,
                    "fsm_only_minus_none": y10 - y00,
                    "rdkit_only_minus_none": y01 - y00,
                    "full_minus_none": y11 - y00,
                    "full_minus_rdkit_only": y11 - y01,
                    "full_minus_fsm_only": y11 - y10,
                }
            )
    return pd.DataFrame(rows)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--base_dir", required=True)
    parser.add_argument("--steps", type=int, default=1000)
    parser.add_argument("--seeds", default="0,1,2")
    parser.add_argument("--checkpoint", default=None)
    parser.add_argument("--output", required=True)
    args = parser.parse_args()

    seeds = [int(value) for value in args.seeds.split(",") if value.strip()]
    rows = [
        load_run(
            args.base_dir,
            condition,
            seed,
            args.steps,
            expected_checkpoint=args.checkpoint,
        )
        for seed in seeds
        for condition in CONDITIONS
    ]
    runs = pd.DataFrame(rows).sort_values(["seed", "fsm", "rdkit"])
    summary = (
        runs.groupby(["condition", "fsm", "rdkit"])[
            list(PERFORMANCE_METRICS + COST_METRICS)
        ]
        .agg(["mean", "std"])
        .reset_index()
    )
    summary.columns = [
        "_".join(str(part) for part in column if str(part))
        if isinstance(column, tuple)
        else str(column)
        for column in summary.columns
    ]
    effects = factorial_effects(runs, seeds)
    effect_columns = [
        column for column in effects.columns if column not in {"seed", "metric"}
    ]
    effect_summary = (
        effects.groupby("metric")[effect_columns]
        .agg(["mean", "std"])
        .reset_index()
    )
    effect_summary.columns = [
        "_".join(str(part) for part in column if str(part))
        if isinstance(column, tuple)
        else str(column)
        for column in effect_summary.columns
    ]

    output = Path(args.output)
    output.parent.mkdir(parents=True, exist_ok=True)
    runs.to_csv(output.with_name(output.stem + "_runs.csv"), index=False)
    summary.to_csv(output, index=False)
    effects.to_csv(output.with_name(output.stem + "_effects_by_seed.csv"), index=False)
    effect_summary.to_csv(
        output.with_name(output.stem + "_effects_summary.csv"), index=False
    )

    print("\nPer-seed raw results:")
    print(
        runs[
            ["seed", "condition", *PERFORMANCE_METRICS, *COST_METRICS]
        ].to_string(index=False, float_format=lambda value: f"{value:.4f}")
    )
    print("\nThree-seed condition means +/- sample SD:")
    print(summary.to_string(index=False, float_format=lambda value: f"{value:.4f}"))
    print("\nFactorial effects (positive improves the named metric):")
    print(
        effect_summary.to_string(
            index=False,
            float_format=lambda value: f"{value:+.4f}",
        )
    )
    print(f"\nSaved summary: {output}")


if __name__ == "__main__":
    main()
