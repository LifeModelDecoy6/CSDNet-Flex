#!/usr/bin/env python
"""Paired three-seed comparison for the ElasticCSDNet FSM ablation."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import pandas as pd


METRICS = (
    "Validity",
    "UniquenessTotal",
    "Quality",
    "Diversity",
    "quality_diversity_product",
    "proposal_acceptance",
    "sanitization_rejection_rate",
)


def _read_boolean_from_log(path: Path, label: str) -> bool | None:
    if not path.exists():
        return None
    prefix = f"{label}:"
    for line in path.read_text().splitlines():
        if line.startswith(prefix):
            value = line.split(":", 1)[1].strip().lower()
            if value in {"true", "false"}:
                return value == "true"
    return None


def _load_run(
    base: Path,
    steps: int,
    profile: str,
    seed: int,
    condition: str,
    protocol: str,
    expected_rdkit: bool = True,
):
    run_dir = base / f"steps{steps}" / profile / f"seed{seed}"
    metrics_path = run_dir / "genmol_denovo_metrics.json"
    diagnostics_path = run_dir / "sampling_diagnostics.json"
    if not metrics_path.exists() or not diagnostics_path.exists():
        raise FileNotFoundError(f"Incomplete run: {run_dir}")

    metrics = json.loads(metrics_path.read_text())
    diagnostics = json.loads(diagnostics_path.read_text())
    fsm_check = diagnostics.get("fsm_check")
    if fsm_check is None:
        fsm_check = _read_boolean_from_log(run_dir / "metrics_log.txt", "FSM check")
    rdkit_check = diagnostics.get("rdkit_kekulize_check")
    if rdkit_check is None:
        rdkit_check = _read_boolean_from_log(
            run_dir / "metrics_log.txt", "RDKit kekulize check"
        )
    strict = bool(diagnostics.get("strict_final_sanitize", False))
    expected_fsm = condition in {"fsm_on", "full_constraints"}
    if fsm_check is not expected_fsm:
        raise ValueError(
            f"{run_dir}: expected FSM={expected_fsm}, observed {fsm_check}"
        )
    if rdkit_check is not expected_rdkit:
        raise ValueError(
            f"{run_dir}: expected RDKit repair={expected_rdkit}, "
            f"observed {rdkit_check}"
        )
    if condition == "full_constraints":
        expected_mode = "online_scan_progressive_completion_neural_then_projection"
        if diagnostics.get("fsm_constraint_mode") != expected_mode:
            raise ValueError(
                f"{run_dir}: complete constraint system was not executed"
            )
    elif condition == "unconstrained":
        if diagnostics.get("fsm_constraint_mode") != "disabled":
            raise ValueError(f"{run_dir}: constraint system was not disabled")
        active_repair_keys = (
            "fsm_neural_repair_rows",
            "fsm_neural_repair_rounds",
            "fsm_hard_projection_rows",
            "fsm_online_online_repair_rows",
            "fsm_online_online_repair_events",
        )
        active = {
            key: diagnostics.get(key)
            for key in active_repair_keys
            if int(diagnostics.get(key, 0) or 0) != 0
        }
        if active:
            raise ValueError(
                f"{run_dir}: unconstrained run contains repair activity: {active}"
            )
    proposals = int(diagnostics.get("proposals", 0))
    accepted = int(diagnostics.get("accepted", 0))
    rejected = int(diagnostics.get("sanitization_rejections", 0))
    empty_rejections = int(diagnostics.get("empty_rejections", 0))
    if proposals <= 0 or accepted <= 0:
        raise ValueError(f"{run_dir}: invalid proposal diagnostics")
    if int(metrics.get("TotalRequested", 0)) != 1000:
        raise ValueError(f"{run_dir}: expected a 1000-molecule denominator")
    if protocol == "strict":
        if not strict:
            raise ValueError(
                f"{run_dir}: strict final sanitization must remain enabled"
            )
    else:
        if strict:
            raise ValueError(f"{run_dir}: raw protocol requires strict=False")
        if proposals != 1000 or accepted != 1000:
            raise ValueError(
                f"{run_dir}: raw protocol requires exactly 1000 proposals "
                f"and returned samples, observed {proposals}/{accepted}"
            )
        if rejected != 0 or empty_rejections != 0:
            raise ValueError(
                f"{run_dir}: raw protocol unexpectedly refilled/rejected samples"
            )
        if int(metrics.get("TotalGenerated", 0)) != 1000:
            raise ValueError(f"{run_dir}: raw metric file must contain 1000 samples")

    return {
        "condition": condition,
        "seed": seed,
        "Validity": float(metrics["Validity"]),
        "UniquenessTotal": float(metrics["UniquenessTotal"]),
        "Quality": float(metrics["Quality"]),
        "Diversity": float(metrics["Diversity"]),
        "quality_diversity_product": float(metrics["Quality"])
        * float(metrics["Diversity"]),
        "proposals": proposals,
        "accepted": accepted,
        "sanitization_rejections": rejected,
        "proposal_acceptance": accepted / proposals,
        "sanitization_rejection_rate": rejected / proposals,
    }


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--fsm_on_dir", required=True)
    parser.add_argument("--fsm_off_dir", required=True)
    parser.add_argument("--steps", type=int, default=1000)
    parser.add_argument("--seeds", default="0,1,2")
    parser.add_argument("--fsm_on_profile", default="elastic_loflex")
    parser.add_argument("--fsm_off_profile", default="elastic_loflex_fsm_off")
    parser.add_argument(
        "--ablation_scope",
        choices=("fsm_increment", "full_constraint_system"),
        default="fsm_increment",
        help=(
            "fsm_increment keeps RDKit repair enabled in both conditions; "
            "full_constraint_system compares the complete FSM/RDKit system "
            "against a condition with both checks disabled"
        ),
    )
    parser.add_argument(
        "--protocol",
        choices=("strict", "raw"),
        default="strict",
        help=(
            "strict compares rejection-and-refill production runs; raw requires "
            "exactly 1000 proposals with strict sanitization disabled"
        ),
    )
    parser.add_argument("--output", required=True)
    args = parser.parse_args()

    seeds = [int(value) for value in args.seeds.split(",") if value.strip()]
    if args.ablation_scope == "full_constraint_system":
        on_condition = "full_constraints"
        off_condition = "unconstrained"
        off_expected_rdkit = False
    else:
        on_condition = "fsm_on"
        off_condition = "fsm_off"
        off_expected_rdkit = True
    rows = []
    for seed in seeds:
        rows.append(
            _load_run(
                Path(args.fsm_on_dir),
                args.steps,
                args.fsm_on_profile,
                seed,
                on_condition,
                args.protocol,
                expected_rdkit=True,
            )
        )
        rows.append(
            _load_run(
                Path(args.fsm_off_dir),
                args.steps,
                args.fsm_off_profile,
                seed,
                off_condition,
                args.protocol,
                expected_rdkit=off_expected_rdkit,
            )
        )

    runs = pd.DataFrame(rows).sort_values(["seed", "condition"])
    summary = runs.groupby("condition")[list(METRICS)].agg(["mean", "std"])
    summary.columns = [f"{metric}_{stat}" for metric, stat in summary.columns]
    summary = summary.reset_index()

    indexed = runs.set_index(["seed", "condition"])
    deltas = []
    for seed in seeds:
        on = indexed.loc[(seed, on_condition)]
        off = indexed.loc[(seed, off_condition)]
        row = {"seed": seed}
        for metric in METRICS:
            row[f"{metric}_off_minus_on"] = float(off[metric] - on[metric])
        row["proposals_off_minus_on"] = int(off["proposals"] - on["proposals"])
        deltas.append(row)
    paired = pd.DataFrame(deltas)

    output = Path(args.output)
    output.parent.mkdir(parents=True, exist_ok=True)
    runs.to_csv(output.with_name(output.stem + "_runs.csv"), index=False)
    paired.to_csv(output.with_name(output.stem + "_paired.csv"), index=False)
    summary.to_csv(output, index=False)

    display = (
        runs[
            [
                "seed",
                "condition",
                "Validity",
                "UniquenessTotal",
                "Quality",
                "Diversity",
                "proposals",
                "proposal_acceptance",
                "sanitization_rejection_rate",
            ]
        ]
        .sort_values(["seed", "condition"], ascending=[True, False])
    )
    print(display.to_string(index=False, float_format=lambda x: f"{x:.4f}"))
    print(f"\nProtocol: {args.protocol}")
    print("Three-seed mean +/- std:")
    print(summary.to_string(index=False, float_format=lambda x: f"{x:.4f}"))
    print(f"\nPaired effect ({off_condition} minus {on_condition}):")
    if args.protocol == "raw":
        print("  Negative validity/quality/uniqueness deltas favor FSM-on.")
    else:
        print(
            "  Negative performance/acceptance deltas and positive "
            "proposal/rejection deltas favor FSM-on."
        )
    print(paired.to_string(index=False, float_format=lambda x: f"{x:+.4f}"))
    print(f"\nSaved summary: {output}")
    print(f"Saved runs: {output.with_name(output.stem + '_runs.csv')}")
    print(f"Saved paired deltas: {output.with_name(output.stem + '_paired.csv')}")


if __name__ == "__main__":
    main()
