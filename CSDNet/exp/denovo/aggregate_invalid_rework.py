#!/usr/bin/env python
"""Aggregate the diagnostic FSM-guided invalid-sample rework experiment."""

from __future__ import annotations

import argparse
import json
from collections import Counter
from pathlib import Path

import pandas as pd


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--input_dir", required=True)
    parser.add_argument("--steps", type=int, default=1000)
    parser.add_argument("--profile", default="elastic_loflex")
    parser.add_argument("--seeds", default="0,1,2")
    parser.add_argument("--output", required=True)
    args = parser.parse_args()

    seeds = [int(value) for value in args.seeds.split(",") if value.strip()]
    rows = []
    initial_reasons = Counter()
    residual_reasons = Counter()
    for seed in seeds:
        run_dir = (
            Path(args.input_dir)
            / f"steps{args.steps}"
            / args.profile
            / f"seed{seed}"
        )
        rework_dir = run_dir / "progressive_rework"
        base_metrics_path = run_dir / "genmol_denovo_metrics.json"
        rework_metrics_path = rework_dir / "genmol_denovo_metrics.json"
        summary_path = rework_dir / "invalid_rework_summary.json"
        for path in (base_metrics_path, rework_metrics_path, summary_path):
            if not path.exists():
                raise FileNotFoundError(f"Incomplete rework run: {path}")

        base = json.loads(base_metrics_path.read_text())
        reworked = json.loads(rework_metrics_path.read_text())
        summary = json.loads(summary_path.read_text())
        initial_reasons.update(summary.get("initial_reason_counts", {}))
        residual_reasons.update(summary.get("residual_reason_counts", {}))
        rows.append(
            {
                "seed": seed,
                "initial_invalid": int(summary["initial_invalid"]),
                "fsm_detected_invalid": int(summary["fsm_detected_invalid"]),
                "fsm_detection_fraction": float(summary["fsm_detection_fraction"]),
                "recovered": int(summary["recovered"]),
                "recovery_fraction": float(summary["recovery_fraction"]),
                "final_invalid": int(summary["final_invalid"]),
                "validity_before": float(base["Validity"]),
                "validity_after": float(reworked["Validity"]),
                "validity_delta": float(reworked["Validity"])
                - float(base["Validity"]),
                "quality_before": float(base["Quality"]),
                "quality_after": float(reworked["Quality"]),
                "quality_delta": float(reworked["Quality"])
                - float(base["Quality"]),
                "uniqueness_before": float(base["UniquenessTotal"]),
                "uniqueness_after": float(reworked["UniquenessTotal"]),
                "diversity_before": float(base["Diversity"]),
                "diversity_after": float(reworked["Diversity"]),
            }
        )

    runs = pd.DataFrame(rows).sort_values("seed")
    output = Path(args.output)
    output.parent.mkdir(parents=True, exist_ok=True)
    runs.to_csv(output, index=False)

    numeric = [column for column in runs.columns if column != "seed"]
    summary = runs[numeric].agg(["mean", "std"])
    summary_path = output.with_name(output.stem + "_mean_std.csv")
    summary.to_csv(summary_path)

    print(runs.to_string(index=False, float_format=lambda value: f"{value:.4f}"))
    print("\nThree-seed mean +/- std:")
    print(summary.to_string(float_format=lambda value: f"{value:.4f}"))
    print(f"\nInitial RDKit failure types: {dict(initial_reasons)}")
    print(f"Residual failure types: {dict(residual_reasons)}")
    print(f"Saved runs: {output}")
    print(f"Saved summary: {summary_path}")


if __name__ == "__main__":
    main()
