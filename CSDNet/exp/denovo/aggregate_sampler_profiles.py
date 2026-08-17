#!/usr/bin/env python
"""Aggregate fixed-budget de novo sampler profiles across seeds."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import pandas as pd

from CSDNet.exp.denovo.sampler_profiles import SAMPLER_PROFILES


METRICS = (
    "Validity",
    "Uniqueness",
    "UniquenessTotal",
    "Quality",
    "Diversity",
    "MeanQEDUniqueValid",
    "MeanSAUniqueValid",
)


def parse_csv(text):
    return [item.strip() for item in text.split(",") if item.strip()]


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--input_dir", required=True)
    parser.add_argument(
        "--profiles",
        default="length_quality,genmol_quality,confidence_quality,balanced_quality",
    )
    parser.add_argument("--seeds", default="0,1,2")
    parser.add_argument("--allow_incomplete", action="store_true")
    parser.add_argument("--output", default=None)
    args = parser.parse_args()

    base = Path(args.input_dir)
    profiles = parse_csv(args.profiles)
    seeds = [int(value) for value in parse_csv(args.seeds)]
    unknown = sorted(set(profiles) - set(SAMPLER_PROFILES))
    if unknown:
        raise SystemExit(f"Unknown profiles: {', '.join(unknown)}")

    rows = []
    missing = []
    for profile in profiles:
        for seed in seeds:
            run_dir = base / profile / f"seed{seed}"
            metric_path = run_dir / "genmol_denovo_metrics.json"
            diagnostic_path = run_dir / "sampling_diagnostics.json"
            if not metric_path.exists():
                missing.append(f"{profile}:seed{seed}")
                continue
            metrics = json.loads(metric_path.read_text())
            diagnostics = (
                json.loads(diagnostic_path.read_text())
                if diagnostic_path.exists()
                else {}
            )
            proposals = int(diagnostics.get("proposals", metrics.get("TotalGenerated", 0)))
            accepted = int(diagnostics.get("accepted", metrics.get("TotalGenerated", 0)))
            row = {
                "profile": profile,
                "seed": seed,
                "proposals": proposals,
                "accepted": accepted,
                "proposal_acceptance": accepted / proposals if proposals else 0.0,
                "sanitization_rejections": int(
                    diagnostics.get("sanitization_rejections", 0)
                ),
            }
            row.update({metric: metrics.get(metric) for metric in METRICS})
            rows.append(row)

    if missing and not args.allow_incomplete:
        raise SystemExit("Missing runs: " + ", ".join(missing))
    if not rows:
        raise SystemExit(f"No completed sampler runs found in {base}")

    frame = pd.DataFrame(rows).sort_values(["profile", "seed"])
    summary = frame.groupby("profile")[list(METRICS)].agg(["mean", "std"])
    summary.columns = [f"{metric}_{stat}" for metric, stat in summary.columns]
    summary = summary.reset_index()
    counts = frame.groupby("profile").size().rename("n_seeds").reset_index()
    summary = counts.merge(summary, on="profile")
    summary = summary.sort_values(
        ["Quality_mean", "Diversity_mean"],
        ascending=[False, False],
    )

    output = Path(args.output) if args.output else base / "sampler_profile_summary.csv"
    output.parent.mkdir(parents=True, exist_ok=True)
    frame.to_csv(output.with_name(output.stem + "_runs.csv"), index=False)
    summary.to_csv(output, index=False)

    print(summary.to_string(index=False, float_format=lambda value: f"{value:.4f}"))
    if missing:
        print("Missing:", ", ".join(missing))
    print(f"Saved: {output}")


if __name__ == "__main__":
    main()
