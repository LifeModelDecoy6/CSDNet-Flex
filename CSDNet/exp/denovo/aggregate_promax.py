#!/usr/bin/env python
"""Aggregate ProMax de novo screens across steps, profiles, and seeds."""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path

import pandas as pd

from CSDNet.exp.denovo.aggregate_sampler_profiles import METRICS, parse_csv
from CSDNet.exp.denovo.sampler_profiles import SAMPLER_PROFILES


DEFAULT_PROFILES = "promax_balanced,promax_quality,promax_diversity"


def _length_histogram_signature(diagnostics):
    histogram = diagnostics.get("sampled_length_histogram")
    if not histogram:
        return ""
    encoded = json.dumps(histogram, sort_keys=True, separators=(",", ":"))
    return hashlib.sha256(encoded.encode()).hexdigest()[:16]


def _mark_pareto_front(frame):
    result = frame.copy()
    result["pareto_nondominated"] = True
    for index, row in result.iterrows():
        dominates = (
            (result["Quality_mean"] >= row["Quality_mean"])
            & (result["Diversity_mean"] >= row["Diversity_mean"])
            & (
                (result["Quality_mean"] > row["Quality_mean"])
                | (result["Diversity_mean"] > row["Diversity_mean"])
            )
        )
        if bool(dominates.any()):
            result.at[index, "pareto_nondominated"] = False
    return result


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--input_dir", required=True)
    parser.add_argument("--steps", default="500,1000")
    parser.add_argument("--profiles", default=DEFAULT_PROFILES)
    parser.add_argument("--seeds", default="0")
    parser.add_argument("--allow_incomplete", action="store_true")
    parser.add_argument("--output", default=None)
    args = parser.parse_args()

    base = Path(args.input_dir)
    steps = [int(value) for value in parse_csv(args.steps)]
    profiles = parse_csv(args.profiles)
    seeds = [int(value) for value in parse_csv(args.seeds)]
    unknown = sorted(set(profiles) - set(SAMPLER_PROFILES))
    if unknown:
        raise SystemExit(f"Unknown profiles: {', '.join(unknown)}")

    rows = []
    missing = []
    for n_steps in steps:
        for profile in profiles:
            for seed in seeds:
                run_dir = base / f"steps{n_steps}" / profile / f"seed{seed}"
                metric_path = run_dir / "genmol_denovo_metrics.json"
                diagnostic_path = run_dir / "sampling_diagnostics.json"
                if not metric_path.exists():
                    missing.append(f"steps{n_steps}:{profile}:seed{seed}")
                    continue
                metrics = json.loads(metric_path.read_text())
                diagnostics = (
                    json.loads(diagnostic_path.read_text())
                    if diagnostic_path.exists()
                    else {}
                )
                proposals = int(
                    diagnostics.get("proposals", metrics.get("TotalGenerated", 0))
                )
                accepted = int(
                    diagnostics.get("accepted", metrics.get("TotalGenerated", 0))
                )
                row = {
                    "n_steps": n_steps,
                    "profile": profile,
                    "seed": seed,
                    "proposals": proposals,
                    "accepted": accepted,
                    "proposal_acceptance": accepted / proposals if proposals else 0.0,
                    "length_histogram_signature": _length_histogram_signature(diagnostics),
                }
                row.update({metric: metrics.get(metric) for metric in METRICS})
                rows.append(row)

    if missing and not args.allow_incomplete:
        raise SystemExit("Missing runs: " + ", ".join(missing))
    if not rows:
        raise SystemExit(f"No completed ProMax runs found in {base}")

    runs = pd.DataFrame(rows).sort_values(["n_steps", "profile", "seed"])
    signatures = runs[runs["length_histogram_signature"] != ""]
    mismatched = (
        signatures.groupby(["n_steps", "seed"])["length_histogram_signature"]
        .nunique()
    )
    mismatched = mismatched[mismatched > 1]
    if not mismatched.empty:
        raise SystemExit(
            "Profiles did not receive identical sampled length multisets for: "
            + ", ".join(f"steps={steps},seed={seed}" for steps, seed in mismatched.index)
        )
    summary = (
        runs.groupby(["n_steps", "profile"])[list(METRICS)]
        .agg(["mean", "std"])
    )
    summary.columns = [f"{metric}_{stat}" for metric, stat in summary.columns]
    summary = summary.reset_index()
    counts = (
        runs.groupby(["n_steps", "profile"])
        .size()
        .rename("n_seeds")
        .reset_index()
    )
    summary = counts.merge(summary, on=["n_steps", "profile"])
    summary["quality_diversity_product"] = (
        summary["Quality_mean"] * summary["Diversity_mean"]
    )
    summary = _mark_pareto_front(summary)
    summary = summary.sort_values(
        ["pareto_nondominated", "Quality_mean", "Diversity_mean"],
        ascending=[False, False, False],
    )

    output = Path(args.output) if args.output else base / "promax_summary.csv"
    output.parent.mkdir(parents=True, exist_ok=True)
    runs.to_csv(output.with_name(output.stem + "_runs.csv"), index=False)
    summary.to_csv(output, index=False)

    columns = [
        "n_steps",
        "profile",
        "n_seeds",
        "Validity_mean",
        "UniquenessTotal_mean",
        "Quality_mean",
        "Diversity_mean",
        "quality_diversity_product",
        "pareto_nondominated",
    ]
    print(summary[columns].to_string(index=False, float_format=lambda x: f"{x:.4f}"))
    if missing:
        print("Missing:", ", ".join(missing))
    print(f"Saved runs: {output.with_name(output.stem + '_runs.csv')}")
    print(f"Saved summary: {output}")


if __name__ == "__main__":
    main()
