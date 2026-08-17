#!/usr/bin/env python
"""Aggregate fragment-constrained V1 results over tasks and random seeds."""

from __future__ import annotations

import argparse
from pathlib import Path

import pandas as pd


# Kept local so aggregation does not import the model/oracle dependency stack.
CANONICAL_TASKS = (
    "linker_design",
    "scaffold_morphing",
    "motif_extension",
    "scaffold_decoration",
    "superstructure_generation",
)


METRICS = (
    "validity",
    "uniqueness",
    "quality",
    "diversity",
    "distance",
    "mean_qed",
    "mean_sa",
)


def parse_seed_list(text):
    return tuple(int(item.strip()) for item in str(text).split(",") if item.strip())


def aggregate(args):
    input_dir = Path(args.input_dir)
    paths = sorted(input_dir.glob("metrics_*_seed*.csv"))
    if not paths:
        raise SystemExit(f"No per-task metric files found in {input_dir}")
    frames = [pd.read_csv(path) for path in paths]
    merged = pd.concat(frames, ignore_index=True)
    merged = merged.drop_duplicates(["task", "name", "seed"], keep="last")

    expected_seeds = parse_seed_list(args.seeds)
    expected_tasks = tuple(CANONICAL_TASKS)
    missing = []
    for seed in expected_seeds:
        for task in expected_tasks:
            count = len(merged[(merged["seed"] == seed) & (merged["task"] == task)])
            if count != args.cases_per_task:
                missing.append(f"seed={seed} task={task}: {count}/{args.cases_per_task}")
    if missing and not args.allow_incomplete:
        raise SystemExit("Incomplete benchmark:\n  " + "\n  ".join(missing))

    by_task_seed = (
        merged.groupby(["task", "seed"], as_index=False)[list(METRICS)].mean()
    )
    mean = by_task_seed.groupby("task")[list(METRICS)].mean().add_suffix("_mean")
    std = (
        by_task_seed.groupby("task")[list(METRICS)]
        .std(ddof=1)
        .fillna(0.0)
        .add_suffix("_std")
    )
    mean_std = mean.join(std).reset_index()
    mean_std.insert(
        1,
        "n_seeds",
        mean_std["task"].map(by_task_seed.groupby("task")["seed"].nunique()),
    )

    overall = {
        "n_tasks": int(merged["task"].nunique()),
        "n_seeds": int(merged["seed"].nunique()),
        "n_cases": int(len(merged)),
    }
    for metric in METRICS:
        values = by_task_seed[metric]
        overall[f"{metric}_mean"] = float(values.mean())
        overall[f"{metric}_std"] = float(values.std(ddof=1)) if len(values) > 1 else 0.0

    output_prefix = Path(args.output_prefix)
    output_prefix.parent.mkdir(parents=True, exist_ok=True)
    merged_path = output_prefix.with_name(output_prefix.name + "_all_metrics.csv")
    task_seed_path = output_prefix.with_name(output_prefix.name + "_by_task_seed.csv")
    mean_std_path = output_prefix.with_name(output_prefix.name + "_mean_std.csv")
    overall_path = output_prefix.with_name(output_prefix.name + "_overall.csv")
    merged.to_csv(merged_path, index=False)
    by_task_seed.to_csv(task_seed_path, index=False)
    mean_std.to_csv(mean_std_path, index=False)
    pd.DataFrame([overall]).to_csv(overall_path, index=False)

    print(mean_std.to_string(index=False))
    print("=" * 88)
    print(pd.DataFrame([overall]).to_string(index=False))
    if missing:
        print("Incomplete entries:")
        for item in missing:
            print(f"  {item}")
    print(f"Saved: {merged_path}")
    print(f"Saved: {task_seed_path}")
    print(f"Saved: {mean_std_path}")
    print(f"Saved: {overall_path}")


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--input_dir", required=True)
    parser.add_argument("--seeds", default="0,1,2")
    parser.add_argument("--cases_per_task", type=int, default=10)
    parser.add_argument("--allow_incomplete", action="store_true")
    parser.add_argument(
        "--output_prefix",
        default="CSDNet/exp/frag/results/fragment_frontier_v1",
    )
    return parser.parse_args()


def main():
    aggregate(parse_args())


if __name__ == "__main__":
    main()
