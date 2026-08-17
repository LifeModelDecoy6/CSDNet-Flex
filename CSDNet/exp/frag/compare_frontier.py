#!/usr/bin/env python
"""Paired comparison of two fragment-constrained benchmark runs."""

from __future__ import annotations

import argparse
from pathlib import Path

import pandas as pd


KEYS = ("task", "name", "seed")
METRICS = ("validity", "uniqueness", "quality", "diversity", "distance")


def load_metrics(directory: str) -> pd.DataFrame:
    paths = sorted(Path(directory).glob("metrics_*_seed*.csv"))
    if not paths:
        raise SystemExit(f"No metric files found in {directory}")
    frame = pd.concat((pd.read_csv(path) for path in paths), ignore_index=True)
    return frame.drop_duplicates(list(KEYS), keep="last")


def compare(args):
    baseline = load_metrics(args.baseline_dir)
    candidate = load_metrics(args.candidate_dir)
    paired = baseline.merge(
        candidate,
        on=list(KEYS),
        how="inner",
        suffixes=("_baseline", "_candidate"),
        validate="one_to_one",
    )
    if paired.empty:
        raise SystemExit("The two runs have no paired task/name/seed rows")

    for metric in METRICS:
        paired[f"{metric}_delta"] = (
            paired[f"{metric}_candidate"] - paired[f"{metric}_baseline"]
        )

    columns = []
    for metric in METRICS:
        columns.extend(
            (
                f"{metric}_baseline",
                f"{metric}_candidate",
                f"{metric}_delta",
            )
        )
    by_task = paired.groupby("task", as_index=False)[columns].mean()

    output_prefix = Path(args.output_prefix)
    output_prefix.parent.mkdir(parents=True, exist_ok=True)
    paired_path = output_prefix.with_name(output_prefix.name + "_paired.csv")
    task_path = output_prefix.with_name(output_prefix.name + "_by_task.csv")
    paired.to_csv(paired_path, index=False)
    by_task.to_csv(task_path, index=False)

    display = ["task"]
    for metric in METRICS[:4]:
        display.extend((f"{metric}_candidate", f"{metric}_delta"))
    print(f"Paired cases: {len(paired)}")
    print(by_task[display].to_string(index=False))
    print(f"Saved: {paired_path}")
    print(f"Saved: {task_path}")


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--baseline_dir", required=True)
    parser.add_argument("--candidate_dir", required=True)
    parser.add_argument("--output_prefix", required=True)
    return parser.parse_args()


def main():
    compare(parse_args())


if __name__ == "__main__":
    main()
