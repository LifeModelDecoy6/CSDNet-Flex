#!/usr/bin/env python
"""Aggregate complete 10k-call PMO summaries across random seeds."""

import argparse
import os
from pathlib import Path

import pandas as pd

from CSDNet.exp.pmo.reporting import PMO_TASKS


def parse_seed_dir(value):
    try:
        seed_text, path = value.split("=", 1)
        seed = int(seed_text)
    except (TypeError, ValueError) as exc:
        raise argparse.ArgumentTypeError(
            f"Expected SEED=PATH, got {value!r}"
        ) from exc
    if not path:
        raise argparse.ArgumentTypeError(f"Missing path in {value!r}")
    return seed, os.path.abspath(os.path.expanduser(path))


def load_seed(seed, input_dir, mode, expected_calls):
    path = os.path.join(input_dir, f"summary_{mode}.csv")
    if not os.path.isfile(path):
        raise SystemExit(f"Seed {seed} PMO summary not found: {path}")
    frame = pd.read_csv(path)
    required = {"mode", "oracle", "seed", "calls", "auc_top10", "avg_top10"}
    missing_columns = sorted(required - set(frame.columns))
    if missing_columns:
        raise SystemExit(f"Missing columns in {path}: {missing_columns}")
    frame = frame[
        (frame["mode"].astype(str) == mode)
        & (pd.to_numeric(frame["seed"], errors="coerce") == seed)
    ].copy()
    frame = frame.drop_duplicates(["oracle", "seed"], keep="last")
    frame["calls"] = pd.to_numeric(frame["calls"], errors="coerce").fillna(0)
    incomplete = frame[frame["calls"] < expected_calls]
    if not incomplete.empty:
        details = incomplete[["oracle", "calls"]].to_string(index=False)
        raise SystemExit(f"Incomplete PMO rows for seed {seed}:\n{details}")
    frame = frame[frame["calls"] >= expected_calls].copy()

    observed = set(frame["oracle"])
    expected = set(PMO_TASKS)
    missing = sorted(expected - observed)
    extra = sorted(observed - expected)
    if missing or extra or len(frame) != len(PMO_TASKS):
        raise SystemExit(
            f"Incomplete PMO seed {seed}: observed={len(frame)}/23, "
            f"missing={missing}, extra={extra}"
        )
    for column in (
        "auc_top1",
        "auc_top10",
        "auc_top100",
        "avg_top1",
        "avg_top10",
        "avg_top100",
        "elapsed_sec",
        "best_score",
    ):
        if column in frame.columns:
            frame[column] = pd.to_numeric(frame[column], errors="coerce")
    frame["source_summary"] = path
    return frame


def sample_std(series):
    return float(series.std(ddof=1)) if len(series) > 1 else 0.0


def main():
    parser = argparse.ArgumentParser(
        description="Aggregate complete PMO runs across random seeds."
    )
    parser.add_argument(
        "--seed_dir",
        action="append",
        type=parse_seed_dir,
        required=True,
        metavar="SEED=PATH",
        help="Repeat once per random seed.",
    )
    parser.add_argument("--output_dir", required=True)
    parser.add_argument("--mode", default="iterative_remask_v9")
    parser.add_argument("--expected_calls", type=int, default=10000)
    parser.add_argument("--expected_seeds", default="0,1,2")
    args = parser.parse_args()

    seed_dirs = dict(args.seed_dir)
    if len(seed_dirs) != len(args.seed_dir):
        raise SystemExit("Each --seed_dir seed label must be unique.")
    expected_seeds = [int(item) for item in args.expected_seeds.split(",") if item]
    if sorted(seed_dirs) != sorted(expected_seeds):
        raise SystemExit(
            f"Expected seed directories {expected_seeds}, got {sorted(seed_dirs)}"
        )

    runs = pd.concat(
        [
            load_seed(
                seed,
                seed_dirs[seed],
                args.mode,
                args.expected_calls,
            )
            for seed in expected_seeds
        ],
        ignore_index=True,
    )
    runs = runs.sort_values(["oracle", "seed"]).reset_index(drop=True)

    seed_rows = []
    for seed, group in runs.groupby("seed", sort=True):
        seed_rows.append(
            {
                "seed": int(seed),
                "n_tasks": len(group),
                "sum_auc_top10": float(group["auc_top10"].sum()),
                "mean_auc_top10": float(group["auc_top10"].mean()),
                "mean_avg_top10": float(group["avg_top10"].mean()),
                "elapsed_hours": (
                    float(group["elapsed_sec"].sum() / 3600.0)
                    if "elapsed_sec" in group
                    else None
                ),
            }
        )
    by_seed = pd.DataFrame(seed_rows)

    task_rows = []
    metric_columns = (
        "auc_top1",
        "auc_top10",
        "auc_top100",
        "avg_top1",
        "avg_top10",
        "avg_top100",
        "best_score",
    )
    for oracle, group in runs.groupby("oracle", sort=True):
        row = {"oracle": oracle, "n_seeds": len(group)}
        for column in metric_columns:
            if column not in group:
                continue
            values = group[column].dropna().astype(float)
            row[f"{column}_mean"] = (
                float(values.mean()) if len(values) else None
            )
            row[f"{column}_std"] = sample_std(values)
        task_rows.append(row)
    by_task = pd.DataFrame(task_rows)

    seed_sums = by_seed["sum_auc_top10"]
    overall = pd.DataFrame(
        [
            {
                "mode": args.mode,
                "oracle_calls_per_task": args.expected_calls,
                "n_seeds": len(expected_seeds),
                "n_tasks": len(PMO_TASKS),
                "sum_auc_top10_3seed_mean": float(seed_sums.mean()),
                "sum_auc_top10_3seed_std": sample_std(seed_sums),
                "mean_auc_top10_over_tasks_and_seeds": float(
                    runs["auc_top10"].mean()
                ),
                "sum_of_per_task_auc_top10_means": float(
                    by_task["auc_top10_mean"].sum()
                ),
            }
        ]
    )

    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    paths = {
        "runs": output_dir / "pmo_all_seed_runs.csv",
        "seed": output_dir / "pmo_by_seed.csv",
        "task": output_dir / "pmo_by_task_3seed.csv",
        "overall": output_dir / "pmo_overall_3seed.csv",
    }
    runs.to_csv(paths["runs"], index=False)
    by_seed.to_csv(paths["seed"], index=False)
    by_task.to_csv(paths["task"], index=False)
    overall.to_csv(paths["overall"], index=False)

    print("PMO three-seed summary")
    print(
        f"Mode={args.mode}, calls={args.expected_calls}/task, "
        f"tasks={len(PMO_TASKS)}"
    )
    for row in by_seed.itertuples(index=False):
        print(
            f"Seed {row.seed}: Sum AUC top-10={row.sum_auc_top10:.4f}, "
            f"mean={row.mean_auc_top10:.4f}"
        )
    record = overall.iloc[0]
    print(
        "\nThree-seed Sum AUC top-10: "
        f"{record.sum_auc_top10_3seed_mean:.4f} +/- "
        f"{record.sum_auc_top10_3seed_std:.4f}"
    )
    for label, path in paths.items():
        print(f"Saved {label}: {path}")


if __name__ == "__main__":
    main()
