#!/usr/bin/env python
"""Strict, definition-explicit aggregation for three-seed Lead Optimization."""

import argparse
import os
from pathlib import Path

import pandas as pd

from CSDNet.exp.lead.aggregate import FNAME_RE, load_start_ds, summarize_file


TARGETS = ("parp1", "fa7", "5ht1b", "braf", "jak2")
START_MOL_INDICES = (0, 1, 2)
SIM_THRESHOLDS = (0.4, 0.6)
TASK_KEYS = ["target", "start_mol_idx", "sim_threshold"]


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


def expected_task_keys():
    return {
        (target, start_idx, threshold)
        for target in TARGETS
        for start_idx in START_MOL_INDICES
        for threshold in SIM_THRESHOLDS
    }


def load_seed(seed, input_dir, planned_total, start_ds_map):
    if not os.path.isdir(input_dir):
        raise SystemExit(f"Seed {seed} directory not found: {input_dir}")
    rows = []
    for name in sorted(os.listdir(input_dir)):
        match = FNAME_RE.match(name)
        if match is None:
            continue
        row = summarize_file(
            os.path.join(input_dir, name),
            planned_total,
            start_ds_map,
        )
        if row is None:
            continue
        if int(row["seed"]) != seed:
            raise SystemExit(
                f"Seed label mismatch in {name}: directory says {seed}, "
                f"filename says {row['seed']}"
            )
        rows.append(row)
    if not rows:
        raise SystemExit(f"No Lead task CSV files found for seed {seed}: {input_dir}")

    frame = pd.DataFrame(rows)
    duplicates = frame.duplicated(TASK_KEYS, keep=False)
    if duplicates.any():
        raise SystemExit(
            f"Duplicate Lead tasks for seed {seed}:\n"
            + frame.loc[duplicates, TASK_KEYS + ["file"]].to_string(index=False)
        )
    observed = set(frame[TASK_KEYS].itertuples(index=False, name=None))
    expected = expected_task_keys()
    missing = sorted(expected - observed)
    extra = sorted(observed - expected)
    if missing or extra:
        raise SystemExit(
            f"Incomplete Lead seed {seed}: observed={len(observed)}/30, "
            f"missing={missing}, extra={extra}"
        )
    return frame


def sample_std(series):
    return float(series.std(ddof=1)) if len(series) > 1 else 0.0


def main():
    parser = argparse.ArgumentParser(
        description="Aggregate complete Lead Optimization runs across random seeds."
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
    parser.add_argument("--planned_total", type=int, default=1000)
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

    first_dir = seed_dirs[expected_seeds[0]]
    start_ds_map = load_start_ds(first_dir)
    runs = pd.concat(
        [
            load_seed(
                seed,
                seed_dirs[seed],
                args.planned_total,
                start_ds_map,
            )
            for seed in expected_seeds
        ],
        ignore_index=True,
    )
    runs = runs.sort_values(TASK_KEYS + ["seed"]).reset_index(drop=True)

    seed_rows = []
    for seed, group in runs.groupby("seed", sort=True):
        seed_rows.append(
            {
                "seed": int(seed),
                "n_tasks": len(group),
                "strict_successes": int(group["strict_success"].sum()),
                "strict_success_rate": float(group["strict_success"].mean()),
                "loose_successes": int(group["loose_success"].sum()),
                "loose_success_rate": float(group["loose_success"].mean()),
                "mean_generated": float(group["generated"].mean()),
                "mean_uniqueness_actual": float(
                    group["uniqueness_actual"].mean()
                ),
            }
        )
    by_seed = pd.DataFrame(seed_rows)

    task_rows = []
    for keys, group in runs.groupby(TASK_KEYS, sort=True):
        strict_count = int(group["strict_success"].sum())
        loose_count = int(group["loose_success"].sum())
        successful_ds = pd.to_numeric(
            group.loc[group["strict_success"], "top_ds"],
            errors="coerce",
        ).dropna()
        task_rows.append(
            {
                **dict(zip(TASK_KEYS, keys)),
                "n_seeds": len(group),
                "strict_successes": strict_count,
                "strict_success_rate": strict_count / len(group),
                "loose_successes": loose_count,
                "loose_success_rate": loose_count / len(group),
                "strict_any_seed": strict_count >= 1,
                "strict_majority": strict_count > len(group) / 2,
                "strict_all_seeds": strict_count == len(group),
                "top_ds_mean_success_only": (
                    float(successful_ds.mean()) if len(successful_ds) else None
                ),
                "top_ds_std_success_only": sample_std(successful_ds),
                "mean_generated": float(group["generated"].mean()),
                "mean_uniqueness_actual": float(
                    group["uniqueness_actual"].mean()
                ),
            }
        )
    by_task = pd.DataFrame(task_rows)

    strict_seed_rates = by_seed["strict_success_rate"]
    loose_seed_rates = by_seed["loose_success_rate"]
    overall = pd.DataFrame(
        [
            {
                "n_seeds": len(expected_seeds),
                "n_benchmark_tasks": len(by_task),
                "n_task_runs": len(runs),
                "strict_successes_over_runs": int(
                    runs["strict_success"].sum()
                ),
                "strict_success_rate_over_runs": float(
                    runs["strict_success"].mean()
                ),
                "strict_rate_seed_mean": float(strict_seed_rates.mean()),
                "strict_rate_seed_std": sample_std(strict_seed_rates),
                "loose_successes_over_runs": int(runs["loose_success"].sum()),
                "loose_success_rate_over_runs": float(
                    runs["loose_success"].mean()
                ),
                "loose_rate_seed_mean": float(loose_seed_rates.mean()),
                "loose_rate_seed_std": sample_std(loose_seed_rates),
                "tasks_strict_any_seed": int(by_task["strict_any_seed"].sum()),
                "tasks_strict_majority": int(by_task["strict_majority"].sum()),
                "tasks_strict_all_seeds": int(
                    by_task["strict_all_seeds"].sum()
                ),
            }
        ]
    )

    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    paths = {
        "runs": output_dir / "lead_all_seed_runs.csv",
        "seed": output_dir / "lead_by_seed.csv",
        "task": output_dir / "lead_by_task_3seed.csv",
        "overall": output_dir / "lead_overall_3seed.csv",
    }
    runs.to_csv(paths["runs"], index=False)
    by_seed.to_csv(paths["seed"], index=False)
    by_task.to_csv(paths["task"], index=False)
    overall.to_csv(paths["overall"], index=False)

    print("Lead Optimization three-seed summary")
    print("Strict = similarity + QED + SA + docking improvement over the seed.")
    print("Loose  = similarity + QED + SA, without docking improvement.")
    print()
    for row in by_seed.itertuples(index=False):
        print(
            f"Seed {row.seed}: strict={row.strict_successes}/30 "
            f"({100 * row.strict_success_rate:.2f}%), "
            f"loose={row.loose_successes}/30 "
            f"({100 * row.loose_success_rate:.2f}%)"
        )
    record = overall.iloc[0]
    print(
        f"\nRun-level strict: {int(record.strict_successes_over_runs)}/"
        f"{int(record.n_task_runs)} "
        f"({100 * record.strict_success_rate_over_runs:.2f}%)"
    )
    print(
        "Per-seed strict rate: "
        f"{100 * record.strict_rate_seed_mean:.2f} +/- "
        f"{100 * record.strict_rate_seed_std:.2f}%"
    )
    print(
        "Task robustness (any / majority / all 3): "
        f"{int(record.tasks_strict_any_seed)} / "
        f"{int(record.tasks_strict_majority)} / "
        f"{int(record.tasks_strict_all_seeds)}"
    )
    for label, path in paths.items():
        print(f"Saved {label}: {path}")


if __name__ == "__main__":
    main()
