#!/usr/bin/env python
"""Compare two aggregated Lead Optimization runs task by task."""

import argparse
import os

import pandas as pd


KEYS = ["target", "start_mol_idx", "sim_threshold", "seed"]


def load_summary(path, label):
    frame = pd.read_csv(path)
    missing = [column for column in KEYS if column not in frame.columns]
    if missing:
        raise SystemExit(f"{label} summary is missing columns: {missing}")
    if frame.duplicated(KEYS).any():
        raise SystemExit(f"{label} summary contains duplicate task rows")
    if "strict_success" not in frame.columns:
        frame["strict_success"] = frame["success"]
    if "loose_success" not in frame.columns:
        frame["loose_success"] = frame["strict_success"]
    keep = KEYS + [
        "strict_success",
        "loose_success",
        "generated",
        "unique",
        "top_ds",
        "top_sim",
        "top_qed",
        "top_sa",
    ]
    return frame[[column for column in keep if column in frame.columns]].copy()


def format_tasks(frame):
    if frame.empty:
        return "  none"
    lines = []
    for row in frame.sort_values(KEYS).itertuples(index=False):
        lines.append(
            f"  {row.target} id{row.start_mol_idx} thr{row.sim_threshold:g} seed{row.seed}"
        )
    return "\n".join(lines)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--baseline", required=True)
    parser.add_argument("--candidate", required=True)
    parser.add_argument("--baseline_label", default="V3")
    parser.add_argument("--candidate_label", default="V4")
    parser.add_argument("--expected_tasks", type=int, default=30)
    parser.add_argument("--output")
    args = parser.parse_args()

    baseline = load_summary(args.baseline, args.baseline_label)
    candidate = load_summary(args.candidate, args.candidate_label)
    merged = baseline.merge(
        candidate,
        on=KEYS,
        how="outer",
        suffixes=("_baseline", "_candidate"),
        indicator=True,
    )

    common = merged[merged["_merge"] == "both"].copy()
    recovered = common[
        (~common["strict_success_baseline"].astype(bool))
        & common["strict_success_candidate"].astype(bool)
    ]
    regressed = common[
        common["strict_success_baseline"].astype(bool)
        & (~common["strict_success_candidate"].astype(bool))
    ]

    print(f"{args.baseline_label}: {len(baseline)}/{args.expected_tasks} tasks, "
          f"strict={int(baseline['strict_success'].sum())}/{len(baseline)}, "
          f"loose={int(baseline['loose_success'].sum())}/{len(baseline)}")
    print(f"{args.candidate_label}: {len(candidate)}/{args.expected_tasks} tasks, "
          f"strict={int(candidate['strict_success'].sum())}/{len(candidate)}, "
          f"loose={int(candidate['loose_success'].sum())}/{len(candidate)}")
    print(f"Common tasks: {len(common)}")
    print(f"Net strict change: {len(recovered) - len(regressed):+d}")
    print(f"Recovered ({len(recovered)}):\n{format_tasks(recovered)}")
    print(f"Regressed ({len(regressed)}):\n{format_tasks(regressed)}")

    baseline_only = merged[merged["_merge"] == "left_only"]
    candidate_only = merged[merged["_merge"] == "right_only"]
    if not baseline_only.empty:
        print(f"Only in {args.baseline_label}:\n{format_tasks(baseline_only)}")
    if not candidate_only.empty:
        print(f"Only in {args.candidate_label}:\n{format_tasks(candidate_only)}")

    if args.output:
        parent = os.path.dirname(os.path.abspath(args.output))
        os.makedirs(parent, exist_ok=True)
        merged.to_csv(args.output, index=False)
        print(f"Saved comparison: {args.output}")


if __name__ == "__main__":
    main()
