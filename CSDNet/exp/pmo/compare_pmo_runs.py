#!/usr/bin/env python
"""Compare two complete PMO summary files with neutral column labels."""

import argparse
import os

import pandas as pd


def load_complete(path, expected_calls):
    frame = pd.read_csv(path)
    required = {"oracle", "seed", "calls", "auc_top10"}
    missing = required - set(frame.columns)
    if missing:
        raise ValueError(f"{path} is missing columns: {sorted(missing)}")
    frame["calls"] = pd.to_numeric(frame["calls"], errors="coerce").fillna(0)
    frame["auc_top10"] = pd.to_numeric(
        frame["auc_top10"], errors="coerce"
    )
    frame = frame[frame["calls"] >= int(expected_calls)].copy()
    return frame.drop_duplicates(["oracle", "seed"], keep="last")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--baseline", required=True)
    parser.add_argument("--candidate", required=True)
    parser.add_argument("--baseline_label", default="baseline")
    parser.add_argument("--candidate_label", default="candidate")
    parser.add_argument("--expected_calls", type=int, default=10000)
    parser.add_argument("--output", default=None)
    args = parser.parse_args()

    baseline = load_complete(args.baseline, args.expected_calls)
    candidate = load_complete(args.candidate, args.expected_calls)
    baseline_auc = f"auc_top10_{args.baseline_label}"
    candidate_auc = f"auc_top10_{args.candidate_label}"
    joined = baseline[["oracle", "seed", "auc_top10"]].merge(
        candidate[["oracle", "seed", "auc_top10"]],
        on=["oracle", "seed"],
        how="outer",
        suffixes=(f"_{args.baseline_label}", f"_{args.candidate_label}"),
        indicator=True,
    )
    joined["delta"] = joined[candidate_auc] - joined[baseline_auc]
    joined = joined.sort_values("delta", ascending=False, na_position="last")

    paired = joined[joined["_merge"] == "both"].copy()
    print(
        f"Coverage: {args.baseline_label}={len(baseline)}, "
        f"{args.candidate_label}={len(candidate)}, paired={len(paired)}"
    )
    if not paired.empty:
        baseline_sum = paired[baseline_auc].sum()
        candidate_sum = paired[candidate_auc].sum()
        print(
            f"Paired sum AUC top-10: {args.baseline_label}={baseline_sum:.4f}, "
            f"{args.candidate_label}={candidate_sum:.4f}, "
            f"delta={candidate_sum - baseline_sum:+.4f}"
        )
        print(
            f"Paired mean AUC top-10: {args.baseline_label}="
            f"{paired[baseline_auc].mean():.4f}, "
            f"{args.candidate_label}={paired[candidate_auc].mean():.4f}"
        )
        improved = int((paired["delta"] > 1e-12).sum())
        regressed = int((paired["delta"] < -1e-12).sum())
        tied = len(paired) - improved - regressed
        print(
            f"Task verdict: improved={improved}, regressed={regressed}, tied={tied}"
        )

    columns = [
        "oracle",
        "seed",
        baseline_auc,
        candidate_auc,
        "delta",
        "_merge",
    ]
    print("\nPer-task comparison:")
    print(
        joined[columns].to_string(
            index=False, float_format=lambda value: f"{value:.4f}"
        )
    )

    if args.output:
        os.makedirs(os.path.dirname(args.output) or ".", exist_ok=True)
        joined.to_csv(args.output, index=False)
        print(f"\nSaved: {args.output}")


if __name__ == "__main__":
    main()
