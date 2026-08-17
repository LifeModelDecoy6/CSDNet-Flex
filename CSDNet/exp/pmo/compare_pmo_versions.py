#!/usr/bin/env python
import argparse
import os

import pandas as pd


def load_complete(path, expected_calls):
    frame = pd.read_csv(path)
    required = {"oracle", "seed", "calls", "auc_top10"}
    missing = required - set(frame.columns)
    if missing:
        raise ValueError(f"{path} is missing columns: {sorted(missing)}")
    frame = frame[frame["calls"] >= expected_calls].copy()
    return frame.drop_duplicates(["oracle", "seed"], keep="last")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--baseline", required=True)
    parser.add_argument("--candidate", required=True)
    parser.add_argument("--expected_calls", type=int, default=10000)
    parser.add_argument("--failure_threshold", type=float, default=0.85)
    parser.add_argument("--control_tolerance", type=float, default=0.02)
    parser.add_argument("--output", default=None)
    args = parser.parse_args()

    baseline = load_complete(args.baseline, args.expected_calls)
    candidate = load_complete(args.candidate, args.expected_calls)
    joined = baseline.merge(
        candidate,
        on=["oracle", "seed"],
        suffixes=("_v8", "_v9"),
        how="outer",
        indicator=True,
    )
    joined["group"] = joined["auc_top10_v8"].map(
        lambda value: "failed" if pd.notna(value) and value < args.failure_threshold else "control"
    )
    joined["delta"] = joined["auc_top10_v9"] - joined["auc_top10_v8"]
    joined = joined.sort_values(["group", "delta", "oracle"], na_position="last")

    display = joined[
        ["oracle", "seed", "group", "auc_top10_v8", "auc_top10_v9", "delta", "_merge"]
    ]
    print(display.to_string(index=False, float_format=lambda value: f"{value:.4f}"))

    paired = joined[joined["_merge"] == "both"]
    print("\nPaired summary:")
    for group, rows in paired.groupby("group", sort=False):
        print(
            f"  {group}: n={len(rows)}, "
            f"V8={rows['auc_top10_v8'].sum():.4f}, "
            f"V9={rows['auc_top10_v9'].sum():.4f}, "
            f"delta={rows['delta'].sum():+.4f}"
        )

    controls = paired[paired["group"] == "control"]
    regressions = controls[controls["delta"] < -args.control_tolerance]
    if controls.empty:
        print("\nControl verdict: not tested yet.")
    elif regressions.empty:
        print(
            f"\nControl verdict: PASS; no control regressed by more than "
            f"{args.control_tolerance:.3f}."
        )
    else:
        names = ", ".join(regressions["oracle"])
        print(
            f"\nControl verdict: FAIL; regression beyond tolerance on: {names}"
        )

    if args.output:
        os.makedirs(os.path.dirname(args.output) or ".", exist_ok=True)
        joined.to_csv(args.output, index=False)
        print(f"Saved: {args.output}")


if __name__ == "__main__":
    main()
