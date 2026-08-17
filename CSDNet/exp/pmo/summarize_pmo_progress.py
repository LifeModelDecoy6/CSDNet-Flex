#!/usr/bin/env python
import argparse
import os

import pandas as pd

from CSDNet.exp.pmo.reporting import PMO_TASKS


def main():
    parser = argparse.ArgumentParser(description="Summarize CSDNet PMO progress from summary CSV.")
    parser.add_argument("--output_dir", required=True)
    parser.add_argument("--mode", default="iterative_remask_v3")
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument(
        "--expected_calls",
        type=int,
        default=None,
        help="Only count rows that reached at least this many oracle calls.",
    )
    args = parser.parse_args()

    path = os.path.join(args.output_dir, f"summary_{args.mode}.csv")
    if not os.path.exists(path):
        raise SystemExit(f"Summary file not found: {path}")

    df = pd.read_csv(path)
    if "seed" in df.columns:
        df = df[df["seed"].astype(int) == args.seed]
    if "mode" in df.columns:
        df = df[df["mode"] == args.mode]

    # If a task was accidentally rerun, keep the most recent row.
    df = df.drop_duplicates(subset=["oracle", "seed"], keep="last")
    incomplete = pd.DataFrame(columns=df.columns)
    if args.expected_calls is not None:
        if "calls" not in df.columns:
            raise SystemExit(f"Summary file has no calls column: {path}")
        calls = pd.to_numeric(df["calls"], errors="coerce").fillna(0).astype(int)
        incomplete = df[calls < args.expected_calls].copy()
        df = df[calls >= args.expected_calls].copy()

    done = set(df["oracle"].tolist())
    missing = [task for task in PMO_TASKS if task not in done]

    print(f"Summary: {path}")
    print(f"Mode: {args.mode}, seed: {args.seed}")
    print(f"Done: {len(done)}/{len(PMO_TASKS)}")
    if args.expected_calls is not None:
        print(f"Required oracle calls per task: {args.expected_calls}")
    if "auc_top10" in df.columns:
        print(f"Current sum AUC top-10: {df['auc_top10'].astype(float).sum():.4f}")
        print(f"Current mean AUC top-10: {df['auc_top10'].astype(float).mean():.4f}")
    if "elapsed_sec" in df.columns:
        print(f"Finished elapsed hours: {df['elapsed_sec'].astype(float).sum() / 3600.0:.2f}")

    print("\nCompleted:")
    for _, row in df.sort_values("oracle").iterrows():
        elapsed = float(row.get("elapsed_sec", 0.0)) / 3600.0
        auc = float(row.get("auc_top10", 0.0))
        print(f"  {row['oracle']}: auc_top10={auc:.4f}, elapsed={elapsed:.2f}h")

    print("\nMissing:")
    if missing:
        for task in missing:
            print(f"  {task}")
    else:
        print("  none")

    if args.expected_calls is not None:
        print("\nIncomplete rows:")
        if incomplete.empty:
            print("  none")
        else:
            for _, row in incomplete.sort_values("oracle").iterrows():
                print(f"  {row['oracle']}: calls={int(row.get('calls', 0))}")


if __name__ == "__main__":
    main()
