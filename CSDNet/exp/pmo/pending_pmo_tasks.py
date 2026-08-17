#!/usr/bin/env python
"""List PMO task indices that have not reached the requested oracle budget."""

import argparse
import csv
from pathlib import Path

from CSDNet.exp.pmo.reporting import PMO_TASKS, completed


def diagnostic_calls(output_dir, mode, oracle, seed):
    path = Path(output_dir) / f"diagnostics_{mode}_{oracle}_{seed}.csv"
    if not path.is_file():
        return 0
    max_calls = 0
    try:
        with path.open(newline="") as handle:
            for row in csv.DictReader(handle):
                max_calls = max(max_calls, int(float(row.get("calls", 0))))
    except (OSError, TypeError, ValueError):
        return 0
    return max_calls


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--output_dir", required=True)
    parser.add_argument("--mode", default="iterative_remask_v8")
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--max_calls", type=int, default=10000)
    parser.add_argument("--format", choices=["table", "indices"], default="table")
    return parser.parse_args()


def main():
    args = parse_args()
    rows = []
    for index, oracle in enumerate(PMO_TASKS):
        yaml_complete = completed(
            args.output_dir,
            args.mode,
            oracle,
            args.seed,
            args.max_calls,
        )
        diag_calls = diagnostic_calls(
            args.output_dir,
            args.mode,
            oracle,
            args.seed,
        )
        is_complete = yaml_complete and diag_calls >= args.max_calls
        rows.append((index, oracle, is_complete, diag_calls))

    if args.format == "indices":
        print(" ".join(str(index) for index, _, done, _ in rows if not done))
        return

    complete_n = sum(done for _, _, done, _ in rows)
    print(f"Complete: {complete_n}/{len(rows)}")
    for index, oracle, done, diag_calls in rows:
        status = "done" if done else "pending"
        print(
            f"{index:2d}  {oracle:30s}  {status:7s}  "
            f"diagnostic_calls={diag_calls}"
        )


if __name__ == "__main__":
    main()
