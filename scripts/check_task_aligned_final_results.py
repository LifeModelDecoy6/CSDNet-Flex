#!/usr/bin/env python
"""Report every missing output cell in the final task-aligned evaluation."""

from __future__ import annotations

import argparse
from pathlib import Path

import pandas as pd

from CSDNet.exp.pmo.reporting import PMO_TASKS


FRAGMENT_TASKS = (
    "linker_design",
    "scaffold_morphing",
    "motif_extension",
    "scaffold_decoration",
    "superstructure_generation",
)
LEAD_TARGETS = ("parp1", "fa7", "5ht1b", "braf", "jak2")


def count_rows(path: Path) -> int:
    try:
        return len(pd.read_csv(path))
    except Exception:
        return 0


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--fragment_dir", required=True)
    parser.add_argument("--lead_prefix", required=True)
    parser.add_argument("--pmo_prefix", required=True)
    parser.add_argument("--pmo_mode", default="iterative_remask_v9")
    args = parser.parse_args()

    fragment_dir = Path(args.fragment_dir)
    fragment_missing = []
    for seed in (0, 1, 2):
        for task in FRAGMENT_TASKS:
            path = fragment_dir / f"metrics_{task}_seed{seed}.csv"
            rows = count_rows(path)
            if rows != 10:
                fragment_missing.append((seed, task, rows))

    lead_missing = []
    for seed in (0, 1, 2):
        directory = Path(f"{args.lead_prefix}{seed}")
        for target in LEAD_TARGETS:
            for start_idx in (0, 1, 2):
                for threshold in (0.4, 0.6):
                    path = directory / (
                        f"{target}_id{start_idx}_thr{threshold}_{seed}.csv"
                    )
                    rows = count_rows(path)
                    if rows == 0:
                        lead_missing.append(
                            (seed, target, start_idx, threshold, rows)
                        )

    pmo_missing = []
    for seed in (0, 1, 2):
        directory = Path(f"{args.pmo_prefix}{seed}")
        path = directory / f"summary_{args.pmo_mode}.csv"
        if not path.is_file():
            pmo_missing.extend((seed, task, 0) for task in PMO_TASKS)
            continue
        frame = pd.read_csv(path)
        if not {"oracle", "seed", "calls"}.issubset(frame.columns):
            pmo_missing.extend((seed, task, 0) for task in PMO_TASKS)
            continue
        frame = frame[
            pd.to_numeric(frame["seed"], errors="coerce").eq(seed)
        ].drop_duplicates("oracle", keep="last")
        frame["calls"] = pd.to_numeric(
            frame["calls"], errors="coerce"
        ).fillna(0)
        calls = {
            str(row.oracle): int(row.calls)
            for row in frame[["oracle", "calls"]].itertuples(index=False)
        }
        for task in PMO_TASKS:
            if calls.get(task, 0) < 10000:
                pmo_missing.append((seed, task, calls.get(task, 0)))

    print(f"Fragment complete: {15 - len(fragment_missing)}/15")
    for row in fragment_missing:
        print("  FRAGMENT MISSING/PARTIAL", row)
    print(f"Lead complete: {90 - len(lead_missing)}/90")
    for row in lead_missing:
        print("  LEAD MISSING", row)
    print(f"PMO complete: {69 - len(pmo_missing)}/69")
    for row in pmo_missing:
        print("  PMO MISSING/PARTIAL", row)

    if fragment_missing or lead_missing or pmo_missing:
        raise SystemExit(1)
    print("All Fragment, Lead and PMO result cells are complete.")


if __name__ == "__main__":
    main()
