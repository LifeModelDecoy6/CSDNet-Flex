#!/usr/bin/env python
import argparse
from pathlib import Path

import pandas as pd


TASKS = [
    "drd2",
    "gsk3b",
    "jnk3",
    "valsartan_smarts",
    "median2",
    "isomers_c7h8n2o2",
]
MODES = ("iterative_remask_v6", "iterative_remask_v8")
MILESTONES = (1000, 3000, 5000, 10000)


def milestone_rows(path, mode, oracle, seed):
    if not path.exists():
        return []
    df = pd.read_csv(path)
    if df.empty or "calls" not in df.columns:
        return []
    df = df.sort_values("calls").drop_duplicates("calls", keep="last")
    rows = []
    for milestone in MILESTONES:
        eligible = df[df["calls"] <= milestone]
        if eligible.empty:
            continue
        row = eligible.iloc[-1]
        rows.append(
            {
                "mode": mode,
                "oracle": oracle,
                "seed": seed,
                "milestone": milestone,
                "calls": int(row["calls"]),
                "auc_top10": float(row["auc_top10"]),
                "avg_top10": float(row["avg_top10"]),
                "best_score": float(row["best_score"]),
                "state": row.get("state", ""),
                "operator_stats": row.get("operator_stats", ""),
            }
        )
    return rows


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--base_out", required=True)
    parser.add_argument("--seed", type=int, default=0)
    args = parser.parse_args()

    base = Path(args.base_out)
    rows = []
    for mode in MODES:
        for oracle in TASKS:
            path = base / mode / f"diagnostics_{mode}_{oracle}_{args.seed}.csv"
            rows.extend(milestone_rows(path, mode, oracle, args.seed))

    if not rows:
        raise SystemExit(f"No paired diagnostic rows found under {base}")

    out = pd.DataFrame(rows)
    output_path = base / "paired_v6_v8_milestones.csv"
    out.to_csv(output_path, index=False)

    table = out.pivot_table(
        index=["oracle", "milestone"],
        columns="mode",
        values="auc_top10",
        aggfunc="last",
    )
    if set(MODES).issubset(table.columns):
        table["v8_minus_v6"] = table[MODES[1]] - table[MODES[0]]
    print(table.to_string())
    print(f"\nSaved: {output_path}")


if __name__ == "__main__":
    main()
