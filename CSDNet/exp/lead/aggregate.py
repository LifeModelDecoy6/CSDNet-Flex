#!/usr/bin/env python
import argparse
import os
import re

import pandas as pd


FNAME_RE = re.compile(
    r"^(?P<target>parp1|fa7|5ht1b|braf|jak2)_id(?P<seed_idx>[0-2])_thr"
    r"(?P<thr>[0-9.]+)_(?P<seed>[0-9]+)\.csv$"
)


def load_start_ds(input_dir):
    active_path = os.path.join(os.path.dirname(os.path.abspath(input_dir)), "docking", "actives.csv")
    if not os.path.exists(active_path):
        active_path = os.path.join("CSDNet", "exp", "lead", "docking", "actives.csv")
    df = pd.read_csv(active_path)
    out = {}
    for target, group in df.groupby("target"):
        group = group.reset_index(drop=True)
        for idx, row in group.iterrows():
            out[(target, idx)] = float(row["DS"])
    return out


def summarize_file(path, planned_total, start_ds_map):
    name = os.path.basename(path)
    match = FNAME_RE.match(name)
    if not match:
        return None
    meta = match.groupdict()
    sim_thr = float(meta["thr"])
    target = meta["target"]
    seed_idx = int(meta["seed_idx"])
    start_ds = start_ds_map.get((target, seed_idx))

    df = pd.read_csv(path, names=["smiles", "DS", "QED", "SA", "SIM", ""])
    raw_n = len(df)
    unique_n = int(df["smiles"].nunique()) if raw_n else 0

    loose = df.drop_duplicates(subset=["smiles"])
    loose = loose[loose["SIM"] >= sim_thr]
    loose = loose[loose["QED"] >= 0.6]
    loose = loose[loose["SA"] >= 6 / 9]
    strict = loose
    if start_ds is not None:
        strict = strict[strict["DS"] > start_ds]
    loose_success = len(loose) > 0
    strict_success = len(strict) > 0
    success = strict_success

    row = {
        "target": target,
        "start_mol_idx": seed_idx,
        "sim_threshold": sim_thr,
        "seed": int(meta["seed"]),
        "generated": raw_n,
        "unique": unique_n,
        "uniqueness_actual": unique_n / raw_n if raw_n else 0.0,
        "unique_over_planned": unique_n / planned_total if planned_total else 0.0,
        "start_ds": start_ds,
        "loose_success": loose_success,
        "strict_success": strict_success,
        "success": success,
        "top_ds": None,
        "top_smiles": None,
        "top_qed": None,
        "top_sa": None,
        "top_sim": None,
        "file": path,
    }
    if success:
        best = strict.sort_values("DS", ascending=False).iloc[0]
        row.update(
            {
                "top_ds": float(best["DS"]),
                "top_smiles": best["smiles"],
                "top_qed": float(best["QED"]),
                "top_sa": float(best["SA"]),
                "top_sim": float(best["SIM"]),
            }
        )
    return row


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--input_dir", default=os.path.join("CSDNet", "exp", "lead", "results"))
    parser.add_argument("--output", default=os.path.join("CSDNet", "exp", "lead", "results", "lead_summary.csv"))
    parser.add_argument("--planned_total", type=int, default=1000)
    args = parser.parse_args()

    rows = []
    start_ds_map = load_start_ds(args.input_dir)
    for name in sorted(os.listdir(args.input_dir)):
        if not name.endswith(".csv") or name == os.path.basename(args.output):
            continue
        row = summarize_file(os.path.join(args.input_dir, name), args.planned_total, start_ds_map)
        if row is not None:
            rows.append(row)

    if not rows:
        raise SystemExit(f"No lead result CSV files found in {args.input_dir}")

    df = pd.DataFrame(rows)
    df = df.sort_values(["target", "start_mol_idx", "sim_threshold", "seed"])
    os.makedirs(os.path.dirname(args.output), exist_ok=True)
    df.to_csv(args.output, index=False)

    success_rate = 100.0 * df["success"].mean()
    loose_success_rate = 100.0 * df["loose_success"].mean()
    print(f"Tasks: {len(df)}")
    print(f"Strict success rate: {success_rate:.2f}%")
    print(f"Loose success rate: {loose_success_rate:.2f}%")
    print(f"Mean generated: {df['generated'].mean():.1f}/{args.planned_total}")
    print(f"Mean uniqueness(actual): {100.0 * df['uniqueness_actual'].mean():.2f}%")
    print(f"Saved: {args.output}")

    by_thr = df.groupby("sim_threshold")["success"].mean().mul(100.0)
    print("\nSuccess by similarity threshold:")
    for thr, val in by_thr.items():
        print(f"  {thr}: {val:.2f}%")

    by_target = df.groupby("target")["success"].mean().mul(100.0)
    print("\nSuccess by target:")
    for target, val in by_target.items():
        print(f"  {target}: {val:.2f}%")


if __name__ == "__main__":
    main()
