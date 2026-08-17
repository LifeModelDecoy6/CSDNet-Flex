#!/usr/bin/env python
"""Compare paired universal-frontier and late bridge lead runs."""

import argparse
from pathlib import Path

import pandas as pd

from CSDNet.exp.lead.aggregate import load_start_ds, summarize_file


TASKS = (
    ("5ht1b", 0, 0.6),
    ("braf", 0, 0.6),
    ("braf", 2, 0.6),
    ("fa7", 1, 0.4),
    ("fa7", 1, 0.6),
    ("fa7", 2, 0.4),
    ("fa7", 2, 0.6),
)
WARMUP_COLUMNS = ("iteration", "smiles", "parent_smiles", "operator")


def result_name(target, start_id, threshold, seed):
    return f"{target}_id{start_id}_thr{threshold}_{seed}.csv"


def diagnostic_name(target, start_id, threshold, seed):
    return f"frontier_diagnostics_{result_name(target, start_id, threshold, seed)}"


def warmup_signature(path, last_iteration):
    if not path.exists():
        return None
    frame = pd.read_csv(path)
    missing = [column for column in WARMUP_COLUMNS if column not in frame.columns]
    if missing:
        raise RuntimeError(f"Missing diagnostic columns in {path}: {missing}")
    frame = frame[pd.to_numeric(frame["iteration"], errors="coerce") <= last_iteration]
    frame = frame.loc[:, WARMUP_COLUMNS].fillna("")
    return sorted(tuple(row) for row in frame.itertuples(index=False, name=None))


def bridge_statistics(path):
    if not path.exists():
        return 0, 0
    frame = pd.read_csv(path)
    if "operator" not in frame.columns:
        return 0, 0
    bridge = frame[frame["operator"] == "pair_bridge"]
    strict = int(bridge.get("strict", pd.Series(dtype=bool)).astype(str).str.lower().eq("true").sum())
    return len(bridge), strict


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--baseline_dir", required=True)
    parser.add_argument("--candidate_dir", required=True)
    parser.add_argument("--output", required=True)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--planned_total", type=int, default=1000)
    parser.add_argument("--warmup_iterations", type=int, default=6)
    parser.add_argument("--min_net_recoveries", type=int, default=2)
    parser.add_argument("--max_regressions", type=int, default=0)
    args = parser.parse_args()

    baseline_dir = Path(args.baseline_dir)
    candidate_dir = Path(args.candidate_dir)
    start_ds = load_start_ds(str(baseline_dir))
    rows = []
    missing = []
    for target, start_id, threshold in TASKS:
        name = result_name(target, start_id, threshold, args.seed)
        baseline_path = baseline_dir / name
        candidate_path = candidate_dir / name
        if not baseline_path.exists() or not candidate_path.exists():
            missing.append(name)
            continue
        baseline = summarize_file(str(baseline_path), args.planned_total, start_ds)
        candidate = summarize_file(str(candidate_path), args.planned_total, start_ds)
        baseline_sig = warmup_signature(
            baseline_dir / diagnostic_name(target, start_id, threshold, args.seed),
            args.warmup_iterations,
        )
        candidate_diag = candidate_dir / diagnostic_name(
            target, start_id, threshold, args.seed
        )
        candidate_sig = warmup_signature(candidate_diag, args.warmup_iterations)
        bridge_evaluated, bridge_strict = bridge_statistics(candidate_diag)
        rows.append(
            {
                "target": target,
                "start_mol_idx": start_id,
                "sim_threshold": threshold,
                "baseline_success": bool(baseline["success"]),
                "candidate_success": bool(candidate["success"]),
                "recovered": bool(candidate["success"] and not baseline["success"]),
                "regressed": bool(baseline["success"] and not candidate["success"]),
                "baseline_generated": int(baseline["generated"]),
                "candidate_generated": int(candidate["generated"]),
                "warmup_match": baseline_sig is not None and baseline_sig == candidate_sig,
                "warmup_rows": 0 if baseline_sig is None else len(baseline_sig),
                "bridge_evaluated": bridge_evaluated,
                "bridge_strict": bridge_strict,
            }
        )

    if missing:
        raise SystemExit("Missing paired lead results: " + ", ".join(missing))

    output = Path(args.output)
    output.parent.mkdir(parents=True, exist_ok=True)
    frame = pd.DataFrame(rows)
    frame.to_csv(output, index=False)
    baseline_successes = int(frame["baseline_success"].sum())
    candidate_successes = int(frame["candidate_success"].sum())
    recovered = int(frame["recovered"].sum())
    regressed = int(frame["regressed"].sum())
    warmup_ok = bool(frame["warmup_match"].all())
    promotion_passed = (
        warmup_ok
        and recovered - regressed >= args.min_net_recoveries
        and regressed <= args.max_regressions
        and int(frame["bridge_evaluated"].sum()) > 0
    )
    print(frame.to_string(index=False))
    print("\nPaired lead bridge screen")
    print(f"  baseline successes:  {baseline_successes}/{len(frame)}")
    print(f"  candidate successes: {candidate_successes}/{len(frame)}")
    print(f"  recovered/regressed: {recovered}/{regressed}")
    print(f"  bridge evaluated:    {int(frame['bridge_evaluated'].sum())}")
    print(f"  pre-bridge match:     {warmup_ok}")
    print(f"  promotion gate passed: {promotion_passed}")
    print(f"Saved: {output}")
    if not warmup_ok:
        raise SystemExit(
            "Control failure: baseline and candidate diverged before the bridge could activate."
        )


if __name__ == "__main__":
    main()
