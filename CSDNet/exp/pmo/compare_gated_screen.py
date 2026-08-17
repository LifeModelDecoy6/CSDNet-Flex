#!/usr/bin/env python
"""Compare paired V9 and evidence-gated V9 PMO trajectories."""

import argparse
from pathlib import Path

import pandas as pd

from CSDNet.exp.pmo.reporting import summarize_buffer


TASKS = (
    "mestranol_similarity",
    "perindopril_mpo",
    "sitagliptin_mpo",
    "troglitazone_rediscovery",
)


def load_scores(path, expected_calls):
    if not path.exists():
        return None
    frame = pd.read_csv(path, names=["smiles", "score"])
    if len(frame) < expected_calls:
        return None
    frame = frame.iloc[:expected_calls].copy()
    frame["score"] = pd.to_numeric(frame["score"], errors="raise")
    return frame


def metrics(frame, expected_calls):
    buffer = {
        row.smiles: [float(row.score), index]
        for index, row in enumerate(frame.itertuples(index=False), start=1)
    }
    return summarize_buffer(buffer, max_oracle_calls=expected_calls, freq_log=100)


def warmup_matches(baseline, candidate, warmup_calls):
    baseline = baseline.iloc[:warmup_calls].reset_index(drop=True)
    candidate = candidate.iloc[:warmup_calls].reset_index(drop=True)
    if len(baseline) != warmup_calls or len(candidate) != warmup_calls:
        return False
    same_smiles = baseline["smiles"].equals(candidate["smiles"])
    max_score_error = (baseline["score"] - candidate["score"]).abs().max()
    return bool(same_smiles and max_score_error <= 1e-10)


def final_gate_state(candidate_dir, oracle, seed):
    path = candidate_dir / (
        f"diagnostics_iterative_remask_v9_gated_{oracle}_{seed}.csv"
    )
    if not path.exists():
        return "missing", float("nan"), float("nan")
    frame = pd.read_csv(path)
    if frame.empty:
        return "empty", float("nan"), float("nan")
    row = frame.iloc[-1]
    return (
        row.get("protected_policy_phase", "unknown"),
        float(row.get("protected_policy_reserve", float("nan"))),
        float(row.get("protected_policy_advantage", float("nan"))),
    )


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--baseline_dir", required=True)
    parser.add_argument("--candidate_dir", required=True)
    parser.add_argument("--output", required=True)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--expected_calls", type=int, default=3000)
    parser.add_argument("--warmup_calls", type=int, default=1000)
    parser.add_argument("--min_sum_gain", type=float, default=0.04)
    parser.add_argument("--max_task_drop", type=float, default=0.02)
    args = parser.parse_args()

    baseline_dir = Path(args.baseline_dir)
    candidate_dir = Path(args.candidate_dir)
    rows = []
    missing = []
    for oracle in TASKS:
        baseline = load_scores(
            baseline_dir / f"{oracle}_{args.seed}.csv", args.expected_calls
        )
        candidate = load_scores(
            candidate_dir / f"{oracle}_{args.seed}.csv", args.expected_calls
        )
        if baseline is None or candidate is None:
            missing.append(oracle)
            continue
        baseline_metrics = metrics(baseline, args.expected_calls)
        candidate_metrics = metrics(candidate, args.expected_calls)
        phase, reserve, advantage = final_gate_state(
            candidate_dir, oracle, args.seed
        )
        rows.append(
            {
                "oracle": oracle,
                "baseline_auc_top10": baseline_metrics["auc_top10"],
                "candidate_auc_top10": candidate_metrics["auc_top10"],
                "delta_auc_top10": (
                    candidate_metrics["auc_top10"] - baseline_metrics["auc_top10"]
                ),
                "baseline_avg_top10": baseline_metrics["avg_top10"],
                "candidate_avg_top10": candidate_metrics["avg_top10"],
                "warmup_match": warmup_matches(
                    baseline, candidate, args.warmup_calls
                ),
                "final_gate_phase": phase,
                "final_gate_reserve": reserve,
                "final_gate_advantage": advantage,
            }
        )

    if missing:
        raise SystemExit("Missing or incomplete paired PMO results: " + ", ".join(missing))

    output = Path(args.output)
    output.parent.mkdir(parents=True, exist_ok=True)
    frame = pd.DataFrame(rows)
    frame.to_csv(output, index=False)
    baseline_sum = float(frame["baseline_auc_top10"].sum())
    candidate_sum = float(frame["candidate_auc_top10"].sum())
    warmup_ok = bool(frame["warmup_match"].all())
    pass_screen = (
        warmup_ok
        and candidate_sum - baseline_sum >= args.min_sum_gain
        and float(frame["delta_auc_top10"].min()) >= -args.max_task_drop
    )
    print(frame.to_string(index=False))
    print("\nPaired PMO evidence-gate screen")
    print(f"  baseline sum:  {baseline_sum:.4f}")
    print(f"  candidate sum: {candidate_sum:.4f}")
    print(f"  delta:         {candidate_sum - baseline_sum:+.4f}")
    print(f"  warmup match:  {warmup_ok}")
    print(f"  promotion gate passed: {pass_screen}")
    print(f"Saved: {output}")
    if not warmup_ok:
        raise SystemExit(
            "Control failure: V9 and gated V9 diverged during the protected warmup."
        )


if __name__ == "__main__":
    main()
