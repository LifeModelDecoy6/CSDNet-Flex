#!/usr/bin/env python
"""Materialize a PMO summary from canonical YAML/CSV oracle histories."""

from __future__ import annotations

import argparse
import os
from pathlib import Path

os.environ.setdefault("TF_CPP_MIN_LOG_LEVEL", "3")

import pandas as pd

from CSDNet.exp.pmo.reporting import (
    PMO_TASKS,
    load_resume_buffer,
    summarize_buffer,
)


SUMMARY_COLUMNS = [
    "mode",
    "oracle",
    "seed",
    "calls",
    "avg_top1",
    "avg_top10",
    "avg_top100",
    "auc_top1",
    "auc_top10",
    "auc_top100",
    "elapsed_sec",
    "nonzero_scores",
    "best_score",
    "unique_recorded",
]


def _existing_rows(path: Path, mode: str, seed: int):
    if not path.exists():
        return pd.DataFrame(columns=SUMMARY_COLUMNS), {}
    frame = pd.read_csv(path)
    for column in SUMMARY_COLUMNS:
        if column not in frame.columns:
            frame[column] = 0
    selected = frame[
        (frame["mode"].astype(str) == mode)
        & (pd.to_numeric(frame["seed"], errors="coerce") == seed)
    ].drop_duplicates(["oracle", "seed"], keep="last")
    elapsed = {
        str(row["oracle"]): float(row.get("elapsed_sec", 0.0))
        for _, row in selected.iterrows()
    }
    untouched = frame.drop(selected.index, errors="ignore")
    return untouched[SUMMARY_COLUMNS], elapsed


def materialize(output_dir, mode, seed, max_oracle_calls, freq_log=100):
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    summary_path = output_dir / f"summary_{mode}.csv"
    untouched, elapsed_by_oracle = _existing_rows(summary_path, mode, seed)

    rows = []
    errors = []
    for oracle in PMO_TASKS:
        try:
            buffer = load_resume_buffer(
                os.fspath(output_dir),
                mode,
                oracle,
                int(seed),
                int(max_oracle_calls),
            )
        except Exception as exc:
            errors.append(f"{oracle}: {type(exc).__name__}: {exc}")
            continue
        if not buffer:
            continue
        metrics = summarize_buffer(
            buffer,
            max_oracle_calls=int(max_oracle_calls),
            freq_log=int(freq_log),
        )
        scores = [float(value[0]) for value in buffer.values()]
        metrics.update(
            {
                "mode": mode,
                "oracle": oracle,
                "seed": int(seed),
                "elapsed_sec": elapsed_by_oracle.get(oracle, 0.0),
                "nonzero_scores": sum(score > 1e-8 for score in scores),
                "best_score": max(scores, default=0.0),
                "unique_recorded": len(buffer),
            }
        )
        rows.append(metrics)

    if not rows:
        detail = "\n".join(f"  {error}" for error in errors)
        message = f"No PMO histories found for mode={mode}, seed={seed} in {output_dir}"
        if detail:
            message += "\nInvalid histories:\n" + detail
        raise SystemExit(message)

    rebuilt = pd.DataFrame(rows)[SUMMARY_COLUMNS]
    if untouched.empty:
        combined = rebuilt.copy()
    else:
        combined = pd.concat([untouched, rebuilt], ignore_index=True)
    combined = combined.drop_duplicates(["mode", "oracle", "seed"], keep="last")
    combined = combined.sort_values(["mode", "seed", "oracle"])

    lock_path = summary_path.with_suffix(summary_path.suffix + ".lock")
    with lock_path.open("w") as lock:
        try:
            import fcntl

            fcntl.flock(lock, fcntl.LOCK_EX)
        except Exception:
            pass
        temporary = summary_path.with_suffix(summary_path.suffix + ".tmp")
        combined.to_csv(temporary, index=False)
        os.replace(temporary, summary_path)
        try:
            fcntl.flock(lock, fcntl.LOCK_UN)
        except Exception:
            pass

    if errors:
        print("Skipped invalid histories:")
        for error in errors:
            print(f"  {error}")
    print(f"Materialized {len(rebuilt)}/{len(PMO_TASKS)} task rows: {summary_path}")
    return summary_path, rebuilt


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--output_dir", required=True)
    parser.add_argument("--mode", required=True)
    parser.add_argument("--seed", type=int, required=True)
    parser.add_argument("--max_oracle_calls", type=int, default=10000)
    parser.add_argument("--freq_log", type=int, default=100)
    args = parser.parse_args()
    materialize(
        args.output_dir,
        args.mode,
        args.seed,
        args.max_oracle_calls,
        args.freq_log,
    )


if __name__ == "__main__":
    main()
