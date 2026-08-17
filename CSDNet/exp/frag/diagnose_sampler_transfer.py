#!/usr/bin/env python
"""Diagnose whether a de novo sampler improvement transfers to fragment infill."""

from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
import pandas as pd


CASE_KEYS = ["task", "name", "seed"]
METRICS = ("validity", "uniqueness", "quality", "diversity", "distance")


def load_family(input_dir: Path, prefix: str) -> pd.DataFrame:
    paths = sorted(input_dir.glob(f"{prefix}_*_seed*.csv"))
    if not paths:
        raise SystemExit(f"No {prefix} files found in {input_dir}")
    return pd.concat([pd.read_csv(path) for path in paths], ignore_index=True)


def as_bool(series: pd.Series) -> pd.Series:
    if pd.api.types.is_bool_dtype(series):
        return series.fillna(False)
    return series.astype(str).str.lower().isin({"true", "1", "yes"})


def compare_metrics(baseline: pd.DataFrame, candidate: pd.DataFrame):
    baseline = baseline.drop_duplicates(CASE_KEYS, keep="last")
    candidate = candidate.drop_duplicates(CASE_KEYS, keep="last")
    merged = baseline.merge(
        candidate,
        on=CASE_KEYS,
        suffixes=("_baseline", "_candidate"),
        validate="one_to_one",
    )
    if len(merged) != len(baseline) or len(merged) != len(candidate):
        raise SystemExit(
            "Baseline and candidate do not contain the same case/seed cells: "
            f"baseline={len(baseline)}, candidate={len(candidate)}, paired={len(merged)}"
        )

    overall = []
    by_task = []
    for metric in METRICS:
        old = pd.to_numeric(merged[f"{metric}_baseline"], errors="coerce")
        new = pd.to_numeric(merged[f"{metric}_candidate"], errors="coerce")
        overall.append(
            {
                "metric": metric,
                "baseline": float(old.mean()),
                "candidate": float(new.mean()),
                "delta": float((new - old).mean()),
                "improved_cases": int((new > old).sum()),
                "regressed_cases": int((new < old).sum()),
                "tied_cases": int((new == old).sum()),
            }
        )
        for task, group in merged.groupby("task", sort=True):
            task_old = pd.to_numeric(
                group[f"{metric}_baseline"], errors="coerce"
            )
            task_new = pd.to_numeric(
                group[f"{metric}_candidate"], errors="coerce"
            )
            by_task.append(
                {
                    "task": task,
                    "metric": metric,
                    "baseline": float(task_old.mean()),
                    "candidate": float(task_new.mean()),
                    "delta": float((task_new - task_old).mean()),
                }
            )
    return merged, pd.DataFrame(overall), pd.DataFrame(by_task)


def attempt_summary(attempts: pd.DataFrame) -> pd.DataFrame:
    attempts = attempts.copy()
    for column in (
        "initial_model_output",
        "initial_structural_success",
        "recovered_structural",
        "model_output",
        "structural_success",
    ):
        if column in attempts:
            attempts[column] = as_bool(attempts[column])
    numeric = (
        "draw_count",
        "target_length",
        "added_tokens",
        "attachment_count",
        "mean_log_prob",
        "refinement_edits",
    )
    for column in numeric:
        if column in attempts:
            attempts[column] = pd.to_numeric(attempts[column], errors="coerce")

    rows = []
    for task, group in attempts.groupby("task", sort=True):
        row = {"task": task, "attempts": len(group)}
        for column in (
            "initial_model_output",
            "initial_structural_success",
            "recovered_structural",
            "model_output",
            "structural_success",
        ):
            if column in group:
                row[f"{column}_rate"] = float(group[column].mean())
        for column in numeric:
            if column in group:
                row[f"mean_{column}"] = float(group[column].mean())
        if "refinement_edits" in group:
            row["refined_attempt_rate"] = float(
                group["refinement_edits"].gt(0).mean()
            )
        rows.append(row)
    return pd.DataFrame(rows)


def quality_bottlenecks(samples: pd.DataFrame) -> pd.DataFrame:
    samples = samples.copy()
    samples["qed"] = pd.to_numeric(samples["qed"], errors="coerce")
    samples["sa"] = pd.to_numeric(samples["sa"], errors="coerce")
    samples["qed_pass"] = samples["qed"] >= 0.6
    samples["sa_pass"] = samples["sa"] <= 4.0
    samples["quality_pass"] = samples["qed_pass"] & samples["sa_pass"]
    rows = []
    for task, group in samples.groupby("task", sort=True):
        rows.append(
            {
                "task": task,
                "unique_valid": len(group),
                "quality_unique": int(group["quality_pass"].sum()),
                "quality_rate_among_unique": float(group["quality_pass"].mean()),
                "qed_fail_rate": float((~group["qed_pass"]).mean()),
                "sa_fail_rate": float((~group["sa_pass"]).mean()),
                "both_fail_rate": float(
                    ((~group["qed_pass"]) & (~group["sa_pass"])).mean()
                ),
                "mean_qed": float(group["qed"].mean()),
                "mean_sa": float(group["sa"].mean()),
            }
        )
    return pd.DataFrame(rows)


def quality_by_editable_size(
    samples: pd.DataFrame,
    attempts: pd.DataFrame,
) -> pd.DataFrame:
    metadata_columns = CASE_KEYS + ["smiles", "profile", "added_tokens", "target_length"]
    available = [column for column in metadata_columns if column in attempts]
    metadata = attempts[available].dropna(subset=["smiles"]).drop_duplicates(
        CASE_KEYS + ["smiles"], keep="last"
    )
    merged = samples.merge(
        metadata,
        on=CASE_KEYS + ["smiles"],
        how="left",
        validate="one_to_one",
    )
    merged["qed"] = pd.to_numeric(merged["qed"], errors="coerce")
    merged["sa"] = pd.to_numeric(merged["sa"], errors="coerce")
    merged["quality_pass"] = (merged["qed"] >= 0.6) & (merged["sa"] <= 4.0)
    merged["added_tokens"] = pd.to_numeric(
        merged.get("added_tokens"), errors="coerce"
    )
    merged["editable_bin"] = pd.cut(
        merged["added_tokens"],
        bins=[-np.inf, 8, 16, 24, 32, np.inf],
        labels=["<=8", "9-16", "17-24", "25-32", ">32"],
    )
    rows = []
    group_columns = ["task", "profile", "editable_bin"]
    for keys, group in merged.dropna(subset=["editable_bin"]).groupby(
        group_columns,
        observed=True,
        sort=True,
    ):
        rows.append(
            {
                **dict(zip(group_columns, keys)),
                "unique_valid": len(group),
                "quality_unique": int(group["quality_pass"].sum()),
                "quality_rate_among_unique": float(group["quality_pass"].mean()),
                "mean_qed": float(group["qed"].mean()),
                "mean_sa": float(group["sa"].mean()),
            }
        )
    return pd.DataFrame(rows)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--baseline_dir", required=True)
    parser.add_argument("--candidate_dir", required=True)
    parser.add_argument("--output_prefix")
    args = parser.parse_args()

    baseline_dir = Path(args.baseline_dir)
    candidate_dir = Path(args.candidate_dir)
    baseline_metrics = load_family(baseline_dir, "metrics")
    candidate_metrics = load_family(candidate_dir, "metrics")
    candidate_attempts = load_family(candidate_dir, "attempts")
    candidate_samples = load_family(candidate_dir, "samples")

    _, overall, by_task = compare_metrics(baseline_metrics, candidate_metrics)
    attempts = attempt_summary(candidate_attempts)
    bottlenecks = quality_bottlenecks(candidate_samples)
    by_editable = quality_by_editable_size(candidate_samples, candidate_attempts)

    print("=" * 88)
    print("PAIRED OVERALL DELTAS")
    print(overall.to_string(index=False, float_format=lambda value: f"{value:.4f}"))
    print("\nQUALITY DELTA BY TASK")
    print(
        by_task[by_task["metric"] == "quality"].to_string(
            index=False,
            float_format=lambda value: f"{value:.4f}",
        )
    )
    print("\nCANDIDATE GENERATION BOTTLENECKS")
    print(attempts.to_string(index=False, float_format=lambda value: f"{value:.4f}"))
    print("\nCANDIDATE QED/SA BOTTLENECKS AMONG UNIQUE VALID OUTPUTS")
    print(bottlenecks.to_string(index=False, float_format=lambda value: f"{value:.4f}"))
    print("\nQUALITY BY EDITABLE TOKEN COUNT")
    print(by_editable.to_string(index=False, float_format=lambda value: f"{value:.4f}"))

    if args.output_prefix:
        prefix = Path(args.output_prefix)
        prefix.parent.mkdir(parents=True, exist_ok=True)
        outputs = {
            "overall": overall,
            "by_task": by_task,
            "attempts": attempts,
            "bottlenecks": bottlenecks,
            "by_editable": by_editable,
        }
        for suffix, frame in outputs.items():
            path = prefix.with_name(f"{prefix.name}_{suffix}.csv")
            frame.to_csv(path, index=False)
            print(f"Saved: {path}")


if __name__ == "__main__":
    main()
