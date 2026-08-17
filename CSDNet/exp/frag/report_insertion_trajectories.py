#!/usr/bin/env python
"""Summarize local elastic-insertion trajectories and within-case collapse."""

from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
import pandas as pd


def _mean_numeric(frame: pd.DataFrame, column: str) -> float:
    if column not in frame:
        return float("nan")
    values = pd.to_numeric(frame[column], errors="coerce")
    return float(values.mean()) if values.notna().any() else float("nan")


def _mean_case_stat(frame: pd.DataFrame, function) -> float:
    values = []
    for _, case in frame.groupby(["seed", "name"], dropna=False):
        value = function(case)
        if value is not None and np.isfinite(value):
            values.append(float(value))
    return float(np.mean(values)) if values else float("nan")


def _prefill_stratum(value) -> str:
    source = str(value)
    if source == "native_one_mask":
        return "native"
    if source.startswith("zinc_lower_"):
        return "zinc_lower"
    if source.startswith("zinc_middle_"):
        return "zinc_middle"
    if source.startswith("zinc_upper_"):
        return "zinc_upper"
    return "legacy"


def _load_attempts(input_dir: Path) -> pd.DataFrame:
    files = sorted(input_dir.glob("attempts_*_seed*.csv"))
    if not files:
        raise SystemExit(f"No attempt files found in {input_dir}")
    attempts = pd.concat(
        [pd.read_csv(path) for path in files],
        ignore_index=True,
        sort=False,
    )
    attempts["structural_success"] = attempts["structural_success"].map(
        lambda value: str(value).lower() in {"1", "true", "yes"}
    )
    return attempts


def _attach_sample_properties(
    attempts: pd.DataFrame,
    input_dir: Path,
) -> pd.DataFrame:
    sample_files = sorted(input_dir.glob("samples_*_seed*.csv"))
    if not sample_files:
        return attempts.assign(
            qed=np.nan,
            sa=np.nan,
            quality_pass=False,
        )
    samples = pd.concat(
        [pd.read_csv(path) for path in sample_files],
        ignore_index=True,
        sort=False,
    )
    keys = ["task", "name", "seed", "smiles"]
    required = set(keys + ["qed", "sa"])
    if not required.issubset(samples.columns):
        return attempts.assign(
            qed=np.nan,
            sa=np.nan,
            quality_pass=False,
        )
    samples = samples[keys + ["qed", "sa"]].drop_duplicates(keys)
    enriched = attempts.drop(columns=["qed", "sa"], errors="ignore").merge(
        samples,
        on=keys,
        how="left",
        validate="many_to_one",
    )
    enriched["qed"] = pd.to_numeric(enriched["qed"], errors="coerce")
    enriched["sa"] = pd.to_numeric(enriched["sa"], errors="coerce")
    enriched["quality_pass"] = (
        enriched["qed"].ge(0.6) & enriched["sa"].le(4.0)
    )
    return enriched


def summarize(input_dir: Path) -> pd.DataFrame:
    attempts = _load_attempts(input_dir)

    rows = []
    for task, frame in attempts.groupby("task", sort=True):
        successful = frame[frame["structural_success"]].copy()
        unique_per_case = _mean_case_stat(
            successful,
            lambda case: float(case["smiles"].dropna().nunique()),
        )
        largest_mode_fraction = _mean_case_stat(
            successful,
            lambda case: (
                float(case["smiles"].value_counts(normalize=True).iloc[0])
                if case["smiles"].notna().any()
                else None
            ),
        )
        learned = (
            pd.to_numeric(frame["learned_inserted_tokens"], errors="coerce")
            if "learned_inserted_tokens" in frame
            else None
        )
        rows.append(
            {
                "task": task,
                "attempts": len(frame),
                "cases": frame[["seed", "name"]].drop_duplicates().shape[0],
                "structural_success": float(frame["structural_success"].mean()),
                "mean_unique_per_case": unique_per_case,
                "mean_largest_mode_fraction": largest_mode_fraction,
                "mean_inserted_tokens": _mean_numeric(frame, "inserted_tokens"),
                "mean_initial_mask_tokens": _mean_numeric(
                    frame,
                    "initial_mask_tokens",
                ),
                "mean_prefill_prior_total": _mean_numeric(
                    frame,
                    "prefill_prior_total",
                ),
                "mean_learned_inserted_tokens": _mean_numeric(
                    frame,
                    "learned_inserted_tokens",
                ),
                "no_learned_growth_fraction": (
                    float(learned.eq(0).mean())
                    if learned is not None and learned.notna().any()
                    else float("nan")
                ),
                "mean_max_open_sites": _mean_numeric(frame, "max_open_sites"),
                "mean_open_site_rate": _mean_numeric(
                    frame,
                    "mean_open_site_rate",
                ),
                "mean_forced_final_unmasks": _mean_numeric(
                    frame,
                    "forced_final_unmasks",
                ),
                "mean_max_sequence_tokens": _mean_numeric(
                    frame,
                    "max_sequence_tokens",
                ),
            }
        )
    return pd.DataFrame(rows)


def summarize_prefill_sources(input_dir: Path) -> pd.DataFrame:
    attempts = _attach_sample_properties(_load_attempts(input_dir), input_dir)
    if "prefill_source" in attempts:
        attempts["prefill_stratum"] = attempts["prefill_source"].map(
            _prefill_stratum
        )
    else:
        attempts["prefill_stratum"] = "legacy"

    rows = []
    for (task, stratum), frame in attempts.groupby(
        ["task", "prefill_stratum"],
        sort=True,
    ):
        successful = frame[frame["structural_success"]].copy()
        unique = successful.dropna(subset=["smiles"]).drop_duplicates(
            ["seed", "name", "smiles"]
        )
        quality_count = int(unique["quality_pass"].sum())
        rows.append(
            {
                "task": task,
                "prefill_stratum": stratum,
                "attempts": len(frame),
                "structural_success": float(frame["structural_success"].mean()),
                "unique_yield": float(len(unique) / len(frame)),
                "quality_yield": float(quality_count / len(frame)),
                "quality_given_unique": (
                    float(quality_count / len(unique))
                    if len(unique)
                    else float("nan")
                ),
                "mean_qed_unique": _mean_numeric(unique, "qed"),
                "mean_sa_unique": _mean_numeric(unique, "sa"),
                "mean_unique_per_case": _mean_case_stat(
                    successful,
                    lambda case: float(case["smiles"].dropna().nunique()),
                ),
                "mean_initial_mask_tokens": _mean_numeric(
                    frame,
                    "initial_mask_tokens",
                ),
                "mean_learned_inserted_tokens": _mean_numeric(
                    frame,
                    "learned_inserted_tokens",
                ),
                "mean_max_sequence_tokens": _mean_numeric(
                    frame,
                    "max_sequence_tokens",
                ),
            }
        )
    return pd.DataFrame(rows)


def summarize_constraint_sources(input_dir: Path) -> pd.DataFrame:
    attempts = _attach_sample_properties(_load_attempts(input_dir), input_dir)
    if "gap_constraint_applied" not in attempts:
        attempts["gap_constraint_applied"] = False
    attempts["gap_constraint_applied"] = attempts[
        "gap_constraint_applied"
    ].map(lambda value: str(value).lower() in {"1", "true", "yes"})

    rows = []
    for (task, constrained), frame in attempts.groupby(
        ["task", "gap_constraint_applied"],
        sort=True,
    ):
        successful = frame[frame["structural_success"]]
        unique = successful.dropna(subset=["smiles"]).drop_duplicates(
            ["seed", "name", "smiles"]
        )
        quality_count = int(unique["quality_pass"].sum())
        rows.append(
            {
                "task": task,
                "chain_atom_constrained": bool(constrained),
                "attempts": len(frame),
                "structural_success": float(frame["structural_success"].mean()),
                "unique_yield": float(len(unique) / len(frame)),
                "quality_yield": float(quality_count / len(frame)),
                "quality_given_unique": (
                    float(quality_count / len(unique))
                    if len(unique)
                    else float("nan")
                ),
                "mean_qed_unique": _mean_numeric(unique, "qed"),
                "mean_sa_unique": _mean_numeric(unique, "sa"),
                "mean_initial_mask_tokens": _mean_numeric(
                    frame,
                    "initial_mask_tokens",
                ),
            }
        )
    return pd.DataFrame(rows)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--input_dir", required=True, type=Path)
    parser.add_argument("--output", type=Path)
    args = parser.parse_args()

    summary = summarize(args.input_dir)
    print(summary.to_string(index=False, float_format=lambda value: f"{value:.4f}"))
    output = args.output or args.input_dir / "insertion_trajectory_summary.csv"
    output.parent.mkdir(parents=True, exist_ok=True)
    summary.to_csv(output, index=False)
    print(f"Saved: {output}")
    by_prefill = summarize_prefill_sources(args.input_dir)
    print("\nBy prefill stratum:")
    print(
        by_prefill.to_string(
            index=False,
            float_format=lambda value: f"{value:.4f}",
        )
    )
    prefill_output = output.with_name(
        f"{output.stem}_by_prefill{output.suffix}"
    )
    by_prefill.to_csv(prefill_output, index=False)
    print(f"Saved: {prefill_output}")
    by_constraint = summarize_constraint_sources(args.input_dir)
    print("\nBy chain-atom constraint:")
    print(
        by_constraint.to_string(
            index=False,
            float_format=lambda value: f"{value:.4f}",
        )
    )
    constraint_output = output.with_name(
        f"{output.stem}_by_constraint{output.suffix}"
    )
    by_constraint.to_csv(constraint_output, index=False)
    print(f"Saved: {constraint_output}")


if __name__ == "__main__":
    main()
