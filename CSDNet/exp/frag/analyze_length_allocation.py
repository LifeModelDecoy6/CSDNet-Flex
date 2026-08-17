#!/usr/bin/env python
"""Diagnose length-arm yield and duplicate collapse in fragment runs."""

from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
import pandas as pd


KEYS = ["task", "name", "seed", "smiles"]
LENGTH_LABELS = ("<=31", "32-39", "40-47", "48-55", "56-63", ">=64")


def _read_many(directory: Path, prefix: str) -> pd.DataFrame:
    paths = sorted(directory.glob(f"{prefix}_*_seed*.csv"))
    if not paths:
        raise SystemExit(f"No {prefix} files found in {directory}")
    return pd.concat((pd.read_csv(path) for path in paths), ignore_index=True)


def _as_bool(series: pd.Series) -> pd.Series:
    if series.dtype == bool:
        return series
    return series.astype(str).str.lower().isin({"true", "1", "yes"})


def _summarize(frame: pd.DataFrame, group: str) -> pd.DataFrame:
    rows = []
    for value, part in frame.groupby(group, observed=True, sort=False):
        successful = part[_as_bool(part["structural_success"]) & part["smiles"].notna()]
        unique = successful.drop_duplicates(KEYS)
        attempts = len(part)
        structural = len(successful)
        unique_count = len(unique)
        quality_count = int(unique["quality_pass"].sum())
        rows.append(
            {
                group: value,
                "attempts": attempts,
                "structural_attempts": structural,
                "structural_rate": structural / attempts if attempts else 0.0,
                "unique_structures": unique_count,
                "unique_yield": unique_count / attempts if attempts else 0.0,
                "duplicate_fraction_successful": (
                    1.0 - unique_count / structural if structural else 0.0
                ),
                "unique_quality": quality_count,
                "unique_quality_yield": quality_count / attempts if attempts else 0.0,
                "quality_fraction_unique": (
                    quality_count / unique_count if unique_count else 0.0
                ),
                "mean_target_length": float(part["target_length"].mean()),
                "mean_added_tokens": float(part["added_tokens"].mean()),
            }
        )
    return pd.DataFrame(rows)


def analyze(args) -> None:
    directory = Path(args.input_dir)
    attempts = _read_many(directory, "attempts")
    samples = _read_many(directory, "samples")
    samples = samples.drop_duplicates(KEYS, keep="last")
    samples["quality_pass"] = (samples["qed"] >= args.qed_threshold) & (
        samples["sa"] <= args.sa_threshold
    )

    attempts = attempts.merge(
        samples[KEYS + ["qed", "sa", "quality_pass"]],
        on=KEYS,
        how="left",
        validate="many_to_one",
    )
    attempts["quality_pass"] = attempts["quality_pass"].eq(True)
    if "profile" not in attempts:
        attempts["profile"] = "unknown"
    attempts["target_length_bin"] = pd.cut(
        attempts["target_length"],
        bins=(-np.inf, 31, 39, 47, 55, 63, np.inf),
        labels=LENGTH_LABELS,
    )

    by_arm = _summarize(attempts, "profile")
    by_length = _summarize(attempts, "target_length_bin")
    output_prefix = Path(args.output_prefix)
    output_prefix.parent.mkdir(parents=True, exist_ok=True)
    arm_path = output_prefix.with_name(output_prefix.name + "_by_arm.csv")
    length_path = output_prefix.with_name(output_prefix.name + "_by_length.csv")
    by_arm.to_csv(arm_path, index=False)
    by_length.to_csv(length_path, index=False)

    print("Length-arm diagnostics")
    print(by_arm.to_string(index=False))
    print("=" * 96)
    print("Total target-length diagnostics")
    print(by_length.to_string(index=False))
    print(f"Saved: {arm_path}")
    print(f"Saved: {length_path}")


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--input_dir", required=True)
    parser.add_argument("--output_prefix", required=True)
    parser.add_argument("--qed_threshold", type=float, default=0.6)
    parser.add_argument("--sa_threshold", type=float, default=4.0)
    return parser.parse_args()


if __name__ == "__main__":
    analyze(parse_args())
