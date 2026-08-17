#!/usr/bin/env python
"""Summarize accepted fixed-length and learned-insertion proposals."""

from __future__ import annotations

import argparse
from pathlib import Path

import pandas as pd


SOURCES = (
    (
        "lead",
        "frontier_diagnostics_*.csv",
        "actual_delta",
    ),
    (
        "pmo",
        "transitions_*.csv",
        "actual_length_delta",
    ),
    (
        "fragment",
        "attempts_*_seed*.csv",
        "actual_delta",
    ),
)


def load_rows(category, directory, pattern, delta_column):
    if directory is None:
        return pd.DataFrame()
    directories = directory if isinstance(directory, list) else [directory]
    paths = sorted(
        path
        for item in directories
        for path in Path(item).glob(pattern)
    )
    frames = []
    for path in paths:
        frame = pd.read_csv(path)
        if "length_mode" not in frame:
            continue
        if delta_column not in frame:
            frame[delta_column] = pd.NA
        frame = frame[["length_mode", delta_column]].copy()
        frame.columns = ["length_mode", "length_delta"]
        frame["category"] = category
        frame["source"] = str(path)
        frames.append(frame)
    return pd.concat(frames, ignore_index=True) if frames else pd.DataFrame()


def summarize(rows):
    if rows.empty:
        return pd.DataFrame()
    rows = rows.copy()
    rows["length_mode"] = rows["length_mode"].fillna("fixed").astype(str)
    rows["length_delta"] = pd.to_numeric(
        rows["length_delta"],
        errors="coerce",
    )
    rows["shrunk"] = rows["length_delta"] < 0
    rows["same_length"] = rows["length_delta"] == 0
    rows["grown"] = rows["length_delta"] > 0
    totals = rows.groupby("category").size().rename("category_total")
    summary = (
        rows.groupby(["category", "length_mode"], as_index=False)
        .agg(
            accepted=("length_mode", "size"),
            measured_deltas=("length_delta", "count"),
            mean_delta=("length_delta", "mean"),
            shrink_fraction=("shrunk", "mean"),
            same_length_fraction=("same_length", "mean"),
            growth_fraction=("grown", "mean"),
        )
    )
    summary["accepted_fraction"] = summary.apply(
        lambda row: row["accepted"] / max(1, totals[row["category"]]),
        axis=1,
    )
    return summary


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--lead_dir", action="append")
    parser.add_argument("--pmo_dir", action="append")
    parser.add_argument("--fragment_dir", action="append")
    parser.add_argument("--output", required=True)
    args = parser.parse_args()

    directories = {
        "lead": args.lead_dir,
        "pmo": args.pmo_dir,
        "fragment": args.fragment_dir,
    }
    frames = [
        load_rows(category, directories[category], pattern, delta_column)
        for category, pattern, delta_column in SOURCES
    ]
    rows = pd.concat(
        [frame for frame in frames if not frame.empty],
        ignore_index=True,
    ) if any(not frame.empty for frame in frames) else pd.DataFrame()
    report = summarize(rows)
    if report.empty:
        raise SystemExit("No insertion-aware diagnostics were found.")
    output = Path(args.output)
    output.parent.mkdir(parents=True, exist_ok=True)
    report.to_csv(output, index=False)
    print(report.to_string(index=False))
    print(f"Saved: {output}")


if __name__ == "__main__":
    main()
