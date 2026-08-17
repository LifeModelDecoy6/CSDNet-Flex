#!/usr/bin/env python
"""Summarize multi-frontier proposal diagnostics."""

import argparse
import re
from pathlib import Path

import pandas as pd


FNAME_RE = re.compile(
    r"frontier_diagnostics_(?P<target>.+)_id(?P<idx>\d+)_thr"
    r"(?P<thr>[0-9.]+)_(?P<seed>\d+)\.csv$"
)


def load_diagnostics(input_dir):
    frames = []
    for path in sorted(Path(input_dir).glob("frontier_diagnostics_*.csv")):
        match = FNAME_RE.match(path.name)
        if match is None:
            continue
        frame = pd.read_csv(path)
        frame["target"] = match.group("target")
        frame["start_mol_idx"] = int(match.group("idx"))
        frame["sim_threshold"] = float(match.group("thr"))
        frame["seed"] = int(match.group("seed"))
        frames.append(frame)
    if not frames:
        raise SystemExit(f"No multi-frontier diagnostics found in {input_dir}")
    return pd.concat(frames, ignore_index=True)


def summarize(input_dir, operator_output, task_output):
    df = load_diagnostics(input_dir)
    for column in (
        "planned_delta",
        "operator_reward",
        "operator_batch_reward",
        "output_residual",
        "output_max_deficit",
        "output_mean_deficit",
        "qed",
        "sa",
        "similarity",
        "constraints_crossed",
        "constraints_regressed",
        "pair_frontiers_gained",
    ):
        if column not in df:
            df[column] = float("nan")
        df[column] = pd.to_numeric(df[column], errors="coerce")
    df["strict"] = df["strict"].astype(str).str.lower().isin({"true", "1", "yes"})
    df["peripheral"] = (
        df["peripheral"].astype(str).str.lower().isin({"true", "1", "yes"})
    )
    df["length_edited"] = df["planned_delta"].fillna(0).ne(0)
    df["quality_ok"] = df["qed"].ge(0.6) & df["sa"].ge(6 / 9)
    df["sim_ok"] = df["similarity"].ge(df["sim_threshold"])
    df["task"] = (
        df["target"].astype(str)
        + ":"
        + df["start_mol_idx"].astype(str)
        + ":"
        + df["sim_threshold"].map(lambda value: f"{value:g}")
    )
    df["strict_task"] = df["task"].where(df["strict"])
    df["strict_smiles"] = df["smiles"].where(df["strict"])

    batch_reward = (
        df.dropna(subset=["operator_batch_reward"])
        .groupby(["operator", "iteration"], dropna=False)["operator_batch_reward"]
        .first()
        .groupby("operator", dropna=False)
        .mean()
        .rename("mean_batch_reward")
    )

    operator = (
        df.groupby("operator", dropna=False)
        .agg(
            evaluated=("smiles", "size"),
            unique=("smiles", "nunique"),
            strict_candidates=("strict", "sum"),
            strict_unique=("strict_smiles", "nunique"),
            strict_tasks=("strict_task", "nunique"),
            mean_reward=("operator_reward", "mean"),
            mean_residual=("output_residual", "mean"),
            peripheral_fraction=("peripheral", "mean"),
            length_edit_fraction=("length_edited", "mean"),
            mean_constraints_crossed=("constraints_crossed", "mean"),
            mean_constraints_regressed=("constraints_regressed", "mean"),
            mean_pair_frontiers_gained=("pair_frontiers_gained", "mean"),
        )
        .reset_index()
    )
    operator = operator.merge(batch_reward.reset_index(), on="operator", how="left")

    task_rows = []
    for task, frame in df.groupby("task"):
        strict_ops = sorted(frame.loc[frame["strict"], "operator"].dropna().unique())
        task_rows.append(
            {
                "task": task,
                "evaluated": len(frame),
                "unique": frame["smiles"].nunique(),
                "sim_ok": int(frame["sim_ok"].sum()),
                "quality_ok": int(frame["quality_ok"].sum()),
                "sim_quality_ok": int((frame["sim_ok"] & frame["quality_ok"]).sum()),
                "strict_candidates": int(frame["strict"].sum()),
                "strict_unique": int(frame.loc[frame["strict"], "smiles"].nunique()),
                "strict_operators": ";".join(strict_ops),
                "best_residual": frame["output_residual"].min(),
                "best_max_deficit": frame["output_max_deficit"].min(),
            }
        )
    task = pd.DataFrame(task_rows).sort_values("task")

    operator_output = Path(operator_output)
    task_output = Path(task_output)
    operator_output.parent.mkdir(parents=True, exist_ok=True)
    task_output.parent.mkdir(parents=True, exist_ok=True)
    operator.to_csv(operator_output, index=False)
    task.to_csv(task_output, index=False)
    print(operator.to_string(index=False))
    print(f"Saved operator summary: {operator_output}")
    print(f"Saved task summary: {task_output}")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--input_dir", required=True)
    parser.add_argument("--operator_output", required=True)
    parser.add_argument("--task_output", required=True)
    args = parser.parse_args()
    summarize(args.input_dir, args.operator_output, args.task_output)


if __name__ == "__main__":
    main()
