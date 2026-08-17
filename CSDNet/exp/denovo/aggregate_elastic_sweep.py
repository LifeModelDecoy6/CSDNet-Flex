#!/usr/bin/env python
import argparse
import json
from pathlib import Path

import pandas as pd

from CSDNet.exp.denovo.aggregate_elastic_runs import METRIC_KEYS


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--base_dir", required=True)
    parser.add_argument("--steps", default="500,800,1000")
    parser.add_argument("--seeds", default="0,1,2")
    parser.add_argument("--output_prefix", default="elastic_denovo_step_sweep")
    args = parser.parse_args()

    base_dir = Path(args.base_dir)
    steps = [int(value) for value in args.steps.split(",") if value]
    seeds = [int(value) for value in args.seeds.split(",") if value]
    rows = []
    for n_steps in steps:
        for seed in seeds:
            path = (
                base_dir
                / f"steps{n_steps}"
                / f"seed{seed}"
                / "genmol_denovo_metrics.json"
            )
            if not path.is_file():
                raise SystemExit(
                    f"Missing metrics for steps={n_steps}, seed={seed}: {path}"
                )
            with path.open() as handle:
                metrics = json.load(handle)
            row = {"n_steps": n_steps, "seed": seed}
            for output_name, source_name in METRIC_KEYS.items():
                row[output_name] = float(metrics[source_name])
            rows.append(row)

    runs = pd.DataFrame(rows).sort_values(["n_steps", "seed"])
    summaries = []
    for n_steps, group in runs.groupby("n_steps", sort=True):
        summary = {"n_steps": int(n_steps), "n": len(group)}
        for metric in METRIC_KEYS:
            summary[f"{metric}_mean"] = float(group[metric].mean())
            summary[f"{metric}_std"] = float(group[metric].std(ddof=1))
        summaries.append(summary)
    summary_frame = pd.DataFrame(summaries)

    runs_path = base_dir / f"{args.output_prefix}_runs.csv"
    summary_path = base_dir / f"{args.output_prefix}_mean_std.csv"
    runs.to_csv(runs_path, index=False)
    summary_frame.to_csv(summary_path, index=False)
    print(runs.to_string(index=False))
    print("=" * 88)
    print(summary_frame.to_string(index=False))
    print(f"Saved runs: {runs_path}")
    print(f"Saved mean/std: {summary_path}")


if __name__ == "__main__":
    main()
