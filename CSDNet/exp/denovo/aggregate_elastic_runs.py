#!/usr/bin/env python
import argparse
import json
from pathlib import Path

import pandas as pd


METRIC_KEYS = {
    "validity": "Validity",
    "uniqueness_valid": "Uniqueness",
    "uniqueness_total": "UniquenessTotal",
    "quality": "Quality",
    "diversity": "Diversity",
    "mean_qed": "MeanQEDUniqueValid",
    "mean_sa": "MeanSAUniqueValid",
}


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--base_dir", required=True)
    parser.add_argument("--seeds", default="0,1,2")
    parser.add_argument("--output_prefix", default="elastic_denovo")
    args = parser.parse_args()

    base_dir = Path(args.base_dir)
    seeds = [int(value) for value in args.seeds.split(",") if value]
    rows = []
    for seed in seeds:
        path = base_dir / f"seed{seed}" / "genmol_denovo_metrics.json"
        if not path.is_file():
            raise SystemExit(f"Missing de novo metrics for seed {seed}: {path}")
        with path.open() as handle:
            metrics = json.load(handle)
        row = {"seed": seed}
        for output_name, source_name in METRIC_KEYS.items():
            row[output_name] = float(metrics[source_name])
        rows.append(row)

    runs = pd.DataFrame(rows)
    summary = {"n": len(runs)}
    for metric in METRIC_KEYS:
        summary[f"{metric}_mean"] = float(runs[metric].mean())
        summary[f"{metric}_std"] = float(runs[metric].std(ddof=1))
    summary_frame = pd.DataFrame([summary])

    runs_path = base_dir / f"{args.output_prefix}_runs.csv"
    summary_path = base_dir / f"{args.output_prefix}_mean_std.csv"
    runs.to_csv(runs_path, index=False)
    summary_frame.to_csv(summary_path, index=False)
    print(runs.to_string(index=False))
    print("=" * 72)
    print(summary_frame.to_string(index=False))
    print(f"Saved runs: {runs_path}")
    print(f"Saved mean/std: {summary_path}")


if __name__ == "__main__":
    main()
