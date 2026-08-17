#!/usr/bin/env python
"""Compare a targeted Lead screen against matching baseline cells."""

import argparse
from pathlib import Path

import pandas as pd

from CSDNet.exp.lead.aggregate import FNAME_RE, load_start_ds


HARD_SCREEN_CELLS = (
    ("5ht1b", 0, 0.4),
    ("5ht1b", 0, 0.6),
    ("braf", 0, 0.6),
    ("fa7", 0, 0.6),
    ("fa7", 1, 0.4),
    ("fa7", 1, 0.6),
    ("fa7", 2, 0.6),
    ("parp1", 2, 0.6),
)


def summarize_run(path, start_ds, sim_threshold):
    if path is None or not path.exists():
        return {
            "rows": 0,
            "loose": False,
            "strict": False,
            "top_ds": float("nan"),
            "best_max_deficit": float("nan"),
        }
    frame = pd.read_csv(
        path,
        names=["smiles", "DS", "QED", "SA", "SIM", "unused"],
    )
    if frame.empty:
        return {
            "rows": 0,
            "loose": False,
            "strict": False,
            "top_ds": float("nan"),
            "best_max_deficit": float("nan"),
        }
    frame = frame.drop_duplicates(subset=["smiles"]).copy()
    quality = (frame["QED"] >= 0.6) & (frame["SA"] >= 6.0 / 9.0)
    similarity = frame["SIM"] >= sim_threshold
    docking = frame["DS"] > start_ds
    loose = quality & similarity
    strict = loose & docking
    deficits = pd.concat(
        [
            (1.0 - frame["DS"] / max(start_ds, 1e-8)).clip(lower=0.0),
            (1.0 - frame["QED"] / 0.6).clip(lower=0.0),
            (1.0 - frame["SA"] / (6.0 / 9.0)).clip(lower=0.0),
            (1.0 - frame["SIM"] / max(sim_threshold, 1e-8)).clip(lower=0.0),
        ],
        axis=1,
    )
    return {
        "rows": int(len(frame)),
        "loose": bool(loose.any()),
        "strict": bool(strict.any()),
        "top_ds": float(frame.loc[strict, "DS"].max()) if strict.any() else float("nan"),
        "best_max_deficit": float(deficits.max(axis=1).min()),
    }


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--baseline_dir", required=True)
    parser.add_argument("--candidate_dir", required=True)
    parser.add_argument("--output", default=None)
    parser.add_argument(
        "--expected_screen",
        choices=["hard8"],
        default=None,
        help="Score every preregistered cell, including absent result files.",
    )
    parser.add_argument("--seed", type=int, default=None)
    args = parser.parse_args()

    baseline_dir = Path(args.baseline_dir)
    candidate_dir = Path(args.candidate_dir)
    start_ds = load_start_ds(str(candidate_dir))
    rows = []
    if args.expected_screen == "hard8":
        if args.seed is None:
            raise SystemExit("--seed is required with --expected_screen=hard8")
        candidate_paths = [
            candidate_dir / f"{target}_id{start_id}_thr{threshold}_{args.seed}.csv"
            for target, start_id, threshold in HARD_SCREEN_CELLS
        ]
    else:
        candidate_paths = sorted(candidate_dir.glob("*.csv"))
    for candidate_path in candidate_paths:
        match = FNAME_RE.match(candidate_path.name)
        if match is None:
            continue
        meta = match.groupdict()
        target = meta["target"]
        start_id = int(meta["seed_idx"])
        threshold = float(meta["thr"])
        baseline_path = baseline_dir / candidate_path.name
        baseline = summarize_run(
            baseline_path if baseline_path.exists() else None,
            start_ds[(target, start_id)],
            threshold,
        )
        candidate = summarize_run(
            candidate_path,
            start_ds[(target, start_id)],
            threshold,
        )
        rows.append(
            {
                "target": target,
                "start_id": start_id,
                "threshold": threshold,
                "baseline_present": baseline_path.exists(),
                "candidate_present": candidate_path.exists(),
                **{f"baseline_{key}": value for key, value in baseline.items()},
                **{f"candidate_{key}": value for key, value in candidate.items()},
                "strict_gain": int(candidate["strict"]) - int(baseline["strict"]),
                "residual_gain": (
                    baseline["best_max_deficit"] - candidate["best_max_deficit"]
                ),
            }
        )
    if not rows:
        raise SystemExit(f"No Lead result files found in {candidate_dir}")

    frame = pd.DataFrame(rows).sort_values(["threshold", "target", "start_id"])
    output = Path(args.output) if args.output else candidate_dir / "paired_screen.csv"
    output.parent.mkdir(parents=True, exist_ok=True)
    frame.to_csv(output, index=False)
    columns = [
        "target",
        "start_id",
        "threshold",
        "baseline_present",
        "candidate_present",
        "baseline_rows",
        "candidate_rows",
        "baseline_strict",
        "candidate_strict",
        "strict_gain",
        "baseline_best_max_deficit",
        "candidate_best_max_deficit",
        "residual_gain",
        "candidate_top_ds",
    ]
    print(frame[columns].to_string(index=False, float_format=lambda x: f"{x:.4f}"))
    print(
        "\nPaired strict: "
        f"baseline={int(frame['baseline_strict'].sum())}/{len(frame)}, "
        f"candidate={int(frame['candidate_strict'].sum())}/{len(frame)}"
    )
    print(f"Saved: {output}")


if __name__ == "__main__":
    main()
