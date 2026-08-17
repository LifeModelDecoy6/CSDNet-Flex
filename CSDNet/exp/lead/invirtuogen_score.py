#!/usr/bin/env python
"""Score CSDNet Lead Optimization outputs using InVirtuoGen Table 4."""

import argparse
import os

import numpy as np
import pandas as pd

from CSDNet.exp.lead.aggregate import FNAME_RE, load_start_ds


TABLE4_REFERENCES = {
    "GenMol": {0.4: -148.7, 0.6: -117.7},
    "RetMol": {0.4: -88.5, 0.6: -25.7},
    "GraphGA": {0.4: -96.3, 0.6: -74.8},
    "InVirtuoGen": {0.4: -152.4, 0.6: -145.7},
}
TABLE4_INCLUSIVE = {
    "InVirtuoGen": {0.4: -160.4, 0.6: -153.1},
}
RESULT_COLUMNS = ["smiles", "DS", "QED", "SA", "SIM", "unused"]
CELL_KEYS = ["target", "start_mol_idx", "sim_threshold"]


def load_task_run(path, planned_total, start_ds_map):
    match = FNAME_RE.match(os.path.basename(path))
    if match is None:
        return None
    meta = match.groupdict()
    target = meta["target"]
    start_idx = int(meta["seed_idx"])
    threshold = float(meta["thr"])
    random_seed = int(meta["seed"])
    start_ds = float(start_ds_map[(target, start_idx)])

    frame = pd.read_csv(path, names=RESULT_COLUMNS)
    frame = frame.dropna(subset=["smiles", "DS", "QED", "SA", "SIM"])
    frame = frame.drop_duplicates(subset=["smiles"])
    constrained = frame[
        (frame["QED"] >= 0.6)
        & (frame["SA"] >= 6.0 / 9.0)
        & (frame["SIM"] >= threshold)
    ].copy()
    constrained = constrained.sort_values("DS", ascending=False)

    top_constrained = constrained.iloc[0] if not constrained.empty else None
    constrained_ds = float(top_constrained["DS"]) if top_constrained is not None else 0.0
    strict_success = bool(constrained_ds > start_ds)
    strict_ds = constrained_ds if strict_success else 0.0

    return {
        "target": target,
        "start_mol_idx": start_idx,
        "sim_threshold": threshold,
        "seed": random_seed,
        "generated": len(frame),
        "planned_total": int(planned_total),
        "start_ds_magnitude": start_ds,
        "strict_success": strict_success,
        "constrained_candidate": top_constrained is not None,
        "strict_ds_magnitude": strict_ds,
        "constraint_only_ds_magnitude": constrained_ds,
        "strict_ds_signed": -strict_ds,
        "constraint_only_ds_signed": -constrained_ds,
        "dock_shortfall": max(0.0, start_ds - constrained_ds),
        "top_constrained_smiles": (
            top_constrained["smiles"] if top_constrained is not None else None
        ),
        "top_constrained_qed": (
            float(top_constrained["QED"]) if top_constrained is not None else np.nan
        ),
        "top_constrained_sa_raw": (
            10.0 - 9.0 * float(top_constrained["SA"])
            if top_constrained is not None
            else np.nan
        ),
        "top_constrained_similarity": (
            float(top_constrained["SIM"]) if top_constrained is not None else np.nan
        ),
        "file": path,
    }


def aggregate_cells(runs, expected_random_seeds=None):
    if expected_random_seeds is None:
        if runs.empty:
            expected_random_seeds = 1
        else:
            expected_random_seeds = int(
                runs.groupby(CELL_KEYS, sort=False).size().max()
            )
    rows = []
    for keys, group in runs.groupby(CELL_KEYS, sort=True):
        strict_values = group["strict_ds_magnitude"].astype(float)
        inclusive_values = group["constraint_only_ds_magnitude"].astype(float)
        n_random_seeds = int(len(group))
        rows.append(
            {
                **dict(zip(CELL_KEYS, keys)),
                "n_random_seeds": n_random_seeds,
                "expected_random_seeds": int(expected_random_seeds),
                "complete_random_seeds": n_random_seeds
                == int(expected_random_seeds),
                "strict_success_runs": int(group["strict_success"].sum()),
                "constraint_candidate_runs": int(
                    group["constrained_candidate"].sum()
                ),
                "strict_ds_signed_mean": -float(strict_values.mean()),
                "strict_ds_signed_zero_filled_mean": -float(
                    strict_values.sum() / expected_random_seeds
                ),
                "strict_ds_signed_std": float(strict_values.std(ddof=1))
                if len(group) > 1
                else 0.0,
                "constraint_only_ds_signed_mean": -float(inclusive_values.mean()),
                "constraint_only_ds_signed_zero_filled_mean": -float(
                    inclusive_values.sum() / expected_random_seeds
                ),
                "constraint_only_ds_signed_std": float(
                    inclusive_values.std(ddof=1)
                )
                if len(group) > 1
                else 0.0,
                "mean_dock_shortfall": float(group["dock_shortfall"].mean()),
            }
        )
    return pd.DataFrame(rows).sort_values(CELL_KEYS).reset_index(drop=True)


def print_reference_gap(threshold, observed_signed):
    observed_magnitude = abs(float(observed_signed))
    for method, values in TABLE4_REFERENCES.items():
        reference = float(values[threshold])
        magnitude_gap = observed_magnitude - abs(reference)
        relation = "ahead" if magnitude_gap > 0 else "behind"
        print(
            f"    vs {method:12s}: {reference:7.1f}, "
            f"{relation} by {abs(magnitude_gap):.3f} DS-sum"
        )


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--input_dir", nargs="+", required=True)
    parser.add_argument("--planned_total", type=int, default=1000)
    parser.add_argument("--expected_cells_per_threshold", type=int, default=15)
    parser.add_argument("--expected_random_seeds", type=int, default=1)
    parser.add_argument("--run_output")
    parser.add_argument("--cell_output")
    args = parser.parse_args()

    start_ds_map = load_start_ds(args.input_dir[0])
    rows = []
    for input_dir in args.input_dir:
        for name in sorted(os.listdir(input_dir)):
            path = os.path.join(input_dir, name)
            if not os.path.isfile(path):
                continue
            row = load_task_run(path, args.planned_total, start_ds_map)
            if row is not None:
                rows.append(row)
    if not rows:
        raise SystemExit(f"No Lead task result files found in {args.input_dir}")

    runs = pd.DataFrame(rows).sort_values(CELL_KEYS + ["seed"]).reset_index(drop=True)
    duplicate_keys = CELL_KEYS + ["seed"]
    if runs.duplicated(duplicate_keys).any():
        duplicated = runs[runs.duplicated(duplicate_keys, keep=False)]
        raise SystemExit(
            "Duplicate task/seed results across input directories:\n"
            + duplicated[duplicate_keys + ["file"]].to_string(index=False)
        )
    cells = aggregate_cells(runs, args.expected_random_seeds)

    run_output = args.run_output or os.path.join(
        args.input_dir[0],
        "invirtuogen_score_runs.csv",
    )
    cell_output = args.cell_output or os.path.join(
        args.input_dir[0],
        "invirtuogen_score_cells.csv",
    )
    runs.to_csv(run_output, index=False)
    cells.to_csv(cell_output, index=False)

    print("InVirtuoGen Table 4 scoring")
    print("DS convention: CSDNet positive magnitude converted to negative docking score")
    print("Official sum: constrained and docking-improved; failed run contributes 0")
    print("Parenthetical sum: best constrained candidate even without docking improvement")
    print()

    total_strict = 0.0
    total_inclusive = 0.0
    all_thresholds_complete = True
    for threshold in sorted(cells["sim_threshold"].unique()):
        subset = cells[cells["sim_threshold"] == threshold]
        strict_sum = float(subset["strict_ds_signed_mean"].sum())
        inclusive_sum = float(subset["constraint_only_ds_signed_mean"].sum())
        total_strict += strict_sum
        total_inclusive += inclusive_sum
        complete_cells = len(subset) == args.expected_cells_per_threshold
        complete_runs = bool(
            (subset["n_random_seeds"] == args.expected_random_seeds).all()
        )
        threshold_complete = complete_cells and complete_runs
        all_thresholds_complete = all_thresholds_complete and threshold_complete
        successes = int(
            runs[runs["sim_threshold"] == threshold]["strict_success"].sum()
        )
        observed_runs = int((runs["sim_threshold"] == threshold).sum())
        print(
            f"delta={threshold:g}: cells={len(subset)}/"
            f"{args.expected_cells_per_threshold}, runs={observed_runs}, "
            f"strict_success={successes}/{observed_runs}"
        )
        if threshold_complete:
            print(f"  official strict sum:      {strict_sum:.3f}")
            print(f"  parenthetical total sum: {inclusive_sum:.3f}")
        else:
            zero_filled_strict = float(
                subset["strict_ds_signed_zero_filled_mean"].sum()
            )
            zero_filled_inclusive = float(
                subset["constraint_only_ds_signed_zero_filled_mean"].sum()
            )
            print(f"  observed-only sum (NOT official): {strict_sum:.3f}")
            print(
                "  missing-as-zero conservative estimate (NOT official): "
                f"{zero_filled_strict:.3f}"
            )
            print(
                "  parenthetical missing-as-zero estimate: "
                f"{zero_filled_inclusive:.3f}"
            )
            print("  WARNING: incomplete cells or random-seed coverage")
        if threshold_complete:
            print_reference_gap(threshold, strict_sum)

    print()
    if all_thresholds_complete:
        print(
            "Combined 30-cell strict sum (not a Table 4 headline): "
            f"{total_strict:.3f}"
        )
        print(
            "Combined parenthetical sum:                       "
            f"{total_inclusive:.3f}"
        )
    else:
        print("No official combined sum: random-seed matrix is incomplete.")

    near = runs[
        (~runs["strict_success"])
        & runs["constrained_candidate"]
        & (runs["dock_shortfall"] <= 0.5)
    ].sort_values("dock_shortfall")
    print("\nConstraint-feasible near misses within 0.5 docking units:")
    if near.empty:
        print("  none")
    else:
        for row in near.itertuples(index=False):
            print(
                f"  {row.target} id{row.start_mol_idx} delta={row.sim_threshold:g} "
                f"seed={row.seed}: best={row.constraint_only_ds_magnitude:.3f}, "
                f"start={row.start_ds_magnitude:.3f}, shortfall={row.dock_shortfall:.3f}"
            )

    print(f"\nSaved run scores:  {run_output}")
    print(f"Saved cell scores: {cell_output}")


if __name__ == "__main__":
    main()
