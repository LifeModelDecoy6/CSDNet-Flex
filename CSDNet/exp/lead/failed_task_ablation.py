#!/usr/bin/env python
"""Select failed lead tasks and report recovery-only ablations."""

import argparse
from pathlib import Path

import pandas as pd

from CSDNet.exp.lead.aggregate import FNAME_RE, load_start_ds, summarize_file


TARGET_ORDER = {
    "parp1": 0,
    "fa7": 1,
    "5ht1b": 2,
    "braf": 3,
    "jak2": 4,
}
TASK_COLUMNS = ["target", "start_mol_idx", "sim_threshold"]


def _bool_series(series):
    if pd.api.types.is_bool_dtype(series):
        return series.fillna(False)
    return (
        series.fillna(False)
        .astype(str)
        .str.strip()
        .str.lower()
        .isin({"1", "true", "yes", "y"})
    )


def _sort_tasks(df):
    if df.empty:
        return df.copy()
    out = df.copy()
    out["_target_order"] = out["target"].map(TARGET_ORDER).fillna(999)
    out = out.sort_values(
        ["_target_order", "start_mol_idx", "sim_threshold", "seed"]
    )
    return out.drop(columns="_target_order")


def _task_key(row):
    return (
        str(row["target"]),
        int(row["start_mol_idx"]),
        float(row["sim_threshold"]),
    )


def _task_label(key):
    target, start_idx, threshold = key
    return f"{target}:{start_idx}:{threshold:g}"


def load_result_dir(input_dir, planned_total=1000):
    """Load raw task CSVs, falling back to an existing aggregate summary."""
    input_dir = Path(input_dir)
    if not input_dir.is_dir():
        raise SystemExit(f"Lead result directory not found: {input_dir}")

    start_ds_map = load_start_ds(str(input_dir))
    rows = []
    for path in sorted(input_dir.glob("*.csv")):
        if FNAME_RE.match(path.name) is None:
            continue
        row = summarize_file(str(path), planned_total, start_ds_map)
        if row is not None:
            rows.append(row)

    if rows:
        df = pd.DataFrame(rows)
    else:
        summaries = [
            input_dir / "lead_summary.csv",
            *sorted(input_dir.glob("lead_summary_*.csv")),
        ]
        summary_path = next((path for path in summaries if path.is_file()), None)
        if summary_path is None:
            raise SystemExit(f"No lead task CSVs or summary found in {input_dir}")
        df = pd.read_csv(summary_path)

    missing = [column for column in TASK_COLUMNS + ["seed"] if column not in df]
    if missing:
        raise SystemExit(
            f"Lead results in {input_dir} are missing columns: {', '.join(missing)}"
        )

    strict_column = "strict_success" if "strict_success" in df else "success"
    if strict_column not in df:
        raise SystemExit(
            f"Lead results in {input_dir} contain neither strict_success nor success"
        )
    df = df.copy()
    df["strict_success"] = _bool_series(df[strict_column])
    if "loose_success" in df:
        df["loose_success"] = _bool_series(df["loose_success"])
    return _sort_tasks(df)


def select_seed(
    df,
    seed,
    expected_tasks=None,
    expected_successes=None,
    label="results",
):
    selected = df[df["seed"].astype(int) == int(seed)].copy()
    if selected.empty:
        available = sorted(df["seed"].astype(int).unique().tolist())
        raise SystemExit(f"No seed={seed} rows in {label}; available seeds: {available}")

    duplicate_mask = selected.duplicated(TASK_COLUMNS, keep=False)
    if duplicate_mask.any():
        duplicates = selected.loc[duplicate_mask, TASK_COLUMNS].drop_duplicates()
        labels = [_task_label(_task_key(row)) for _, row in duplicates.iterrows()]
        raise SystemExit(
            f"Duplicate seed={seed} tasks in {label}: {', '.join(labels)}"
        )

    if expected_tasks is not None and len(selected) != expected_tasks:
        raise SystemExit(
            f"Expected {expected_tasks} seed={seed} tasks in {label}, found "
            f"{len(selected)}. Refusing to select from an incomplete or mixed baseline."
        )
    successes = int(selected["strict_success"].sum())
    if expected_successes is not None and successes != expected_successes:
        raise SystemExit(
            f"Expected {expected_successes} strict successes for seed={seed} in "
            f"{label}, found {successes}. Refusing to use the wrong baseline."
        )
    return _sort_tasks(selected)


def command_select(args):
    baseline = load_result_dir(args.baseline_dir, args.planned_total)
    baseline = select_seed(
        baseline,
        args.seed,
        expected_tasks=args.expected_tasks,
        expected_successes=args.expected_successes,
        label=str(args.baseline_dir),
    )
    failures = baseline[~baseline["strict_success"]].copy()

    output = Path(args.output)
    output.parent.mkdir(parents=True, exist_ok=True)
    labels = [_task_label(_task_key(row)) for _, row in failures.iterrows()]
    output.write_text("".join(f"{label}\n" for label in labels), encoding="ascii")

    successes = int(baseline["strict_success"].sum())
    print(
        f"Baseline seed={args.seed}: strict={successes}/{len(baseline)}, "
        f"failed={len(failures)}"
    )
    for label in labels:
        print(f"  {label}")
    print(f"Saved failed-task list: {output}")


def command_select_remaining(args):
    baseline = select_seed(
        load_result_dir(args.baseline_dir, args.planned_total),
        args.baseline_seed,
        expected_tasks=args.expected_tasks,
        expected_successes=args.expected_successes,
        label=str(args.baseline_dir),
    )
    baseline_failures = baseline[~baseline["strict_success"]].copy()
    failure_keys = {_task_key(row) for _, row in baseline_failures.iterrows()}

    candidate = select_seed(
        load_result_dir(args.candidate_dir, args.planned_total),
        args.candidate_seed,
        expected_tasks=args.expected_candidate_tasks,
        expected_successes=None,
        label=str(args.candidate_dir),
    )
    candidate_keys = {_task_key(row) for _, row in candidate.iterrows()}
    missing = failure_keys - candidate_keys
    unexpected = candidate_keys - failure_keys
    if missing or unexpected:
        details = []
        if missing:
            details.append(
                "missing=" + ",".join(sorted(_task_label(key) for key in missing))
            )
        if unexpected:
            details.append(
                "unexpected="
                + ",".join(sorted(_task_label(key) for key in unexpected))
            )
        raise SystemExit(
            "Candidate screen does not exactly cover the baseline failures: "
            + " ".join(details)
        )

    remaining = candidate[~candidate["strict_success"]].copy()
    labels = [_task_label(_task_key(row)) for _, row in remaining.iterrows()]
    output = Path(args.output)
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text("".join(f"{label}\n" for label in labels), encoding="ascii")

    recovered = int(candidate["strict_success"].sum())
    print(
        f"Candidate screen seed={args.candidate_seed}: "
        f"recovered={recovered}/{len(candidate)}, remaining={len(remaining)}"
    )
    for label in labels:
        print(f"  {label}")
    print(f"Saved remaining-task list: {output}")


def _prefer_candidate(previous, candidate):
    if previous is None:
        return candidate
    previous_strict = bool(previous["strict_success"])
    candidate_strict = bool(candidate["strict_success"])
    if candidate_strict != previous_strict:
        return candidate if candidate_strict else previous
    if candidate_strict:
        previous_ds = previous.get("top_ds")
        candidate_ds = candidate.get("top_ds")
        previous_ds = -float("inf") if pd.isna(previous_ds) else float(previous_ds)
        candidate_ds = -float("inf") if pd.isna(candidate_ds) else float(candidate_ds)
        return candidate if candidate_ds > previous_ds else previous
    return candidate


def command_report(args):
    baseline_all = load_result_dir(args.baseline_dir, args.planned_total)
    baseline = select_seed(
        baseline_all,
        args.baseline_seed,
        expected_tasks=args.expected_tasks,
        expected_successes=args.expected_successes,
        label=str(args.baseline_dir),
    )
    baseline_failures = baseline[~baseline["strict_success"]].copy()
    failure_keys = {_task_key(row) for _, row in baseline_failures.iterrows()}

    candidate_dirs = (
        args.candidate_dir
        if isinstance(args.candidate_dir, list)
        else [args.candidate_dir]
    )
    candidate_by_key = {}
    for candidate_dir in candidate_dirs:
        candidate = select_seed(
            load_result_dir(candidate_dir, args.planned_total),
            args.candidate_seed,
            expected_tasks=None,
            expected_successes=None,
            label=str(candidate_dir),
        )
        candidate = candidate[
            candidate.apply(lambda row: _task_key(row) in failure_keys, axis=1)
        ].copy()
        for _, row in candidate.iterrows():
            key = _task_key(row)
            candidate_by_key[key] = _prefer_candidate(candidate_by_key.get(key), row)

    records = []
    for _, base_row in baseline_failures.iterrows():
        key = _task_key(base_row)
        candidate_row = candidate_by_key.get(key)
        completed = candidate_row is not None
        recovered = completed and bool(candidate_row["strict_success"])
        records.append(
            {
                "target": key[0],
                "start_mol_idx": key[1],
                "sim_threshold": key[2],
                "baseline_seed": args.baseline_seed,
                "candidate_seed": args.candidate_seed,
                "completed": completed,
                "recovered": recovered,
                "candidate_loose_success": (
                    bool(candidate_row.get("loose_success", False))
                    if completed
                    else False
                ),
                "generated": candidate_row.get("generated") if completed else None,
                "unique": candidate_row.get("unique") if completed else None,
                "uniqueness_actual": (
                    candidate_row.get("uniqueness_actual") if completed else None
                ),
                "top_ds": candidate_row.get("top_ds") if completed else None,
                "top_sim": candidate_row.get("top_sim") if completed else None,
                "top_qed": candidate_row.get("top_qed") if completed else None,
                "top_sa": candidate_row.get("top_sa") if completed else None,
            }
        )

    report = _sort_tasks(pd.DataFrame(records).assign(seed=args.candidate_seed))
    report = report.drop(columns="seed")
    output = Path(args.output)
    output.parent.mkdir(parents=True, exist_ok=True)
    report.to_csv(output, index=False)

    completed = int(report["completed"].sum())
    recovered = int(report["recovered"].sum())
    baseline_successes = int(baseline["strict_success"].sum())
    projected_successes = baseline_successes + recovered
    print(
        f"Failed-task screen: completed={completed}/{len(report)}, "
        f"recovered={recovered}/{len(report)}"
    )
    print(
        "Screening projection (assumes prior successful tasks do not regress): "
        f"{projected_successes}/{len(baseline)} "
        f"({100.0 * projected_successes / len(baseline):.2f}%)"
    )
    recovered_rows = report[report["recovered"]]
    if not recovered_rows.empty:
        print("Recovered tasks:")
        for _, row in recovered_rows.iterrows():
            print(f"  {_task_label(_task_key(row))}")
    missing_rows = report[~report["completed"]]
    if not missing_rows.empty:
        print("Missing candidate results:")
        for _, row in missing_rows.iterrows():
            print(f"  {_task_label(_task_key(row))}")
    print(f"Saved recovery report: {output}")
    print("A full 30-task run is required before reporting an overall success rate.")


def parse_args():
    parser = argparse.ArgumentParser()
    subparsers = parser.add_subparsers(dest="command", required=True)

    select_parser = subparsers.add_parser("select")
    select_parser.add_argument("--baseline_dir", required=True)
    select_parser.add_argument("--seed", type=int, default=0)
    select_parser.add_argument("--expected_tasks", type=int, default=30)
    select_parser.add_argument("--expected_successes", type=int)
    select_parser.add_argument("--planned_total", type=int, default=1000)
    select_parser.add_argument("--output", required=True)
    select_parser.set_defaults(func=command_select)

    remaining_parser = subparsers.add_parser("select-remaining")
    remaining_parser.add_argument("--baseline_dir", required=True)
    remaining_parser.add_argument("--candidate_dir", required=True)
    remaining_parser.add_argument("--baseline_seed", type=int, default=0)
    remaining_parser.add_argument("--candidate_seed", type=int, default=0)
    remaining_parser.add_argument("--expected_tasks", type=int, default=30)
    remaining_parser.add_argument("--expected_successes", type=int)
    remaining_parser.add_argument("--expected_candidate_tasks", type=int)
    remaining_parser.add_argument("--planned_total", type=int, default=1000)
    remaining_parser.add_argument("--output", required=True)
    remaining_parser.set_defaults(func=command_select_remaining)

    report_parser = subparsers.add_parser("report")
    report_parser.add_argument("--baseline_dir", required=True)
    report_parser.add_argument("--candidate_dir", action="append", required=True)
    report_parser.add_argument("--baseline_seed", type=int, default=0)
    report_parser.add_argument("--candidate_seed", type=int, default=0)
    report_parser.add_argument("--expected_tasks", type=int, default=30)
    report_parser.add_argument("--expected_successes", type=int)
    report_parser.add_argument("--planned_total", type=int, default=1000)
    report_parser.add_argument("--output", required=True)
    report_parser.set_defaults(func=command_report)
    return parser.parse_args()


def main():
    args = parse_args()
    args.func(args)


if __name__ == "__main__":
    main()
