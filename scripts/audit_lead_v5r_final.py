#!/usr/bin/env python
"""Audit coverage, integrity, and Table-4 scoring for final Lead runs."""

from __future__ import annotations

import argparse
import os
import re
from pathlib import Path

import pandas as pd


TARGETS = ("parp1", "fa7", "5ht1b", "braf", "jak2")
START_INDICES = (0, 1, 2)
THRESHOLDS = (0.4, 0.6)
RESULT_COLUMNS = ("smiles", "DS", "QED", "SA", "SIM", "unused")
REFERENCE_SCORES = {
    "GenMol": {0.4: -148.7, 0.6: -117.7},
    "RetMol": {0.4: -88.5, 0.6: -25.7},
    "GraphGA": {0.4: -96.3, 0.6: -74.8},
    "InVirtuoGen": {0.4: -152.4, 0.6: -145.7},
}
CHUNK_TASKS = {
    0: tuple(
        (target, start_index, 0.4)
        for target in TARGETS
        for start_index in START_INDICES
    ),
    1: (("fa7", 1, 0.6),),
    2: (("5ht1b", 0, 0.6),),
    3: (("fa7", 0, 0.6), ("fa7", 2, 0.6)),
    4: (
        ("parp1", 0, 0.6),
        ("parp1", 1, 0.6),
        ("parp1", 2, 0.6),
        ("5ht1b", 1, 0.6),
        ("5ht1b", 2, 0.6),
    ),
    5: tuple(
        (target, start_index, 0.6)
        for target in ("braf", "jak2")
        for start_index in START_INDICES
    ),
}
ERROR_PATTERN = re.compile(
    r"Traceback|CUDA error|out of memory|OOM|walltime.*(?:exceed|kill)|job killed|"
    r"verification failed|task failed.*status=[1-9]",
    re.IGNORECASE,
)
TASK_START_PATTERN = re.compile(
    r"Lead v5 start: target=(?P<target>\S+) id=(?P<start>[0-2]) "
    r"threshold=(?P<threshold>0\.[46]) seed=(?P<seed>[0-2])"
)
ITERATION_PATTERN = re.compile(r"\[Iter\s+(?P<iteration>\d+)\]")


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--prefix",
        default=(
            "CSDNet/exp/lead/"
            "results_elastic_joint_frontier_v5r_final_base50k_seed"
        ),
        help="Result-directory prefix immediately before the seed number.",
    )
    parser.add_argument("--seeds", default="0,1,2")
    parser.add_argument("--log_dir", default="logs")
    parser.add_argument(
        "--run_tag_prefix",
        default="elastic_joint_v5r_final_base50k_s",
    )
    parser.add_argument("--sampler_profile", default="elastic_joint_frontier_v5")
    parser.add_argument("--oracle_budget", type=int, default=1000)
    parser.add_argument("--max_iterations", type=int, default=10)
    parser.add_argument("--output_dir")
    return parser.parse_args()


def load_start_scores(prefix):
    prefix_path = Path(prefix).resolve()
    candidates = (
        prefix_path.parent / "docking" / "actives.csv",
        Path("CSDNet/exp/lead/docking/actives.csv"),
    )
    path = next((candidate for candidate in candidates if candidate.is_file()), None)
    if path is None:
        raise SystemExit("Lead actives.csv was not found")
    frame = pd.read_csv(path)
    scores = {}
    for target, group in frame.groupby("target", sort=False):
        for start_index, row in group.reset_index(drop=True).iterrows():
            scores[(str(target), int(start_index))] = float(row["DS"])
    return scores


def read_result(path, threshold, start_ds, oracle_budget):
    result = {
        "result_exists": path.is_file(),
        "calls": 0,
        "unique_smiles": 0,
        "duplicate_rows": 0,
        "malformed_rows": 0,
        "feasibility_leaks": 0,
        "budget_excess": 0,
        "loose_success": False,
        "strict_success": False,
        "best_feasible_ds": float("nan"),
        "strict_score_signed": 0.0,
        "dock_shortfall": float("nan"),
    }
    if not path.is_file() or path.stat().st_size == 0:
        return result
    try:
        frame = pd.read_csv(path, names=RESULT_COLUMNS, header=None)
    except Exception:
        result["malformed_rows"] = 1
        return result
    result["calls"] = int(len(frame))
    result["unique_smiles"] = int(frame["smiles"].nunique(dropna=True))
    result["duplicate_rows"] = result["calls"] - result["unique_smiles"]
    result["budget_excess"] = max(0, result["calls"] - oracle_budget)
    for column in ("DS", "QED", "SA", "SIM"):
        frame[column] = pd.to_numeric(frame[column], errors="coerce")
    numeric = frame[["DS", "QED", "SA", "SIM"]].notna().all(axis=1)
    nonempty_smiles = frame["smiles"].notna() & frame["smiles"].astype(str).ne("")
    valid_rows = numeric & nonempty_smiles
    result["malformed_rows"] = int((~valid_rows).sum())
    feasible = (
        valid_rows
        & frame["QED"].ge(0.6)
        & frame["SA"].ge(6.0 / 9.0)
        & frame["SIM"].ge(threshold - 1e-12)
    )
    result["feasibility_leaks"] = int((valid_rows & ~feasible).sum())
    constrained = frame.loc[feasible]
    result["loose_success"] = bool(len(constrained))
    if len(constrained):
        best = float(constrained["DS"].max())
        result["best_feasible_ds"] = best
        result["dock_shortfall"] = max(0.0, start_ds - best)
        result["strict_success"] = best > start_ds
        if result["strict_success"]:
            result["strict_score_signed"] = -best
    return result


def marker_paths(directory, run_tag, sampler_profile, seed, chunk, task):
    target, start_index, threshold = task
    task_marker = directory / (
        f".{run_tag}_{target}_id{start_index}_thr{threshold}.done"
    )
    chunk_marker = directory / (
        f".seed{seed}_{sampler_profile}_chunk{chunk}.done"
    )
    return task_marker, chunk_marker


def find_chunk(task):
    for chunk, tasks in CHUNK_TASKS.items():
        if task in tasks:
            return chunk
    raise RuntimeError(f"Task is absent from the final chunk map: {task}")


def inspect_log(path):
    if not path.is_file():
        return "MISSING", float("nan"), ""
    age_minutes = max(0.0, (pd.Timestamp.now().timestamp() - path.stat().st_mtime) / 60.0)
    try:
        text = path.read_text(errors="replace")
    except OSError as exc:
        return "UNREADABLE", age_minutes, str(exc)
    lines = text.splitlines()
    matches = [line.strip() for line in lines if ERROR_PATTERN.search(line)]
    completion = [
        line.strip() for line in lines if "Lead v5r chunk complete:" in line
    ]
    if completion and "failed=0" in completion[-1]:
        return "OK", age_minutes, ""
    if completion and "failed=1" in completion[-1]:
        return "ERROR", age_minutes, completion[-1]
    return ("ERROR" if matches else "ACTIVE"), age_minutes, (matches[-1] if matches else "")


def parse_task_log(path):
    """Return the latest outer-loop progress segment for each task in a chunk."""
    if not path.is_file():
        return {}
    try:
        lines = path.read_text(errors="replace").splitlines()
    except OSError:
        return {}
    segments = {}
    current = None
    for line in lines:
        start = TASK_START_PATTERN.search(line)
        if start:
            current = (
                start.group("target"),
                int(start.group("start")),
                float(start.group("threshold")),
                int(start.group("seed")),
            )
            segments[current] = {
                "outer_iteration": 0,
                "log_task_done": False,
                "log_no_feasible": False,
                "log_task_failed": False,
            }
            continue
        if current is None:
            continue
        iteration = ITERATION_PATTERN.search(line)
        if iteration:
            segments[current]["outer_iteration"] = max(
                segments[current]["outer_iteration"],
                int(iteration.group("iteration")),
            )
        if "Lead v5r done:" in line:
            segments[current]["log_task_done"] = True
        if "Lead v5r produced no feasible docking candidate:" in line:
            segments[current]["log_no_feasible"] = True
        if ERROR_PATTERN.search(line):
            segments[current]["log_task_failed"] = True
    return segments


def task_status(row, max_iterations):
    integrity_error = any(
        int(row[column]) > 0
        for column in (
            "duplicate_rows",
            "malformed_rows",
            "feasibility_leaks",
            "budget_excess",
        )
    )
    if row["task_marker"] and row["result_exists"] and not integrity_error:
        return "DONE"
    if row["task_marker"] and not row["result_exists"]:
        return "BAD_MARKER"
    if (
        row["log_no_feasible"]
        and not row["log_task_failed"]
        and int(row["outer_iteration"]) >= int(max_iterations)
        and not row["result_exists"]
    ):
        return "DONE_ZERO"
    if row["log_task_failed"]:
        return "FAILED"
    if row["result_exists"]:
        return "PARTIAL"
    if int(row["outer_iteration"]) > 0:
        return "INCOMPLETE"
    return "MISSING"


def fmt_best(value):
    return "-" if pd.isna(value) else f"{float(value):.2f}"


def main():
    args = parse_args()
    seeds = tuple(int(value) for value in args.seeds.split(",") if value)
    start_scores = load_start_scores(args.prefix)
    rows = []
    chunk_rows = []

    for seed in seeds:
        directory = Path(f"{args.prefix}{seed}")
        run_tag = f"{args.run_tag_prefix}{seed}"
        task_log_progress = {}
        for chunk, tasks in CHUNK_TASKS.items():
            log = Path(args.log_dir) / f"csdnet_lead_{run_tag}_chunk{chunk}_live.log"
            task_log_progress.update(parse_task_log(log))
            log_state, log_age, log_error = inspect_log(log)
            _, chunk_marker = marker_paths(
                directory,
                run_tag,
                args.sampler_profile,
                seed,
                chunk,
                tasks[0],
            )
            chunk_rows.append(
                {
                    "seed": seed,
                    "chunk": chunk,
                    "chunk_marker": chunk_marker.is_file(),
                    "log_state": log_state,
                    "log_age_min": log_age,
                    "last_error": log_error,
                    "log": os.fspath(log),
                }
            )

        for target in TARGETS:
            for start_index in START_INDICES:
                for threshold in THRESHOLDS:
                    task = (target, start_index, threshold)
                    chunk = find_chunk(task)
                    path = directory / (
                        f"{target}_id{start_index}_thr{threshold}_{seed}.csv"
                    )
                    start_ds = start_scores[(target, start_index)]
                    result = read_result(
                        path,
                        threshold,
                        start_ds,
                        args.oracle_budget,
                    )
                    task_marker, chunk_marker = marker_paths(
                        directory,
                        run_tag,
                        args.sampler_profile,
                        seed,
                        chunk,
                        task,
                    )
                    row = {
                        "seed": seed,
                        "chunk": chunk,
                        "target": target,
                        "start_mol_idx": start_index,
                        "sim_threshold": threshold,
                        "start_ds": start_ds,
                        "task_marker": task_marker.is_file(),
                        "chunk_marker": chunk_marker.is_file(),
                        "file": os.fspath(path),
                        **result,
                    }
                    row.update(
                        task_log_progress.get(
                            (target, start_index, threshold, seed),
                            {
                                "outer_iteration": 0,
                                "log_task_done": False,
                                "log_no_feasible": False,
                                "log_task_failed": False,
                            },
                        )
                    )
                    row["status"] = task_status(row, args.max_iterations)
                    rows.append(row)

    frame = pd.DataFrame(rows)
    chunks = pd.DataFrame(chunk_rows)
    output_dir = Path(
        args.output_dir
        or f"{args.prefix.removesuffix('seed')}audit"
    )
    output_dir.mkdir(parents=True, exist_ok=True)
    frame.to_csv(output_dir / "lead_audit_runs.csv", index=False)
    chunks.to_csv(output_dir / "lead_audit_chunks.csv", index=False)

    print("Lead v5r final audit")
    print("Feasible = QED >= 0.6, normalized SA >= 6/9, similarity >= delta.")
    print("Strict = feasible and docking score improves over the starting molecule.\n")
    for seed in seeds:
        subset = frame[frame["seed"].eq(seed)]
        counts = subset["status"].value_counts().to_dict()
        complete_count = counts.get("DONE", 0) + counts.get("DONE_ZERO", 0)
        print("=" * 104)
        print(
            f"SEED {seed}: COMPLETE={complete_count}/30 "
            f"DONE={counts.get('DONE', 0)} "
            f"DONE_ZERO={counts.get('DONE_ZERO', 0)} "
            f"PARTIAL={counts.get('PARTIAL', 0)} "
            f"INCOMPLETE={counts.get('INCOMPLETE', 0)} "
            f"FAILED={counts.get('FAILED', 0)} "
            f"MISSING={counts.get('MISSING', 0)} "
            f"BAD_MARKER={counts.get('BAD_MARKER', 0)}"
        )
        print(
            "chunk task                 status    iter calls strict loose "
            "best/start leaks dup"
        )
        print("-" * 104)
        for row in subset.sort_values(["chunk", "target", "start_mol_idx"]).itertuples():
            task = f"{row.target}:id{row.start_mol_idx}:d{row.sim_threshold:g}"
            print(
                f"{row.chunk:5d} {task:20s} {row.status:9s} "
                f"{row.outer_iteration:4d} {row.calls:5d} "
                f"{str(bool(row.strict_success)):6s} {str(bool(row.loose_success)):5s} "
                f"{fmt_best(row.best_feasible_ds):>5s}/{row.start_ds:.2f} "
                f"{row.feasibility_leaks:5d} {row.duplicate_rows:3d}"
            )
        print()
        for threshold in THRESHOLDS:
            threshold_rows = subset[subset["sim_threshold"].eq(threshold)]
            completed = threshold_rows["status"].isin(("DONE", "DONE_ZERO"))
            score = float(threshold_rows["strict_score_signed"].sum())
            print(
                f"delta={threshold:g}: done={int(completed.sum())}/15, "
                f"calls={int(threshold_rows['calls'].sum())}/15000, "
                f"strict={int(threshold_rows['strict_success'].sum())}/15, "
                f"strict_DS_sum={score:.3f}"
                + (" [OFFICIAL SEED SUM]" if completed.all() else " [STAGE ONLY]")
            )

    print("\n" + "=" * 104)
    print("THREE-SEED SCORE AUDIT")
    all_integrity_ok = not frame[
        ["duplicate_rows", "malformed_rows", "feasibility_leaks", "budget_excess"]
    ].astype(int).any(axis=None)
    for threshold in THRESHOLDS:
        threshold_rows = frame[frame["sim_threshold"].eq(threshold)]
        complete = threshold_rows["status"].isin(("DONE", "DONE_ZERO")).all()
        seed_scores = threshold_rows.groupby("seed")["strict_score_signed"].sum()
        mean_score = float(seed_scores.reindex(seeds, fill_value=0.0).mean())
        std_score = float(seed_scores.reindex(seeds, fill_value=0.0).std(ddof=1))
        print(
            f"delta={threshold:g}: complete={complete}, "
            f"strict_DS_sum={mean_score:.3f} +/- {std_score:.3f}"
            + (" [OFFICIAL]" if complete and all_integrity_ok else " [NOT OFFICIAL]")
        )
        if complete and all_integrity_ok:
            for method, reference in REFERENCE_SCORES.items():
                gap = abs(mean_score) - abs(reference[threshold])
                relation = "ahead" if gap > 0 else "behind"
                print(
                    f"  vs {method:12s} {reference[threshold]:7.1f}: "
                    f"{relation} by {abs(gap):.3f}"
                )

    integrity = frame[
        ["duplicate_rows", "malformed_rows", "feasibility_leaks", "budget_excess"]
    ].sum()
    print("\nIntegrity totals:")
    for column, value in integrity.items():
        print(f"  {column}: {int(value)}")
    print("\nChunk/log audit:")
    for row in chunks.itertuples(index=False):
        age = "-" if pd.isna(row.log_age_min) else f"{row.log_age_min:.1f}m"
        detail = f" | {row.last_error}" if row.last_error else ""
        print(
            f"  seed={row.seed} chunk={row.chunk} marker={row.chunk_marker} "
            f"log={row.log_state} idle={age}{detail}"
        )
    print(f"\nSaved run audit:   {output_dir / 'lead_audit_runs.csv'}")
    print(f"Saved chunk audit: {output_dir / 'lead_audit_chunks.csv'}")


if __name__ == "__main__":
    main()
