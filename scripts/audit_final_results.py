#!/usr/bin/env python3
"""Fast, dependency-free audit of the dissertation's final benchmark evidence."""

from __future__ import annotations

import argparse
import csv
import json
import math
import statistics
import sys
from collections import Counter, defaultdict
from datetime import datetime, timezone
from pathlib import Path


class AuditError(RuntimeError):
    pass


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--root", default=".", help="Project root containing CSDNet/ and results/.")
    parser.add_argument(
        "--lock",
        default=str(Path(__file__).with_name("final_result_lock.json")),
        help="Machine-readable final-result lock.",
    )
    parser.add_argument(
        "--benchmarks",
        default="denovo,fragment_infill,lead,pmo",
        help="Comma-separated subset to audit.",
    )
    parser.add_argument("--output-json")
    parser.add_argument("--lead-summary-csv")
    return parser.parse_args()


def fail(message):
    raise AuditError(message)


def require_file(path):
    if not path.is_file() or path.stat().st_size == 0:
        fail(f"Missing or empty required file: {path}")
    return path


def require_dir(path):
    if not path.is_dir():
        fail(f"Missing required directory: {path}")
    return path


def read_csv(path):
    require_file(path)
    with path.open(newline="", encoding="utf-8-sig") as handle:
        return list(csv.DictReader(handle))


def count_nonempty_lines(path):
    require_file(path)
    with path.open(encoding="utf-8", errors="replace") as handle:
        return sum(bool(line.strip()) for line in handle)


def as_float(value, label):
    try:
        result = float(value)
    except (TypeError, ValueError) as exc:
        fail(f"Expected a number for {label}, got {value!r}: {exc}")
    if not math.isfinite(result):
        fail(f"Expected a finite number for {label}, got {value!r}")
    return result


def as_int(value, label):
    number = as_float(value, label)
    rounded = int(round(number))
    if not math.isclose(number, rounded, abs_tol=1e-9):
        fail(f"Expected an integer for {label}, got {value!r}")
    return rounded


def as_bool(value, label):
    text = str(value).strip().lower()
    if text in {"true", "1", "yes"}:
        return True
    if text in {"false", "0", "no", ""}:
        return False
    fail(f"Expected a Boolean for {label}, got {value!r}")


def assert_close(actual, expected, label, tolerance=5e-4):
    if not math.isclose(float(actual), float(expected), rel_tol=0.0, abs_tol=tolerance):
        fail(f"{label}: expected {expected}, observed {actual}")


def assert_exact_keys(observed, expected, label):
    observed = set(observed)
    expected = set(expected)
    if observed != expected:
        fail(
            f"{label}: missing={sorted(expected - observed)}, "
            f"extra={sorted(observed - expected)}"
        )


def resolve_recorded_path(root, value):
    path = Path(str(value))
    return path if path.is_absolute() else root / path


def audit_denovo(root, config):
    directory = require_dir(root / config["directory"])
    aggregate_path = directory / config["aggregate"]
    aggregate = read_csv(aggregate_path)
    if len(aggregate) != 1:
        fail(f"De novo aggregate must contain one row: {aggregate_path}")
    row = aggregate[0]
    if row.get("profile") != config["profile"]:
        fail(f"Wrong de novo profile: {row.get('profile')!r}")
    if as_int(row.get("n_steps"), "de novo n_steps") != int(config["steps"]):
        fail("Wrong de novo step count")
    if as_int(row.get("n_seeds"), "de novo n_seeds") != len(config["seeds"]):
        fail("Wrong de novo seed count")
    for column, expected in config["expected"].items():
        assert_close(as_float(row.get(column), f"de novo {column}"), expected, f"de novo {column}")

    runs = read_csv(directory / config["runs"])
    run_by_seed = {as_int(item.get("seed"), "de novo run seed"): item for item in runs}
    assert_exact_keys(run_by_seed, config["seeds"], "de novo run seeds")
    if len(runs) != len(run_by_seed):
        fail("Duplicate de novo seed rows")
    generated_counts = {}
    for seed in config["seeds"]:
        run = run_by_seed[seed]
        if run.get("profile") != config["profile"]:
            fail(f"De novo seed {seed} used profile {run.get('profile')!r}")
        if as_int(run.get("n_steps"), f"de novo seed {seed} n_steps") != int(config["steps"]):
            fail(f"De novo seed {seed} used the wrong number of steps")
        if as_int(run.get("accepted"), f"de novo seed {seed} accepted") != int(config["molecules_per_seed"]):
            fail(f"De novo seed {seed} did not deliver 1000 molecules")
        seed_dir = require_dir(
            directory / f"steps{config['steps']}" / config["profile"] / f"seed{seed}"
        )
        generated = seed_dir / "generated_mols.txt"
        generated_counts[str(seed)] = count_nonempty_lines(generated)
        if generated_counts[str(seed)] != int(config["molecules_per_seed"]):
            fail(f"De novo seed {seed}: expected 1000 generated lines")
        require_file(seed_dir / "genmol_denovo_metrics.json")
        require_file(seed_dir / "sampling_diagnostics.json")

    newest_input = max(
        (directory / f"steps{config['steps']}" / config["profile"] / f"seed{s}" / "genmol_denovo_metrics.json").stat().st_mtime
        for s in config["seeds"]
    )
    if aggregate_path.stat().st_mtime + 1e-6 < newest_input:
        fail("De novo aggregate is older than a per-seed metrics file")
    return {
        "directory": config["directory"],
        "profile": config["profile"],
        "seeds": config["seeds"],
        "generated_lines": generated_counts,
        "metrics": {key: as_float(row[key], key) for key in config["expected"]},
    }


def audit_fragment(root, config):
    directory = require_dir(root / config["directory"])
    aggregate_path = directory / config["aggregate"]
    aggregate = read_csv(aggregate_path)
    if len(aggregate) != 1:
        fail(f"Fragment aggregate must contain one row: {aggregate_path}")
    row = aggregate[0]
    if as_int(row.get("n_tasks"), "fragment n_tasks") != len(config["tasks"]):
        fail("Wrong fragment task count")
    if as_int(row.get("n_seeds"), "fragment n_seeds") != len(config["seeds"]):
        fail("Wrong fragment seed count")
    expected_cases = len(config["tasks"]) * len(config["seeds"]) * int(config["cases_per_task_seed"])
    if as_int(row.get("n_cases"), "fragment n_cases") != expected_cases:
        fail("Wrong fragment case count")
    for column, expected in config["expected"].items():
        assert_close(as_float(row.get(column), f"fragment {column}"), expected, f"fragment {column}")

    by_task_rows = read_csv(directory / config["by_task_seed"])
    expected_keys = {(task, int(seed)) for task in config["tasks"] for seed in config["seeds"]}
    keyed = {}
    newest_raw = 0.0
    for item in by_task_rows:
        key = (item.get("task"), as_int(item.get("seed"), "fragment seed"))
        if key in keyed:
            fail(f"Duplicate fragment task/seed row: {key}")
        keyed[key] = item
    assert_exact_keys(keyed, expected_keys, "fragment task/seed coverage")

    for task, seed in sorted(expected_keys):
        paths = {
            "metrics": directory / f"metrics_{task}_seed{seed}.csv",
            "summary": directory / f"summary_{task}_seed{seed}.csv",
            "samples": directory / f"samples_{task}_seed{seed}.csv",
            "attempts": directory / f"attempts_{task}_seed{seed}.csv",
        }
        for path in paths.values():
            require_file(path)
            newest_raw = max(newest_raw, path.stat().st_mtime)
        if len(read_csv(paths["metrics"])) != int(config["cases_per_task_seed"]):
            fail(f"Fragment {task} seed {seed}: expected 10 case-metric rows")
        if len(read_csv(paths["summary"])) != 1:
            fail(f"Fragment {task} seed {seed}: expected one summary row")
        if len(read_csv(paths["attempts"])) != int(config["proposals_per_task_seed"]):
            fail(f"Fragment {task} seed {seed}: expected 1000 attempt rows")
    if aggregate_path.stat().st_mtime + 1e-6 < newest_raw:
        fail("Fragment aggregate is older than a final raw task file")
    return {
        "directory": config["directory"],
        "task_seed_cells": len(keyed),
        "attempt_rows": len(keyed) * int(config["proposals_per_task_seed"]),
        "metrics": {key: as_float(row[key], key) for key in config["expected"]},
        "version_note": config.get("note"),
    }


def audit_lead(root, config):
    prefix = config["directory_prefix"]
    audit_dir = require_dir(root / config["audit_directory"])
    audit_path = audit_dir / config["audit_runs"]
    chunk_path = audit_dir / config["audit_chunks"]
    rows = read_csv(audit_path)
    chunks = read_csv(chunk_path)
    expected_keys = {
        (int(seed), target, int(index), float(threshold))
        for seed in config["seeds"]
        for target in config["targets"]
        for index in config["start_indices"]
        for threshold in config["thresholds"]
    }
    keyed = {}
    score_sums = defaultdict(float)
    calls = defaultdict(int)
    strict = defaultdict(int)
    status_counts = defaultdict(Counter)
    newest_result = 0.0

    for item in rows:
        seed = as_int(item.get("seed"), "Lead seed")
        target = item.get("target")
        index = as_int(item.get("start_mol_idx"), "Lead start index")
        threshold = round(as_float(item.get("sim_threshold"), "Lead threshold"), 1)
        key = (seed, target, index, threshold)
        if key in keyed:
            fail(f"Duplicate Lead audit cell: {key}")
        keyed[key] = item
        status = item.get("status")
        if status not in {"DONE", "DONE_ZERO"}:
            fail(f"Non-terminal Lead cell {key}: status={status!r}")
        status_counts[seed][status] += 1
        for column in ("duplicate_rows", "malformed_rows", "feasibility_leaks", "budget_excess"):
            if as_int(item.get(column), f"Lead {key} {column}") != 0:
                fail(f"Lead integrity failure in {key}: {column}={item.get(column)}")

        expected_file = root / f"{prefix}{seed}" / f"{target}_id{index}_thr{threshold}_{seed}.csv"
        recorded_file = Path(str(item.get("file", "")))
        if recorded_file.name != expected_file.name:
            fail(
                f"Lead audit cell {key} records the wrong result basename: "
                f"{recorded_file.name!r}"
            )
        cell_calls = as_int(item.get("calls"), f"Lead {key} calls")
        calls[(seed, threshold)] += cell_calls
        strict[(seed, threshold)] += int(as_bool(item.get("strict_success"), f"Lead {key} strict"))
        score_sums[(seed, threshold)] += as_float(
            item.get("strict_score_signed"), f"Lead {key} strict score"
        )
        if status == "DONE":
            if not as_bool(item.get("result_exists"), f"Lead {key} result_exists"):
                fail(f"Lead DONE cell lacks result evidence: {key}")
            require_file(expected_file)
            newest_result = max(newest_result, expected_file.stat().st_mtime)
            if count_nonempty_lines(expected_file) != cell_calls:
                fail(f"Lead {key}: audit calls do not match result rows")
        else:
            if cell_calls != 0 or as_bool(item.get("result_exists"), f"Lead {key} result_exists"):
                fail(f"Lead DONE_ZERO cell unexpectedly has result rows: {key}")
            if expected_file.exists() and expected_file.stat().st_size:
                fail(f"Lead DONE_ZERO cell unexpectedly has a non-empty CSV: {expected_file}")
            if as_int(item.get("outer_iteration"), f"Lead {key} iterations") < int(config["max_iterations"]):
                fail(f"Lead DONE_ZERO cell stopped before the final iteration: {key}")
            if not as_bool(item.get("log_no_feasible"), f"Lead {key} log_no_feasible"):
                fail(f"Lead DONE_ZERO cell lacks no-feasible log evidence: {key}")
            if as_bool(item.get("strict_success"), f"Lead {key} strict") or as_bool(
                item.get("loose_success"), f"Lead {key} loose"
            ):
                fail(f"Lead DONE_ZERO cell is incorrectly marked successful: {key}")

    assert_exact_keys(keyed, expected_keys, "Lead terminal-cell coverage")
    if len(rows) != len(keyed):
        fail("Duplicate Lead audit rows")
    expected_chunk_keys = {(int(seed), chunk) for seed in config["seeds"] for chunk in range(6)}
    observed_chunk_keys = {
        (as_int(row.get("seed"), "Lead chunk seed"), as_int(row.get("chunk"), "Lead chunk"))
        for row in chunks
    }
    assert_exact_keys(observed_chunk_keys, expected_chunk_keys, "Lead chunk audit coverage")
    if audit_path.stat().st_mtime + 1e-6 < newest_result:
        fail("Lead audit is older than a final result CSV")

    summary_rows = []
    for seed in config["seeds"]:
        expected_zero = int(config["expected_done_zero_by_seed"][str(seed)])
        if status_counts[seed]["DONE_ZERO"] != expected_zero:
            fail(
                f"Lead seed {seed}: expected {expected_zero} DONE_ZERO cells, "
                f"observed {status_counts[seed]['DONE_ZERO']}"
            )
        if sum(status_counts[seed].values()) != 30:
            fail(f"Lead seed {seed}: expected 30 terminal cells")
        for threshold in config["thresholds"]:
            threshold = round(float(threshold), 1)
            expected_score = config["expected_seed_score_sums"][str(seed)][str(threshold)]
            assert_close(
                score_sums[(seed, threshold)],
                expected_score,
                f"Lead seed {seed} delta={threshold} score sum",
                tolerance=1e-6,
            )
            threshold_rows = [
                row
                for (cell_seed, _target, _index, cell_threshold), row in keyed.items()
                if cell_seed == seed and cell_threshold == threshold
            ]
            summary_rows.append(
                {
                    "scope": "seed",
                    "seed": seed,
                    "sim_threshold": threshold,
                    "terminal_cells": len(threshold_rows),
                    "result_cells": sum(row["status"] == "DONE" for row in threshold_rows),
                    "done_zero_cells": sum(row["status"] == "DONE_ZERO" for row in threshold_rows),
                    "oracle_calls": calls[(seed, threshold)],
                    "strict_successes": strict[(seed, threshold)],
                    "strict_score_signed": score_sums[(seed, threshold)],
                    "three_seed_mean": "",
                    "three_seed_sample_std": "",
                }
            )

    three_seed = {}
    for threshold in config["thresholds"]:
        threshold = round(float(threshold), 1)
        values = [score_sums[(seed, threshold)] for seed in config["seeds"]]
        mean = statistics.mean(values)
        std = statistics.stdev(values)
        expected = config["expected_three_seed"][str(threshold)]
        assert_close(mean, expected["mean"], f"Lead delta={threshold} mean", tolerance=1e-6)
        assert_close(std, expected["sample_std"], f"Lead delta={threshold} SD", tolerance=1e-6)
        three_seed[str(threshold)] = {"seed_sums": values, "mean": mean, "sample_std": std}
        summary_rows.append(
            {
                "scope": "three_seed",
                "seed": "",
                "sim_threshold": threshold,
                "terminal_cells": 45,
                "result_cells": sum(row["status"] == "DONE" for key, row in keyed.items() if key[3] == threshold),
                "done_zero_cells": sum(row["status"] == "DONE_ZERO" for key, row in keyed.items() if key[3] == threshold),
                "oracle_calls": sum(calls[(seed, threshold)] for seed in config["seeds"]),
                "strict_successes": sum(strict[(seed, threshold)] for seed in config["seeds"]),
                "strict_score_signed": "",
                "three_seed_mean": mean,
                "three_seed_sample_std": std,
            }
        )
    return {
        "directory_prefix": prefix,
        "audit_directory": config["audit_directory"],
        "terminal_cells": len(keyed),
        "result_cells": sum(row["status"] == "DONE" for row in rows),
        "done_zero_cells": sum(row["status"] == "DONE_ZERO" for row in rows),
        "status_by_seed": {str(seed): dict(status_counts[seed]) for seed in config["seeds"]},
        "three_seed": three_seed,
        "protocol_note": config.get("note"),
        "summary_rows": summary_rows,
    }


def audit_pmo(root, config):
    prefix = config["directory_prefix"]
    expected_tasks = set(config["tasks"])
    seed_sums = {}
    newest_summary = 0.0
    for seed in config["seeds"]:
        directory = require_dir(root / f"{prefix}{seed}")
        summary_path = directory / f"summary_{config['mode']}.csv"
        rows = read_csv(summary_path)
        keyed = {}
        for item in rows:
            if item.get("mode") != config["mode"] or as_int(item.get("seed"), "PMO seed") != int(seed):
                fail(f"PMO seed {seed} summary contains a foreign mode/seed row")
            task = item.get("oracle")
            if task in keyed:
                fail(f"PMO seed {seed} contains duplicate task {task}")
            keyed[task] = item
        assert_exact_keys(keyed, expected_tasks, f"PMO seed {seed} task coverage")
        if len(rows) != len(keyed):
            fail(f"PMO seed {seed} contains duplicate summary rows")
        total = 0.0
        for task, item in keyed.items():
            calls = as_int(item.get("calls"), f"PMO seed {seed} {task} calls")
            if calls != int(config["oracle_calls_per_task"]):
                fail(f"PMO seed {seed} {task}: expected exactly 10000 calls, observed {calls}")
            history = directory / f"{task}_{seed}.csv"
            if count_nonempty_lines(history) != calls:
                fail(f"PMO seed {seed} {task}: history rows do not match calls")
            total += as_float(item.get("auc_top10"), f"PMO seed {seed} {task} AUC")
        assert_close(
            total,
            config["expected_seed_sums"][str(seed)],
            f"PMO seed {seed} sum AUC top-10",
        )
        seed_sums[str(seed)] = total
        newest_summary = max(newest_summary, summary_path.stat().st_mtime)

    summary_dir = require_dir(root / config["summary_directory"])
    by_seed_path = summary_dir / "pmo_by_seed.csv"
    overall_path = summary_dir / "pmo_overall_3seed.csv"
    by_seed = read_csv(by_seed_path)
    if len(by_seed) != len(config["seeds"]):
        fail("PMO three-seed by-seed table must contain three rows")
    for item in by_seed:
        seed = as_int(item.get("seed"), "PMO aggregate seed")
        assert_close(
            as_float(item.get("sum_auc_top10"), f"PMO aggregate seed {seed}"),
            config["expected_seed_sums"][str(seed)],
            f"PMO aggregate seed {seed}",
        )
        if as_int(item.get("n_tasks"), f"PMO seed {seed} n_tasks") != len(expected_tasks):
            fail(f"PMO aggregate seed {seed} has the wrong task count")
    overall = read_csv(overall_path)
    if len(overall) != 1:
        fail("PMO overall table must contain one row")
    overall_row = overall[0]
    if overall_row.get("mode") != config["mode"]:
        fail("PMO overall table has the wrong mode")
    if as_int(overall_row.get("n_seeds"), "PMO overall n_seeds") != 3:
        fail("PMO overall table has the wrong seed count")
    if as_int(overall_row.get("n_tasks"), "PMO overall n_tasks") != len(expected_tasks):
        fail("PMO overall table has the wrong task count")
    assert_close(
        as_float(overall_row.get("sum_auc_top10_3seed_mean"), "PMO overall mean"),
        config["expected_three_seed"]["mean"],
        "PMO three-seed mean",
    )
    assert_close(
        as_float(overall_row.get("sum_auc_top10_3seed_std"), "PMO overall SD"),
        config["expected_three_seed"]["sample_std"],
        "PMO three-seed SD",
    )
    if overall_path.stat().st_mtime + 1e-6 < newest_summary:
        fail("PMO three-seed aggregate is older than a final seed summary")
    return {
        "directory_prefix": prefix,
        "summary_directory": config["summary_directory"],
        "mode": config["mode"],
        "task_seed_histories": len(expected_tasks) * len(config["seeds"]),
        "oracle_calls_per_history": config["oracle_calls_per_task"],
        "seed_sums": seed_sums,
        "three_seed_mean": as_float(overall_row["sum_auc_top10_3seed_mean"], "PMO mean"),
        "three_seed_sample_std": as_float(overall_row["sum_auc_top10_3seed_std"], "PMO SD"),
    }


def write_lead_summary(path, rows):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    fields = [
        "scope",
        "seed",
        "sim_threshold",
        "terminal_cells",
        "result_cells",
        "done_zero_cells",
        "oracle_calls",
        "strict_successes",
        "strict_score_signed",
        "three_seed_mean",
        "three_seed_sample_std",
    ]
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()
        writer.writerows(rows)


def main():
    args = parse_args()
    root = Path(args.root).expanduser().resolve()
    lock_path = Path(args.lock).expanduser().resolve()
    require_file(lock_path)
    with lock_path.open(encoding="utf-8") as handle:
        lock = json.load(handle)
    selected = [item.strip() for item in args.benchmarks.split(",") if item.strip()]
    available = {"denovo", "fragment_infill", "lead", "pmo"}
    unknown = sorted(set(selected) - available)
    if unknown:
        fail(f"Unknown benchmark names: {unknown}")
    require_file(root / lock["checkpoint"])
    auditors = {
        "denovo": audit_denovo,
        "fragment_infill": audit_fragment,
        "lead": audit_lead,
        "pmo": audit_pmo,
    }
    results = {}
    for name in selected:
        print(f"Auditing {name} ...", flush=True)
        results[name] = auditors[name](root, lock["benchmarks"][name])
        print(f"  PASS: {name}", flush=True)
    report = {
        "status": "PASS",
        "audited_at_utc": datetime.now(timezone.utc).isoformat(),
        "root": str(root),
        "lock": str(lock_path),
        "lock_version": lock["lock_version"],
        "checkpoint": lock["checkpoint"],
        "benchmarks": results,
    }
    if args.output_json:
        output = Path(args.output_json)
        output.parent.mkdir(parents=True, exist_ok=True)
        output.write_text(json.dumps(report, indent=2, sort_keys=True) + "\n", encoding="utf-8")
        print(f"Saved audit report: {output}")
    if args.lead_summary_csv:
        if "lead" not in results:
            fail("--lead-summary-csv requires the Lead benchmark to be audited")
        write_lead_summary(args.lead_summary_csv, results["lead"].pop("summary_rows"))
        print(f"Saved Lead score summary: {args.lead_summary_csv}")
    print("FINAL RESULT AUDIT: PASS")


if __name__ == "__main__":
    try:
        main()
    except (AuditError, KeyError, json.JSONDecodeError) as exc:
        print(f"FINAL RESULT AUDIT FAILED: {exc}", file=sys.stderr)
        raise SystemExit(2)
