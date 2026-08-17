#!/usr/bin/env python
"""Lightweight PMO history and reporting helpers.

This module intentionally avoids importing the optimizer, model, RDKit, PyTorch,
or Transformers.  Reporting commands run on login nodes and should only need the
saved oracle histories plus NumPy/YAML.
"""

from __future__ import annotations

import os

import numpy as np
import yaml


PMO_TASKS = (
    "albuterol_similarity",
    "amlodipine_mpo",
    "celecoxib_rediscovery",
    "deco_hop",
    "drd2",
    "fexofenadine_mpo",
    "gsk3b",
    "isomers_c7h8n2o2",
    "isomers_c9h10n2o2pf2cl",
    "jnk3",
    "median1",
    "median2",
    "mestranol_similarity",
    "osimertinib_mpo",
    "perindopril_mpo",
    "qed",
    "ranolazine_mpo",
    "scaffold_hop",
    "sitagliptin_mpo",
    "thiothixene_rediscovery",
    "troglitazone_rediscovery",
    "valsartan_smarts",
    "zaleplon_mpo",
)


def result_path(output_dir, mode, oracle_name, seed):
    return os.path.join(
        output_dir,
        f"results_CSDNet_{mode}_{oracle_name}_{seed}.yaml",
    )


def completed(output_dir, mode, oracle_name, seed, max_oracle_calls):
    path = result_path(output_dir, mode, oracle_name, seed)
    if not os.path.exists(path):
        return False
    try:
        with open(path) as handle:
            data = yaml.safe_load(handle) or {}
    except Exception:
        return False
    return isinstance(data, dict) and len(data) >= max_oracle_calls


def load_resume_buffer(output_dir, mode, oracle_name, seed, max_oracle_calls):
    """Load and cross-check the canonical YAML/CSV oracle history."""
    path = result_path(output_dir, mode, oracle_name, seed)
    score_path = os.path.join(output_dir, f"{oracle_name}_{seed}.csv")
    yaml_rows = []
    if os.path.exists(path):
        with open(path) as handle:
            data = yaml.safe_load(handle) or {}
        if not isinstance(data, dict):
            raise RuntimeError(f"Resume checkpoint is not a mapping: {path}")
        seen_calls = set()
        for smiles, value in data.items():
            if (
                not isinstance(smiles, str)
                or not isinstance(value, (list, tuple))
                or len(value) < 2
            ):
                raise RuntimeError(
                    f"Malformed resume entry in {path}: {smiles!r}: {value!r}"
                )
            score = float(value[0])
            call_index = int(value[1])
            if call_index < 1 or call_index > max_oracle_calls:
                raise RuntimeError(f"Invalid call index {call_index} in {path}")
            if call_index in seen_calls:
                raise RuntimeError(f"Duplicate call index {call_index} in {path}")
            seen_calls.add(call_index)
            yaml_rows.append((call_index, smiles, score))
        yaml_rows.sort()

    csv_rows = []
    if os.path.exists(score_path):
        seen_smiles = set()
        with open(score_path) as handle:
            for line_number, raw in enumerate(handle, start=1):
                raw = raw.rstrip("\n")
                if not raw:
                    continue
                try:
                    smiles, score_text = raw.rsplit(",", 1)
                    score = float(score_text)
                except Exception as exc:
                    raise RuntimeError(
                        f"Malformed score row {line_number} in {score_path}"
                    ) from exc
                if smiles in seen_smiles:
                    raise RuntimeError(
                        f"Duplicate molecule at row {line_number} in {score_path}"
                    )
                seen_smiles.add(smiles)
                csv_rows.append((len(csv_rows) + 1, smiles, score))
                if len(csv_rows) >= max_oracle_calls:
                    break

    rows = csv_rows if len(csv_rows) > len(yaml_rows) else yaml_rows
    if yaml_rows and csv_rows:
        shared = min(len(yaml_rows), len(csv_rows))
        for yaml_row, csv_row in zip(yaml_rows[:shared], csv_rows[:shared]):
            if (
                yaml_row[1] != csv_row[1]
                or abs(yaml_row[2] - csv_row[2]) > 1e-8
            ):
                raise RuntimeError(
                    f"YAML and score CSV histories diverge at call {yaml_row[0]} "
                    f"for {oracle_name}; refusing an unsafe resume."
                )

    expected = list(range(1, len(rows) + 1))
    observed = [call_index for call_index, _, _ in rows]
    if observed != expected:
        raise RuntimeError(
            f"Non-contiguous oracle call history for {oracle_name}: "
            f"expected 1..{len(rows)}, got {observed[:3]}...{observed[-3:]}"
        )
    return {
        smiles: [score, call_index]
        for call_index, smiles, score in rows
    }


def top_auc(buffer, top_n, finish, freq_log, max_oracle_calls):
    """Compute the benchmark AUC using the original PMO implementation."""
    area = 0.0
    previous = 0.0
    called = 0
    ordered_results = sorted(buffer.items(), key=lambda item: item[1][1])
    for index in range(
        freq_log,
        min(len(buffer), max_oracle_calls),
        freq_log,
    ):
        current = sorted(
            ordered_results[:index],
            key=lambda item: item[1][0],
            reverse=True,
        )[:top_n]
        top_n_now = float(np.mean([item[1][0] for item in current]))
        area += freq_log * (top_n_now + previous) / 2
        previous = top_n_now
        called = index
    current = sorted(
        ordered_results,
        key=lambda item: item[1][0],
        reverse=True,
    )[:top_n]
    top_n_now = float(np.mean([item[1][0] for item in current]))
    area += (len(buffer) - called) * (top_n_now + previous) / 2
    if finish and len(buffer) < max_oracle_calls:
        area += (max_oracle_calls - len(buffer)) * top_n_now
    return area / max_oracle_calls


def summarize_buffer(buffer, max_oracle_calls, freq_log):
    if not buffer:
        return {
            "calls": 0,
            "avg_top1": 0.0,
            "avg_top10": 0.0,
            "avg_top100": 0.0,
            "auc_top1": 0.0,
            "auc_top10": 0.0,
            "auc_top100": 0.0,
        }
    ordered_by_time = dict(
        sorted(buffer.items(), key=lambda item: item[1][1])
    )
    top = sorted(
        buffer.items(),
        key=lambda item: item[1][0],
        reverse=True,
    )[:100]
    scores = [item[1][0] for item in top]
    return {
        "calls": len(buffer),
        "avg_top1": float(np.max(scores)),
        "avg_top10": float(np.mean(sorted(scores, reverse=True)[:10])),
        "avg_top100": float(np.mean(scores)),
        "auc_top1": float(
            top_auc(ordered_by_time, 1, True, freq_log, max_oracle_calls)
        ),
        "auc_top10": float(
            top_auc(ordered_by_time, 10, True, freq_log, max_oracle_calls)
        ),
        "auc_top100": float(
            top_auc(ordered_by_time, 100, True, freq_log, max_oracle_calls)
        ),
    }
