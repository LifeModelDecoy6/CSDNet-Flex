#!/usr/bin/env python
"""Run fair direct infilling on the five fragment-constrained tasks."""

from __future__ import annotations

import argparse
import csv
import hashlib
import random
from pathlib import Path

import numpy as np
import pandas as pd
import torch
from rdkit import RDLogger

from CSDNet.exp.frag.direct_infill import build_masked_template, load_length_prior
from CSDNet.exp.frag.run_linker_design import evaluate_samples
from CSDNet.exp.frag.task_head import (
    CANONICAL_TASKS,
    assess_candidate,
    build_constraint_spec,
    normalize_task,
)
from CSDNet.exp.pmo.optimizer import (
    load_csdnet_model,
    sample_csdnet_local_remask,
)
from CSDNet.util.tokenizer import tokenize_smiles


RDLogger.DisableLog("rdApp.*")


def stable_case_seed(seed: int, task: str, name: str) -> int:
    random_task = "linker_design" if task == "scaffold_morphing" else task
    payload = f"direct-infill-v1:{seed}:{random_task}:{name}".encode()
    return int.from_bytes(hashlib.sha256(payload).digest()[:4], "big")


def _write_rows(path: Path, rows):
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    pd.DataFrame(rows).to_csv(temporary, index=False)
    temporary.replace(path)


def _read_rows(path: Path):
    if not path.exists() or path.stat().st_size == 0:
        return []
    try:
        return pd.read_csv(path).to_dict("records")
    except pd.errors.EmptyDataError:
        return []


def _validate_fixed_tokens(template, tokenizer):
    tokens = tokenize_smiles(template.seed_smiles)
    masked = set()
    for plan in template.edit_plans:
        masked.update(range(int(plan["start"]), int(plan["stop"])))
    unknown = [
        token
        for index, token in enumerate(tokens)
        if index not in masked and token not in tokenizer.vocab
    ]
    if unknown:
        raise ValueError(f"Frozen template contains unknown tokens: {unknown}")


def run_case(args, model_bundle, row, length_prior):
    spec = build_constraint_spec(args.task, row)
    case_seed = stable_case_seed(args.seed, spec.task, spec.name)
    random.seed(case_seed)
    np.random.seed(case_seed)
    torch.manual_seed(case_seed)
    rng = random.Random(case_seed)
    model, tokenizer, device = model_bundle

    templates = [
        build_masked_template(
            spec,
            max_len=args.max_len,
            length_prior=length_prior,
            min_added_tokens=args.min_added_tokens,
            rng=rng,
        )
        for _ in range(args.num_samples)
    ]
    for template in templates:
        _validate_fixed_tokens(template, tokenizer)

    outputs = sample_csdnet_local_remask(
        model=model,
        tk=tokenizer,
        seed_smiles=[template.seed_smiles for template in templates],
        max_len=args.max_len,
        device=device,
        batch_size=args.batch_size,
        n_steps=args.n_steps,
        use_fsm_check=not args.disable_fsm_check,
        use_rdkit_kekulize_check=not args.disable_rdkit_kekulize_check,
        rdkit_check_interval=args.rdkit_check_interval,
        max_sample_retries=args.max_sample_retries,
        violation_neighborhood=args.violation_neighborhood,
        temperature_start=args.temperature_start,
        temperature_end=args.temperature_end,
        temperature_power=args.temperature_power,
        edit_plans=[list(template.edit_plans) for template in templates],
        return_seed_indices=True,
    )
    outputs_by_attempt = {int(index): smiles for smiles, index in outputs}
    accepted = []
    attempts = []
    for index, template in enumerate(templates):
        assessment = assess_candidate(outputs_by_attempt.get(index), spec)
        if assessment.structural_success:
            accepted.append(assessment.smiles)
        attempts.append(
            {
                "task": spec.task,
                "name": spec.name,
                "seed": args.seed,
                "case_seed": case_seed,
                "attempt": index,
                "target_length": template.target_length,
                "added_tokens": template.added_tokens,
                "attachment_count": template.attachment_count,
                "model_output": assessment.valid,
                "smiles": assessment.smiles,
                "connected": assessment.connected,
                "no_dummies": assessment.no_dummies,
                "preserved": assessment.preserved,
                "required": assessment.required,
                "structural_success": assessment.structural_success,
            }
        )
    return spec, accepted, attempts


def run_task(args):
    from tdc import Evaluator, Oracle

    args.task = normalize_task(args.task)
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    data = pd.read_csv(args.fragments_csv)
    length_prior = load_length_prior(args.length_prior)
    model_bundle = load_csdnet_model(args)
    oracle_qed = Oracle("qed")
    oracle_sa = Oracle("sa")
    diversity_evaluator = Evaluator("diversity")

    stem = f"{args.task}_seed{args.seed}"
    metrics_path = output_dir / f"metrics_{stem}.csv"
    samples_path = output_dir / f"samples_{stem}.csv"
    attempts_path = output_dir / f"attempts_{stem}.csv"
    summary_path = output_dir / f"summary_{stem}.csv"
    metric_rows = _read_rows(metrics_path) if args.resume else []
    sample_rows = _read_rows(samples_path) if args.resume else []
    attempt_rows = _read_rows(attempts_path) if args.resume else []
    completed = {str(record["name"]) for record in metric_rows}

    for _, row in data.iterrows():
        name = str(row["name"])
        if name in completed:
            print(f"[{args.task}] {name}: already complete, skipping")
            continue
        print("=" * 72)
        print(f"[{args.task}] {name}: direct attempts={args.num_samples}")
        spec, accepted, attempts = run_case(args, model_bundle, row, length_prior)
        metrics, unique_records = evaluate_samples(
            spec.original,
            accepted,
            args.num_samples,
            oracle_qed,
            oracle_sa,
            diversity_evaluator,
        )
        metrics.update(
            {
                "task": spec.task,
                "name": spec.name,
                "seed": args.seed,
                "method": "direct_infill_v1",
                "original": spec.original,
                "fragment": spec.fragment,
                "geometry": spec.geometry,
                "raw_attempts": args.num_samples,
                "model_outputs": sum(item["model_output"] for item in attempts),
                "structural_successes": len(accepted),
                "mean_added_tokens": float(
                    np.mean([item["added_tokens"] for item in attempts])
                ),
            }
        )
        metric_rows.append(metrics)
        attempt_rows.extend(attempts)
        sample_rows.extend(
            {
                "task": spec.task,
                "name": spec.name,
                "seed": args.seed,
                **record,
            }
            for record in unique_records
        )
        print(
            f"[{args.task}] {name}: validity={metrics['validity']:.3f} "
            f"uniqueness={metrics['uniqueness']:.3f} "
            f"quality={metrics['quality']:.3f} "
            f"outputs={metrics['model_outputs']}/{args.num_samples}"
        )
        _write_rows(metrics_path, metric_rows)
        _write_rows(samples_path, sample_rows)
        _write_rows(attempts_path, attempt_rows)

    frame = pd.DataFrame(metric_rows)
    summary = {
        "task": args.task,
        "seed": args.seed,
        "method": "direct_infill_v1",
        "n_cases": len(frame),
    }
    for metric in (
        "validity",
        "uniqueness",
        "quality",
        "diversity",
        "distance",
        "mean_qed",
        "mean_sa",
    ):
        summary[f"{metric}_mean"] = float(frame[metric].mean()) if len(frame) else 0.0
    with summary_path.open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(summary))
        writer.writeheader()
        writer.writerow(summary)
    print(pd.DataFrame([summary]).to_string(index=False))
    print(f"Saved: {metrics_path}")
    print(f"Saved: {samples_path}")
    print(f"Saved: {attempts_path}")
    print(f"Saved: {summary_path}")


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--task", required=True, choices=(*CANONICAL_TASKS, "superstructure_design")
    )
    parser.add_argument("--ckpt_path", required=True)
    parser.add_argument("--vocab", default="csdnet_vocab.pkl")
    parser.add_argument("--fragments_csv", default="data/fragments.csv")
    parser.add_argument("--length_prior", default="data/len.pk")
    parser.add_argument(
        "--output_dir",
        default="CSDNet/exp/frag/results/direct_infill_v1",
    )
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--cpu", action="store_true")
    parser.add_argument("--resume", action="store_true")
    parser.add_argument("--num_samples", type=int, default=100)
    parser.add_argument("--min_added_tokens", type=int, default=4)
    parser.add_argument("--max_len", type=int, default=128)
    parser.add_argument("--batch_size", type=int, default=64)
    parser.add_argument("--n_steps", type=int, default=160)
    parser.add_argument("--temperature_start", type=float, default=1.20)
    parser.add_argument("--temperature_end", type=float, default=0.18)
    parser.add_argument("--temperature_power", type=float, default=1.6)
    parser.add_argument("--disable_fsm_check", action="store_true")
    parser.add_argument("--disable_rdkit_kekulize_check", action="store_true")
    parser.add_argument("--rdkit_check_interval", type=int, default=25)
    parser.add_argument("--max_sample_retries", type=int, default=2)
    parser.add_argument("--violation_neighborhood", type=int, default=2)
    return parser.parse_args()


def main():
    args = parse_args()
    for path in (args.fragments_csv, args.length_prior, args.vocab, args.ckpt_path):
        if not Path(path).exists():
            raise SystemExit(f"Cannot find required file: {path}")
    run_task(args)


if __name__ == "__main__":
    main()
