#!/usr/bin/env python
import argparse
import csv
import os
import random
from pathlib import Path

import numpy as np
import pandas as pd
import torch
from rdkit import Chem, RDLogger

from CSDNet.exp.frag.run_linker_design import (
    build_query,
    canonical_smiles,
    evaluate_samples,
    generate_linker_samples,
)
from CSDNet.exp.pmo.optimizer import (
    load_csdnet_model,
    sample_csdnet_local_remask,
    tokenizable,
)


RDLogger.DisableLog("rdApp.*")


TASKS = [
    "linker_design_onestep",
    "motif_extension",
    "scaffold_decoration",
    "superstructure_generation",
]


def task_column(task):
    if task == "linker_design_onestep":
        return "linker_design"
    return task


def contains_query(smiles, query):
    mol = Chem.MolFromSmiles(smiles)
    if mol is None or query is None:
        return False
    return mol.HasSubstructMatch(query)


def generate_completion_samples(args, model_bundle, original, fragment):
    model, tk, device = model_bundle
    query = build_query(fragment)
    original = canonical_smiles(original)
    if original is None or query is None:
        return []

    out = []
    rounds = 0
    while len(out) < args.num_samples and rounds < args.max_rounds:
        rounds += 1
        request_n = min(args.candidate_batch_size, args.num_samples - len(out))
        seed_batch = [original for _ in range(request_n)]
        candidates = sample_csdnet_local_remask(
            model=model,
            tk=tk,
            seed_smiles=seed_batch,
            max_len=args.max_len,
            device=device,
            batch_size=args.batch_size,
            n_steps=args.n_steps,
            remask_fraction=args.remask_fraction,
            min_remask_tokens=args.min_remask_tokens,
            span_prob=args.span_prob,
            use_fsm_check=not args.disable_fsm_check,
            use_rdkit_kekulize_check=not args.disable_rdkit_kekulize_check,
            rdkit_check_interval=args.rdkit_check_interval,
            max_sample_retries=args.max_sample_retries,
            violation_neighborhood=args.violation_neighborhood,
            temperature_start=args.temperature_start,
            temperature_end=args.temperature_end,
            temperature_power=args.temperature_power,
        )
        for smi in candidates:
            can = canonical_smiles(smi)
            if can is None:
                continue
            if not tokenizable(can, tk, args.max_len):
                continue
            if not contains_query(can, query):
                continue
            out.append(can)
            if len(out) >= args.num_samples:
                break
    return out


def configure_strategy(args, strategy):
    args = argparse.Namespace(**vars(args))
    if strategy == "wide_refine":
        args.remask_fraction = args.wide_remask_fraction
        args.temperature_start = args.wide_temperature_start
        args.span_prob = args.wide_span_prob
    elif strategy != "local_refine":
        raise ValueError(f"Unsupported strategy: {strategy}")
    return args


def run_one_strategy(args, strategy, tasks):
    args = configure_strategy(args, strategy)
    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)

    from tdc import Evaluator, Oracle

    os.makedirs(args.output_dir, exist_ok=True)
    model_bundle = load_csdnet_model(args)
    oracle_qed = Oracle("qed")
    oracle_sa = Oracle("sa")
    diversity_evaluator = Evaluator("diversity")
    data = pd.read_csv(args.fragments_csv)

    all_metrics = []
    all_samples = []

    for task in tasks:
        col = task_column(task)
        task_metrics = []
        print("=" * 72)
        print(f"[{strategy}] Task: {task} (column={col})")
        for _, row in data.iterrows():
            name = row["name"]
            original = row["smiles"]
            fragment = row[col]
            print(f"[{strategy}:{task}] {name}: generating {args.num_samples} samples")
            try:
                if task == "linker_design_onestep":
                    samples = generate_linker_samples(args, model_bundle, fragment)
                else:
                    samples = generate_completion_samples(args, model_bundle, original, fragment)
            except Exception as exc:
                print(
                    f"[{strategy}:{task}] {name}: failed with "
                    f"{type(exc).__name__}: {exc}"
                )
                samples = []

            metrics, unique_records = evaluate_samples(
                original,
                samples,
                args.num_samples,
                oracle_qed,
                oracle_sa,
                diversity_evaluator,
            )
            metrics.update(
                {
                    "strategy": strategy,
                    "task": task,
                    "name": name,
                    "original": original,
                    "fragment": fragment,
                }
            )
            task_metrics.append(metrics)
            all_metrics.append(metrics)

            for rec in unique_records:
                all_samples.append(
                    {
                        "strategy": strategy,
                        "task": task,
                        "name": name,
                        "smiles": rec["smiles"],
                        "qed": rec["qed"],
                        "sa": rec["sa"],
                    }
                )

            print(
                f"[{strategy}:{task}] {name}: validity={metrics['validity']:.3f} "
                f"uniqueness={metrics['uniqueness']:.3f} "
                f"quality={metrics['quality']:.3f}"
            )

        write_task_outputs(args.output_dir, strategy, task, task_metrics)

    write_all_outputs(args.output_dir, strategy, all_metrics, all_samples)


def metric_columns():
    return [
        "validity",
        "uniqueness",
        "diversity",
        "distance",
        "quality",
        "mean_qed",
        "mean_sa",
    ]


def summarize(rows, strategy, task=None):
    df = pd.DataFrame(rows)
    out = {
        "strategy": strategy,
        "task": task or "all",
        "n_cases": len(df),
    }
    for col in metric_columns():
        out[f"{col}_mean"] = float(df[col].mean()) if len(df) else 0.0
    return out


def write_task_outputs(output_dir, strategy, task, rows):
    metrics_path = os.path.join(output_dir, f"{task}_metrics_{strategy}.csv")
    summary_path = os.path.join(output_dir, f"{task}_summary_{strategy}.csv")
    pd.DataFrame(rows).to_csv(metrics_path, index=False)
    summary = summarize(rows, strategy=strategy, task=task)
    with open(summary_path, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=list(summary.keys()))
        writer.writeheader()
        writer.writerow(summary)
    print(f"Saved: {metrics_path}")
    print(f"Saved: {summary_path}")


def write_all_outputs(output_dir, strategy, metrics_rows, sample_rows):
    metrics_path = os.path.join(output_dir, f"fragment_remaining_metrics_{strategy}.csv")
    samples_path = os.path.join(output_dir, f"fragment_remaining_samples_{strategy}.csv")
    summary_path = os.path.join(output_dir, f"fragment_remaining_summary_{strategy}.csv")
    by_task_path = os.path.join(output_dir, f"fragment_remaining_by_task_{strategy}.csv")

    metrics_df = pd.DataFrame(metrics_rows)
    metrics_df.to_csv(metrics_path, index=False)
    pd.DataFrame(sample_rows).to_csv(samples_path, index=False)

    summary = summarize(metrics_rows, strategy=strategy)
    with open(summary_path, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=list(summary.keys()))
        writer.writeheader()
        writer.writerow(summary)

    grouped = [
        summarize(group.to_dict("records"), strategy=strategy, task=task)
        for task, group in metrics_df.groupby("task")
    ]
    pd.DataFrame(grouped).to_csv(by_task_path, index=False)

    print("=" * 72)
    print(f"Strategy: {strategy}")
    for col in ["validity", "uniqueness", "diversity", "distance", "quality"]:
        print(f"{col}: {summary[f'{col}_mean']:.4f}")
    print(f"Saved: {metrics_path}")
    print(f"Saved: {samples_path}")
    print(f"Saved: {summary_path}")
    print(f"Saved: {by_task_path}")


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--tasks",
        default=",".join(TASKS),
        help="Comma-separated tasks from: " + ",".join(TASKS),
    )
    parser.add_argument("--strategy", choices=["local_refine", "wide_refine", "both"], default="both")
    parser.add_argument("--ckpt_path", type=str, required=True)
    parser.add_argument("--vocab", type=str, default="csdnet_vocab.pkl")
    parser.add_argument("--fragments_csv", type=str, default="data/fragments.csv")
    parser.add_argument(
        "--output_dir",
        type=str,
        default=os.path.join("CSDNet", "exp", "frag", "results", "fragment_remaining"),
    )
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--cpu", action="store_true")

    parser.add_argument("--num_samples", type=int, default=100)
    parser.add_argument("--candidate_batch_size", type=int, default=128)
    parser.add_argument("--max_rounds", type=int, default=12)
    parser.add_argument("--max_len", type=int, default=128)
    parser.add_argument("--batch_size", type=int, default=64)
    parser.add_argument("--n_steps", type=int, default=160)
    parser.add_argument("--remask_fraction", type=float, default=0.22)
    parser.add_argument("--min_remask_tokens", type=int, default=2)
    parser.add_argument("--span_prob", type=float, default=0.78)
    parser.add_argument("--temperature_start", type=float, default=1.10)
    parser.add_argument("--temperature_end", type=float, default=0.18)
    parser.add_argument("--temperature_power", type=float, default=1.6)

    parser.add_argument("--wide_remask_fraction", type=float, default=0.36)
    parser.add_argument("--wide_temperature_start", type=float, default=1.30)
    parser.add_argument("--wide_span_prob", type=float, default=0.85)

    parser.add_argument("--disable_fsm_check", action="store_true")
    parser.add_argument("--disable_rdkit_kekulize_check", action="store_true")
    parser.add_argument("--rdkit_check_interval", type=int, default=25)
    parser.add_argument("--max_sample_retries", type=int, default=2)
    parser.add_argument("--violation_neighborhood", type=int, default=2)
    return parser.parse_args()


def main():
    args = parse_args()
    if not Path(args.fragments_csv).exists():
        raise SystemExit(f"Cannot find fragments CSV: {args.fragments_csv}")
    tasks = [task.strip() for task in args.tasks.split(",") if task.strip()]
    bad = sorted(set(tasks) - set(TASKS))
    if bad:
        raise SystemExit(f"Unsupported tasks: {bad}. Supported: {TASKS}")
    strategies = ["local_refine", "wide_refine"] if args.strategy == "both" else [args.strategy]
    for strategy in strategies:
        run_one_strategy(args, strategy, tasks)


if __name__ == "__main__":
    main()
