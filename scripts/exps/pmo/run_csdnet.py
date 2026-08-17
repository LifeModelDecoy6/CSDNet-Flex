#!/usr/bin/env python
import argparse
import os
import sys
from pathlib import Path
from time import time

import yaml
from tdc import Oracle

REPO_ROOT = Path(__file__).resolve().parents[3]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from scripts.exps.pmo.main.csdnet.run import CSDNetOptimizer, PMO_TASKS, load_csdnet_model


def completed(output_dir, mode, oracle_name, seed, max_oracle_calls):
    path = os.path.join(output_dir, f"results_CSDNet_{mode}_{oracle_name}_{seed}.yaml")
    if not os.path.exists(path):
        return False
    try:
        with open(path) as f:
            data = yaml.safe_load(f) or {}
    except Exception:
        return False
    return len(data) >= max_oracle_calls


def clear_incomplete_outputs(output_dir, mode, oracle_name, seed):
    paths = [
        os.path.join(output_dir, f"{oracle_name}_{seed}.csv"),
        os.path.join(output_dir, f"results_CSDNet_{mode}_{oracle_name}_{seed}.yaml"),
    ]
    for path in paths:
        if os.path.exists(path):
            os.remove(path)


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--mode", choices=["motif_seeded", "iterative_remask"], required=True)
    parser.add_argument("-o", "--oracle", default="all",
                        choices=["all", *PMO_TASKS])
    parser.add_argument("--output_dir", type=str, default=None)
    parser.add_argument("--resume", action="store_true",
                        help="Skip oracle tasks whose YAML result already has max_oracle_calls entries.")
    parser.add_argument("--max_oracle_calls", type=int, default=10000)
    parser.add_argument("--freq_log", type=int, default=100)
    parser.add_argument("-s", "--seed", type=int, default=1)
    parser.add_argument("--n_jobs", type=int, default=-1)

    parser.add_argument("--ckpt_path", type=str, required=True)
    parser.add_argument("--vocab", type=str, default="csdnet_vocab.pkl")
    parser.add_argument("--data_dir", type=str, default="csdnet_data/pubchem_10m_with_props_v2")
    parser.add_argument("--ref_sample_n", type=int, default=50000)
    parser.add_argument("--cpu", action="store_true")

    parser.add_argument("--max_len", type=int, default=128)
    parser.add_argument("--batch_size", type=int, default=64)
    parser.add_argument("--candidate_batch_size", type=int, default=128)
    parser.add_argument("--n_steps", type=int, default=250)
    parser.add_argument("--population_size", type=int, default=100)
    parser.add_argument("--elite_size", type=int, default=100)
    parser.add_argument("--elite_seed_prob", type=float, default=0.5)

    parser.add_argument("--motif_min_atoms", type=int, default=4)
    parser.add_argument("--motif_max_atoms", type=int, default=36)
    parser.add_argument("--remask_fraction", type=float, default=0.35)
    parser.add_argument("--min_remask_tokens", type=int, default=2)
    parser.add_argument("--span_prob", type=float, default=0.7)

    parser.add_argument("--disable_fsm_check", action="store_true")
    parser.add_argument("--disable_rdkit_kekulize_check", action="store_true")
    parser.add_argument("--rdkit_check_interval", type=int, default=25)
    parser.add_argument("--max_sample_retries", type=int, default=2)
    parser.add_argument("--violation_neighborhood", type=int, default=2)
    parser.add_argument("--temperature_start", type=float, default=1.2)
    parser.add_argument("--temperature_end", type=float, default=0.2)
    parser.add_argument("--temperature_power", type=float, default=1.5)
    return parser.parse_args()


def main():
    start = time()
    args = parse_args()
    if args.output_dir is None:
        args.output_dir = os.path.join(
            "scripts",
            "exps",
            "pmo",
            "main",
            "csdnet",
            "results",
            args.mode,
        )
    os.makedirs(args.output_dir, exist_ok=True)

    tasks = PMO_TASKS if args.oracle == "all" else [args.oracle]
    print(f"CSDNet PMO mode: {args.mode}")
    print(f"Tasks: {len(tasks)} ({', '.join(tasks)})")
    print(f"Output dir: {args.output_dir}")
    print(f"Max oracle calls per task: {args.max_oracle_calls}")

    model_bundle = load_csdnet_model(args)
    for oracle_name in tasks:
        if args.resume and completed(
            args.output_dir,
            args.mode,
            oracle_name,
            args.seed,
            args.max_oracle_calls,
        ):
            print(f"[skip] {oracle_name}: completed result found.")
            continue
        clear_incomplete_outputs(args.output_dir, args.mode, oracle_name, args.seed)

        print("=" * 80)
        print(f"Optimizing PMO oracle: {oracle_name}")
        args.oracle = oracle_name
        oracle = Oracle(name=oracle_name)
        optimizer = CSDNetOptimizer(args=args, model_bundle=model_bundle)
        optimizer.optimize(oracle=oracle, config={}, seed=args.seed)

    hours = (time() - start) / 3600.0
    print(f"CSDNet PMO {args.mode} finished in {hours:.2f} hours")


if __name__ == "__main__":
    main()
