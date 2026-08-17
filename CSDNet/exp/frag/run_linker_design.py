#!/usr/bin/env python
import argparse
import csv
import os
import random
from pathlib import Path

import numpy as np
import pandas as pd
import torch
from rdkit import Chem, DataStructs, RDLogger
from rdkit.Chem import AllChem

from CSDNet.exp.pmo.optimizer import (
    attach_fragments,
    canonical_smiles,
    load_csdnet_model,
    sample_csdnet_local_remask,
    tokenizable,
)


RDLogger.DisableLog("rdApp.*")


LINKER_TEMPLATES = [
    "[1*]C[1*]",
    "[1*]CC[1*]",
    "[1*]CCC[1*]",
    "[1*]CCCC[1*]",
    "[1*]CO[1*]",
    "[1*]OC[1*]",
    "[1*]CCO[1*]",
    "[1*]OCC[1*]",
    "[1*]CNC[1*]",
    "[1*]NCC[1*]",
    "[1*]CCN[1*]",
    "[1*]OCCO[1*]",
    "[1*]NCCN[1*]",
    "[1*]C=C[1*]",
    "[1*]C#C[1*]",
]


def normalize_dummy_atoms(smiles):
    mol = Chem.MolFromSmiles(smiles)
    if mol is None:
        return None
    for atom in mol.GetAtoms():
        if atom.GetAtomicNum() == 0:
            atom.SetIsotope(1)
    return Chem.MolToSmiles(mol, canonical=True)


def remove_dummy_atoms(smiles):
    mol = Chem.MolFromSmiles(smiles)
    if mol is None:
        return None
    try:
        cleaned = Chem.DeleteSubstructs(mol, Chem.MolFromSmiles("*"))
        if cleaned is not None and cleaned.GetNumHeavyAtoms() > 0:
            return Chem.MolToSmiles(cleaned, canonical=True)
    except Exception:
        pass

    rw = Chem.RWMol(mol)
    dummy = [atom.GetIdx() for atom in rw.GetAtoms() if atom.GetAtomicNum() == 0]
    for idx in sorted(dummy, reverse=True):
        rw.RemoveAtom(idx)
    mol = rw.GetMol()
    try:
        Chem.SanitizeMol(mol)
    except Exception:
        return None
    if mol.GetNumHeavyAtoms() == 0:
        return None
    return Chem.MolToSmiles(mol, canonical=True)


def build_query(fragment):
    mol = Chem.MolFromSmiles(fragment)
    if mol is None:
        return None
    try:
        query = Chem.DeleteSubstructs(mol, Chem.MolFromSmiles("*"))
        if query is not None and query.GetNumHeavyAtoms() > 0:
            return query
    except Exception:
        pass
    clean = remove_dummy_atoms(fragment)
    if clean is None:
        return None
    mol = Chem.MolFromSmiles(clean)
    if mol is None:
        return None
    return mol


def parse_fragment_pair(fragment_pair):
    parts = str(fragment_pair).split(".")
    if len(parts) != 2:
        raise ValueError(f"linker_design fragment should contain two fragments: {fragment_pair}")
    left, right = parts
    left_norm = normalize_dummy_atoms(left)
    right_norm = normalize_dummy_atoms(right)
    if left_norm is None or right_norm is None:
        raise ValueError(f"Could not parse fragment pair: {fragment_pair}")
    left_query = build_query(left)
    right_query = build_query(right)
    if left_query is None or right_query is None:
        raise ValueError(f"Could not build substructure queries: {fragment_pair}")
    return left_norm, right_norm, left_query, right_query


def build_linked_seed(left_fragment, right_fragment, linker):
    first = attach_fragments(left_fragment, linker)
    if first is None:
        return None
    second = attach_fragments(first, right_fragment)
    return canonical_smiles(second) if second else None


def build_seed_pool(fragment_pair, extra_linkers=None):
    left, right, left_query, right_query = parse_fragment_pair(fragment_pair)
    linkers = list(LINKER_TEMPLATES)
    if extra_linkers:
        linkers.extend(extra_linkers)
    seeds = []
    seen = set()
    for linker in linkers:
        for a, b in [(left, right), (right, left)]:
            seed = build_linked_seed(a, b, linker)
            if seed is None or seed in seen:
                continue
            mol = Chem.MolFromSmiles(seed)
            if mol is None:
                continue
            if not mol.HasSubstructMatch(left_query) or not mol.HasSubstructMatch(right_query):
                continue
            seen.add(seed)
            seeds.append(seed)
    return seeds, left_query, right_query


def contains_both_fragments(smiles, left_query, right_query):
    mol = Chem.MolFromSmiles(smiles)
    if mol is None:
        return False
    return mol.HasSubstructMatch(left_query) and mol.HasSubstructMatch(right_query)


def morgan_fp(smiles):
    mol = Chem.MolFromSmiles(smiles)
    if mol is None:
        return None
    return AllChem.GetMorganFingerprintAsBitVect(mol, 2, 1024)


def mean_distance_to_original(original, smiles_list):
    fp0 = morgan_fp(original)
    fps = [fp for smi in smiles_list for fp in [morgan_fp(smi)] if fp is not None]
    if fp0 is None or not fps:
        return 0.0
    sims = DataStructs.BulkTanimotoSimilarity(fp0, fps)
    return float(np.mean([1.0 - sim for sim in sims]))


def evaluate_samples(original, samples, num_samples, oracle_qed, oracle_sa, diversity_evaluator):
    validity = len(samples) / num_samples if num_samples else 0.0
    if not samples:
        return {
            "validity": 0.0,
            "uniqueness": 0.0,
            "diversity": 0.0,
            "distance": 0.0,
            "quality": 0.0,
            "mean_qed": 0.0,
            "mean_sa": 0.0,
            "n_samples": 0,
            "n_unique": 0,
        }, []

    df = pd.DataFrame({"smiles": samples})
    df["qed"] = oracle_qed(df["smiles"].tolist())
    df["sa"] = oracle_sa(df["smiles"].tolist())
    unique_df = df.drop_duplicates("smiles").copy()
    unique_smiles = unique_df["smiles"].tolist()
    uniqueness = len(unique_df) / len(df) if len(df) else 0.0
    diversity = diversity_evaluator(unique_smiles) if len(unique_smiles) > 1 else 0.0
    distance = mean_distance_to_original(original, unique_smiles)
    quality_df = unique_df[(unique_df["qed"] >= 0.6) & (unique_df["sa"] <= 4.0)]
    quality = len(quality_df) / num_samples if num_samples else 0.0
    metrics = {
        "validity": float(validity),
        "uniqueness": float(uniqueness),
        "diversity": float(diversity),
        "distance": float(distance),
        "quality": float(quality),
        "mean_qed": float(unique_df["qed"].mean()) if len(unique_df) else 0.0,
        "mean_sa": float(unique_df["sa"].mean()) if len(unique_df) else 0.0,
        "n_samples": int(len(df)),
        "n_unique": int(len(unique_df)),
    }
    return metrics, unique_df.to_dict("records")


def generate_linker_samples(args, model_bundle, fragment_pair):
    model, tk, device = model_bundle
    seeds, left_query, right_query = build_seed_pool(fragment_pair)
    if not seeds:
        return []

    out = []
    rounds = 0
    max_rounds = args.max_rounds
    while len(out) < args.num_samples and rounds < max_rounds:
        rounds += 1
        request_n = min(args.candidate_batch_size, args.num_samples - len(out))
        seed_batch = [random.choice(seeds) for _ in range(request_n)]
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
            if not contains_both_fragments(can, left_query, right_query):
                continue
            out.append(can)
            if len(out) >= args.num_samples:
                break
    return out


def run_strategy(args, strategy):
    args = argparse.Namespace(**vars(args))
    if strategy == "wide_refine":
        args.remask_fraction = args.wide_remask_fraction
        args.temperature_start = args.wide_temperature_start
        args.span_prob = args.wide_span_prob
    elif strategy != "local_refine":
        raise ValueError(f"Unsupported strategy: {strategy}")

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
    metrics_rows = []
    sample_rows = []

    for _, row in data.iterrows():
        name = row["name"]
        original = row["smiles"]
        fragment_pair = row["linker_design"]
        print(f"[{strategy}] {name}: generating {args.num_samples} linker samples")
        try:
            samples = generate_linker_samples(args, model_bundle, fragment_pair)
        except Exception as exc:
            print(f"[{strategy}] {name}: failed to generate samples: {type(exc).__name__}: {exc}")
            samples = []

        metrics, unique_records = evaluate_samples(
            original,
            samples,
            args.num_samples,
            oracle_qed,
            oracle_sa,
            diversity_evaluator,
        )
        metrics.update({"strategy": strategy, "name": name, "original": original})
        metrics_rows.append(metrics)

        for rec in unique_records:
            sample_rows.append(
                {
                    "strategy": strategy,
                    "name": name,
                    "smiles": rec["smiles"],
                    "qed": rec["qed"],
                    "sa": rec["sa"],
                }
            )
        print(
            f"[{strategy}] {name}: validity={metrics['validity']:.3f} "
            f"uniqueness={metrics['uniqueness']:.3f} quality={metrics['quality']:.3f}"
        )

    metrics_path = os.path.join(args.output_dir, f"linker_design_metrics_{strategy}.csv")
    samples_path = os.path.join(args.output_dir, f"linker_design_samples_{strategy}.csv")
    pd.DataFrame(metrics_rows).to_csv(metrics_path, index=False)
    pd.DataFrame(sample_rows).to_csv(samples_path, index=False)

    mean = pd.DataFrame(metrics_rows)[
        ["validity", "uniqueness", "diversity", "distance", "quality", "mean_qed", "mean_sa"]
    ].mean()
    summary = {
        "strategy": strategy,
        "n_cases": len(metrics_rows),
        **{f"{key}_mean": float(val) for key, val in mean.items()},
    }
    summary_path = os.path.join(args.output_dir, f"linker_design_summary_{strategy}.csv")
    with open(summary_path, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=list(summary.keys()))
        writer.writeheader()
        writer.writerow(summary)

    print("=" * 60)
    print(f"Strategy: {strategy}")
    for key in ["validity", "uniqueness", "diversity", "distance", "quality"]:
        print(f"{key}: {summary[f'{key}_mean']:.4f}")
    print(f"Saved: {metrics_path}")
    print(f"Saved: {samples_path}")
    print(f"Saved: {summary_path}")


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--strategy", choices=["local_refine", "wide_refine", "both"], default="local_refine")
    parser.add_argument("--ckpt_path", type=str, required=True)
    parser.add_argument("--vocab", type=str, default="csdnet_vocab.pkl")
    parser.add_argument("--fragments_csv", type=str, default="data/fragments.csv")
    parser.add_argument("--output_dir", type=str, default=os.path.join("CSDNet", "exp", "frag", "results"))
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
    strategies = ["local_refine", "wide_refine"] if args.strategy == "both" else [args.strategy]
    for strategy in strategies:
        run_strategy(args, strategy)


if __name__ == "__main__":
    main()
