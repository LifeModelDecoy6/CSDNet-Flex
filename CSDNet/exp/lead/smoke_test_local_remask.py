#!/usr/bin/env python
import argparse
import csv
import os
import random
from pathlib import Path

import numpy as np
import torch
from rdkit import Chem, DataStructs
from rdkit.Chem import AllChem

from CSDNet.exp.pmo.optimizer import (
    canonical_smiles,
    load_csdnet_model,
    sample_csdnet_local_remask,
)


PROFILES = {
    "micro": {"remask": 0.06, "temperature": 0.92, "span_prob": 0.70},
    "small": {"remask": 0.12, "temperature": 0.98, "span_prob": 0.76},
    "medium": {"remask": 0.22, "temperature": 1.10, "span_prob": 0.84},
    "legacy": {"remask": 0.35, "temperature": 1.20, "span_prob": 0.70},
}


def tanimoto(smiles_a, smiles_b):
    mol_a = Chem.MolFromSmiles(smiles_a)
    mol_b = Chem.MolFromSmiles(smiles_b)
    if mol_a is None or mol_b is None:
        return 0.0
    fp_a = AllChem.GetMorganFingerprintAsBitVect(mol_a, 2, 2048)
    fp_b = AllChem.GetMorganFingerprintAsBitVect(mol_b, 2, 2048)
    return float(DataStructs.TanimotoSimilarity(fp_a, fp_b))


def write_csv(path, rows, fieldnames):
    os.makedirs(os.path.dirname(path), exist_ok=True)
    with open(path, "w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--ckpt_path", required=True)
    parser.add_argument("--vocab", default="csdnet_vocab.pkl")
    parser.add_argument("--output_dir", default="CSDNet/exp/lead/smoke_results")
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--repeats", type=int, default=2)
    parser.add_argument("--batch_size", type=int, default=64)
    parser.add_argument("--n_steps", type=int, default=220)
    parser.add_argument("--max_len", type=int, default=256)
    parser.add_argument("--cpu", action="store_true")
    parser.add_argument("--disable_fsm_check", action="store_true")
    parser.add_argument("--disable_rdkit_kekulize_check", action="store_true")
    args = parser.parse_args()

    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    model, tokenizer, device = load_csdnet_model(args)

    active_path = Path(__file__).resolve().parent / "docking" / "actives.csv"
    with active_path.open() as handle:
        base_seeds = [canonical_smiles(row["smiles"]) for row in csv.DictReader(handle)]
    base_seeds = [smiles for smiles in base_seeds if smiles is not None]
    seeds = base_seeds * max(1, args.repeats)

    detail_rows = []
    summary_rows = []
    for profile_idx, (profile_name, profile) in enumerate(PROFILES.items()):
        profile_seed = args.seed + profile_idx * 1009
        random.seed(profile_seed)
        np.random.seed(profile_seed)
        torch.manual_seed(profile_seed)
        outputs = sample_csdnet_local_remask(
            model=model,
            tk=tokenizer,
            seed_smiles=seeds,
            max_len=args.max_len,
            device=device,
            batch_size=args.batch_size,
            n_steps=args.n_steps,
            remask_fraction=profile["remask"],
            min_remask_tokens=1,
            span_prob=profile["span_prob"],
            use_fsm_check=not args.disable_fsm_check,
            use_rdkit_kekulize_check=not args.disable_rdkit_kekulize_check,
            rdkit_check_interval=25,
            max_sample_retries=2,
            violation_neighborhood=2,
            temperature_start=profile["temperature"],
            temperature_end=0.18,
            temperature_power=1.6,
            length_delta_choices="0",
            length_edit_prob=0.0,
            return_seed_indices=True,
        )
        by_index = {source_idx: smiles for smiles, source_idx in outputs}
        similarities = []
        valid_outputs = []
        identity_count = 0
        for source_idx, parent in enumerate(seeds):
            child = by_index.get(source_idx)
            similarity = tanimoto(parent, child) if child else 0.0
            if child:
                valid_outputs.append(child)
                similarities.append(similarity)
                identity_count += int(child == parent)
            detail_rows.append(
                {
                    "profile": profile_name,
                    "source_idx": source_idx,
                    "parent_smiles": parent,
                    "child_smiles": child or "",
                    "valid": int(child is not None),
                    "identity": int(child == parent) if child else 0,
                    "similarity": similarity,
                }
            )

        requested = len(seeds)
        output_count = len(valid_outputs)
        summary = {
            "profile": profile_name,
            "requested": requested,
            "outputs": output_count,
            "output_rate": output_count / requested if requested else 0.0,
            "unique_outputs": len(set(valid_outputs)),
            "uniqueness": (
                len(set(valid_outputs)) / output_count if output_count else 0.0
            ),
            "identity_rate": identity_count / output_count if output_count else 0.0,
            "mean_similarity": float(np.mean(similarities)) if similarities else 0.0,
        }
        summary_rows.append(summary)
        print(
            f"{profile_name}: outputs={output_count}/{requested} "
            f"rate={summary['output_rate']:.3f} unique={summary['uniqueness']:.3f} "
            f"identity={summary['identity_rate']:.3f} "
            f"mean_sim={summary['mean_similarity']:.3f}"
        )

    write_csv(
        os.path.join(args.output_dir, "local_remask_details.csv"),
        detail_rows,
        [
            "profile",
            "source_idx",
            "parent_smiles",
            "child_smiles",
            "valid",
            "identity",
            "similarity",
        ],
    )
    write_csv(
        os.path.join(args.output_dir, "local_remask_summary.csv"),
        summary_rows,
        [
            "profile",
            "requested",
            "outputs",
            "output_rate",
            "unique_outputs",
            "uniqueness",
            "identity_rate",
            "mean_similarity",
        ],
    )
    print(f"Saved smoke-test results to {args.output_dir}")


if __name__ == "__main__":
    main()
