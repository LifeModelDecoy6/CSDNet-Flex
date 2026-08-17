#!/usr/bin/env python
"""Diagnose and progressively rework invalid fixed-budget de novo samples."""

from __future__ import annotations

import argparse
import csv
import json
import pickle
import random
from collections import Counter
from pathlib import Path

import numpy as np
import torch
from rdkit import Chem, RDLogger

from CSDNet.util.checkpoint import load_backbone_from_checkpoint
from CSDNet.util.elastic_sampling import _repair_final_sequences
from CSDNet.util.fsm import ValenceFSMTracker
from CSDNet.util.metrics import classify_invalid
from CSDNet.util.tokenizer import SMILESTokenizer


RDLogger.DisableLog("rdApp.*")


def _trim_padding(token_ids, pad_id):
    return [token_id for token_id in token_ids if token_id != pad_id]


def _is_valid(smiles):
    if not smiles:
        return False
    try:
        return Chem.MolFromSmiles(smiles) is not None
    except Exception:
        return False


def _fsm_diagnostics(smiles, tokenizer, tracker, max_len):
    try:
        sequence = _trim_padding(
            tokenizer.encode(smiles, max_len=max_len),
            tokenizer.pad_id,
        )
    except Exception:
        return None, [], []
    penalties = tracker.compute_penalties(
        torch.tensor([sequence], dtype=torch.long)
    )[0]
    positions = penalties.lt(0).nonzero(as_tuple=False).flatten().tolist()
    tokens = [tokenizer.inv[sequence[position]] for position in positions]
    return sequence, positions, tokens


def _load_smiles(path):
    # Empty lines are retained as failed proposals if they ever occur.
    return path.read_text().splitlines()


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--input", required=True)
    parser.add_argument("--ckpt_path", required=True)
    parser.add_argument("--vocab", default="csdnet_vocab.pkl")
    parser.add_argument("--output_dir", required=True)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--max_len", type=int, default=256)
    parser.add_argument("--batch_size", type=int, default=128)
    parser.add_argument("--max_sample_retries", type=int, default=3)
    parser.add_argument("--progressive_steps", type=int, default=8)
    parser.add_argument("--violation_neighborhood", type=int, default=2)
    parser.add_argument("--temperature", type=float, default=1.0)
    parser.add_argument("--top_k", type=int, default=0)
    parser.add_argument("--top_p", type=float, default=0.5)
    return parser.parse_args()


def main():
    args = parse_args()
    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(args.seed)

    input_path = Path(args.input)
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    smiles = _load_smiles(input_path)

    with open(args.vocab, "rb") as handle:
        tokenizer = SMILESTokenizer(pickle.load(handle))
    tracker = ValenceFSMTracker(tokenizer)

    invalid_indices = [
        index for index, value in enumerate(smiles) if not _is_valid(value)
    ]
    initial_reasons = Counter(
        classify_invalid(smiles[index]) if smiles[index] else "empty"
        for index in invalid_indices
    )
    rows = []
    encoded = []
    encoded_indices = []
    for index in invalid_indices:
        value = smiles[index]
        sequence, positions, tokens = _fsm_diagnostics(
            value,
            tokenizer,
            tracker,
            args.max_len,
        )
        row = {
            "index": index,
            "original_smiles": value,
            "initial_reason": classify_invalid(value) if value else "empty",
            "encoded": sequence is not None,
            "sequence_tokens": len(sequence) if sequence is not None else 0,
            "fsm_violation_count": len(positions),
            "fsm_violation_positions": " ".join(map(str, positions)),
            "fsm_violation_tokens": " ".join(tokens),
            "fsm_detected": bool(positions),
        }
        rows.append(row)
        if sequence is not None:
            encoded.append(sequence)
            encoded_indices.append(len(rows) - 1)

    device = "cuda" if torch.cuda.is_available() else "cpu"
    model = None
    if encoded:
        model = load_backbone_from_checkpoint(
            args.ckpt_path,
            tokenizer,
            device=device,
        ).to(device)
        model.eval()

    reworked = list(smiles)
    repair_totals = Counter()
    for start in range(0, len(encoded), args.batch_size):
        batch = encoded[start : start + args.batch_size]
        repaired, repair_diagnostics = _repair_final_sequences(
            model=model,
            tk=tokenizer,
            sequences=batch,
            device=device,
            use_fsm_check=True,
            use_rdkit_kekulize_check=True,
            max_sample_retries=args.max_sample_retries,
            violation_neighborhood=args.violation_neighborhood,
            temperature=args.temperature,
            top_k=args.top_k,
            top_p=args.top_p,
            progressive_steps=args.progressive_steps,
            prefer_fsm_localization=True,
            return_diagnostics=True,
        )
        repair_totals.update(repair_diagnostics)
        for local_index, sequence in enumerate(repaired):
            row_index = encoded_indices[start + local_index]
            source_index = rows[row_index]["index"]
            if tokenizer.eos_id in sequence:
                sequence = sequence[: sequence.index(tokenizer.eos_id) + 1]
            candidate = tokenizer.decode(sequence).strip("'\"")
            recovered = _is_valid(candidate)
            rows[row_index].update(
                {
                    "reworked_smiles": candidate,
                    "recovered": recovered,
                    "final_reason": "" if recovered else (
                        classify_invalid(candidate) if candidate else "empty"
                    ),
                }
            )
            if recovered:
                reworked[source_index] = candidate

    for row in rows:
        if "recovered" not in row:
            row.update(
                {
                    "reworked_smiles": "",
                    "recovered": False,
                    "final_reason": "unencodable",
                }
            )

    recovered_rows = [row for row in rows if row["recovered"]]
    residual_reasons = Counter(
        row["final_reason"] for row in rows if not row["recovered"]
    )
    summary = {
        "input": str(input_path),
        "total_proposals": len(smiles),
        "initial_invalid": len(rows),
        "initial_validity": (
            (len(smiles) - len(rows)) / len(smiles) if smiles else 0.0
        ),
        "initial_reason_counts": dict(initial_reasons),
        "fsm_detected_invalid": sum(row["fsm_detected"] for row in rows),
        "fsm_detection_fraction": (
            sum(row["fsm_detected"] for row in rows) / len(rows) if rows else 0.0
        ),
        "recovered": len(recovered_rows),
        "recovery_fraction": len(recovered_rows) / len(rows) if rows else 0.0,
        "final_invalid": len(rows) - len(recovered_rows),
        "final_validity": (
            (len(smiles) - len(rows) + len(recovered_rows)) / len(smiles)
            if smiles
            else 0.0
        ),
        "residual_reason_counts": dict(residual_reasons),
        "max_sample_retries": args.max_sample_retries,
        "progressive_steps": args.progressive_steps,
        "violation_neighborhood": args.violation_neighborhood,
        "temperature": args.temperature,
        "top_k": args.top_k,
        "top_p": args.top_p,
        "repair_diagnostics": dict(repair_totals),
    }

    (output_dir / "generated_mols_reworked.txt").write_text(
        "\n".join(reworked) + "\n"
    )
    with (output_dir / "invalid_rework_attempts.csv").open("w", newline="") as handle:
        fieldnames = list(rows[0]) if rows else ["index", "original_smiles"]
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)
    (output_dir / "invalid_rework_summary.json").write_text(
        json.dumps(summary, indent=2, sort_keys=True) + "\n"
    )

    print(json.dumps(summary, indent=2, sort_keys=True))
    print(f"Saved: {output_dir / 'generated_mols_reworked.txt'}")
    print(f"Saved: {output_dir / 'invalid_rework_attempts.csv'}")
    print(f"Saved: {output_dir / 'invalid_rework_summary.json'}")


if __name__ == "__main__":
    main()
