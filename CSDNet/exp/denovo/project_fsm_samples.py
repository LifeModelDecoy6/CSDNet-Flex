#!/usr/bin/env python
"""Replay the deterministic FSM projection on a fixed proposal file."""

from __future__ import annotations

import argparse
import csv
import json
import pickle
from collections import Counter
from pathlib import Path

from rdkit import Chem, RDLogger

from CSDNet.util.fsm import ValenceFSMTracker
from CSDNet.util.metrics import classify_invalid
from CSDNet.util.tokenizer import SMILESTokenizer


RDLogger.DisableLog("rdApp.*")


def _is_valid(smiles):
    if not smiles:
        return False
    try:
        return Chem.MolFromSmiles(smiles) is not None
    except Exception:
        return False


def _trim_padding(sequence, pad_id):
    return [token_id for token_id in sequence if token_id != pad_id]


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--input", required=True)
    parser.add_argument("--output_dir", required=True)
    parser.add_argument("--vocab", default="csdnet_vocab.pkl")
    parser.add_argument("--max_len", type=int, default=256)
    parser.add_argument(
        "--fail_on_valid_change",
        action=argparse.BooleanOptionalAction,
        default=True,
    )
    return parser.parse_args()


def main():
    args = parse_args()
    input_path = Path(args.input)
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    with open(args.vocab, "rb") as handle:
        tokenizer = SMILESTokenizer(pickle.load(handle))
    tracker = ValenceFSMTracker(tokenizer).syntax_tracker

    inputs = input_path.read_text().splitlines()
    outputs = []
    rows = []
    totals = Counter()
    projection_totals = Counter()

    for index, original in enumerate(inputs):
        original_valid = _is_valid(original)
        totals["input_valid"] += int(original_valid)
        totals["input_invalid"] += int(not original_valid)
        sequence = _trim_padding(
            tokenizer.encode(original, max_len=args.max_len),
            tokenizer.pad_id,
        )
        projected, _, _, diagnostics = tracker.project_completed_sequence(
            sequence
        )
        projected_smiles = tokenizer.decode(projected).strip("'\"")
        changed = projected_smiles != original

        if original_valid:
            if changed and args.fail_on_valid_change:
                raise RuntimeError(
                    "FSM projection changed a valid proposal at row "
                    f"{index}: {original!r} -> {projected_smiles!r}"
                )
            accepted = original
        else:
            accepted = projected_smiles if _is_valid(projected_smiles) else original

        final_valid = _is_valid(accepted)
        recovered = not original_valid and final_valid
        totals["changed"] += int(changed)
        totals["recovered"] += int(recovered)
        totals["final_valid"] += int(final_valid)
        totals["final_invalid"] += int(not final_valid)
        projection_totals.update(diagnostics)
        outputs.append(accepted)
        rows.append(
            {
                "index": index,
                "original_valid": original_valid,
                "changed": changed,
                "recovered": recovered,
                "final_valid": final_valid,
                "initial_reason": "" if original_valid else classify_invalid(original),
                "final_reason": "" if final_valid else classify_invalid(accepted),
                "original_smiles": original,
                "projected_smiles": projected_smiles,
                "accepted_smiles": accepted,
            }
        )

    total = len(inputs)
    summary = {
        "input": str(input_path),
        "total_proposals": total,
        **dict(totals),
        "input_validity": totals["input_valid"] / total if total else 0.0,
        "final_validity": totals["final_valid"] / total if total else 0.0,
        "valid_rows_changed": sum(
            row["original_valid"] and row["changed"] for row in rows
        ),
        "projection_diagnostics": dict(projection_totals),
    }

    output_path = output_dir / "generated_mols_fsm_projected.txt"
    output_path.write_text("\n".join(outputs) + "\n")
    with (output_dir / "fsm_projection_rows.csv").open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]) if rows else [])
        if rows:
            writer.writeheader()
            writer.writerows(rows)
    (output_dir / "fsm_projection_summary.json").write_text(
        json.dumps(summary, indent=2, sort_keys=True) + "\n"
    )

    print(json.dumps(summary, indent=2, sort_keys=True))
    print(f"Saved: {output_path}")
    print(f"Saved: {output_dir / 'fsm_projection_rows.csv'}")
    print(f"Saved: {output_dir / 'fsm_projection_summary.json'}")


if __name__ == "__main__":
    main()
