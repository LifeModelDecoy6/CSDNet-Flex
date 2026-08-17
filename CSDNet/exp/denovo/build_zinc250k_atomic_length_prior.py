#!/usr/bin/env python
"""Build a ZINC250K sequence-length prior in CSDNet atomic token space."""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import pickle
from collections import Counter
from pathlib import Path

import numpy as np
from rdkit import Chem

from CSDNet.util.length_prior import ATOMIC_LENGTH_PRIOR_SCHEMA
from CSDNet.util.tokenizer import SMILESTokenizer, tokenize_smiles


def _iter_smiles(path, smiles_col):
    suffix = path.suffix.lower()
    if suffix in {".csv", ".tsv"}:
        delimiter = "\t" if suffix == ".tsv" else ","
        with path.open(newline="") as handle:
            reader = csv.DictReader(handle, delimiter=delimiter)
            if not reader.fieldnames:
                raise SystemExit(f"No header found in {path}")
            lookup = {name.strip().lower(): name for name in reader.fieldnames}
            if smiles_col == "auto":
                key = next(
                    (lookup[name] for name in ("smiles", "canonical_smiles", "text") if name in lookup),
                    None,
                )
            else:
                key = smiles_col if smiles_col in reader.fieldnames else lookup.get(smiles_col.lower())
            if key is None:
                raise SystemExit(
                    f"Could not identify a SMILES column in {path}; columns={reader.fieldnames}"
                )
            for row in reader:
                value = (row.get(key) or "").strip()
                if value:
                    yield value
        return

    with path.open() as handle:
        for line in handle:
            value = line.strip().split()[0] if line.strip() else ""
            if value:
                yield value


def _sha256(path):
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--input", required=True)
    parser.add_argument("--output", required=True)
    parser.add_argument("--vocab", default="csdnet_vocab.pkl")
    parser.add_argument("--smiles_col", default="auto")
    parser.add_argument("--max_len", type=int, default=128)
    parser.add_argument("--no_canonicalize", action="store_true")
    parser.add_argument(
        "--keep_stereochemistry",
        action="store_true",
        help="Keep stereochemical SMILES tokens; off by default to match the CSDNet vocabulary.",
    )
    args = parser.parse_args()

    input_path = Path(args.input)
    output_path = Path(args.output)
    vocab_path = Path(args.vocab)
    if not input_path.is_file():
        raise SystemExit(f"Input not found: {input_path}")
    if not vocab_path.is_file():
        raise SystemExit(f"Vocabulary not found: {vocab_path}")

    with vocab_path.open("rb") as handle:
        tk = SMILESTokenizer(pickle.load(handle))

    lengths = []
    counters = Counter()
    for raw_smiles in _iter_smiles(input_path, args.smiles_col):
        counters["rows"] += 1
        mol = Chem.MolFromSmiles(raw_smiles)
        if mol is None:
            counters["invalid"] += 1
            continue
        if not args.keep_stereochemistry:
            Chem.RemoveStereochemistry(mol)
        smiles = Chem.MolToSmiles(
            mol,
            canonical=not args.no_canonicalize,
            isomericSmiles=args.keep_stereochemistry,
        )
        tokens = tokenize_smiles(smiles)
        if "".join(tokens) != smiles:
            counters["tokenization_mismatch"] += 1
            continue
        if tk.unk_id != -1 and any(token not in tk.vocab for token in tokens):
            counters["unknown_token"] += 1
            continue
        length = len(tokens) + 2
        if length > args.max_len:
            counters["too_long"] += 1
            continue
        lengths.append(length)

    if not lengths:
        raise SystemExit("No usable molecules remained after validation")

    histogram = Counter(lengths)
    payload = {
        "schema": ATOMIC_LENGTH_PRIOR_SCHEMA,
        "dataset": "ZINC250K",
        "source_file": input_path.name,
        "source_sha256": _sha256(input_path),
        "vocab_file": vocab_path.name,
        "vocab_sha256": _sha256(vocab_path),
        "tokenizer": "csdnet_atomic_smiles",
        "canonical_smiles": not args.no_canonicalize,
        "stereochemistry_removed": not args.keep_stereochemistry,
        "include_special_tokens": True,
        "max_len": args.max_len,
        "source_rows": counters["rows"],
        "accepted_rows": len(lengths),
        "rejections": {
            key: counters[key]
            for key in ("invalid", "tokenization_mismatch", "unknown_token", "too_long")
        },
        "summary": {
            "minimum": min(lengths),
            "maximum": max(lengths),
            "mean": float(np.mean(lengths)),
            "median": float(np.median(lengths)),
        },
        "histogram": {str(key): value for key, value in sorted(histogram.items())},
        "lengths": lengths,
    }
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(json.dumps(payload, indent=2, sort_keys=True))

    print(f"Saved: {output_path}")
    print(f"Source rows: {counters['rows']}")
    print(f"Accepted: {len(lengths)}")
    print(f"Rejected: {payload['rejections']}")
    print(
        "Atomic sequence length (BOS/EOS included): "
        f"min={min(lengths)}, mean={np.mean(lengths):.2f}, "
        f"median={np.median(lengths):.1f}, max={max(lengths)}"
    )


if __name__ == "__main__":
    main()
