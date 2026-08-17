"""Tokenization-aware sequence-length priors for de novo sampling."""

from __future__ import annotations

import json
from collections import Counter
from pathlib import Path


ATOMIC_LENGTH_PRIOR_SCHEMA = "csdnet_atomic_smiles_length_prior_v1"


def load_atomic_length_prior(path, max_len=None):
    """Load and validate a CSDNet atomic-SMILES length prior."""
    prior_path = Path(path)
    payload = json.loads(prior_path.read_text())
    if not isinstance(payload, dict):
        raise ValueError(f"Length prior must be a JSON object: {prior_path}")
    if payload.get("schema") != ATOMIC_LENGTH_PRIOR_SCHEMA:
        raise ValueError(
            "Length prior has an incompatible or missing schema. "
            "GenMol data/len.pk contains SAFE-token lengths and cannot be "
            "used directly by the CSDNet atomic tokenizer. Build the prior "
            "with CSDNet.exp.denovo.build_zinc250k_atomic_length_prior."
        )
    if payload.get("tokenizer") != "csdnet_atomic_smiles":
        raise ValueError("Length prior was not built with the CSDNet atomic tokenizer")
    if payload.get("include_special_tokens") is not True:
        raise ValueError("Length prior must include BOS and EOS tokens")

    lengths = [int(value) for value in payload.get("lengths", [])]
    if not lengths or min(lengths) < 3:
        raise ValueError(f"Length prior contains no valid sequence lengths: {prior_path}")
    if max_len is not None and max(lengths) > int(max_len):
        raise ValueError(
            f"Length prior maximum {max(lengths)} exceeds evaluator max_len={max_len}; "
            "rebuild the prior with the same max_len instead of clipping it at runtime"
        )

    metadata = {key: value for key, value in payload.items() if key != "lengths"}
    metadata["path"] = str(prior_path)
    metadata["count"] = len(lengths)
    metadata["minimum"] = min(lengths)
    metadata["maximum"] = max(lengths)
    metadata["histogram"] = {
        str(key): value for key, value in sorted(Counter(lengths).items())
    }
    return lengths, metadata
