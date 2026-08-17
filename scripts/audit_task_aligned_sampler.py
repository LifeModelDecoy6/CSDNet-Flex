#!/usr/bin/env python
"""Fail-fast audit for the final task-aligned Fragment/Lead/PMO evaluation."""

from __future__ import annotations

import argparse
import hashlib
import inspect
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

import torch

from CSDNet.exp.denovo.sampler_profiles import SAMPLER_PROFILES
from CSDNet.exp.frag.direct_infill import load_length_prior as load_fragment_length_prior
from CSDNet.exp.pmo.optimizer import (
    progressive_global_sampler_kwargs,
    resolve_local_sampler_profile,
)
from CSDNet.util.checkpoint import infer_backbone_config
from CSDNet.util.length_prior import load_atomic_length_prior
from CSDNet.util.sampling import sample_csdnet

def require_text(path: Path, fragments: tuple[str, ...]) -> None:
    if not path.is_file():
        raise SystemExit(f"Missing source file: {path}")
    text = path.read_text()
    missing = [fragment for fragment in fragments if fragment not in text]
    if missing:
        raise SystemExit(f"{path} is missing required integration markers: {missing}")


def profile_digest(name: str) -> str:
    payload = json.dumps(
        SAMPLER_PROFILES[name],
        sort_keys=True,
        separators=(",", ":"),
    )
    return hashlib.sha256(payload.encode()).hexdigest()


def load_checkpoint(path: Path):
    kwargs = {"map_location": "cpu", "weights_only": False}
    try:
        return torch.load(path, mmap=True, **kwargs)
    except TypeError:
        return torch.load(path, **kwargs)


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--checkpoint", required=True)
    parser.add_argument("--atomic_length_prior", required=True)
    parser.add_argument("--vocab", default="csdnet_vocab.pkl")
    parser.add_argument("--max_len", type=int, default=128)
    args = parser.parse_args()

    checkpoint_path = Path(args.checkpoint)
    prior_path = Path(args.atomic_length_prior)
    vocab_path = Path(args.vocab)
    if not checkpoint_path.is_file() or checkpoint_path.stat().st_size == 0:
        raise SystemExit(f"Missing or empty checkpoint: {checkpoint_path}")
    if not vocab_path.is_file() or vocab_path.stat().st_size == 0:
        raise SystemExit(f"Missing or empty tokenizer vocabulary: {vocab_path}")

    local = SAMPLER_PROFILES["promax_task_adaptive_local"]
    refined = SAMPLER_PROFILES["promax_task_adaptive_refine"]
    fragment = SAMPLER_PROFILES["promax_fragment_conditional_refine"]
    global_profile = SAMPLER_PROFILES["promax_progressive_length_coupled"]

    assert resolve_local_sampler_profile("task_local") == "task_adaptive_local"
    assert resolve_local_sampler_profile("task_refine") == "task_adaptive_refine"
    assert local["local_confidence_uses_editable_length"] is True
    assert local["local_temperature_mode"] == "operator_scaled"
    assert local["all_position_refine_steps"] == 0
    assert refined["all_position_refine_steps"] > 0
    assert refined["all_position_max_total_edits"] > 0
    assert refined["all_position_verify_masked"] is True
    assert refined["all_position_rdkit_each_step"] is True
    assert fragment["local_confidence_uses_editable_length"] is True
    assert fragment["all_position_refine_steps"] > 0
    assert global_profile["confidence_length_adaptive"] is True
    assert global_profile["all_position_refine_steps"] == 0

    global_kwargs = progressive_global_sampler_kwargs("task_adaptive_local")
    accepted_global_kwargs = set(inspect.signature(sample_csdnet).parameters)
    unsupported = sorted(set(global_kwargs) - accepted_global_kwargs)
    if unsupported:
        raise SystemExit(
            "PMO global restart leaks local-only sampler arguments: "
            f"{unsupported}"
        )
    for required_key in (
        "progressive_commit",
        "confidence_length_adaptive",
        "temperature_start",
        "temperature_end",
        "remask_power",
    ):
        if required_key not in global_kwargs:
            raise SystemExit(
                f"PMO global restart is missing trajectory control: {required_key}"
            )

    lengths, prior_metadata = load_atomic_length_prior(
        prior_path,
        max_len=args.max_len,
    )
    if len(lengths) < 100:
        raise SystemExit(
            f"Atomic length prior is unexpectedly small: {len(lengths)} entries"
        )
    expected_prior_metadata = {
        "dataset": "ZINC250K",
        "tokenizer": "csdnet_atomic_smiles",
        "canonical_smiles": True,
        "stereochemistry_removed": True,
        "include_special_tokens": True,
        "max_len": args.max_len,
    }
    mismatches = {
        key: (prior_metadata.get(key), expected)
        for key, expected in expected_prior_metadata.items()
        if prior_metadata.get(key) != expected
    }
    if mismatches:
        raise SystemExit(f"Atomic length prior metadata mismatch: {mismatches}")
    if prior_metadata.get("accepted_rows") != len(lengths):
        raise SystemExit(
            "Atomic length prior count does not match accepted_rows: "
            f"{len(lengths)} != {prior_metadata.get('accepted_rows')}"
        )
    if len(lengths) < 200_000:
        raise SystemExit(
            "Atomic length prior does not look like the validated ZINC250K prior: "
            f"accepted_rows={len(lengths)}"
        )
    if prior_metadata.get("vocab_sha256") != sha256_file(vocab_path):
        raise SystemExit(
            "Atomic length prior was built with a different tokenizer vocabulary"
        )

    fragment_lengths = load_fragment_length_prior(
        prior_path,
        max_len=args.max_len,
    )
    expected_fragment_lengths = [length - 2 for length in lengths]
    if fragment_lengths != expected_fragment_lengths:
        raise SystemExit(
            "Fragment length conversion is not exactly atomic sequence length - 2 "
            "for BOS/EOS"
        )

    checkpoint = load_checkpoint(checkpoint_path)
    state_dict = checkpoint.get("state_dict", checkpoint)
    config = infer_backbone_config(state_dict, checkpoint)
    if config["architecture_type"] != "csdnet":
        raise SystemExit(
            "Final task-aligned evaluation requires the ordinary CSDNet "
            f"checkpoint, got {config['architecture_type']}"
        )
    if not config["corruption_level_conditioning"]:
        raise SystemExit(
            "Checkpoint lacks corruption-level conditioning required for "
            "conditional refinement"
        )
    if config["max_position_embeddings"] < args.max_len:
        raise SystemExit(
            "Checkpoint position limit is smaller than the requested max_len: "
            f"{config['max_position_embeddings']} < {args.max_len}"
        )
    global_step = int(checkpoint.get("global_step", -1))

    require_text(
        ROOT / "CSDNet/exp/frag/run_direct_infill_v2.py",
        (
            '"conditional_progressive_refine"',
            '"empirical_atomic"',
            'getattr(model, "corruption_level_conditioning", False)',
        ),
    )
    require_text(
        ROOT / "CSDNet/exp/denovo/run_90m_promax_3seed_1gpu.pbs",
        (
            'LENGTH_PRIOR="${LENGTH_PRIOR:?LENGTH_PRIOR must be provided}"',
            'LENGTH_PRIOR="$LENGTH_PRIOR"',
        ),
    )
    require_text(
        ROOT / "CSDNet/exp/denovo/run_90m_normcbi_promax_seed_1gpu.pbs",
        (
            '--length_prior_path "$LENGTH_PRIOR"',
            "--sampler_profile \"$PROFILE\"",
        ),
    )
    require_text(
        ROOT / "CSDNet/exp/frag/run_fragment_conditional_refine_v7_task_1gpu.pbs",
        (
            '--length_prior "$LENGTH_PRIOR"',
            "--length_allocation_policy empirical_atomic",
            "--local_sampler_profile conditional_progressive_refine",
        ),
    )
    require_text(
        ROOT / "CSDNet/exp/lead/run.py",
        (
            "frontier_local_sampler_profile",
            'return "task_adaptive_refine"',
            'return "task_adaptive_local"',
            "local_sampler_profile=self.frontier_local_sampler_profile(",
        ),
    )
    require_text(
        ROOT / "CSDNet/exp/lead/run_lead_bounded15_6chunk_1gpu.pbs",
        (
            "export NUM_ITER=15",
            "export BUDGET_COMPLETION=0",
            "export BUDGET_COMPLETION_UNTIL_BUDGET=0",
            "export CSDNET_LOCAL_SAMPLER_PROFILE=",
        ),
    )
    require_text(
        ROOT / "CSDNet/exp/pmo/optimizer.py",
        (
            'weights["elite_refine"]',
            'operator == "elite_refine"',
            "atomic_length_prior=getattr",
            "_v9_task_reference_lengths",
            "ref_lengths=self._v9_task_reference_lengths(",
            "local_sampler_profile=root_local_profile",
        ),
    )
    require_text(
        ROOT / "CSDNet/exp/pmo/run_iterative_remask_v9_10k_task_1gpu.pbs",
        ('--atomic_length_prior "$ATOMIC_LENGTH_PRIOR"',),
    )
    require_text(
        ROOT / "CSDNet/exp/pmo/run_iterative_remask_v9_10k_task_3seed_1gpu.pbs",
        ('ATOMIC_LENGTH_PRIOR="$ATOMIC_LENGTH_PRIOR"',),
    )

    del state_dict
    del checkpoint
    print("Task-aligned sampler audit: PASS")
    print(
        "Checkpoint: "
        f"step={global_step} architecture={config['architecture_type']} "
        f"H={config['hidden_size']} L={config['num_layers']} "
        f"max_positions={config['max_position_embeddings']} refinement=True"
    )
    print(
        "Atomic length prior: "
        f"n={len(lengths)} range={prior_metadata['minimum']}-"
        f"{prior_metadata['maximum']} tokenizer=atomic vocab_sha256=verified"
    )
    print(
        "PMO global restart: "
        f"{len(global_kwargs)} accepted trajectory arguments; "
        "no local-only arguments"
    )
    for name in (
        "promax_progressive_length_coupled",
        "promax_fragment_conditional_refine",
        "promax_task_adaptive_local",
        "promax_task_adaptive_refine",
    ):
        print(f"Profile {name}: sha256={profile_digest(name)}")


if __name__ == "__main__":
    main()
