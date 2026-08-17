#!/usr/bin/env python3
"""Fail-fast audit for the recommended LoFlex-aligned 6M CSDNet."""

import argparse
import math
import pickle
import sys
from pathlib import Path

import torch

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from CSDNet.model.elastic_lightning_module import (  # noqa: E402
    ElasticCSDNetLightningModule,
)
from CSDNet.model.elastic_schedule import ElasticKumaSchedule  # noqa: E402
from csdnet_tokenizer import SMILESTokenizer, tokenize_smiles  # noqa: E402


EXPECTED = {
    "hidden_size": 256,
    "num_layers": 8,
    "num_heads": 8,
    "intermediate": 1024,
    "max_position_embeddings": 256,
    "position_embedding_type": "rotary",
    "rate_parameterization": "exp",
    "theta_rate_min": 0.0,
    "phi_rate_min": 1.01,
    "rate_output_bias": -4.0,
    "fixed_unmask_rate": 1.0,
    "kuma_shape_a": 1.0,
    "loflex_objective": True,
    "policy_time_conditioning": True,
    "fragment_corruption_prob": 0.15,
    "fragment_internal_probability": 0.25,
    "fragment_terminal_probability": 0.25,
    "fragment_dual_probability": 0.25,
    "fragment_multi_probability": 0.25,
    "mdm_corruption_prob": 0.10,
    "refine_corruption_prob": 0.05,
    "length_loss_normalizer": 256.0,
}


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--vocab", default="csdnet_vocab.pkl")
    parser.add_argument("--checkpoint", default="")
    parser.add_argument(
        "--expected_global_step",
        type=int,
        default=-1,
        help="Fail unless the checkpoint has this exact global step.",
    )
    return parser.parse_args()


def equal(actual, expected):
    if isinstance(expected, float):
        return actual is not None and math.isclose(
            float(actual),
            expected,
            rel_tol=0.0,
            abs_tol=1e-8,
        )
    return actual == expected


def load_tokenizer(path):
    with Path(path).open("rb") as handle:
        return SMILESTokenizer(pickle.load(handle))


def dry_module(tokenizer):
    return ElasticCSDNetLightningModule(
        vocab_size=tokenizer.vocab_size,
        pad_id=tokenizer.pad_id,
        mask_id=tokenizer.mask_id,
        bos_id=tokenizer.bos_id,
        eos_id=tokenizer.eos_id,
        unk_id=tokenizer.unk_id,
        scaffold_ids=tokenizer.scaffold_ids,
        aromatic_ids=tokenizer.aromatic_ids,
        cond_dim=0,
        hidden_size=EXPECTED["hidden_size"],
        num_layers=EXPECTED["num_layers"],
        num_heads=EXPECTED["num_heads"],
        intermediate=EXPECTED["intermediate"],
        max_position_embeddings=EXPECTED["max_position_embeddings"],
        position_embedding_type=EXPECTED["position_embedding_type"],
        rate_parameterization=EXPECTED["rate_parameterization"],
        theta_rate_min=EXPECTED["theta_rate_min"],
        phi_rate_min=EXPECTED["phi_rate_min"],
        rate_output_bias=EXPECTED["rate_output_bias"],
        fixed_unmask_rate=EXPECTED["fixed_unmask_rate"],
        kuma_shape_a=EXPECTED["kuma_shape_a"],
        loflex_objective=EXPECTED["loflex_objective"],
        policy_time_conditioning=EXPECTED["policy_time_conditioning"],
        fragment_corruption_prob=EXPECTED["fragment_corruption_prob"],
        fragment_internal_probability=EXPECTED[
            "fragment_internal_probability"
        ],
        fragment_terminal_probability=EXPECTED[
            "fragment_terminal_probability"
        ],
        fragment_dual_probability=EXPECTED["fragment_dual_probability"],
        fragment_multi_probability=EXPECTED["fragment_multi_probability"],
        mdm_corruption_prob=EXPECTED["mdm_corruption_prob"],
        refine_corruption_prob=EXPECTED["refine_corruption_prob"],
        length_loss_normalizer=EXPECTED["length_loss_normalizer"],
        gradient_checkpointing=False,
    )


def audit_atom_level_tokenizer(tokenizer, failures):
    probe = "CC(=O)Nc1ccccc1Cl"
    expected_tokens = [
        "C",
        "C",
        "(",
        "=",
        "O",
        ")",
        "N",
        "c",
        "1",
        "c",
        "c",
        "c",
        "c",
        "c",
        "1",
        "Cl",
    ]
    actual_tokens = tokenize_smiles(probe)
    encoded = tokenizer.encode(probe, max_len=64)
    roundtrip = tokenizer.decode(encoded)
    contains_unknown = tokenizer.unk_id in encoded
    atom_level_ok = (
        actual_tokens == expected_tokens
        and roundtrip == probe
        and not contains_unknown
    )
    status = "OK" if atom_level_ok else "FAIL"
    print(
        f"{status:4s} atom-level SMILES tokenization: "
        f"tokens={actual_tokens!r}, roundtrip={roundtrip!r}"
    )
    if not atom_level_ok:
        failures.append("atom_level_tokenization")


def audit_checkpoint_state(checkpoint, tokenizer, module, failures):
    state = checkpoint.get("state_dict", checkpoint)
    required = {
        "backbone.esm.esm.embeddings.word_embeddings.weight",
        "backbone.theta_insertion_head.out.weight",
        "backbone.phi_insertion_head.out.weight",
    }
    missing_required = sorted(required.difference(state))
    if missing_required:
        print(f"FAIL checkpoint tensors missing: {missing_required}")
        failures.append("checkpoint_tensors")
    else:
        print("OK   checkpoint contains atom-MDM embedding and theta/phi insertion heads")

    embedding = state.get(
        "backbone.esm.esm.embeddings.word_embeddings.weight"
    )
    embedding_ok = (
        embedding is not None
        and embedding.ndim == 2
        and embedding.shape[0] == tokenizer.vocab_size
        and embedding.shape[1] == EXPECTED["hidden_size"]
    )
    status = "OK" if embedding_ok else "FAIL"
    shape = tuple(embedding.shape) if embedding is not None else None
    print(
        f"{status:4s} checkpoint token embedding: "
        f"shape={shape}, expected=({tokenizer.vocab_size}, "
        f"{EXPECTED['hidden_size']})"
    )
    if not embedding_ok:
        failures.append("checkpoint_vocab_embedding")

    unmask_keys = sorted(key for key in state if "_unmask_head." in key)
    if unmask_keys:
        print(f"FAIL learned unmask heads found despite fixed-unmask mode: {unmask_keys[:4]}")
        failures.append("fixed_unmask_checkpoint")
    else:
        print("OK   checkpoint uses fixed unmask rate; no learned unmask heads")

    expected_ema = {
        "ema." + name.replace(".", "___")
        for name, parameter in module.backbone.named_parameters()
        if parameter.requires_grad
    }
    missing_ema = sorted(expected_ema.difference(state))
    if missing_ema:
        print(
            f"FAIL checkpoint EMA coverage: missing "
            f"{len(missing_ema)}/{len(expected_ema)} trainable tensors"
        )
        failures.append("checkpoint_ema")
    else:
        print(
            f"OK   checkpoint EMA coverage: "
            f"{len(expected_ema)}/{len(expected_ema)} trainable tensors"
        )


def main():
    args = parse_args()
    tokenizer = load_tokenizer(args.vocab)
    module = dry_module(tokenizer)
    parameters = sum(
        parameter.numel()
        for parameter in module.backbone.parameters()
        if parameter.requires_grad
    )
    print(f"Dry architecture parameters: {parameters:,} ({parameters / 1e6:.3f}M)")

    failures = []
    audit_atom_level_tokenizer(tokenizer, failures)
    if args.checkpoint:
        path = Path(args.checkpoint)
        if not path.is_file() or path.stat().st_size == 0:
            raise SystemExit(f"Missing or empty checkpoint: {path}")
        checkpoint = torch.load(path, map_location="cpu", weights_only=False)
        values = checkpoint.get("hyper_parameters", {})
        global_step = checkpoint.get("global_step", "unknown")
        print(f"Checkpoint global_step: {global_step}")
        audit_checkpoint_state(checkpoint, tokenizer, module, failures)
        if (
            args.expected_global_step >= 0
            and global_step != args.expected_global_step
        ):
            failures.append("global_step")
    else:
        values = dict(module.hparams)

    for name, expected in EXPECTED.items():
        actual = values.get(name)
        status = "OK" if equal(actual, expected) else "FAIL"
        print(f"{status:4s} {name}: actual={actual!r}, expected={expected!r}")
        if status == "FAIL":
            failures.append(name)

    schedule = ElasticKumaSchedule(shape_a=1.0, regularizer_mode="loflex")
    time = torch.tensor([0.1, 0.5, 0.9])
    uniform_error = (schedule.cdf(time, torch.ones_like(time)) - time).abs().max()
    print(f"Kuma(a=1,b=1) uniform-CDF max error: {uniform_error.item():.3e}")
    if uniform_error.item() > 1e-6:
        failures.append("uniform_schedule")

    theta_floor = float(module.backbone.theta_rate_min)
    phi_floor = float(module.backbone.phi_rate_min)
    if theta_floor != 0.0 or phi_floor < 1.01:
        failures.append("rate_support")

    if failures:
        raise SystemExit("AUDIT FAILED: " + ", ".join(sorted(set(failures))))
    print(
        "AUDIT PASSED: atom-level LoFlex-adapted 6M configuration is "
        "internally consistent."
    )


if __name__ == "__main__":
    main()
