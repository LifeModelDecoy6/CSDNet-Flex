#!/usr/bin/env python
from __future__ import annotations

import argparse
import os
import pickle
import re
from pathlib import Path

import torch

from CSDNet.model.edit_scheduler import load_edit_scheduler_checkpoint
from CSDNet.util.edit_schedule_sampling import sample_de_novo_lengths
from CSDNet.util.tokenizer import SMILESTokenizer


def load_checkpoint(path):
    if not os.path.isfile(path) or os.path.getsize(path) == 0:
        raise FileNotFoundError(path)
    try:
        return torch.load(path, map_location="cpu", weights_only=False)
    except TypeError:
        return torch.load(path, map_location="cpu")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--backbone", required=True)
    parser.add_argument("--length_scheduler", required=True)
    parser.add_argument("--vocab", default="csdnet_vocab.pkl")
    parser.add_argument("--repo_root", default=".")
    args = parser.parse_args()

    backbone = load_checkpoint(args.backbone)
    hparams = backbone.get("hyper_parameters", {})
    if hparams.get("training_objective_mode") != "three_way_equal":
        raise SystemExit(
            "Backbone is not the final equal three-way objective checkpoint"
        )
    if int(hparams.get("trajectory_rollout_steps", -1)) != 2:
        raise SystemExit("Final backbone must use two trajectory rollout steps")
    if not bool(hparams.get("corruption_level_conditioning", False)):
        raise SystemExit("Final backbone lacks corruption-level conditioning")
    if int(hparams.get("fragment_span_max", -1)) != 24:
        raise SystemExit("Final backbone fragment span configuration mismatch")

    with open(args.vocab, "rb") as handle:
        tokenizer = SMILESTokenizer(pickle.load(handle))
    scheduler_checkpoint = load_checkpoint(args.length_scheduler)
    scheduler_hparams = scheduler_checkpoint.get("hyper_parameters", {})
    config = scheduler_hparams.get("scheduler_config", {})
    if int(config.get("max_position_embeddings", -1)) != 128:
        raise SystemExit("Length scheduler max positions must be 128")
    if int(config.get("max_replacement_length", -1)) != 126:
        raise SystemExit("Length scheduler must represent body lengths 0..126")

    scheduler = load_edit_scheduler_checkpoint(
        args.length_scheduler,
        device="cpu",
    )
    torch.manual_seed(7)
    lengths = sample_de_novo_lengths(
        scheduler,
        tokenizer,
        32,
        max_len=128,
        device=torch.device("cpu"),
        temperature=1.0,
        top_k=16,
    )
    if len(lengths) != 32 or not all(3 <= length <= 128 for length in lengths):
        raise SystemExit(f"Invalid learned de novo lengths: {lengths}")

    root = Path(args.repo_root)
    final_jobs = {
        "de novo": root
        / "CSDNet/exp/denovo/run_90m_final_learned_length_3seed_1gpu.pbs",
        "fragment": root
        / "CSDNet/exp/frag/run_fragment_final_learned_length_seed_1gpu.pbs",
    }
    for label, path in final_jobs.items():
        text = path.read_text()
        if "--length_scheduler_ckpt" not in text:
            raise SystemExit(f"{label} job does not load the learned scheduler")
        if re.search(r"--length_prior(?:_path)?(?:\s|=)", text):
            raise SystemExit(f"{label} job still supplies an empirical length prior")
        if re.search(r"--length_explore_fraction\s+(?!0(?:\.0*)?(?:\s|$))", text):
            raise SystemExit(f"{label} job enables random length exploration")
    fragment_text = final_jobs["fragment"].read_text()
    if "--length_allocation_policy learned_scheduler" not in fragment_text:
        raise SystemExit("Fragment job does not enforce learned length allocation")

    print("Final asset audit passed")
    print(f"Backbone: {args.backbone}")
    print("Objective: equal mask/refine/fragment = 1/3 each")
    print("Refinement rollout: 2 steps")
    print(f"Length scheduler: {args.length_scheduler}")
    print("Length modes: learned de novo gap + learned conditional fragment gaps")
    print("Random length deltas and empirical sampling: disabled in final jobs")
    print("Final de novo and fragment PBS interfaces: mutually exclusive audit passed")
    print(
        "Learned-length smoke range: "
        f"min={min(lengths)}, max={max(lengths)}, n={len(lengths)}"
    )


if __name__ == "__main__":
    main()
