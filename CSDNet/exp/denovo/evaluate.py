#!/usr/bin/env python
import argparse
import json
import os
import pickle
import random

import numpy as np
import torch

from CSDNet.util.checkpoint import load_backbone_from_checkpoint
from CSDNet.util.conditioning import build_condition_from_args
from CSDNet.exp.denovo.sampler_profiles import (
    SAMPLER_PROFILES,
    add_block_refinement_arguments,
    add_confidence_planning_arguments,
    add_length_adaptive_arguments,
    apply_sampler_profile,
    block_refinement_kwargs,
    confidence_planning_kwargs,
    length_adaptive_kwargs,
)
from CSDNet.util.metrics import calculate_basic_metrics, write_generation_structure_summary
from CSDNet.util.length_prior import load_atomic_length_prior
from CSDNet.model.edit_scheduler import load_edit_scheduler_checkpoint
from CSDNet.util.reference import build_reference_from_disk, build_reference_from_hf_stream
from CSDNet.util.sampling import sample_csdnet
from CSDNet.util.tokenizer import SMILESTokenizer


def parse_args():
    parser = argparse.ArgumentParser(description="Evaluate CSDNet de novo generation.")
    parser.add_argument("--ckpt_path", required=True)
    parser.add_argument("--data_dir", default="csdnet_data/pubchem_10m_with_props_v2")
    parser.add_argument("--hf_dataset", default=None)
    parser.add_argument("--hf_split", default="train")
    parser.add_argument("--hf_smiles_col", default="auto")
    parser.add_argument("--hf_safe_col", default="safe")
    parser.add_argument("--disable_hf_safe_decode", action="store_true")
    parser.add_argument("--hf_streaming", action="store_true")
    parser.add_argument("--hf_token_env", default="HF_TOKEN")
    parser.add_argument("--hf_token", default=None)
    parser.add_argument("--hf_ref_sample_n", type=int, default=50000)
    parser.add_argument("--hf_novelty_sample_n", type=int, default=200000)
    parser.add_argument("--hf_scan_limit", type=int, default=1000000)
    parser.add_argument("--hf_skip_unknown", action="store_true", default=True)
    parser.add_argument("--hf_keep_unknown", action="store_false", dest="hf_skip_unknown")
    parser.add_argument("--hf_skip_long", action="store_true", default=True)
    parser.add_argument("--hf_keep_long", action="store_false", dest="hf_skip_long")
    parser.add_argument(
        "--length_prior_path",
        default=None,
        help="Tokenizer-aware JSON length prior; overrides reference-dataset lengths only.",
    )
    parser.add_argument(
        "--length_scheduler_ckpt",
        default=None,
        help=(
            "External ZINC-trained learned-length scheduler. When supplied, "
            "empirical/random length controls are disabled."
        ),
    )
    parser.add_argument("--length_scheduler_temperature", type=float, default=1.0)
    parser.add_argument("--length_scheduler_top_k", type=int, default=16)
    parser.add_argument("--vocab", default="csdnet_vocab.pkl")
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--n_mol", type=int, default=1000)
    parser.add_argument("--batch_size", type=int, default=100)
    parser.add_argument("--n_steps", type=int, default=500)
    parser.add_argument("--max_len", type=int, default=128)
    parser.add_argument("--output_dir", default="results/csdnet_eval")
    parser.add_argument("--w", type=float, default=2.0)
    parser.add_argument("--disable_fsm_check", action="store_true")
    parser.add_argument("--disable_rdkit_kekulize_check", action="store_true")
    parser.add_argument("--rdkit_check_interval", type=int, default=25)
    parser.add_argument("--max_sample_retries", type=int, default=3)
    parser.add_argument("--violation_neighborhood", type=int, default=2)
    parser.add_argument("--temperature_start", type=float, default=1.5)
    parser.add_argument("--temperature_end", type=float, default=0.25)
    parser.add_argument("--temperature_power", type=float, default=1.5)
    parser.add_argument("--top_k", type=int, default=0)
    parser.add_argument("--top_p", type=float, default=1.0)
    parser.add_argument("--gumbel_scale", type=float, default=1.0)
    parser.add_argument("--length_quantile_low", type=float, default=0.0)
    parser.add_argument("--length_quantile_high", type=float, default=1.0)
    parser.add_argument("--length_min", type=int, default=0)
    parser.add_argument("--length_max", type=int, default=0)
    parser.add_argument("--length_explore_fraction", type=float, default=0.0)
    parser.add_argument(
        "--length_batching",
        choices=("random", "sorted"),
        default="random",
    )
    parser.add_argument("--remask_power", type=float, default=1.0)
    add_length_adaptive_arguments(parser)
    add_confidence_planning_arguments(parser)
    add_block_refinement_arguments(parser)
    parser.add_argument("--strict_final_sanitize", action="store_true")
    parser.add_argument("--max_refill_factor", type=float, default=1)
    parser.add_argument(
        "--sampler_profile",
        choices=("custom", *SAMPLER_PROFILES),
        default="custom",
    )
    parser.add_argument(
        "--unmask_selection",
        choices=("random", "top_prob"),
        default="top_prob",
    )
    parser.add_argument("--cond", type=float, nargs=5, default=None)
    parser.add_argument("--qed", type=float, default=None)
    parser.add_argument("--logp", type=float, default=None)
    parser.add_argument("--sa", type=float, default=None)
    parser.add_argument("--tpsa", type=float, default=None)
    parser.add_argument("--mw", type=float, default=None)
    args = parser.parse_args()
    apply_sampler_profile(args)
    return args


def main():
    args = parse_args()
    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(args.seed)

    if args.length_scheduler_ckpt and args.length_prior_path:
        raise ValueError(
            "--length_scheduler_ckpt replaces --length_prior_path; do not mix "
            "learned and empirical length selection."
        )
    with open(args.vocab, "rb") as f:
        tk = SMILESTokenizer(pickle.load(f))

    if args.hf_dataset:
        ref_lengths, train_set = build_reference_from_hf_stream(args, tk)
    else:
        ref_lengths, train_set = build_reference_from_disk(args, tk)
    length_prior_metadata = None
    if args.length_prior_path:
        ref_lengths, length_prior_metadata = load_atomic_length_prior(
            args.length_prior_path,
            max_len=args.max_len,
        )
        print(
            "Loaded atomic length prior: "
            f"{length_prior_metadata['path']} "
            f"(n={len(ref_lengths)}, range={min(ref_lengths)}-{max(ref_lengths)})"
        )

    device = "cuda" if torch.cuda.is_available() else "cpu"
    model = load_backbone_from_checkpoint(args.ckpt_path, tk, device=device).to(device)
    model.eval()
    length_scheduler = None
    if args.length_scheduler_ckpt:
        length_scheduler = load_edit_scheduler_checkpoint(
            args.length_scheduler_ckpt,
            device=device,
        )
        print(
            "Loaded learned de novo length scheduler: "
            f"{args.length_scheduler_ckpt}"
        )

    cond_tensor, mode_str, cond_info = build_condition_from_args(
        args,
        cond_dim=getattr(model, "cond_dim", 0),
    )
    print(f"CSDNet evaluation mode: {mode_str}")
    print(f"FSM: {not args.disable_fsm_check}, RDKit kekulize: {not args.disable_rdkit_kekulize_check}")

    generated, sampling_diagnostics = sample_csdnet(
        model=model,
        tk=tk,
        ref_lengths=ref_lengths,
        n_mol=args.n_mol,
        cond=cond_tensor,
        w=args.w,
        device=device,
        batch_size=args.batch_size,
        n_steps=args.n_steps,
        use_fsm_check=not args.disable_fsm_check,
        use_rdkit_kekulize_check=not args.disable_rdkit_kekulize_check,
        rdkit_check_interval=args.rdkit_check_interval,
        max_sample_retries=args.max_sample_retries,
        violation_neighborhood=args.violation_neighborhood,
        temperature_start=args.temperature_start,
        temperature_end=args.temperature_end,
        temperature_power=args.temperature_power,
        top_k=args.top_k,
        top_p=args.top_p,
        gumbel_scale=args.gumbel_scale,
        length_quantile_low=args.length_quantile_low,
        length_quantile_high=args.length_quantile_high,
        length_min=args.length_min,
        length_max=args.length_max,
        length_explore_fraction=args.length_explore_fraction,
        length_batching=args.length_batching,
        remask_power=args.remask_power,
        **length_adaptive_kwargs(args),
        **confidence_planning_kwargs(args),
        **block_refinement_kwargs(args),
        unmask_selection=args.unmask_selection,
        strict_final_sanitize=args.strict_final_sanitize,
        max_refill_factor=args.max_refill_factor,
        length_scheduler=length_scheduler,
        length_scheduler_temperature=args.length_scheduler_temperature,
        length_scheduler_top_k=args.length_scheduler_top_k,
        return_diagnostics=True,
    )
    sampling_diagnostics.update(
        {
            "checkpoint": os.path.abspath(args.ckpt_path),
            "fsm_check": not args.disable_fsm_check,
            "rdkit_kekulize_check": not args.disable_rdkit_kekulize_check,
            "seed": int(args.seed),
            "n_steps": int(args.n_steps),
            "sampler_profile": args.sampler_profile,
        }
    )
    print("Sampling diagnostics:", sampling_diagnostics)
    if length_prior_metadata is not None:
        sampling_diagnostics["length_prior"] = length_prior_metadata

    os.makedirs(args.output_dir, exist_ok=True)
    generated_path = os.path.join(args.output_dir, "generated_mols.txt")
    metrics_path = os.path.join(args.output_dir, "metrics_log.txt")
    diagnostics_path = os.path.join(args.output_dir, "sampling_diagnostics.json")
    with open(generated_path, "w") as f:
        for smi in generated:
            f.write(smi + "\n")
    with open(diagnostics_path, "w") as f:
        json.dump(sampling_diagnostics, f, indent=2, sort_keys=True)

    metrics = calculate_basic_metrics(generated, train_set)
    write_generation_structure_summary(generated, args.output_dir, input_label=generated_path)

    print(f"Validity: {metrics['Validity']:.2f}%")
    print(f"Uniqueness: {metrics['Uniqueness']:.2f}%")
    print(f"Novelty: {metrics['Novelty']:.2f}%")
    print(f"IntDiv: {metrics['IntDiv']:.4f}")

    with open(metrics_path, "w") as f:
        f.write(f"Seed: {args.seed}\n")
        f.write(f"Steps: {args.n_steps}, CFG: {args.w}\n")
        f.write(f"Sampler profile: {args.sampler_profile}\n")
        f.write(f"Unmask selection: {args.unmask_selection}\n")
        f.write(f"FSM check: {not args.disable_fsm_check}\n")
        f.write(f"RDKit kekulize check: {not args.disable_rdkit_kekulize_check}\n")
        f.write(f"Strict final sanitize: {args.strict_final_sanitize}\n")
        f.write(
            "Length profile: "
            f"min={args.length_min}, max={args.length_max}, "
            f"explore_fraction={args.length_explore_fraction}\n"
        )
        f.write(f"Length batching: {args.length_batching}\n")
        f.write(
            "Length prior: "
            f"{args.length_scheduler_ckpt or args.length_prior_path or 'reference dataset encoded with CSDNet tokenizer'}\n"
        )
        f.write(f"Remask power: {args.remask_power}\n")
        f.write(f"Length-adaptive sampler: {args.length_adaptive}\n")
        f.write(f"Confidence temperature: {args.confidence_temperature}\n")
        f.write(f"Progressive commitment: {args.progressive_commit}\n")
        f.write(
            "Block refinement: "
            f"steps={args.block_refine_steps}, "
            f"span_max={args.block_refine_span_max}, "
            f"candidates={args.block_refine_candidates}, "
            f"temperature={args.block_refine_temperature}, "
            f"accept_margin={args.block_refine_accept_margin}\n"
        )
        if args.length_adaptive:
            f.write(
                "Length-adaptive transition: "
                f"{args.adaptive_length_low}-{args.adaptive_length_high}\n"
            )
            f.write(
                "Length-adaptive short endpoint: "
                f"temperature={args.adaptive_temperature_start_short}->"
                f"{args.adaptive_temperature_end_short}, "
                f"temperature_power={args.adaptive_temperature_power_short}, "
                f"gumbel_scale={args.adaptive_gumbel_scale_short}, "
                f"remask_power={args.adaptive_remask_power_short}\n"
            )
        if cond_info is not None:
            values, active_props = cond_info
            f.write("Cond active props: " + ",".join(active_props) + "\n")
            for prop in active_props:
                f.write(f"Cond {prop}: {values[prop]}\n")
        else:
            f.write("Cond: None (de novo)\n")
        for key, value in metrics.items():
            f.write(f"{key}: {value}\n")
    print(f"Generated molecules saved: {generated_path}")
    print(f"Metrics saved: {metrics_path}")


if __name__ == "__main__":
    main()
