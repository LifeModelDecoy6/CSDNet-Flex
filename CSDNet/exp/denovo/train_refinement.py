#!/usr/bin/env python
"""Fine-tune CSDNet with mixed masked and all-position refinement supervision."""

from __future__ import annotations

import argparse
import os
import pickle

import lightning as L
import torch
from lightning.pytorch.callbacks import LearningRateMonitor, ModelCheckpoint
from torch.utils.data import DataLoader

from CSDNet.exp.denovo.sampler_profiles import SAMPLER_PROFILES
from CSDNet.model.lightning_module import CSDNetLightningModule, SimpleEMA
from CSDNet.util.checkpoint import load_backbone_from_checkpoint
from CSDNet.util.data import CSDNetDeNovoDataset, make_collate_fn
from CSDNet.util.hf_smiles import extract_smiles_from_row
from CSDNet.util.tokenizer import SMILESTokenizer


def canonicalize_example(example, keep_stereochemistry=False):
    from rdkit import Chem

    smiles = extract_smiles_from_row(example)
    molecule = Chem.MolFromSmiles(smiles) if smiles else None
    if molecule is None:
        return {"text": "", "_canonical_valid": False}
    if not keep_stereochemistry:
        Chem.RemoveStereochemistry(molecule)
    canonical = Chem.MolToSmiles(
        molecule,
        canonical=True,
        isomericSmiles=keep_stereochemistry,
    )
    return {"text": canonical, "_canonical_valid": bool(canonical)}


def parse_args():
    parser = argparse.ArgumentParser(
        description=(
            "Fine-tune CSDNet with masked, draft-refinement, and fragment-span "
            "supervision."
        )
    )
    parser.add_argument("--data_dir", default="csdnet_data/pubchem_10m_with_props_v2")
    parser.add_argument("--vocab", default="csdnet_vocab.pkl")
    parser.add_argument("--init_from", required=True)
    parser.add_argument("--resume_from", default="")
    parser.add_argument("--checkpoint_dir", required=True)
    parser.add_argument("--run_name", default="csdnet-90m-pubchem10m-refinement")
    parser.add_argument("--max_len", type=int, default=256)
    parser.add_argument("--dynamic_padding", action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument("--pad_to_multiple_of", type=int, default=8)
    parser.add_argument("--batch_size", type=int, default=32)
    parser.add_argument("--num_workers", type=int, default=4)
    parser.add_argument("--devices", type=int, default=1)
    parser.add_argument("--strategy", default="auto")
    parser.add_argument("--precision", default="16-mixed")
    parser.add_argument("--accumulate_grad_batches", type=int, default=64)
    parser.add_argument(
        "--gradient_checkpointing",
        action=argparse.BooleanOptionalAction,
        default=False,
        help=(
            "Activation checkpointing is disabled by default because trajectory "
            "refinement performs multiple backbone forwards per training step."
        ),
    )
    parser.add_argument("--epochs", type=int, default=1)
    parser.add_argument("--lr", type=float, default=5e-5)
    parser.add_argument("--warmup_steps", type=int, default=250)
    parser.add_argument(
        "--lr_schedule",
        choices=("cosine", "constant_with_warmup"),
        default="cosine",
    )
    parser.add_argument("--weight_decay", type=float, default=0.0)
    parser.add_argument("--gradient_clip_val", type=float, default=1.0)
    parser.add_argument("--checkpoint_every_n_steps", type=int, default=500)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--max_samples", type=int, default=0)
    parser.add_argument("--canonicalize_smiles", action="store_true")
    parser.add_argument("--keep_stereochemistry", action="store_true")

    parser.add_argument("--refinement_loss_weight", type=float, default=0.50)
    parser.add_argument("--refinement_warmup_steps", type=int, default=250)
    parser.add_argument("--refinement_corruption_min", type=float, default=0.05)
    parser.add_argument("--refinement_corruption_max", type=float, default=0.50)
    parser.add_argument("--refinement_clean_weight", type=float, default=0.20)
    parser.add_argument("--refinement_mask_fraction", type=float, default=0.15)
    parser.add_argument(
        "--refinement_corruption_mode",
        choices=("uniform", "trajectory"),
        default="uniform",
    )
    parser.add_argument("--trajectory_rollout_steps", type=int, default=2)
    parser.add_argument("--trajectory_rollout_decay", type=float, default=0.50)
    parser.add_argument(
        "--training_objective_mode",
        choices=("mask_refine", "three_way_equal"),
        default="mask_refine",
    )
    parser.add_argument("--fragment_span_min", type=int, default=1)
    parser.add_argument("--fragment_span_max", type=int, default=24)
    parser.add_argument(
        "--fragment_span_continue_probability",
        type=float,
        default=0.78,
    )
    parser.add_argument(
        "--fragment_internal_probability",
        type=float,
        default=0.67,
    )
    parser.add_argument(
        "--trajectory_profile",
        choices=tuple(sorted(SAMPLER_PROFILES)),
        default="promax_progressive_length_coupled",
    )
    return parser.parse_args()


def trajectory_profile_kwargs(profile_name):
    profile = SAMPLER_PROFILES[profile_name]
    if not profile.get("progressive_commit", False):
        raise ValueError(
            "trajectory refinement requires a progressive-commit sampler profile"
        )
    return {
        "trajectory_length_low": profile["adaptive_length_low"],
        "trajectory_length_high": profile["adaptive_length_high"],
        "trajectory_temperature_start": profile["temperature_start"],
        "trajectory_temperature_end": profile["temperature_end"],
        "trajectory_temperature_power": profile["temperature_power"],
        "trajectory_remask_power": profile["remask_power"],
        "trajectory_gumbel_scale": profile["gumbel_scale"],
        "trajectory_temperature_start_short": profile[
            "adaptive_temperature_start_short"
        ],
        "trajectory_temperature_end_short": profile[
            "adaptive_temperature_end_short"
        ],
        "trajectory_temperature_power_short": profile[
            "adaptive_temperature_power_short"
        ],
        "trajectory_remask_power_short": profile[
            "adaptive_remask_power_short"
        ],
        "trajectory_gumbel_scale_short": profile[
            "adaptive_gumbel_scale_short"
        ],
        "trajectory_confidence_temperature": profile.get(
            "confidence_temperature",
            0.0,
        ),
        "trajectory_confidence_length_adaptive": profile.get(
            "confidence_length_adaptive",
            False,
        ),
        "trajectory_confidence_length_low": profile.get(
            "adaptive_confidence_length_low",
            28.0,
        ),
        "trajectory_confidence_length_high": profile.get(
            "adaptive_confidence_length_high",
            34.0,
        ),
        "trajectory_confidence_temperature_short": profile.get(
            "adaptive_confidence_temperature_short",
            1.0,
        ),
    }


def initialize_from_ema(model, checkpoint_path, tokenizer):
    source = load_backbone_from_checkpoint(
        checkpoint_path,
        tokenizer,
        device="cpu",
        use_ema=True,
    )
    missing, unexpected = model.backbone.load_state_dict(
        source.state_dict(),
        strict=False,
    )
    allowed_missing = {
        key for key in missing if key.startswith("corruption_level_embedding.")
    }
    disallowed_missing = sorted(set(missing) - allowed_missing)
    if disallowed_missing or unexpected:
        raise RuntimeError(
            "Warm-start mismatch: "
            f"missing={disallowed_missing}, unexpected={sorted(unexpected)}"
        )
    del source

    if model.use_ema:
        model.ema = SimpleEMA(model.backbone, decay=model.ema_decay)
    print(
        "Warm-started from EMA backbone; new corruption-level parameters: "
        f"{len(allowed_missing)} keys. Optimizer and scheduler start fresh."
    )


def main():
    args = parse_args()
    L.seed_everything(args.seed, workers=True)

    if args.epochs <= 0:
        raise ValueError("epochs must be positive")
    global_batch = args.batch_size * args.devices * args.accumulate_grad_batches
    if global_batch != 2048:
        raise ValueError(
            f"Expected effective global batch 2048, got {global_batch}."
        )
    if not os.path.isfile(args.init_from) or os.path.getsize(args.init_from) == 0:
        raise FileNotFoundError(f"Missing warm-start checkpoint: {args.init_from}")
    if args.resume_from and (
        not os.path.isfile(args.resume_from)
        or os.path.getsize(args.resume_from) == 0
    ):
        raise FileNotFoundError(f"Missing resume checkpoint: {args.resume_from}")

    with open(args.vocab, "rb") as handle:
        tokenizer = SMILESTokenizer(pickle.load(handle))

    required_positions = args.max_len + tokenizer.pad_id + 1
    if required_positions > 257:
        raise ValueError(
            "The 90M source checkpoint has 257 absolute position embeddings; "
            f"requested layout requires {required_positions}."
        )

    try:
        from datasets import DatasetDict, load_from_disk
    except ImportError as exc:
        raise SystemExit("Missing datasets dependency.") from exc

    if os.path.isdir(args.data_dir):
        dataset = load_from_disk(args.data_dir)
    elif os.path.isfile(args.data_dir):
        suffix = os.path.splitext(args.data_dir)[1].lower()
        if suffix not in {".csv", ".tsv"}:
            raise ValueError(f"Unsupported dataset file: {args.data_dir}")
        delimiter = "\t" if suffix == ".tsv" else ","
        from datasets import load_dataset

        dataset = load_dataset(
            "csv",
            data_files=args.data_dir,
            delimiter=delimiter,
            split="train",
        )
    else:
        raise FileNotFoundError(f"Missing dataset: {args.data_dir}")
    if isinstance(dataset, DatasetDict):
        dataset = dataset["train"]
    if args.canonicalize_smiles:
        dataset = dataset.map(
            canonicalize_example,
            fn_kwargs={"keep_stereochemistry": args.keep_stereochemistry},
            num_proc=max(1, args.num_workers),
            desc="Canonicalizing refinement SMILES",
        )
        dataset = dataset.filter(
            lambda example: bool(example["_canonical_valid"]),
            num_proc=max(1, args.num_workers),
            desc="Removing invalid refinement SMILES",
        )
    if args.max_samples > 0:
        dataset = dataset.select(range(min(args.max_samples, len(dataset))))
    dataset = dataset.shuffle(seed=args.seed)

    train_dataset = CSDNetDeNovoDataset(
        dataset,
        tokenizer,
        max_len=args.max_len,
        use_aromatic_cbi=True,
    )
    train_loader = DataLoader(
        train_dataset,
        batch_size=args.batch_size,
        shuffle=True,
        num_workers=args.num_workers,
        collate_fn=make_collate_fn(
            tokenizer,
            include_cond=True,
            use_aromatic_cbi=True,
            dynamic_padding=args.dynamic_padding,
            pad_to_multiple_of=args.pad_to_multiple_of,
        ),
        pin_memory=True,
        persistent_workers=args.num_workers > 0,
        drop_last=True,
    )

    trajectory_kwargs = trajectory_profile_kwargs(args.trajectory_profile)
    model = CSDNetLightningModule(
        vocab_size=tokenizer.vocab_size,
        pad_id=tokenizer.pad_id,
        mask_id=tokenizer.mask_id,
        bos_id=tokenizer.bos_id,
        eos_id=tokenizer.eos_id,
        unk_id=tokenizer.unk_id,
        scaffold_ids=tokenizer.scaffold_ids,
        aromatic_ids=tokenizer.aromatic_ids,
        cond_dim=0,
        use_cbi=True,
        cbi_weight=2.0,
        use_aromatic_cbi=True,
        aromatic_cbi_weight=1.2,
        normalized_cbi=True,
        lr=args.lr,
        warmup_steps=args.warmup_steps,
        lr_schedule=args.lr_schedule,
        weight_decay=args.weight_decay,
        adam_beta1=0.9,
        adam_beta2=0.999,
        adam_eps=1e-8,
        ema_decay=0.9999,
        hidden_size=768,
        num_layers=12,
        num_heads=12,
        intermediate=3072,
        max_position_embeddings=257,
        hidden_dropout_prob=0.1,
        attention_probs_dropout_prob=0.1,
        layer_norm_eps=1e-12,
        initializer_range=0.02,
        position_embedding_type="absolute",
        gradient_checkpointing=args.gradient_checkpointing,
        drop_cond_prob=0.0,
        corruption_level_conditioning=True,
        refinement_loss_weight=args.refinement_loss_weight,
        refinement_warmup_steps=args.refinement_warmup_steps,
        refinement_corruption_min=args.refinement_corruption_min,
        refinement_corruption_max=args.refinement_corruption_max,
        refinement_clean_weight=args.refinement_clean_weight,
        refinement_mask_fraction=args.refinement_mask_fraction,
        refinement_corruption_mode=args.refinement_corruption_mode,
        trajectory_rollout_steps=args.trajectory_rollout_steps,
        trajectory_rollout_decay=args.trajectory_rollout_decay,
        training_objective_mode=args.training_objective_mode,
        fragment_span_min=args.fragment_span_min,
        fragment_span_max=args.fragment_span_max,
        fragment_span_continue_probability=(
            args.fragment_span_continue_probability
        ),
        fragment_internal_probability=args.fragment_internal_probability,
        **trajectory_kwargs,
    )
    if not args.resume_from:
        initialize_from_ema(model, args.init_from, tokenizer)

    n_params = sum(parameter.numel() for parameter in model.backbone.parameters())
    expected_steps_per_epoch = len(dataset) // global_batch
    expected_steps = expected_steps_per_epoch * args.epochs
    print("=" * 72)
    print("CSDNet 90M final multi-objective fine-tuning")
    print(f"Dataset: {args.data_dir} ({len(dataset):,} molecules)")
    print(f"Backbone parameters: {n_params / 1e6:.2f}M")
    print(f"Devices: {args.devices}; effective global batch: {global_batch}")
    print(f"Gradient checkpointing: {args.gradient_checkpointing}")
    print(
        f"Epochs: {args.epochs}; expected optimizer steps: "
        f"approximately {expected_steps:,}"
    )
    print(
        "Canonicalization: "
        f"enabled={args.canonicalize_smiles}, "
        f"keep_stereochemistry={args.keep_stereochemistry}"
    )
    if args.training_objective_mode == "three_way_equal":
        print(
            "Objective: exactly 1/3 masked diffusion + 1/3 trajectory "
            "refinement + 1/3 contiguous fragment infill"
        )
        print(
            "Fragment corruption: "
            f"span={args.fragment_span_min}-{args.fragment_span_max}, "
            "geometric_continue="
            f"{args.fragment_span_continue_probability:.2f}, "
            f"internal_probability={args.fragment_internal_probability:.2f}"
        )
    else:
        print(
            "Mixed objective: "
            f"beta={args.refinement_loss_weight}, "
            f"corruption={args.refinement_corruption_min:.2f}-"
            f"{args.refinement_corruption_max:.2f}, "
            f"clean_weight={args.refinement_clean_weight:.2f}"
        )
    print(
        "Refinement trajectory: "
        f"mode={args.refinement_corruption_mode}, "
        f"profile={args.trajectory_profile}, "
        f"rollout={args.trajectory_rollout_steps}x"
        f"{args.trajectory_rollout_decay:.2f}"
    )
    print(f"Output: {args.checkpoint_dir}")
    print(f"Resume checkpoint: {args.resume_from or 'none'}")
    print("=" * 72)

    callbacks = [LearningRateMonitor(logging_interval="step")]
    if args.checkpoint_every_n_steps > 0:
        callbacks.insert(
            0,
            ModelCheckpoint(
                dirpath=args.checkpoint_dir,
                filename=args.run_name + "-{step}",
                save_top_k=0,
                save_last=True,
                every_n_train_steps=args.checkpoint_every_n_steps,
                save_on_train_epoch_end=True,
            ),
        )

    trainer_kwargs = dict(
        max_epochs=args.epochs,
        max_steps=-1,
        accelerator="gpu",
        devices=args.devices,
        precision=args.precision,
        callbacks=callbacks,
        log_every_n_steps=25,
        num_sanity_val_steps=0,
        limit_val_batches=0,
        accumulate_grad_batches=args.accumulate_grad_batches,
        gradient_clip_val=args.gradient_clip_val,
    )
    if args.strategy != "auto":
        trainer_kwargs["strategy"] = args.strategy
    trainer = L.Trainer(**trainer_kwargs)
    trainer.fit(
        model,
        train_loader,
        ckpt_path=args.resume_from or None,
    )

    os.makedirs(args.checkpoint_dir, exist_ok=True)
    final_path = os.path.join(args.checkpoint_dir, "last.ckpt")
    trainer.save_checkpoint(final_path)
    print(f"Final refinement checkpoint saved: {final_path}")


if __name__ == "__main__":
    main()
