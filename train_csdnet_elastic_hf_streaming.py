#!/usr/bin/env python
import argparse
import csv
import math
import os
import pickle
from pathlib import Path

import lightning as L
import torch
from lightning.pytorch.callbacks import LearningRateMonitor, ModelCheckpoint
from torch.utils.data import DataLoader, Dataset

from CSDNet.config import parse_args_with_yaml_config
from CSDNet.model.elastic_lightning_module import ElasticCSDNetLightningModule
from CSDNet.model.lightning_module import SimpleEMA
from CSDNet.util.checkpoint import load_backbone_from_checkpoint
from csdnet_hf_smiles import resolve_hf_token
from csdnet_tokenizer import SMILESTokenizer, tokenize_smiles
from train_csdnet_hf_streaming import (
    HFStreamingSMILESDataset,
    PerRunTimer,
    make_streaming_collate_fn,
)


class LocalCSVSMILESDataset(Dataset):
    """Validated local SMILES data for finite-epoch fine-tuning."""

    def __init__(
        self,
        path,
        smiles_col,
        tokenizer,
        max_len,
        skip_unknown=True,
        skip_long=True,
        use_aromatic_cbi=True,
    ):
        self.path = Path(path)
        self.tk = tokenizer
        self.max_len = int(max_len)
        self.use_aromatic_cbi = bool(use_aromatic_cbi)
        if not self.path.is_file() or self.path.stat().st_size == 0:
            raise FileNotFoundError(f"Missing or empty local CSV: {self.path}")

        self.smiles = []
        skipped_empty = 0
        skipped_long_count = 0
        skipped_unknown_count = 0
        with self.path.open(newline="", encoding="utf-8") as handle:
            reader = csv.DictReader(handle)
            fieldnames = list(reader.fieldnames or [])
            lookup = {name.strip().lower(): name for name in fieldnames}
            requested = str(smiles_col).strip()
            if requested.lower() == "auto":
                column = lookup.get("smiles") or lookup.get("text")
            else:
                column = requested if requested in fieldnames else lookup.get(
                    requested.lower()
                )
            if column is None:
                raise ValueError(
                    f"SMILES column {smiles_col!r} not found in {fieldnames}."
                )

            for row in reader:
                smiles = str(row.get(column, "") or "").strip()
                if not smiles:
                    skipped_empty += 1
                    continue
                tokens = tokenize_smiles(smiles)
                if skip_long and len(tokens) + 2 > self.max_len:
                    skipped_long_count += 1
                    continue
                if skip_unknown and any(
                    token not in self.tk.vocab for token in tokens
                ):
                    skipped_unknown_count += 1
                    continue
                self.smiles.append(smiles)

        if not self.smiles:
            raise ValueError(f"No usable SMILES found in {self.path}.")
        print(
            "Local SMILES dataset: "
            f"path={self.path} usable={len(self.smiles)} "
            f"empty={skipped_empty} long={skipped_long_count} "
            f"unknown={skipped_unknown_count}"
        )

    def __len__(self):
        return len(self.smiles)

    def __getitem__(self, index):
        smiles = self.smiles[index]
        item = {
            "input_ids": torch.tensor(
                self.tk.encode(smiles, self.max_len),
                dtype=torch.long,
            ),
            "cond": torch.empty(0, dtype=torch.float),
        }
        if self.use_aromatic_cbi:
            item["aromatic_mask"] = torch.tensor(
                self.tk.aromatic_context_mask(smiles, self.max_len),
                dtype=torch.bool,
            )
        return item


def parse_args():
    parser = argparse.ArgumentParser(
        description="Train variable-length Elastic CSDNet on SAFE-GPT."
    )
    parser.add_argument(
        "--config",
        default=None,
        help="Flat YAML training config. Explicit CLI options override it.",
    )
    parser.add_argument("--hf_dataset", default="datamol-io/safe-gpt")
    parser.add_argument("--hf_split", default="train")
    parser.add_argument("--smiles_col", default="auto")
    parser.add_argument("--safe_col", default="safe")
    parser.add_argument("--disable_safe_decode", action="store_true")
    parser.add_argument("--hf_token_env", default="HF_TOKEN")
    parser.add_argument("--hf_token", default=None)
    parser.add_argument(
        "--local_csv",
        default="",
        help="Use a finite local CSV instead of the HF streaming dataset.",
    )
    parser.add_argument("--local_smiles_col", default="smiles")
    parser.add_argument(
        "--epochs",
        type=int,
        default=0,
        help="Required finite epoch count when --local_csv is used.",
    )
    parser.add_argument("--vocab", default="csdnet_vocab.pkl")
    parser.add_argument("--max_len", type=int, default=256)
    parser.add_argument("--batch_size", type=int, default=256)
    parser.add_argument("--num_workers", type=int, default=4)
    parser.add_argument(
        "--dynamic_padding",
        action="store_true",
        help="Crop each batch to its longest sequence before the GPU forward.",
    )
    parser.add_argument(
        "--pad_to_multiple_of",
        type=int,
        default=8,
        help="Round dynamic batch lengths up for Tensor Core efficiency.",
    )
    parser.add_argument("--shuffle_buffer", type=int, default=50000)
    parser.add_argument("--stream_retries", type=int, default=20)
    parser.add_argument("--stream_retry_backoff", type=float, default=30.0)
    parser.add_argument("--max_steps", type=int, default=100000)
    parser.add_argument(
        "--max_time",
        type=str,
        default=None,
        help=(
            "Optional Lightning wall-clock limit in DD:HH:MM:SS format. "
            "Set it below the PBS walltime to allow a final checkpoint."
        ),
    )
    parser.add_argument("--lr", type=float, default=5e-4)
    parser.add_argument("--min_lr", type=float, default=None)
    parser.add_argument("--warmup_steps", type=int, default=2000)
    parser.add_argument(
        "--lr_schedule",
        choices=("cosine", "constant_with_warmup"),
        default="cosine",
    )
    parser.add_argument("--weight_decay", type=float, default=0.01)
    parser.add_argument("--adam_beta1", type=float, default=0.9)
    parser.add_argument("--adam_beta2", type=float, default=0.98)
    parser.add_argument("--adam_eps", type=float, default=1e-8)
    parser.add_argument("--ema_decay", type=float, default=0.9999)
    parser.add_argument("--cbi_weight", type=float, default=2.0)
    cbi_group = parser.add_mutually_exclusive_group()
    cbi_group.add_argument(
        "--enable_cbi",
        action="store_true",
        dest="use_cbi",
    )
    cbi_group.add_argument(
        "--disable_cbi",
        action="store_false",
        dest="use_cbi",
    )
    parser.set_defaults(use_cbi=False)
    parser.add_argument(
        "--normalized_cbi",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="Preserve each molecule's loss mass under CBI/AroCBI weighting.",
    )
    parser.add_argument("--aromatic_cbi_weight", type=float, default=1.2)
    parser.add_argument("--aromatic_cbi_final_weight", type=float, default=1.0)
    parser.add_argument(
        "--aromatic_cbi_anneal_start_fraction",
        type=float,
        default=0.10,
    )
    parser.add_argument(
        "--aromatic_cbi_anneal_end_fraction",
        type=float,
        default=0.90,
    )
    parser.add_argument(
        "--structured_corruption_prob",
        type=float,
        default=0.10,
    )
    parser.add_argument("--structured_span_min", type=int, default=2)
    parser.add_argument("--structured_span_max", type=int, default=8)
    parser.add_argument("--papl_alpha", type=float, default=3.0)
    parser.add_argument("--papl_tau", type=float, default=1.0)
    parser.add_argument("--hidden_size", type=int, default=256)
    parser.add_argument("--num_layers", type=int, default=8)
    parser.add_argument("--num_heads", type=int, default=8)
    parser.add_argument("--intermediate", type=int, default=1024)
    parser.add_argument("--max_position_embeddings", type=int, default=None)
    parser.add_argument("--position_embedding_type", default="rotary")
    parser.add_argument("--hidden_dropout_prob", type=float, default=0.1)
    parser.add_argument(
        "--attention_probs_dropout_prob",
        type=float,
        default=0.1,
    )
    parser.add_argument("--layer_norm_eps", type=float, default=1e-12)
    parser.add_argument("--initializer_range", type=float, default=0.02)
    parser.add_argument("--rate_min", type=float, default=0.001)
    parser.add_argument("--rate_max", type=float, default=20.0)
    parser.add_argument("--rate_initial", type=float, default=1.0)
    parser.add_argument(
        "--rate_parameterization",
        choices=("sigmoid", "exp", "softplus"),
        default="sigmoid",
    )
    parser.add_argument("--theta_rate_min", type=float, default=None)
    parser.add_argument("--phi_rate_min", type=float, default=None)
    parser.add_argument("--rate_output_bias", type=float, default=None)
    parser.add_argument("--fixed_unmask_rate", type=float, default=1.0)
    parser.add_argument("--kuma_shape_a", type=float, default=2.0)
    parser.add_argument("--insertion_loss_weight", type=float, default=1.0)
    parser.add_argument("--reinforce_weight", type=float, default=1.0)
    parser.add_argument("--schedule_regularizer_weight", type=float, default=1.0)
    parser.add_argument(
        "--loflex_aligned",
        action=argparse.BooleanOptionalAction,
        default=False,
        help="Use the LoFlex-aligned objective and regularizer.",
    )
    parser.add_argument(
        "--policy_time_conditioning",
        action=argparse.BooleanOptionalAction,
        default=False,
        help="Condition the clean ordering policy on sampled observation time.",
    )
    parser.add_argument("--fragment_corruption_prob", type=float, default=0.0)
    parser.add_argument("--fragment_internal_probability", type=float, default=0.25)
    parser.add_argument("--fragment_terminal_probability", type=float, default=0.25)
    parser.add_argument("--fragment_dual_probability", type=float, default=0.25)
    parser.add_argument("--fragment_multi_probability", type=float, default=0.25)
    parser.add_argument("--mdm_corruption_prob", type=float, default=0.0)
    parser.add_argument("--refine_corruption_prob", type=float, default=0.0)
    parser.add_argument("--refine_fraction_min", type=float, default=0.05)
    parser.add_argument("--refine_fraction_max", type=float, default=0.20)
    parser.add_argument("--length_loss_normalizer", type=float, default=None)
    parser.add_argument(
        "--disable_gradient_checkpointing",
        action="store_true",
        help="Disable activation checkpointing (uses more GPU memory).",
    )
    parser.add_argument(
        "--checkpoint_dir",
        default=(
            "csdnet_checkpoints_6m_elastic_safegpt_"
            "arocbi12anneal_struct10_100k"
        ),
    )
    parser.add_argument("--checkpoint_every_n_steps", type=int, default=10000)
    parser.add_argument("--resume_from_checkpoint", default=None)
    parser.add_argument(
        "--init_from_checkpoint",
        default=None,
        help=(
            "Warm-start backbone from checkpoint EMA weights while resetting "
            "optimizer, scheduler, EMA history, epoch, and global step."
        ),
    )
    parser.add_argument(
        "--run_name",
        default="csdnet-6m-elastic-safegpt-arocbi12anneal-struct10-100k",
    )
    parser.add_argument("--precision", default="bf16-mixed")
    parser.add_argument("--devices", type=int, default=4)
    parser.add_argument("--strategy", default="ddp")
    parser.add_argument("--accumulate_grad_batches", type=int, default=1)
    parser.add_argument("--gradient_clip_val", type=float, default=0.0)
    parser.add_argument("--skip_unknown", action="store_true", default=True)
    parser.add_argument(
        "--keep_unknown",
        action="store_false",
        dest="skip_unknown",
    )
    parser.add_argument("--skip_long", action="store_true", default=True)
    parser.add_argument(
        "--keep_long_truncated",
        action="store_false",
        dest="skip_long",
    )
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument(
        "--require_complete",
        action="store_true",
        help="Exit non-zero after saving if a timer stopped training early.",
    )
    return parse_args_with_yaml_config(parser)


def validate_args(args):
    positive_integer_fields = (
        "max_len",
        "batch_size",
        "max_steps",
        "devices",
        "accumulate_grad_batches",
        "hidden_size",
        "num_layers",
        "num_heads",
        "intermediate",
    )
    for name in positive_integer_fields:
        if int(getattr(args, name)) < 1:
            raise ValueError(f"--{name} must be positive.")
    if args.max_len < 4:
        raise ValueError("--max_len must leave room for BOS, content, and EOS.")
    if args.num_workers < 0:
        raise ValueError("--num_workers cannot be negative.")
    if bool(args.local_csv) != (args.epochs > 0):
        raise ValueError(
            "--local_csv and a positive --epochs must be provided together."
        )
    if args.resume_from_checkpoint and args.init_from_checkpoint:
        raise ValueError(
            "Use either --resume_from_checkpoint or --init_from_checkpoint, "
            "not both."
        )
    if args.pad_to_multiple_of < 1:
        raise ValueError("--pad_to_multiple_of must be positive.")
    if not 0 <= args.warmup_steps <= args.max_steps:
        raise ValueError("--warmup_steps must be between 0 and --max_steps.")
    if args.hidden_size % args.num_heads != 0:
        raise ValueError("--hidden_size must be divisible by --num_heads.")
    if args.max_position_embeddings is not None:
        required_positions = (
            args.max_len
            if args.position_embedding_type == "rotary"
            else args.max_len + 1
        )
        if args.max_position_embeddings < required_positions:
            raise ValueError(
                "--max_position_embeddings must be at least "
                f"{required_positions} for max_len={args.max_len}."
            )
    if args.checkpoint_every_n_steps < 0:
        raise ValueError("--checkpoint_every_n_steps cannot be negative.")
    geometry_probabilities = (
        args.fragment_internal_probability,
        args.fragment_terminal_probability,
        args.fragment_dual_probability,
        args.fragment_multi_probability,
    )
    if any(value < 0.0 for value in geometry_probabilities) or not math.isclose(
        sum(geometry_probabilities),
        1.0,
        rel_tol=0.0,
        abs_tol=1e-8,
    ):
        raise ValueError(
            "Fragment geometry probabilities must be non-negative and sum to one."
        )
    if args.loflex_aligned:
        expected = {
            "fixed_unmask_rate": 1.0,
            "kuma_shape_a": 1.0,
            "theta_rate_min": 0.0,
            "phi_rate_min": 1.01,
        }
        for name, value in expected.items():
            actual = getattr(args, name)
            if actual is None or abs(float(actual) - value) > 1e-8:
                raise ValueError(
                    f"LoFlex alignment requires --{name} {value}; got {actual}."
                )
        if args.rate_parameterization != "exp":
            raise ValueError(
                "LoFlex alignment requires --rate_parameterization exp."
            )
        if not args.policy_time_conditioning:
            raise ValueError(
                "LoFlex alignment requires --policy_time_conditioning."
            )


def main():
    args = parse_args()
    validate_args(args)
    L.seed_everything(args.seed, workers=True)
    token = None
    if not args.local_csv:
        token = resolve_hf_token(
            args.hf_token,
            args.hf_token_env,
            required=True,
        )

    with open(args.vocab, "rb") as handle:
        tokenizer = SMILESTokenizer(pickle.load(handle))
    max_position_embeddings = (
        args.max_position_embeddings
        if args.max_position_embeddings is not None
        else (
            args.max_len
            if args.position_embedding_type == "rotary"
            else args.max_len + tokenizer.pad_id + 1
        )
    )
    required_positions = (
        args.max_len
        if args.position_embedding_type == "rotary"
        else args.max_len + tokenizer.pad_id + 1
    )
    if max_position_embeddings < required_positions:
        raise ValueError(
            "max_position_embeddings must be at least "
            f"{required_positions} for max_len={args.max_len} and "
            f"pad_id={tokenizer.pad_id}; got {max_position_embeddings}."
        )

    if args.local_csv:
        dataset = LocalCSVSMILESDataset(
            path=args.local_csv,
            smiles_col=args.local_smiles_col,
            tokenizer=tokenizer,
            max_len=args.max_len,
            skip_unknown=args.skip_unknown,
            skip_long=args.skip_long,
            use_aromatic_cbi=True,
        )
    else:
        dataset = HFStreamingSMILESDataset(
            dataset_name=args.hf_dataset,
            split=args.hf_split,
            smiles_col=args.smiles_col,
            safe_col=args.safe_col,
            tokenizer=tokenizer,
            max_len=args.max_len,
            token=token,
            allow_safe_decode=not args.disable_safe_decode,
            shuffle_buffer=args.shuffle_buffer,
            seed=args.seed,
            skip_unknown=args.skip_unknown,
            skip_long=args.skip_long,
            use_aromatic_cbi=True,
            stream_retries=args.stream_retries,
            stream_retry_backoff=args.stream_retry_backoff,
        )
    loader = DataLoader(
        dataset,
        batch_size=args.batch_size,
        shuffle=bool(args.local_csv),
        drop_last=bool(args.local_csv),
        num_workers=args.num_workers,
        collate_fn=make_streaming_collate_fn(
            tokenizer,
            use_aromatic_cbi=True,
            dynamic_padding=args.dynamic_padding,
            pad_to_multiple_of=args.pad_to_multiple_of,
        ),
        pin_memory=True,
        persistent_workers=args.num_workers > 0,
    )
    examples_per_step = (
        args.batch_size
        * args.devices
        * args.accumulate_grad_batches
    )
    optimizer_steps_per_epoch = None
    examples_per_epoch = None
    estimated_optimizer_steps = args.max_steps
    if args.local_csv:
        batches_per_rank = math.ceil(len(dataset) / args.devices) // args.batch_size
        optimizer_steps_per_epoch = math.ceil(
            batches_per_rank / args.accumulate_grad_batches
        )
        if optimizer_steps_per_epoch < 1:
            raise ValueError(
                "Local dataset is too small for the requested global batch size."
            )
        examples_per_epoch = (
            batches_per_rank * args.batch_size * args.devices
        )
        estimated_optimizer_steps = optimizer_steps_per_epoch * args.epochs

    model = ElasticCSDNetLightningModule(
        vocab_size=tokenizer.vocab_size,
        pad_id=tokenizer.pad_id,
        mask_id=tokenizer.mask_id,
        bos_id=tokenizer.bos_id,
        eos_id=tokenizer.eos_id,
        unk_id=tokenizer.unk_id,
        scaffold_ids=tokenizer.scaffold_ids,
        aromatic_ids=tokenizer.aromatic_ids,
        cond_dim=0,
        use_cbi=args.use_cbi,
        cbi_weight=args.cbi_weight,
        use_aromatic_cbi=True,
        normalized_cbi=args.normalized_cbi,
        aromatic_cbi_weight=args.aromatic_cbi_weight,
        aromatic_cbi_final_weight=args.aromatic_cbi_final_weight,
        aromatic_cbi_anneal_steps=estimated_optimizer_steps,
        aromatic_cbi_anneal_start_fraction=(
            args.aromatic_cbi_anneal_start_fraction
        ),
        aromatic_cbi_anneal_end_fraction=(
            args.aromatic_cbi_anneal_end_fraction
        ),
        structured_corruption_prob=args.structured_corruption_prob,
        structured_span_min=args.structured_span_min,
        structured_span_max=args.structured_span_max,
        lr=args.lr,
        min_lr=args.min_lr,
        warmup_steps=args.warmup_steps,
        lr_schedule=args.lr_schedule,
        weight_decay=args.weight_decay,
        adam_beta1=args.adam_beta1,
        adam_beta2=args.adam_beta2,
        adam_eps=args.adam_eps,
        papl_alpha=args.papl_alpha,
        papl_tau=args.papl_tau,
        ema_decay=args.ema_decay,
        hidden_size=args.hidden_size,
        num_layers=args.num_layers,
        num_heads=args.num_heads,
        intermediate=args.intermediate,
        max_position_embeddings=max_position_embeddings,
        position_embedding_type=args.position_embedding_type,
        hidden_dropout_prob=args.hidden_dropout_prob,
        attention_probs_dropout_prob=args.attention_probs_dropout_prob,
        layer_norm_eps=args.layer_norm_eps,
        initializer_range=args.initializer_range,
        rate_min=args.rate_min,
        rate_max=args.rate_max,
        rate_initial=args.rate_initial,
        rate_parameterization=args.rate_parameterization,
        theta_rate_min=args.theta_rate_min,
        phi_rate_min=args.phi_rate_min,
        rate_output_bias=args.rate_output_bias,
        fixed_unmask_rate=args.fixed_unmask_rate,
        kuma_shape_a=args.kuma_shape_a,
        insertion_loss_weight=args.insertion_loss_weight,
        reinforce_weight=args.reinforce_weight,
        schedule_regularizer_weight=args.schedule_regularizer_weight,
        loflex_objective=args.loflex_aligned,
        policy_time_conditioning=args.policy_time_conditioning,
        fragment_corruption_prob=args.fragment_corruption_prob,
        fragment_internal_probability=args.fragment_internal_probability,
        fragment_terminal_probability=args.fragment_terminal_probability,
        fragment_dual_probability=args.fragment_dual_probability,
        fragment_multi_probability=args.fragment_multi_probability,
        mdm_corruption_prob=args.mdm_corruption_prob,
        refine_corruption_prob=args.refine_corruption_prob,
        refine_fraction_min=args.refine_fraction_min,
        refine_fraction_max=args.refine_fraction_max,
        length_loss_normalizer=(
            args.max_len
            if args.length_loss_normalizer is None
            else args.length_loss_normalizer
        ),
        drop_cond_prob=0.0,
        gradient_checkpointing=not args.disable_gradient_checkpointing,
    )
    if args.init_from_checkpoint:
        source = load_backbone_from_checkpoint(
            args.init_from_checkpoint,
            tokenizer,
            device="cpu",
            use_ema=True,
        )
        model.backbone.load_state_dict(source.state_dict(), strict=True)
        model.ema = SimpleEMA(model.backbone, decay=args.ema_decay)
        print(
            "Warm-started backbone from EMA weights and reset training state: "
            f"{args.init_from_checkpoint}"
        )
    parameter_count = sum(
        parameter.numel()
        for parameter in model.backbone.parameters()
    )
    print("=" * 72)
    print("Elastic CSDNet SAFE-GPT streaming training")
    print(f"Backbone and rate-head parameters: {parameter_count / 1e6:.2f}M")
    print(
        f"architecture=elastic_csdnet position={args.position_embedding_type} "
        f"CBI={'on' if args.use_cbi else 'off'} "
        f"normalized_CBI={args.normalized_cbi} "
        f"aroCBI={args.aromatic_cbi_weight:g}"
        f"->{args.aromatic_cbi_final_weight:g} "
        f"structured={args.structured_corruption_prob:g}"
        f"[{args.structured_span_min},{args.structured_span_max}] "
        f"fixed_unmask={args.fixed_unmask_rate:g} "
        f"rate={args.rate_parameterization} "
        f"loflex_aligned={args.loflex_aligned} "
        f"aux_modes=fragment:{args.fragment_corruption_prob:g},"
        f"mdm:{args.mdm_corruption_prob:g},"
        f"refine:{args.refine_corruption_prob:g} "
        "fragment_geometry="
        f"{args.fragment_internal_probability:g}/"
        f"{args.fragment_terminal_probability:g}/"
        f"{args.fragment_dual_probability:g}/"
        f"{args.fragment_multi_probability:g} "
        f"dynamic_padding={args.dynamic_padding} "
        f"batch/GPU={args.batch_size} devices={args.devices} "
        f"accumulate={args.accumulate_grad_batches} "
        f"global_batch="
        f"{args.batch_size * args.devices * args.accumulate_grad_batches} "
        f"steps={args.max_steps} lr={args.lr:g} "
        f"min_lr={args.min_lr if args.min_lr is not None else 'legacy'} "
        f"schedule={args.lr_schedule} warmup={args.warmup_steps}"
    )
    if args.local_csv:
        print(
            f"finite_epochs={args.epochs} usable_rows={len(dataset)} "
            f"estimated_optimizer_steps_per_epoch={optimizer_steps_per_epoch} "
            f"estimated_optimizer_steps={estimated_optimizer_steps} "
            f"estimated_examples_per_epoch={examples_per_epoch} "
            f"estimated_examples_seen={examples_per_epoch * args.epochs}"
        )
    else:
        print(
            f"nominal_examples_per_step={examples_per_step} "
            f"nominal_training_examples={examples_per_step * args.max_steps} "
            f"nominal_warmup_examples={examples_per_step * args.warmup_steps}"
        )
    print("=" * 72)

    callbacks = [LearningRateMonitor(logging_interval="step")]
    if args.max_time:
        callbacks.append(PerRunTimer(duration=args.max_time))
    if args.checkpoint_every_n_steps > 0:
        callbacks.insert(
            0,
            ModelCheckpoint(
                dirpath=args.checkpoint_dir,
                filename=args.run_name + "-{step}",
                save_top_k=0,
                save_last=True,
                every_n_train_steps=args.checkpoint_every_n_steps,
                save_on_train_epoch_end=False,
            ),
        )

    trainer_kwargs = {
        "max_epochs": args.epochs if args.local_csv else 999,
        "max_steps": -1 if args.local_csv else args.max_steps,
        "accelerator": "gpu",
        "devices": args.devices,
        "precision": args.precision,
        "callbacks": callbacks,
        "log_every_n_steps": 25,
        "num_sanity_val_steps": 0,
        "limit_val_batches": 0,
        "accumulate_grad_batches": args.accumulate_grad_batches,
        "gradient_clip_val": args.gradient_clip_val,
    }
    if args.strategy != "auto":
        trainer_kwargs["strategy"] = args.strategy
    trainer = L.Trainer(**trainer_kwargs)
    trainer.fit(
        model,
        loader,
        ckpt_path=args.resume_from_checkpoint,
    )

    os.makedirs(args.checkpoint_dir, exist_ok=True)
    final_checkpoint = os.path.join(args.checkpoint_dir, "last.ckpt")
    trainer.save_checkpoint(final_checkpoint)
    print(f"Final checkpoint saved: {final_checkpoint}")
    completed_epochs = int(
        trainer.fit_loop.epoch_progress.current.completed
    )
    print(
        f"Training progress: global_step={trainer.global_step} "
        f"completed_epochs={completed_epochs}"
    )
    if args.require_complete:
        if args.local_csv and completed_epochs < args.epochs:
            raise RuntimeError(
                "Finite-epoch training stopped before completion: "
                f"{completed_epochs}/{args.epochs} epochs. The checkpoint is "
                "safe to resume."
            )
        if not args.local_csv and trainer.global_step < args.max_steps:
            raise RuntimeError(
                "Streaming training stopped before completion: "
                f"{trainer.global_step}/{args.max_steps} steps. The checkpoint "
                "is safe to resume."
            )


if __name__ == "__main__":
    main()
