#!/usr/bin/env python
"""Train a LoFlex-inspired external length and edit-order scheduler on ZINC."""

from __future__ import annotations

import argparse
import os
import pickle

import lightning as L
from lightning.pytorch.callbacks import LearningRateMonitor, ModelCheckpoint
from torch.utils.data import DataLoader

from CSDNet.model.edit_scheduler import EditScheduleNet
from CSDNet.model.edit_scheduler_data import EditScheduleCorruptionCollator
from CSDNet.model.edit_scheduler_lightning import EditSchedulerLightningModule
from CSDNet.util.checkpoint import load_backbone_from_checkpoint
from CSDNet.util.data import CSDNetDeNovoDataset
from CSDNet.util.tokenizer import SMILESTokenizer


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--data", default="data/zinc250k.csv")
    parser.add_argument("--teacher_ckpt", required=True)
    parser.add_argument("--vocab", default="csdnet_vocab.pkl")
    parser.add_argument("--checkpoint_dir", required=True)
    parser.add_argument("--run_name", default="csdnet-zinc250k-length-scheduler")
    parser.add_argument("--resume_from", default="")
    parser.add_argument("--max_len", type=int, default=128)
    parser.add_argument("--epochs", type=int, default=10)
    parser.add_argument("--batch_size", type=int, default=16)
    parser.add_argument("--accumulate_grad_batches", type=int, default=128)
    parser.add_argument("--num_workers", type=int, default=8)
    parser.add_argument("--precision", default="16-mixed")
    parser.add_argument("--devices", type=int, default=1)
    parser.add_argument("--strategy", default="auto")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--validation_size", type=int, default=5000)

    parser.add_argument("--hidden_size", type=int, default=256)
    parser.add_argument("--num_layers", type=int, default=4)
    parser.add_argument("--num_heads", type=int, default=8)
    parser.add_argument("--intermediate", type=int, default=1024)
    parser.add_argument("--dropout", type=float, default=0.1)
    parser.add_argument("--max_gaps", type=int, default=4)
    parser.add_argument("--max_span_length", type=int, default=24)
    parser.add_argument("--max_replacement_length", type=int, default=126)
    parser.add_argument("--zero_gap_probability", type=float, default=0.20)
    parser.add_argument("--pure_insertion_probability", type=float, default=0.30)
    parser.add_argument("--unchanged_length_probability", type=float, default=0.35)
    parser.add_argument("--unconditional_gap_probability", type=float, default=0.15)

    parser.add_argument("--lr", type=float, default=3e-4)
    parser.add_argument("--warmup_steps", type=int, default=100)
    parser.add_argument("--weight_decay", type=float, default=0.01)
    parser.add_argument("--order_loss_weight", type=float, default=0.25)
    parser.add_argument("--calibration_loss_weight", type=float, default=0.10)
    parser.add_argument("--rate_regularizer_weight", type=float, default=0.01)
    parser.add_argument("--order_temperature", type=float, default=0.7)
    parser.add_argument("--checkpoint_every_n_steps", type=int, default=100)
    return parser.parse_args()


def main():
    args = parse_args()
    L.seed_everything(args.seed, workers=True)
    if not os.path.isfile(args.data):
        raise FileNotFoundError(f"Missing ZINC250K data: {args.data}")
    if not os.path.isfile(args.teacher_ckpt):
        raise FileNotFoundError(f"Missing frozen teacher: {args.teacher_ckpt}")
    if args.max_replacement_length < args.max_len - 2:
        raise ValueError(
            "The scheduler must represent every de novo body length: "
            "max_replacement_length >= max_len - 2"
        )
    global_batch = (
        args.batch_size * args.devices * args.accumulate_grad_batches
    )
    if global_batch != 2048:
        raise ValueError(
            f"Expected effective global batch 2048, got {global_batch}"
        )

    from datasets import load_dataset

    with open(args.vocab, "rb") as handle:
        tokenizer = SMILESTokenizer(pickle.load(handle))
    dataset = load_dataset("csv", data_files=args.data, split="train")
    validation_size = min(args.validation_size, max(1, len(dataset) // 20))
    split = dataset.train_test_split(
        test_size=validation_size,
        seed=args.seed,
        shuffle=True,
    )
    train_dataset = CSDNetDeNovoDataset(
        split["train"],
        tokenizer,
        max_len=args.max_len,
        use_aromatic_cbi=False,
    )
    validation_dataset = CSDNetDeNovoDataset(
        split["test"],
        tokenizer,
        max_len=args.max_len,
        use_aromatic_cbi=False,
    )

    collator = EditScheduleCorruptionCollator(
        tokenizer=tokenizer,
        max_len=args.max_len,
        max_gaps=args.max_gaps,
        max_span_length=args.max_span_length,
        max_replacement_length=args.max_replacement_length,
        zero_gap_probability=args.zero_gap_probability,
        pure_insertion_probability=args.pure_insertion_probability,
        unchanged_length_probability=args.unchanged_length_probability,
        unconditional_gap_probability=args.unconditional_gap_probability,
    )
    loader_kwargs = {
        "batch_size": args.batch_size,
        "num_workers": args.num_workers,
        "collate_fn": collator,
        "pin_memory": True,
        "persistent_workers": args.num_workers > 0,
    }
    train_loader = DataLoader(
        train_dataset,
        shuffle=True,
        drop_last=True,
        **loader_kwargs,
    )
    validation_loader = DataLoader(
        validation_dataset,
        shuffle=False,
        drop_last=False,
        **loader_kwargs,
    )

    teacher = load_backbone_from_checkpoint(
        args.teacher_ckpt,
        tokenizer,
        device="cpu",
        use_ema=True,
    )
    teacher.requires_grad_(False)
    teacher.eval()
    scheduler = EditScheduleNet(
        vocab_size=tokenizer.vocab_size,
        pad_token_id=tokenizer.pad_id,
        mask_token_id=tokenizer.mask_id,
        hidden_size=args.hidden_size,
        num_layers=args.num_layers,
        num_heads=args.num_heads,
        intermediate=args.intermediate,
        max_position_embeddings=args.max_len,
        max_replacement_length=args.max_replacement_length,
        dropout=args.dropout,
    )
    module = EditSchedulerLightningModule(
        scheduler=scheduler,
        teacher=teacher,
        teacher_checkpoint=args.teacher_ckpt,
        lr=args.lr,
        warmup_steps=args.warmup_steps,
        lr_schedule="cosine",
        weight_decay=args.weight_decay,
        order_loss_weight=args.order_loss_weight,
        calibration_loss_weight=args.calibration_loss_weight,
        rate_regularizer_weight=args.rate_regularizer_weight,
        order_temperature=args.order_temperature,
    )

    trainable = sum(
        parameter.numel()
        for parameter in scheduler.parameters()
        if parameter.requires_grad
    )
    print("=" * 72)
    print("ZINC250K learned length scheduler")
    print(f"Frozen teacher: {args.teacher_ckpt}")
    print(f"Trainable scheduler: {trainable / 1e6:.3f}M parameters")
    print(f"Train/validation: {len(train_dataset):,}/{len(validation_dataset):,}")
    print(f"Epochs: {args.epochs}; effective global batch: {global_batch}")
    print(
        "Length supervision: conditional fragment gaps plus "
        f"{args.unconditional_gap_probability:.0%} full-molecule gaps"
    )
    print("Random length deltas: disabled; the scheduler predicts all lengths")
    print("=" * 72)

    callbacks = [
        ModelCheckpoint(
            dirpath=args.checkpoint_dir,
            filename=args.run_name + "-{step}",
            save_top_k=0,
            save_last=True,
            every_n_train_steps=args.checkpoint_every_n_steps,
            save_on_train_epoch_end=True,
        ),
        LearningRateMonitor(logging_interval="step"),
    ]
    trainer_kwargs = {
        "max_epochs": args.epochs,
        "accelerator": "gpu",
        "devices": args.devices,
        "precision": args.precision,
        "callbacks": callbacks,
        "log_every_n_steps": 10,
        "num_sanity_val_steps": 0,
        "accumulate_grad_batches": args.accumulate_grad_batches,
        "gradient_clip_val": 1.0,
        "limit_val_batches": 20,
        "check_val_every_n_epoch": 1,
    }
    if args.strategy != "auto":
        trainer_kwargs["strategy"] = args.strategy
    trainer = L.Trainer(**trainer_kwargs)
    trainer.fit(
        module,
        train_loader,
        validation_loader,
        ckpt_path=args.resume_from or None,
    )
    os.makedirs(args.checkpoint_dir, exist_ok=True)
    final_path = os.path.join(args.checkpoint_dir, "last.ckpt")
    trainer.save_checkpoint(final_path)
    print(f"Final learned length scheduler saved: {final_path}")


if __name__ == "__main__":
    main()
