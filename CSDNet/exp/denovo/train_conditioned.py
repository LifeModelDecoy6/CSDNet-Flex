#!/usr/bin/env python
import argparse
import os
import pickle

import lightning as L
from lightning.pytorch.callbacks import LearningRateMonitor, ModelCheckpoint
from torch.utils.data import DataLoader

from CSDNet.config import get_model_config
from CSDNet.model.lightning_module import CSDNetLightningModule, PROP_NAMES
from CSDNet.util.data import CSDNetPropertyDataset, make_collate_fn
from CSDNet.util.tokenizer import SMILESTokenizer


def apply_model_size(args):
    cfg = get_model_config(args.model_size)
    for key in ("hidden_size", "num_layers", "num_heads", "intermediate", "lr", "warmup_steps", "batch_size"):
        if getattr(args, key) is None:
            setattr(args, key, cfg[key])
    return args


def parse_args():
    parser = argparse.ArgumentParser(description="Train property-conditioned CSDNet on local Arrow data.")
    parser.add_argument("--model_size", choices=["6m", "30m", "90m"], default="6m")
    parser.add_argument("--data_dir", default="csdnet_data/pubchem_10m_with_props_v2")
    parser.add_argument("--vocab", default="csdnet_vocab.pkl")
    parser.add_argument("--max_len", type=int, default=128)
    parser.add_argument("--batch_size", type=int, default=None)
    parser.add_argument("--num_workers", type=int, default=8)
    parser.add_argument("--epochs", type=int, default=40)
    parser.add_argument("--max_steps", type=int, default=-1)
    parser.add_argument("--lr", type=float, default=None)
    parser.add_argument("--warmup_steps", type=int, default=None)
    parser.add_argument("--ema_decay", type=float, default=0.9999)
    parser.add_argument("--cbi_weight", type=float, default=2.0)
    parser.add_argument("--use_aromatic_cbi", action="store_true")
    parser.add_argument("--aromatic_cbi_weight", type=float, default=3.0)
    parser.add_argument("--hidden_size", type=int, default=None)
    parser.add_argument("--num_layers", type=int, default=None)
    parser.add_argument("--num_heads", type=int, default=None)
    parser.add_argument("--intermediate", type=int, default=None)
    parser.add_argument("--checkpoint_dir", default="csdnet_checkpoints_conditioned")
    parser.add_argument("--run_name", default="csdnet-conditioned")
    parser.add_argument("--disable_validation", action="store_true")
    parser.add_argument("--precision", default="bf16-mixed")
    parser.add_argument("--devices", type=int, default=1)
    parser.add_argument("--strategy", default="auto")
    parser.add_argument("--resume_from", default=None)
    parser.add_argument("--partial_cond", action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument("--all_prop_mask_prob", type=float, default=0.1)
    parser.add_argument("--full_prop_keep_prob", type=float, default=0.1)
    parser.add_argument("--per_prop_mask_prob", type=float, default=0.5)
    parser.add_argument("--drop_cond_prob", type=float, default=0.1)
    return apply_model_size(parser.parse_args())


def main():
    args = parse_args()
    with open(args.vocab, "rb") as f:
        tk = SMILESTokenizer(pickle.load(f))

    try:
        from datasets import load_from_disk
    except ImportError as exc:
        raise SystemExit("Missing datasets dependency.") from exc

    hf_ds = load_from_disk(args.data_dir).shuffle(seed=42)
    dataset_kwargs = dict(
        max_len=args.max_len,
        partial_cond=args.partial_cond,
        all_prop_mask_prob=args.all_prop_mask_prob,
        full_prop_keep_prob=args.full_prop_keep_prob,
        per_prop_mask_prob=args.per_prop_mask_prob,
        use_aromatic_cbi=args.use_aromatic_cbi,
    )

    if args.disable_validation:
        train_ds = CSDNetPropertyDataset(hf_ds, tk, **dataset_kwargs)
        val_ds = None
    else:
        val_size = min(10000, len(hf_ds) // 20)
        train_ds = CSDNetPropertyDataset(hf_ds.select(range(val_size, len(hf_ds))), tk, **dataset_kwargs)
        val_ds = CSDNetPropertyDataset(hf_ds.select(range(val_size)), tk, **dataset_kwargs)

    train_loader = DataLoader(
        train_ds,
        batch_size=args.batch_size,
        shuffle=True,
        num_workers=args.num_workers,
        collate_fn=make_collate_fn(
            tk,
            include_cond=True,
            use_aromatic_cbi=args.use_aromatic_cbi,
        ),
        pin_memory=True,
    )
    val_loader = None
    if val_ds is not None:
        val_loader = DataLoader(
            val_ds,
            batch_size=args.batch_size,
            shuffle=False,
            num_workers=args.num_workers,
            collate_fn=make_collate_fn(
                tk,
                include_cond=True,
                use_aromatic_cbi=args.use_aromatic_cbi,
            ),
            pin_memory=True,
        )

    cond_dim = len(PROP_NAMES) * 2 if args.partial_cond else len(PROP_NAMES)
    model = CSDNetLightningModule(
        vocab_size=tk.vocab_size,
        pad_id=tk.pad_id,
        mask_id=tk.mask_id,
        bos_id=tk.bos_id,
        eos_id=tk.eos_id,
        unk_id=tk.unk_id,
        scaffold_ids=tk.scaffold_ids,
        aromatic_ids=tk.aromatic_ids,
        cond_dim=cond_dim,
        cbi_weight=args.cbi_weight,
        use_aromatic_cbi=args.use_aromatic_cbi,
        aromatic_cbi_weight=args.aromatic_cbi_weight,
        lr=args.lr,
        warmup_steps=args.warmup_steps,
        ema_decay=args.ema_decay,
        hidden_size=args.hidden_size,
        num_layers=args.num_layers,
        num_heads=args.num_heads,
        intermediate=args.intermediate,
        drop_cond_prob=args.drop_cond_prob,
    )
    n_params = sum(p.numel() for p in model.backbone.parameters())
    print(f"CSDNet conditioned model size: {n_params / 1e6:.2f}M, cond_dim={cond_dim}")

    if args.disable_validation:
        checkpoint_callback = ModelCheckpoint(
            dirpath=args.checkpoint_dir,
            filename=args.run_name + "-{epoch:02d}-{step}",
            save_top_k=0,
            save_last=True,
        )
    else:
        checkpoint_callback = ModelCheckpoint(
            dirpath=args.checkpoint_dir,
            filename=args.run_name + "-{epoch:02d}-{val_loss:.4f}",
            monitor="val_loss",
            mode="min",
            save_top_k=3,
            save_last=True,
        )

    trainer_kwargs = dict(
        max_epochs=args.epochs,
        max_steps=args.max_steps,
        accelerator="gpu",
        devices=args.devices,
        precision=args.precision,
        callbacks=[checkpoint_callback, LearningRateMonitor(logging_interval="step")],
        log_every_n_steps=50,
        num_sanity_val_steps=0 if args.disable_validation else 2,
        limit_val_batches=0 if args.disable_validation else 1.0,
    )
    if args.strategy != "auto":
        trainer_kwargs["strategy"] = args.strategy
    if not args.disable_validation:
        trainer_kwargs["val_check_interval"] = 0.25
    trainer = L.Trainer(**trainer_kwargs)

    trainer.fit(model, train_loader, val_loader, ckpt_path=args.resume_from)
    os.makedirs(args.checkpoint_dir, exist_ok=True)
    final_ckpt = os.path.join(args.checkpoint_dir, "last.ckpt")
    trainer.save_checkpoint(final_ckpt)
    print(f"Final checkpoint saved: {final_ckpt}")


if __name__ == "__main__":
    main()
