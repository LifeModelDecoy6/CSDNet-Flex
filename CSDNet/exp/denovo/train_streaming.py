#!/usr/bin/env python
import argparse
import os
import pickle

import lightning as L
from lightning.pytorch.callbacks import LearningRateMonitor, ModelCheckpoint
from torch.utils.data import DataLoader

from CSDNet.config import get_model_config
from CSDNet.model.lightning_module import CSDNetLightningModule
from CSDNet.util.data import HFStreamingSMILESDataset, make_collate_fn
from CSDNet.util.hf_smiles import resolve_hf_token
from CSDNet.util.tokenizer import SMILESTokenizer


def apply_model_size(args):
    cfg = get_model_config(args.model_size)
    for key in ("hidden_size", "num_layers", "num_heads", "intermediate", "lr", "warmup_steps", "batch_size"):
        if getattr(args, key) is None:
            setattr(args, key, cfg[key])
    return args


def parse_args():
    parser = argparse.ArgumentParser(description="Train unconditional CSDNet from HF streaming SMILES/SAFE data.")
    parser.add_argument("--model_size", choices=["6m", "30m", "90m"], default="6m")
    parser.add_argument("--hf_dataset", default="datamol-io/safe-gpt")
    parser.add_argument("--hf_split", default="train")
    parser.add_argument("--smiles_col", default="auto")
    parser.add_argument("--safe_col", default="safe")
    parser.add_argument("--disable_safe_decode", action="store_true")
    parser.add_argument("--hf_token_env", default="HF_TOKEN")
    parser.add_argument("--hf_token", default=None)
    parser.add_argument("--vocab", default="csdnet_vocab.pkl")
    parser.add_argument("--max_len", type=int, default=128)
    parser.add_argument(
        "--dynamic_padding",
        action="store_true",
        help="Trim each batch to its longest non-padding sequence.",
    )
    parser.add_argument(
        "--pad_to_multiple_of",
        type=int,
        default=1,
        help="Round the dynamic batch length up to this multiple; 1 is exact.",
    )
    parser.add_argument("--max_position_embeddings", type=int, default=256)
    parser.add_argument("--batch_size", type=int, default=None)
    parser.add_argument("--num_workers", type=int, default=4)
    parser.add_argument("--shuffle_buffer", type=int, default=10000)
    parser.add_argument("--stream_retries", type=int, default=20)
    parser.add_argument("--stream_retry_backoff", type=float, default=30.0)
    parser.add_argument("--max_steps", type=int, default=80000)
    parser.add_argument("--lr", type=float, default=None)
    parser.add_argument("--warmup_steps", type=int, default=None)
    parser.add_argument("--ema_decay", type=float, default=0.9999)
    parser.add_argument("--cbi_weight", type=float, default=2.0)
    parser.add_argument("--use_aromatic_cbi", action="store_true")
    parser.add_argument("--aromatic_cbi_weight", type=float, default=3.0)
    parser.add_argument(
        "--normalized_aromatic_cbi",
        action="store_true",
        help="Preserve per-molecule loss mass while emphasizing aromatic positions.",
    )
    parser.add_argument(
        "--normalized_cbi",
        action="store_true",
        help="Normalize the combined CBI and AroCBI allocation per molecule.",
    )
    parser.add_argument("--hidden_size", type=int, default=None)
    parser.add_argument("--num_layers", type=int, default=None)
    parser.add_argument("--num_heads", type=int, default=None)
    parser.add_argument("--intermediate", type=int, default=None)
    parser.add_argument("--checkpoint_dir", default="csdnet_checkpoints_6m_safegpt")
    parser.add_argument("--checkpoint_every_n_steps", type=int, default=10000)
    parser.add_argument("--run_name", default="csdnet-6m-safegpt")
    parser.add_argument("--precision", default="bf16-mixed")
    parser.add_argument("--devices", type=int, default=1)
    parser.add_argument("--strategy", default="auto")
    parser.add_argument(
        "--resume_from",
        default=None,
        help="Resume training from a Lightning checkpoint.",
    )
    parser.add_argument("--skip_unknown", action="store_true", default=True)
    parser.add_argument("--keep_unknown", action="store_false", dest="skip_unknown")
    parser.add_argument("--skip_long", action="store_true", default=True)
    parser.add_argument("--keep_long_truncated", action="store_false", dest="skip_long")
    parser.add_argument("--seed", type=int, default=42)
    return apply_model_size(parser.parse_args())


def main():
    args = parse_args()
    L.seed_everything(args.seed, workers=True)
    token = resolve_hf_token(args.hf_token, args.hf_token_env, required=True)

    with open(args.vocab, "rb") as f:
        tk = SMILESTokenizer(pickle.load(f))
    required_positions = args.max_len + tk.pad_id + 1
    if args.max_position_embeddings < required_positions:
        raise ValueError(
            "max_position_embeddings must be at least "
            f"{required_positions} for max_len={args.max_len} and "
            f"pad_id={tk.pad_id}; got {args.max_position_embeddings}."
        )

    train_ds = HFStreamingSMILESDataset(
        dataset_name=args.hf_dataset,
        split=args.hf_split,
        smiles_col=args.smiles_col,
        safe_col=args.safe_col,
        tokenizer=tk,
        max_len=args.max_len,
        token=token,
        allow_safe_decode=not args.disable_safe_decode,
        shuffle_buffer=args.shuffle_buffer,
        seed=args.seed,
        skip_unknown=args.skip_unknown,
        skip_long=args.skip_long,
        use_aromatic_cbi=args.use_aromatic_cbi,
        stream_retries=args.stream_retries,
        stream_retry_backoff=args.stream_retry_backoff,
    )
    train_loader = DataLoader(
        train_ds,
        batch_size=args.batch_size,
        num_workers=args.num_workers,
        collate_fn=make_collate_fn(
            tk,
            include_cond=True,
            use_aromatic_cbi=args.use_aromatic_cbi,
            dynamic_padding=args.dynamic_padding,
            pad_to_multiple_of=args.pad_to_multiple_of,
        ),
        pin_memory=True,
    )

    model = CSDNetLightningModule(
        vocab_size=tk.vocab_size,
        pad_id=tk.pad_id,
        mask_id=tk.mask_id,
        bos_id=tk.bos_id,
        eos_id=tk.eos_id,
        unk_id=tk.unk_id,
        scaffold_ids=tk.scaffold_ids,
        aromatic_ids=tk.aromatic_ids,
        cond_dim=0,
        cbi_weight=args.cbi_weight,
        use_aromatic_cbi=args.use_aromatic_cbi,
        aromatic_cbi_weight=args.aromatic_cbi_weight,
        normalized_aromatic_cbi=args.normalized_aromatic_cbi,
        normalized_cbi=args.normalized_cbi,
        lr=args.lr,
        warmup_steps=args.warmup_steps,
        ema_decay=args.ema_decay,
        hidden_size=args.hidden_size,
        num_layers=args.num_layers,
        num_heads=args.num_heads,
        intermediate=args.intermediate,
        max_position_embeddings=args.max_position_embeddings,
        drop_cond_prob=0.0,
    )
    n_params = sum(p.numel() for p in model.backbone.parameters())
    print(f"CSDNet streaming model size: {n_params / 1e6:.2f}M")
    print(
        "Chemical CBI normalization: "
        f"{'all_cbi' if args.normalized_cbi else ('aromatic_only' if args.normalized_aromatic_cbi else 'off')} "
        f"(aromatic_enabled={args.use_aromatic_cbi}, "
        f"aromatic_weight={args.aromatic_cbi_weight})"
    )
    print(
        "Sequence layout: "
        f"global_max_len={args.max_len}, "
        f"dynamic_padding={args.dynamic_padding}, "
        f"pad_to_multiple_of={args.pad_to_multiple_of}, "
        f"position_embeddings={args.max_position_embeddings}"
    )

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
                save_on_train_epoch_end=False,
            ),
        )

    trainer_kwargs = dict(
        max_epochs=999,
        max_steps=args.max_steps,
        accelerator="gpu",
        devices=args.devices,
        precision=args.precision,
        callbacks=callbacks,
        log_every_n_steps=50,
        num_sanity_val_steps=0,
        limit_val_batches=0,
    )
    if args.strategy != "auto":
        trainer_kwargs["strategy"] = args.strategy
    trainer = L.Trainer(**trainer_kwargs)

    trainer.fit(model, train_loader, ckpt_path=args.resume_from)
    os.makedirs(args.checkpoint_dir, exist_ok=True)
    final_ckpt = os.path.join(args.checkpoint_dir, "last.ckpt")
    trainer.save_checkpoint(final_ckpt)
    print(f"Final checkpoint saved: {final_ckpt}")


if __name__ == "__main__":
    main()
