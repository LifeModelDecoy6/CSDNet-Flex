#!/usr/bin/env python
import argparse
import os
import pickle
import time

import lightning as L
import torch
from torch.utils.data import DataLoader, IterableDataset, get_worker_info
from lightning.pytorch.callbacks import LearningRateMonitor, ModelCheckpoint, Timer

from CSDNet.config import parse_args_with_yaml_config
from CSDNet.model.lightning_module import CSDNetLightningModule
from csdnet_tokenizer import SMILESTokenizer, tokenize_smiles
from csdnet_hf_smiles import (
    extract_smiles_from_row,
    load_dataset_with_optional_token,
    resolve_hf_token,
)


class PerRunTimer(Timer):
    """Apply the wall-clock limit independently to each resumed allocation."""

    def load_state_dict(self, state_dict):
        # A resumed PBS job receives a fresh wall-clock allowance. Optimizer,
        # scheduler, model, EMA, and global-step state still come from ckpt.
        self._offset = 0


def _is_transient_stream_error(exc):
    text = f"{type(exc).__module__}.{type(exc).__name__}: {exc}".lower()
    needles = (
        "readtimeout",
        "connecttimeout",
        "connectionerror",
        "chunkedencodingerror",
        "protocolerror",
        "temporarily unavailable",
        "connection reset",
        "remote end closed connection",
        "httpsconnectionpool",
    )
    return any(needle in text for needle in needles)


class HFStreamingSMILESDataset(IterableDataset):
    def __init__(
        self,
        dataset_name,
        split,
        smiles_col,
        safe_col,
        tokenizer,
        max_len,
        token,
        allow_safe_decode=True,
        shuffle_buffer=10000,
        seed=42,
        skip_unknown=True,
        skip_long=True,
        use_aromatic_cbi=False,
        stream_retries=20,
        stream_retry_backoff=30.0,
        repeat_stream=True,
    ):
        self.dataset_name = dataset_name
        self.split = split
        self.smiles_col = smiles_col
        self.safe_col = safe_col
        self.tk = tokenizer
        self.max_len = max_len
        self.token = token
        self.allow_safe_decode = allow_safe_decode
        self.shuffle_buffer = shuffle_buffer
        self.seed = seed
        self.skip_unknown = skip_unknown
        self.skip_long = skip_long
        self.use_aromatic_cbi = use_aromatic_cbi
        self.stream_retries = stream_retries
        self.stream_retry_backoff = stream_retry_backoff
        self.repeat_stream = repeat_stream

    def _make_stream(self, cycle=0):
        ds = load_dataset_with_optional_token(
            self.dataset_name,
            split=self.split,
            streaming=True,
            token=self.token,
        )
        if self.shuffle_buffer > 0:
            ds = ds.shuffle(
                buffer_size=self.shuffle_buffer,
                seed=self.seed + cycle,
            )
        return ds

    @staticmethod
    def _distributed_info():
        if torch.distributed.is_available() and torch.distributed.is_initialized():
            return torch.distributed.get_rank(), torch.distributed.get_world_size()
        return 0, 1

    def __iter__(self):
        worker = get_worker_info()
        rank, world_size = self._distributed_info()
        worker_id = worker.id if worker is not None else 0
        num_workers = worker.num_workers if worker is not None else 1
        num_shards = world_size * num_workers
        shard_index = rank * num_workers + worker_id

        unk_id = self.tk.unk_id

        attempt = 0
        cycle = 0
        while True:
            try:
                stream = self._make_stream(cycle=cycle)

                # Streaming datasets are iterable, so DDP ranks must be explicitly
                # sharded; otherwise all GPUs would see the same rows.
                if hasattr(stream, "shard"):
                    stream = stream.shard(num_shards=num_shards, index=shard_index)
                    iterator = stream
                else:
                    iterator = (
                        row for idx, row in enumerate(stream)
                        if idx % num_shards == shard_index
                    )

                for row in iterator:
                    smi = extract_smiles_from_row(
                        row,
                        smiles_col=self.smiles_col,
                        safe_col=self.safe_col,
                        allow_safe_decode=self.allow_safe_decode,
                    )
                    if not smi:
                        continue

                    tokens = tokenize_smiles(smi)
                    if self.skip_long and len(tokens) + 2 > self.max_len:
                        continue
                    if self.skip_unknown and any(tok not in self.tk.vocab for tok in tokens):
                        continue

                    input_ids = self.tk.encode(smi, self.max_len)
                    if self.skip_unknown and unk_id != -1 and unk_id in input_ids:
                        continue

                    item = {
                        "input_ids": torch.tensor(input_ids, dtype=torch.long),
                        "cond": torch.empty(0, dtype=torch.float),
                    }
                    if self.use_aromatic_cbi:
                        item["aromatic_mask"] = torch.tensor(
                            self.tk.aromatic_context_mask(smi, self.max_len),
                            dtype=torch.bool,
                        )
                    yield item

                if not self.repeat_stream:
                    return

                # Different DDP shards can contain different numbers of rows
                # after validity and length filtering. Keeping every worker
                # infinite avoids ranks entering epoch-end collectives while
                # another rank is still reducing its final training batch.
                cycle += 1
                attempt = 0
            except Exception as exc:
                if (not _is_transient_stream_error(exc)) or attempt >= self.stream_retries:
                    raise
                attempt += 1
                sleep_s = self.stream_retry_backoff * min(8, 2 ** (attempt - 1))
                print(
                    "[HF streaming] transient read error on "
                    f"rank={rank}, worker={worker_id}; retry "
                    f"{attempt}/{self.stream_retries} in {sleep_s:.1f}s: "
                    f"{type(exc).__name__}: {exc}",
                    flush=True,
                )
                time.sleep(sleep_s)


def make_streaming_collate_fn(
    tk,
    use_aromatic_cbi=False,
    dynamic_padding=False,
    pad_to_multiple_of=1,
):
    if pad_to_multiple_of < 1:
        raise ValueError("pad_to_multiple_of must be positive.")

    def collate_fn(batch):
        input_ids = torch.stack([b["input_ids"] for b in batch])
        if dynamic_padding:
            active_lengths = input_ids.ne(tk.pad_id).sum(dim=1)
            batch_length = max(2, int(active_lengths.max().item()))
            batch_length = min(
                input_ids.size(1),
                (
                    (batch_length + pad_to_multiple_of - 1)
                    // pad_to_multiple_of
                )
                * pad_to_multiple_of,
            )
            input_ids = input_ids[:, :batch_length].contiguous()
        out = {
            "input_ids": input_ids,
            "attention_mask": (input_ids != tk.pad_id).long(),
            "cond": torch.empty(input_ids.size(0), 0, dtype=torch.float),
        }
        if use_aromatic_cbi:
            aromatic_mask = torch.stack([b["aromatic_mask"] for b in batch])
            out["aromatic_mask"] = aromatic_mask[
                :, :input_ids.size(1)
            ].contiguous()
        return out

    return collate_fn


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--config",
        type=str,
        default=None,
        help="Flat YAML training config. Explicit CLI options override it.",
    )
    parser.add_argument("--hf_dataset", type=str, default="datamol-io/safe-gpt")
    parser.add_argument("--hf_split", type=str, default="train")
    parser.add_argument("--smiles_col", type=str, default="auto")
    parser.add_argument("--safe_col", type=str, default="safe")
    parser.add_argument("--disable_safe_decode", action="store_true",
                        help="auto 模式下如果没有 SMILES 列，则默认尝试从 SAFE 列 decode；此开关可关闭 fallback")
    parser.add_argument("--hf_token_env", type=str, default="HF_TOKEN")
    parser.add_argument("--hf_token", type=str, default=None)
    parser.add_argument("--vocab", type=str, default="csdnet_vocab.pkl")
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
    parser.add_argument(
        "--max_position_embeddings",
        type=int,
        default=256,
        help="Size of the absolute ESM position-embedding table.",
    )
    parser.add_argument("--batch_size", type=int, default=256)
    parser.add_argument("--num_workers", type=int, default=4)
    parser.add_argument("--shuffle_buffer", type=int, default=10000)
    parser.add_argument("--stream_retries", type=int, default=20)
    parser.add_argument("--stream_retry_backoff", type=float, default=30.0)
    parser.add_argument(
        "--repeat_stream",
        action=argparse.BooleanOptionalAction,
        default=True,
        help=(
            "Repeat finite streaming shards until max_steps is reached. "
            "This must remain enabled for DDP because filtering can make "
            "rank-local shards end at different batch counts."
        ),
    )
    parser.add_argument("--max_steps", type=int, default=80000)
    parser.add_argument(
        "--max_time",
        type=str,
        default=None,
        help=(
            "Optional Lightning wall-clock limit in DD:HH:MM:SS format. "
            "Set it below the scheduler walltime so final checkpointing can "
            "finish before PBS terminates the job."
        ),
    )
    parser.add_argument("--lr", type=float, default=5e-4)
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
    cbi_group.add_argument("--enable_cbi", action="store_true", dest="use_cbi")
    cbi_group.add_argument("--disable_cbi", action="store_false", dest="use_cbi")
    parser.set_defaults(use_cbi=True)
    parser.add_argument("--use_aromatic_cbi", action="store_true")
    parser.add_argument("--aromatic_cbi_weight", type=float, default=3.0)
    parser.add_argument(
        "--normalized_aromatic_cbi",
        action="store_true",
        help="Preserve each molecule's total loss mass while emphasizing aromatic positions.",
    )
    parser.add_argument(
        "--normalized_cbi",
        action="store_true",
        help="Normalize the combined CBI and AroCBI allocation per molecule.",
    )
    parser.add_argument("--hidden_size", type=int, default=256)
    parser.add_argument("--num_layers", type=int, default=8)
    parser.add_argument("--num_heads", type=int, default=8)
    parser.add_argument("--intermediate", type=int, default=1024)
    parser.add_argument("--hidden_dropout_prob", type=float, default=0.1)
    parser.add_argument(
        "--attention_probs_dropout_prob",
        type=float,
        default=0.1,
    )
    parser.add_argument("--layer_norm_eps", type=float, default=1e-12)
    parser.add_argument("--initializer_range", type=float, default=0.02)
    parser.add_argument("--position_embedding_type", default="absolute")
    parser.add_argument(
        "--gradient_checkpointing",
        action=argparse.BooleanOptionalAction,
        default=False,
    )
    parser.add_argument("--checkpoint_dir", type=str, default="csdnet_checkpoints_6m_safegpt_80k")
    parser.add_argument(
        "--checkpoint_every_n_steps",
        type=int,
        default=10000,
        help="Save streaming-training checkpoints by step. Set <=0 to keep only the final manual save.",
    )
    parser.add_argument("--run_name", type=str, default="csdnet-6m-safegpt-80k")
    parser.add_argument("--precision", type=str, default="bf16-mixed")
    parser.add_argument("--devices", type=int, default=1)
    parser.add_argument("--strategy", type=str, default="auto")
    parser.add_argument("--accumulate_grad_batches", type=int, default=1)
    parser.add_argument("--gradient_clip_val", type=float, default=0.0)
    parser.add_argument(
        "--resume_from",
        type=str,
        default=None,
        help="Resume training from a Lightning checkpoint.",
    )
    parser.add_argument("--skip_unknown", action="store_true", default=True)
    parser.add_argument("--keep_unknown", action="store_false", dest="skip_unknown")
    parser.add_argument("--skip_long", action="store_true", default=True)
    parser.add_argument("--keep_long_truncated", action="store_false", dest="skip_long")
    parser.add_argument("--seed", type=int, default=42)
    args = parse_args_with_yaml_config(parser)

    L.seed_everything(args.seed, workers=True)
    token = resolve_hf_token(args.hf_token, args.hf_token_env, required=True)

    print("=" * 50)
    print("🚀 加载 Tokenizer 与 SAFE-GPT streaming 配置...")
    with open(args.vocab, "rb") as f:
        tk = SMILESTokenizer(pickle.load(f))
    required_positions = args.max_len + tk.pad_id + 1
    if args.max_position_embeddings < required_positions:
        raise ValueError(
            "max_position_embeddings must be at least "
            f"{required_positions} for max_len={args.max_len} and "
            f"pad_id={tk.pad_id}; got {args.max_position_embeddings}."
        )
    print(
        f"✔️ vocab={tk.vocab_size}, dataset={args.hf_dataset}, "
        f"split={args.hf_split}, col={args.smiles_col}, "
        f"safe_fallback={not args.disable_safe_decode}"
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
        repeat_stream=args.repeat_stream,
    )
    train_loader = DataLoader(
        train_ds,
        batch_size=args.batch_size,
        num_workers=args.num_workers,
        collate_fn=make_streaming_collate_fn(
            tk,
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
        use_cbi=args.use_cbi,
        cbi_weight=args.cbi_weight,
        use_aromatic_cbi=args.use_aromatic_cbi,
        aromatic_cbi_weight=args.aromatic_cbi_weight,
        normalized_aromatic_cbi=args.normalized_aromatic_cbi,
        normalized_cbi=args.normalized_cbi,
        lr=args.lr,
        warmup_steps=args.warmup_steps,
        lr_schedule=args.lr_schedule,
        weight_decay=args.weight_decay,
        adam_beta1=args.adam_beta1,
        adam_beta2=args.adam_beta2,
        adam_eps=args.adam_eps,
        ema_decay=args.ema_decay,
        hidden_size=args.hidden_size,
        num_layers=args.num_layers,
        num_heads=args.num_heads,
        intermediate=args.intermediate,
        max_position_embeddings=args.max_position_embeddings,
        hidden_dropout_prob=args.hidden_dropout_prob,
        attention_probs_dropout_prob=args.attention_probs_dropout_prob,
        layer_norm_eps=args.layer_norm_eps,
        initializer_range=args.initializer_range,
        position_embedding_type=args.position_embedding_type,
        gradient_checkpointing=args.gradient_checkpointing,
        drop_cond_prob=0.0,
    )
    n_params = sum(p.numel() for p in model.backbone.parameters())
    print(f"✔️ Backbone 参数量: {n_params / 1e6:.2f}M")
    print(
        f"✔️ CBI: scaffold_weight={args.cbi_weight}, "
        f"aromatic_cbi={'on' if args.use_aromatic_cbi else 'off'}, "
        f"aromatic_weight={args.aromatic_cbi_weight}, "
        f"normalization="
        f"{'all_cbi' if args.normalized_cbi else ('aromatic_only' if args.normalized_aromatic_cbi else 'off')}"
    )
    print(
        "✔️ Sequence layout: "
        f"global_max_len={args.max_len}, "
        f"dynamic_padding={args.dynamic_padding}, "
        f"pad_to_multiple_of={args.pad_to_multiple_of}, "
        f"repeat_stream={args.repeat_stream}, "
        f"position_embeddings={args.max_position_embeddings}"
    )
    print(
        "✔️ GenMol alignment: "
        f"H={args.hidden_size}, L={args.num_layers}, "
        f"heads={args.num_heads}, FFN={args.intermediate}, "
        f"dropout={args.hidden_dropout_prob}/"
        f"{args.attention_probs_dropout_prob}, "
        f"layer_norm_eps={args.layer_norm_eps:g}, "
        f"position={args.position_embedding_type}, "
        f"global_batch="
        f"{args.batch_size * args.devices * args.accumulate_grad_batches}, "
        f"optimizer=AdamW(lr={args.lr:g}, wd={args.weight_decay:g}, "
        f"betas=({args.adam_beta1:g},{args.adam_beta2:g})), "
        f"schedule={args.lr_schedule}, warmup={args.warmup_steps}"
    )

    lr_monitor = LearningRateMonitor(logging_interval="step")
    callbacks = [lr_monitor]
    if args.max_time:
        callbacks.append(PerRunTimer(duration=args.max_time))
    if args.checkpoint_every_n_steps > 0:
        checkpoint_callback = ModelCheckpoint(
            dirpath=args.checkpoint_dir,
            filename=args.run_name + "-{step}",
            save_top_k=0,
            save_last=True,
            every_n_train_steps=args.checkpoint_every_n_steps,
            save_on_train_epoch_end=False,
        )
        callbacks.insert(0, checkpoint_callback)

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
        accumulate_grad_batches=args.accumulate_grad_batches,
        gradient_clip_val=args.gradient_clip_val,
    )
    if args.strategy != "auto":
        trainer_kwargs["strategy"] = args.strategy
    trainer = L.Trainer(**trainer_kwargs)

    print("\n🔥 开始 SAFE-GPT streaming 训练")
    print("=" * 50)
    trainer.fit(model, train_loader, ckpt_path=args.resume_from)
    os.makedirs(args.checkpoint_dir, exist_ok=True)
    final_ckpt = os.path.join(args.checkpoint_dir, "last.ckpt")
    trainer.save_checkpoint(final_ckpt)
    print(f"✔️ 最终 checkpoint 已保存: {final_ckpt}")


if __name__ == "__main__":
    main()
