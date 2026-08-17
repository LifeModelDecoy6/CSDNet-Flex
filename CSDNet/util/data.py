import random
import time

import torch
from torch.utils.data import Dataset, IterableDataset, get_worker_info

from CSDNet.model.lightning_module import PROP_NAMES, PROP_SCALES
from CSDNet.util.hf_smiles import extract_smiles_from_row, load_dataset_with_optional_token
from CSDNet.util.tokenizer import tokenize_smiles


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


class CSDNetPropertyDataset(Dataset):
    def __init__(
        self,
        hf_ds,
        tk,
        max_len=128,
        partial_cond=True,
        all_prop_mask_prob=0.1,
        full_prop_keep_prob=0.1,
        per_prop_mask_prob=0.5,
        use_aromatic_cbi=True,
    ):
        self.ds = hf_ds
        self.tk = tk
        self.max_len = max_len
        self.partial_cond = partial_cond
        self.all_prop_mask_prob = all_prop_mask_prob
        self.full_prop_keep_prob = full_prop_keep_prob
        self.per_prop_mask_prob = per_prop_mask_prob
        self.use_aromatic_cbi = use_aromatic_cbi
        if self.all_prop_mask_prob < 0 or self.full_prop_keep_prob < 0:
            raise ValueError("Condition-drop probabilities must be non-negative")
        if self.all_prop_mask_prob + self.full_prop_keep_prob > 1:
            raise ValueError("all_prop_mask_prob + full_prop_keep_prob must be <= 1")
        if not 0 <= self.per_prop_mask_prob <= 1:
            raise ValueError("per_prop_mask_prob must be in [0, 1]")

    def __len__(self):
        return len(self.ds)

    def _sample_property_mask(self):
        n_props = len(PROP_NAMES)
        draw = random.random()
        if draw < self.all_prop_mask_prob:
            return torch.zeros(n_props, dtype=torch.float)
        if draw < self.all_prop_mask_prob + self.full_prop_keep_prob:
            return torch.ones(n_props, dtype=torch.float)

        while True:
            mask = torch.tensor(
                [0.0 if random.random() < self.per_prop_mask_prob else 1.0 for _ in range(n_props)],
                dtype=torch.float,
            )
            n_kept = int(mask.sum().item())
            if 0 < n_kept < n_props:
                return mask

    def __getitem__(self, idx):
        row = self.ds[idx]
        smi = row.get("text", row.get("smiles", ""))
        values = torch.tensor(
            [
                float(row.get("qed", 0.5)),
                float(row.get("logp", 0.0)) / PROP_SCALES["logp"],
                float(row.get("sa", 1.0)) / PROP_SCALES["sa"],
                float(row.get("tpsa", 0.0)) / PROP_SCALES["tpsa"],
                float(row.get("mw", 0.0)) / PROP_SCALES["mw"],
            ],
            dtype=torch.float,
        )
        if self.partial_cond:
            mask = self._sample_property_mask()
            cond = torch.cat([values * mask, mask], dim=0)
        else:
            cond = values

        item = {
            "input_ids": torch.tensor(self.tk.encode(smi, self.max_len), dtype=torch.long),
            "cond": cond,
        }
        if self.use_aromatic_cbi:
            item["aromatic_mask"] = torch.tensor(
                self.tk.aromatic_context_mask(smi, self.max_len),
                dtype=torch.bool,
            )
        return item


class CSDNetDeNovoDataset(Dataset):
    def __init__(self, hf_ds, tk, max_len=128, use_aromatic_cbi=False):
        self.ds = hf_ds
        self.tk = tk
        self.max_len = max_len
        self.use_aromatic_cbi = use_aromatic_cbi

    def __len__(self):
        return len(self.ds)

    def __getitem__(self, idx):
        row = self.ds[idx]
        smi = row.get("text", row.get("smiles", ""))
        item = {
            "input_ids": torch.tensor(self.tk.encode(smi, self.max_len), dtype=torch.long),
            "cond": torch.empty(0, dtype=torch.float),
        }
        if self.use_aromatic_cbi:
            item["aromatic_mask"] = torch.tensor(
                self.tk.aromatic_context_mask(smi, self.max_len),
                dtype=torch.bool,
            )
        return item


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

    def _make_stream(self):
        ds = load_dataset_with_optional_token(
            self.dataset_name,
            split=self.split,
            streaming=True,
            token=self.token,
        )
        if self.shuffle_buffer > 0:
            ds = ds.shuffle(buffer_size=self.shuffle_buffer, seed=self.seed)
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
        while True:
            try:
                stream = self._make_stream()

                if hasattr(stream, "shard"):
                    iterator = stream.shard(num_shards=num_shards, index=shard_index)
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
                return
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


def make_collate_fn(
    tk,
    include_cond=True,
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
        }
        if include_cond:
            conds = [b["cond"] for b in batch]
            if conds and conds[0].numel() == 0:
                out["cond"] = torch.empty(input_ids.size(0), 0, dtype=torch.float)
            else:
                out["cond"] = torch.stack(conds)
        if use_aromatic_cbi:
            aromatic_mask = torch.stack([b["aromatic_mask"] for b in batch])
            out["aromatic_mask"] = aromatic_mask[
                :, :input_ids.size(1)
            ].contiguous()
        return out

    return collate_fn
