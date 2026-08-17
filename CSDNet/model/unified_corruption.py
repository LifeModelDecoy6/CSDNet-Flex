import math
import random
from dataclasses import dataclass

import torch


MODE_MDM = 0
MODE_VARIABLE = 1
MODE_FRAGMENT = 2
MODE_REFINE = 3
MODE_NAMES = ("mdm", "variable", "fragment", "refine")


@dataclass
class _StateToken:
    token_id: int
    clean_index: int | None
    target_id: int = -100
    token_weight: float = 0.0
    delete_label: float = -100.0
    aromatic: bool = False


class UnifiedCorruptionCollator:
    """Build aligned molecular edit states for a single shared forward pass.

    Four row types are mixed in every batch:

    * masked diffusion on the original atom-tokenized sequence;
    * variable-length projection with removed, masked and visible tokens;
    * known- and unknown-length fragment infilling across internal, terminal,
      long, and multi-span deletions;
    * all-position refinement with substitutions, masks, and spurious tokens.

    The clean alignment is used only to construct labels.  The model receives
    the corrupted sequence and an independently sampled diffusion time.
    """

    def __init__(
        self,
        tokenizer,
        max_len=256,
        max_gap_count=8,
        mode_probabilities=(0.35, 0.25, 0.25, 0.15),
        dynamic_padding=True,
        pad_to_multiple_of=8,
        fragment_unknown_length_probability=0.60,
        fragment_long_span_probability=0.20,
        fragment_multi_span_probability=0.25,
        refine_clean_weight=0.10,
        seed=42,
    ):
        self.tk = tokenizer
        self.max_len = int(max_len)
        self.max_gap_count = int(max_gap_count)
        self.dynamic_padding = bool(dynamic_padding)
        self.pad_to_multiple_of = int(pad_to_multiple_of)
        self.fragment_unknown_length_probability = float(
            fragment_unknown_length_probability
        )
        self.fragment_long_span_probability = float(
            fragment_long_span_probability
        )
        self.fragment_multi_span_probability = float(
            fragment_multi_span_probability
        )
        self.refine_clean_weight = float(refine_clean_weight)
        self.seed = int(seed)
        probabilities = torch.tensor(mode_probabilities, dtype=torch.float64)
        if probabilities.numel() != 4 or (probabilities < 0).any():
            raise ValueError("mode_probabilities must contain four non-negative values.")
        if float(probabilities.sum()) <= 0:
            raise ValueError("At least one mode probability must be positive.")
        self.mode_probabilities = (
            probabilities / probabilities.sum()
        ).tolist()
        if self.max_len < 4:
            raise ValueError("max_len must be at least four.")
        if self.max_gap_count < 1:
            raise ValueError("max_gap_count must be positive.")
        if self.pad_to_multiple_of < 1:
            raise ValueError("pad_to_multiple_of must be positive.")

        special = {
            self.tk.pad_id,
            self.tk.mask_id,
            self.tk.bos_id,
            self.tk.eos_id,
            self.tk.unk_id,
        }
        self.candidate_token_ids = [
            idx for idx in range(self.tk.vocab_size) if idx not in special
        ]
        if not self.candidate_token_ids:
            raise ValueError("Tokenizer has no ordinary candidate tokens.")

    def __call__(self, batch):
        # DataLoader workers are independently seeded by Lightning.  Deriving a
        # local generator avoids mutating the application's global RNG state.
        local_seed = torch.randint(0, 2**31 - 1, (1,)).item() + self.seed
        rng = random.Random(local_seed)
        modes = self._balanced_modes(len(batch), rng)
        rows = []
        for item, mode in zip(batch, modes):
            clean_ids = item["input_ids"].tolist()
            active_len = 0
            for token_id in clean_ids:
                if token_id == self.tk.pad_id:
                    break
                active_len += 1
            clean_ids = clean_ids[:active_len]
            if len(clean_ids) < 2:
                continue
            aromatic = item.get("aromatic_mask")
            if aromatic is None:
                aromatic_flags = [False] * len(clean_ids)
            else:
                aromatic_flags = aromatic[:active_len].bool().tolist()
            rows.append(
                self._make_row(clean_ids, aromatic_flags, mode, rng)
            )
        if not rows:
            raise RuntimeError("Unified collator received no valid molecular rows.")
        return self._pad_rows(rows)

    def _balanced_modes(self, batch_size, rng):
        raw = [probability * batch_size for probability in self.mode_probabilities]
        counts = [int(math.floor(value)) for value in raw]
        remainder = batch_size - sum(counts)
        order = sorted(
            range(4), key=lambda idx: raw[idx] - counts[idx], reverse=True
        )
        for idx in order[:remainder]:
            counts[idx] += 1
        modes = []
        for mode, count in enumerate(counts):
            modes.extend([mode] * count)
        rng.shuffle(modes)
        return modes

    def _make_row(self, clean_ids, aromatic, mode, rng):
        if mode == MODE_MDM:
            return self._make_mdm(clean_ids, aromatic, rng)
        if mode == MODE_VARIABLE:
            return self._make_variable(clean_ids, aromatic, rng)
        if mode == MODE_FRAGMENT:
            return self._make_fragment(clean_ids, aromatic, rng)
        if mode == MODE_REFINE:
            return self._make_refine(clean_ids, aromatic, rng)
        raise ValueError(f"Unknown corruption mode: {mode}")

    @staticmethod
    def _sample_time(rng, low=0.02, high=0.98):
        return rng.uniform(float(low), float(high))

    def _base_tokens(self, clean_ids, aromatic):
        return [
            _StateToken(
                token_id=token_id,
                clean_index=index,
                aromatic=bool(aromatic[index]),
            )
            for index, token_id in enumerate(clean_ids)
        ]

    def _make_mdm(self, clean_ids, aromatic, rng):
        time = self._sample_time(rng)
        corruption = 1.0 - time
        states = self._base_tokens(clean_ids, aromatic)
        body_indices = list(range(1, len(states) - 1))
        masked = []
        for index in body_indices:
            if rng.random() < corruption:
                masked.append(index)
        if body_indices and not masked:
            masked = [rng.choice(body_indices)]
        for index in masked:
            states[index].token_id = self.tk.mask_id
            states[index].target_id = clean_ids[index]
            states[index].token_weight = 1.0 / max(corruption, 0.05)
        for index in body_indices:
            states[index].delete_label = 0.0
        return self._finalize(states, MODE_MDM, 1.0 - time, supervise_gaps=True)

    def _make_variable(self, clean_ids, aromatic, rng):
        time = self._sample_time(rng, 0.03, 0.97)
        insertion_survival = time ** 0.75
        unmask_probability = time
        states = [
            _StateToken(
                token_id=clean_ids[0],
                clean_index=0,
                aromatic=bool(aromatic[0]),
            )
        ]
        for index in range(1, len(clean_ids) - 1):
            if rng.random() > insertion_survival:
                continue
            visible = rng.random() < unmask_probability
            states.append(
                _StateToken(
                    token_id=clean_ids[index] if visible else self.tk.mask_id,
                    clean_index=index,
                    target_id=-100 if visible else clean_ids[index],
                    token_weight=0.0 if visible else 1.0,
                    delete_label=0.0,
                    aromatic=bool(aromatic[index]),
                )
            )
        states.append(
            _StateToken(
                token_id=clean_ids[-1],
                clean_index=len(clean_ids) - 1,
                aromatic=bool(aromatic[-1]),
            )
        )
        return self._finalize(
            states, MODE_VARIABLE, 1.0 - time, supervise_gaps=True
        )

    def _make_fragment(self, clean_ids, aromatic, rng):
        body_length = len(clean_ids) - 2
        if body_length <= 0:
            return self._make_mdm(clean_ids, aromatic, rng)
        spans = self._sample_fragment_spans(body_length, rng)
        removed = set()
        for start, end in spans:
            removed.update(range(start + 1, end + 1))
        unknown_length = rng.random() < self.fragment_unknown_length_probability
        states = []
        for index, token_id in enumerate(clean_ids):
            if index in removed and unknown_length:
                continue
            if index in removed:
                states.append(
                    _StateToken(
                        token_id=self.tk.mask_id,
                        clean_index=index,
                        target_id=token_id,
                        token_weight=1.0,
                        delete_label=0.0,
                        aromatic=bool(aromatic[index]),
                    )
                )
            else:
                states.append(
                    _StateToken(
                        token_id=token_id,
                        clean_index=index,
                        delete_label=(
                            0.0 if 0 < index < len(clean_ids) - 1 else -100.0
                        ),
                        aromatic=bool(aromatic[index]),
                    )
                )
        # Fragment completion has no access to the hidden target length.  Time
        # is therefore sampled independently and never encodes that length.
        time = self._sample_time(rng, 0.05, 0.75)
        return self._finalize(
            states, MODE_FRAGMENT, 1.0 - time, supervise_gaps=True
        )

    def _make_refine(self, clean_ids, aromatic, rng):
        severity = rng.uniform(0.05, 0.35)
        states = []
        for index, token_id in enumerate(clean_ids):
            special = index == 0 or index == len(clean_ids) - 1
            if special:
                states.append(
                    _StateToken(
                        token_id=token_id,
                        clean_index=index,
                        aromatic=bool(aromatic[index]),
                    )
                )
                continue
            corrupted = rng.random() < severity
            state_token = token_id
            if corrupted and rng.random() < 0.25:
                state_token = self.tk.mask_id
            elif corrupted:
                state_token = self._different_token(token_id, rng)
            states.append(
                _StateToken(
                    token_id=state_token,
                    clean_index=index,
                    target_id=token_id,
                    token_weight=1.0 if corrupted else self.refine_clean_weight,
                    delete_label=0.0,
                    aromatic=bool(aromatic[index]),
                )
            )

        max_extra = min(4, max(0, self.max_len - len(states)))
        expected_extra = severity * max(1, len(clean_ids) - 2) * 0.18
        n_extra = min(max_extra, self._sample_poisson(expected_extra, rng))
        for _ in range(n_extra):
            if len(states) >= self.max_len:
                break
            insert_at = rng.randint(1, len(states) - 1)
            states.insert(
                insert_at,
                _StateToken(
                    token_id=rng.choice(self.candidate_token_ids),
                    clean_index=None,
                    target_id=-100,
                    token_weight=0.0,
                    delete_label=1.0,
                    aromatic=False,
                ),
            )
        return self._finalize(
            states, MODE_REFINE, severity, supervise_gaps=False
        )

    def _different_token(self, target_id, rng):
        if len(self.candidate_token_ids) == 1:
            return self.candidate_token_ids[0]
        candidate = rng.choice(self.candidate_token_ids)
        while candidate == target_id:
            candidate = rng.choice(self.candidate_token_ids)
        return candidate

    @staticmethod
    def _sample_poisson(rate, rng):
        if rate <= 0:
            return 0
        threshold = math.exp(-rate)
        product = 1.0
        count = 0
        while product > threshold:
            product *= rng.random()
            count += 1
        return max(0, count - 1)

    def _sample_fragment_spans(self, body_length, rng):
        if body_length == 1:
            return [(0, 1)]
        if rng.random() < self.fragment_multi_span_probability and body_length >= 6:
            n_spans = 2 if body_length < 16 or rng.random() < 0.75 else 3
            occupied = set()
            spans = []
            for _ in range(n_spans * 8):
                if len(spans) >= n_spans:
                    break
                span_length = self._sample_span_length(body_length, rng, long=False)
                start = rng.randint(0, max(0, body_length - span_length))
                positions = set(range(start, start + span_length))
                guard = set(range(max(0, start - 1), min(body_length, start + span_length + 1)))
                if positions & occupied:
                    continue
                spans.append((start, start + span_length))
                occupied.update(guard)
            if spans:
                return sorted(spans)

        span_length = self._sample_span_length(
            body_length,
            rng,
            long=rng.random() < self.fragment_long_span_probability,
        )
        task_draw = rng.random()
        if task_draw < 0.25:
            start = 0
        elif task_draw < 0.50:
            start = body_length - span_length
        elif body_length > span_length + 2:
            start = rng.randint(1, body_length - span_length - 1)
        else:
            start = rng.randint(0, body_length - span_length)
        return [(start, start + span_length)]

    def _sample_span_length(self, body_length, rng, long=False):
        if long:
            low = min(body_length, max(4, body_length // 3))
            high = min(body_length, max(low, min(64, (3 * body_length) // 4)))
            return rng.randint(low, high)
        cap = min(body_length, 32)
        length = 1
        while length < cap and rng.random() < 0.78:
            length += 1
        if rng.random() < 0.15 and cap >= 8:
            length = rng.randint(8, cap)
        return max(1, min(length, body_length))

    def _finalize(self, states, mode, corruption_level, supervise_gaps):
        if len(states) > self.max_len:
            states = states[: self.max_len]
            if states[-1].token_id != self.tk.eos_id:
                states[-1] = _StateToken(
                    token_id=self.tk.eos_id,
                    clean_index=None,
                )
        gap_labels = [-100] * len(states)
        gap_exact = [0.0] * len(states)
        if supervise_gaps:
            for index in range(len(states) - 1):
                left = states[index].clean_index
                right = states[index + 1].clean_index
                if left is None or right is None or right <= left:
                    continue
                missing = right - left - 1
                gap_labels[index] = min(missing, self.max_gap_count)
                gap_exact[index] = float(min(missing, self.max_gap_count))
        return {
            "states": states,
            "gap_labels": gap_labels,
            "gap_exact": gap_exact,
            "mode": int(mode),
            "corruption_level": float(max(0.0, min(1.0, corruption_level))),
        }

    def _pad_rows(self, rows):
        max_row_len = max(len(row["states"]) for row in rows)
        if self.dynamic_padding:
            padded_len = (
                (max_row_len + self.pad_to_multiple_of - 1)
                // self.pad_to_multiple_of
            ) * self.pad_to_multiple_of
            padded_len = min(self.max_len, max(2, padded_len))
        else:
            padded_len = self.max_len

        batch_size = len(rows)
        input_ids = torch.full(
            (batch_size, padded_len), self.tk.pad_id, dtype=torch.long
        )
        attention_mask = torch.zeros(batch_size, padded_len, dtype=torch.long)
        token_labels = torch.full(
            (batch_size, padded_len), -100, dtype=torch.long
        )
        token_weights = torch.zeros(batch_size, padded_len, dtype=torch.float32)
        gap_labels = torch.full(
            (batch_size, padded_len), -100, dtype=torch.long
        )
        gap_exact = torch.zeros(batch_size, padded_len, dtype=torch.float32)
        delete_labels = torch.full(
            (batch_size, padded_len), -100.0, dtype=torch.float32
        )
        aromatic_mask = torch.zeros(batch_size, padded_len, dtype=torch.bool)
        mode_ids = torch.empty(batch_size, dtype=torch.long)
        corruption_level = torch.empty(batch_size, dtype=torch.float32)

        for row_index, row in enumerate(rows):
            states = row["states"][:padded_len]
            length = len(states)
            input_ids[row_index, :length] = torch.tensor(
                [state.token_id for state in states], dtype=torch.long
            )
            attention_mask[row_index, :length] = 1
            token_labels[row_index, :length] = torch.tensor(
                [state.target_id for state in states], dtype=torch.long
            )
            token_weights[row_index, :length] = torch.tensor(
                [state.token_weight for state in states], dtype=torch.float32
            )
            delete_labels[row_index, :length] = torch.tensor(
                [state.delete_label for state in states], dtype=torch.float32
            )
            aromatic_mask[row_index, :length] = torch.tensor(
                [state.aromatic for state in states], dtype=torch.bool
            )
            gap_labels[row_index, :length] = torch.tensor(
                row["gap_labels"][:length], dtype=torch.long
            )
            gap_exact[row_index, :length] = torch.tensor(
                row["gap_exact"][:length], dtype=torch.float32
            )
            mode_ids[row_index] = row["mode"]
            corruption_level[row_index] = row["corruption_level"]

        return {
            "input_ids": input_ids,
            "attention_mask": attention_mask,
            "token_labels": token_labels,
            "token_weights": token_weights,
            "gap_labels": gap_labels,
            "gap_exact": gap_exact,
            "delete_labels": delete_labels,
            "aromatic_mask": aromatic_mask,
            "mode_ids": mode_ids,
            "corruption_level": corruption_level,
            "cond": torch.empty(batch_size, 0, dtype=torch.float32),
        }
