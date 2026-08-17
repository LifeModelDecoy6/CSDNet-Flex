import math
import random

import torch


def _pad_rows(rows, value, dtype):
    maximum = max(len(row) for row in rows)
    tensor = torch.full(
        (len(rows), maximum),
        value,
        dtype=dtype,
    )
    for index, row in enumerate(rows):
        tensor[index, : len(row)] = torch.tensor(row, dtype=dtype)
    return tensor


class EditScheduleCorruptionCollator:
    """
    Build multi-gap length-recovery examples from complete token sequences.

    A positive gap removes a real clean span and labels its original length.
    A zero gap removes no clean tokens and teaches the scheduler to decline an
    unnecessary insertion. The observed removed length is deliberately noisy,
    preventing the scheduler from learning an identity mapping.
    """

    def __init__(
        self,
        tokenizer,
        max_len=256,
        max_gaps=4,
        max_span_length=24,
        max_replacement_length=32,
        zero_gap_probability=0.25,
        pure_insertion_probability=0.25,
        unchanged_length_probability=0.35,
        unconditional_gap_probability=0.15,
    ):
        self.tk = tokenizer
        self.max_len = int(max_len)
        self.max_gaps = int(max_gaps)
        self.max_span_length = int(max_span_length)
        self.max_replacement_length = int(max_replacement_length)
        self.zero_gap_probability = float(zero_gap_probability)
        self.pure_insertion_probability = float(
            pure_insertion_probability
        )
        self.unchanged_length_probability = float(
            unchanged_length_probability
        )
        self.unconditional_gap_probability = float(
            unconditional_gap_probability
        )
        if self.max_gaps < 1:
            raise ValueError("max_gaps must be positive.")
        if not 0.0 <= self.zero_gap_probability <= 1.0:
            raise ValueError("zero_gap_probability must be in [0, 1].")
        if not 0.0 <= self.unconditional_gap_probability <= 1.0:
            raise ValueError(
                "unconditional_gap_probability must be in [0, 1]."
            )

    @staticmethod
    def _sample_gap_count():
        draw = random.random()
        if draw < 0.55:
            return 1
        if draw < 0.85:
            return 2
        if draw < 0.95:
            return 3
        return 4

    def _sample_span_length(self, maximum):
        # A geometric-like distribution keeps local edits common while
        # retaining a useful tail of larger fragment gaps.
        draw = max(1e-8, random.random())
        length = 1 + int(math.log(draw) / math.log(0.78))
        return min(maximum, max(1, length))

    def _sample_spans(self, body_length):
        target_count = min(self.max_gaps, self._sample_gap_count())
        spans = []
        removable_limit = max(1, body_length - 1)
        attempts = 0
        while len(spans) < target_count and attempts < 100:
            attempts += 1
            maximum = min(
                self.max_span_length,
                self.max_replacement_length,
                removable_limit,
            )
            if maximum < 1:
                break
            length = self._sample_span_length(maximum)
            start = random.randint(0, body_length - length)
            stop = start + length
            if any(
                not (stop + 1 <= old_start or start >= old_stop + 1)
                for old_start, old_stop in spans
            ):
                continue
            if sum(old_stop - old_start for old_start, old_stop in spans) + length >= body_length:
                continue
            spans.append((start, stop))
        if not spans:
            start = random.randint(0, max(0, body_length - 1))
            spans = [(start, min(body_length, start + 1))]
        return sorted(spans)

    def _observed_removed_length(self, target_length):
        draw = random.random()
        if draw < self.pure_insertion_probability:
            return 0
        if draw < (
            self.pure_insertion_probability
            + self.unchanged_length_probability
        ):
            return target_length
        delta_choices = [
            value
            for value in range(-6, 7)
            if value != 0
        ]
        return min(
            self.max_replacement_length,
            max(0, target_length + random.choice(delta_choices)),
        )

    def _build_example(self, clean_ids):
        active = [
            int(token)
            for token in clean_ids
            if int(token) != self.tk.pad_id
        ]
        if len(active) < 3:
            raise ValueError("A clean sequence must contain BOS, body, and EOS.")
        body = active[1:-1]
        unconditional = (
            len(body) <= self.max_replacement_length
            and random.random() < self.unconditional_gap_probability
        )
        spans = (
            [(0, len(body))]
            if unconditional
            else self._sample_spans(len(body))
        )

        events = []
        for start, stop in spans:
            target_length = stop - start
            events.append(
                {
                    "boundary": start,
                    "stop": stop,
                    "target_length": target_length,
                    "removed_length": (
                        0
                        if unconditional
                        else self._observed_removed_length(target_length)
                    ),
                }
            )

        compressed_length = (
            len(active)
            - sum(event["target_length"] for event in events)
            + len(events)
        )
        if (
            not unconditional
            and len(events) < self.max_gaps
            and compressed_length < self.max_len
            and random.random() < self.zero_gap_probability
        ):
            blocked = set()
            for event in events:
                blocked.update(
                    range(event["boundary"], event["stop"] + 1)
                )
            candidates = [
                boundary
                for boundary in range(len(body) + 1)
                if boundary not in blocked
            ]
            if candidates:
                events.append(
                    {
                        "boundary": random.choice(candidates),
                        "stop": None,
                        "target_length": 0,
                        "removed_length": (
                            0
                            if random.random() < 0.5
                            else random.randint(
                                1,
                                min(8, self.max_replacement_length),
                            )
                        ),
                    }
                )
        events.sort(
            key=lambda event: (
                event["boundary"],
                event["target_length"] == 0,
            )
        )
        for gap_id, event in enumerate(events):
            event["gap_id"] = gap_id

        event_at = {}
        for event in events:
            event_at.setdefault(event["boundary"], []).append(event)

        scheduler_ids = [self.tk.bos_id]
        scheduler_gap_mask = [False]
        scheduler_gap_ids = [-1]
        scheduler_removed_lengths = [0]
        scheduler_length_labels = [-100]
        body_index = 0
        while body_index <= len(body):
            local_events = event_at.get(body_index, [])
            positive = [
                event
                for event in local_events
                if event["target_length"] > 0
            ]
            zero = [
                event
                for event in local_events
                if event["target_length"] == 0
            ]
            for event in positive + zero:
                scheduler_ids.append(self.tk.mask_id)
                scheduler_gap_mask.append(True)
                scheduler_gap_ids.append(event["gap_id"])
                scheduler_removed_lengths.append(event["removed_length"])
                scheduler_length_labels.append(event["target_length"])
            if positive:
                body_index = positive[0]["stop"]
                continue
            if body_index == len(body):
                break
            scheduler_ids.append(body[body_index])
            scheduler_gap_mask.append(False)
            scheduler_gap_ids.append(-1)
            scheduler_removed_lengths.append(0)
            scheduler_length_labels.append(-100)
            body_index += 1

        scheduler_ids.append(self.tk.eos_id)
        scheduler_gap_mask.append(False)
        scheduler_gap_ids.append(-1)
        scheduler_removed_lengths.append(0)
        scheduler_length_labels.append(-100)
        if len(scheduler_ids) > self.max_len:
            raise RuntimeError(
                "Corruption produced a scheduler sequence above max_len."
            )

        teacher_ids = list(active)
        teacher_labels = [-100] * len(active)
        teacher_gap_ids = [-1] * len(active)
        for event in events:
            if event["target_length"] == 0:
                continue
            for body_position in range(
                event["boundary"],
                event["stop"],
            ):
                sequence_position = body_position + 1
                teacher_labels[sequence_position] = teacher_ids[
                    sequence_position
                ]
                teacher_ids[sequence_position] = self.tk.mask_id
                teacher_gap_ids[sequence_position] = event["gap_id"]

        removed_tokens = sum(
            event["target_length"]
            for event in events
        )
        return {
            "scheduler_ids": scheduler_ids,
            "scheduler_gap_mask": scheduler_gap_mask,
            "scheduler_gap_ids": scheduler_gap_ids,
            "scheduler_removed_lengths": scheduler_removed_lengths,
            "scheduler_length_labels": scheduler_length_labels,
            "teacher_ids": teacher_ids,
            "teacher_labels": teacher_labels,
            "teacher_gap_ids": teacher_gap_ids,
            "gap_count": len(events),
            "corruption_fraction": (
                removed_tokens / max(1, len(body))
            ),
        }

    def __call__(self, batch):
        examples = [
            self._build_example(row["input_ids"].tolist())
            for row in batch
        ]
        scheduler_input_ids = _pad_rows(
            [row["scheduler_ids"] for row in examples],
            self.tk.pad_id,
            torch.long,
        )
        teacher_input_ids = _pad_rows(
            [row["teacher_ids"] for row in examples],
            self.tk.pad_id,
            torch.long,
        )
        return {
            "input_ids": scheduler_input_ids,
            "attention_mask": scheduler_input_ids.ne(self.tk.pad_id).long(),
            "gap_mask": _pad_rows(
                [row["scheduler_gap_mask"] for row in examples],
                False,
                torch.bool,
            ),
            "gap_ids": _pad_rows(
                [row["scheduler_gap_ids"] for row in examples],
                -1,
                torch.long,
            ),
            "removed_lengths": _pad_rows(
                [row["scheduler_removed_lengths"] for row in examples],
                0,
                torch.long,
            ),
            "length_labels": _pad_rows(
                [row["scheduler_length_labels"] for row in examples],
                -100,
                torch.long,
            ),
            "teacher_input_ids": teacher_input_ids,
            "teacher_attention_mask": teacher_input_ids.ne(
                self.tk.pad_id
            ).long(),
            "teacher_labels": _pad_rows(
                [row["teacher_labels"] for row in examples],
                -100,
                torch.long,
            ),
            "teacher_gap_ids": _pad_rows(
                [row["teacher_gap_ids"] for row in examples],
                -1,
                torch.long,
            ),
            "gap_count": torch.tensor(
                [row["gap_count"] for row in examples],
                dtype=torch.long,
            ),
            "corruption_fraction": torch.tensor(
                [row["corruption_fraction"] for row in examples],
                dtype=torch.float32,
            ),
        }
