from __future__ import annotations

from copy import deepcopy

import torch


def _bounded_categorical_sample(
    logits,
    *,
    minimum,
    maximum,
    temperature=1.0,
    top_k=0,
):
    minimum = max(0, int(minimum))
    maximum = min(int(maximum), logits.numel() - 1)
    if maximum < minimum:
        raise ValueError(
            f"Empty learned-length support: minimum={minimum}, maximum={maximum}"
        )
    if temperature <= 0.0:
        raise ValueError("length scheduler temperature must be positive")

    bounded = torch.full_like(logits, -torch.inf, dtype=torch.float32)
    bounded[minimum : maximum + 1] = (
        logits[minimum : maximum + 1].float() / float(temperature)
    )
    support = maximum - minimum + 1
    if 0 < int(top_k) < support:
        values, indices = torch.topk(bounded, int(top_k))
        filtered = torch.full_like(bounded, -torch.inf)
        filtered[indices] = values
        bounded = filtered
    return int(torch.distributions.Categorical(logits=bounded).sample().item())


@torch.no_grad()
def sample_de_novo_lengths(
    scheduler,
    tokenizer,
    count,
    *,
    max_len,
    device,
    temperature=1.0,
    top_k=16,
    batch_size=512,
):
    """Sample full atomic-token lengths from a learned unconditional gap."""
    count = int(count)
    if count <= 0:
        return []
    maximum_body = min(
        int(max_len) - 2,
        int(scheduler.max_replacement_length),
    )
    if maximum_body < 1:
        raise ValueError("max_len leaves no room for molecular body tokens")

    scheduler.eval()
    lengths = []
    for start in range(0, count, int(batch_size)):
        local = min(int(batch_size), count - start)
        input_ids = torch.tensor(
            [[tokenizer.bos_id, tokenizer.mask_id, tokenizer.eos_id]] * local,
            dtype=torch.long,
            device=device,
        )
        attention = torch.ones_like(input_ids)
        gap_mask = torch.zeros_like(input_ids, dtype=torch.bool)
        gap_mask[:, 1] = True
        removed = torch.zeros_like(input_ids)
        corruption = torch.ones(local, device=device)
        output = scheduler(
            input_ids=input_ids,
            attention_mask=attention,
            gap_mask=gap_mask,
            removed_lengths=removed,
            corruption_fraction=corruption,
        )
        for row_logits in output["length_logits"][:, 1]:
            body_length = _bounded_categorical_sample(
                row_logits,
                minimum=1,
                maximum=maximum_body,
                temperature=temperature,
                top_k=top_k,
            )
            lengths.append(body_length + 2)
    return lengths


def _scheduler_example(tokenizer, smiles, plans, max_len):
    active = [
        int(token)
        for token in tokenizer.encode(smiles, max_len)
        if int(token) != tokenizer.pad_id
    ]
    if len(active) < 3:
        raise ValueError(f"Could not encode edit-scheduler seed: {smiles}")
    body = active[1:-1]
    normalized = []
    for gap_id, plan in enumerate(plans):
        start = int(plan["start"])
        stop = int(plan["stop"])
        if not 0 <= start < stop <= len(body):
            raise ValueError(
                f"Invalid edit plan for body length {len(body)}: {plan}"
            )
        normalized.append((start, stop, gap_id, plan))
    normalized.sort(key=lambda row: row[0])
    for left, right in zip(normalized, normalized[1:]):
        if left[1] > right[0]:
            raise ValueError("Learned-length edit plans may not overlap")

    ids = [tokenizer.bos_id]
    gap_mask = [False]
    gap_ids = [-1]
    removed_lengths = [0]
    body_index = 0
    for start, stop, gap_id, _ in normalized:
        ids.extend(body[body_index:start])
        gap_mask.extend([False] * (start - body_index))
        gap_ids.extend([-1] * (start - body_index))
        removed_lengths.extend([0] * (start - body_index))
        ids.append(tokenizer.mask_id)
        gap_mask.append(True)
        gap_ids.append(gap_id)
        removed_lengths.append(stop - start)
        body_index = stop
    ids.extend(body[body_index:])
    gap_mask.extend([False] * (len(body) - body_index))
    gap_ids.extend([-1] * (len(body) - body_index))
    removed_lengths.extend([0] * (len(body) - body_index))
    ids.append(tokenizer.eos_id)
    gap_mask.append(False)
    gap_ids.append(-1)
    removed_lengths.append(0)
    return {
        "input_ids": ids,
        "gap_mask": gap_mask,
        "gap_ids": gap_ids,
        "removed_lengths": removed_lengths,
        "body_length": len(body),
        "normalized": normalized,
    }


def _pad(rows, value, *, dtype, device):
    width = max(len(row) for row in rows)
    output = torch.full(
        (len(rows), width),
        value,
        dtype=dtype,
        device=device,
    )
    for index, row in enumerate(rows):
        output[index, : len(row)] = torch.tensor(
            row,
            dtype=dtype,
            device=device,
        )
    return output


@torch.no_grad()
def schedule_replacement_lengths(
    scheduler,
    tokenizer,
    seed_smiles,
    edit_plans,
    *,
    max_len,
    device,
    temperature=1.0,
    top_k=8,
    minimum_replacement=1,
):
    """Replace fixed/random gap sizes with context-conditioned predictions."""
    if len(seed_smiles) != len(edit_plans):
        raise ValueError("seed_smiles and edit_plans must have equal length")
    if not seed_smiles:
        return [], []

    examples = [
        _scheduler_example(tokenizer, smiles, plans, max_len)
        for smiles, plans in zip(seed_smiles, edit_plans)
    ]
    maximum_positions = int(scheduler.max_position_embeddings)
    if any(len(example["input_ids"]) > maximum_positions for example in examples):
        raise ValueError(
            "Collapsed edit context exceeds scheduler position embeddings"
        )

    input_ids = _pad(
        [example["input_ids"] for example in examples],
        tokenizer.pad_id,
        dtype=torch.long,
        device=device,
    )
    gap_mask = _pad(
        [example["gap_mask"] for example in examples],
        False,
        dtype=torch.bool,
        device=device,
    )
    removed = _pad(
        [example["removed_lengths"] for example in examples],
        0,
        dtype=torch.long,
        device=device,
    )
    attention = input_ids.ne(tokenizer.pad_id).long()
    corruption = torch.tensor(
        [
            sum(example["removed_lengths"])
            / max(1, example["body_length"])
            for example in examples
        ],
        dtype=torch.float32,
        device=device,
    )
    scheduler.eval()
    output = scheduler(
        input_ids=input_ids,
        attention_mask=attention,
        gap_mask=gap_mask,
        removed_lengths=removed,
        corruption_fraction=corruption,
    )

    scheduled_rows = []
    diagnostics = []
    for row_index, (example, original_plans) in enumerate(
        zip(examples, edit_plans)
    ):
        plans = deepcopy(list(original_plans))
        predictions = [None] * len(plans)
        rates = [0.0] * len(plans)
        for position, gap_id in enumerate(example["gap_ids"]):
            if gap_id < 0:
                continue
            plan = plans[gap_id]
            minimum = int(plan.get("min_replacement_len", minimum_replacement))
            maximum = int(
                plan.get(
                    "max_replacement_len",
                    scheduler.max_replacement_length,
                )
            )
            predictions[gap_id] = _bounded_categorical_sample(
                output["length_logits"][row_index, position],
                minimum=minimum,
                maximum=maximum,
                temperature=temperature,
                top_k=top_k,
            )
            rates[gap_id] = float(
                output["insertion_rate"][row_index, position].item()
            )

        fixed_tokens = example["body_length"] - sum(
            int(plan["stop"]) - int(plan["start"])
            for plan in plans
        )
        available = max(0, int(max_len) - 2 - fixed_tokens)
        total = sum(int(value) for value in predictions)
        if total > available:
            # Remove excess from the least preferred gaps first while
            # respecting their hard minima. This is deterministic conditional
            # capacity enforcement, not a random length fallback.
            for gap_id in sorted(range(len(plans)), key=lambda idx: rates[idx]):
                minimum = int(
                    plans[gap_id].get(
                        "min_replacement_len",
                        minimum_replacement,
                    )
                )
                reducible = max(0, int(predictions[gap_id]) - minimum)
                reduction = min(reducible, total - available)
                predictions[gap_id] -= reduction
                total -= reduction
                if total <= available:
                    break
        if total > available:
            raise ValueError(
                "Learned replacement lengths cannot fit within max_len"
            )

        for gap_id, prediction in enumerate(predictions):
            plans[gap_id]["replacement_len"] = int(prediction)
            plans[gap_id].pop("delta", None)
            plans[gap_id]["length_mode"] = "learned_scheduler"
        scheduled_rows.append(tuple(plans))
        diagnostics.append(
            {
                "replacement_lengths": [int(value) for value in predictions],
                "insertion_rates": rates,
                "target_body_length": fixed_tokens + total,
            }
        )
    return scheduled_rows, diagnostics
