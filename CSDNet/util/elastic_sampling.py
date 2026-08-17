import inspect
import math
from collections import Counter

import torch

from CSDNet.model.elastic_schedule import ElasticKumaSchedule
from CSDNet.util.fsm import (
    ValenceFSMTracker,
    compute_rdkit_kekulize_penalties,
    expand_violation_mask,
    prepare_rdkit_kekulize_checker,
    rdkit_smiles_is_valid,
)
from CSDNet.util.tokenizer import tokenize_smiles


def _filter_logits(logits, top_k=0, top_p=1.0, min_tokens_to_keep=1):
    filtered = logits
    vocabulary_size = logits.size(-1)
    min_tokens_to_keep = min(
        vocabulary_size,
        max(1, int(min_tokens_to_keep)),
    )
    if top_k and 0 < top_k < vocabulary_size:
        threshold = torch.topk(filtered, top_k, dim=-1).values[..., -1, None]
        filtered = filtered.masked_fill(filtered < threshold, -1e9)
    if top_p is not None and 0.0 < top_p < 1.0:
        sorted_logits, sorted_indices = torch.sort(
            filtered,
            descending=True,
            dim=-1,
        )
        sorted_probabilities = torch.softmax(sorted_logits, dim=-1)
        cumulative = torch.cumsum(sorted_probabilities, dim=-1)
        remove = cumulative > top_p
        remove[..., 1:] = remove[..., :-1].clone()
        remove[..., 0] = False
        remove[..., :min_tokens_to_keep] = False
        scatter_remove = torch.zeros_like(remove)
        scatter_remove.scatter_(-1, sorted_indices, remove)
        filtered = filtered.masked_fill(scatter_remove, -1e9)
    return filtered


def _pad_sequences(sequences, pad_id, device):
    max_length = max(len(sequence) for sequence in sequences)
    tensor = torch.full(
        (len(sequences), max_length),
        pad_id,
        device=device,
        dtype=torch.long,
    )
    for row, sequence in enumerate(sequences):
        tensor[row, : len(sequence)] = torch.tensor(
            sequence,
            device=device,
            dtype=torch.long,
        )
    return tensor


def _strip_padding(tensor, pad_id):
    result = []
    for row in tensor.detach().cpu().tolist():
        result.append([token for token in row if token != pad_id])
    return result


def _pad_boolean_sequences(sequences, device):
    max_length = max(len(sequence) for sequence in sequences)
    tensor = torch.zeros(
        (len(sequences), max_length),
        device=device,
        dtype=torch.bool,
    )
    for row, sequence in enumerate(sequences):
        tensor[row, : len(sequence)] = torch.tensor(
            sequence,
            device=device,
            dtype=torch.bool,
        )
    return tensor


def _sample_tokens(
    logits,
    tk,
    temperature,
    top_k,
    top_p,
    min_tokens_to_keep=1,
    return_confidence=False,
):
    logits = _mask_special_token_logits(logits, tk)
    scaled_logits = logits / max(float(temperature), 1e-4)
    confidence = torch.softmax(scaled_logits.float(), dim=-1).amax(dim=-1)
    sample_logits = _filter_logits(
        scaled_logits,
        top_k=top_k,
        top_p=top_p,
        min_tokens_to_keep=min_tokens_to_keep,
    )
    sampled = torch.distributions.Categorical(logits=sample_logits).sample()
    if not return_confidence:
        return sampled
    return sampled, confidence


def _mask_special_token_logits(logits, tk):
    logits = logits.clone()
    special_ids = [
        tk.bos_id,
        tk.eos_id,
        tk.mask_id,
        tk.pad_id,
        getattr(tk, "unk_id", -1),
    ]
    for token_id in special_ids:
        if token_id is not None and token_id >= 0:
            logits[..., token_id] = -1e9
    return logits


def _repair_model_logits(model, tensor, attention, repair_time):
    """Condition repairs when supported without changing legacy backbones."""
    try:
        parameters = inspect.signature(model.forward).parameters
    except (TypeError, ValueError):
        parameters = {}
    accepts_kwargs = any(
        parameter.kind is inspect.Parameter.VAR_KEYWORD
        for parameter in parameters.values()
    )
    if "t" in parameters or accepts_kwargs:
        return model(tensor, attention, t=repair_time)
    if "corruption_level" in parameters:
        corruption_level = (1.0 - repair_time).clamp(0.0, 1.0)
        return model(
            tensor,
            attention,
            corruption_level=corruption_level,
        )
    return model(tensor, attention)


def _constrain_token_logits(logits, constraint_sequences, tk):
    """Mask token logits at positions whose template has a structural role."""
    if constraint_sequences is None:
        return logits
    constrained = logits.clone()
    chain_atom_ids = [
        tk.vocab[token] for token in ("B", "C", "N", "O", "P", "S") if token in tk.vocab
    ]
    has_chain_atoms = any(
        constraint == "chain_atom"
        for sequence in constraint_sequences
        for constraint in sequence
    )
    if has_chain_atoms and not chain_atom_ids:
        raise RuntimeError(
            "chain_atom constraints require at least one compatible token"
        )
    if not has_chain_atoms:
        return constrained

    positions = torch.zeros(
        constrained.shape[:2],
        device=constrained.device,
        dtype=torch.bool,
    )
    for row, sequence in enumerate(constraint_sequences):
        limit = min(len(sequence), positions.size(1))
        if limit:
            positions[row, :limit] = torch.tensor(
                [value == "chain_atom" for value in sequence[:limit]],
                device=positions.device,
                dtype=torch.bool,
            )
    allowed = torch.zeros(
        constrained.size(-1),
        device=constrained.device,
        dtype=torch.bool,
    )
    allowed[chain_atom_ids] = True
    constrained.masked_fill_(
        positions.unsqueeze(-1) & ~allowed.view(1, 1, -1),
        -1e9,
    )
    return constrained


def _position_constraint_sequences(states):
    """Resolve dynamic gap constraints into per-token sampling roles."""
    resolved = []
    for state in states:
        sequence = list(state.get("constraints", [None] * len(state["tokens"])))
        gap_ids = state.get("gap_ids", [-1] * len(state["tokens"]))
        editable = state.get("editable", [False] * len(state["tokens"]))
        for gap_id, gap in enumerate(state["gaps"]):
            if gap.get("constraint") != "atom_bounded":
                continue
            positions = [
                position
                for position, (member, is_editable) in enumerate(zip(gap_ids, editable))
                if int(member) == gap_id and bool(is_editable)
            ]
            if positions:
                sequence[positions[0]] = "chain_atom"
                sequence[positions[-1]] = "chain_atom"
        resolved.append(sequence)
    return resolved


def _select_unmask_events(
    current_mask,
    unmask_probability,
    confidence,
    selection,
):
    proposed = (torch.rand_like(unmask_probability) < unmask_probability) & current_mask
    if selection == "random":
        return proposed
    if selection != "top_prob":
        raise ValueError(
            f"unmask_selection must be 'random' or 'top_prob', got {selection!r}"
        )

    event_counts = proposed.sum(dim=-1, keepdim=True)
    if not event_counts.any():
        return proposed
    scores = confidence.masked_fill(~current_mask, -torch.inf)
    ordering = scores.argsort(dim=-1, descending=True)
    ranks = torch.empty_like(ordering)
    rank_values = (
        torch.arange(
            scores.size(-1),
            device=scores.device,
        )
        .unsqueeze(0)
        .expand_as(ordering)
    )
    ranks.scatter_(1, ordering, rank_values)
    return (ranks < event_counts) & current_mask


@torch.no_grad()
def _progressive_final_unmask(
    model,
    tk,
    tensor,
    *,
    cond=None,
    guidance_weight=2.0,
    constraint_sequences=None,
    temperature=1.0,
    top_k=0,
    top_p=1.0,
    deterministic=True,
    progressive_steps=8,
    time_start=0.90,
    time_end=0.99,
):
    """Complete residual masks in confidence order with refreshed context."""
    if int(progressive_steps) < 1:
        raise ValueError("progressive final-unmask steps must be positive")
    if not 0.0 < float(time_start) <= float(time_end) < 1.0:
        raise ValueError("final-unmask times must satisfy 0 < start <= end < 1")

    initial_masks = tensor.eq(tk.mask_id).sum(dim=1)
    forward_calls = 0
    for step in range(int(progressive_steps)):
        remaining = tensor.eq(tk.mask_id)
        if not remaining.any():
            break
        attention = tensor.ne(tk.pad_id).long()
        progress = (step + 0.5) / max(1, int(progressive_steps))
        time_value = (
            float(time_start) + (float(time_end) - float(time_start)) * progress
        )
        time_tensor = torch.full(
            (tensor.size(0),),
            time_value,
            device=tensor.device,
            dtype=torch.float32,
        )
        if cond is not None:
            conditional = model(
                tensor,
                attention,
                cond=cond,
                drop_cond=False,
                t=time_tensor,
            )
            unconditional = model(
                tensor,
                attention,
                cond=cond,
                drop_cond=True,
                t=time_tensor,
            )
            logits = (
                float(guidance_weight) * conditional
                + (1.0 - float(guidance_weight)) * unconditional
            )
            forward_calls += 2
        else:
            logits = model(tensor, attention, t=time_tensor)
            forward_calls += 1
        logits = _constrain_token_logits(
            logits,
            constraint_sequences,
            tk,
        )
        clean_logits = _mask_special_token_logits(logits, tk)
        scaled_logits = clean_logits / max(float(temperature), 1e-4)
        confidence = torch.softmax(
            scaled_logits.float(),
            dim=-1,
        ).amax(dim=-1)
        if deterministic:
            proposed_tokens = scaled_logits.argmax(dim=-1)
        else:
            sample_logits = _filter_logits(
                scaled_logits,
                top_k=top_k,
                top_p=top_p,
            )
            proposed_tokens = torch.distributions.Categorical(
                logits=sample_logits
            ).sample()

        steps_left = int(progressive_steps) - step
        remaining_counts = remaining.sum(dim=1)
        commit_counts = torch.div(
            remaining_counts + steps_left - 1,
            steps_left,
            rounding_mode="floor",
        )
        scores = confidence.masked_fill(~remaining, -torch.inf)
        ordering = scores.argsort(dim=1, descending=True)
        ranks = torch.empty_like(ordering)
        rank_values = (
            torch.arange(
                ordering.size(1),
                device=tensor.device,
            )
            .unsqueeze(0)
            .expand_as(ordering)
        )
        ranks.scatter_(1, ordering, rank_values)
        commit = (ranks < commit_counts.unsqueeze(1)) & remaining
        tensor.masked_scatter_(commit, proposed_tokens[commit])

    # The ceiling schedule commits every residual mask on the last round.
    if tensor.eq(tk.mask_id).any():
        raise RuntimeError("progressive final unmask left unresolved masks")
    return tensor, {
        "forced_final_unmasks": int(initial_masks.sum().item()),
        "forced_final_unmask_rows": int(initial_masks.gt(0).sum().item()),
        "final_unmask_forward_calls": int(forward_calls),
        "final_unmask_progressive_steps": int(progressive_steps),
        "final_unmask_time_start": float(time_start),
        "final_unmask_time_end": float(time_end),
    }


def _apply_insertions(
    sequences,
    insertion_counts,
    mask_id,
    max_length,
):
    expanded = []
    for sequence, counts in zip(sequences, insertion_counts):
        remaining = max(0, max_length - len(sequence))
        output = []
        for position, token in enumerate(sequence):
            count = int(counts[position]) if position < len(counts) else 0
            if position == 0:
                count = 0
            count = min(count, remaining)
            if count:
                output.extend([mask_id] * count)
                remaining -= count
            output.append(token)
        expanded.append(output[:max_length])
    return expanded


def _fit_gap_insertions_to_capacity(
    insertions,
    capacity,
    gap_capacities=None,
):
    """Share sequence and optional per-gap budgets across insertion events."""
    remaining = max(0, int(capacity))
    gap_remaining = (
        {
            int(gap_id): max(0, int(gap_capacity))
            for gap_id, gap_capacity in gap_capacities.items()
        }
        if gap_capacities is not None
        else None
    )
    bounded = []
    for position, gap_id, requested in insertions:
        limits = [max(0, int(requested)), remaining]
        if gap_remaining is not None:
            limits.append(gap_remaining.get(int(gap_id), 0))
        count = min(limits)
        if count > 0:
            bounded.append((position, gap_id, count))
            remaining -= count
            if gap_remaining is not None:
                gap_remaining[int(gap_id)] -= count
        if remaining <= 0:
            break
    return bounded


def _apply_local_gap_insertions(
    state,
    insertions,
    mask_id,
    *,
    recursive_gap_insertions,
):
    """Materialize local insertions while preserving editable-gap lineage."""
    inserted = 0
    constraints = state.setdefault(
        "constraints",
        [None] * len(state["tokens"]),
    )
    gap_ids = state.setdefault(
        "gap_ids",
        [-1] * len(state["tokens"]),
    )
    for position, gap_id, count in reversed(insertions):
        count = max(0, int(count))
        if count <= 0:
            continue
        state["tokens"][position:position] = [mask_id] * count
        state["editable"][position:position] = [True] * count
        gap_constraint = state["gaps"][gap_id].get("constraint")
        static_constraint = "chain_atom" if gap_constraint == "chain_atom" else None
        constraints[position:position] = [static_constraint] * count
        gap_ids[position:position] = [int(gap_id)] * count
        anchor = int(gap_id) if recursive_gap_insertions else -1
        state["anchors"][position:position] = [anchor] * count
        state["gaps"][gap_id]["inserted"] += count
        inserted += count
    return inserted


@torch.no_grad()
def _repair_final_sequences(
    model,
    tk,
    sequences,
    device,
    use_fsm_check,
    use_rdkit_kekulize_check,
    max_sample_retries,
    violation_neighborhood,
    temperature,
    top_k,
    top_p,
    editable_sequences=None,
    token_constraint_sequences=None,
    progressive_steps=1,
    prefer_fsm_localization=False,
    hard_syntax_projection=True,
    repair_time_start=0.65,
    repair_time_end=0.95,
    sequence_validators=None,
    return_diagnostics=False,
):
    if sequence_validators is not None and len(sequence_validators) != len(sequences):
        raise ValueError("sequence_validators must match the sequence batch")
    if (
        not use_fsm_check
        and not use_rdkit_kekulize_check
        and sequence_validators is None
    ):
        if return_diagnostics:
            return sequences, {
                "hard_projection_rows": 0,
                "neural_repair_rows": 0,
                "neural_repair_rounds": 0,
            }
        return sequences
    if not 0.0 < float(repair_time_start) <= float(repair_time_end) < 1.0:
        raise ValueError("repair times must satisfy 0 < start <= end < 1")

    tracker = ValenceFSMTracker(tk) if use_fsm_check else None
    rdkit_checker = (
        prepare_rdkit_kekulize_checker(tk, tracker)
        if use_rdkit_kekulize_check
        else None
    )
    repair_diagnostics = Counter()

    normalized_sequences = []
    normalized_editable = []
    normalized_constraints = []
    for row, sequence in enumerate(sequences):
        sequence = list(sequence)
        row_editable = (
            editable_sequences[row] if editable_sequences is not None else None
        )
        row_constraints = (
            token_constraint_sequences[row]
            if token_constraint_sequences is not None
            else None
        )
        if row_editable is None:
            row_editable = [
                token_id not in {tk.pad_id, tk.bos_id, tk.eos_id}
                for token_id in sequence
            ]
        else:
            row_editable = list(row_editable)
        if row_constraints is None:
            row_constraints = [None] * len(sequence)
        else:
            row_constraints = list(row_constraints)
        normalized_sequences.append(sequence)
        normalized_editable.append(row_editable)
        normalized_constraints.append(row_constraints)

    sequences = normalized_sequences
    editable_sequences = normalized_editable
    token_constraint_sequences = normalized_constraints
    validators = (
        list(sequence_validators)
        if sequence_validators is not None
        else [None] * len(sequences)
    )
    tensor = _pad_sequences(sequences, tk.pad_id, device)
    non_special = tensor.ne(tk.pad_id) & tensor.ne(tk.bos_id) & tensor.ne(tk.eos_id)
    editable = _pad_boolean_sequences(editable_sequences, tensor.device)
    editable = editable[:, : tensor.size(1)] & non_special

    def evaluate_rows(values):
        penalties = torch.zeros_like(values, dtype=torch.float)
        row_editable = editable[:, : values.size(1)]
        if row_editable.size(1) < values.size(1):
            row_editable = torch.nn.functional.pad(
                row_editable,
                (0, values.size(1) - row_editable.size(1)),
                value=False,
            )
        fsm_penalties = None
        if use_fsm_check:
            fsm_penalties = tracker.compute_penalties(values)
            penalties += fsm_penalties
        validity_rank = torch.ones(
            values.size(0),
            device=values.device,
            dtype=torch.long,
        )
        if rdkit_checker is not None:
            chemistry, focus_ids = rdkit_checker
            rdkit_penalties = compute_rdkit_kekulize_penalties(
                values,
                tk,
                chemistry,
                focus_ids,
            )
            if prefer_fsm_localization and fsm_penalties is not None:
                # RDKit parser failures cannot be localized and therefore mark
                # an entire row. Preserve the FSM's specific diagnosis first;
                # RDKit remains the fallback on the next round if that local
                # repair does not resolve the molecule.
                localized_rows = fsm_penalties.lt(0).any(dim=1)
                rdkit_penalties.masked_fill_(localized_rows.unsqueeze(1), 0.0)
            penalties += rdkit_penalties
            ranks = []
            for sequence in _strip_padding(values, tk.pad_id):
                smiles = tk.decode(sequence).strip("'\"")
                molecule = chemistry.MolFromSmiles(smiles, sanitize=False)
                if molecule is None:
                    ranks.append(2)
                    continue
                try:
                    chemistry.SanitizeMol(molecule)
                except Exception:
                    ranks.append(1)
                else:
                    ranks.append(0)
            validity_rank = torch.tensor(
                ranks,
                device=values.device,
                dtype=torch.long,
            )
            penalties.masked_fill_(validity_rank.eq(0).unsqueeze(1), 0.0)
        else:
            validity_rank = penalties.lt(0).any(dim=1).long()
        constraint_failures = []
        decoded = _strip_padding(values, tk.pad_id)
        for sequence, validator in zip(decoded, validators):
            if validator is None:
                constraint_failures.append(False)
                continue
            smiles = tk.decode(sequence).strip("'\"")
            try:
                constraint_failures.append(not bool(validator(smiles)))
            except Exception:
                constraint_failures.append(True)
        constraint_invalid = torch.tensor(
            constraint_failures,
            device=values.device,
            dtype=torch.bool,
        )
        if constraint_invalid.any():
            # A protected-fragment failure cannot be localized reliably in a
            # linearized SMILES string. Re-open only the caller-declared
            # editable gap; protected context remains immutable.
            penalties.masked_fill_(
                constraint_invalid.unsqueeze(1) & row_editable,
                -1.0,
            )
            validity_rank = torch.maximum(
                validity_rank,
                constraint_invalid.long(),
            )
        violation_count = penalties.lt(0).sum(dim=1).long()
        score = validity_rank * 10000 + violation_count
        return penalties, validity_rank, score, constraint_invalid

    penalties, validity_rank, best_score, constraint_invalid = evaluate_rows(tensor)
    best_tensor = tensor.clone()
    initially_invalid = validity_rank.gt(0)
    repair_diagnostics["initial_invalid_rows"] = int(initially_invalid.sum().item())
    repair_diagnostics["initial_constraint_invalid_rows"] = int(
        constraint_invalid.sum().item()
    )
    repair_diagnostics["neural_repair_rows"] = int(initially_invalid.sum().item())

    for _ in range(max(1, max_sample_retries) * 4):
        violations = (penalties < 0) & non_special
        invalid_rows = validity_rank.gt(0)
        if not invalid_rows.any():
            break
        violations &= invalid_rows.unsqueeze(1)
        if not violations.any():
            violations = editable & invalid_rows.unsqueeze(1)
        repair = expand_violation_mask(
            violations,
            editable,
            radius=violation_neighborhood,
        )
        repair &= invalid_rows.unsqueeze(1)
        if not repair.any():
            break
        repair_diagnostics["neural_repair_rounds"] += 1
        tensor.masked_fill_(repair, tk.mask_id)
        attention = tensor.ne(tk.pad_id).long()
        repair_steps = max(1, int(progressive_steps))
        repair_fraction = repair.sum(dim=1).float() / non_special.sum(
            dim=1
        ).float().clamp(min=1.0)
        row_start_time = (1.0 - 1.5 * repair_fraction).clamp(
            min=float(repair_time_start),
            max=float(repair_time_end),
        )
        if repair_steps == 1:
            logits = _constrain_token_logits(
                _repair_model_logits(
                    model,
                    tensor,
                    attention,
                    row_start_time,
                ),
                token_constraint_sequences,
                tk,
            )
            sampled = _sample_tokens(
                logits,
                tk,
                temperature=temperature,
                top_k=top_k,
                top_p=top_p,
            )
            tensor.masked_scatter_(repair, sampled[repair])
        else:
            for repair_step in range(repair_steps):
                remaining = tensor.eq(tk.mask_id) & repair
                if not remaining.any():
                    break
                progress = (repair_step + 0.5) / repair_steps
                repair_time = (
                    row_start_time
                    + (float(repair_time_end) - row_start_time) * progress
                )
                logits = _constrain_token_logits(
                    _repair_model_logits(
                        model,
                        tensor,
                        attention,
                        repair_time,
                    ),
                    token_constraint_sequences,
                    tk,
                )
                sampled, confidence = _sample_tokens(
                    logits,
                    tk,
                    temperature=temperature,
                    top_k=top_k,
                    top_p=top_p,
                    return_confidence=True,
                )
                steps_left = repair_steps - repair_step
                remaining_counts = remaining.sum(dim=1)
                commit_counts = torch.div(
                    remaining_counts + steps_left - 1,
                    steps_left,
                    rounding_mode="floor",
                )
                scores = confidence.masked_fill(~remaining, -torch.inf)
                ordering = scores.argsort(dim=1, descending=True)
                ranks = torch.empty_like(ordering)
                rank_values = (
                    torch.arange(
                        ordering.size(1),
                        device=tensor.device,
                    )
                    .unsqueeze(0)
                    .expand_as(ordering)
                )
                ranks.scatter_(1, ordering, rank_values)
                commit = (ranks < commit_counts.unsqueeze(1)) & remaining
                tensor.masked_scatter_(commit, sampled[commit])

        (
            candidate_penalties,
            candidate_rank,
            candidate_score,
            _,
        ) = evaluate_rows(tensor)
        improved = candidate_score < best_score
        if improved.any():
            best_tensor[improved] = tensor[improved]
            best_score[improved] = candidate_score[improved]
        tensor = best_tensor.clone()
        penalties, validity_rank, _, constraint_invalid = evaluate_rows(tensor)

    repair_diagnostics["neural_recovered_rows"] = int(
        (initially_invalid & validity_rank.eq(0)).sum().item()
    )

    # A deterministic grammar projection is the final safety net, not the
    # first repair. This gives the backbone every opportunity to produce a
    # chemically plausible local correction while guaranteeing that a simple
    # branch/ring/aromatic syntax failure cannot survive merely because the
    # stochastic repair sampled another bad token. Valid proposals are never
    # presented to this projector.
    if tracker is not None and hard_syntax_projection:
        best_sequences = _strip_padding(best_tensor, tk.pad_id)
        candidate_sequences = list(best_sequences)
        candidate_editable = list(editable_sequences)
        candidate_constraints = list(token_constraint_sequences)
        residual_rows = validity_rank.gt(0).nonzero(as_tuple=False).flatten().tolist()
        for row in residual_rows:
            (
                projected,
                projected_editable,
                projected_constraints,
                projection_stats,
            ) = tracker.syntax_tracker.project_completed_sequence(
                best_sequences[row],
                editable=editable_sequences[row],
                constraints=token_constraint_sequences[row],
            )
            repair_diagnostics.update(projection_stats)
            changed = projection_stats.get("sequence_changed", 0) > 0
            repair_diagnostics["hard_projection_rows"] += int(changed)
            candidate_sequences[row] = projected
            candidate_editable[row] = projected_editable
            candidate_constraints[row] = projected_constraints

        if residual_rows:
            candidate_tensor = _pad_sequences(
                candidate_sequences,
                tk.pad_id,
                device,
            )
            _, candidate_rank, candidate_score, _ = evaluate_rows(candidate_tensor)
            final_sequences = []
            for row, (original, candidate) in enumerate(
                zip(best_sequences, candidate_sequences)
            ):
                if candidate_score[row] < best_score[row]:
                    final_sequences.append(candidate)
                    best_score[row] = candidate_score[row]
                    validity_rank[row] = candidate_rank[row]
                else:
                    final_sequences.append(original)
            best_tensor = _pad_sequences(final_sequences, tk.pad_id, device)

        # Neural repair can improve a parser failure into a molecule that
        # parses but still fails sanitization. In that case, projecting only
        # the intermediate state would lose a valid deterministic projection
        # of the original proposal. Preserve that original branch explicitly
        # so stochastic repair can never lower the hard validity floor.
        residual_rows = validity_rank.gt(0).nonzero(as_tuple=False).flatten().tolist()
        if residual_rows:
            current_sequences = _strip_padding(best_tensor, tk.pad_id)
            original_fallbacks = list(current_sequences)
            for row in residual_rows:
                (
                    projected,
                    _,
                    _,
                    projection_stats,
                ) = tracker.syntax_tracker.project_completed_sequence(
                    sequences[row],
                    editable=editable_sequences[row],
                    constraints=token_constraint_sequences[row],
                )
                repair_diagnostics.update(projection_stats)
                original_fallbacks[row] = projected

            fallback_tensor = _pad_sequences(
                original_fallbacks,
                tk.pad_id,
                device,
            )
            _, fallback_rank, fallback_score, _ = evaluate_rows(fallback_tensor)
            final_sequences = []
            for row, (current, fallback) in enumerate(
                zip(current_sequences, original_fallbacks)
            ):
                if fallback_score[row] < best_score[row]:
                    final_sequences.append(fallback)
                    best_score[row] = fallback_score[row]
                    validity_rank[row] = fallback_rank[row]
                    repair_diagnostics["hard_original_fallback_rows"] += 1
                else:
                    final_sequences.append(current)
            best_tensor = _pad_sequences(final_sequences, tk.pad_id, device)

    repaired = _strip_padding(best_tensor, tk.pad_id)
    if return_diagnostics:
        for key in (
            "initial_invalid_rows",
            "initial_constraint_invalid_rows",
            "neural_repair_rows",
            "neural_repair_rounds",
            "neural_recovered_rows",
            "hard_projection_rows",
            "hard_original_fallback_rows",
            "final_constraint_invalid_rows",
        ):
            repair_diagnostics.setdefault(key, 0)
        repair_diagnostics["final_invalid_rows"] = int(validity_rank.gt(0).sum().item())
        _, _, _, final_constraint_invalid = evaluate_rows(best_tensor)
        repair_diagnostics["final_constraint_invalid_rows"] = int(
            final_constraint_invalid.sum().item()
        )
        repair_diagnostics["repair_time_start"] = float(repair_time_start)
        repair_diagnostics["repair_time_end"] = float(repair_time_end)
        return repaired, dict(repair_diagnostics)
    return repaired


def _normalize_local_infill_plans(
    tokens,
    plans,
    max_replacement_len,
):
    normalized = []
    for plan in plans if isinstance(plans, (list, tuple)) else [plans]:
        if not plan:
            continue
        start = max(0, min(len(tokens) - 1, int(plan.get("start", 0))))
        stop = max(start + 1, min(len(tokens), int(plan.get("stop", start + 1))))
        removed = stop - start
        minimum = int(plan.get("min_replacement_len", 0))
        maximum = int(plan.get("max_replacement_len", max_replacement_len))
        initial = int(plan.get("initial_replacement_len", 0))
        constraint = plan.get("token_constraint")
        if constraint not in (None, "chain_atom", "atom_bounded"):
            raise ValueError(f"Unsupported token constraint: {constraint!r}")
        minimum = max(0, minimum)
        maximum = max(minimum, maximum)
        initial = max(0, min(initial, maximum))
        normalized.append(
            {
                "start": start,
                "stop": stop,
                "removed": removed,
                "minimum": minimum,
                "maximum": maximum,
                "initial": initial,
                "constraint": constraint,
            }
        )

    normalized.sort(key=lambda row: (row["start"], row["stop"]))
    merged = []
    for row in normalized:
        if not merged or row["start"] > merged[-1]["stop"]:
            merged.append(dict(row))
            continue
        previous = merged[-1]
        previous["stop"] = max(previous["stop"], row["stop"])
        previous["removed"] = previous["stop"] - previous["start"]
        previous["minimum"] += row["minimum"]
        previous["maximum"] += row["maximum"]
        previous["initial"] += row["initial"]
        if previous["constraint"] != row["constraint"]:
            previous["constraint"] = None
    return merged


def _build_local_infill_state(
    smiles,
    plans,
    tk,
    max_len,
    recursive_gap_insertions=False,
):
    tokens = tokenize_smiles(smiles)[: max(0, int(max_len) - 2)]
    if not tokens:
        return None
    normalized = _normalize_local_infill_plans(
        tokens,
        plans,
        max_replacement_len=max_len,
    )
    if not normalized:
        return None

    removed_positions = set()
    for plan in normalized:
        removed_positions.update(range(plan["start"], plan["stop"]))
    kept = [
        (position, tk.vocab.get(token, getattr(tk, "unk_id", -1)))
        for position, token in enumerate(tokens)
        if position not in removed_positions
    ]
    if any(token_id is None or token_id < 0 for _, token_id in kept):
        return None

    sequence = [tk.bos_id] + [token_id for _, token_id in kept] + [tk.eos_id]
    editable = [False] * len(sequence)
    constraints = [None] * len(sequence)
    gap_ids = [-1] * len(sequence)
    anchors = [-1] * len(sequence)
    gaps = []
    kept_positions = [position for position, _ in kept]
    for gap_id, plan in enumerate(normalized):
        anchor_body_index = next(
            (
                index
                for index, position in enumerate(kept_positions)
                if position >= plan["stop"]
            ),
            len(kept),
        )
        anchor_index = anchor_body_index + 1
        if anchors[anchor_index] >= 0:
            existing = gaps[anchors[anchor_index]]
            existing["minimum"] += plan["minimum"]
            existing["maximum"] += plan["maximum"]
            existing["removed"] += plan["removed"]
            existing["initial"] += plan["initial"]
            if existing["constraint"] != plan["constraint"]:
                existing["constraint"] = None
            continue
        anchors[anchor_index] = len(gaps)
        gaps.append(
            {
                "minimum": int(plan["minimum"]),
                "maximum": int(plan["maximum"]),
                "removed": int(plan["removed"]),
                "initial": int(plan["initial"]),
                "inserted": 0,
                "constraint": plan["constraint"],
            }
        )

    available = max(0, int(max_len) - len(sequence))
    for gap in gaps:
        gap["maximum"] = min(gap["maximum"], gap["inserted"] + available)
        gap["minimum"] = min(gap["minimum"], gap["maximum"])
        gap["initial"] = min(gap["initial"], gap["maximum"])

    initial_insertions = []
    for position, gap_id in enumerate(anchors):
        if gap_id < 0:
            continue
        count = int(gaps[gap_id]["initial"])
        if count > 0:
            initial_insertions.append((position, gap_id, count))
    initial_insertions = _fit_gap_insertions_to_capacity(
        initial_insertions,
        available,
    )
    state = {
        "tokens": sequence,
        "editable": editable,
        "constraints": constraints,
        "gap_ids": gap_ids,
        "anchors": anchors,
        "gaps": gaps,
    }
    _apply_local_gap_insertions(
        state,
        initial_insertions,
        tk.mask_id,
        recursive_gap_insertions=recursive_gap_insertions,
    )
    state["trajectory"] = {
        "initial_inserted_tokens": sum(int(gap["inserted"]) for gap in state["gaps"]),
        "learned_inserted_tokens": 0,
        "insertion_event_sites": 0,
        "insertion_steps": 0,
        "unmask_events": 0,
        "forced_final_unmasks": 0,
        "open_site_observations": 0,
        "open_site_rate_sum": 0.0,
        "max_open_sites": sum(anchor >= 0 for anchor in state["anchors"]),
        "max_sequence_tokens": len(state["tokens"]),
        "online_fsm_repair_events": 0,
        "online_fsm_remasked_tokens": 0,
    }
    return state


@torch.no_grad()
def sample_elastic_local_infill(
    model,
    tk,
    seed_smiles,
    edit_plans,
    max_len,
    device,
    batch_size=64,
    n_steps=120,
    use_fsm_check=True,
    use_rdkit_kekulize_check=True,
    max_sample_retries=2,
    violation_neighborhood=2,
    fsm_repair_progressive_steps=8,
    fsm_repair_prefer_localization=True,
    temperature_start=1.2,
    temperature_end=0.2,
    temperature_power=1.5,
    top_k=0,
    top_p=1.0,
    nucleus_min_tokens_start=1,
    nucleus_min_tokens_end=1,
    max_insertions_per_step=4,
    insertion_rate_scale=1.0,
    unmask_selection="top_prob",
    deterministic_final_unmask=True,
    recursive_gap_insertions=False,
    trajectory_mode="coupled",
    planning_fraction=0.5,
    fill_mode="absorbing",
    fill_remask_power=0.8,
    fill_gumbel_scale=0.65,
    sequence_validators=None,
    return_seed_indices=False,
    return_diagnostics=False,
):
    """Fill local gaps using the learned insertion dynamics.

    The caller chooses which span may change.  The model chooses how many
    tokens to put back, so one operation can shrink, preserve, or grow the
    sequence without a hand-written length delta. Only explicit per-plan hard
    limits and the global ``max_len`` constrain replacement length. With
    ``recursive_gap_insertions=True``, every token materialized inside an
    editable gap remains an insertion site. This matches the reverse process
    used during elastic training while fixed context stays closed.

    ``trajectory_mode="plan_then_fill"`` separates the two learned event
    families without introducing a hand-written length prior. The insertion
    head first plans the editable gap lengths while all new slots remain
    masked. Length is then frozen and the token head jointly fills those slots.
    This is useful for syntax-coupled representations such as ordinary SMILES,
    where committing token content while a local span is still growing can
    strand unmatched branch or ring syntax.
    """
    if not getattr(model, "is_elastic", False):
        return []
    if not seed_smiles or edit_plans is None:
        return []
    if sequence_validators is not None and len(sequence_validators) != len(seed_smiles):
        raise ValueError("sequence_validators must match seed_smiles")
    if n_steps <= 0:
        raise ValueError("n_steps must be positive")
    if float(insertion_rate_scale) <= 0.0:
        raise ValueError("insertion_rate_scale must be positive")
    if int(fsm_repair_progressive_steps) < 1:
        raise ValueError("fsm_repair_progressive_steps must be positive")
    if int(nucleus_min_tokens_start) < 1 or int(nucleus_min_tokens_end) < 1:
        raise ValueError("nucleus minimum token counts must be positive")
    if trajectory_mode not in {"coupled", "plan_then_fill"}:
        raise ValueError(
            "trajectory_mode must be 'coupled' or 'plan_then_fill', "
            f"got {trajectory_mode!r}"
        )
    if trajectory_mode == "plan_then_fill":
        if n_steps < 4:
            raise ValueError("plan_then_fill requires at least four steps")
        if not 0.0 < float(planning_fraction) < 1.0:
            raise ValueError("planning_fraction must be in (0, 1)")
    if fill_mode not in {"absorbing", "progressive_remask"}:
        raise ValueError(
            f"fill_mode must be 'absorbing' or 'progressive_remask', got {fill_mode!r}"
        )
    if fill_mode == "progressive_remask":
        if trajectory_mode != "plan_then_fill":
            raise ValueError(
                "progressive_remask requires trajectory_mode='plan_then_fill'"
            )
        if float(fill_remask_power) <= 0.0:
            raise ValueError("fill_remask_power must be positive")
        if float(fill_gumbel_scale) < 0.0:
            raise ValueError("fill_gumbel_scale must be non-negative")

    model.eval()
    schedule = ElasticKumaSchedule(shape_a=float(getattr(model, "kuma_shape_a", 2.0)))
    local_fsm_tracker = ValenceFSMTracker(tk) if use_fsm_check else None
    local_rdkit_checker = (
        prepare_rdkit_kekulize_checker(tk, local_fsm_tracker)
        if use_rdkit_kekulize_check
        else None
    )
    generated = []
    for offset in range(0, len(seed_smiles), max(1, int(batch_size))):
        seeds = seed_smiles[offset : offset + batch_size]
        plans = edit_plans[offset : offset + batch_size]
        states = []
        source_indices = []
        state_validators = []
        for local_index, (smiles, plan) in enumerate(zip(seeds, plans)):
            state = _build_local_infill_state(
                smiles,
                plan,
                tk=tk,
                max_len=max_len,
                recursive_gap_insertions=recursive_gap_insertions,
            )
            if state is None:
                continue
            states.append(state)
            source_indices.append(offset + local_index)
            if sequence_validators is not None:
                state_validators.append(sequence_validators[offset + local_index])
        if not states:
            continue

        planning_steps = (
            max(2, min(n_steps - 2, int(round(n_steps * planning_fraction))))
            if trajectory_mode == "plan_then_fill"
            else 0
        )
        for step in range(n_steps):
            if trajectory_mode == "plan_then_fill":
                planning = step < planning_steps
                phase_step = step if planning else step - planning_steps
                phase_steps = planning_steps if planning else n_steps - planning_steps
                allow_unmask = not planning
                # Match the LoFlex terminal convention: the last integration
                # point of a length trajectory does not create fresh masks.
                allow_insert = planning and phase_step < phase_steps - 1
            else:
                planning = False
                phase_step = step
                phase_steps = n_steps
                allow_unmask = True
                # Match the terminal tau-leaping convention used by LoFlexMDM:
                # the final integration point only resolves remaining masks.
                allow_insert = step < n_steps - 1
            time = max(
                schedule.eps,
                min(1.0 - schedule.eps, (phase_step + 0.5) / phase_steps),
            )
            next_time = max(
                time,
                min(
                    1.0 - schedule.eps,
                    (phase_step + 1.5) / phase_steps,
                ),
            )
            progress = phase_step / max(phase_steps - 1, 1)
            nucleus_min_tokens = max(
                1,
                int(
                    round(
                        float(nucleus_min_tokens_end)
                        + (
                            float(nucleus_min_tokens_start)
                            - float(nucleus_min_tokens_end)
                        )
                        * (1.0 - progress)
                    )
                ),
            )
            temperature = (
                temperature_end
                + (temperature_start - temperature_end)
                * (1.0 - progress) ** temperature_power
            )

            tensor = _pad_sequences(
                [state["tokens"] for state in states],
                tk.pad_id,
                device,
            )
            attention = tensor.ne(tk.pad_id).long()
            time_tensor = torch.full(
                (len(states),),
                time,
                device=device,
                dtype=torch.float32,
            )
            rates = model(
                tensor,
                attention,
                t=time_tensor,
                return_aux=True,
                rate_family="theta",
            )
            editable = _pad_boolean_sequences(
                [state["editable"] for state in states],
                device,
            )
            current_masks = tensor.eq(tk.mask_id) & editable
            if allow_unmask:
                logits = _constrain_token_logits(
                    rates["logits"],
                    _position_constraint_sequences(states),
                    tk,
                )
                sampled_tokens, confidence = _sample_tokens(
                    logits,
                    tk,
                    temperature=temperature,
                    top_k=top_k,
                    top_p=top_p,
                    min_tokens_to_keep=nucleus_min_tokens,
                    return_confidence=True,
                )
                if fill_mode == "progressive_remask":
                    # Fill every currently masked slot, then return the lowest
                    # confidence editable slots to mask. This is a single
                    # progressive trajectory, not best-of-N filtering.
                    unmask_events = current_masks
                else:
                    unmask_hazard = schedule.hazard(
                        time_tensor.unsqueeze(-1),
                        rates["b_unmask"],
                    )
                    unmask_probability = 1.0 - torch.exp(
                        -unmask_hazard * (next_time - time)
                    )
                    unmask_events = _select_unmask_events(
                        current_mask=current_masks,
                        unmask_probability=unmask_probability,
                        confidence=confidence,
                        selection=unmask_selection,
                    )
                tensor.masked_scatter_(
                    unmask_events,
                    sampled_tokens[unmask_events],
                )
            else:
                unmask_events = torch.zeros_like(current_masks)
            token_rows = tensor.detach().cpu().tolist()
            unmask_counts = (
                unmask_events.sum(dim=1).detach().cpu().tolist()
                if return_diagnostics
                else None
            )
            for row, state in enumerate(states):
                state["tokens"] = token_rows[row][: len(state["tokens"])]
                if unmask_counts is not None:
                    state["trajectory"]["unmask_events"] += int(unmask_counts[row])

            if allow_unmask and fill_mode == "progressive_remask":
                confidence_scores = torch.log(confidence.clamp_min(1e-8))
                confidence_scores = confidence_scores + 0.10 * torch.log(
                    rates["b_unmask"].float().clamp_min(1e-8)
                )
                score_tensor = torch.full_like(
                    confidence_scores,
                    -torch.inf,
                )
                for row, state in enumerate(states):
                    stored = state.get("fill_scores")
                    if stored is None:
                        stored = [
                            -math.inf if is_editable else 0.0
                            for is_editable in state["editable"]
                        ]
                    score_tensor[row, : len(stored)] = torch.tensor(
                        stored,
                        device=device,
                        dtype=score_tensor.dtype,
                    )
                score_tensor.masked_scatter_(
                    unmask_events,
                    confidence_scores[unmask_events],
                )

                if phase_step < phase_steps - 1:
                    progress_after_step = (phase_step + 1) / phase_steps
                    remask_rate = math.cos(
                        progress_after_step * math.pi * 0.5
                    ) ** float(fill_remask_power)
                    editable_counts = editable.sum(dim=1)
                    remask_counts = (editable_counts.float() * remask_rate).long()
                    ranking_scores = score_tensor.masked_fill(
                        ~editable,
                        1000.0,
                    )
                    if fill_gumbel_scale > 0.0 and remask_rate > 0.0:
                        uniform = torch.rand_like(ranking_scores).clamp_(
                            min=1e-10,
                            max=1.0 - 1e-10,
                        )
                        gumbel = -torch.log(-torch.log(uniform))
                        ranking_scores = ranking_scores + (
                            gumbel * float(fill_gumbel_scale) * remask_rate
                        )
                    ordering = ranking_scores.argsort(dim=1)
                    ranks = torch.empty_like(ordering)
                    rank_values = (
                        torch.arange(
                            ordering.size(1),
                            device=device,
                        )
                        .unsqueeze(0)
                        .expand_as(ordering)
                    )
                    ranks.scatter_(1, ordering, rank_values)
                    remask = (ranks < remask_counts.unsqueeze(1)) & editable
                    tensor.masked_fill_(remask, tk.mask_id)
                    score_tensor.masked_fill_(remask, -torch.inf)

                token_rows = tensor.detach().cpu().tolist()
                score_rows = score_tensor.detach().cpu().tolist()
                for row, state in enumerate(states):
                    state["tokens"] = token_rows[row][: len(state["tokens"])]
                    state["fill_scores"] = score_rows[row][: len(state["tokens"])]

            # Once a local trajectory becomes fully decoded, apply the same
            # late transactional scan used by de novo generation. Protected
            # context is never remasked; only editable positions in the local
            # gap can be returned to the model.
            local_fsm_start = int(0.8 * phase_steps)
            should_check_fsm = (
                allow_unmask
                and local_fsm_tracker is not None
                and phase_step >= local_fsm_start
                and (phase_step % 5 == 0 or phase_step == phase_steps - 1)
            )
            should_check_rdkit = (
                allow_unmask
                and local_rdkit_checker is not None
                and phase_step >= local_fsm_start
                and (phase_step % 10 == 0 or phase_step == phase_steps - 1)
            )
            if should_check_fsm or should_check_rdkit:
                complete_rows = ~(tensor.eq(tk.mask_id) & attention.bool()).any(dim=1)
                complete_indices = complete_rows.nonzero(as_tuple=False).flatten()
                penalties = torch.zeros_like(tensor, dtype=torch.float)
                if complete_indices.numel() > 0:
                    if should_check_fsm:
                        penalties[complete_indices] += (
                            local_fsm_tracker.compute_penalties(
                                tensor[complete_indices]
                            )
                        )
                    if should_check_rdkit:
                        chemistry, focus_ids = local_rdkit_checker
                        penalties[complete_indices] += compute_rdkit_kekulize_penalties(
                            tensor[complete_indices],
                            tk,
                            chemistry,
                            focus_ids,
                        )
                violation_positions = penalties < 0
                editable_decoded = editable & attention.bool() & tensor.ne(tk.mask_id)
                repair_mask = expand_violation_mask(
                    violation_positions,
                    editable_decoded,
                    radius=violation_neighborhood,
                )
                if repair_mask.any():
                    tensor.masked_fill_(repair_mask, tk.mask_id)
                    token_rows = tensor.detach().cpu().tolist()
                    repair_counts = repair_mask.sum(dim=1).cpu().tolist()
                    for row, state in enumerate(states):
                        state["tokens"] = token_rows[row][: len(state["tokens"])]
                        if repair_counts[row] <= 0:
                            continue
                        state["trajectory"]["online_fsm_repair_events"] += 1
                        state["trajectory"]["online_fsm_remasked_tokens"] += int(
                            repair_counts[row]
                        )
                        if "fill_scores" in state:
                            for position in (
                                repair_mask[row]
                                .nonzero(as_tuple=False)
                                .flatten()
                                .tolist()
                            ):
                                if position < len(state["fill_scores"]):
                                    state["fill_scores"][position] = -math.inf

            open_positions_by_row = []
            open_gap_ids_by_row = []
            for state in states:
                open_positions = []
                open_gap_ids = []
                for position, gap_id in enumerate(state["anchors"]):
                    if not allow_insert:
                        break
                    if gap_id < 0:
                        continue
                    gap = state["gaps"][gap_id]
                    remaining = min(
                        gap["maximum"] - gap["inserted"],
                        int(max_len) - len(state["tokens"]),
                    )
                    if remaining <= 0:
                        continue
                    open_positions.append(position)
                    open_gap_ids.append(gap_id)
                open_positions_by_row.append(open_positions)
                open_gap_ids_by_row.append(open_gap_ids)

            recursive_counts = None
            recursive_rate_sums = None
            if recursive_gap_insertions:
                open_mask_cpu = torch.zeros(
                    rates["b_ins"].shape,
                    dtype=torch.bool,
                )
                for row, positions in enumerate(open_positions_by_row):
                    if positions:
                        open_mask_cpu[row, positions] = True
                open_mask = open_mask_cpu.to(device=device)
                insertion_rates = rates["b_ins"].float() * float(insertion_rate_scale)
                insertion_hazard = schedule.hazard(
                    time_tensor.unsqueeze(-1),
                    insertion_rates,
                )
                insertion_intensity = insertion_hazard * (next_time - time)
                insertion_intensity.masked_fill_(~open_mask, 0.0)
                sampled_counts = torch.poisson(insertion_intensity).long()
                sampled_counts.clamp_(max=max(0, int(max_insertions_per_step)))
                recursive_counts = sampled_counts.detach().cpu().tolist()
                if return_diagnostics:
                    recursive_rate_sums = (
                        insertion_rates.masked_fill(~open_mask, 0.0)
                        .sum(dim=1)
                        .detach()
                        .cpu()
                        .tolist()
                    )

            for row, state in enumerate(states):
                insertions = []
                open_positions = open_positions_by_row[row]
                open_gap_ids = open_gap_ids_by_row[row]

                open_site_count = len(open_positions)
                open_site_rate_sum = 0.0
                if open_positions:
                    if recursive_gap_insertions:
                        counts = recursive_counts[row]
                        if recursive_rate_sums is not None:
                            open_site_rate_sum = float(recursive_rate_sums[row])
                        for position, gap_id, count in zip(
                            open_positions,
                            open_gap_ids,
                            (counts[position] for position in open_positions),
                        ):
                            if count > 0:
                                insertions.append((position, gap_id, count))
                    else:
                        # Preserve the legacy random-number consumption order
                        # for PMO/Lead callers that do not opt into recursion.
                        for position, gap_id in zip(
                            open_positions,
                            open_gap_ids,
                        ):
                            gap = state["gaps"][gap_id]
                            remaining = min(
                                gap["maximum"] - gap["inserted"],
                                int(max_len) - len(state["tokens"]),
                            )
                            rate = rates["b_ins"][row, position].float() * float(
                                insertion_rate_scale
                            )
                            if return_diagnostics:
                                open_site_rate_sum += float(rate.item())
                            hazard = schedule.hazard(
                                torch.tensor(time, device=device),
                                rate,
                            )
                            count = int(
                                torch.poisson(hazard * (next_time - time)).item()
                            )
                            count = min(
                                count,
                                max(0, int(max_insertions_per_step)),
                                remaining,
                            )
                            if count > 0:
                                insertions.append((position, gap_id, count))

                insertions = _fit_gap_insertions_to_capacity(
                    insertions,
                    int(max_len) - len(state["tokens"]),
                    gap_capacities={
                        gap_id: gap["maximum"] - gap["inserted"]
                        for gap_id, gap in enumerate(state["gaps"])
                    },
                )
                inserted = _apply_local_gap_insertions(
                    state,
                    insertions,
                    tk.mask_id,
                    recursive_gap_insertions=recursive_gap_insertions,
                )
                trajectory = state["trajectory"]
                trajectory["learned_inserted_tokens"] += inserted
                trajectory["insertion_event_sites"] += len(insertions)
                trajectory["insertion_steps"] += int(inserted > 0)
                if return_diagnostics:
                    trajectory["open_site_observations"] += open_site_count
                    trajectory["open_site_rate_sum"] += open_site_rate_sum
                trajectory["max_open_sites"] = max(
                    trajectory["max_open_sites"],
                    sum(anchor >= 0 for anchor in state["anchors"]),
                )
                trajectory["max_sequence_tokens"] = max(
                    trajectory["max_sequence_tokens"],
                    len(state["tokens"]),
                )

        for state in states:
            minimum_insertions = []
            for position, gap_id in enumerate(state["anchors"]):
                if gap_id < 0:
                    continue
                gap = state["gaps"][gap_id]
                missing = min(
                    max(0, gap["minimum"] - gap["inserted"]),
                    int(max_len) - len(state["tokens"]),
                )
                if missing <= 0:
                    continue
                minimum_insertions.append((position, gap_id, missing))
            minimum_insertions = _fit_gap_insertions_to_capacity(
                minimum_insertions,
                int(max_len) - len(state["tokens"]),
                gap_capacities={
                    gap_id: gap["maximum"] - gap["inserted"]
                    for gap_id, gap in enumerate(state["gaps"])
                },
            )
            inserted = _apply_local_gap_insertions(
                state,
                minimum_insertions,
                tk.mask_id,
                recursive_gap_insertions=recursive_gap_insertions,
            )
            state["trajectory"]["learned_inserted_tokens"] += inserted

        tensor = _pad_sequences(
            [state["tokens"] for state in states],
            tk.pad_id,
            device,
        )
        attention = tensor.ne(tk.pad_id).long()
        remaining_masks = tensor.eq(tk.mask_id)
        forced_unmask_counts = (
            remaining_masks.sum(dim=1).detach().cpu().tolist()
            if return_diagnostics
            else None
        )
        if remaining_masks.any():
            tensor, _ = _progressive_final_unmask(
                model=model,
                tk=tk,
                tensor=tensor,
                constraint_sequences=_position_constraint_sequences(states),
                temperature=temperature_end,
                top_k=top_k,
                top_p=top_p,
                deterministic=deterministic_final_unmask,
                progressive_steps=fsm_repair_progressive_steps,
            )
        token_rows = tensor.detach().cpu().tolist()
        for row, state in enumerate(states):
            if forced_unmask_counts is not None:
                state["trajectory"]["forced_final_unmasks"] = int(
                    forced_unmask_counts[row]
                )
            state["tokens"] = token_rows[row][: len(state["tokens"])]

        repaired = _repair_final_sequences(
            model=model,
            tk=tk,
            sequences=[state["tokens"] for state in states],
            device=device,
            use_fsm_check=use_fsm_check,
            use_rdkit_kekulize_check=use_rdkit_kekulize_check,
            max_sample_retries=max_sample_retries,
            violation_neighborhood=violation_neighborhood,
            temperature=temperature_end,
            top_k=top_k,
            top_p=top_p,
            editable_sequences=[state["editable"] for state in states],
            token_constraint_sequences=[
                sequence for sequence in _position_constraint_sequences(states)
            ],
            progressive_steps=fsm_repair_progressive_steps,
            prefer_fsm_localization=fsm_repair_prefer_localization,
            sequence_validators=(
                state_validators if sequence_validators is not None else None
            ),
            return_diagnostics=return_diagnostics,
        )
        if return_diagnostics:
            sequences, final_repair_diagnostics = repaired
        else:
            sequences = repaired
            final_repair_diagnostics = {}
        for source_index, state, sequence in zip(
            source_indices,
            states,
            sequences,
        ):
            if tk.eos_id in sequence:
                sequence = sequence[: sequence.index(tk.eos_id) + 1]
            smiles = tk.decode(sequence).strip("'\"")
            diagnostics = {
                "length_mode": (
                    "learned_recursive_insertion"
                    if recursive_gap_insertions
                    else "learned_insertion"
                ),
                "trajectory_mode": trajectory_mode,
                "planning_steps": planning_steps,
                "fill_mode": fill_mode,
                "insertion_rate_scale": float(insertion_rate_scale),
                "fsm_repair_progressive_steps": int(fsm_repair_progressive_steps),
                "fsm_repair_prefer_localization": bool(fsm_repair_prefer_localization),
                "fsm_constraint_mode": (
                    "online_scan_progressive_completion_neural_then_projection"
                    if use_fsm_check
                    else "disabled"
                ),
                "fsm_check_enabled": bool(use_fsm_check),
                "rdkit_kekulize_check_enabled": bool(use_rdkit_kekulize_check),
                "condition_validator_enabled": bool(sequence_validators is not None),
                "nucleus_min_tokens_start": int(nucleus_min_tokens_start),
                "nucleus_min_tokens_end": int(nucleus_min_tokens_end),
                "removed_tokens": sum(int(gap["removed"]) for gap in state["gaps"]),
                "inserted_tokens": sum(int(gap["inserted"]) for gap in state["gaps"]),
                "chain_atom_constrained_tokens": sum(
                    constraint == "chain_atom"
                    for constraint in _position_constraint_sequences([state])[0]
                ),
            }
            trajectory = state["trajectory"]
            diagnostics.update(
                {
                    f"condition_repair_{key}": value
                    for key, value in sorted(final_repair_diagnostics.items())
                    if "constraint" in key
                }
            )
            diagnostics.update(
                {
                    "initial_inserted_tokens": int(
                        trajectory["initial_inserted_tokens"]
                    ),
                    "learned_inserted_tokens": int(
                        trajectory["learned_inserted_tokens"]
                    ),
                    "insertion_event_sites": int(trajectory["insertion_event_sites"]),
                    "insertion_steps": int(trajectory["insertion_steps"]),
                    "unmask_events": int(trajectory["unmask_events"]),
                    "forced_final_unmasks": int(trajectory["forced_final_unmasks"]),
                    "max_open_sites": int(trajectory["max_open_sites"]),
                    "max_sequence_tokens": int(trajectory["max_sequence_tokens"]),
                    "online_fsm_repair_events": int(
                        trajectory["online_fsm_repair_events"]
                    ),
                    "online_fsm_remasked_tokens": int(
                        trajectory["online_fsm_remasked_tokens"]
                    ),
                    "mean_open_site_rate": (
                        float(trajectory["open_site_rate_sum"])
                        / max(
                            1,
                            int(trajectory["open_site_observations"]),
                        )
                    ),
                }
            )
            diagnostics["actual_delta"] = (
                diagnostics["inserted_tokens"] - diagnostics["removed_tokens"]
            )
            if return_seed_indices and return_diagnostics:
                generated.append((smiles, source_index, diagnostics))
            elif return_seed_indices:
                generated.append((smiles, source_index))
            elif return_diagnostics:
                generated.append((smiles, diagnostics))
            else:
                generated.append(smiles)
    return generated


@torch.no_grad()
def sample_elastic_csdnet(
    model,
    tk,
    ref_lengths,
    n_mol,
    cond=None,
    w=2.0,
    device="cuda",
    batch_size=128,
    n_steps=500,
    use_fsm_check=True,
    use_rdkit_kekulize_check=True,
    rdkit_check_interval=25,
    max_sample_retries=3,
    violation_neighborhood=2,
    temperature_start=1.5,
    temperature_end=0.25,
    temperature_power=1.5,
    top_k=0,
    top_p=1.0,
    gumbel_scale=1.0,
    length_quantile_low=0.0,
    length_quantile_high=1.0,
    length_min=0,
    length_max=0,
    unmask_selection="top_prob",
    strict_final_sanitize=False,
    max_refill_factor=1.25,
    deterministic_final_unmask=True,
    return_diagnostics=False,
):
    """
    Generate molecules by alternating learned gap insertion and token unmasking.

    ref_lengths is accepted for API compatibility but intentionally unused:
    sequence length is generated by the learned insertion process.
    """
    del (
        ref_lengths,
        gumbel_scale,
        length_quantile_low,
        length_quantile_high,
    )
    if n_steps <= 0:
        raise ValueError("n_steps must be positive")
    if float(max_refill_factor) < 1.0:
        raise ValueError("max_refill_factor must be at least 1.0")
    if int(rdkit_check_interval) < 1:
        raise ValueError("rdkit_check_interval must be positive")
    model.eval()
    maximum_length = int(length_max) if int(length_max) > 0 else 256
    maximum_length = max(3, maximum_length)
    minimum_length = max(3, int(length_min)) if int(length_min) > 0 else 3
    schedule = ElasticKumaSchedule(shape_a=float(getattr(model, "kuma_shape_a", 2.0)))

    fsm_tracker = ValenceFSMTracker(tk) if use_fsm_check else None
    rdkit_checker = (
        prepare_rdkit_kekulize_checker(tk, fsm_tracker)
        if use_rdkit_kekulize_check
        else None
    )
    final_checker = None
    if strict_final_sanitize:
        final_checker = rdkit_checker or prepare_rdkit_kekulize_checker(tk)
        if final_checker is None:
            raise RuntimeError("strict_final_sanitize requires RDKit")

    generated = []
    proposal_count = 0
    empty_rejections = 0
    sanitization_rejections = 0
    fsm_repair_diagnostics = Counter()
    online_fsm_diagnostics = Counter()
    max_proposals = max(
        n_mol,
        int(math.ceil(n_mol * float(max_refill_factor))),
    )
    while len(generated) < n_mol:
        remaining_proposals = max_proposals - proposal_count
        if remaining_proposals <= 0:
            raise RuntimeError(
                "Final sanitization refill budget exhausted: "
                f"accepted={len(generated)}/{n_mol}, proposals={proposal_count}"
            )
        current_batch_size = min(
            batch_size,
            n_mol - len(generated),
            remaining_proposals,
        )
        proposal_count += current_batch_size
        sequences = [[tk.bos_id, tk.eos_id] for _ in range(current_batch_size)]
        cond_batch = None
        if cond is not None:
            cond_batch = cond.repeat(current_batch_size, 1).to(device)

        for step in range(n_steps):
            time = max(
                schedule.eps,
                min(1.0 - schedule.eps, (step + 0.5) / n_steps),
            )
            next_time = max(
                time,
                min(1.0 - schedule.eps, (step + 1.5) / n_steps),
            )
            progress = step / max(n_steps - 1, 1)
            temperature = (
                temperature_end
                + (temperature_start - temperature_end)
                * (1.0 - progress) ** temperature_power
            )

            tensor = _pad_sequences(sequences, tk.pad_id, device)
            attention = tensor.ne(tk.pad_id).long()
            time_tensor = torch.full(
                (current_batch_size,),
                time,
                device=device,
                dtype=torch.float32,
            )
            if cond_batch is not None:
                conditional = model(
                    tensor,
                    attention,
                    cond=cond_batch,
                    drop_cond=False,
                    t=time_tensor,
                    return_aux=True,
                )
                unconditional = model(
                    tensor,
                    attention,
                    cond=cond_batch,
                    drop_cond=True,
                    t=time_tensor,
                    return_aux=True,
                )
                logits = w * conditional["logits"] + (1.0 - w) * unconditional["logits"]
                rates = conditional
            else:
                rates = model(
                    tensor,
                    attention,
                    t=time_tensor,
                    return_aux=True,
                    rate_family="theta",
                )
                logits = rates["logits"]

            sampled_tokens, confidence = _sample_tokens(
                logits,
                tk,
                temperature=temperature,
                top_k=top_k,
                top_p=top_p,
                return_confidence=True,
            )
            current_mask = tensor.eq(tk.mask_id) & attention.bool()
            unmask_hazard = schedule.hazard(
                time_tensor.unsqueeze(-1),
                rates["b_unmask"],
            )
            unmask_probability = 1.0 - torch.exp(-unmask_hazard * (next_time - time))
            unmask_event = _select_unmask_events(
                current_mask=current_mask,
                unmask_probability=unmask_probability,
                confidence=confidence,
                selection=unmask_selection,
            )
            tensor.masked_scatter_(
                unmask_event,
                sampled_tokens[unmask_event],
            )
            fsm_start_step = int(0.8 * n_steps)
            should_check_fsm = (
                fsm_tracker is not None
                and step >= fsm_start_step
                and (step % 5 == 0 or step == n_steps - 1)
            )
            should_check_rdkit = (
                rdkit_checker is not None
                and step >= fsm_start_step
                and (step % int(rdkit_check_interval) == 0 or step == n_steps - 1)
            )
            if should_check_fsm or should_check_rdkit:
                complete_rows = ~(tensor.eq(tk.mask_id) & attention.bool()).any(dim=1)
                complete_indices = complete_rows.nonzero(as_tuple=False).flatten()
                decoded_positions = (
                    attention.bool()
                    & tensor.ne(tk.pad_id)
                    & tensor.ne(tk.bos_id)
                    & tensor.ne(tk.eos_id)
                    & tensor.ne(tk.mask_id)
                )
                penalties = torch.zeros_like(tensor, dtype=torch.float)
                if complete_indices.numel() > 0:
                    if should_check_fsm:
                        penalties[complete_indices] += fsm_tracker.compute_penalties(
                            tensor[complete_indices]
                        )
                        online_fsm_diagnostics["fsm_checked_rows"] += int(
                            complete_indices.numel()
                        )
                    if should_check_rdkit:
                        chemistry, focus_ids = rdkit_checker
                        penalties[complete_indices] += compute_rdkit_kekulize_penalties(
                            tensor[complete_indices],
                            tk,
                            chemistry,
                            focus_ids,
                        )
                        online_fsm_diagnostics["rdkit_checked_rows"] += int(
                            complete_indices.numel()
                        )
                violation_positions = (penalties < 0) & decoded_positions
                if violation_positions.any():
                    affected_rows = violation_positions.any(dim=1)
                    repair_mask = expand_violation_mask(
                        violation_positions,
                        decoded_positions,
                        radius=violation_neighborhood,
                    )
                    tensor.masked_fill_(repair_mask, tk.mask_id)
                    online_fsm_diagnostics["online_repair_events"] += 1
                    online_fsm_diagnostics["online_repair_rows"] += int(
                        affected_rows.sum().item()
                    )
                    online_fsm_diagnostics["online_remasked_tokens"] += int(
                        repair_mask.sum().item()
                    )
            sequences = _strip_padding(tensor, tk.pad_id)

            if step < n_steps - 1:
                insertion_hazard = schedule.hazard(
                    time_tensor.unsqueeze(-1),
                    rates["b_ins"],
                )
                insertion_counts = torch.poisson(
                    insertion_hazard * (next_time - time)
                ).long()
                insertion_counts = insertion_counts.clamp(max=4)
                insertion_counts = insertion_counts * attention.long()
                sequences = _apply_insertions(
                    sequences,
                    insertion_counts.detach().cpu().tolist(),
                    mask_id=tk.mask_id,
                    max_length=maximum_length,
                )

        for row, sequence in enumerate(sequences):
            missing = max(0, minimum_length - len(sequence))
            if missing:
                sequence[-1:-1] = [tk.mask_id] * missing
                sequences[row] = sequence[:maximum_length]
        tensor = _pad_sequences(sequences, tk.pad_id, device)
        attention = tensor.ne(tk.pad_id).long()
        remaining_masks = tensor.eq(tk.mask_id)
        if remaining_masks.any():
            tensor, final_unmask_diagnostics = _progressive_final_unmask(
                model=model,
                tk=tk,
                tensor=tensor,
                cond=cond_batch,
                guidance_weight=w,
                temperature=temperature_end,
                top_k=top_k,
                top_p=top_p,
                deterministic=deterministic_final_unmask,
                progressive_steps=8,
            )
            online_fsm_diagnostics.update(final_unmask_diagnostics)
        sequences = _strip_padding(tensor, tk.pad_id)
        sequences, batch_repair_diagnostics = _repair_final_sequences(
            model=model,
            tk=tk,
            sequences=sequences,
            device=device,
            use_fsm_check=use_fsm_check,
            use_rdkit_kekulize_check=use_rdkit_kekulize_check,
            max_sample_retries=max_sample_retries,
            violation_neighborhood=violation_neighborhood,
            temperature=temperature_end,
            top_k=top_k,
            top_p=top_p,
            progressive_steps=8,
            prefer_fsm_localization=True,
            hard_syntax_projection=True,
            repair_time_start=0.65,
            repair_time_end=0.95,
            return_diagnostics=True,
        )
        fsm_repair_diagnostics.update(batch_repair_diagnostics)

        for sequence in sequences:
            if tk.eos_id in sequence:
                sequence = sequence[: sequence.index(tk.eos_id) + 1]
            smiles = tk.decode(sequence).strip("'\"")
            if not smiles:
                empty_rejections += 1
                continue
            if strict_final_sanitize and not rdkit_smiles_is_valid(
                smiles,
                final_checker[0],
            ):
                sanitization_rejections += 1
                continue
            generated.append(smiles)
            if len(generated) >= n_mol:
                break

    if return_diagnostics:
        return generated, {
            "proposals": proposal_count,
            "accepted": len(generated),
            "empty_rejections": empty_rejections,
            "sanitization_rejections": sanitization_rejections,
            "strict_final_sanitize": bool(strict_final_sanitize),
            "fsm_tracker_active": fsm_tracker is not None,
            "rdkit_constraint_active": rdkit_checker is not None,
            "fsm_constraint_mode": (
                "online_scan_progressive_completion_neural_then_projection"
                if use_fsm_check
                else "disabled"
            ),
            "fsm_repair_progressive_steps": 8,
            "fsm_repair_time_start": 0.65,
            "fsm_repair_time_end": 0.95,
            **{
                f"fsm_{key}": value
                for key, value in sorted(fsm_repair_diagnostics.items())
            },
            **{
                f"fsm_online_{key}": value
                for key, value in sorted(online_fsm_diagnostics.items())
            },
        }
    return generated
