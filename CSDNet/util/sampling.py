import math
from collections import Counter

import numpy as np
import torch

from CSDNet.util.edit_schedule_sampling import sample_de_novo_lengths

from CSDNet.util.fsm import (
    ValenceFSMTracker,
    compute_rdkit_kekulize_penalties,
    expand_violation_mask,
    prepare_rdkit_kekulize_checker,
    rdkit_smiles_is_valid,
)


def _prepare_length_pool(
    ref_lengths,
    length_quantile_low=0.0,
    length_quantile_high=1.0,
    length_min=0,
    length_max=0,
):
    arr = np.asarray([int(x) for x in ref_lengths if int(x) >= 3], dtype=np.int64)
    if arr.size == 0:
        raise ValueError("ref_lengths must contain at least one valid sequence length")

    low_q = float(np.clip(length_quantile_low, 0.0, 1.0))
    high_q = float(np.clip(length_quantile_high, low_q, 1.0))
    lo = int(np.floor(np.quantile(arr, low_q)))
    hi = int(np.ceil(np.quantile(arr, high_q)))
    if length_min and length_min > 0:
        lo = max(lo, int(length_min))
    if length_max and length_max > 0:
        hi = min(hi, int(length_max))

    pool = arr[(arr >= lo) & (arr <= hi)]
    if pool.size == 0:
        return arr.tolist()
    return pool.tolist()


def _filter_sampling_logits(logits, top_k=0, top_p=1.0):
    if (not top_k or top_k <= 0) and (top_p is None or top_p >= 1.0):
        return logits

    filtered = logits
    vocab_size = filtered.size(-1)
    if top_k and top_k > 0 and top_k < vocab_size:
        kth = torch.topk(filtered, top_k, dim=-1).values[..., -1, None]
        filtered = filtered.masked_fill(filtered < kth, -1e9)

    if top_p is not None and 0.0 < top_p < 1.0:
        sorted_logits, sorted_indices = torch.sort(filtered, descending=True, dim=-1)
        sorted_probs = torch.softmax(sorted_logits, dim=-1)
        cumulative_probs = torch.cumsum(sorted_probs, dim=-1)
        sorted_remove = cumulative_probs > top_p
        sorted_remove[..., 1:] = sorted_remove[..., :-1].clone()
        sorted_remove[..., 0] = False
        remove = torch.zeros_like(sorted_remove, dtype=torch.bool)
        remove.scatter_(-1, sorted_indices, sorted_remove)
        filtered = filtered.masked_fill(remove, -1e9)

    return filtered


def _cosine_remask_rate(step, n_steps, power=1.0):
    if n_steps <= 0:
        raise ValueError("n_steps must be positive")
    if power <= 0:
        raise ValueError("remask_power must be positive")
    progress = min(max(float(step) / float(n_steps), 0.0), 1.0)
    return math.cos(progress * math.pi * 0.5) ** float(power)


def _cosine_remask_rates(step, n_steps, powers):
    """Vectorized cosine remask rate for per-sequence schedule powers."""
    if n_steps <= 0:
        raise ValueError("n_steps must be positive")
    if torch.any(powers <= 0):
        raise ValueError("remask_power must be positive")
    progress = min(max(float(step) / float(n_steps), 0.0), 1.0)
    base = math.cos(progress * math.pi * 0.5)
    return torch.pow(torch.full_like(powers, base), powers)


def _smooth_length_mix(lengths, low, high):
    """Map sequence lengths to a smooth short-to-long interpolation weight."""
    low = float(low)
    high = float(high)
    if high <= low:
        raise ValueError("adaptive_length_high must exceed adaptive_length_low")
    mix = ((lengths.float() - low) / (high - low)).clamp(0.0, 1.0)
    return mix * mix * (3.0 - 2.0 * mix)


def _length_conditioned_confidence_temperatures(
    sampling_temperatures,
    length_mix,
    short_temperature=1.0,
):
    """Interpolate confidence scaling without changing token sampling.

    Short sequences use an independently calibrated posterior, while long
    sequences recover the temperature-coupled confidence that is effective
    when more token decisions must remain mutually consistent.
    """
    if float(short_temperature) <= 0.0:
        raise ValueError("adaptive confidence temperature must be positive")
    return float(short_temperature) + (
        sampling_temperatures - float(short_temperature)
    ) * length_mix


def _refresh_progressive_scores(
    tokens,
    output_scores,
    log_probs,
    committed_positions,
    refresh_rows,
    gain_weight=0.0,
):
    """Rescore prior commitments under the current coupled distribution.

    Sampling and commitment stay separate: this function only updates the
    confidence used by the next remasking decision. A token that is no longer
    supported by its evolving context can therefore be masked on the next
    step, but it is never overwritten in place.
    """
    if float(gain_weight) < 0.0:
        raise ValueError("progressive_refresh_gain_weight must be non-negative")
    refresh_positions = committed_positions & refresh_rows.unsqueeze(1)
    if not refresh_positions.any():
        return output_scores, {
            "positions": 0,
            "contradictions": 0,
        }

    current_log_probs = log_probs.gather(
        2,
        tokens.unsqueeze(-1),
    ).squeeze(-1)
    best_log_probs = log_probs.max(dim=-1).values
    alternative_gain = (best_log_probs - current_log_probs).clamp(min=0.0)
    refreshed_scores = current_log_probs - float(gain_weight) * alternative_gain
    output_scores = torch.where(
        refresh_positions,
        refreshed_scores,
        output_scores,
    )
    contradictions = refresh_positions & alternative_gain.gt(0.0)
    return output_scores, {
        "positions": int(refresh_positions.sum().item()),
        "contradictions": int(contradictions.sum().item()),
    }


@torch.no_grad()
def _block_refine_tokens(
    model,
    x,
    output_scores,
    non_special,
    tk,
    steps,
    span_max=4,
    candidates=3,
    temperature=0.75,
    accept_margin=0.0,
    fsm_tracker=None,
    cond_batch=None,
    guidance_weight=2.0,
):
    """Refine low-confidence spans with model-likelihood and FSM gating."""
    if steps <= 0:
        return x, output_scores, {"steps": 0, "accepted_rows": 0, "proposals": 0}
    if span_max <= 0 or candidates <= 0 or temperature <= 0.0:
        raise ValueError("block-refinement span, candidate count, and temperature must be positive")

    batch_size, seq_len = x.shape
    accepted_rows = 0
    proposal_count = 0
    invalid_proposals = 0
    unk_id = getattr(tk, "unk_id", tk.vocab.get("<unk>", -1))

    for refine_step in range(int(steps)):
        progress = refine_step / max(int(steps) - 1, 1)
        span_size = max(1, int(round(float(span_max) * (1.0 - progress))))
        edit_mask = torch.zeros_like(non_special)

        for row in range(batch_size):
            positions = torch.nonzero(non_special[row], as_tuple=False).flatten()
            if positions.numel() == 0:
                continue
            width = min(span_size, int(positions.numel()))
            row_scores = output_scores[row, positions]
            window_scores = torch.stack(
                [row_scores[start:start + width].mean()
                 for start in range(positions.numel() - width + 1)]
            )
            start = int(torch.argmin(window_scores).item())
            edit_mask[row, positions[start:start + width]] = True

        if not edit_mask.any():
            break

        proposal_input = x.clone()
        proposal_input.masked_fill_(edit_mask, tk.mask_id)
        attention_mask = proposal_input.ne(tk.pad_id).long()
        if cond_batch is not None:
            logits_cond = model(
                proposal_input,
                attention_mask,
                cond=cond_batch,
                drop_cond=False,
            )
            logits_uncond = model(
                proposal_input,
                attention_mask,
                cond=cond_batch,
                drop_cond=True,
            )
            logits = guidance_weight * logits_cond + (1.0 - guidance_weight) * logits_uncond
        else:
            logits = model(proposal_input, attention_mask)

        for special_id in (tk.bos_id, tk.eos_id, tk.mask_id, tk.pad_id, unk_id):
            if special_id != -1:
                logits[:, :, special_id] = -1e9

        model_log_probs = torch.nn.functional.log_softmax(logits, dim=-1)
        sample_logits = logits / float(temperature)
        sampled = [sample_logits.argmax(dim=-1)]
        distribution = torch.distributions.Categorical(logits=sample_logits)
        for _ in range(int(candidates) - 1):
            sampled.append(distribution.sample())
        sampled = torch.stack(sampled, dim=0)

        current_token_scores = model_log_probs.gather(2, x.unsqueeze(-1)).squeeze(-1)
        current_scores = (current_token_scores * edit_mask.float()).sum(dim=1)
        candidate_token_scores = model_log_probs.unsqueeze(0).expand(
            int(candidates), -1, -1, -1
        ).gather(3, sampled.unsqueeze(-1)).squeeze(-1)
        candidate_scores = (
            candidate_token_scores * edit_mask.unsqueeze(0).float()
        ).sum(dim=2)

        candidate_sequences = x.unsqueeze(0).expand(int(candidates), -1, -1).clone()
        candidate_sequences = torch.where(
            edit_mask.unsqueeze(0),
            sampled,
            candidate_sequences,
        )
        proposal_count += int(candidates) * batch_size

        if fsm_tracker is not None:
            flat_candidates = candidate_sequences.reshape(-1, seq_len)
            valid = ~fsm_tracker.compute_penalties(flat_candidates).lt(0).any(dim=1)
            valid = valid.view(int(candidates), batch_size)
            invalid_proposals += int((~valid).sum().item())
            candidate_scores = candidate_scores.masked_fill(~valid, -math.inf)

        best_scores, best_indices = candidate_scores.max(dim=0)
        accept = best_scores > current_scores + float(accept_margin)
        if not accept.any():
            continue

        rows = torch.arange(batch_size, device=x.device)
        best_sequences = candidate_sequences[best_indices, rows]
        best_token_scores = candidate_token_scores[best_indices, rows]
        accepted_mask = edit_mask & accept.unsqueeze(1)
        x = torch.where(accepted_mask, best_sequences, x)
        output_scores = torch.where(accepted_mask, best_token_scores, output_scores)
        accepted_rows += int(accept.sum().item())

    return x, output_scores, {
        "steps": int(steps),
        "accepted_rows": accepted_rows,
        "proposals": proposal_count,
        "invalid_proposals": invalid_proposals,
    }


def _forward_with_corruption_level(
    model,
    tokens,
    attention_mask,
    corruption_level,
    cond_batch=None,
    guidance_weight=2.0,
):
    """Run a refinement-conditioned backbone without hiding filled tokens."""
    if not getattr(model, "corruption_level_conditioning", False):
        raise ValueError(
            "all-position refinement requires a checkpoint trained with "
            "corruption-level conditioning"
        )
    if cond_batch is None:
        return model(
            tokens,
            attention_mask,
            corruption_level=corruption_level,
        )
    logits_cond = model(
        tokens,
        attention_mask,
        cond=cond_batch,
        drop_cond=False,
        corruption_level=corruption_level,
    )
    logits_uncond = model(
        tokens,
        attention_mask,
        cond=cond_batch,
        drop_cond=True,
        corruption_level=corruption_level,
    )
    return (
        guidance_weight * logits_cond
        + (1.0 - guidance_weight) * logits_uncond
    )


@torch.no_grad()
def _all_position_refine_tokens(
    model,
    x,
    output_scores,
    non_special,
    tk,
    steps,
    corruption_start=0.25,
    corruption_end=0.05,
    corruption_power=1.5,
    max_edits=4,
    max_total_edits=0,
    min_logprob_gain=0.05,
    proposal_masked=False,
    verify_masked=False,
    verify_min_logprob_gain=0.25,
    prevent_revisit=False,
    patience=0,
    rdkit_each_step=False,
    fsm_tracker=None,
    rdkit_checker=None,
    cond_batch=None,
    guidance_weight=2.0,
    context_non_special=None,
):
    """Repair a filled draft using the model's all-position denoising scores.

    By default the input remains fully visible. ``proposal_masked`` instead
    re-masks the least-confident editable positions before proposing their
    replacements, which is useful for conditional spans. No molecular property
    or external score participates in proposal or acceptance.
    """
    if steps <= 0:
        return x, output_scores, {
            "steps": 0,
            "accepted_rows": 0,
            "accepted_edits": 0,
        }
    if not 0.0 < float(corruption_end) <= float(corruption_start) <= 1.0:
        raise ValueError(
            "all-position corruption rates must satisfy "
            "0 < end <= start <= 1"
        )
    if float(corruption_power) <= 0.0:
        raise ValueError("all-position corruption power must be positive")
    if int(max_edits) <= 0:
        raise ValueError("all-position max_edits must be positive")
    if int(max_total_edits) < 0:
        raise ValueError("all-position max_total_edits must be non-negative")
    if float(min_logprob_gain) < 0.0:
        raise ValueError("all-position min_logprob_gain must be non-negative")
    if float(verify_min_logprob_gain) < 0.0:
        raise ValueError(
            "all-position verify_min_logprob_gain must be non-negative"
        )
    if int(patience) < 0:
        raise ValueError("all-position patience must be non-negative")
    if not getattr(model, "corruption_level_conditioning", False):
        raise ValueError(
            "all-position refinement requires a refinement-trained checkpoint"
        )

    batch_size, seq_len = x.shape
    original = x.clone()
    original_scores = output_scores.clone()
    valid_lengths = non_special.sum(dim=1).clamp(min=1)
    if context_non_special is None:
        context_non_special = non_special
    if context_non_special.shape != non_special.shape:
        raise ValueError("context_non_special must match the token tensor shape")
    context_lengths = context_non_special.sum(dim=1).clamp(min=1)
    unk_id = getattr(tk, "unk_id", tk.vocab.get("<unk>", -1))
    special_ids = (tk.bos_id, tk.eos_id, tk.mask_id, tk.pad_id, unk_id)
    accepted_rows = 0
    accepted_edits = 0
    no_gain_rows = 0
    structurally_rejected_rows = 0
    single_edit_fallback_rows = 0
    verification_rejected_rows = 0
    rdkit_rejected_rows = 0
    converged_early = False
    edited_positions = torch.zeros_like(non_special)
    accepted_edit_counts = torch.zeros(
        batch_size,
        dtype=torch.long,
        device=x.device,
    )
    stale_steps = torch.zeros(batch_size, dtype=torch.long, device=x.device)
    active_rows = torch.ones(batch_size, dtype=torch.bool, device=x.device)

    ranks_template = torch.arange(seq_len, device=x.device).unsqueeze(0)
    ranks_template = ranks_template.expand(batch_size, -1)

    for refine_step in range(int(steps)):
        if int(max_total_edits) > 0:
            active_rows = active_rows & accepted_edit_counts.lt(
                int(max_total_edits)
            )
            if not active_rows.any():
                converged_early = True
                break
        progress = refine_step / max(int(steps) - 1, 1)
        corruption = float(corruption_end) + (
            float(corruption_start) - float(corruption_end)
        ) * (1.0 - progress) ** float(corruption_power)
        edit_counts = torch.ceil(valid_lengths.float() * corruption).long()
        edit_counts = edit_counts.clamp(min=1, max=int(max_edits))
        proposal_input = x
        proposal_positions = None
        if proposal_masked:
            proposal_candidates = non_special & active_rows.unsqueeze(1)
            if prevent_revisit:
                proposal_candidates = proposal_candidates & ~edited_positions
            ranked_scores = output_scores.masked_fill(
                ~proposal_candidates,
                math.inf,
            )
            sorted_probe_indices = torch.argsort(
                ranked_scores,
                dim=1,
                descending=False,
            )
            probe_ranks = torch.zeros_like(sorted_probe_indices).scatter_(
                1,
                sorted_probe_indices,
                ranks_template,
            )
            proposal_positions = (
                proposal_candidates
                & (probe_ranks < edit_counts.unsqueeze(1))
            )
            if not proposal_positions.any():
                converged_early = True
                break
            proposal_input = x.masked_fill(proposal_positions, tk.mask_id)
            corruption_level = (
                proposal_positions.sum(dim=1).float()
                / context_lengths.float()
            )
        else:
            corruption_level = torch.full(
                (batch_size,),
                corruption,
                device=x.device,
                dtype=torch.float32,
            )
        attention_mask = proposal_input.ne(tk.pad_id).long()
        logits = _forward_with_corruption_level(
            model=model,
            tokens=proposal_input,
            attention_mask=attention_mask,
            corruption_level=corruption_level,
            cond_batch=cond_batch,
            guidance_weight=guidance_weight,
        )
        for token_id in special_ids:
            if token_id != -1:
                logits[:, :, token_id] = -1e9

        log_probs = torch.nn.functional.log_softmax(logits, dim=-1)
        current_log_probs = log_probs.gather(2, x.unsqueeze(-1)).squeeze(-1)
        proposed_log_probs, proposed_tokens = log_probs.max(dim=-1)
        gains = proposed_log_probs - current_log_probs
        eligible = (
            non_special
            & active_rows.unsqueeze(1)
            & proposed_tokens.ne(x)
            & gains.gt(float(min_logprob_gain))
        )
        if proposal_positions is not None:
            eligible = eligible & proposal_positions
        if prevent_revisit:
            eligible = eligible & ~edited_positions
        if not eligible.any():
            no_gain_rows += int(active_rows.sum().item())
            converged_early = True
            break

        ranked_gains = gains.masked_fill(~eligible, -math.inf)
        sorted_indices = torch.argsort(ranked_gains, dim=1, descending=True)
        ranks = torch.zeros_like(sorted_indices).scatter_(
            1,
            sorted_indices,
            ranks_template,
        )
        edit_mask = eligible & (ranks < edit_counts.unsqueeze(1))
        no_gain_rows += int((~edit_mask.any(dim=1)).sum().item())
        if not edit_mask.any():
            converged_early = True
            break

        if verify_masked:
            verify_input = x.masked_fill(edit_mask, tk.mask_id)
            verify_level = (
                edit_mask.sum(dim=1).float()
                / valid_lengths.float().clamp(min=1.0)
            )
            verify_logits = _forward_with_corruption_level(
                model=model,
                tokens=verify_input,
                attention_mask=verify_input.ne(tk.pad_id).long(),
                corruption_level=verify_level,
                cond_batch=cond_batch,
                guidance_weight=guidance_weight,
            )
            for token_id in special_ids:
                if token_id != -1:
                    verify_logits[:, :, token_id] = -1e9
            verify_log_probs = torch.nn.functional.log_softmax(
                verify_logits,
                dim=-1,
            )
            verify_current = verify_log_probs.gather(
                2,
                x.unsqueeze(-1),
            ).squeeze(-1)
            verify_proposed = verify_log_probs.gather(
                2,
                proposed_tokens.unsqueeze(-1),
            ).squeeze(-1)
            verified = (
                verify_proposed - verify_current
            ).gt(float(verify_min_logprob_gain))
            rejected_by_verification = (
                edit_mask.any(dim=1)
                & ~(edit_mask & verified).any(dim=1)
            )
            verification_rejected_rows += int(
                rejected_by_verification.sum().item()
            )
            edit_mask = edit_mask & verified

        if not edit_mask.any():
            if int(patience) <= 0:
                converged_early = True
                break
            stale_steps[active_rows] += 1
            active_rows = active_rows & stale_steps.lt(int(patience))
            if not active_rows.any():
                converged_early = True
                break
            continue

        candidate = torch.where(edit_mask, proposed_tokens, x)
        accepted_mask = edit_mask.clone()
        row_valid = torch.ones(batch_size, dtype=torch.bool, device=x.device)

        if fsm_tracker is not None:
            row_valid = ~fsm_tracker.compute_penalties(candidate).lt(0).any(dim=1)
            rejected = ~row_valid & edit_mask.any(dim=1)
            structurally_rejected_rows += int(rejected.sum().item())

            # A simultaneous edit can violate syntax even when its strongest
            # component is useful. Retry only the highest-gain edit per row.
            if rejected.any():
                single_mask = eligible & ranks.eq(0) & rejected.unsqueeze(1)
                single_candidate = torch.where(single_mask, proposed_tokens, x)
                single_valid = ~fsm_tracker.compute_penalties(
                    single_candidate
                ).lt(0).any(dim=1)
                use_single = rejected & single_valid & single_mask.any(dim=1)
                candidate = torch.where(
                    use_single[:, None],
                    single_candidate,
                    candidate,
                )
                accepted_mask = torch.where(
                    use_single[:, None],
                    single_mask,
                    accepted_mask,
                )
                row_valid = row_valid | use_single
                single_edit_fallback_rows += int(use_single.sum().item())

        if rdkit_each_step and rdkit_checker is not None:
            chem = rdkit_checker[0]
            for row in torch.nonzero(
                row_valid & accepted_mask.any(dim=1),
                as_tuple=False,
            ).flatten().tolist():
                sequence = candidate[row].detach().cpu().tolist()
                if tk.eos_id in sequence:
                    sequence = sequence[:sequence.index(tk.eos_id) + 1]
                smiles = tk.decode(sequence).strip("'\"")
                if not rdkit_smiles_is_valid(smiles, chem):
                    row_valid[row] = False
                    rdkit_rejected_rows += 1

        accept_rows = row_valid & accepted_mask.any(dim=1)
        if not accept_rows.any():
            if int(patience) > 0:
                stale_steps[active_rows] += 1
                active_rows = active_rows & stale_steps.lt(int(patience))
                if not active_rows.any():
                    converged_early = True
                    break
            continue
        final_edit_mask = accepted_mask & accept_rows.unsqueeze(1)
        x = torch.where(final_edit_mask, candidate, x)
        output_scores = torch.where(
            final_edit_mask,
            proposed_log_probs,
            output_scores,
        )
        accepted_rows += int(accept_rows.sum().item())
        accepted_edits += int(final_edit_mask.sum().item())
        accepted_edit_counts += final_edit_mask.sum(dim=1)
        edited_positions = edited_positions | final_edit_mask
        if int(patience) > 0:
            stale_steps = torch.where(
                accept_rows,
                torch.zeros_like(stale_steps),
                stale_steps + active_rows.long(),
            )
            active_rows = active_rows & stale_steps.lt(int(patience))
            if not active_rows.any():
                converged_early = True
                break
        if int(max_total_edits) > 0:
            active_rows = active_rows & accepted_edit_counts.lt(
                int(max_total_edits)
            )
            if not active_rows.any():
                converged_early = True
                break

    rdkit_rollbacks = 0
    if rdkit_checker is not None:
        chem = rdkit_checker[0]
        for row in range(batch_size):
            sequence = x[row].detach().cpu().tolist()
            if tk.eos_id in sequence:
                sequence = sequence[:sequence.index(tk.eos_id) + 1]
            smiles = tk.decode(sequence).strip("'\"")
            if not rdkit_smiles_is_valid(smiles, chem):
                x[row] = original[row]
                output_scores[row] = original_scores[row]
                rdkit_rollbacks += 1

    return x, output_scores, {
        "steps": int(steps),
        "accepted_rows": accepted_rows,
        "accepted_edits": accepted_edits,
        "proposal_masked": bool(proposal_masked),
        "max_total_edits": int(max_total_edits),
        "rows_at_edit_cap": int(
            (
                accepted_edit_counts.ge(int(max_total_edits)).sum().item()
                if int(max_total_edits) > 0
                else 0
            )
        ),
        "no_gain_rows": no_gain_rows,
        "structurally_rejected_rows": structurally_rejected_rows,
        "single_edit_fallback_rows": single_edit_fallback_rows,
        "verification_rejected_rows": verification_rejected_rows,
        "rdkit_rejected_rows": rdkit_rejected_rows,
        "rdkit_rollbacks": rdkit_rollbacks,
        "converged_early": converged_early,
    }


@torch.no_grad()
def sample_csdnet(
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
    length_explore_fraction=0.0,
    length_batching="random",
    remask_power=1.0,
    length_adaptive=False,
    adaptive_length_low=28.0,
    adaptive_length_high=40.0,
    adaptive_temperature_start_short=1.8,
    adaptive_temperature_end_short=0.35,
    adaptive_temperature_power_short=1.25,
    adaptive_gumbel_scale_short=1.35,
    adaptive_remask_power_short=0.8,
    confidence_temperature=0.0,
    confidence_length_adaptive=False,
    adaptive_confidence_length_low=28.0,
    adaptive_confidence_length_high=34.0,
    adaptive_confidence_temperature_short=1.0,
    progressive_commit=False,
    progressive_refresh_confidence=False,
    progressive_refresh_start=0.50,
    progressive_refresh_gain_weight=0.0,
    block_refine_steps=0,
    block_refine_span_max=4,
    block_refine_candidates=3,
    block_refine_temperature=0.75,
    block_refine_accept_margin=0.0,
    all_position_refine_steps=0,
    all_position_corruption_start=0.25,
    all_position_corruption_end=0.05,
    all_position_corruption_power=1.5,
    all_position_max_edits=4,
    all_position_max_total_edits=0,
    all_position_min_logprob_gain=0.05,
    all_position_verify_masked=False,
    all_position_verify_min_logprob_gain=0.25,
    all_position_prevent_revisit=False,
    all_position_patience=0,
    all_position_rdkit_each_step=False,
    unmask_selection="top_prob",
    strict_final_sanitize=False,
    max_refill_factor=1.25,
    length_scheduler=None,
    length_scheduler_temperature=1.0,
    length_scheduler_top_k=16,
    return_diagnostics=False,
):
    """Generate SMILES with masked discrete diffusion and optional FSM repair."""

    if getattr(model, "is_elastic", False):
        if length_scheduler is not None:
            raise ValueError(
                "External learned-length scheduling is implemented for the "
                "ordinary fixed-backbone sampler, not ElasticCSDNet."
            )
        if length_adaptive:
            raise ValueError("length-adaptive sampling is not supported by elastic sampling")
        if length_batching != "random":
            raise ValueError("sorted length batching is not supported by elastic sampling")
        if int(block_refine_steps) > 0:
            raise ValueError("block refinement is not yet supported by elastic sampling")
        if int(all_position_refine_steps) > 0:
            raise ValueError("all-position refinement is not supported by elastic sampling")
        if progressive_refresh_confidence:
            raise ValueError(
                "progressive confidence refresh is not supported by elastic sampling"
            )
        if confidence_length_adaptive:
            raise ValueError(
                "length-adaptive confidence is not supported by elastic sampling"
            )
        from CSDNet.util.elastic_sampling import sample_elastic_csdnet

        return sample_elastic_csdnet(
            model=model,
            tk=tk,
            ref_lengths=ref_lengths,
            n_mol=n_mol,
            cond=cond,
            w=w,
            device=device,
            batch_size=batch_size,
            n_steps=n_steps,
            use_fsm_check=use_fsm_check,
            use_rdkit_kekulize_check=use_rdkit_kekulize_check,
            rdkit_check_interval=rdkit_check_interval,
            max_sample_retries=max_sample_retries,
            violation_neighborhood=violation_neighborhood,
            temperature_start=temperature_start,
            temperature_end=temperature_end,
            temperature_power=temperature_power,
            top_k=top_k,
            top_p=top_p,
            gumbel_scale=gumbel_scale,
            length_quantile_low=length_quantile_low,
            length_quantile_high=length_quantile_high,
            length_min=length_min,
            length_max=length_max,
            unmask_selection=unmask_selection,
            strict_final_sanitize=strict_final_sanitize,
            max_refill_factor=max_refill_factor,
            return_diagnostics=return_diagnostics,
        )

    model.eval()
    if n_steps <= 0:
        raise ValueError("n_steps must be positive")
    if not ref_lengths:
        raise ValueError("ref_lengths must contain at least one sequence length")
    if length_scheduler is not None:
        if (
            float(length_quantile_low) != 0.0
            or float(length_quantile_high) != 1.0
            or int(length_min) != 0
            or int(length_max) != 0
            or float(length_explore_fraction) != 0.0
        ):
            raise ValueError(
                "Learned de novo length is mutually exclusive with empirical "
                "quantiles, hard length ranges, and random exploration."
            )
        if float(length_scheduler_temperature) <= 0.0:
            raise ValueError("length_scheduler_temperature must be positive")
    if not 0.0 <= float(length_explore_fraction) <= 1.0:
        raise ValueError("length_explore_fraction must be in [0, 1]")
    if length_batching not in {"random", "sorted"}:
        raise ValueError("length_batching must be 'random' or 'sorted'")
    if float(remask_power) <= 0.0:
        raise ValueError("remask_power must be positive")
    if length_adaptive:
        if float(adaptive_length_high) <= float(adaptive_length_low):
            raise ValueError("adaptive_length_high must exceed adaptive_length_low")
        if min(
            float(adaptive_temperature_start_short),
            float(adaptive_temperature_end_short),
            float(adaptive_temperature_power_short),
            float(adaptive_remask_power_short),
        ) <= 0.0:
            raise ValueError("adaptive temperatures and powers must be positive")
        if float(adaptive_gumbel_scale_short) < 0.0:
            raise ValueError("adaptive_gumbel_scale_short must be non-negative")
    if float(max_refill_factor) < 1.0:
        raise ValueError("max_refill_factor must be at least 1.0")
    if float(confidence_temperature) < 0.0:
        raise ValueError("confidence_temperature must be non-negative")
    if confidence_length_adaptive:
        if float(adaptive_confidence_length_high) <= float(
            adaptive_confidence_length_low
        ):
            raise ValueError(
                "adaptive confidence length high must exceed length low"
            )
        if float(adaptive_confidence_temperature_short) <= 0.0:
            raise ValueError(
                "adaptive confidence temperature must be positive"
            )
    if not 0.0 < float(progressive_refresh_start) <= 1.0:
        raise ValueError("progressive_refresh_start must be in (0, 1]")
    if float(progressive_refresh_gain_weight) < 0.0:
        raise ValueError(
            "progressive_refresh_gain_weight must be non-negative"
        )
    if progressive_refresh_confidence and not progressive_commit:
        raise ValueError(
            "progressive confidence refresh requires progressive_commit=True"
        )
    if progressive_refresh_confidence and not getattr(
        model,
        "corruption_level_conditioning",
        False,
    ):
        raise ValueError(
            "progressive confidence refresh requires a refinement-trained "
            "checkpoint with corruption-level conditioning"
        )
    if int(block_refine_steps) < 0:
        raise ValueError("block_refine_steps must be non-negative")
    if int(all_position_refine_steps) < 0:
        raise ValueError("all_position_refine_steps must be non-negative")

    fsm_start_step = int(n_steps * 0.8)
    retry_step = int(n_steps * 0.6)

    fsm_tracker = None
    if use_fsm_check:
        fsm_tracker = ValenceFSMTracker(tk)
    rdkit_checker = None
    if use_rdkit_kekulize_check or strict_final_sanitize:
        rdkit_checker = prepare_rdkit_kekulize_checker(tk, fsm_tracker)
    if strict_final_sanitize and rdkit_checker is None:
        raise RuntimeError("strict_final_sanitize requires RDKit")

    generated = []
    unk_id = getattr(tk, "unk_id", tk.vocab.get("<unk>", -1))
    length_pool = None
    full_length_pool = None
    if length_scheduler is None:
        length_pool = _prepare_length_pool(
            ref_lengths,
            length_quantile_low=length_quantile_low,
            length_quantile_high=length_quantile_high,
            length_min=length_min,
            length_max=length_max,
        )
        full_length_pool = _prepare_length_pool(ref_lengths)
    max_proposals = max(
        n_mol,
        int(math.ceil(n_mol * float(max_refill_factor))),
    )
    proposal_count = 0
    empty_rejections = 0
    sanitization_rejections = 0
    sampled_length_counts = Counter()
    pending_lengths = []
    block_refine_diagnostics = Counter()
    all_position_refine_diagnostics = Counter()
    progressive_refresh_diagnostics = Counter()

    def draw_lengths(count):
        if length_scheduler is not None:
            values = sample_de_novo_lengths(
                length_scheduler,
                tk,
                count,
                max_len=min(
                    int(length_scheduler.max_position_embeddings),
                    int(length_scheduler.max_replacement_length) + 2,
                ),
                device=device,
                temperature=length_scheduler_temperature,
                top_k=length_scheduler_top_k,
                batch_size=max(1, min(batch_size, 512)),
            )
            if length_batching == "sorted":
                values.sort()
            return values
        values = []
        for _ in range(count):
            pool = length_pool
            if (
                length_explore_fraction > 0.0
                and np.random.random() < length_explore_fraction
            ):
                pool = full_length_pool
            values.append(max(3, int(np.random.choice(pool))))
        if length_batching == "sorted":
            values.sort()
        return values

    while len(generated) < n_mol:
        remaining_proposals = max_proposals - proposal_count
        if remaining_proposals <= 0:
            raise RuntimeError(
                "Final sanitization refill budget exhausted: "
                f"accepted={len(generated)}/{n_mol}, proposals={proposal_count}"
            )
        if not pending_lengths:
            draw_count = min(n_mol - len(generated), remaining_proposals)
            pending_lengths = draw_lengths(draw_count)
        bsz = min(batch_size, len(pending_lengths), remaining_proposals)
        lengths = pending_lengths[:bsz]
        del pending_lengths[:bsz]
        sampled_length_counts.update(lengths)
        maxL = max(lengths)
        proposal_count += bsz

        x = torch.full((bsz, maxL), tk.mask_id, device=device, dtype=torch.long)
        x[:, 0] = tk.bos_id
        for b, L in enumerate(lengths):
            x[b, L - 1] = tk.eos_id
            if L < maxL:
                x[b, L:] = tk.pad_id

        output_scores = torch.zeros_like(x, dtype=torch.float)
        non_special = (x != tk.pad_id) & (x != tk.bos_id) & (x != tk.eos_id)
        valid_lens = non_special.sum(dim=1, keepdim=True).float()
        ranks_template = torch.arange(maxL, device=device).unsqueeze(0).expand(bsz, -1)

        length_tensor = torch.as_tensor(lengths, device=device, dtype=torch.float32)
        if length_adaptive:
            length_mix = _smooth_length_mix(
                length_tensor,
                adaptive_length_low,
                adaptive_length_high,
            )
        else:
            length_mix = torch.ones_like(length_tensor)
        if confidence_length_adaptive:
            confidence_length_mix = _smooth_length_mix(
                length_tensor,
                adaptive_confidence_length_low,
                adaptive_confidence_length_high,
            )
        else:
            confidence_length_mix = None

        def interpolate(short_value, long_value):
            return float(short_value) + (
                float(long_value) - float(short_value)
            ) * length_mix

        row_temperature_start = interpolate(
            adaptive_temperature_start_short,
            temperature_start,
        )
        row_temperature_end = interpolate(
            adaptive_temperature_end_short,
            temperature_end,
        )
        row_temperature_power = interpolate(
            adaptive_temperature_power_short,
            temperature_power,
        )
        row_gumbel_scale = interpolate(
            adaptive_gumbel_scale_short,
            gumbel_scale,
        ).unsqueeze(1)
        row_remask_power = interpolate(
            adaptive_remask_power_short,
            remask_power,
        ).unsqueeze(1)

        cond_batch = None
        if cond is not None:
            cond_batch = cond.repeat(bsz, 1).to(device)

        step = 0
        active_rows = torch.ones(bsz, device=device, dtype=torch.bool)
        retries = torch.zeros(bsz, device=device, dtype=torch.long)
        while step < n_steps:
            if not active_rows.any():
                break

            active_positions = non_special & active_rows.unsqueeze(1)
            sample_positions = active_positions
            if progressive_commit:
                sample_positions = sample_positions & (x == tk.mask_id)
            committed_positions = active_positions & (x != tk.mask_id)
            remaining_mask_ratio = (
                ((x == tk.mask_id) & active_positions).sum(dim=1).float()
                / active_positions.sum(dim=1).float().clamp(min=1.0)
            )
            amask = (x != tk.pad_id).long()

            if cond_batch is not None:
                logits_cond = model(x, amask, cond=cond_batch, drop_cond=False)
                logits_uncond = model(x, amask, cond=cond_batch, drop_cond=True)
                logits = w * logits_cond + (1 - w) * logits_uncond
            else:
                logits = model(x, amask)

            logits[:, :, tk.bos_id] = -1e9
            logits[:, :, tk.eos_id] = -1e9
            logits[:, :, tk.mask_id] = -1e9
            logits[:, :, tk.pad_id] = -1e9
            if unk_id != -1:
                logits[:, :, unk_id] = -1e9

            progress = step / max(n_steps - 1, 1)
            temperature = row_temperature_end + (
                row_temperature_start - row_temperature_end
            ) * (1.0 - progress) ** row_temperature_power
            sample_logits = _filter_sampling_logits(
                logits / temperature[:, None, None],
                top_k=top_k,
                top_p=top_p,
            )
            cur_tokens = torch.distributions.Categorical(logits=sample_logits).sample()

            confidence_logits = sample_logits
            if confidence_length_mix is not None:
                confidence_temperatures = (
                    _length_conditioned_confidence_temperatures(
                        sampling_temperatures=temperature,
                        length_mix=confidence_length_mix,
                        short_temperature=adaptive_confidence_temperature_short,
                    )
                )
                confidence_logits = logits / confidence_temperatures[:, None, None]
            elif float(confidence_temperature) > 0.0:
                confidence_logits = logits / float(confidence_temperature)
            lm_log_probs = torch.nn.functional.log_softmax(
                confidence_logits,
                dim=-1,
            )
            cur_scores = torch.gather(lm_log_probs, 2, cur_tokens.unsqueeze(-1)).squeeze(-1)

            x.masked_scatter_(sample_positions, cur_tokens[sample_positions])
            output_scores.masked_scatter_(
                sample_positions,
                cur_scores[sample_positions],
            )

            if progressive_refresh_confidence:
                refresh_rows = (
                    active_rows
                    & remaining_mask_ratio.le(float(progressive_refresh_start))
                )
                output_scores, refresh_stats = _refresh_progressive_scores(
                    tokens=x,
                    output_scores=output_scores,
                    log_probs=lm_log_probs,
                    committed_positions=committed_positions,
                    refresh_rows=refresh_rows,
                    gain_weight=progressive_refresh_gain_weight,
                )
                if refresh_stats["positions"] > 0:
                    progressive_refresh_diagnostics["steps"] += 1
                    progressive_refresh_diagnostics["positions"] += refresh_stats[
                        "positions"
                    ]
                    progressive_refresh_diagnostics[
                        "contradictions"
                    ] += refresh_stats["contradictions"]

            should_check_fsm = (
                use_fsm_check
                and step >= fsm_start_step
                and (step % 5 == 0 or step == n_steps - 1)
            )
            should_check_rdkit = (
                rdkit_checker is not None
                and step >= fsm_start_step
                and (step % max(1, rdkit_check_interval) == 0 or step == n_steps - 1)
            )
            if should_check_fsm or should_check_rdkit:
                penalties = torch.zeros_like(output_scores)
                if should_check_fsm:
                    penalties += fsm_tracker.compute_penalties(x)
                if should_check_rdkit:
                    chem, rdkit_focus_ids = rdkit_checker
                    penalties += compute_rdkit_kekulize_penalties(
                        x,
                        tk,
                        chem,
                        rdkit_focus_ids,
                    )
                penalties = penalties.masked_fill(~active_positions, 0.0)
                output_scores += penalties

                violation_positions = (penalties < 0) & active_positions
                if violation_positions.any() and step != n_steps - 1:
                    repair_mask = expand_violation_mask(
                        violation_positions,
                        active_positions,
                        radius=violation_neighborhood,
                    )
                    x.masked_fill_(repair_mask, tk.mask_id)
                    output_scores.masked_fill_(repair_mask, -math.inf)

                if step == n_steps - 1:
                    bad_rows = violation_positions.any(dim=1) & active_rows
                    retry_rows = bad_rows & (retries < max_sample_retries)
                    if retry_rows.any():
                        retries[retry_rows] += 1
                        active_rows = retry_rows

                        retry_positions = non_special & retry_rows.unsqueeze(1)
                        retry_violation_positions = violation_positions & retry_rows.unsqueeze(1)
                        retry_repair_mask = expand_violation_mask(
                            retry_violation_positions,
                            retry_positions,
                            radius=violation_neighborhood,
                        )
                        x.masked_fill_(retry_repair_mask, tk.mask_id)
                        output_scores.masked_fill_(retry_repair_mask, -math.inf)

                        retry_rate = _cosine_remask_rates(
                            retry_step + 1,
                            n_steps,
                            row_remask_power,
                        )
                        retry_cutoff_len = (valid_lens * retry_rate).long()
                        retry_scores = output_scores.masked_fill(~retry_positions, 1000.0)
                        retry_sorted_idx = torch.argsort(retry_scores, dim=1)
                        retry_ranks = torch.zeros_like(retry_sorted_idx).scatter_(
                            1,
                            retry_sorted_idx,
                            ranks_template,
                        )
                        retry_mask = (retry_ranks < retry_cutoff_len) & retry_positions
                        x.masked_fill_(retry_mask, tk.mask_id)
                        output_scores.masked_fill_(retry_mask, -math.inf)

                        step = retry_step
                        continue

                    active_rows = torch.zeros_like(active_rows)
                    break

            t = step + 1
            remask_rate = _cosine_remask_rates(t, n_steps, row_remask_power)
            cutoff_len = (valid_lens * remask_rate).long()

            scores = output_scores.masked_fill(~active_positions, 1000.0)
            gumbel = -torch.log(-torch.log(torch.rand_like(scores) + 1e-10) + 1e-10)
            scores = scores + gumbel * row_gumbel_scale * remask_rate

            sorted_idx = torch.argsort(scores, dim=1)
            ranks = torch.zeros_like(sorted_idx).scatter_(1, sorted_idx, ranks_template)

            bottom_mask = (ranks < cutoff_len) & active_positions
            x.masked_fill_(bottom_mask, tk.mask_id)
            output_scores.masked_fill_(bottom_mask, -math.inf)

            step += 1

        if int(block_refine_steps) > 0:
            x, output_scores, refine_diagnostics = _block_refine_tokens(
                model=model,
                x=x,
                output_scores=output_scores,
                non_special=non_special,
                tk=tk,
                steps=block_refine_steps,
                span_max=block_refine_span_max,
                candidates=block_refine_candidates,
                temperature=block_refine_temperature,
                accept_margin=block_refine_accept_margin,
                fsm_tracker=fsm_tracker if use_fsm_check else None,
                cond_batch=cond_batch,
                guidance_weight=w,
            )
            block_refine_diagnostics.update(refine_diagnostics)

        if int(all_position_refine_steps) > 0:
            x, output_scores, refine_diagnostics = _all_position_refine_tokens(
                model=model,
                x=x,
                output_scores=output_scores,
                non_special=non_special,
                tk=tk,
                steps=all_position_refine_steps,
                corruption_start=all_position_corruption_start,
                corruption_end=all_position_corruption_end,
                corruption_power=all_position_corruption_power,
                max_edits=all_position_max_edits,
                max_total_edits=all_position_max_total_edits,
                min_logprob_gain=all_position_min_logprob_gain,
                verify_masked=all_position_verify_masked,
                verify_min_logprob_gain=all_position_verify_min_logprob_gain,
                prevent_revisit=all_position_prevent_revisit,
                patience=all_position_patience,
                rdkit_each_step=all_position_rdkit_each_step,
                fsm_tracker=fsm_tracker if use_fsm_check else None,
                rdkit_checker=rdkit_checker,
                cond_batch=cond_batch,
                guidance_weight=w,
            )
            all_position_refine_diagnostics.update(refine_diagnostics)

        for i in range(bsz):
            seq = x[i].cpu().tolist()
            if tk.eos_id in seq:
                seq = seq[: seq.index(tk.eos_id) + 1]
            smi = tk.decode(seq).strip("'\"")
            if not smi:
                empty_rejections += 1
                continue
            if strict_final_sanitize and not rdkit_smiles_is_valid(
                smi,
                rdkit_checker[0],
            ):
                sanitization_rejections += 1
                continue
            generated.append(smi)

    if return_diagnostics:
        return generated, {
            "proposals": proposal_count,
            "accepted": len(generated),
            "empty_rejections": empty_rejections,
            "sanitization_rejections": sanitization_rejections,
            "strict_final_sanitize": bool(strict_final_sanitize),
            "length_batching": length_batching,
            "length_source": (
                "learned_scheduler"
                if length_scheduler is not None
                else "empirical_reference"
            ),
            "length_scheduler_temperature": (
                float(length_scheduler_temperature)
                if length_scheduler is not None
                else None
            ),
            "length_scheduler_top_k": (
                int(length_scheduler_top_k)
                if length_scheduler is not None
                else None
            ),
            "length_adaptive": bool(length_adaptive),
            "adaptive_length_low": float(adaptive_length_low),
            "adaptive_length_high": float(adaptive_length_high),
            "adaptive_short_parameters": {
                "temperature_start": float(adaptive_temperature_start_short),
                "temperature_end": float(adaptive_temperature_end_short),
                "temperature_power": float(adaptive_temperature_power_short),
                "gumbel_scale": float(adaptive_gumbel_scale_short),
                "remask_power": float(adaptive_remask_power_short),
            },
            "adaptive_long_parameters": {
                "temperature_start": float(temperature_start),
                "temperature_end": float(temperature_end),
                "temperature_power": float(temperature_power),
                "gumbel_scale": float(gumbel_scale),
                "remask_power": float(remask_power),
            },
            "confidence_temperature": float(confidence_temperature),
            "confidence_length_adaptive": bool(confidence_length_adaptive),
            "adaptive_confidence_length_low": float(
                adaptive_confidence_length_low
            ),
            "adaptive_confidence_length_high": float(
                adaptive_confidence_length_high
            ),
            "adaptive_confidence_temperature_short": float(
                adaptive_confidence_temperature_short
            ),
            "progressive_commit": bool(progressive_commit),
            "progressive_refresh": {
                "enabled": bool(progressive_refresh_confidence),
                "start_mask_ratio": float(progressive_refresh_start),
                "gain_weight": float(progressive_refresh_gain_weight),
                **dict(progressive_refresh_diagnostics),
            },
            "block_refinement": {
                "steps_per_batch": int(block_refine_steps),
                "span_max": int(block_refine_span_max),
                "candidates": int(block_refine_candidates),
                "temperature": float(block_refine_temperature),
                "accept_margin": float(block_refine_accept_margin),
                **dict(block_refine_diagnostics),
            },
            "all_position_refinement": {
                "steps_per_batch": int(all_position_refine_steps),
                "corruption_start": float(all_position_corruption_start),
                "corruption_end": float(all_position_corruption_end),
                "corruption_power": float(all_position_corruption_power),
                "max_edits": int(all_position_max_edits),
                "max_total_edits": int(all_position_max_total_edits),
                "min_logprob_gain": float(all_position_min_logprob_gain),
                "verify_masked": bool(all_position_verify_masked),
                "verify_min_logprob_gain": float(
                    all_position_verify_min_logprob_gain
                ),
                "prevent_revisit": bool(all_position_prevent_revisit),
                "patience": int(all_position_patience),
                "rdkit_each_step": bool(all_position_rdkit_each_step),
                **dict(all_position_refine_diagnostics),
            },
            "sampled_length_histogram": dict(sorted(sampled_length_counts.items())),
        }
    return generated
