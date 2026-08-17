import math
from dataclasses import dataclass

import torch


@dataclass
class _DynamicToken:
    token_id: int
    frozen: bool = False
    left_region: int | None = None
    right_region: int | None = None


class UnifiedDynamicSampler:
    """Model-aligned sampler for fixed- and variable-length molecular states.

    The same loop is used for de novo generation and fragment infilling.  An
    editable gap is represented by a region shared by the right side of its
    left boundary and the left side of its right boundary.  Newly inserted
    masks inherit that region, so repeated insertion remains local without a
    task-name-specific policy.
    """

    def __init__(
        self,
        model,
        pad_id,
        mask_id,
        bos_id,
        eos_id,
        valid_token_ids,
        max_len=256,
        max_gap_count=8,
        token_logit_filter=None,
    ):
        self.model = getattr(model, "backbone", model)
        self.pad_id = int(pad_id)
        self.mask_id = int(mask_id)
        self.bos_id = int(bos_id)
        self.eos_id = int(eos_id)
        self.valid_token_ids = tuple(int(x) for x in valid_token_ids)
        self.max_len = int(max_len)
        self.max_gap_count = int(max_gap_count)
        self.token_logit_filter = token_logit_filter
        if not self.valid_token_ids:
            raise ValueError("valid_token_ids cannot be empty.")

    @property
    def device(self):
        return next(self.model.parameters()).device

    @torch.no_grad()
    def sample_de_novo(
        self,
        batch_size,
        num_steps=500,
        temperature_start=1.35,
        temperature_end=0.20,
        insertion_temperature=0.85,
        insertion_stop=0.72,
        refinement_start=0.58,
        deletion_threshold=0.94,
        replacement_logprob_gain=0.35,
        stochastic=True,
    ):
        states = []
        for _ in range(int(batch_size)):
            states.append(
                [
                    _DynamicToken(
                        self.bos_id, frozen=True, right_region=0
                    ),
                    _DynamicToken(
                        self.eos_id, frozen=True, left_region=0
                    ),
                ]
            )
        return self._sample(
            states,
            region_limits=[{0: (0, self.max_len - 2)} for _ in states],
            num_steps=num_steps,
            temperature_start=temperature_start,
            temperature_end=temperature_end,
            insertion_temperature=insertion_temperature,
            insertion_stop=insertion_stop,
            refinement_start=refinement_start,
            deletion_threshold=deletion_threshold,
            replacement_logprob_gain=replacement_logprob_gain,
            stochastic=stochastic,
        )

    @torch.no_grad()
    def sample_fragment(
        self,
        input_ids,
        frozen_mask=None,
        editable_gaps=None,
        region_limits=None,
        return_frozen_mask=False,
        num_steps=500,
        **sampling_kwargs,
    ):
        """Fill one or more molecular regions.

        ``input_ids`` is a list of active token-id lists including BOS/EOS.
        ``frozen_mask`` marks supplied context. Existing mask tokens should be
        unfrozen. ``editable_gaps`` gives the gap indices after which unknown-
        length content may be inserted. A known-length task can simply provide
        mask tokens and an empty gap list.
        """
        if torch.is_tensor(input_ids):
            input_ids = input_ids.tolist()
        if frozen_mask is None:
            frozen_mask = [
                [token_id != self.mask_id for token_id in row]
                for row in input_ids
            ]
        elif torch.is_tensor(frozen_mask):
            frozen_mask = frozen_mask.bool().tolist()
        if editable_gaps is None:
            editable_gaps = [[] for _ in input_ids]
        if region_limits is None:
            region_limits = [None for _ in input_ids]
        states = []
        normalized_limits = []
        for row_ids, row_frozen, row_gaps in zip(
            input_ids, frozen_mask, editable_gaps
        ):
            active_ids = [int(x) for x in row_ids if int(x) != self.pad_id]
            row_frozen = list(row_frozen)[: len(active_ids)]
            if len(active_ids) < 2:
                raise ValueError("Each fragment state must contain BOS and EOS.")
            row = [
                _DynamicToken(token_id, frozen=bool(is_frozen))
                for token_id, is_frozen in zip(active_ids, row_frozen)
            ]
            for region, gap_index in enumerate(row_gaps):
                gap_index = int(gap_index)
                if gap_index < 0 or gap_index >= len(row) - 1:
                    raise ValueError(
                        f"Editable gap {gap_index} is outside a row of length {len(row)}."
                    )
                row[gap_index].right_region = region
                row[gap_index + 1].left_region = region
            states.append(row)
        for row, row_gaps, limits in zip(states, editable_gaps, region_limits):
            if limits is None:
                normalized_limits.append(
                    {
                        region: (0, self.max_len - len(row))
                        for region in range(len(row_gaps))
                    }
                )
                continue
            if isinstance(limits, dict):
                normalized_limits.append(
                    {
                        int(region): (int(bounds[0]), int(bounds[1]))
                        for region, bounds in limits.items()
                    }
                )
            else:
                normalized_limits.append(
                    {
                        region: (int(bounds[0]), int(bounds[1]))
                        for region, bounds in enumerate(limits)
                    }
                )
        return self._sample(
            states,
            region_limits=normalized_limits,
            num_steps=num_steps,
            return_frozen_mask=return_frozen_mask,
            **sampling_kwargs,
        )

    def _tensorize(self, states):
        width = max(len(row) for row in states)
        input_ids = torch.full(
            (len(states), width),
            self.pad_id,
            dtype=torch.long,
            device=self.device,
        )
        attention_mask = torch.zeros_like(input_ids)
        for row_index, row in enumerate(states):
            ids = torch.tensor(
                [token.token_id for token in row],
                dtype=torch.long,
                device=self.device,
            )
            input_ids[row_index, : ids.numel()] = ids
            attention_mask[row_index, : ids.numel()] = 1
        return input_ids, attention_mask

    @staticmethod
    def _gap_region(left, right):
        if left.right_region is None:
            return None
        if left.right_region != right.left_region:
            return None
        return left.right_region

    def _valid_logits(self, logits, row_ids, position):
        selected = logits.new_full(logits.shape, -torch.inf)
        valid_ids = torch.tensor(
            self.valid_token_ids, device=logits.device, dtype=torch.long
        )
        selected[valid_ids] = logits[valid_ids]
        if self.token_logit_filter is not None:
            selected = self.token_logit_filter(
                row_ids=row_ids,
                position=int(position),
                logits=selected,
            )
        return selected

    @staticmethod
    def _temperature(progress, start, end):
        return float(end) + (float(start) - float(end)) * (1.0 - progress) ** 1.5

    def _draw_token(self, logits, temperature, stochastic):
        if not stochastic or temperature <= 1e-5:
            return int(logits.argmax().item())
        probabilities = torch.softmax(logits.float() / temperature, dim=-1)
        if not torch.isfinite(probabilities).all() or probabilities.sum() <= 0:
            return int(logits.argmax().item())
        return int(torch.multinomial(probabilities, 1).item())

    def _draw_gap_count(self, logits, temperature, stochastic):
        if not stochastic or temperature <= 1e-5:
            return int(logits.argmax().item())
        probabilities = torch.softmax(logits.float() / temperature, dim=-1)
        if not torch.isfinite(probabilities).all() or probabilities.sum() <= 0:
            return int(logits.argmax().item())
        return int(torch.multinomial(probabilities, 1).item())

    @torch.no_grad()
    def _sample(
        self,
        states,
        region_limits=None,
        num_steps=500,
        temperature_start=1.35,
        temperature_end=0.20,
        insertion_temperature=0.85,
        insertion_stop=0.72,
        refinement_start=0.58,
        deletion_threshold=0.94,
        replacement_logprob_gain=0.35,
        stochastic=True,
        return_frozen_mask=False,
    ):
        num_steps = int(num_steps)
        if num_steps < 2:
            raise ValueError("num_steps must be at least two.")
        self.model.eval()
        if region_limits is None:
            region_limits = [{} for _ in states]

        for step in range(num_steps):
            progress = (step + 1) / float(num_steps)
            input_ids, attention_mask = self._tensorize(states)
            output = self.model(
                input_ids=input_ids,
                attention_mask=attention_mask,
                corruption_level=torch.full(
                    (len(states),),
                    1.0 - progress,
                    device=self.device,
                    dtype=torch.float32,
                ),
                return_aux=True,
            )
            token_temperature = self._temperature(
                progress, temperature_start, temperature_end
            )

            # Operate on object references from the pre-insertion state.  Gap
            # insertions are applied in reverse order so tensor positions and
            # references remain aligned for the rest of this iteration.
            for row_index, row in enumerate(states):
                original = list(row)
                row_ids = [token.token_id for token in original]

                if progress <= insertion_stop and len(row) < self.max_len:
                    insertions = []
                    for gap_index in range(len(original) - 1):
                        region = self._gap_region(
                            original[gap_index], original[gap_index + 1]
                        )
                        if region is None:
                            continue
                        count = self._draw_gap_count(
                            output["gap_logits"][row_index, gap_index],
                            insertion_temperature,
                            stochastic,
                        )
                        if count <= 0:
                            count = 0
                        current_region_tokens = sum(
                            token.left_region == region
                            and token.right_region == region
                            for token in row
                        )
                        minimum, maximum = region_limits[row_index].get(
                            region, (0, self.max_len)
                        )
                        if step == 0 and current_region_tokens < minimum:
                            count = max(count, minimum - current_region_tokens)
                        count = min(
                            count,
                            max(0, maximum - current_region_tokens),
                        )
                        if count <= 0:
                            continue
                        room = self.max_len - len(row) - sum(
                            item[1] for item in insertions
                        )
                        count = min(count, self.max_gap_count, room)
                        if count > 0:
                            insertions.append((gap_index, count, region))
                    for gap_index, count, region in reversed(insertions):
                        new_tokens = [
                            _DynamicToken(
                                self.mask_id,
                                frozen=False,
                                left_region=region,
                                right_region=region,
                            )
                            for _ in range(count)
                        ]
                        row[gap_index + 1 : gap_index + 1] = new_tokens

                mask_candidates = []
                for position, token in enumerate(original):
                    if token.token_id != self.mask_id or token.frozen:
                        continue
                    logits = self._valid_logits(
                        output["logits"][row_index, position], row_ids, position
                    )
                    probability = torch.softmax(logits.float(), dim=-1).max()
                    calibrated = torch.sigmoid(
                        output["confidence_logits"][row_index, position].float()
                    )
                    score = float((probability * calibrated).item())
                    mask_candidates.append((score, position, token, logits))
                if mask_candidates:
                    remaining_steps = max(1, num_steps - step)
                    reveal = max(
                        1,
                        math.ceil(len(mask_candidates) / remaining_steps),
                    )
                    if progress > insertion_stop:
                        reveal = max(
                            reveal,
                            math.ceil(
                                len(mask_candidates)
                                * min(1.0, (progress - insertion_stop) / max(1e-6, 1.0 - insertion_stop))
                            ),
                        )
                    mask_candidates.sort(key=lambda item: item[0], reverse=True)
                    for _score, _position, token, logits in mask_candidates[:reveal]:
                        token.token_id = self._draw_token(
                            logits, token_temperature, stochastic
                        )

                if progress >= refinement_start:
                    delete_candidates = []
                    replace_candidates = []
                    for position, token in enumerate(original):
                        if token.frozen or token.token_id == self.mask_id:
                            continue
                        delete_probability = torch.sigmoid(
                            output["delete_logits"][row_index, position].float()
                        ).item()
                        if delete_probability >= deletion_threshold:
                            delete_candidates.append(
                                (delete_probability, token)
                            )

                        logits = self._valid_logits(
                            output["logits"][row_index, position],
                            row_ids,
                            position,
                        ).float()
                        log_probs = torch.log_softmax(logits, dim=-1)
                        proposal = int(log_probs.argmax().item())
                        current = int(token.token_id)
                        if proposal == current or current >= log_probs.numel():
                            continue
                        gain = float(
                            (log_probs[proposal] - log_probs[current]).item()
                        )
                        if gain >= replacement_logprob_gain:
                            replace_candidates.append((gain, token, proposal))

                    if delete_candidates and len(row) > 2:
                        delete_candidates.sort(reverse=True, key=lambda item: item[0])
                        max_delete = 1 if progress < 0.85 else 2
                        for _score, token in delete_candidates[:max_delete]:
                            identity_index = next(
                                (
                                    index
                                    for index, candidate in enumerate(row)
                                    if candidate is token
                                ),
                                None,
                            )
                            if identity_index is not None and len(row) > 2:
                                row.pop(identity_index)
                    if replace_candidates:
                        replace_candidates.sort(reverse=True, key=lambda item: item[0])
                        max_replace = 1 if progress < 0.85 else 2
                        for _gain, token, proposal in replace_candidates[:max_replace]:
                            if any(candidate is token for candidate in row):
                                token.token_id = proposal

        # Guarantee a complete sequence after the last insertion step.
        input_ids, attention_mask = self._tensorize(states)
        output = self.model(
            input_ids=input_ids,
            attention_mask=attention_mask,
            corruption_level=torch.zeros(
                len(states), device=self.device, dtype=torch.float32
            ),
            return_aux=True,
        )
        for row_index, row in enumerate(states):
            row_ids = [token.token_id for token in row]
            for position, token in enumerate(row):
                if token.token_id != self.mask_id:
                    continue
                logits = self._valid_logits(
                    output["logits"][row_index, position], row_ids, position
                )
                token.token_id = int(logits.argmax().item())
        result = [
            torch.tensor(
                [token.token_id for token in row], dtype=torch.long
            )
            for row in states
        ]
        if not return_frozen_mask:
            return result
        editable = [
            [not token.frozen for token in row]
            for row in states
        ]
        return result, editable


@torch.no_grad()
def sample_unified_local_infill(
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
    temperature_start=1.2,
    temperature_end=0.2,
    max_growth=8,
    max_shrink=8,
    return_seed_indices=False,
    return_diagnostics=False,
):
    """Adapter from existing atom-span plans to unified dynamic infilling."""
    if not getattr(model, "is_unified", False):
        return []
    if not seed_smiles or edit_plans is None:
        return []

    # Reuse the established plan normalizer so task heads retain their exact
    # protected spans and explicit min/max length constraints.
    from CSDNet.util.elastic_sampling import (
        _build_local_infill_state,
        _repair_final_sequences,
    )

    special = {tk.pad_id, tk.mask_id, tk.bos_id, tk.eos_id, tk.unk_id}
    valid_ids = [
        token_id for token_id in range(tk.vocab_size) if token_id not in special
    ]
    sampler = UnifiedDynamicSampler(
        model=model,
        pad_id=tk.pad_id,
        mask_id=tk.mask_id,
        bos_id=tk.bos_id,
        eos_id=tk.eos_id,
        valid_token_ids=valid_ids,
        max_len=max_len,
        max_gap_count=int(getattr(model, "max_gap_count", 8)),
    )

    def legacy_bounded_plan(plan):
        rows = plan if isinstance(plan, (list, tuple)) else [plan]
        bounded = []
        for row in rows:
            if not isinstance(row, dict):
                continue
            item = dict(row)
            removed = max(
                1,
                int(item.get("stop", 1)) - int(item.get("start", 0)),
            )
            item.setdefault(
                "min_replacement_len",
                max(0, removed - max(0, int(max_shrink))),
            )
            item.setdefault(
                "max_replacement_len",
                removed + max(0, int(max_growth)),
            )
            bounded.append(item)
        if isinstance(plan, (list, tuple)):
            return bounded
        return bounded[0] if bounded else None

    generated = []
    from rdkit import Chem
    for offset in range(0, len(seed_smiles), max(1, int(batch_size))):
        input_rows = []
        frozen_rows = []
        editable_rows = []
        limit_rows = []
        metadata = []
        for local_index, (smiles, plan) in enumerate(
            zip(
                seed_smiles[offset : offset + batch_size],
                edit_plans[offset : offset + batch_size],
            )
        ):
            state = _build_local_infill_state(
                smiles,
                legacy_bounded_plan(plan),
                tk=tk,
                max_len=max_len,
            )
            if state is None:
                continue
            anchor_pairs = sorted(
                (
                    gap_id,
                    position,
                )
                for position, gap_id in enumerate(state["anchors"])
                if gap_id >= 0
            )
            editable = []
            limits = []
            for gap_id, anchor_position in anchor_pairs:
                editable.append(max(0, anchor_position - 1))
                gap = state["gaps"][gap_id]
                limits.append((gap["minimum"], gap["maximum"]))
            if not editable:
                continue
            input_rows.append(state["tokens"])
            frozen_rows.append([True] * len(state["tokens"]))
            editable_rows.append(editable)
            limit_rows.append(limits)
            metadata.append(
                {
                    "source_index": offset + local_index,
                    "removed_tokens": sum(
                        int(gap["removed"]) for gap in state["gaps"]
                    ),
                    "initial_length": len(state["tokens"]),
                }
            )
        if not input_rows:
            continue

        sequences, editable_masks = sampler.sample_fragment(
            input_rows,
            frozen_mask=frozen_rows,
            editable_gaps=editable_rows,
            region_limits=limit_rows,
            num_steps=n_steps,
            temperature_start=temperature_start,
            temperature_end=temperature_end,
            return_frozen_mask=True,
        )
        repaired = _repair_final_sequences(
            model=model,
            tk=tk,
            sequences=[sequence.tolist() for sequence in sequences],
            device=device,
            use_fsm_check=use_fsm_check,
            use_rdkit_kekulize_check=use_rdkit_kekulize_check,
            max_sample_retries=max_sample_retries,
            violation_neighborhood=violation_neighborhood,
            temperature=temperature_end,
            top_k=0,
            top_p=1.0,
            editable_sequences=editable_masks,
        )
        for sequence, info in zip(repaired, metadata):
            if tk.eos_id in sequence:
                sequence = sequence[: sequence.index(tk.eos_id) + 1]
            smiles = tk.decode(sequence).strip("'\"")
            molecule = Chem.MolFromSmiles(smiles)
            if molecule is None:
                continue
            smiles = Chem.MolToSmiles(molecule, canonical=True)
            inserted = max(0, len(sequence) - info["initial_length"])
            diagnostics = {
                "length_mode": "unified_learned_insertion",
                "removed_tokens": info["removed_tokens"],
                "inserted_tokens": inserted,
                "actual_delta": inserted - info["removed_tokens"],
            }
            if return_seed_indices and return_diagnostics:
                generated.append((smiles, info["source_index"], diagnostics))
            elif return_seed_indices:
                generated.append((smiles, info["source_index"]))
            elif return_diagnostics:
                generated.append((smiles, diagnostics))
            else:
                generated.append(smiles)
    return generated
