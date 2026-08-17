import math
import random

import lightning as L
import numpy as np
import torch
import torch.nn.functional as F

from CSDNet.model.elastic_backbone import ElasticCSDNetBackbone
from CSDNet.model.elastic_schedule import (
    ElasticKumaSchedule,
    apply_structured_span_mask,
    bregman_poisson,
    sample_variable_length_state,
)
from CSDNet.model.lightning_module import SimpleEMA
from CSDNet.model.loss_weighting import redistribute_priority_with_fixed_mass


class ElasticCSDNetLightningModule(L.LightningModule):
    """
    Joint token, insertion, and learned-order training for a compact CSDNet.

    Two variable-length trajectories provide a leave-one-out baseline for the
    learned clean-sequence schedule. The noisy-sequence heads are trained by
    token reconstruction and Poisson Bregman rate matching.
    """

    def __init__(
        self,
        vocab_size,
        pad_id,
        mask_id,
        bos_id,
        eos_id,
        unk_id,
        scaffold_ids,
        aromatic_ids=None,
        cond_dim=0,
        use_ema=True,
        use_cbi=False,
        cbi_weight=2.0,
        use_aromatic_cbi=True,
        normalized_cbi=True,
        aromatic_cbi_weight=1.2,
        aromatic_cbi_final_weight=1.0,
        aromatic_cbi_anneal_steps=100000,
        aromatic_cbi_anneal_start_fraction=0.10,
        aromatic_cbi_anneal_end_fraction=0.90,
        structured_corruption_prob=0.10,
        structured_span_min=2,
        structured_span_max=8,
        lr=5e-4,
        min_lr=None,
        warmup_steps=2000,
        lr_schedule="cosine",
        weight_decay=0.01,
        adam_beta1=0.9,
        adam_beta2=0.98,
        adam_eps=1e-8,
        papl_alpha=3.0,
        papl_tau=1.0,
        ema_decay=0.9999,
        hidden_size=256,
        num_layers=8,
        num_heads=8,
        intermediate=1024,
        max_position_embeddings=256,
        position_embedding_type="rotary",
        hidden_dropout_prob=0.1,
        attention_probs_dropout_prob=0.1,
        layer_norm_eps=1e-12,
        initializer_range=0.02,
        rate_min=0.001,
        rate_max=20.0,
        rate_initial=1.0,
        rate_parameterization="sigmoid",
        theta_rate_min=None,
        phi_rate_min=None,
        rate_output_bias=None,
        fixed_unmask_rate=1.0,
        kuma_shape_a=2.0,
        insertion_loss_weight=1.0,
        reinforce_weight=1.0,
        schedule_regularizer_weight=1.0,
        loflex_objective=False,
        policy_time_conditioning=False,
        fragment_corruption_prob=0.0,
        fragment_internal_probability=0.25,
        fragment_terminal_probability=0.25,
        fragment_dual_probability=0.25,
        fragment_multi_probability=0.25,
        mdm_corruption_prob=0.0,
        refine_corruption_prob=0.0,
        refine_fraction_min=0.05,
        refine_fraction_max=0.20,
        length_loss_normalizer=256.0,
        drop_cond_prob=0.0,
        gradient_checkpointing=True,
    ):
        super().__init__()
        self.save_hyperparameters(ignore=["scaffold_ids", "aromatic_ids"])
        self.architecture_type = "elastic_csdnet"

        self.pad_id = int(pad_id)
        self.mask_id = int(mask_id)
        self.bos_id = int(bos_id)
        self.eos_id = int(eos_id)
        self.unk_id = int(unk_id)
        self.lr = float(lr)
        self.min_lr = (
            None if min_lr is None else float(min_lr)
        )
        self.warmup = int(warmup_steps)
        self.lr_schedule = str(lr_schedule)
        self.weight_decay = float(weight_decay)
        self.adam_beta1 = float(adam_beta1)
        self.adam_beta2 = float(adam_beta2)
        self.adam_eps = float(adam_eps)
        if self.lr_schedule not in {"cosine", "constant_with_warmup"}:
            raise ValueError(
                "lr_schedule must be 'cosine' or 'constant_with_warmup', "
                f"got {self.lr_schedule!r}"
            )
        if self.min_lr is not None and not 0.0 <= self.min_lr <= self.lr:
            raise ValueError("min_lr must lie between zero and lr.")
        self.alpha = float(papl_alpha)
        self.tau = float(papl_tau)
        self.ema_decay = float(ema_decay)
        self.use_ema = bool(use_ema)
        self.use_cbi = bool(use_cbi)
        self.cbi_weight = float(cbi_weight)
        self.use_aromatic_cbi = bool(use_aromatic_cbi)
        self.normalized_cbi = bool(normalized_cbi)
        self.aromatic_cbi_weight = float(aromatic_cbi_weight)
        self.aromatic_cbi_final_weight = float(aromatic_cbi_final_weight)
        self.aromatic_cbi_anneal_steps = int(aromatic_cbi_anneal_steps)
        self.aromatic_cbi_anneal_start_fraction = float(
            aromatic_cbi_anneal_start_fraction
        )
        self.aromatic_cbi_anneal_end_fraction = float(
            aromatic_cbi_anneal_end_fraction
        )
        self.structured_corruption_prob = float(structured_corruption_prob)
        self.structured_span_min = int(structured_span_min)
        self.structured_span_max = int(structured_span_max)
        self.insertion_loss_weight = float(insertion_loss_weight)
        self.reinforce_weight = float(reinforce_weight)
        self.schedule_regularizer_weight = float(schedule_regularizer_weight)
        self.loflex_objective = bool(loflex_objective)
        self.policy_time_conditioning = bool(policy_time_conditioning)
        self.fragment_corruption_prob = float(fragment_corruption_prob)
        self.fragment_geometry_probabilities = (
            float(fragment_internal_probability),
            float(fragment_terminal_probability),
            float(fragment_dual_probability),
            float(fragment_multi_probability),
        )
        self.mdm_corruption_prob = float(mdm_corruption_prob)
        self.refine_corruption_prob = float(refine_corruption_prob)
        self.refine_fraction_min = float(refine_fraction_min)
        self.refine_fraction_max = float(refine_fraction_max)
        self.length_loss_normalizer = float(length_loss_normalizer)
        self.fixed_unmask_rate = (
            None
            if fixed_unmask_rate is None
            else float(fixed_unmask_rate)
        )
        self.drop_cond_prob = float(drop_cond_prob)
        if not 0.0 <= self.structured_corruption_prob <= 1.0:
            raise ValueError("structured_corruption_prob must be in [0, 1].")
        aligned_probability = (
            self.fragment_corruption_prob
            + self.mdm_corruption_prob
            + self.refine_corruption_prob
        )
        if any(
            probability < 0.0
            for probability in (
                self.fragment_corruption_prob,
                self.mdm_corruption_prob,
                self.refine_corruption_prob,
            )
        ) or aligned_probability > 1.0:
            raise ValueError(
                "Aligned corruption probabilities must be non-negative and "
                "sum to at most one."
            )
        if any(
            probability < 0.0
            for probability in self.fragment_geometry_probabilities
        ) or not math.isclose(
            sum(self.fragment_geometry_probabilities),
            1.0,
            rel_tol=0.0,
            abs_tol=1e-8,
        ):
            raise ValueError(
                "Fragment geometry probabilities must be non-negative and "
                "sum to one."
            )
        if not 0.0 < self.refine_fraction_min <= self.refine_fraction_max <= 1.0:
            raise ValueError(
                "Refinement fractions must satisfy 0 < min <= max <= 1."
            )
        if self.length_loss_normalizer <= 0:
            raise ValueError("length_loss_normalizer must be positive.")
        if self.normalized_cbi and not (
            self.use_cbi or self.use_aromatic_cbi
        ):
            raise ValueError(
                "normalized_cbi requires at least one chemical weighting mode."
            )
        if self.structured_span_min < 1:
            raise ValueError("structured_span_min must be positive.")
        if self.structured_span_max < self.structured_span_min:
            raise ValueError(
                "structured_span_max must be at least structured_span_min."
            )
        if self.aromatic_cbi_anneal_steps < 1:
            raise ValueError("aromatic_cbi_anneal_steps must be positive.")
        if not (
            0.0
            <= self.aromatic_cbi_anneal_start_fraction
            <= self.aromatic_cbi_anneal_end_fraction
            <= 1.0
        ):
            raise ValueError(
                "AROCBI anneal fractions must satisfy 0 <= start <= end <= 1."
            )

        scaffold_buffer = torch.zeros(vocab_size, dtype=torch.bool)
        for token_id in scaffold_ids:
            scaffold_buffer[int(token_id)] = True
        self.register_buffer("_scaffold_buf", scaffold_buffer)

        aromatic_buffer = torch.zeros(vocab_size, dtype=torch.bool)
        for token_id in aromatic_ids or frozenset():
            aromatic_buffer[int(token_id)] = True
        self.register_buffer("_aromatic_buf", aromatic_buffer)

        self.backbone = ElasticCSDNetBackbone(
            vocab_size=vocab_size,
            cond_dim=cond_dim,
            hidden_size=hidden_size,
            num_layers=num_layers,
            num_heads=num_heads,
            intermediate=intermediate,
            pad_token_id=self.pad_id,
            mask_token_id=self.mask_id,
            max_position_embeddings=max_position_embeddings,
            position_embedding_type=position_embedding_type,
            hidden_dropout_prob=hidden_dropout_prob,
            attention_probs_dropout_prob=attention_probs_dropout_prob,
            layer_norm_eps=layer_norm_eps,
            initializer_range=initializer_range,
            rate_min=rate_min,
            rate_max=rate_max,
            rate_initial=rate_initial,
            rate_parameterization=rate_parameterization,
            theta_rate_min=theta_rate_min,
            phi_rate_min=phi_rate_min,
            rate_output_bias=rate_output_bias,
            fixed_unmask_rate=self.fixed_unmask_rate,
            kuma_shape_a=kuma_shape_a,
            gradient_checkpointing=gradient_checkpointing,
        )
        self.schedule = ElasticKumaSchedule(
            shape_a=kuma_shape_a,
            regularizer_mode=("loflex" if self.loflex_objective else "legacy"),
        )
        self.ema = (
            SimpleEMA(self.backbone, decay=self.ema_decay)
            if self.use_ema
            else None
        )

    def _special_mask(self, token_ids):
        special = (
            (token_ids == self.bos_id)
            | (token_ids == self.eos_id)
            | (token_ids == self.pad_id)
        )
        if self.unk_id != -1:
            special = special | (token_ids == self.unk_id)
        return special

    @staticmethod
    def _phi_schedule_time(t, use_observation_time=False):
        """Match the selected clean-policy parameterization."""
        return t if use_observation_time else torch.ones_like(t)

    @staticmethod
    def _masked_row_mean(values, mask):
        mask_float = mask.to(dtype=values.dtype)
        return (values * mask_float).sum(dim=-1) / mask_float.sum(
            dim=-1
        ).clamp(min=1.0)

    def _sample_aligned_modes(self, batch_size, device, is_train):
        modes = torch.zeros(batch_size, device=device, dtype=torch.long)
        if not is_train or not self.loflex_objective:
            return modes

        draws = torch.rand(batch_size, device=device)
        boundary = self.fragment_corruption_prob
        modes[draws < boundary] = 1
        boundary += self.mdm_corruption_prob
        modes[(draws >= boundary - self.mdm_corruption_prob) & (draws < boundary)] = 2
        boundary += self.refine_corruption_prob
        modes[
            (draws >= boundary - self.refine_corruption_prob)
            & (draws < boundary)
        ] = 3
        return modes

    def _sample_fragment_active_mask(self, active, fragment_rows):
        """Choose a balanced, task-agnostic editable-span geometry."""
        output = active.clone()
        geometry = torch.full(
            (active.size(0),),
            -1,
            device=active.device,
            dtype=torch.long,
        )
        probabilities = torch.tensor(
            self.fragment_geometry_probabilities,
            device=active.device,
            dtype=torch.float32,
        )

        def mark_random_spans(row_output, positions, count, minimum, maximum):
            available = torch.ones(
                positions.numel(),
                device=active.device,
                dtype=torch.bool,
            )
            selected_count = 0
            for _ in range(count):
                runs = []
                start = None
                for index in range(positions.numel() + 1):
                    enabled = index < positions.numel() and bool(available[index])
                    if enabled and start is None:
                        start = index
                    if (not enabled or index == positions.numel()) and start is not None:
                        if index - start >= minimum:
                            runs.append((start, index))
                        start = None
                if not runs:
                    break
                run_index = int(
                    torch.randint(len(runs), (1,), device=active.device).item()
                )
                run_start, run_stop = runs[run_index]
                upper = min(maximum, run_stop - run_start)
                lower = min(minimum, upper)
                length = int(
                    torch.randint(
                        lower,
                        upper + 1,
                        (1,),
                        device=active.device,
                    ).item()
                )
                offset = int(
                    torch.randint(
                        run_stop - run_start - length + 1,
                        (1,),
                        device=active.device,
                    ).item()
                )
                span_start = run_start + offset
                span_stop = span_start + length
                row_output[positions[span_start:span_stop]] = True
                # Preserve at least one context token between independent gaps.
                available[
                    max(0, span_start - 1) : min(
                        positions.numel(),
                        span_stop + 1,
                    )
                ] = False
                selected_count += 1
            return selected_count

        for row_tensor in fragment_rows.nonzero(as_tuple=False).flatten():
            row = int(row_tensor.item())
            positions = active[row].nonzero(as_tuple=False).flatten()
            output[row].zero_()
            if positions.numel() == 0:
                continue
            geometry_id = int(
                torch.multinomial(probabilities, 1).item()
            )
            geometry[row] = geometry_id
            token_count = int(positions.numel())

            if geometry_id == 0 and token_count >= 3:
                maximum = min(self.structured_span_max, token_count - 2)
                minimum = min(self.structured_span_min, maximum)
                span_length = int(
                    torch.randint(
                        minimum,
                        maximum + 1,
                        (1,),
                        device=active.device,
                    ).item()
                )
                start = int(
                    torch.randint(
                        1,
                        token_count - span_length,
                        (1,),
                        device=active.device,
                    ).item()
                )
                output[row, positions[start : start + span_length]] = True
            elif geometry_id == 1 and token_count >= 2:
                maximum = min(self.structured_span_max, token_count - 1)
                minimum = min(self.structured_span_min, maximum)
                span_length = int(
                    torch.randint(
                        minimum,
                        maximum + 1,
                        (1,),
                        device=active.device,
                    ).item()
                )
                if bool(torch.randint(2, (1,), device=active.device).item()):
                    output[row, positions[:span_length]] = True
                else:
                    output[row, positions[-span_length:]] = True
            elif geometry_id == 2:
                mark_random_spans(
                    output[row],
                    positions,
                    count=2,
                    minimum=min(self.structured_span_min, 2),
                    maximum=min(self.structured_span_max, 12),
                )
            else:
                requested = int(
                    torch.randint(3, 7, (1,), device=active.device).item()
                )
                mark_random_spans(
                    output[row],
                    positions,
                    count=requested,
                    minimum=1,
                    maximum=min(4, self.structured_span_max),
                )

            if not output[row].any():
                fallback_length = min(
                    max(1, self.structured_span_min),
                    token_count,
                )
                start = int(
                    torch.randint(
                        token_count - fallback_length + 1,
                        (1,),
                        device=active.device,
                    ).item()
                )
                output[row, positions[start : start + fallback_length]] = True
        return output, geometry

    def _apply_visible_token_corruption(self, state, clean_ids, active, rows):
        """Create realistic visible-token errors for the refinement branch."""
        replacement_mask = torch.zeros_like(clean_ids, dtype=torch.bool)
        pool = clean_ids[active]
        if pool.numel() == 0:
            state["replacement_mask"] = replacement_mask
            return state

        for row_tensor in rows.nonzero(as_tuple=False).flatten():
            row = int(row_tensor.item())
            positions = active[row].nonzero(as_tuple=False).flatten()
            if positions.numel() == 0:
                continue
            fraction = self.refine_fraction_min + torch.rand(
                (), device=clean_ids.device
            ) * (self.refine_fraction_max - self.refine_fraction_min)
            count = max(1, int(round(float(fraction) * positions.numel())))
            chosen = positions[
                torch.randperm(positions.numel(), device=clean_ids.device)[:count]
            ]
            replacement = pool[
                torch.randint(
                    pool.numel(),
                    (chosen.numel(),),
                    device=clean_ids.device,
                )
            ]
            same = replacement.eq(clean_ids[row, chosen])
            if same.any() and pool.numel() > 1:
                replacement[same] = pool[
                    torch.randint(
                        pool.numel(),
                        (int(same.sum().item()),),
                        device=clean_ids.device,
                    )
                ]
            state["input_ids"][row, chosen] = replacement
            replacement_mask[row, chosen] = True
        state["replacement_mask"] = replacement_mask
        return state

    def _aromatic_weight(self, step=None):
        if not self.use_aromatic_cbi:
            return 1.0
        if step is None:
            trainer = getattr(self, "_trainer", None)
            step = int(trainer.global_step) if trainer is not None else 0
        progress = min(
            max(float(step) / float(self.aromatic_cbi_anneal_steps), 0.0),
            1.0,
        )
        start = self.aromatic_cbi_anneal_start_fraction
        end = self.aromatic_cbi_anneal_end_fraction
        if progress <= start:
            mixture = 0.0
        elif progress >= end or end <= start:
            mixture = 1.0
        else:
            phase = (progress - start) / (end - start)
            mixture = 0.5 - 0.5 * math.cos(math.pi * phase)
        return (
            self.aromatic_cbi_weight
            + mixture
            * (self.aromatic_cbi_final_weight - self.aromatic_cbi_weight)
        )

    def _token_multiplier(self, target_ids, valid, log_target, aromatic_mask):
        with torch.no_grad():
            scaled = log_target.masked_fill(~valid, -1e9) / (self.tau + 1e-8)
            papl = torch.softmax(scaled, dim=-1)
            papl = papl * valid.sum(dim=-1, keepdim=True).float().clamp(min=1.0)
        multiplier = 1.0 + self.alpha * papl

        chemical_multiplier = torch.ones_like(log_target)
        if self.use_cbi:
            chemical_multiplier = torch.where(
                self._scaffold_buf[target_ids],
                torch.full_like(chemical_multiplier, self.cbi_weight),
                chemical_multiplier,
            )
        if self.use_aromatic_cbi:
            aromatic_weight = self._aromatic_weight()
            if aromatic_mask is None:
                is_aromatic = self._aromatic_buf[target_ids]
            else:
                is_aromatic = aromatic_mask.bool()
            chemical_multiplier = torch.where(
                is_aromatic,
                torch.maximum(
                    chemical_multiplier,
                    torch.full_like(
                        chemical_multiplier,
                        aromatic_weight,
                    ),
                ),
                chemical_multiplier,
            )
        if getattr(self, "normalized_cbi", True):
            return redistribute_priority_with_fixed_mass(
                multiplier,
                chemical_multiplier,
                valid,
            )
        return multiplier * chemical_multiplier

    def _trajectory_loss(
        self,
        clean_ids,
        clean_aromatic_mask,
        t,
        state,
        phi_insertion_hazard,
        phi_unmask_hazard,
        cond,
        drop_cond,
        theta=None,
    ):
        noisy_ids = state["input_ids"]
        theta_time = state.get("model_time", t)
        noisy_attention = noisy_ids.ne(self.pad_id)
        if theta is None:
            theta = self.backbone(
                noisy_ids,
                noisy_attention.long(),
                cond=cond,
                drop_cond=drop_cond,
                t=theta_time,
                return_aux=True,
                rate_family="theta",
            )
        source_positions = state["source_positions"]
        target_ids = torch.gather(clean_ids, 1, source_positions)
        target_aromatic = (
            torch.gather(clean_aromatic_mask, 1, source_positions)
            if clean_aromatic_mask is not None
            else None
        )
        target_unmask_hazard = torch.gather(
            phi_unmask_hazard,
            1,
            source_positions,
        )

        log_probabilities = F.log_softmax(theta["logits"].float(), dim=-1)
        log_target = torch.gather(
            log_probabilities,
            2,
            target_ids.unsqueeze(-1),
        ).squeeze(-1)
        masked_valid = (
            noisy_ids.eq(self.mask_id)
            & noisy_attention
            & ~self._special_mask(target_ids)
        )
        replacement_mask = state.get("replacement_mask")
        if replacement_mask is None:
            replacement_mask = torch.zeros_like(masked_valid)
        else:
            replacement_mask = replacement_mask.bool() & noisy_attention
        token_valid = masked_valid | replacement_mask
        structured_mask = state.get("structured_mask")
        if structured_mask is None:
            structured_mask = torch.zeros_like(token_valid)
        else:
            structured_mask = structured_mask.bool() & token_valid
        process_mask = masked_valid & ~structured_mask
        multiplier = self._token_multiplier(
            target_ids,
            token_valid,
            log_target,
            target_aromatic,
        )
        reconstruction_weight = torch.where(
            structured_mask | replacement_mask,
            torch.ones_like(target_unmask_hazard),
            target_unmask_hazard,
        )
        if not self.loflex_objective:
            reconstruction_weight = reconstruction_weight.detach()
        reconstruction = (
            -reconstruction_weight * log_target * multiplier
        )

        unmask_loss = torch.where(
            token_valid,
            reconstruction,
            torch.zeros_like(reconstruction),
        )
        if self.fixed_unmask_rate is None:
            theta_unmask_hazard = self.schedule.hazard(
                theta_time.unsqueeze(-1),
                theta["b_unmask"],
            )
            unmask_rate_loss = bregman_poisson(
                (
                    target_unmask_hazard
                    if self.loflex_objective
                    else target_unmask_hazard.detach()
                ),
                theta_unmask_hazard,
            )
            unmask_loss = unmask_loss + torch.where(
                process_mask,
                unmask_rate_loss,
                torch.zeros_like(unmask_rate_loss),
            )
        unmask_loss = self._masked_row_mean(unmask_loss, token_valid)

        theta_insertion_hazard = self.schedule.hazard(
            theta_time.unsqueeze(-1),
            theta["b_ins"],
        )
        insertion_loss = bregman_poisson(
            (
                state["gap_rate_target"]
                if self.loflex_objective
                else state["gap_rate_target"].detach()
            ),
            theta_insertion_hazard,
        )
        predictable_gap = state["gap_mask"]
        insertion_loss = torch.where(
            predictable_gap,
            insertion_loss,
            torch.zeros_like(insertion_loss),
        )
        if self.loflex_objective:
            insertion_loss = insertion_loss.sum(dim=-1) / self.length_loss_normalizer
        else:
            insertion_loss = self._masked_row_mean(
                insertion_loss,
                predictable_gap,
            )

        total = unmask_loss + self.insertion_loss_weight * insertion_loss
        return total, unmask_loss, insertion_loss

    def _merged_theta_outputs(self, states, t, cond, drop_cond):
        """Evaluate both noisy trajectories in one shared backbone call."""
        if len(states) != 2:
            raise ValueError("Exactly two trajectories are required.")
        batch_size = states[0]["input_ids"].size(0)
        if any(
            state["input_ids"].shape != states[0]["input_ids"].shape
            for state in states[1:]
        ):
            raise ValueError("Trajectory tensors must have matching shapes.")

        noisy_ids = torch.cat(
            [state["input_ids"] for state in states],
            dim=0,
        )
        noisy_attention = noisy_ids.ne(self.pad_id).long()
        merged_time = torch.cat(
            [state.get("model_time", t) for state in states],
            dim=0,
        )
        merged_cond = (
            None
            if cond is None
            else torch.cat([cond, cond], dim=0)
        )
        merged = self.backbone(
            noisy_ids,
            noisy_attention,
            cond=merged_cond,
            drop_cond=drop_cond,
            t=merged_time,
            return_aux=True,
            rate_family="theta",
        )

        outputs = []
        for trajectory_index in range(2):
            start = trajectory_index * batch_size
            end = start + batch_size
            outputs.append(
                {
                    name: (
                        value[start:end]
                        if torch.is_tensor(value)
                        and value.ndim > 0
                        and value.size(0) == 2 * batch_size
                        else value
                    )
                    for name, value in merged.items()
                }
            )
        return outputs

    def _trajectory_log_probability(self, phi, t, active, state):
        dropped, masked, unmasked = self.schedule.state_log_probabilities(
            t,
            phi["b_ins"],
            phi["b_unmask"],
        )
        selected = torch.where(
            state["deleted"],
            dropped,
            torch.where(state["masked"], masked, unmasked),
        )
        total = (selected * active.float()).sum(dim=-1)
        if self.loflex_objective:
            return total
        return total / active.sum(dim=-1).clamp(min=1)

    def _step(self, batch, is_train):
        clean_ids = batch["input_ids"]
        attention_mask = batch["attention_mask"].bool()
        cond = batch.get("cond")
        if cond is not None and cond.numel() == 0:
            cond = None
        aromatic_mask = batch.get("aromatic_mask")
        if aromatic_mask is not None:
            aromatic_mask = aromatic_mask.bool()

        base_fixed = self._special_mask(clean_ids) | ~attention_mask
        active = attention_mask & ~base_fixed
        t = self.schedule.sample_time(clean_ids.size(0), clean_ids.device)
        drop_cond = (
            cond is not None
            and is_train
            and random.random() < self.drop_cond_prob
        )

        modes = self._sample_aligned_modes(
            clean_ids.size(0),
            clean_ids.device,
            is_train,
        )
        fragment_rows = modes.eq(1)
        mdm_rows = modes.eq(2)
        refine_rows = modes.eq(3)
        policy_rows = modes.le(1)
        process_active, fragment_geometry = self._sample_fragment_active_mask(
            active,
            fragment_rows,
        )
        fixed = base_fixed | (fragment_rows.unsqueeze(-1) & ~process_active)
        policy_active = process_active & policy_rows.unsqueeze(-1)

        phi = self.backbone(
            clean_ids,
            attention_mask.long(),
            cond=cond,
            drop_cond=drop_cond,
            t=self._phi_schedule_time(
                t,
                use_observation_time=(
                    self.loflex_objective
                    and self.policy_time_conditioning
                ),
            ),
            return_aux=True,
            rate_family="phi",
            compute_logits=False,
        )
        phi_insertion_hazard = self.schedule.hazard(
            t.unsqueeze(-1),
            phi["b_ins"],
        )
        phi_unmask_hazard = self.schedule.hazard(
            t.unsqueeze(-1),
            phi["b_unmask"],
        )

        structured_selected = torch.zeros(
            clean_ids.size(0),
            dtype=torch.bool,
            device=clean_ids.device,
        )
        if (
            not self.loflex_objective
            and is_train
            and self.structured_corruption_prob > 0.0
        ):
            structured_selected = (
                torch.rand(clean_ids.size(0), device=clean_ids.device)
                < self.structured_corruption_prob
            )

        states = []
        structured_applied = torch.zeros_like(structured_selected)
        for _ in range(2):
            insertion_times = self.schedule.sample_event_time(phi["b_ins"])
            unmask_times = self.schedule.sample_truncated_event_time(
                insertion_times,
                phi["b_unmask"],
            )
            no_insertion_rows = mdm_rows | refine_rows
            insertion_times = torch.where(
                no_insertion_rows.unsqueeze(-1) & active,
                torch.zeros_like(insertion_times),
                insertion_times,
            )
            if mdm_rows.any():
                mdm_unmask_times = self.schedule.sample_truncated_event_time(
                    torch.zeros_like(insertion_times),
                    phi["b_unmask"],
                )
                unmask_times = torch.where(
                    mdm_rows.unsqueeze(-1) & active,
                    mdm_unmask_times,
                    unmask_times,
                )
            unmask_times = torch.where(
                refine_rows.unsqueeze(-1) & active,
                torch.zeros_like(unmask_times),
                unmask_times,
            )
            state = sample_variable_length_state(
                clean_ids=clean_ids,
                t=t,
                insertion_times=insertion_times,
                unmask_times=unmask_times,
                insertion_hazard=(
                    phi_insertion_hazard
                    if self.loflex_objective
                    else phi_insertion_hazard.detach()
                ),
                fixed=fixed,
                mask_id=self.mask_id,
                pad_id=self.pad_id,
            )
            state["model_time"] = torch.where(
                refine_rows,
                0.75 + 0.25 * t,
                t,
            )
            if refine_rows.any():
                state = self._apply_visible_token_corruption(
                    state,
                    clean_ids,
                    active,
                    refine_rows,
                )
            if structured_selected.any():
                process_mask = state["input_ids"].eq(self.mask_id)
                state, applied = apply_structured_span_mask(
                    state=state,
                    fixed=fixed,
                    mask_id=self.mask_id,
                    pad_id=self.pad_id,
                    selected_samples=structured_selected,
                    min_span=self.structured_span_min,
                    max_span=self.structured_span_max,
                )
                state["structured_mask"] = (
                    state["input_ids"].eq(self.mask_id) & ~process_mask
                )
                structured_applied = structured_applied | applied
            states.append(state)

        theta_outputs = self._merged_theta_outputs(
            states=states,
            t=t,
            cond=cond,
            drop_cond=drop_cond,
        )
        trajectory_results = [
            self._trajectory_loss(
                clean_ids=clean_ids,
                clean_aromatic_mask=aromatic_mask,
                t=t,
                state=state,
                phi_insertion_hazard=phi_insertion_hazard,
                phi_unmask_hazard=phi_unmask_hazard,
                cond=cond,
                drop_cond=drop_cond,
                theta=theta,
            )
            for state, theta in zip(states, theta_outputs)
        ]
        loss_one, unmask_one, insertion_one = trajectory_results[0]
        loss_two, unmask_two, insertion_two = trajectory_results[1]
        theta_loss = 0.5 * (loss_one + loss_two)

        log_prob_one = self._trajectory_log_probability(
            phi,
            t,
            policy_active,
            states[0],
        )
        log_prob_two = self._trajectory_log_probability(
            phi,
            t,
            policy_active,
            states[1],
        )
        raw_advantage = loss_one.detach() - loss_two.detach()
        if self.loflex_objective:
            advantage = raw_advantage
        else:
            advantage_scale = (
                0.5 * (loss_one.detach().abs() + loss_two.detach().abs())
                + 1.0
            )
            advantage = (raw_advantage / advantage_scale).clamp(
                min=-2.0,
                max=2.0,
            )
        reinforce_loss = 0.5 * advantage * (
            log_prob_one - log_prob_two
        )
        reinforce_loss = torch.where(
            structured_applied | ~policy_rows,
            torch.zeros_like(reinforce_loss),
            reinforce_loss,
        )
        regularizer = self.schedule.regularizer(
            phi["b_ins"],
            (
                phi["b_unmask"]
                if self.loflex_objective
                or self.fixed_unmask_rate is None
                else None
            ),
            policy_active,
        )
        regularizer = torch.where(
            policy_rows,
            regularizer,
            torch.zeros_like(regularizer),
        )
        total_loss = (
            theta_loss
            + self.reinforce_weight * reinforce_loss
            + self.schedule_regularizer_weight * regularizer
        )
        fragment_denominator = fragment_rows.sum().clamp(min=1)
        return {
            "loss": total_loss.mean(),
            "theta_loss": theta_loss.mean().detach(),
            "unmask_loss": (0.5 * (unmask_one + unmask_two)).mean().detach(),
            "insertion_loss": (
                0.5 * (insertion_one + insertion_two)
            ).mean().detach(),
            "reinforce_loss": reinforce_loss.mean().detach(),
            "schedule_regularizer": regularizer.mean().detach(),
            "aromatic_cbi_weight": torch.tensor(
                self._aromatic_weight(),
                device=clean_ids.device,
            ),
            "structured_corruption_fraction": (
                structured_applied.float().mean().detach()
            ),
            "fragment_corruption_fraction": fragment_rows.float().mean().detach(),
            "fragment_internal_fraction": (
                fragment_geometry.eq(0).sum() / fragment_denominator
            ).detach(),
            "fragment_terminal_fraction": (
                fragment_geometry.eq(1).sum() / fragment_denominator
            ).detach(),
            "fragment_dual_fraction": (
                fragment_geometry.eq(2).sum() / fragment_denominator
            ).detach(),
            "fragment_multi_fraction": (
                fragment_geometry.eq(3).sum() / fragment_denominator
            ).detach(),
            "mdm_corruption_fraction": mdm_rows.float().mean().detach(),
            "refine_corruption_fraction": refine_rows.float().mean().detach(),
            "mean_phi_insertion_rate": (
                (phi["b_ins"] * policy_active.float()).sum()
                / policy_active.sum().clamp(min=1)
            ).detach(),
            "mean_phi_unmask_rate": (
                (phi["b_unmask"] * policy_active.float()).sum()
                / policy_active.sum().clamp(min=1)
            ).detach(),
        }

    def training_step(self, batch, _):
        losses = self._step(batch, is_train=True)
        self.log("train_loss", losses["loss"], on_step=True, prog_bar=True)
        for name, value in losses.items():
            if name != "loss":
                self.log(f"train_{name}", value, on_step=True)
        return losses["loss"]

    def optimizer_step(self, *args, **kwargs):
        super().optimizer_step(*args, **kwargs)
        if self.ema is not None:
            self.ema.update(self.backbone)

    def validation_step(self, batch, _):
        losses = self._step(batch, is_train=False)
        self.log("val_loss", losses["loss"], prog_bar=True, sync_dist=True)

    def configure_optimizers(self):
        optimizer = torch.optim.AdamW(
            self.parameters(),
            lr=self.lr,
            weight_decay=self.weight_decay,
            betas=(self.adam_beta1, self.adam_beta2),
            eps=self.adam_eps,
        )

        def learning_rate_multiplier(step):
            if step < self.warmup:
                return (step + 1) / max(self.warmup, 1)
            if self.lr_schedule == "constant_with_warmup":
                return 1.0
            try:
                total_steps = self.trainer.estimated_stepping_batches
                if total_steps == float("inf"):
                    total_steps = 100000
            except Exception:
                total_steps = 100000
            progress = (step - self.warmup) / max(
                total_steps - self.warmup,
                1,
            )
            minimum_ratio = (
                0.1
                if self.min_lr is None
                else self.min_lr / max(self.lr, 1e-12)
            )
            return minimum_ratio + (1.0 - minimum_ratio) * 0.5 * (
                1.0 + np.cos(np.pi * min(progress, 1.0))
            )

        scheduler = torch.optim.lr_scheduler.LambdaLR(
            optimizer,
            learning_rate_multiplier,
        )
        return {
            "optimizer": optimizer,
            "lr_scheduler": {"scheduler": scheduler, "interval": "step"},
        }
