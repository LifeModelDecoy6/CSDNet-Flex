import random
from contextlib import contextmanager

import lightning as L
import numpy as np
import torch
import torch.nn.functional as F

from CSDNet.model.backbone import CSDNetBackbone
from CSDNet.model.loss_weighting import redistribute_priority_with_fixed_mass


PROP_NAMES = ("qed", "logp", "sa", "tpsa", "mw")
PROP_SCALES = {
    "qed": 1.0,
    "logp": 5.0,
    "sa": 10.0,
    "tpsa": 140.0,
    "mw": 500.0,
}


def build_mdlm(vocab_size, mask_id):
    try:
        from bionemo.moco.distributions.prior import DiscreteMaskedPrior
        from bionemo.moco.distributions.time import UniformTimeDistribution
        from bionemo.moco.interpolants import MDLM
        from bionemo.moco.schedules.noise.continuous_noise_transforms import (
            LogLinearExpNoiseTransform,
        )
    except ImportError as exc:
        raise RuntimeError("Missing bionemo-moco dependency.") from exc

    return MDLM(
        time_distribution=UniformTimeDistribution(),
        prior_distribution=DiscreteMaskedPrior(num_classes=vocab_size, mask_dim=mask_id),
        noise_schedule=LogLinearExpNoiseTransform(),
    )


class SimpleEMA(torch.nn.Module):
    def __init__(self, model: torch.nn.Module, decay: float = 0.9999):
        super().__init__()
        self.decay = decay
        self.shadow_names = []
        for n, p in model.named_parameters():
            if p.requires_grad:
                safe_name = n.replace(".", "___")
                self.register_buffer(safe_name, p.detach().clone())
                self.shadow_names.append((n, safe_name))

    @torch.no_grad()
    def update(self, model: torch.nn.Module) -> None:
        for n, p in model.named_parameters():
            if p.requires_grad:
                safe_name = n.replace(".", "___")
                if hasattr(self, safe_name):
                    shadow = getattr(self, safe_name)
                    shadow.mul_(self.decay).add_(p.data, alpha=1.0 - self.decay)

    @contextmanager
    def apply(self, model: torch.nn.Module):
        orig = {n: p.data.clone() for n, p in model.named_parameters() if p.requires_grad}
        param_dict = dict(model.named_parameters())
        for n, safe_name in self.shadow_names:
            if hasattr(self, safe_name) and n in param_dict:
                shadow = getattr(self, safe_name)
                param_dict[n].data.copy_(shadow.to(param_dict[n].device))
        try:
            yield
        finally:
            for n, p in model.named_parameters():
                if p.requires_grad and n in orig:
                    p.data.copy_(orig[n])


class CSDNetLightningModule(L.LightningModule):
    def __init__(
        self,
        vocab_size: int,
        pad_id: int,
        mask_id: int,
        bos_id: int,
        eos_id: int,
        unk_id: int,
        scaffold_ids: frozenset[int],
        aromatic_ids: frozenset[int] | None = None,
        cond_dim: int = 5,
        use_ema: bool = True,
        use_cbi: bool = True,
        cbi_weight: float = 2.0,
        use_aromatic_cbi: bool = False,
        aromatic_cbi_weight: float = 3.0,
        normalized_aromatic_cbi: bool = False,
        normalized_cbi: bool = False,
        lr: float = 5e-4,
        warmup_steps: int = 2000,
        lr_schedule: str = "cosine",
        weight_decay: float = 0.01,
        adam_beta1: float = 0.9,
        adam_beta2: float = 0.98,
        adam_eps: float = 1e-8,
        papl_alpha: float = 3.0,
        papl_tau: float = 1.0,
        ema_decay: float = 0.9999,
        hidden_size: int = 768,
        num_layers: int = 12,
        num_heads: int = 12,
        intermediate: int = 3072,
        max_position_embeddings: int = 256,
        hidden_dropout_prob: float = 0.1,
        attention_probs_dropout_prob: float = 0.1,
        layer_norm_eps: float = 1e-12,
        initializer_range: float = 0.02,
        position_embedding_type: str = "absolute",
        gradient_checkpointing: bool = False,
        drop_cond_prob: float = 0.1,
        corruption_level_conditioning: bool = False,
        refinement_loss_weight: float = 0.0,
        refinement_warmup_steps: int = 0,
        refinement_corruption_min: float = 0.05,
        refinement_corruption_max: float = 0.50,
        refinement_clean_weight: float = 0.20,
        refinement_mask_fraction: float = 0.15,
        refinement_corruption_mode: str = "uniform",
        trajectory_length_low: float = 30.0,
        trajectory_length_high: float = 38.0,
        trajectory_temperature_start: float = 1.20,
        trajectory_temperature_end: float = 0.15,
        trajectory_temperature_power: float = 1.50,
        trajectory_remask_power: float = 1.35,
        trajectory_gumbel_scale: float = 0.65,
        trajectory_temperature_start_short: float = 1.80,
        trajectory_temperature_end_short: float = 0.35,
        trajectory_temperature_power_short: float = 1.25,
        trajectory_remask_power_short: float = 0.80,
        trajectory_gumbel_scale_short: float = 1.35,
        trajectory_confidence_temperature: float = 0.0,
        trajectory_confidence_length_adaptive: bool = False,
        trajectory_confidence_length_low: float = 28.0,
        trajectory_confidence_length_high: float = 34.0,
        trajectory_confidence_temperature_short: float = 1.0,
        trajectory_rollout_steps: int = 2,
        trajectory_rollout_decay: float = 0.50,
        training_objective_mode: str = "mask_refine",
        fragment_span_min: int = 1,
        fragment_span_max: int = 24,
        fragment_span_continue_probability: float = 0.78,
        fragment_internal_probability: float = 0.67,
    ):
        super().__init__()
        self.save_hyperparameters(ignore=["scaffold_ids", "aromatic_ids"])
        self.architecture_type = "csdnet"

        self.pad_id = pad_id
        self.mask_id = mask_id
        self.bos_id = bos_id
        self.eos_id = eos_id
        self.unk_id = unk_id
        self.lr = lr
        self.warmup = warmup_steps
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
        self.alpha = papl_alpha
        self.tau = papl_tau
        self.ema_decay = ema_decay
        self.hidden_size = hidden_size
        self.num_layers = num_layers
        self.num_heads = num_heads
        self.intermediate = intermediate
        self.use_ema = use_ema
        self.use_cbi = use_cbi
        self.cbi_w = cbi_weight
        self.use_aromatic_cbi = use_aromatic_cbi
        self.aromatic_cbi_w = aromatic_cbi_weight
        self.normalized_aromatic_cbi = normalized_aromatic_cbi
        self.normalized_cbi = normalized_cbi
        self.drop_cond_prob = drop_cond_prob
        self.corruption_level_conditioning = bool(corruption_level_conditioning)
        self.refinement_loss_weight = float(refinement_loss_weight)
        self.refinement_warmup_steps = int(refinement_warmup_steps)
        self.refinement_corruption_min = float(refinement_corruption_min)
        self.refinement_corruption_max = float(refinement_corruption_max)
        self.refinement_clean_weight = float(refinement_clean_weight)
        self.refinement_mask_fraction = float(refinement_mask_fraction)
        self.refinement_corruption_mode = str(refinement_corruption_mode)
        self.trajectory_length_low = float(trajectory_length_low)
        self.trajectory_length_high = float(trajectory_length_high)
        self.trajectory_temperature_start = float(trajectory_temperature_start)
        self.trajectory_temperature_end = float(trajectory_temperature_end)
        self.trajectory_temperature_power = float(trajectory_temperature_power)
        self.trajectory_remask_power = float(trajectory_remask_power)
        self.trajectory_gumbel_scale = float(trajectory_gumbel_scale)
        self.trajectory_temperature_start_short = float(
            trajectory_temperature_start_short
        )
        self.trajectory_temperature_end_short = float(
            trajectory_temperature_end_short
        )
        self.trajectory_temperature_power_short = float(
            trajectory_temperature_power_short
        )
        self.trajectory_remask_power_short = float(
            trajectory_remask_power_short
        )
        self.trajectory_gumbel_scale_short = float(
            trajectory_gumbel_scale_short
        )
        self.trajectory_confidence_temperature = float(
            trajectory_confidence_temperature
        )
        self.trajectory_confidence_length_adaptive = bool(
            trajectory_confidence_length_adaptive
        )
        self.trajectory_confidence_length_low = float(
            trajectory_confidence_length_low
        )
        self.trajectory_confidence_length_high = float(
            trajectory_confidence_length_high
        )
        self.trajectory_confidence_temperature_short = float(
            trajectory_confidence_temperature_short
        )
        self.trajectory_rollout_steps = int(trajectory_rollout_steps)
        self.trajectory_rollout_decay = float(trajectory_rollout_decay)
        self.training_objective_mode = str(training_objective_mode)
        self.fragment_span_min = int(fragment_span_min)
        self.fragment_span_max = int(fragment_span_max)
        self.fragment_span_continue_probability = float(
            fragment_span_continue_probability
        )
        self.fragment_internal_probability = float(
            fragment_internal_probability
        )
        if self.cbi_w <= 0 or self.aromatic_cbi_w <= 0:
            raise ValueError("CBI weights must be strictly positive")
        if self.normalized_aromatic_cbi and not self.use_aromatic_cbi:
            raise ValueError(
                "normalized_aromatic_cbi requires use_aromatic_cbi=True"
            )
        if self.normalized_cbi and self.normalized_aromatic_cbi:
            raise ValueError(
                "normalized_cbi and normalized_aromatic_cbi are mutually exclusive"
            )
        if self.normalized_cbi and not (self.use_cbi or self.use_aromatic_cbi):
            raise ValueError("normalized_cbi requires at least one CBI mode")
        if not 0.0 <= self.refinement_loss_weight <= 1.0:
            raise ValueError("refinement_loss_weight must be in [0, 1]")
        if not 0.0 <= self.refinement_corruption_min <= self.refinement_corruption_max <= 1.0:
            raise ValueError("refinement corruption bounds must satisfy 0 <= min <= max <= 1")
        if self.refinement_clean_weight < 0.0:
            raise ValueError("refinement_clean_weight must be non-negative")
        if not 0.0 <= self.refinement_mask_fraction <= 1.0:
            raise ValueError("refinement_mask_fraction must be in [0, 1]")
        if self.refinement_corruption_mode not in {"uniform", "trajectory"}:
            raise ValueError(
                "refinement_corruption_mode must be 'uniform' or 'trajectory'"
            )
        if self.trajectory_length_high <= self.trajectory_length_low:
            raise ValueError("trajectory_length_high must exceed trajectory_length_low")
        if (
            self.trajectory_confidence_length_adaptive
            and self.trajectory_confidence_length_high
            <= self.trajectory_confidence_length_low
        ):
            raise ValueError(
                "trajectory_confidence_length_high must exceed "
                "trajectory_confidence_length_low"
            )
        if self.trajectory_confidence_temperature < 0.0:
            raise ValueError("trajectory_confidence_temperature must be non-negative")
        if self.trajectory_confidence_temperature_short <= 0.0:
            raise ValueError(
                "trajectory_confidence_temperature_short must be positive"
            )
        if min(
            self.trajectory_temperature_start,
            self.trajectory_temperature_end,
            self.trajectory_temperature_power,
            self.trajectory_remask_power,
            self.trajectory_gumbel_scale,
            self.trajectory_temperature_start_short,
            self.trajectory_temperature_end_short,
            self.trajectory_temperature_power_short,
            self.trajectory_remask_power_short,
            self.trajectory_gumbel_scale_short,
        ) <= 0.0:
            raise ValueError("trajectory temperatures and powers must be positive")
        if self.trajectory_rollout_steps <= 0:
            raise ValueError("trajectory_rollout_steps must be positive")
        if not 0.0 < self.trajectory_rollout_decay < 1.0:
            raise ValueError("trajectory_rollout_decay must be in (0, 1)")
        if self.training_objective_mode not in {
            "mask_refine",
            "three_way_equal",
        }:
            raise ValueError(
                "training_objective_mode must be 'mask_refine' or "
                "'three_way_equal'"
            )
        if not 1 <= self.fragment_span_min <= self.fragment_span_max:
            raise ValueError(
                "fragment span bounds must satisfy 1 <= min <= max"
            )
        if not 0.0 < self.fragment_span_continue_probability < 1.0:
            raise ValueError(
                "fragment_span_continue_probability must be in (0, 1)"
            )
        if not 0.0 <= self.fragment_internal_probability <= 1.0:
            raise ValueError(
                "fragment_internal_probability must be in [0, 1]"
            )
        if self.refinement_loss_weight > 0.0 and not self.corruption_level_conditioning:
            raise ValueError(
                "refinement supervision requires corruption_level_conditioning=True"
            )

        sc_tensor = torch.zeros(vocab_size, dtype=torch.bool)
        for sid in scaffold_ids:
            sc_tensor[sid] = True
        self.register_buffer("_scaffold_buf", sc_tensor)

        aromatic_ids = aromatic_ids or frozenset()
        ar_tensor = torch.zeros(vocab_size, dtype=torch.bool)
        for aid in aromatic_ids:
            ar_tensor[aid] = True
        self.register_buffer("_aromatic_buf", ar_tensor)

        refinement_tokens = torch.ones(vocab_size, dtype=torch.bool)
        for special_id in (pad_id, mask_id, bos_id, eos_id, unk_id):
            if 0 <= special_id < vocab_size:
                refinement_tokens[special_id] = False
        self.register_buffer("_refinement_token_buf", refinement_tokens)

        self.backbone = CSDNetBackbone(
            vocab_size=vocab_size,
            cond_dim=cond_dim,
            hidden_size=hidden_size,
            num_layers=num_layers,
            num_heads=num_heads,
            intermediate=intermediate,
            pad_token_id=pad_id,
            mask_token_id=mask_id,
            max_position_embeddings=max_position_embeddings,
            hidden_dropout_prob=hidden_dropout_prob,
            attention_probs_dropout_prob=attention_probs_dropout_prob,
            layer_norm_eps=layer_norm_eps,
            initializer_range=initializer_range,
            position_embedding_type=position_embedding_type,
            gradient_checkpointing=gradient_checkpointing,
            corruption_level_conditioning=self.corruption_level_conditioning,
        )
        self.mdlm = build_mdlm(vocab_size, mask_id)

        if self.use_ema:
            self.ema = SimpleEMA(self.backbone, decay=self.ema_decay)
        else:
            self.ema = None

    def _logits(
        self,
        x,
        amask,
        cond=None,
        drop_cond=False,
        corruption_level=None,
    ):
        return self.backbone(
            x,
            attention_mask=amask,
            cond=cond,
            drop_cond=drop_cond,
            corruption_level=corruption_level,
        )

    def _apply_chemical_weighting(self, weights, x0, valid, aromatic_mask=None):
        cbi_multiplier = torch.ones_like(weights)
        if self.use_cbi:
            is_sc = self._scaffold_buf[x0]
            cbi_multiplier = torch.where(
                is_sc,
                torch.full_like(weights, self.cbi_w),
                cbi_multiplier,
            )
        if self.use_aromatic_cbi:
            if aromatic_mask is not None:
                is_ar = aromatic_mask.to(device=x0.device, dtype=torch.bool)
            else:
                is_ar = self._aromatic_buf[x0]
            aromatic_multiplier = torch.full_like(weights, self.aromatic_cbi_w)
            combined_multiplier = torch.where(
                is_ar,
                torch.maximum(cbi_multiplier, aromatic_multiplier),
                cbi_multiplier,
            )
        else:
            combined_multiplier = cbi_multiplier

        if self.normalized_cbi:
            return redistribute_priority_with_fixed_mass(
                weights,
                combined_multiplier,
                valid,
            )

        weights = weights * cbi_multiplier
        if not self.use_aromatic_cbi:
            return weights

        aromatic_relative_multiplier = combined_multiplier / cbi_multiplier
        if self.normalized_aromatic_cbi:
            return redistribute_priority_with_fixed_mass(
                weights,
                aromatic_relative_multiplier,
                valid,
            )
        return weights * aromatic_relative_multiplier

    def _loss(self, logits, x0, xt, t, amask, aromatic_mask=None):
        B, L, _ = logits.shape
        t_ = t.view(B, 1).expand(B, L)

        special = (x0 == self.bos_id) | (x0 == self.eos_id) | (x0 == self.pad_id)
        if self.unk_id != -1:
            special = special | (x0 == self.unk_id)
        valid = (xt == self.mask_id) & ~special & amask.bool()

        lp = F.log_softmax(logits, dim=-1)
        tl = lp.gather(2, x0.unsqueeze(2)).squeeze(2)
        w = 1.0 / (t_ + 1e-5)

        with torch.no_grad():
            sc = tl.masked_fill(~valid, -1e9) / (self.tau + 1e-8)
            wp = torch.softmax(sc, dim=-1)
            n_v = valid.sum(-1, keepdim=True).float().clamp(min=1)
            wp = wp * n_v
        w = w * (1.0 + self.alpha * wp)

        w = self._apply_chemical_weighting(
            w,
            x0,
            valid,
            aromatic_mask=aromatic_mask,
        )

        return (-tl * w * valid.float()).sum() / valid.sum().clamp(min=1)

    def _trajectory_temperatures(self, valid, corruption_ratios):
        mix = self._trajectory_length_mix(valid)

        def interpolate(short_value, long_value):
            return float(short_value) + (
                float(long_value) - float(short_value)
            ) * mix

        temperature_start = interpolate(
            self.trajectory_temperature_start_short,
            self.trajectory_temperature_start,
        )
        temperature_end = interpolate(
            self.trajectory_temperature_end_short,
            self.trajectory_temperature_end,
        )
        temperature_power = interpolate(
            self.trajectory_temperature_power_short,
            self.trajectory_temperature_power,
        )
        remask_power = self._trajectory_remask_powers(valid)

        remask_ratio = corruption_ratios.clamp(1e-6, 1.0)
        cosine_value = remask_ratio.pow(1.0 / remask_power)
        progress = (2.0 / np.pi) * torch.acos(cosine_value.clamp(0.0, 1.0))
        return temperature_end + (temperature_start - temperature_end) * (
            1.0 - progress
        ).pow(temperature_power)

    def _trajectory_length_mix(self, valid):
        lengths = valid.sum(dim=1).float() + 2.0
        mix = (
            (lengths - self.trajectory_length_low)
            / (self.trajectory_length_high - self.trajectory_length_low)
        ).clamp(0.0, 1.0)
        return mix * mix * (3.0 - 2.0 * mix)

    def _trajectory_gumbel_scales(self, valid):
        mix = self._trajectory_length_mix(valid)
        return self.trajectory_gumbel_scale_short + (
            self.trajectory_gumbel_scale - self.trajectory_gumbel_scale_short
        ) * mix

    def _trajectory_remask_powers(self, valid):
        mix = self._trajectory_length_mix(valid)
        return self.trajectory_remask_power_short + (
            self.trajectory_remask_power - self.trajectory_remask_power_short
        ) * mix

    def _sample_trajectory_corruption_ratios(self, valid):
        powers = self._trajectory_remask_powers(valid)
        low = torch.full_like(powers, self.refinement_corruption_min)
        high = torch.full_like(powers, self.refinement_corruption_max)

        def ratio_to_progress(ratios):
            cosine_value = ratios.clamp(0.0, 1.0).pow(1.0 / powers)
            return (2.0 / np.pi) * torch.acos(cosine_value.clamp(0.0, 1.0))

        progress_start = ratio_to_progress(high)
        progress_end = ratio_to_progress(low)
        progress = progress_start + torch.rand_like(powers) * (
            progress_end - progress_start
        )
        cosine_value = torch.cos(progress * np.pi * 0.5)
        return cosine_value.clamp(0.0, 1.0).pow(powers)

    def _trajectory_confidence_temperatures(self, valid, sampling_temperatures):
        if self.trajectory_confidence_length_adaptive:
            lengths = valid.sum(dim=1).float() + 2.0
            mix = (
                (lengths - self.trajectory_confidence_length_low)
                / (
                    self.trajectory_confidence_length_high
                    - self.trajectory_confidence_length_low
                )
            ).clamp(0.0, 1.0)
            mix = mix * mix * (3.0 - 2.0 * mix)
            short = self.trajectory_confidence_temperature_short
            return short + (sampling_temperatures - short) * mix
        if self.trajectory_confidence_temperature > 0.0:
            return torch.full_like(
                sampling_temperatures,
                self.trajectory_confidence_temperature,
            )
        return sampling_temperatures

    def _sample_refinement_corruption(
        self,
        x0,
        amask,
        cond=None,
        drop_cond=False,
    ):
        special = (x0 == self.bos_id) | (x0 == self.eos_id) | (x0 == self.pad_id)
        if self.unk_id != -1:
            special = special | (x0 == self.unk_id)
        valid = amask.bool() & ~special

        batch_size = x0.size(0)
        low = self.refinement_corruption_min
        high = self.refinement_corruption_max
        if self.refinement_corruption_mode == "trajectory":
            ratios = self._sample_trajectory_corruption_ratios(valid)
            corrupt = torch.zeros_like(valid)
            for row in range(batch_size):
                choices = torch.nonzero(valid[row], as_tuple=False).flatten()
                if choices.numel() == 0:
                    continue
                count = max(1, int(choices.numel() * ratios[row].item()))
                selected = choices[
                    torch.randperm(choices.numel(), device=x0.device)[:count]
                ]
                corrupt[row, selected] = True
        else:
            ratios = torch.rand(batch_size, device=x0.device) * (high - low) + low
            corrupt = (torch.rand_like(x0, dtype=torch.float) < ratios[:, None]) & valid
            for row in range(batch_size):
                if valid[row].any() and not corrupt[row].any():
                    choices = torch.nonzero(valid[row], as_tuple=False).flatten()
                    selected = choices[
                        torch.randint(choices.numel(), (1,), device=x0.device)
                    ]
                    corrupt[row, selected] = True

        observed_ratio = (
            corrupt.sum(dim=1).float()
            / valid.sum(dim=1).float().clamp(min=1.0)
        )
        if self.refinement_corruption_mode == "trajectory":
            draft = torch.where(
                corrupt,
                torch.full_like(x0, self.mask_id),
                x0,
            )
            was_training = self.backbone.training
            self.backbone.eval()
            try:
                with torch.no_grad():
                    rollout_scores = torch.full_like(
                        draft,
                        -torch.inf,
                        dtype=torch.float,
                    )
                    current_mask = corrupt.clone()
                    for rollout_step in range(self.trajectory_rollout_steps):
                        if not current_mask.any():
                            break
                        current_ratio = (
                            current_mask.sum(dim=1).float()
                            / valid.sum(dim=1).float().clamp(min=1.0)
                        )
                        proposal_logits = self._logits(
                            draft,
                            amask,
                            cond=cond,
                            drop_cond=drop_cond,
                            corruption_level=current_ratio,
                        )
                        illegal = ~self._refinement_token_buf
                        proposal_logits = proposal_logits.masked_fill(
                            illegal.view(1, 1, -1),
                            -torch.finfo(proposal_logits.dtype).max,
                        )
                        temperatures = self._trajectory_temperatures(
                            valid,
                            current_ratio,
                        )
                        sample_logits = proposal_logits / temperatures[:, None, None]
                        sampled = torch.distributions.Categorical(
                            logits=sample_logits
                        ).sample()
                        confidence_temperatures = (
                            self._trajectory_confidence_temperatures(
                                valid,
                                temperatures,
                            )
                        )
                        confidence_logits = (
                            proposal_logits
                            / confidence_temperatures[:, None, None]
                        )
                        sampled_scores = F.log_softmax(
                            confidence_logits,
                            dim=-1,
                        ).gather(2, sampled.unsqueeze(2)).squeeze(2)
                        draft.masked_scatter_(current_mask, sampled[current_mask])
                        rollout_scores.masked_scatter_(
                            current_mask,
                            sampled_scores[current_mask].float(),
                        )

                        next_mask = torch.zeros_like(current_mask)
                        gumbel_scales = self._trajectory_gumbel_scales(valid)
                        next_ratios = current_ratio * self.trajectory_rollout_decay
                        for row in range(batch_size):
                            candidates = torch.nonzero(
                                corrupt[row],
                                as_tuple=False,
                            ).flatten()
                            keep = int(
                                valid[row].sum().item()
                                * next_ratios[row].item()
                            )
                            if keep <= 0:
                                continue
                            keep = min(keep, candidates.numel())
                            scores = rollout_scores[row, candidates]
                            uniform = torch.rand_like(scores).clamp_(1e-6, 1.0 - 1e-6)
                            gumbel = -torch.log(-torch.log(uniform))
                            ranking = scores + (
                                gumbel
                                * gumbel_scales[row]
                                * next_ratios[row]
                            )
                            selected = candidates[torch.argsort(ranking)[:keep]]
                            next_mask[row, selected] = True
                        draft.masked_fill_(next_mask, self.mask_id)
                        rollout_scores.masked_fill_(next_mask, -torch.inf)
                        current_mask = next_mask
            finally:
                self.backbone.train(was_training)
            replacements = draft
        else:
            legal_ids = torch.nonzero(
                self._refinement_token_buf,
                as_tuple=False,
            ).flatten()
            sampled_idx = torch.randint(
                legal_ids.numel(),
                x0.shape,
                device=x0.device,
            )
            replacements = legal_ids[sampled_idx]
            same = replacements.eq(x0) & corrupt
            replacements[same] = legal_ids[
                (sampled_idx[same] + 1) % legal_ids.numel()
            ]

        if self.refinement_corruption_mode != "trajectory":
            use_mask = (
                torch.rand_like(x0, dtype=torch.float) < self.refinement_mask_fraction
            ) & corrupt
            replacements = torch.where(
                use_mask,
                torch.full_like(replacements, self.mask_id),
                replacements,
            )
        xt = torch.where(corrupt, replacements, x0)
        actual_corrupt = valid & xt.ne(x0)
        for row in range(batch_size):
            if valid[row].any() and not actual_corrupt[row].any():
                choices = torch.nonzero(valid[row], as_tuple=False).flatten()
                selected = choices[torch.randint(choices.numel(), (1,), device=x0.device)]
                xt[row, selected] = self.mask_id
                actual_corrupt[row, selected] = True
        observed_ratio = (
            actual_corrupt.sum(dim=1).float()
            / valid.sum(dim=1).float().clamp(min=1.0)
        )
        if self.refinement_corruption_mode == "trajectory":
            observed_ratio = (
                xt.eq(self.mask_id).logical_and(valid).sum(dim=1).float()
                / valid.sum(dim=1).float().clamp(min=1.0)
            )
        return xt, actual_corrupt, valid, observed_ratio

    def _refinement_loss(
        self,
        logits,
        x0,
        corrupt,
        valid,
        aromatic_mask=None,
    ):
        token_loss = F.cross_entropy(
            logits.transpose(1, 2),
            x0,
            reduction="none",
        )
        weights = torch.where(
            corrupt,
            torch.ones_like(token_loss),
            torch.full_like(token_loss, self.refinement_clean_weight),
        )
        weights = self._apply_chemical_weighting(
            weights,
            x0,
            valid,
            aromatic_mask=aromatic_mask,
        )
        weighted_valid = weights * valid.float()
        return (token_loss * weighted_valid).sum() / weighted_valid.sum().clamp(min=1.0)

    def _sample_fragment_span_corruption(self, x0, amask):
        """Mask one contiguous linker/extension span while preserving context."""
        special = (
            (x0 == self.bos_id)
            | (x0 == self.eos_id)
            | (x0 == self.pad_id)
        )
        if self.unk_id != -1:
            special = special | (x0 == self.unk_id)
        valid = amask.bool() & ~special
        corrupt = torch.zeros_like(valid)

        for row in range(x0.size(0)):
            positions = torch.nonzero(valid[row], as_tuple=False).flatten()
            body_length = int(positions.numel())
            if body_length == 0:
                continue

            # Keep at least one visible body token whenever possible. Internal
            # spans additionally retain context on both sides of the gap.
            maximum = min(
                self.fragment_span_max,
                max(1, body_length - 1),
            )
            span_length = min(self.fragment_span_min, maximum)
            while (
                span_length < maximum
                and random.random()
                < self.fragment_span_continue_probability
            ):
                span_length += 1

            can_be_internal = body_length - span_length >= 2
            use_internal = (
                can_be_internal
                and random.random() < self.fragment_internal_probability
            )
            if use_internal:
                start = random.randint(1, body_length - span_length - 1)
            elif body_length > span_length:
                # Edge spans explicitly train motif extension and decoration;
                # random choice keeps prefix and suffix completion symmetric.
                start = 0 if random.random() < 0.5 else body_length - span_length
            else:
                start = 0
            selected = positions[start : start + span_length]
            corrupt[row, selected] = True

        xt = x0.masked_fill(corrupt, self.mask_id)
        level = (
            corrupt.sum(dim=1).float()
            / valid.sum(dim=1).float().clamp(min=1.0)
        )
        return xt, corrupt, level

    def _fragment_infill_loss(
        self,
        logits,
        x0,
        corrupt,
        aromatic_mask=None,
    ):
        """Cross-entropy only on the conditionally hidden fragment span."""
        token_loss = F.cross_entropy(
            logits.transpose(1, 2),
            x0,
            reduction="none",
        )
        weights = torch.ones_like(token_loss)
        weights = self._apply_chemical_weighting(
            weights,
            x0,
            corrupt,
            aromatic_mask=aromatic_mask,
        )
        weighted = weights * corrupt.float()
        return (token_loss * weighted).sum() / weighted.sum().clamp(min=1.0)

    def _three_way_equal_step(
        self,
        ids,
        amsk,
        cond,
        aromatic_mask,
        drop_cond,
    ):
        """Equal mask/refine/fragment objective independent of row rounding."""
        batch_size = ids.size(0)
        if batch_size < 3:
            raise ValueError(
                "three_way_equal requires a per-device batch of at least 3"
            )

        offset = int(self.global_step) % 3
        assignment = (
            torch.arange(batch_size, device=ids.device) + offset
        ) % 3
        mask_rows = torch.nonzero(assignment == 0, as_tuple=False).flatten()
        refine_rows = torch.nonzero(assignment == 1, as_tuple=False).flatten()
        fragment_rows = torch.nonzero(assignment == 2, as_tuple=False).flatten()

        mask_ids = ids[mask_rows]
        mask_attention = amsk[mask_rows]
        t = self.mdlm.sample_time(mask_rows.numel()).to(ids.device)
        masked_xt = self.mdlm.forward_process(mask_ids, t)
        mask_active = mask_attention.bool() & mask_ids.ne(self.pad_id)
        masked_level = (
            masked_xt.eq(self.mask_id).sum(dim=1).float()
            / mask_active.sum(dim=1).float().clamp(min=1.0)
        )

        refine_cond = cond[refine_rows] if cond is not None else None
        refine_xt, refine_corrupt, refine_valid, refine_level = (
            self._sample_refinement_corruption(
                ids[refine_rows],
                amsk[refine_rows],
                cond=refine_cond,
                drop_cond=drop_cond,
            )
        )
        fragment_xt, fragment_corrupt, fragment_level = (
            self._sample_fragment_span_corruption(
                ids[fragment_rows],
                amsk[fragment_rows],
            )
        )

        xt = ids.clone()
        corruption_level = ids.new_zeros(
            batch_size,
            dtype=torch.float32,
        )
        xt[mask_rows] = masked_xt
        xt[refine_rows] = refine_xt
        xt[fragment_rows] = fragment_xt
        corruption_level[mask_rows] = masked_level
        corruption_level[refine_rows] = refine_level
        corruption_level[fragment_rows] = fragment_level

        logits = self._logits(
            xt,
            amsk,
            cond=cond,
            drop_cond=drop_cond,
            corruption_level=corruption_level,
        )

        def aromatic(rows):
            return aromatic_mask[rows] if aromatic_mask is not None else None

        mask_loss = self._loss(
            logits[mask_rows],
            mask_ids,
            masked_xt,
            t,
            mask_attention,
            aromatic_mask=aromatic(mask_rows),
        )
        refinement_loss = self._refinement_loss(
            logits[refine_rows],
            ids[refine_rows],
            refine_corrupt,
            refine_valid,
            aromatic_mask=aromatic(refine_rows),
        )
        fragment_loss = self._fragment_infill_loss(
            logits[fragment_rows],
            ids[fragment_rows],
            fragment_corrupt,
            aromatic_mask=aromatic(fragment_rows),
        )
        loss = (mask_loss + refinement_loss + fragment_loss) / 3.0
        self._last_refinement_metrics = {
            "mask_loss": mask_loss.detach(),
            "refinement_loss": refinement_loss.detach(),
            "fragment_loss": fragment_loss.detach(),
            "refinement_weight": 1.0 / 3.0,
            "refinement_corruption_rate": refine_level.mean().detach(),
            "fragment_corruption_rate": fragment_level.mean().detach(),
        }
        return loss

    def _current_refinement_weight(self):
        target = self.refinement_loss_weight
        if target <= 0.0 or self.refinement_warmup_steps <= 0:
            return target
        progress = min((int(self.global_step) + 1) / self.refinement_warmup_steps, 1.0)
        return target * progress

    def _step(self, batch, is_train=True):
        ids = batch["input_ids"]
        amsk = batch["attention_mask"]
        cond = batch.get("cond")
        if cond is not None and cond.numel() == 0:
            cond = None
        aromatic_mask = batch.get("aromatic_mask")

        drop_cond = False
        if cond is not None and is_train and random.random() < self.drop_cond_prob:
            drop_cond = True

        if self.training_objective_mode == "three_way_equal":
            return self._three_way_equal_step(
                ids,
                amsk,
                cond,
                aromatic_mask,
                drop_cond,
            )

        beta = self._current_refinement_weight()
        batch_size = ids.shape[0]
        if beta <= 0.0 or batch_size < 2:
            t = self.mdlm.sample_time(batch_size).to(ids.device)
            xt = self.mdlm.forward_process(ids, t)
            active = amsk.bool() & ids.ne(self.pad_id)
            corruption_level = (
                xt.eq(self.mask_id).sum(dim=1).float()
                / active.sum(dim=1).float().clamp(min=1.0)
            )
            logits = self._logits(
                xt,
                amsk,
                cond=cond,
                drop_cond=drop_cond,
                corruption_level=corruption_level,
            )
            loss = self._loss(
                logits,
                ids,
                xt,
                t,
                amsk,
                aromatic_mask=aromatic_mask,
            )
            self._last_refinement_metrics = {
                "mask_loss": loss.detach(),
                "refinement_loss": torch.zeros_like(loss.detach()),
                "refinement_weight": 0.0,
            }
            return loss

        n_refine = min(batch_size - 1, max(1, int(round(batch_size * beta))))
        n_mask = batch_size - n_refine

        t = self.mdlm.sample_time(n_mask).to(ids.device)
        masked_xt = self.mdlm.forward_process(ids[:n_mask], t)
        refine_cond = cond[n_mask:] if cond is not None else None
        refine_xt, corrupt, refine_valid, refine_level = (
            self._sample_refinement_corruption(
                ids[n_mask:],
                amsk[n_mask:],
                cond=refine_cond,
                drop_cond=drop_cond,
            )
        )
        xt = torch.cat([masked_xt, refine_xt], dim=0)

        masked_active = amsk[:n_mask].bool() & ids[:n_mask].ne(self.pad_id)
        masked_level = (
            masked_xt.eq(self.mask_id).sum(dim=1).float()
            / masked_active.sum(dim=1).float().clamp(min=1.0)
        )
        corruption_level = torch.cat([masked_level, refine_level], dim=0)

        logits = self._logits(
            xt,
            amsk,
            cond=cond,
            drop_cond=drop_cond,
            corruption_level=corruption_level,
        )
        mask_aromatic = aromatic_mask[:n_mask] if aromatic_mask is not None else None
        refine_aromatic = aromatic_mask[n_mask:] if aromatic_mask is not None else None
        mask_loss = self._loss(
            logits[:n_mask],
            ids[:n_mask],
            masked_xt,
            t,
            amsk[:n_mask],
            aromatic_mask=mask_aromatic,
        )
        refinement_loss = self._refinement_loss(
            logits[n_mask:],
            ids[n_mask:],
            corrupt,
            refine_valid,
            aromatic_mask=refine_aromatic,
        )
        loss = (1.0 - beta) * mask_loss + beta * refinement_loss
        self._last_refinement_metrics = {
            "mask_loss": mask_loss.detach(),
            "refinement_loss": refinement_loss.detach(),
            "refinement_weight": beta,
            "refinement_corruption_rate": refine_level.mean().detach(),
        }
        return loss

    def training_step(self, batch, _):
        loss = self._step(batch, is_train=True)
        self.log("train_loss", loss, on_step=True, prog_bar=True)
        metrics = getattr(self, "_last_refinement_metrics", None)
        if metrics is not None and self.refinement_loss_weight > 0.0:
            self.log("train_mask_loss", metrics["mask_loss"], on_step=True)
            self.log("train_refinement_loss", metrics["refinement_loss"], on_step=True)
            self.log(
                "train_refinement_weight",
                float(metrics["refinement_weight"]),
                on_step=True,
            )
            if "refinement_corruption_rate" in metrics:
                self.log(
                    "train_refinement_corruption_rate",
                    metrics["refinement_corruption_rate"],
                    on_step=True,
                )
            if "fragment_loss" in metrics:
                self.log(
                    "train_fragment_loss",
                    metrics["fragment_loss"],
                    on_step=True,
                )
            if "fragment_corruption_rate" in metrics:
                self.log(
                    "train_fragment_corruption_rate",
                    metrics["fragment_corruption_rate"],
                    on_step=True,
                )
        return loss

    def optimizer_step(self, *args, **kwargs):
        super().optimizer_step(*args, **kwargs)
        if self.ema is not None:
            self.ema.update(self.backbone)

    def validation_step(self, batch, _):
        loss = self._step(batch, is_train=False)
        self.log("val_loss", loss, prog_bar=True, sync_dist=True)

    def configure_optimizers(self):
        opt = torch.optim.AdamW(
            self.parameters(),
            lr=self.lr,
            weight_decay=self.weight_decay,
            betas=(self.adam_beta1, self.adam_beta2),
            eps=self.adam_eps,
        )

        def lr_lam(step):
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

            prog = (step - self.warmup) / max(total_steps - self.warmup, 1)
            return 0.1 + 0.9 * 0.5 * (1 + np.cos(np.pi * min(prog, 1.0)))

        sch = torch.optim.lr_scheduler.LambdaLR(opt, lr_lam)
        return {"optimizer": opt, "lr_scheduler": {"scheduler": sch, "interval": "step"}}
