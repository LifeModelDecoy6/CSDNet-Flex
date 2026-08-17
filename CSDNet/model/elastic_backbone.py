import math

import torch
import torch.nn as nn
import torch.nn.functional as F
from transformers import EsmConfig, EsmForMaskedLM


def timestep_embedding(t, dim, max_period=10000):
    """Sinusoidal embedding for continuous diffusion time in [0, 1]."""
    half = dim // 2
    frequencies = torch.exp(
        -math.log(max_period)
        * torch.arange(half, device=t.device, dtype=torch.float32)
        / max(half, 1)
    )
    angles = t.float().unsqueeze(-1) * frequencies.unsqueeze(0)
    embedding = torch.cat([torch.cos(angles), torch.sin(angles)], dim=-1)
    if dim % 2:
        embedding = F.pad(embedding, (0, 1))
    return embedding


class PositiveRateHead(nn.Module):
    """Small positive scalar head used for insertion and unmasking rates."""

    def __init__(
        self,
        hidden_size,
        min_rate=0.05,
        max_rate=20.0,
        initial_rate=1.0,
        scalar_fn="sigmoid",
        output_bias=None,
    ):
        super().__init__()
        self.min_rate = float(min_rate)
        self.max_rate = float(max_rate)
        self.scalar_fn = str(scalar_fn)
        if self.scalar_fn not in {"sigmoid", "exp", "softplus"}:
            raise ValueError(
                "scalar_fn must be 'sigmoid', 'exp', or 'softplus'."
            )
        if self.max_rate <= self.min_rate:
            raise ValueError("max_rate must exceed min_rate.")
        self.norm = nn.LayerNorm(hidden_size)
        self.proj = nn.Linear(hidden_size, hidden_size)
        self.out = nn.Linear(hidden_size, 1)

        if output_bias is not None:
            initial_bias = float(output_bias)
        elif self.scalar_fn == "sigmoid":
            fraction = (
                (float(initial_rate) - self.min_rate)
                / max(self.max_rate - self.min_rate, 1e-8)
            )
            fraction = min(max(fraction, 1e-5), 1.0 - 1e-5)
            initial_bias = math.log(fraction / (1.0 - fraction))
        elif self.scalar_fn == "exp":
            initial_bias = math.log(
                max(float(initial_rate) - self.min_rate, 1e-5)
            )
        else:
            target = max(float(initial_rate) - self.min_rate, 1e-5)
            initial_bias = math.log(math.expm1(target))
        # Keep the initial rates near one while allowing gradients to reach
        # the shared features from the first optimizer step.
        nn.init.normal_(self.out.weight, mean=0.0, std=1e-3)
        nn.init.constant_(self.out.bias, initial_bias)

    def forward(self, hidden_states):
        hidden = F.gelu(self.proj(self.norm(hidden_states.float())))
        raw_rate = self.out(hidden).squeeze(-1)
        if self.scalar_fn == "sigmoid":
            unit_rate = torch.sigmoid(raw_rate)
            rate = self.min_rate + (
                self.max_rate - self.min_rate
            ) * unit_rate
        elif self.scalar_fn == "exp":
            rate = self.min_rate + torch.exp(raw_rate)
            rate = rate.clamp(max=self.max_rate)
        else:
            rate = self.min_rate + F.softplus(raw_rate)
            rate = rate.clamp(max=self.max_rate)
        return rate.to(dtype=hidden_states.dtype)


class ElasticCSDNetBackbone(nn.Module):
    """
    Variable-length CSDNet with shared token, insertion, and order dynamics.

    The default forward path returns only token logits, matching the original
    CSDNetBackbone API used by PMO and lead optimization. Set return_aux=True
    to receive the learned insertion and unmasking rates.
    """

    architecture_type = "elastic_csdnet"
    is_elastic = True

    def __init__(
        self,
        vocab_size,
        cond_dim=5,
        hidden_size=768,
        num_layers=12,
        num_heads=12,
        intermediate=3072,
        pad_token_id=0,
        mask_token_id=None,
        max_position_embeddings=256,
        position_embedding_type="rotary",
        rate_min=0.001,
        rate_max=20.0,
        rate_initial=1.0,
        rate_parameterization="sigmoid",
        theta_rate_min=None,
        phi_rate_min=None,
        rate_output_bias=None,
        fixed_unmask_rate=1.0,
        kuma_shape_a=2.0,
        gradient_checkpointing=True,
        hidden_dropout_prob=0.1,
        attention_probs_dropout_prob=0.1,
        layer_norm_eps=1e-12,
        initializer_range=0.02,
    ):
        super().__init__()
        cfg = EsmConfig(
            vocab_size=vocab_size,
            hidden_size=hidden_size,
            num_hidden_layers=num_layers,
            num_attention_heads=num_heads,
            intermediate_size=intermediate,
            max_position_embeddings=max_position_embeddings,
            pad_token_id=pad_token_id,
            mask_token_id=mask_token_id,
            position_embedding_type=position_embedding_type,
            hidden_dropout_prob=hidden_dropout_prob,
            attention_probs_dropout_prob=attention_probs_dropout_prob,
            layer_norm_eps=layer_norm_eps,
            initializer_range=initializer_range,
            use_cache=False,
        )
        self.esm = EsmForMaskedLM(cfg)
        self.cond_dim = int(cond_dim)
        self.mask_token_id = mask_token_id
        self.position_embedding_type = position_embedding_type
        self.rate_min = float(rate_min)
        self.rate_max = float(rate_max)
        self.rate_initial = float(rate_initial)
        self.rate_parameterization = str(rate_parameterization)
        self.theta_rate_min = float(
            self.rate_min if theta_rate_min is None else theta_rate_min
        )
        self.phi_rate_min = float(
            self.rate_min if phi_rate_min is None else phi_rate_min
        )
        self.rate_output_bias = (
            None if rate_output_bias is None else float(rate_output_bias)
        )
        self.fixed_unmask_rate = (
            None
            if fixed_unmask_rate is None
            else float(fixed_unmask_rate)
        )
        if self.fixed_unmask_rate is not None and self.fixed_unmask_rate <= 0:
            raise ValueError("fixed_unmask_rate must be positive or None.")
        self.kuma_shape_a = float(kuma_shape_a)
        self.gradient_checkpointing = bool(gradient_checkpointing)
        if self.gradient_checkpointing:
            try:
                self.esm.gradient_checkpointing_enable(
                    gradient_checkpointing_kwargs={
                        "use_reentrant": False,
                    }
                )
            except TypeError as exc:
                raise RuntimeError(
                    "Elastic CSDNet requires a transformers version that "
                    "supports non-reentrant gradient checkpointing."
                ) from exc

        self.time_proj = nn.Sequential(
            nn.Linear(hidden_size, hidden_size),
            nn.SiLU(),
            nn.Linear(hidden_size, hidden_size),
        )

        if self.cond_dim > 0:
            self.cond_proj = nn.Sequential(
                nn.Linear(self.cond_dim, hidden_size),
                nn.SiLU(),
                nn.Linear(hidden_size, hidden_size),
            )
            self.null_cond = nn.Parameter(torch.zeros(1, hidden_size))

        common_rate_kwargs = {
            "hidden_size": hidden_size,
            "max_rate": self.rate_max,
            "initial_rate": self.rate_initial,
            "scalar_fn": self.rate_parameterization,
            "output_bias": self.rate_output_bias,
        }
        self.theta_insertion_head = PositiveRateHead(
            min_rate=self.theta_rate_min,
            **common_rate_kwargs,
        )
        self.phi_insertion_head = PositiveRateHead(
            min_rate=self.phi_rate_min,
            **common_rate_kwargs,
        )
        if self.fixed_unmask_rate is None:
            self.theta_unmask_head = PositiveRateHead(
                min_rate=self.theta_rate_min,
                **common_rate_kwargs,
            )
            self.phi_unmask_head = PositiveRateHead(
                min_rate=self.phi_rate_min,
                **common_rate_kwargs,
            )

    def _infer_time(self, input_ids, attention_mask):
        if self.mask_token_id is None:
            return torch.full(
                (input_ids.size(0),),
                0.5,
                device=input_ids.device,
                dtype=torch.float32,
            )
        active = attention_mask.bool()
        denominator = active.sum(dim=1).clamp(min=1)
        masked = ((input_ids == self.mask_token_id) & active).sum(dim=1)
        # Clean data is at t=1 and a fully masked canvas is near t=0.
        return (1.0 - masked.float() / denominator.float()).clamp(1e-4, 1.0)

    def _condition_inputs(self, input_ids, t, cond, drop_cond):
        inputs_embeds = self.esm.esm.embeddings.word_embeddings(input_ids)
        time_emb = timestep_embedding(t, inputs_embeds.size(-1))
        inputs_embeds = inputs_embeds + self.time_proj(time_emb).unsqueeze(1)

        if self.cond_dim > 0:
            null_emb = self.null_cond.expand(input_ids.size(0), -1)
            projected_cond = self.cond_proj(cond) if cond is not None else None
            if projected_cond is not None and not drop_cond:
                cond_emb = projected_cond + 0.0 * null_emb
            else:
                cond_emb = null_emb
                if projected_cond is not None:
                    cond_emb = cond_emb + 0.0 * projected_cond
            inputs_embeds = inputs_embeds + cond_emb.unsqueeze(1)
        return inputs_embeds

    def forward(
        self,
        input_ids,
        attention_mask,
        cond=None,
        drop_cond=False,
        t=None,
        return_aux=False,
        rate_family="theta",
        compute_logits=True,
    ):
        if t is None:
            t = self._infer_time(input_ids, attention_mask)
        elif not torch.is_tensor(t):
            t = torch.full(
                (input_ids.size(0),),
                float(t),
                device=input_ids.device,
                dtype=torch.float32,
            )
        else:
            t = t.to(device=input_ids.device, dtype=torch.float32).reshape(-1)
            if t.numel() == 1 and input_ids.size(0) > 1:
                t = t.expand(input_ids.size(0))

        inputs_embeds = self._condition_inputs(input_ids, t, cond, drop_cond)
        if compute_logits:
            output = self.esm(
                inputs_embeds=inputs_embeds,
                attention_mask=attention_mask,
                output_hidden_states=return_aux,
            )
            logits = self._attach_ddp_unused_parameter_anchors(output.logits)
            if not return_aux:
                return logits
            hidden_states = output.hidden_states[-1]
        else:
            if not return_aux:
                raise ValueError(
                    "compute_logits=False requires return_aux=True."
                )
            output = self.esm.esm(
                inputs_embeds=inputs_embeds,
                attention_mask=attention_mask,
            )
            logits = None
            hidden_states = output.last_hidden_state
        if rate_family == "theta":
            insertion_rate = self.theta_insertion_head(hidden_states)
            unmask_head = getattr(self, "theta_unmask_head", None)
        elif rate_family == "phi":
            insertion_rate = self.phi_insertion_head(hidden_states)
            unmask_head = getattr(self, "phi_unmask_head", None)
        else:
            raise ValueError(f"Unknown rate family: {rate_family}")

        active = attention_mask.to(dtype=hidden_states.dtype)
        insertion_rate = insertion_rate * active
        if unmask_head is None:
            unmask_rate = torch.full_like(
                insertion_rate,
                self.fixed_unmask_rate,
            )
        else:
            unmask_rate = unmask_head(hidden_states)
        unmask_rate = unmask_rate * active
        result = {
            "b_ins": insertion_rate,
            "b_unmask": unmask_rate,
            "hidden_states": hidden_states,
            "t": t,
        }
        if logits is not None:
            result["logits"] = logits
            result["vocab_logits"] = logits
        return result

    def _attach_ddp_unused_parameter_anchors(self, logits):
        anchors = []
        position_embeddings = getattr(
            self.esm.esm.embeddings,
            "position_embeddings",
            None,
        )
        if position_embeddings is not None and position_embeddings.weight.requires_grad:
            anchors.append(position_embeddings.weight.sum())

        contact_head = getattr(self.esm.esm, "contact_head", None)
        if contact_head is not None:
            anchors.extend(
                parameter.sum()
                for parameter in contact_head.parameters()
                if parameter.requires_grad
            )
        if not anchors:
            return logits

        anchor = anchors[0]
        for item in anchors[1:]:
            anchor = anchor + item
        return logits + anchor.to(dtype=logits.dtype) * 0.0
