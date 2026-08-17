import torch
import torch.nn as nn
from transformers import EsmConfig, EsmForMaskedLM


class CorruptionLevelEmbedding(nn.Module):
    """Zero-initialized continuous corruption-level conditioning."""

    def __init__(self, hidden_size, n_frequencies=16):
        super().__init__()
        frequencies = torch.exp(
            torch.linspace(0.0, 6.0, int(n_frequencies))
        )
        self.register_buffer("frequencies", frequencies)
        self.proj = nn.Sequential(
            nn.Linear(2 * int(n_frequencies), hidden_size),
            nn.SiLU(),
            nn.Linear(hidden_size, hidden_size),
        )
        nn.init.zeros_(self.proj[-1].weight)
        nn.init.zeros_(self.proj[-1].bias)

    def forward(self, level):
        phase = level.float().clamp(0.0, 1.0).unsqueeze(-1) * self.frequencies
        features = torch.cat([torch.sin(phase), torch.cos(phase)], dim=-1)
        return self.proj(features)


class CSDNetBackbone(nn.Module):
    """ESM-style masked-token backbone used by CSDNet."""

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
        hidden_dropout_prob=0.1,
        attention_probs_dropout_prob=0.1,
        layer_norm_eps=1e-12,
        initializer_range=0.02,
        position_embedding_type="absolute",
        gradient_checkpointing=False,
        corruption_level_conditioning=False,
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
            hidden_dropout_prob=hidden_dropout_prob,
            attention_probs_dropout_prob=attention_probs_dropout_prob,
            layer_norm_eps=layer_norm_eps,
            initializer_range=initializer_range,
            position_embedding_type=position_embedding_type,
            use_cache=False,
        )
        self.esm = EsmForMaskedLM(cfg)
        self.position_embedding_type = str(position_embedding_type)
        self.gradient_checkpointing = bool(gradient_checkpointing)
        if self.gradient_checkpointing:
            try:
                self.esm.gradient_checkpointing_enable(
                    gradient_checkpointing_kwargs={"use_reentrant": False}
                )
            except TypeError as exc:
                raise RuntimeError(
                    "CSDNet requires a transformers version that supports "
                    "non-reentrant gradient checkpointing."
                ) from exc

        self.cond_dim = cond_dim
        if cond_dim > 0:
            self.cond_proj = nn.Sequential(
                nn.Linear(cond_dim, hidden_size),
                nn.SiLU(),
                nn.Linear(hidden_size, hidden_size),
            )
            self.null_cond = nn.Parameter(torch.zeros(1, hidden_size))

        self.mask_token_id = mask_token_id
        self.corruption_level_conditioning = bool(corruption_level_conditioning)
        if self.corruption_level_conditioning:
            self.corruption_level_embedding = CorruptionLevelEmbedding(hidden_size)

    def forward(
        self,
        input_ids,
        attention_mask,
        cond=None,
        drop_cond=False,
        corruption_level=None,
    ):
        inputs_embeds = self.esm.esm.embeddings.word_embeddings(input_ids)

        if self.cond_dim > 0:
            null_emb = self.null_cond.expand(input_ids.size(0), -1)
            if cond is not None:
                projected_cond = self.cond_proj(cond)
            else:
                projected_cond = None

            if projected_cond is not None and not drop_cond:
                c_emb = projected_cond + 0.0 * null_emb
            else:
                c_emb = null_emb
                if projected_cond is not None:
                    c_emb = c_emb + 0.0 * projected_cond
            inputs_embeds = inputs_embeds + c_emb.unsqueeze(1)

        if self.corruption_level_conditioning:
            if corruption_level is None:
                active = attention_mask.bool()
                masked = input_ids.eq(self.mask_token_id) & active
                corruption_level = (
                    masked.sum(dim=1).float()
                    / active.sum(dim=1).float().clamp(min=1.0)
                )
            corruption_emb = self.corruption_level_embedding(corruption_level)
            inputs_embeds = inputs_embeds + corruption_emb.unsqueeze(1)

        out = self.esm(inputs_embeds=inputs_embeds, attention_mask=attention_mask)
        return self._attach_ddp_unused_parameter_anchors(out.logits)

    def _attach_ddp_unused_parameter_anchors(self, logits):
        anchors = []
        position_embeddings = getattr(self.esm.esm.embeddings, "position_embeddings", None)
        if position_embeddings is not None and position_embeddings.weight.requires_grad:
            anchors.append(position_embeddings.weight.sum())

        contact_head = getattr(self.esm.esm, "contact_head", None)
        if contact_head is not None:
            anchors.extend(p.sum() for p in contact_head.parameters() if p.requires_grad)

        if not anchors:
            return logits

        anchor = anchors[0]
        for item in anchors[1:]:
            anchor = anchor + item
        return logits + anchor.to(dtype=logits.dtype) * 0.0
