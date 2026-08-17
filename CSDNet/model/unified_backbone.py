import torch
import torch.nn as nn
from transformers import EsmConfig, EsmForMaskedLM

from CSDNet.model.backbone import CorruptionLevelEmbedding


class UnifiedCSDNetBackbone(nn.Module):
    """One molecular backbone for token, insertion, and deletion dynamics.

    Gap logits at position ``i`` describe the number of missing tokens between
    active positions ``i`` and ``i + 1``.  The final sequence position never
    represents a valid gap.  A categorical count head is intentionally used
    instead of a free positive rate: the count target is directly supervised,
    bounded, and can be applied repeatedly when a gap contains more tokens than
    ``max_gap_count``.
    """

    architecture_type = "unified_csdnet"
    is_unified = True
    is_variable_length = True

    def __init__(
        self,
        vocab_size,
        hidden_size=768,
        num_layers=12,
        num_heads=12,
        intermediate=3072,
        pad_token_id=0,
        mask_token_id=None,
        max_position_embeddings=512,
        max_gap_count=8,
        position_embedding_type="rotary",
        gradient_checkpointing=True,
        hidden_dropout_prob=0.1,
        attention_probs_dropout_prob=0.1,
        layer_norm_eps=1e-12,
        initializer_range=0.02,
    ):
        super().__init__()
        config = EsmConfig(
            vocab_size=int(vocab_size),
            hidden_size=int(hidden_size),
            num_hidden_layers=int(num_layers),
            num_attention_heads=int(num_heads),
            intermediate_size=int(intermediate),
            max_position_embeddings=int(max_position_embeddings),
            pad_token_id=int(pad_token_id),
            mask_token_id=mask_token_id,
            position_embedding_type=str(position_embedding_type),
            hidden_dropout_prob=float(hidden_dropout_prob),
            attention_probs_dropout_prob=float(attention_probs_dropout_prob),
            layer_norm_eps=float(layer_norm_eps),
            initializer_range=float(initializer_range),
            use_cache=False,
        )
        self.esm = EsmForMaskedLM(config)
        self.mask_token_id = mask_token_id
        self.max_gap_count = int(max_gap_count)
        self.position_embedding_type = str(position_embedding_type)
        self.gradient_checkpointing = bool(gradient_checkpointing)
        if self.max_gap_count < 1:
            raise ValueError("max_gap_count must be positive.")

        if self.gradient_checkpointing:
            try:
                self.esm.gradient_checkpointing_enable(
                    gradient_checkpointing_kwargs={"use_reentrant": False}
                )
            except TypeError as exc:
                raise RuntimeError(
                    "Unified CSDNet requires non-reentrant gradient "
                    "checkpointing support in transformers."
                ) from exc

        self.corruption_level_embedding = CorruptionLevelEmbedding(hidden_size)

        gap_features = 4 * int(hidden_size)
        gap_hidden = max(128, int(hidden_size) // 2)
        self.gap_head = nn.Sequential(
            nn.LayerNorm(gap_features),
            nn.Linear(gap_features, gap_hidden),
            nn.SiLU(),
            nn.Dropout(float(hidden_dropout_prob)),
            nn.Linear(gap_hidden, self.max_gap_count + 1),
        )
        self.delete_head = nn.Sequential(
            nn.LayerNorm(int(hidden_size)),
            nn.Linear(int(hidden_size), max(64, int(hidden_size) // 4)),
            nn.SiLU(),
            nn.Linear(max(64, int(hidden_size) // 4), 1),
        )
        self.confidence_head = nn.Sequential(
            nn.LayerNorm(int(hidden_size)),
            nn.Linear(int(hidden_size), max(64, int(hidden_size) // 4)),
            nn.SiLU(),
            nn.Linear(max(64, int(hidden_size) // 4), 1),
        )

        # Start from conservative edit dynamics.  The token model learns as a
        # normal MDM while the auxiliary heads acquire evidence from data.
        nn.init.zeros_(self.gap_head[-1].weight)
        nn.init.zeros_(self.gap_head[-1].bias)
        self.gap_head[-1].bias.data[0] = 2.0
        nn.init.zeros_(self.delete_head[-1].weight)
        nn.init.constant_(self.delete_head[-1].bias, -4.0)
        nn.init.zeros_(self.confidence_head[-1].weight)
        nn.init.zeros_(self.confidence_head[-1].bias)

    def forward(
        self,
        input_ids,
        attention_mask,
        corruption_level=None,
        return_aux=False,
        t=None,
        cond=None,
        drop_cond=False,
        **_unused,
    ):
        del cond, drop_cond
        if corruption_level is None and t is not None:
            if not torch.is_tensor(t):
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
            corruption_level = 1.0 - t
        if corruption_level is None:
            active = attention_mask.bool()
            if self.mask_token_id is None:
                corruption_level = torch.full(
                    (input_ids.size(0),),
                    0.5,
                    device=input_ids.device,
                    dtype=torch.float32,
                )
            else:
                masked = input_ids.eq(self.mask_token_id) & active
                corruption_level = (
                    masked.sum(dim=1).float()
                    / active.sum(dim=1).float().clamp(min=1.0)
                )
        corruption_level = corruption_level.to(
            device=input_ids.device, dtype=torch.float32
        ).reshape(-1)

        inputs_embeds = self.esm.esm.embeddings.word_embeddings(input_ids)
        level_emb = self.corruption_level_embedding(corruption_level)
        inputs_embeds = inputs_embeds + level_emb.unsqueeze(1)
        output = self.esm(
            inputs_embeds=inputs_embeds,
            attention_mask=attention_mask,
            output_hidden_states=return_aux,
        )
        logits = self._attach_ddp_unused_parameter_anchors(output.logits)
        if not return_aux:
            return logits

        hidden = output.hidden_states[-1]
        left = hidden[:, :-1]
        right = hidden[:, 1:]
        pair = torch.cat(
            [left, right, left * right, left - right], dim=-1
        )
        gap_logits = self.gap_head(pair)
        gap_logits = torch.cat(
            [
                gap_logits,
                gap_logits.new_zeros(
                    gap_logits.size(0), 1, gap_logits.size(-1)
                ),
            ],
            dim=1,
        )
        return {
            "logits": logits,
            "vocab_logits": logits,
            "gap_logits": gap_logits,
            "delete_logits": self.delete_head(hidden).squeeze(-1),
            "confidence_logits": self.confidence_head(hidden).squeeze(-1),
            "hidden_states": hidden,
            "corruption_level": corruption_level,
        }

    def _attach_ddp_unused_parameter_anchors(self, logits):
        anchors = []
        position_embeddings = getattr(
            self.esm.esm.embeddings, "position_embeddings", None
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
