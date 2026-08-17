import math

import torch
import torch.nn as nn
import torch.nn.functional as F


def scalar_timestep_embedding(t, dim, max_period=10000):
    """Sinusoidal embedding for a scalar corruption level in [0, 1]."""
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


class PositiveInsertionRate(nn.Module):
    """Bounded positive rate used to rank data-dependent insertion events."""

    def __init__(
        self,
        hidden_size,
        min_rate=0.05,
        max_rate=20.0,
        initial_rate=1.0,
    ):
        super().__init__()
        if not 0.0 < min_rate < max_rate:
            raise ValueError("Insertion rates must satisfy 0 < min < max.")
        self.min_rate = float(min_rate)
        self.max_rate = float(max_rate)
        self.norm = nn.LayerNorm(hidden_size)
        self.proj = nn.Linear(hidden_size, hidden_size)
        self.out = nn.Linear(hidden_size, 1)

        fraction = (
            (float(initial_rate) - self.min_rate)
            / (self.max_rate - self.min_rate)
        )
        fraction = min(max(fraction, 1e-5), 1.0 - 1e-5)
        nn.init.normal_(self.out.weight, mean=0.0, std=1e-3)
        nn.init.constant_(self.out.bias, math.log(fraction / (1.0 - fraction)))

    def forward(self, hidden_states):
        hidden = F.gelu(self.proj(self.norm(hidden_states.float())))
        unit_rate = torch.sigmoid(self.out(hidden)).squeeze(-1)
        rate = self.min_rate + (self.max_rate - self.min_rate) * unit_rate
        return rate.to(dtype=hidden_states.dtype)


class EditScheduleNet(nn.Module):
    """
    Lightweight external insertion and edit-order scheduler.

    The molecular generator remains frozen. This module predicts a categorical
    replacement length at every proposed gap and a positive insertion rate
    whose relative magnitude defines a data-dependent edit order. Token
    unmasking is intentionally left to the original CSDNet sampler.
    """

    architecture_type = "edit_schedule_net"

    def __init__(
        self,
        vocab_size,
        pad_token_id,
        mask_token_id,
        hidden_size=256,
        num_layers=4,
        num_heads=8,
        intermediate=1024,
        max_position_embeddings=256,
        max_replacement_length=32,
        dropout=0.1,
        rate_min=0.05,
        rate_max=20.0,
        rate_initial=1.0,
    ):
        super().__init__()
        if hidden_size % num_heads != 0:
            raise ValueError("hidden_size must be divisible by num_heads.")
        if max_replacement_length < 1:
            raise ValueError("max_replacement_length must be positive.")

        self.vocab_size = int(vocab_size)
        self.pad_token_id = int(pad_token_id)
        self.mask_token_id = int(mask_token_id)
        self.hidden_size = int(hidden_size)
        self.num_layers = int(num_layers)
        self.num_heads = int(num_heads)
        self.intermediate = int(intermediate)
        self.max_position_embeddings = int(max_position_embeddings)
        self.max_replacement_length = int(max_replacement_length)
        self.dropout_probability = float(dropout)
        self.rate_min = float(rate_min)
        self.rate_max = float(rate_max)
        self.rate_initial = float(rate_initial)

        self.token_embedding = nn.Embedding(
            self.vocab_size,
            self.hidden_size,
            padding_idx=self.pad_token_id,
        )
        self.position_embedding = nn.Embedding(
            self.max_position_embeddings,
            self.hidden_size,
        )
        self.gap_embedding = nn.Embedding(2, self.hidden_size)
        self.removed_length_embedding = nn.Embedding(
            self.max_replacement_length + 1,
            self.hidden_size,
        )
        self.time_projection = nn.Sequential(
            nn.Linear(self.hidden_size, self.hidden_size),
            nn.SiLU(),
            nn.Linear(self.hidden_size, self.hidden_size),
        )
        self.input_norm = nn.LayerNorm(self.hidden_size)
        self.input_dropout = nn.Dropout(self.dropout_probability)

        layer = nn.TransformerEncoderLayer(
            d_model=self.hidden_size,
            nhead=self.num_heads,
            dim_feedforward=self.intermediate,
            dropout=self.dropout_probability,
            activation="gelu",
            batch_first=True,
            norm_first=True,
        )
        self.encoder = nn.TransformerEncoder(
            layer,
            num_layers=self.num_layers,
            norm=nn.LayerNorm(self.hidden_size),
            enable_nested_tensor=False,
        )
        self.length_head = nn.Sequential(
            nn.LayerNorm(self.hidden_size),
            nn.Linear(self.hidden_size, self.hidden_size),
            nn.GELU(),
            nn.Linear(
                self.hidden_size,
                self.max_replacement_length + 1,
            ),
        )
        self.insertion_rate_head = PositiveInsertionRate(
            hidden_size=self.hidden_size,
            min_rate=self.rate_min,
            max_rate=self.rate_max,
            initial_rate=self.rate_initial,
        )

    def forward(
        self,
        input_ids,
        attention_mask,
        gap_mask,
        removed_lengths,
        corruption_fraction=None,
    ):
        batch_size, sequence_length = input_ids.shape
        if sequence_length > self.max_position_embeddings:
            raise ValueError(
                f"Sequence length {sequence_length} exceeds scheduler maximum "
                f"{self.max_position_embeddings}."
            )
        positions = torch.arange(
            sequence_length,
            device=input_ids.device,
        ).unsqueeze(0)
        positions = positions.expand(batch_size, -1)
        gap_mask = gap_mask.bool()
        removed_lengths = removed_lengths.clamp(
            min=0,
            max=self.max_replacement_length,
        )

        hidden = (
            self.token_embedding(input_ids)
            + self.position_embedding(positions)
            + self.gap_embedding(gap_mask.long())
            + self.removed_length_embedding(removed_lengths)
        )
        if corruption_fraction is None:
            corruption_fraction = torch.zeros(
                batch_size,
                device=input_ids.device,
                dtype=torch.float32,
            )
        corruption_fraction = corruption_fraction.to(
            device=input_ids.device,
            dtype=torch.float32,
        ).reshape(-1)
        time_embedding = scalar_timestep_embedding(
            corruption_fraction,
            self.hidden_size,
        )
        hidden = hidden + self.time_projection(time_embedding).unsqueeze(1)
        hidden = self.input_dropout(self.input_norm(hidden))
        hidden = self.encoder(
            hidden,
            src_key_padding_mask=~attention_mask.bool(),
        )

        length_logits = self.length_head(hidden)
        insertion_rate = self.insertion_rate_head(hidden)
        insertion_rate = insertion_rate * gap_mask.to(insertion_rate.dtype)
        order_logits = torch.log(insertion_rate.float().clamp(min=1e-8))
        order_logits = order_logits.masked_fill(~gap_mask, -torch.inf)
        return {
            "length_logits": length_logits,
            "insertion_rate": insertion_rate,
            "order_logits": order_logits,
            "hidden_states": hidden,
        }

    def config_dict(self):
        return {
            "vocab_size": self.vocab_size,
            "pad_token_id": self.pad_token_id,
            "mask_token_id": self.mask_token_id,
            "hidden_size": self.hidden_size,
            "num_layers": self.num_layers,
            "num_heads": self.num_heads,
            "intermediate": self.intermediate,
            "max_position_embeddings": self.max_position_embeddings,
            "max_replacement_length": self.max_replacement_length,
            "dropout": self.dropout_probability,
            "rate_min": self.rate_min,
            "rate_max": self.rate_max,
            "rate_initial": self.rate_initial,
        }


def load_edit_scheduler_checkpoint(checkpoint_path, device="cpu"):
    """Load only the external scheduler weights from a Lightning checkpoint."""
    try:
        checkpoint = torch.load(
            checkpoint_path,
            map_location="cpu",
            weights_only=False,
        )
    except TypeError:
        checkpoint = torch.load(checkpoint_path, map_location="cpu")

    hyperparameters = checkpoint.get("hyper_parameters", {})
    config = dict(hyperparameters.get("scheduler_config", {}))
    if not config:
        required = {
            "vocab_size",
            "pad_token_id",
            "mask_token_id",
            "hidden_size",
            "num_layers",
            "num_heads",
            "intermediate",
            "max_position_embeddings",
            "max_replacement_length",
            "dropout",
            "rate_min",
            "rate_max",
            "rate_initial",
        }
        config = {
            key: hyperparameters[key]
            for key in required
            if key in hyperparameters
        }
    missing_config = {
        "vocab_size",
        "pad_token_id",
        "mask_token_id",
    } - set(config)
    if missing_config:
        raise ValueError(
            "Scheduler checkpoint lacks required configuration: "
            f"{sorted(missing_config)}"
        )

    model = EditScheduleNet(**config)
    state_dict = checkpoint.get("state_dict", checkpoint)
    scheduler_state = {
        key[len("scheduler."):]: value
        for key, value in state_dict.items()
        if key.startswith("scheduler.")
    }
    if not scheduler_state:
        scheduler_state = state_dict
    missing, unexpected = model.load_state_dict(scheduler_state, strict=False)
    if missing or unexpected:
        raise RuntimeError(
            "Invalid edit scheduler checkpoint. "
            f"Missing={missing}, unexpected={unexpected}"
        )
    model.to(device)
    model.eval()
    return model
