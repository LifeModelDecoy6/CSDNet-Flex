from __future__ import annotations

import torch


def redistribute_priority_with_fixed_mass(
    base_weights: torch.Tensor,
    relative_multiplier: torch.Tensor,
    valid_mask: torch.Tensor,
    eps: float = 1e-12,
) -> torch.Tensor:
    """Redistribute token weights while preserving each sample's valid weight mass.

    For sample b, the returned weights satisfy

        sum_i valid[b, i] * output[b, i]
        = sum_i valid[b, i] * base_weights[b, i].

    ``relative_multiplier`` changes only the within-sample allocation. This is
    useful for emphasizing aromatic positions without giving aromatic-rich
    molecules a larger total gradient budget.
    """
    if base_weights.shape != relative_multiplier.shape:
        raise ValueError("base_weights and relative_multiplier must have the same shape")
    if base_weights.shape != valid_mask.shape:
        raise ValueError("base_weights and valid_mask must have the same shape")

    valid = valid_mask.to(dtype=base_weights.dtype)
    boosted = base_weights * relative_multiplier
    base_mass = (base_weights * valid).sum(dim=-1, keepdim=True)
    boosted_mass = (boosted * valid).sum(dim=-1, keepdim=True)
    scale = torch.where(
        boosted_mass > eps,
        base_mass / boosted_mass.clamp_min(eps),
        torch.ones_like(boosted_mass),
    )
    return boosted * scale
