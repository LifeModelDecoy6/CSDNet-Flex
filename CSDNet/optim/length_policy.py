"""Budget-preserving length proposals shared by constrained samplers."""

from __future__ import annotations

import random
from dataclasses import dataclass
from math import floor
from typing import Mapping


@dataclass(frozen=True)
class LengthProposal:
    """One proposal arm and a stratified draw within its length support."""

    arm: str
    length_space: str
    support_min: int
    support_max: int
    quantile: float


def _largest_remainder_counts(
    total: int,
    fractions: Mapping[str, float],
) -> dict[str, int]:
    if total < 0:
        raise ValueError("total must be non-negative")
    if not fractions:
        raise ValueError("at least one allocation fraction is required")
    if any(value < 0.0 for value in fractions.values()):
        raise ValueError("allocation fractions must be non-negative")
    weight_sum = float(sum(fractions.values()))
    if weight_sum <= 0.0:
        raise ValueError("allocation fractions must have positive mass")

    exact = {
        name: total * float(weight) / weight_sum
        for name, weight in fractions.items()
    }
    counts = {name: floor(value) for name, value in exact.items()}
    remainder = total - sum(counts.values())
    order = sorted(
        fractions,
        key=lambda name: (exact[name] - counts[name], name),
        reverse=True,
    )
    for name in order[:remainder]:
        counts[name] += 1
    return counts


def _stratified_quantiles(count: int, rng: random.Random) -> list[float]:
    if count <= 0:
        return []
    values = [(index + rng.random()) / count for index in range(count)]
    rng.shuffle(values)
    return values


class ProtectedLengthAllocator:
    """Mix a global total-length prior with protected local and explore arms.

    The allocator is deliberately non-adaptive. It consumes no target identity,
    molecular property, or benchmark outcome and preserves the requested raw
    proposal budget exactly.
    """

    def __init__(
        self,
        *,
        total_quality_range: tuple[int, int] = (32, 47),
        global_fraction: float = 0.40,
        local_fraction: float = 0.40,
        explore_fraction: float = 0.20,
    ) -> None:
        lower, upper = (int(value) for value in total_quality_range)
        if lower > upper:
            raise ValueError("total_quality_range must be ordered")
        self.total_quality_range = (lower, upper)
        self.fractions = {
            "global_quality": float(global_fraction),
            "local_quality": float(local_fraction),
            "explore": float(explore_fraction),
        }
        if any(value < 0.0 for value in self.fractions.values()):
            raise ValueError("length-arm fractions must be non-negative")
        if sum(self.fractions.values()) <= 0.0:
            raise ValueError("length-arm fractions must have positive mass")

    def allocate(
        self,
        num_samples: int,
        *,
        local_added_range: tuple[int, int],
        explore_added_range: tuple[int, int],
        rng: random.Random,
    ) -> list[LengthProposal]:
        supports = {
            "global_quality": ("total", *self.total_quality_range),
            "local_quality": ("added", *local_added_range),
            "explore": ("added", *explore_added_range),
        }
        proposals: list[LengthProposal] = []
        for arm, count in _largest_remainder_counts(
            int(num_samples), self.fractions
        ).items():
            length_space, lower, upper = supports[arm]
            lower, upper = int(lower), int(upper)
            if lower > upper:
                raise ValueError(f"invalid support for {arm}: {(lower, upper)}")
            proposals.extend(
                LengthProposal(
                    arm=arm,
                    length_space=length_space,
                    support_min=lower,
                    support_max=upper,
                    quantile=quantile,
                )
                for quantile in _stratified_quantiles(count, rng)
            )
        rng.shuffle(proposals)
        return proposals
