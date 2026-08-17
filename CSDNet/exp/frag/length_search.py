"""Task-agnostic non-parametric search over fixed-model infill lengths."""

from __future__ import annotations

import math
import random
from collections import Counter
from dataclasses import dataclass
from typing import Iterable, Sequence

import numpy as np


@dataclass(frozen=True)
class LengthProposal:
    added_tokens: int
    source: str


@dataclass
class LengthStats:
    attempts: int = 0
    structural: int = 0
    novel: int = 0

    @property
    def structural_rate(self) -> float:
        return (self.structural + 2.0) / (self.attempts + 3.0)

    @property
    def novelty_rate(self) -> float:
        return (self.novel + 1.0) / (self.structural + 2.0)


class NonParametricLengthController:
    """Coarse-to-fine length search with a protected training-prior floor.

    The controller observes only structural constraint success and canonical
    novelty. It never consumes task names, targets, QED, SA, distance, or
    benchmark quality values.
    """

    def __init__(
        self,
        added_length_prior: Sequence[int],
        *,
        minimum: int,
        maximum: int,
        warmup_attempts: int = 20,
        prior_floor: float = 0.65,
        exploration_fraction: float = 0.10,
        ucb_scale: float = 0.10,
        softmax_temperature: float = 0.18,
        refinement_radius: int = 2,
        feasible_lengths: Sequence[int] | None = None,
    ):
        minimum = int(minimum)
        maximum = int(maximum)
        if minimum < 1 or maximum < minimum:
            raise ValueError("Invalid added-token interval.")
        if not 0.0 <= prior_floor <= 1.0:
            raise ValueError("prior_floor must be in [0, 1].")
        if not 0.0 <= exploration_fraction <= 1.0:
            raise ValueError("exploration_fraction must be in [0, 1].")
        if prior_floor + exploration_fraction > 1.0:
            raise ValueError("prior and exploration fractions exceed one.")
        if softmax_temperature <= 0.0:
            raise ValueError("softmax_temperature must be positive.")

        if feasible_lengths is None:
            support = tuple(range(minimum, maximum + 1))
        else:
            support = tuple(
                sorted(
                    {
                        int(value)
                        for value in feasible_lengths
                        if minimum <= int(value) <= maximum
                    }
                )
            )
            if not support:
                raise ValueError("feasible_lengths contains no usable values.")

        def nearest_feasible(value: int) -> int:
            target = min(maximum, max(minimum, int(value)))
            return min(support, key=lambda item: (abs(item - target), item))

        clipped = [nearest_feasible(value) for value in added_length_prior]
        if not clipped:
            raise ValueError("added_length_prior cannot be empty.")

        self.minimum = support[0]
        self.maximum = support[-1]
        self.support = support
        self.warmup_attempts = max(0, int(warmup_attempts))
        self.prior_floor = float(prior_floor)
        self.exploration_fraction = float(exploration_fraction)
        self.ucb_scale = float(ucb_scale)
        self.softmax_temperature = float(softmax_temperature)
        self.refinement_radius = max(1, int(refinement_radius))

        counts = Counter(clipped)
        # Additive smoothing keeps every feasible neighbouring length reachable.
        self.prior_weights = {
            length: float(counts.get(length, 0) + 1)
            for length in self.support
        }
        maximum_weight = max(self.prior_weights.values())
        self.prior_density = {
            length: weight / maximum_weight
            for length, weight in self.prior_weights.items()
        }
        quantiles = np.quantile(
            np.asarray(clipped, dtype=np.int64),
            [0.05, 0.20, 0.35, 0.50, 0.65, 0.80, 0.95],
            method="nearest",
        )
        self.anchors = tuple(sorted({int(value) for value in quantiles}))
        self.active = set(self.anchors)
        self.stats = {length: LengthStats() for length in self.support}
        self.total_attempts = 0

    @staticmethod
    def _counts(batch_size: int, fractions: Sequence[float]) -> list[int]:
        raw = [batch_size * float(value) for value in fractions]
        counts = [int(math.floor(value)) for value in raw]
        remainder = batch_size - sum(counts)
        order = sorted(
            range(len(raw)),
            key=lambda index: (raw[index] - counts[index], -index),
            reverse=True,
        )
        for index in order[:remainder]:
            counts[index] += 1
        return counts

    @staticmethod
    def _weighted_choice(
        values: Sequence[int],
        weights: Sequence[float],
        rng: random.Random,
    ) -> int:
        return int(rng.choices(list(values), weights=list(weights), k=1)[0])

    def _score(self, length: int) -> float:
        item = self.stats[length]
        uncertainty = self.ucb_scale * math.sqrt(
            math.log(self.total_attempts + 2.0) / (item.attempts + 1.0)
        )
        return (
            0.60 * item.structural_rate
            + 0.25 * item.novelty_rate
            + 0.15 * self.prior_density[length]
            + uncertainty
        )

    def _adaptive_choice(self, rng: random.Random) -> int:
        values = sorted(self.active)
        scores = [self._score(length) for length in values]
        peak = max(scores)
        weights = [
            math.exp((score - peak) / self.softmax_temperature)
            for score in scores
        ]
        return self._weighted_choice(values, weights, rng)

    def _coverage_choice(
        self,
        rng: random.Random,
        reserved: Counter[int] | None = None,
    ) -> int:
        reserved = reserved or Counter()
        minimum_attempts = min(
            self.stats[length].attempts + reserved[length]
            for length in self.support
        )
        candidates = [
            length
            for length in self.support
            if self.stats[length].attempts + reserved[length] == minimum_attempts
        ]
        return int(rng.choice(candidates))

    def _prior_choice(self, rng: random.Random) -> int:
        values = list(self.support)
        weights = [self.prior_weights[length] for length in values]
        return self._weighted_choice(values, weights, rng)

    def allocate(
        self,
        batch_size: int,
        rng: random.Random,
    ) -> list[LengthProposal]:
        batch_size = max(0, int(batch_size))
        if batch_size == 0:
            return []

        remaining_warmup = max(0, self.warmup_attempts - self.total_attempts)
        warmup_count = min(batch_size, remaining_warmup)
        proposals = []
        reserved: Counter[int] = Counter()
        for _ in range(warmup_count):
            least = min(
                self.stats[length].attempts + reserved[length]
                for length in self.anchors
            )
            candidates = [
                length
                for length in self.anchors
                if self.stats[length].attempts + reserved[length] == least
            ]
            length = int(rng.choice(candidates))
            reserved[length] += 1
            proposals.append(LengthProposal(length, "warmup"))

        remaining = batch_size - warmup_count
        if remaining:
            prior_count, exploration_count, adaptive_count = self._counts(
                remaining,
                (
                    self.prior_floor,
                    self.exploration_fraction,
                    1.0 - self.prior_floor - self.exploration_fraction,
                ),
            )
            proposals.extend(
                LengthProposal(self._prior_choice(rng), "prior_floor")
                for _ in range(prior_count)
            )
            for _ in range(exploration_count):
                length = self._coverage_choice(rng, reserved)
                reserved[length] += 1
                proposals.append(LengthProposal(length, "coverage"))
            proposals.extend(
                LengthProposal(self._adaptive_choice(rng), "adaptive")
                for _ in range(adaptive_count)
            )

        rng.shuffle(proposals)
        return proposals

    def update(
        self,
        observations: Iterable[tuple[LengthProposal, bool, bool]],
    ) -> None:
        for proposal, structural_success, novel in observations:
            length = int(proposal.added_tokens)
            item = self.stats[length]
            item.attempts += 1
            item.structural += int(bool(structural_success))
            item.novel += int(bool(novel) and bool(structural_success))
            self.total_attempts += 1

        observed = [
            length
            for length in self.support
            if self.stats[length].attempts > 0
        ]
        if not observed:
            return
        leaders = sorted(
            observed,
            key=lambda length: (
                self._score(length),
                self.stats[length].structural,
                -abs(length - self.anchors[len(self.anchors) // 2]),
            ),
            reverse=True,
        )[:3]
        self.active.update(self.anchors)
        for leader in leaders:
            lower = max(self.minimum, leader - self.refinement_radius)
            upper = min(self.maximum, leader + self.refinement_radius)
            self.active.update(
                length
                for length in self.support
                if lower <= length <= upper
            )

    def records(self) -> list[dict]:
        rows = []
        for length in self.support:
            item = self.stats[length]
            rows.append(
                {
                    "added_tokens": length,
                    "active": length in self.active,
                    "prior_density": self.prior_density[length],
                    "attempts": item.attempts,
                    "structural": item.structural,
                    "novel": item.novel,
                    "structural_rate": item.structural_rate,
                    "novelty_rate": item.novelty_rate,
                    "score": self._score(length),
                }
            )
        return rows
