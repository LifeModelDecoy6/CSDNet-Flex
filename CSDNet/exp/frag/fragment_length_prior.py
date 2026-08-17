"""Geometry-conditioned lower-bound priors for fragment infilling."""

from __future__ import annotations

import json
import random
from dataclasses import dataclass
from pathlib import Path


FRAGMENT_GAP_PRIOR_SCHEMA = "csdnet_fragment_gap_length_prior_v1"
SUPPORTED_GEOMETRIES = (
    "multi_anchor",
    "single_attachment",
    "multi_attachment",
    "substructure_expand",
)


PREFILL_ALLOCATION_PROFILES = {
    "balanced": {
        geometry: (25, 50, 25) for geometry in SUPPORTED_GEOMETRIES
    },
    # Calibrated once on seed 0 and then frozen. The policy observes only the
    # attachment geometry; it never reads a task name, target molecule or
    # molecular-property oracle at generation time.
    "geometry_calibrated": {
        "multi_anchor": (5, 25, 70),
        "single_attachment": (5, 75, 20),
        "multi_attachment": (5, 25, 70),
        "substructure_expand": (100, 0, 0),
    },
    # Preserve a large high-quality anchor while restoring the empirical
    # upper tail that is otherwise truncated before elastic decoding begins.
    # The fourth entry is the upper-tail allocation. Single-attachment and
    # substructure-expansion geometries already cover their observed modes
    # well, so their frozen v4 allocation is retained.
    "diversity_guarded": {
        "multi_anchor": (10, 30, 30, 30),
        "single_attachment": (5, 75, 20, 0),
        "multi_attachment": (10, 30, 30, 30),
        "substructure_expand": (100, 0, 0, 0),
    },
}


@dataclass(frozen=True)
class PrefillProposal:
    lengths: tuple[int, ...]
    source: str
    measure: str
    quantile: float | None
    prior_total: int


def _weighted_quantile(histogram: dict[str, int], quantile: float) -> int:
    values = sorted(
        (int(value), int(count))
        for value, count in histogram.items()
        if int(value) > 0 and int(count) > 0
    )
    if not values:
        raise ValueError("Length histogram contains no positive observations")
    quantile = min(max(float(quantile), 0.0), 1.0)
    total = sum(count for _, count in values)
    threshold = quantile * max(0, total - 1)
    cumulative = 0
    for value, count in values:
        cumulative += count
        if cumulative > threshold:
            return value
    return values[-1][0]


def _positive_partition(
    total: int,
    parts: int,
    *,
    maximum_per_part: int,
    rng: random.Random,
) -> tuple[int, ...]:
    parts = max(1, int(parts))
    maximum_per_part = max(1, int(maximum_per_part))
    total = min(
        max(parts, int(total)),
        parts * maximum_per_part,
    )
    allocation = [1] * parts
    remaining = total - parts
    if remaining <= 0:
        return tuple(allocation)

    weights = [rng.gammavariate(1.0, 1.0) for _ in range(parts)]
    weight_total = sum(weights) or 1.0
    raw = [remaining * weight / weight_total for weight in weights]
    additions = [min(maximum_per_part - 1, int(value)) for value in raw]
    for index, value in enumerate(additions):
        allocation[index] += value
    remaining = total - sum(allocation)
    order = sorted(
        range(parts),
        key=lambda index: (raw[index] - int(raw[index]), rng.random()),
        reverse=True,
    )
    while remaining > 0:
        progressed = False
        for index in order:
            if allocation[index] >= maximum_per_part:
                continue
            allocation[index] += 1
            remaining -= 1
            progressed = True
            if remaining == 0:
                break
        if not progressed:
            break
    return tuple(allocation)


class FragmentGapLengthPrior:
    """Validated ZINC-derived prior used only to initialize editable gaps.

    Initial masks cannot be deleted by the current reverse process. Sampling
    therefore keeps an explicit short-length anchor. Optional upper-tail
    proposals restore empirical support without making every trajectory long.
    The allocation between native, lower, middle and upper strata is explicit
    and can be conditioned on attachment geometry.
    """

    def __init__(self, payload: dict, *, path: str | Path | None = None):
        if payload.get("schema") != FRAGMENT_GAP_PRIOR_SCHEMA:
            raise ValueError("Incompatible fragment gap length-prior schema")
        if payload.get("tokenizer") != "csdnet_atomic_smiles":
            raise ValueError("Fragment prior must use the CSDNet atom tokenizer")
        geometries = payload.get("geometries")
        if not isinstance(geometries, dict):
            raise ValueError("Fragment prior has no geometry distributions")
        missing = [
            geometry
            for geometry in SUPPORTED_GEOMETRIES
            if geometry not in geometries
        ]
        if missing:
            raise ValueError(f"Fragment prior is missing geometries: {missing}")
        self.payload = payload
        self.path = str(path) if path is not None else None
        self.bin_width = int(payload.get("fixed_bin_width", 8))
        self.minimum_group_count = int(payload.get("minimum_group_count", 32))
        if self.bin_width < 1 or self.minimum_group_count < 1:
            raise ValueError("Fragment prior bin settings must be positive")

    @classmethod
    def load(cls, path: str | Path) -> "FragmentGapLengthPrior":
        path = Path(path)
        return cls(json.loads(path.read_text()), path=path)

    def _distribution(
        self,
        *,
        geometry: str,
        fixed_tokens: int,
        gap_count: int,
    ) -> tuple[dict, str]:
        if geometry not in SUPPORTED_GEOMETRIES:
            raise ValueError(f"Unsupported prior geometry: {geometry!r}")
        geometry_data = self.payload["geometries"][geometry]
        gap_count = max(1, int(gap_count))
        target_bin = (max(0, int(fixed_tokens)) // self.bin_width) * self.bin_width

        candidates = []
        for key, group in geometry_data.get("groups", {}).items():
            fixed_bin, group_gaps = (int(value) for value in key.split(":"))
            if group_gaps != gap_count:
                continue
            if int(group.get("count", 0)) < self.minimum_group_count:
                continue
            candidates.append((abs(fixed_bin - target_bin), fixed_bin, group))
        if candidates:
            distance, fixed_bin, group = min(
                candidates,
                key=lambda item: item[:2],
            )
            if distance <= 2 * self.bin_width:
                return group, f"fixed_bin_{fixed_bin}_gaps_{gap_count}"

        gap_group = geometry_data.get("gap_groups", {}).get(str(gap_count))
        if gap_group and int(gap_group.get("count", 0)) > 0:
            return gap_group, f"geometry_gaps_{gap_count}"
        global_group = geometry_data.get("global")
        if not global_group or int(global_group.get("count", 0)) <= 0:
            raise ValueError(f"No usable length observations for {geometry}")
        return global_group, "geometry_global"

    def propose(
        self,
        *,
        geometry: str,
        fixed_tokens: int,
        gap_count: int,
        attempt_index: int,
        constrained_atoms: bool,
        maximum_total: int,
        maximum_per_gap: int,
        rng: random.Random,
        allocation_profile: str = "balanced",
    ) -> PrefillProposal:
        if geometry not in SUPPORTED_GEOMETRIES:
            raise ValueError(f"Unsupported prior geometry: {geometry!r}")
        gap_count = max(1, int(gap_count))
        maximum_total = max(gap_count, int(maximum_total))
        maximum_per_gap = max(1, int(maximum_per_gap))
        if allocation_profile not in PREFILL_ALLOCATION_PROFILES:
            raise ValueError(
                f"Unsupported prefill allocation profile: "
                f"{allocation_profile!r}"
            )
        allocation = PREFILL_ALLOCATION_PROFILES[allocation_profile][geometry]
        if sum(allocation) != 100 or any(value < 0 for value in allocation):
            raise ValueError("Prefill allocation counts must sum to 100")
        if len(allocation) == 3:
            native_count, lower_count, middle_count = allocation
            upper_count = 0
        elif len(allocation) == 4:
            native_count, lower_count, middle_count, upper_count = allocation
        else:
            raise ValueError("Prefill allocation must contain 3 or 4 strata")

        # A deterministic permutation preserves the requested allocation over
        # every 100 attempts and avoids coupling source strata to batch order.
        stratum_slot = (int(attempt_index) * 37 + 11) % 100
        if stratum_slot < native_count:
            return PrefillProposal(
                lengths=tuple([1] * gap_count),
                source="native_one_mask",
                measure="atoms" if constrained_atoms else "tokens",
                quantile=None,
                prior_total=gap_count,
            )

        group, source = self._distribution(
            geometry=geometry,
            fixed_tokens=fixed_tokens,
            gap_count=gap_count,
        )
        measure = "atoms" if constrained_atoms else "tokens"
        histogram_key = (
            "atom_histogram" if measure == "atoms" else "token_histogram"
        )
        histogram = group.get(histogram_key, {})
        if not histogram:
            fallback = "tokens" if measure == "atoms" else "atoms"
            fallback_key = (
                "atom_histogram"
                if fallback == "atoms"
                else "token_histogram"
            )
            histogram = group.get(fallback_key, {})
            measure = fallback
        lower_end = native_count + lower_count
        middle_end = lower_end + middle_count
        if stratum_slot < lower_end:
            stratum = "lower"
            quantile_low, quantile_high = 0.10, 0.40
            stratum_rank = stratum_slot - native_count
            stratum_count = lower_count
        elif stratum_slot < middle_end:
            stratum = "middle"
            quantile_low, quantile_high = 0.40, 0.55
            stratum_rank = stratum_slot - lower_end
            stratum_count = middle_count
        else:
            if upper_count <= 0:
                raise RuntimeError("Proposal entered an empty upper stratum")
            stratum = "upper"
            quantile_low, quantile_high = 0.55, 0.90
            stratum_rank = stratum_slot - middle_end
            stratum_count = upper_count

        if allocation_profile == "diversity_guarded":
            # Latin-hypercube coverage prevents a finite 100-proposal case
            # from accidentally clustering at one length while preserving
            # the same uniform marginal distribution within each stratum.
            unit_quantile = (stratum_rank + rng.random()) / stratum_count
            quantile = quantile_low + (
                quantile_high - quantile_low
            ) * unit_quantile
        else:
            quantile = rng.uniform(quantile_low, quantile_high)
        prior_total = _weighted_quantile(histogram, quantile)
        total = min(
            max(gap_count, prior_total),
            maximum_total,
            maximum_per_gap * gap_count,
        )
        return PrefillProposal(
            lengths=_positive_partition(
                total,
                gap_count,
                maximum_per_part=maximum_per_gap,
                rng=rng,
            ),
            source=f"zinc_{stratum}_{source}",
            measure=measure,
            quantile=quantile,
            prior_total=prior_total,
        )


def apply_prefill_lengths(plans, lengths):
    plans = [dict(plan) for plan in plans]
    lengths = tuple(int(value) for value in lengths)
    if len(plans) != len(lengths):
        raise ValueError("One prefill length is required for every editable gap")
    for plan, length in zip(plans, lengths):
        maximum = int(plan.get("max_replacement_len", length))
        plan["initial_replacement_len"] = min(max(1, length), maximum)
    return tuple(plans)
