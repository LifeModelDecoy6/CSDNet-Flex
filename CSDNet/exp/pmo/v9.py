"""Compatibility API for the lineage-aware PMO V9 policy.

The implementation now lives in :mod:`CSDNet.optim.frontier`. Keeping these
names preserves old checkpoints, tests, scripts, and result-recovery commands
while making V9 an adapter configuration of the shared optimizer.
"""

from CSDNet.optim.frontier import (
    AdaptiveOperatorBandit as V9BatchBandit,
    FrontierRecord as V9Record,
    FrontierRootStats as V9RootStats,
    LineageFrontierArchive as V9LineageArchive,
    RestoredPMOFrontierAdapter,
    allocate_weighted_counts,
    scalar_batch_frontier_reward as batch_frontier_reward,
)


def _adapter(**kwargs):
    return RestoredPMOFrontierAdapter(**kwargs)


def classify_v9_state(
    calls,
    warmup_calls,
    avg_top10,
    nonzero_rate,
    stagnant_calls,
    largest_root_fraction,
    saturation_threshold,
    sparse_threshold,
    stagnation_patience,
    collapse_threshold,
):
    """Return the historical V9 state using the shared PMO adapter."""
    return _adapter(
        warmup_calls=warmup_calls,
        saturation_threshold=saturation_threshold,
        sparse_threshold=sparse_threshold,
        stagnation_patience=stagnation_patience,
        collapse_threshold=collapse_threshold,
    ).classify(
        calls=calls,
        avg_top10=avg_top10,
        nonzero_rate=nonzero_rate,
        stagnant_calls=stagnant_calls,
        largest_root_fraction=largest_root_fraction,
    )


def v9_root_fraction(state):
    return float(_adapter().group_fractions(state)["root"])


def v9_root_weights(state):
    return _adapter().operator_priors("root", state)


def v9_local_weights(state):
    return _adapter().operator_priors("local", state)


__all__ = [
    "V9BatchBandit",
    "V9LineageArchive",
    "V9Record",
    "V9RootStats",
    "allocate_weighted_counts",
    "batch_frontier_reward",
    "classify_v9_state",
    "v9_local_weights",
    "v9_root_fraction",
    "v9_root_weights",
]
