"""Task-agnostic frontier ranking, state, and trust-region utilities."""

from __future__ import annotations

import math
import random
from collections import Counter
from dataclasses import dataclass
from typing import Iterable, Mapping


@dataclass(frozen=True)
class FrontierRecord:
    """One scalar-objective candidate and its root-lineage provenance."""

    smiles: str
    score: float
    root_id: str
    root_operator: str
    depth: int
    operator: str
    created_call: int


@dataclass
class FrontierRootStats:
    """Delayed credit accumulated by an independent proposal lineage."""

    root_id: str
    root_operator: str
    root_smiles: str
    root_score: float
    best_score: float
    created_call: int
    descendants: int = 0
    frontier_credit: float = 0.0
    top10_entries: int = 0


class LineageFrontierArchive:
    """Scalar frontier archive with protected slots for independent roots.

    This is the lineage mechanism used by the best PMO V9 policy. Keeping it
    in the shared optimizer makes lineage preservation, delayed root credit,
    and parent selection reusable without putting proposal generators in task
    adapters.
    """

    def __init__(self, score_slots, lineage_slots, root_ucb_weight=0.04):
        self.score_slots = max(1, int(score_slots))
        self.lineage_slots = max(1, int(lineage_slots))
        self.root_ucb_weight = max(0.0, float(root_ucb_weight))
        self.records: dict[str, FrontierRecord] = {}
        self.roots: dict[str, FrontierRootStats] = {}

    def add_root(
        self,
        smiles,
        score,
        root_id,
        root_operator,
        created_call,
    ):
        record = FrontierRecord(
            smiles=str(smiles),
            score=float(score),
            root_id=str(root_id),
            root_operator=str(root_operator),
            depth=0,
            operator=str(root_operator),
            created_call=int(created_call),
        )
        self.records[record.smiles] = record
        self.roots[record.root_id] = FrontierRootStats(
            root_id=record.root_id,
            root_operator=record.root_operator,
            root_smiles=record.smiles,
            root_score=record.score,
            best_score=record.score,
            created_call=record.created_call,
        )
        return record

    def add_child(
        self,
        smiles,
        score,
        parent,
        operator,
        created_call,
        frontier_gain=0.0,
        entered_top10=False,
    ):
        record = FrontierRecord(
            smiles=str(smiles),
            score=float(score),
            root_id=parent.root_id,
            root_operator=parent.root_operator,
            depth=int(parent.depth) + 1,
            operator=str(operator),
            created_call=int(created_call),
        )
        self.records[record.smiles] = record
        root = self.roots[record.root_id]
        root.descendants += 1
        root.best_score = max(root.best_score, record.score)
        root.frontier_credit += max(0.0, float(frontier_gain))
        root.top10_entries += int(bool(entered_top10))
        return record

    def _best_per_root(self):
        best = {}
        for record in self.records.values():
            old = best.get(record.root_id)
            if old is None or record.score > old.score:
                best[record.root_id] = record
        return list(best.values())

    def _root_priority(self, record):
        root = self.roots[record.root_id]
        total = max(1, sum(item.descendants + 1 for item in self.roots.values()))
        explore = self.root_ucb_weight * math.sqrt(
            math.log(total + 1.0) / (root.descendants + 1.0)
        )
        delayed = min(0.08, root.frontier_credit * 0.5)
        return float(record.score) + explore + delayed

    def parent_pools(self):
        score_pool = sorted(
            self.records.values(), key=lambda row: row.score, reverse=True
        )[: self.score_slots]
        lineage_pool = sorted(
            self._best_per_root(), key=self._root_priority, reverse=True
        )[: self.lineage_slots]
        return score_pool, lineage_pool

    def choose_parent(self, rng: random.Random, lineage_probability=0.30):
        score_pool, lineage_pool = self.parent_pools()
        if lineage_pool and rng.random() < float(lineage_probability):
            return rng.choice(lineage_pool)
        if score_pool:
            return rng.choice(score_pool)
        if lineage_pool:
            return rng.choice(lineage_pool)
        return None

    def metrics(self, top_n=None):
        if not self.records:
            return {
                "root_count": 0.0,
                "largest_root_fraction": 0.0,
                "lineage_entropy": 0.0,
            }
        limit = top_n or (self.score_slots + self.lineage_slots)
        rows = sorted(
            self.records.values(), key=lambda row: row.score, reverse=True
        )[: max(1, int(limit))]
        counts = Counter(row.root_id for row in rows)
        probabilities = [count / len(rows) for count in counts.values()]
        entropy = -sum(prob * math.log(prob + 1e-12) for prob in probabilities)
        if len(probabilities) > 1:
            entropy /= math.log(len(probabilities))
        else:
            entropy = 0.0
        return {
            "root_count": float(len(counts)),
            "largest_root_fraction": float(max(probabilities)),
            "lineage_entropy": float(entropy),
        }


def constraint_rank(item):
    """Return a scale-free lexicographic rank; lower values are better.

    Feasible candidates are ordered only by the task objective (``dock`` for
    lead optimization). Infeasible candidates are ordered by the number of
    violated constraints, then their largest and mean normalized violations.
    """
    checks = item.get("checks", {})
    deficits = item.get("deficits", {})
    objective = float(item.get("dock", item.get("score", 0.0)))
    if checks and all(bool(value) for value in checks.values()):
        return (0, 0, 0.0, 0.0, -objective)
    values = [max(0.0, float(value)) for value in deficits.values()]
    violated = sum(value > 0.0 for value in values)
    max_violation = max(values, default=0.0)
    mean_violation = sum(values) / max(1, len(values))
    return (1, violated, max_violation, mean_violation, -objective)


def rank_improved(child, parent, tolerance=1e-6):
    """Whether ``child`` improves on ``parent`` under constraint domination."""
    child_rank = constraint_rank(child)
    parent_rank = constraint_rank(parent)
    if child_rank[:2] != parent_rank[:2]:
        return child_rank[:2] < parent_rank[:2]
    for child_value, parent_value in zip(child_rank[2:], parent_rank[2:]):
        if child_value < parent_value - tolerance:
            return True
        if child_value > parent_value + tolerance:
            return False
    return False


def lineage_metrics(items, limit=100):
    """Summarize root-lineage concentration in a ranked archive."""
    rows = list(items)[: max(1, int(limit))]
    if not rows:
        return {"root_count": 0, "largest_root_fraction": 0.0, "lineage_entropy": 0.0}
    counts = Counter(str(item.get("root_id") or "unknown") for item in rows)
    total = sum(counts.values())
    probabilities = [count / total for count in counts.values()]
    entropy = -sum(prob * math.log(prob + 1e-12) for prob in probabilities)
    if len(probabilities) > 1:
        entropy /= math.log(len(probabilities))
    else:
        entropy = 0.0
    return {
        "root_count": len(counts),
        "largest_root_fraction": max(probabilities),
        "lineage_entropy": entropy,
    }


def classify_frontier_state(
    iteration,
    warmup_iterations,
    has_feasible,
    stagnant_iterations,
    plateau_patience,
    largest_root_fraction,
    collapse_threshold,
):
    """Classify search state without task- or target-specific rules."""
    if int(iteration) < int(warmup_iterations):
        return "warmup"
    if has_feasible:
        return "refine"
    if float(largest_root_fraction) >= float(collapse_threshold):
        return "collapsed"
    if int(stagnant_iterations) >= int(plateau_patience):
        return "plateau"
    return "search"


def trust_region_fraction(
    parent,
    target_keys,
    base_fraction,
    min_fraction=0.04,
    max_fraction=0.30,
    deficit_scale=0.18,
    slack_scale=0.25,
):
    """Choose edit radius from target deficit and preserved-constraint slack."""
    target_keys = set(target_keys)
    deficits = parent.get("deficits", {})
    normalized = parent.get("normalized", {})
    checks = parent.get("checks", {})
    target_deficit = max(
        (max(0.0, float(deficits.get(key, 0.0))) for key in target_keys),
        default=max((float(value) for value in deficits.values()), default=0.0),
    )
    preserved_margins = [
        max(0.0, float(normalized.get(key, 0.0)) - 1.0)
        for key, satisfied in checks.items()
        if key not in target_keys and satisfied
    ]
    if preserved_margins:
        slack = min(1.0, min(preserved_margins) / max(float(slack_scale), 1e-8))
    else:
        slack = 1.0
    radius = float(base_fraction) + float(deficit_scale) * target_deficit * (
        0.35 + 0.65 * slack
    )
    return max(float(min_fraction), min(float(max_fraction), radius))


def integrated_operator_weights(state):
    """State-conditioned proposal priors shared by constrained tasks."""
    if state == "warmup":
        return {
            "start_repair": 0.28,
            "joint_repair": 0.42,
            "lineage_restart": 0.30,
        }
    if state == "refine":
        return {
            "start_repair": 0.05,
            "dock_refine": 0.50,
            "similarity_repair": 0.10,
            "quality_repair": 0.10,
            "joint_repair": 0.10,
            "lineage_restart": 0.15,
        }
    if state in {"plateau", "collapsed"}:
        return {
            "start_repair": 0.22,
            "dock_refine": 0.08,
            "similarity_repair": 0.10,
            "quality_repair": 0.10,
            "joint_repair": 0.20,
            "lineage_restart": 0.30,
        }
    return {
        "start_repair": 0.18,
        "dock_refine": 0.08,
        "similarity_repair": 0.12,
        "quality_repair": 0.12,
        "joint_repair": 0.32,
        "lineage_restart": 0.18,
    }


def allocate_weighted_counts(total, weighted):
    """Allocate an integer proposal budget without losing any slots."""
    total = max(0, int(total))
    rows = [(str(name), max(0.0, float(weight))) for name, weight in weighted]
    weight_sum = sum(weight for _, weight in rows)
    if total == 0 or not rows or weight_sum <= 0.0:
        return {}
    raw = [(name, total * weight / weight_sum) for name, weight in rows]
    counts = {name: int(math.floor(value)) for name, value in raw}
    remainder = total - sum(counts.values())
    fractions = sorted(
        ((value - math.floor(value), name) for name, value in raw),
        reverse=True,
    )
    for _, name in fractions[:remainder]:
        counts[name] += 1
    return {name: count for name, count in counts.items() if count > 0}


def allocate_insertion_flags(total, fraction, rng=None):
    """Allocate an exact learned-insertion sub-budget within a proposal batch."""
    total = max(0, int(total))
    fraction = max(0.0, min(1.0, float(fraction)))
    learned = min(total, max(0, int(round(total * fraction))))
    flags = [True] * learned + [False] * (total - learned)
    if rng is None:
        random.shuffle(flags)
    else:
        rng.shuffle(flags)
    return flags


class AdaptiveOperatorBandit:
    """Batch-updated UCB policy shared by PMO and constrained search."""

    def __init__(
        self,
        operators: Iterable[str],
        alpha=0.20,
        temperature=2.0,
        ucb_weight=0.30,
        min_multiplier=0.05,
        base_floor=0.30,
    ):
        self.alpha = max(0.0, min(1.0, float(alpha)))
        self.temperature = max(0.0, float(temperature))
        self.ucb_weight = max(0.0, float(ucb_weight))
        self.min_multiplier = max(0.0, float(min_multiplier))
        self.base_floor = max(0.0, float(base_floor))
        self.stats = {
            str(operator): {
                "ema": 0.50,
                "batches": 0.0,
                "evaluated": 0.0,
                "last_reward": 0.0,
            }
            for operator in operators
        }

    def weighted(self, base_weights):
        total_batches = max(
            1.0,
            sum(float(row["batches"]) for row in self.stats.values()),
        )
        weighted = []
        for operator, base in base_weights.items():
            if float(base) <= 0.0:
                continue
            row = self.stats.setdefault(
                operator,
                {
                    "ema": 0.50,
                    "batches": 0.0,
                    "evaluated": 0.0,
                    "last_reward": 0.0,
                },
            )
            exploit = math.exp(
                self.temperature * (float(row["ema"]) - 0.50)
            )
            explore = self.ucb_weight * math.sqrt(
                math.log(total_batches + 1.0) / (float(row["batches"]) + 1.0)
            )
            multiplier = max(
                self.min_multiplier,
                min(4.0, exploit + explore),
            )
            weighted.append(
                (operator, float(base) * (self.base_floor + multiplier))
            )
        return weighted

    def update(self, operator, reward, evaluated):
        row = self.stats.setdefault(
            operator,
            {
                "ema": 0.50,
                "batches": 0.0,
                "evaluated": 0.0,
                "last_reward": 0.0,
            },
        )
        value = max(0.0, min(1.0, float(reward))) if evaluated > 0 else 0.0
        row["ema"] = (
            (1.0 - self.alpha) * float(row["ema"]) + self.alpha * value
        )
        row["batches"] = float(row["batches"]) + 1.0
        row["evaluated"] = float(row["evaluated"]) + max(0, int(evaluated))
        row["last_reward"] = value

    def delayed_credit(self, operator, reward, alpha=0.05):
        row = self.stats.get(operator)
        if row is None:
            return
        rate = max(0.0, min(1.0, float(alpha)))
        value = max(0.0, min(1.0, float(reward)))
        row["ema"] = (1.0 - rate) * float(row["ema"]) + rate * value

    def snapshot(self):
        return {
            operator: {
                "ema": float(row["ema"]),
                "batches": int(row["batches"]),
                "evaluated": int(row["evaluated"]),
                "last_reward": float(row["last_reward"]),
            }
            for operator, row in sorted(self.stats.items())
        }

    def load_snapshot(self, snapshot):
        for operator, saved in dict(snapshot or {}).items():
            row = self.stats.setdefault(
                str(operator),
                {
                    "ema": 0.50,
                    "batches": 0.0,
                    "evaluated": 0.0,
                    "last_reward": 0.0,
                },
            )
            row["ema"] = max(0.0, min(1.0, float(saved.get("ema", 0.50))))
            row["batches"] = max(0.0, float(saved.get("batches", 0.0)))
            row["evaluated"] = max(0.0, float(saved.get("evaluated", 0.0)))
            row["last_reward"] = max(
                0.0,
                min(1.0, float(saved.get("last_reward", 0.0))),
            )


def scalar_batch_frontier_reward(
    scores,
    before_scores,
    before_top10,
    after_top10,
    parent_scores,
    frontier_scale,
    delta_scale,
):
    """Order-invariant PMO credit based on frontier movement and quality."""
    values = [float(value) for value in scores]
    history = [float(value) for value in before_scores]
    parents = list(parent_scores)
    if not values:
        return 0.0, {
            "frontier_gain": 0.0,
            "frontier_signal": 0.0,
            "entry_rate": 0.0,
            "tail_signal": 0.0,
            "positive_delta_signal": 0.0,
        }

    if history:
        percentiles = [
            sum(old <= value for old in history) / len(history)
            for value in values
        ]
    else:
        percentiles = [max(0.0, min(1.0, value)) for value in values]

    merged = sorted(history + values, reverse=True)
    top_n = min(10, len(merged))
    threshold = merged[top_n - 1]
    entries = min(top_n, sum(value >= threshold - 1e-12 for value in values))
    entry_rate = entries / max(1, min(10, len(values)))

    tail_n = max(1, int(math.ceil(0.20 * len(values))))
    percentile_tail = sorted(value**4 for value in percentiles)[-tail_n:]
    tail_signal = sum(percentile_tail) / len(percentile_tail)
    frontier_gain = max(0.0, float(after_top10) - float(before_top10))
    frontier_signal = min(
        1.0,
        frontier_gain / max(1e-6, float(frontier_scale)),
    )

    positive_deltas = [
        max(0.0, score - float(parent))
        for score, parent in zip(values, parents)
        if parent is not None
    ]
    if positive_deltas:
        positive_delta_signal = sum(
            min(1.0, value / max(float(delta_scale), 1e-6))
            for value in positive_deltas
        ) / len(positive_deltas)
    else:
        positive_delta_signal = 0.0

    entry_signal = min(1.0, entry_rate / 0.10)
    reward = (
        0.55 * frontier_signal
        + 0.25 * entry_signal
        + 0.15 * tail_signal
        + 0.05 * positive_delta_signal
    )
    return max(0.0, min(1.0, reward)), {
        "frontier_gain": frontier_gain,
        "frontier_signal": frontier_signal,
        "entry_rate": entry_rate,
        "tail_signal": tail_signal,
        "positive_delta_signal": positive_delta_signal,
    }


@dataclass(frozen=True)
class ScalarFrontierAdapter:
    """Task adapter for a bounded scalar oracle such as a PMO objective."""

    warmup_calls: int = 1000
    saturation_threshold: float = 0.90
    sparse_threshold: float = 0.25
    stagnation_patience: int = 800
    collapse_threshold: float = 0.60

    def classify(
        self,
        *,
        calls,
        avg_top10,
        nonzero_rate,
        stagnant_calls,
        largest_root_fraction,
    ):
        if int(calls) < int(self.warmup_calls):
            return "warmup"
        if float(avg_top10) >= float(self.saturation_threshold):
            return "saturated"
        if float(largest_root_fraction) >= float(self.collapse_threshold):
            return "collapsed"
        if int(stagnant_calls) >= int(self.stagnation_patience):
            return "plateau"
        if float(nonzero_rate) < float(self.sparse_threshold):
            return "sparse"
        return "search"

    @staticmethod
    def group_fractions(state, context=None):
        root = {
            "warmup": 1.00,
            "saturated": 0.06,
            "collapsed": 0.32,
            "plateau": 0.28,
            "sparse": 0.30,
            "search": 0.18,
            "fallback": 1.00,
        }.get(state, 0.18)
        return {"root": root, "local": max(0.0, 1.0 - root)}

    @staticmethod
    def insertion_fraction(state, context=None):
        """Budget learned local length edits from task-agnostic search state."""
        return {
            "warmup": 0.08,
            "saturated": 0.06,
            "collapsed": 0.30,
            "plateau": 0.28,
            "sparse": 0.24,
            "search": 0.18,
        }.get(state, 0.18)

    @staticmethod
    def operator_priors(group, state, context=None):
        if group == "root":
            if state == "warmup":
                return {"attach_only": 1.0}
            if state == "saturated":
                return {
                    "attach_only": 0.75,
                    "motif_restart": 0.10,
                    "fragment_anchor": 0.15,
                }
            if state in {"collapsed", "plateau"}:
                return {
                    "attach_only": 0.40,
                    "motif_restart": 0.30,
                    "fragment_anchor": 0.30,
                }
            if state == "sparse":
                return {
                    "attach_only": 0.45,
                    "motif_restart": 0.30,
                    "fragment_anchor": 0.25,
                }
            return {
                "attach_only": 0.55,
                "motif_restart": 0.20,
                "fragment_anchor": 0.25,
            }

        if state == "saturated":
            return {
                "elite_tiny": 0.34,
                "elite_small": 0.32,
                "elite_medium": 0.08,
                "diverse_medium": 0.04,
                "graph_shrink": 0.22,
            }
        if state in {"collapsed", "plateau", "sparse"}:
            return {
                "elite_tiny": 0.15,
                "elite_small": 0.25,
                "elite_medium": 0.15,
                "diverse_medium": 0.15,
                "graph_shrink": 0.22,
                "graph_swap": 0.06,
                "graph_expand": 0.01,
                "rescue_large": 0.01,
            }
        return {
            "elite_tiny": 0.20,
            "elite_small": 0.28,
            "elite_medium": 0.14,
            "diverse_medium": 0.12,
            "graph_shrink": 0.21,
            "graph_swap": 0.035,
            "graph_expand": 0.005,
            "rescue_large": 0.01,
        }


@dataclass(frozen=True)
class RestoredPMOFrontierAdapter(ScalarFrontierAdapter):
    """Compatibility adapter for the best PMO V9 policy.

    The state thresholds, root fraction, and operator priors are inherited
    unchanged from V9. The class exists to make the task adapter explicit while
    proposal generation, bandit learning, lineage, and recovery stay shared.
    """


@dataclass(frozen=True)
class PMOFrontierAdapter(ScalarFrontierAdapter):
    """Scalar PMO policy with evidence-driven local-edit allocation.

    The V9 diagnostics showed that large graph shrinking consumed substantial
    oracle budget without entering the top-10 frontier. This adapter keeps that
    operator available for recovery, but moves the normal search budget toward
    small elite edits and lets the low-floor bandit suppress persistently weak
    operators.
    """

    @staticmethod
    def operator_priors(group, state, context=None):
        if group == "root":
            return ScalarFrontierAdapter.operator_priors(group, state, context)
        if state == "saturated":
            return {
                "elite_tiny": 0.44,
                "elite_small": 0.40,
                "elite_medium": 0.07,
                "diverse_medium": 0.03,
                "graph_shrink": 0.05,
                "graph_swap": 0.01,
            }
        if state in {"collapsed", "plateau", "sparse"}:
            return {
                "elite_tiny": 0.22,
                "elite_small": 0.30,
                "elite_medium": 0.16,
                "diverse_medium": 0.16,
                "graph_shrink": 0.06,
                "graph_swap": 0.06,
                "graph_expand": 0.02,
                "rescue_large": 0.02,
            }
        return {
            "elite_tiny": 0.32,
            "elite_small": 0.36,
            "elite_medium": 0.14,
            "diverse_medium": 0.10,
            "graph_shrink": 0.04,
            "graph_swap": 0.025,
            "graph_expand": 0.005,
            "rescue_large": 0.01,
        }


def constrained_batch_frontier_reward(
    transitions,
    *,
    tail_fraction=0.20,
    mean_weight=0.15,
    regression_penalty=0.30,
    residual_scale=0.20,
):
    """Credit constrained proposals by frontier progress without hiding harm.

    Lead-optimization successes are sparse, so the upper tail carries most of
    the reward. Constraint regressions are charged separately at batch level;
    an operator cannot compensate for repeatedly destroying satisfied
    constraints merely by producing many mediocre novel molecules.
    """
    rows = [dict(row) for row in transitions]
    if not rows:
        return 0.0, {
            "tail_signal": 0.0,
            "mean_signal": 0.0,
            "strict_rate": 0.0,
            "rank_improvement_rate": 0.0,
            "regression_rate": 0.0,
        }

    values = []
    regressed_rows = 0
    for row in rows:
        residual_gain = float(row.get("residual_gain", 0.0))
        gain_signal = max(
            -1.0,
            min(1.0, residual_gain / max(1e-8, float(residual_scale))),
        )
        crossed = min(1.0, max(0.0, float(row.get("crossed", 0.0))))
        pair_gain = min(1.0, max(0.0, float(row.get("pair_gain", 0.0))))
        regressed = max(0.0, float(row.get("regressed", 0.0)))
        regressed_rows += int(regressed > 0.0)
        value = (
            0.28 * float(bool(row.get("rank_improved", False)))
            + 0.24 * float(bool(row.get("strict", False)))
            + 0.13 * crossed
            + 0.10 * pair_gain
            + 0.08 * float(bool(row.get("admitted", False)))
            + 0.17 * max(0.0, gain_signal)
            - float(regression_penalty) * min(1.0, regressed)
            - 0.10 * max(0.0, -gain_signal)
        )
        values.append(max(0.0, min(1.0, value)))

    tail_n = max(1, int(math.ceil(float(tail_fraction) * len(values))))
    tail_signal = sum(sorted(values)[-tail_n:]) / tail_n
    mean_signal = sum(values) / len(values)
    regression_rate = regressed_rows / len(rows)
    reward = (
        (1.0 - float(mean_weight)) * tail_signal
        + float(mean_weight) * mean_signal
    ) * (1.0 - 0.50 * regression_rate)
    return max(0.0, min(1.0, reward)), {
        "tail_signal": tail_signal,
        "mean_signal": mean_signal,
        "strict_rate": sum(bool(row.get("strict", False)) for row in rows)
        / len(rows),
        "rank_improvement_rate": sum(
            bool(row.get("rank_improved", False)) for row in rows
        )
        / len(rows),
        "regression_rate": regression_rate,
    }


@dataclass(frozen=True)
class LeadFrontierAdapter:
    """Hard-constraint adapter for lead optimization.

    All decisions depend on observed deficits and search dynamics, never on a
    target or seed identity. The adapter therefore shares one policy over all
    benchmark tasks while retaining the hard feasibility semantics of lead
    optimization.
    """

    warmup_iterations: int = 2
    plateau_patience: int = 2
    collapse_threshold: float = 0.70
    bridge_deficit: float = 0.15
    need_weight: float = 1.50
    tail_fraction: float = 0.20
    mean_weight: float = 0.15
    regression_penalty: float = 0.30

    def classify(
        self,
        *,
        iteration,
        has_feasible,
        has_pair_feasible,
        best_max_deficit,
        stagnant_iterations,
        largest_root_fraction,
        available_operators=None,
        constraint_needs=None,
        completion_operators=None,
        has_generated_similarity_feasible=True,
        similarity_threshold=None,
    ):
        if int(iteration) < int(self.warmup_iterations):
            return "warmup"
        if bool(has_feasible):
            return "refine"
        if float(largest_root_fraction) >= float(self.collapse_threshold):
            return "collapsed"
        if int(stagnant_iterations) >= int(self.plateau_patience):
            return "plateau"
        if bool(has_pair_feasible) or float(best_max_deficit) <= float(
            self.bridge_deficit
        ):
            return "bridge"
        return "search"

    @staticmethod
    def group_fractions(state, context=None):
        return {"proposal": 1.0}

    def operator_priors(self, group, state, context=None):
        if group != "proposal":
            return {}
        priors = {
            "warmup": {
                "legacy": 0.35,
                "start_repair": 0.20,
                "joint_repair": 0.25,
                "lineage_restart": 0.20,
            },
            "search": {
                "legacy": 0.20,
                "start_repair": 0.12,
                "dock_refine": 0.08,
                "similarity_repair": 0.14,
                "quality_repair": 0.12,
                "joint_repair": 0.22,
                "lineage_restart": 0.12,
            },
            "bridge": {
                "legacy": 0.08,
                "start_repair": 0.05,
                "dock_refine": 0.30,
                "similarity_repair": 0.14,
                "quality_repair": 0.10,
                "joint_repair": 0.23,
                "lineage_restart": 0.10,
            },
            "refine": {
                "legacy": 0.05,
                "start_repair": 0.04,
                "dock_refine": 0.50,
                "similarity_repair": 0.12,
                "quality_repair": 0.10,
                "joint_repair": 0.12,
                "lineage_restart": 0.07,
            },
            "plateau": {
                "legacy": 0.12,
                "start_repair": 0.15,
                "dock_refine": 0.08,
                "similarity_repair": 0.14,
                "quality_repair": 0.12,
                "joint_repair": 0.20,
                "lineage_restart": 0.19,
            },
            "collapsed": {
                "legacy": 0.12,
                "start_repair": 0.20,
                "dock_refine": 0.06,
                "similarity_repair": 0.13,
                "quality_repair": 0.11,
                "joint_repair": 0.18,
                "lineage_restart": 0.20,
            },
        }.get(state, {})
        context = dict(context or {})
        available = set(context.get("available_operators") or priors)
        needs = dict(context.get("constraint_needs") or {})
        weighted = {}
        for operator, prior in priors.items():
            if operator not in available:
                continue
            need = max(0.0, float(needs.get(operator, 0.0)))
            weighted[operator] = float(prior) * (1.0 + self.need_weight * need)
        return weighted

    @staticmethod
    def trust_region_bounds(state, configured_min, configured_max):
        cap = {
            "warmup": 0.24,
            "search": 0.22,
            "bridge": 0.16,
            "refine": 0.12,
            "plateau": 0.26,
            "collapsed": 0.28,
        }.get(state, 0.22)
        lower = max(0.0, float(configured_min))
        return lower, max(lower, min(float(configured_max), cap))

    @staticmethod
    def length_edit_scale(state, context=None):
        return {
            "warmup": 0.50,
            "search": 0.45,
            "bridge": 0.35,
            "refine": 0.15,
            "plateau": 0.70,
            "collapsed": 0.85,
        }.get(state, 0.45)

    @staticmethod
    def insertion_fraction(state, context=None):
        """Reserve bounded learned-infill slots without target-specific rules."""
        return {
            "locked": 0.00,
            "baseline": 0.12,
            "warmup": 0.18,
            "search": 0.20,
            "seed_anchor": 0.20,
            "bridge": 0.14,
            "bridge_probe": 0.16,
            "completion": 0.10,
            "explore": 0.28,
            "refine": 0.08,
            "dock_polish": 0.06,
            "similarity_boundary": 0.06,
            "quality_boundary": 0.08,
            "late_route_probe": 0.12,
            "late_route_commit": 0.08,
            "late_anchor_probe": 0.14,
            "late_anchor_route_probe": 0.12,
            "late_anchor_route_commit": 0.08,
            "late_route_portfolio": 0.12,
            "late_seed_probe": 0.14,
            "plateau": 0.30,
            "collapsed": 0.34,
        }.get(state, 0.20)

    def batch_reward(self, transitions):
        return constrained_batch_frontier_reward(
            transitions,
            tail_fraction=self.tail_fraction,
            mean_weight=self.mean_weight,
            regression_penalty=self.regression_penalty,
        )


@dataclass(frozen=True)
class LeadFrontierAdapterV21(LeadFrontierAdapter):
    """Constraint-completion lead policy with a stable proposal floor.

    This adapter uses only observed frontier occupancy, normalized deficits,
    lineage concentration, and stagnation. It never inspects a target name.
    Pair-feasible candidates take precedence over collapse/plateau recovery so
    that the remaining constraint is completed with a small trust-region edit.
    """

    bridge_deficit: float = 0.15

    def classify(
        self,
        *,
        iteration,
        has_feasible,
        has_pair_feasible,
        best_max_deficit,
        stagnant_iterations,
        largest_root_fraction,
        available_operators=None,
        constraint_needs=None,
        completion_operators=None,
        has_generated_similarity_feasible=True,
        similarity_threshold=None,
    ):
        if int(iteration) < int(self.warmup_iterations):
            return "warmup"
        if bool(has_feasible):
            return "refine"
        if bool(has_pair_feasible) or float(best_max_deficit) <= float(
            self.bridge_deficit
        ):
            return "bridge"
        if float(largest_root_fraction) >= float(self.collapse_threshold):
            return "collapsed"
        if int(stagnant_iterations) >= int(self.plateau_patience):
            return "plateau"
        return "search"

    @staticmethod
    def _available(priors, context):
        available = set(dict(context or {}).get("available_operators") or priors)
        return {
            operator: float(weight)
            for operator, weight in priors.items()
            if operator in available and float(weight) > 0.0
        }

    def operator_priors(self, group, state, context=None):
        if group != "proposal":
            return {}
        context = dict(context or {})
        completion = [
            operator
            for operator in context.get("completion_operators", ())
            if operator in set(context.get("available_operators") or ())
        ]

        if state == "warmup":
            return self._available(
                {
                    "legacy": 0.50,
                    "joint_repair": 0.20,
                    "start_repair": 0.15,
                    "lineage_restart": 0.15,
                },
                context,
            )

        if state == "bridge":
            priors = {
                "legacy": 0.10,
                "joint_repair": 0.15,
                "start_repair": 0.05,
                "scaffold_rescue": 0.05,
                "lineage_restart": 0.05,
            }
            if completion:
                share = 0.60 / len(completion)
                for operator in completion:
                    priors[operator] = priors.get(operator, 0.0) + share
            else:
                priors["joint_repair"] += 0.40
                priors["legacy"] += 0.10
                priors["start_repair"] += 0.10
            return self._available(priors, context)

        if not bool(context.get("has_generated_similarity_feasible", True)):
            return self._available(
                {
                    "legacy": 0.30,
                    "scaffold_rescue": 0.25,
                    "joint_repair": 0.20,
                    "start_repair": 0.15,
                    "lineage_restart": 0.10,
                },
                context,
            )

        priors = {
            "search": {
                "legacy": 0.25,
                "start_repair": 0.12,
                "scaffold_rescue": 0.05,
                "dock_refine": 0.05,
                "similarity_repair": 0.10,
                "quality_repair": 0.08,
                "joint_repair": 0.22,
                "lineage_restart": 0.13,
            },
            "refine": {
                "legacy": 0.10,
                "start_repair": 0.05,
                "dock_refine": 0.50,
                "similarity_repair": 0.10,
                "quality_repair": 0.10,
                "joint_repair": 0.10,
                "lineage_restart": 0.05,
            },
            "plateau": {
                "legacy": 0.20,
                "start_repair": 0.12,
                "scaffold_rescue": 0.10,
                "dock_refine": 0.06,
                "similarity_repair": 0.10,
                "quality_repair": 0.08,
                "joint_repair": 0.18,
                "lineage_restart": 0.16,
            },
            "collapsed": {
                "legacy": 0.20,
                "start_repair": 0.15,
                "scaffold_rescue": 0.15,
                "dock_refine": 0.04,
                "similarity_repair": 0.08,
                "quality_repair": 0.06,
                "joint_repair": 0.14,
                "lineage_restart": 0.18,
            },
        }.get(state, {})
        available = self._available(priors, context)
        needs = dict(context.get("constraint_needs") or {})
        return {
            operator: weight
            * (1.0 + 0.50 * self.need_weight * max(0.0, float(needs.get(operator, 0.0))))
            for operator, weight in available.items()
        }

    def operator_floors(self, group, state, context=None):
        if group != "proposal":
            return {}
        context = dict(context or {})
        if state == "warmup":
            floors = {"legacy": 0.50}
        elif state == "bridge":
            completion = [
                operator
                for operator in context.get("completion_operators", ())
                if operator in set(context.get("available_operators") or ())
            ]
            floors = {"legacy": 0.10}
            if completion:
                share = 0.50 / len(completion)
                floors.update({operator: share for operator in completion})
        elif not bool(context.get("has_generated_similarity_feasible", True)):
            floors = {"legacy": 0.25, "scaffold_rescue": 0.15}
        elif state == "refine":
            floors = {"legacy": 0.10}
        else:
            floors = {"legacy": 0.20}
        return self._available(floors, context)

    @staticmethod
    def trust_region_bounds(state, configured_min, configured_max):
        cap = {
            "warmup": 0.22,
            "search": 0.20,
            "bridge": 0.12,
            "refine": 0.10,
            "plateau": 0.24,
            "collapsed": 0.26,
        }.get(state, 0.20)
        lower = max(0.0, float(configured_min))
        return lower, max(lower, min(float(configured_max), cap))

    @staticmethod
    def length_edit_scale(state, context=None):
        context = dict(context or {})
        if state == "bridge":
            return 0.20 if int(context.get("stagnant_iterations", 0)) >= 2 else 0.0
        return {
            "warmup": 0.35,
            "search": 0.35,
            "refine": 0.10,
            "plateau": 0.60,
            "collapsed": 0.70,
        }.get(state, 0.35)


@dataclass(frozen=True)
class RestoredLeadFrontierAdapter(LeadFrontierAdapter):
    """Lead adapter that protects the historical Pareto/fragment sampler.

    The old generator remains a fixed proposal floor in every search state.
    Shared repair operators may use only the residual budget, so online bandit
    updates cannot erase the path that produced the best 25/30 result. All
    values depend on observed feasibility state rather than target identity.
    """

    legacy_floor_warmup: float = 1.00
    legacy_floor_search: float = 0.85
    legacy_floor_bridge: float = 0.70
    legacy_floor_refine: float = 0.85
    legacy_floor_recovery: float = 0.75
    completion_floor: float = 0.15

    def classify(
        self,
        *,
        iteration,
        has_feasible,
        has_pair_feasible,
        best_max_deficit,
        stagnant_iterations,
        largest_root_fraction,
        available_operators=None,
        constraint_needs=None,
        completion_operators=None,
        has_generated_similarity_feasible=True,
        similarity_threshold=None,
    ):
        if int(iteration) < int(self.warmup_iterations):
            return "warmup"
        if bool(has_feasible):
            return "refine"
        if bool(has_pair_feasible) or float(best_max_deficit) <= float(
            self.bridge_deficit
        ):
            return "bridge"
        if float(largest_root_fraction) >= float(self.collapse_threshold):
            return "collapsed"
        if int(stagnant_iterations) >= int(self.plateau_patience):
            return "plateau"
        return "search"

    @staticmethod
    def _available(values, context):
        available = set(dict(context or {}).get("available_operators") or values)
        return {
            operator: float(weight)
            for operator, weight in values.items()
            if operator in available and float(weight) > 0.0
        }

    def operator_priors(self, group, state, context=None):
        if group != "proposal":
            return {}
        context = dict(context or {})
        priors = {
            "warmup": {"legacy": 1.0},
            "search": {
                "legacy": 1.00,
                "start_repair": 0.18,
                "joint_repair": 0.28,
                "lineage_restart": 0.18,
                "similarity_repair": 0.12,
                "quality_repair": 0.12,
                "dock_refine": 0.12,
            },
            "bridge": {
                "legacy": 1.00,
                "start_repair": 0.06,
                "joint_repair": 0.18,
                "lineage_restart": 0.08,
                "similarity_repair": 0.20,
                "quality_repair": 0.20,
                "dock_refine": 0.28,
            },
            "refine": {
                "legacy": 1.00,
                "joint_repair": 0.10,
                "lineage_restart": 0.05,
                "similarity_repair": 0.10,
                "quality_repair": 0.10,
                "dock_refine": 0.55,
            },
            "plateau": {
                "legacy": 1.00,
                "start_repair": 0.18,
                "joint_repair": 0.28,
                "lineage_restart": 0.26,
                "similarity_repair": 0.10,
                "quality_repair": 0.10,
                "dock_refine": 0.08,
            },
            "collapsed": {
                "legacy": 1.00,
                "start_repair": 0.22,
                "joint_repair": 0.24,
                "lineage_restart": 0.30,
                "similarity_repair": 0.08,
                "quality_repair": 0.08,
                "dock_refine": 0.08,
            },
        }.get(state, {"legacy": 1.0})
        priors = self._available(priors, context)
        needs = dict(context.get("constraint_needs") or {})
        completion = set(context.get("completion_operators") or ())
        return {
            operator: weight
            * (1.0 + self.need_weight * max(0.0, float(needs.get(operator, 0.0))))
            * (1.75 if operator in completion else 1.0)
            for operator, weight in priors.items()
        }

    def operator_floors(self, group, state, context=None):
        if group != "proposal":
            return {}
        legacy = {
            "warmup": self.legacy_floor_warmup,
            "search": self.legacy_floor_search,
            "bridge": self.legacy_floor_bridge,
            "refine": self.legacy_floor_refine,
            "plateau": self.legacy_floor_recovery,
            "collapsed": self.legacy_floor_recovery,
        }.get(state, self.legacy_floor_search)
        floors = {"legacy": legacy}
        if state == "bridge":
            completion = [
                operator
                for operator in dict(context or {}).get("completion_operators", ())
                if operator != "legacy"
            ]
            if completion:
                share = min(self.completion_floor, 1.0 - legacy) / len(completion)
                floors.update({operator: share for operator in completion})
        return self._available(floors, context)

    @staticmethod
    def trust_region_bounds(state, configured_min, configured_max):
        cap = {
            "warmup": 0.25,
            "search": 0.22,
            "bridge": 0.14,
            "refine": 0.12,
            "plateau": 0.25,
            "collapsed": 0.28,
        }.get(state, 0.22)
        lower = max(0.0, float(configured_min))
        return lower, max(lower, min(float(configured_max), cap))

    @staticmethod
    def length_edit_scale(state, context=None):
        return {
            "warmup": 0.0,
            "search": 0.20,
            "bridge": 0.05,
            "refine": 0.05,
            "plateau": 0.35,
            "collapsed": 0.45,
        }.get(state, 0.20)


@dataclass(frozen=True)
class LeadBestUnionAdapter(LeadFrontierAdapter):
    """Integrate the strongest lead proposal families without task routing.

    ``legacy`` preserves the broad fragment sampler used by the 25/30 Pareto
    baseline, while ``legacy_local`` supplies the smaller-edit regime that
    recovered complementary strict tasks in later frontier experiments. The
    adapter only sees constraint occupancy, the requested similarity threshold,
    lineage concentration, and online operator rewards. It never sees target or
    seed identity.
    """

    base_floor_warmup: float = 0.60
    base_floor_search: float = 0.50
    base_floor_bridge: float = 0.35
    base_floor_refine: float = 0.45
    base_floor_recovery: float = 0.45
    completion_floor: float = 0.30

    def classify(
        self,
        *,
        iteration,
        has_feasible,
        has_pair_feasible,
        best_max_deficit,
        stagnant_iterations,
        largest_root_fraction,
        available_operators=None,
        constraint_needs=None,
        completion_operators=None,
        has_generated_similarity_feasible=True,
        similarity_threshold=None,
    ):
        if int(iteration) < int(self.warmup_iterations):
            return "warmup"
        if bool(has_feasible):
            return "refine"
        if bool(has_pair_feasible) or float(best_max_deficit) <= float(
            self.bridge_deficit
        ):
            return "bridge"
        if float(largest_root_fraction) >= float(self.collapse_threshold):
            return "collapsed"
        if int(stagnant_iterations) >= int(self.plateau_patience):
            return "plateau"
        return "search"

    @staticmethod
    def _available(values, context):
        available = set(dict(context or {}).get("available_operators") or values)
        return {
            operator: float(weight)
            for operator, weight in values.items()
            if operator in available and float(weight) > 0.0
        }

    @staticmethod
    def _strict_similarity(context):
        threshold = dict(context or {}).get("similarity_threshold")
        return threshold is not None and float(threshold) >= 0.55

    def operator_priors(self, group, state, context=None):
        if group != "proposal":
            return {}
        context = dict(context or {})
        priors = {
            "warmup": {
                "legacy": 0.45,
                "legacy_local": 0.20,
                "start_repair": 0.10,
                "joint_repair": 0.15,
                "lineage_restart": 0.10,
            },
            "search": {
                "legacy": 0.30,
                "legacy_local": 0.20,
                "start_repair": 0.10,
                "dock_refine": 0.06,
                "similarity_repair": 0.07,
                "quality_repair": 0.06,
                "joint_repair": 0.13,
                "lineage_restart": 0.08,
            },
            "bridge": {
                "legacy": 0.18,
                "legacy_local": 0.17,
                "start_repair": 0.05,
                "dock_refine": 0.22,
                "similarity_repair": 0.10,
                "quality_repair": 0.08,
                "joint_repair": 0.12,
                "lineage_restart": 0.08,
            },
            "refine": {
                "legacy": 0.25,
                "legacy_local": 0.20,
                "start_repair": 0.03,
                "dock_refine": 0.27,
                "similarity_repair": 0.07,
                "quality_repair": 0.07,
                "joint_repair": 0.06,
                "lineage_restart": 0.05,
            },
            "plateau": {
                "legacy": 0.25,
                "legacy_local": 0.20,
                "start_repair": 0.10,
                "dock_refine": 0.04,
                "similarity_repair": 0.05,
                "quality_repair": 0.04,
                "joint_repair": 0.15,
                "lineage_restart": 0.17,
            },
            "collapsed": {
                "legacy": 0.25,
                "legacy_local": 0.20,
                "start_repair": 0.13,
                "dock_refine": 0.03,
                "similarity_repair": 0.04,
                "quality_repair": 0.03,
                "joint_repair": 0.14,
                "lineage_restart": 0.18,
            },
        }.get(state, {"legacy": 0.60, "legacy_local": 0.20})

        if self._strict_similarity(context):
            priors["legacy_local"] = priors.get("legacy_local", 0.0) * 1.35
            priors["similarity_repair"] = priors.get("similarity_repair", 0.0) * 1.25
            priors["legacy"] = priors.get("legacy", 0.0) * 0.90

        completion = set(context.get("completion_operators") or ())
        needs = dict(context.get("constraint_needs") or {})
        available = self._available(priors, context)
        return {
            operator: weight
            * (1.0 + 0.75 * self.need_weight * max(0.0, float(needs.get(operator, 0.0))))
            * (1.60 if operator in completion else 1.0)
            for operator, weight in available.items()
        }

    def _base_floors(self, total, context, state):
        strict = self._strict_similarity(context)
        if state == "warmup":
            local_share = 0.34
        else:
            local_share = 0.55 if strict else 0.32
        return {
            "legacy": float(total) * (1.0 - local_share),
            "legacy_local": float(total) * local_share,
        }

    def operator_floors(self, group, state, context=None):
        if group != "proposal":
            return {}
        context = dict(context or {})
        total = {
            "warmup": self.base_floor_warmup,
            "search": self.base_floor_search,
            "bridge": self.base_floor_bridge,
            "refine": self.base_floor_refine,
            "plateau": self.base_floor_recovery,
            "collapsed": self.base_floor_recovery,
        }.get(state, self.base_floor_search)
        floors = self._base_floors(total, context, state)
        if state == "bridge":
            completion = [
                operator
                for operator in context.get("completion_operators", ())
                if operator not in {"legacy", "legacy_local"}
            ]
            if completion:
                share = min(self.completion_floor, 1.0 - total) / len(completion)
                floors.update({operator: share for operator in completion})
        return self._available(floors, context)

    @staticmethod
    def trust_region_bounds(state, configured_min, configured_max):
        cap = {
            "warmup": 0.24,
            "search": 0.21,
            "bridge": 0.12,
            "refine": 0.10,
            "plateau": 0.24,
            "collapsed": 0.27,
        }.get(state, 0.21)
        lower = max(0.0, float(configured_min))
        return lower, max(lower, min(float(configured_max), cap))

    @staticmethod
    def length_edit_scale(state, context=None):
        scale = {
            "warmup": 0.15,
            "search": 0.30,
            "bridge": 0.10,
            "refine": 0.05,
            "plateau": 0.45,
            "collapsed": 0.55,
        }.get(state, 0.25)
        if LeadBestUnionAdapter._strict_similarity(context):
            scale *= 0.50
        return scale


class UnifiedFrontierEngine:
    """Shared online acquisition engine with task-specific adapters.

    The engine never inspects an oracle name. It only sees the observed search
    trajectory, lineage concentration, and operator outcomes supplied by the
    task adapter.
    """

    def __init__(
        self,
        adapter,
        operator_groups: Mapping[str, Iterable[str]],
        bandit_configs=None,
        frontier_min_scale=0.01,
        delta_min_scale=0.03,
    ):
        self.adapter = adapter
        self.frontier_min_scale = max(1e-8, float(frontier_min_scale))
        self.delta_min_scale = max(1e-8, float(delta_min_scale))
        configs = dict(bandit_configs or {})
        self.bandits = {}
        for group, operators in operator_groups.items():
            self.bandits[group] = AdaptiveOperatorBandit(
                operators,
                **dict(configs.get(group, {})),
            )
        self.frontier_history = []
        self.delta_history = []
        self.last_state = "warmup"
        self.last_context = {}

    def classify(self, **context):
        self.last_context = dict(context)
        self.last_state = self.adapter.classify(**context)
        return self.last_state

    def allocate(self, total, state=None, context=None):
        state = state or self.last_state
        context = self.last_context if context is None else dict(context)
        total = max(0, int(total))
        fractions = self.adapter.group_fractions(state, context)
        group_counts = allocate_weighted_counts(total, fractions.items())
        allocation = {}
        for group, count in group_counts.items():
            bandit = self.bandits.get(group)
            if bandit is None:
                continue
            priors = self.adapter.operator_priors(group, state, context)
            floor_provider = getattr(self.adapter, "operator_floors", None)
            floors = (
                floor_provider(group, state, context)
                if callable(floor_provider)
                else {}
            )
            floors = {
                operator: max(0.0, float(fraction))
                for operator, fraction in dict(floors).items()
                if operator in priors and float(fraction) > 0.0
            }
            floor_sum = sum(floors.values())
            if floor_sum > 1.0:
                floors = {
                    operator: fraction / floor_sum
                    for operator, fraction in floors.items()
                }
            fixed = {
                operator: int(math.floor(count * fraction))
                for operator, fraction in floors.items()
            }
            remaining = count - sum(fixed.values())
            residual_priors = {
                operator: max(0.0, float(weight) - floors.get(operator, 0.0))
                for operator, weight in priors.items()
            }
            if not any(residual_priors.values()):
                residual_priors = priors
            adaptive = allocate_weighted_counts(
                remaining,
                bandit.weighted(residual_priors),
            )
            for operator, amount in adaptive.items():
                fixed[operator] = fixed.get(operator, 0) + amount
            allocation[group] = {
                operator: amount
                for operator, amount in fixed.items()
                if amount > 0
            }
        return allocation

    def update_scalar_batch(
        self,
        *,
        group,
        operator,
        scores,
        before_scores,
        before_top10,
        after_top10,
        parent_scores,
    ):
        scores = list(scores)
        parent_scores = list(parent_scores)
        frontier_scale = max(
            self.frontier_min_scale,
            _median(self.frontier_history[-64:]),
        )
        delta_scale = max(
            self.delta_min_scale,
            _median(self.delta_history[-512:]),
        )
        reward, parts = scalar_batch_frontier_reward(
            scores=scores,
            before_scores=before_scores,
            before_top10=before_top10,
            after_top10=after_top10,
            parent_scores=parent_scores,
            frontier_scale=frontier_scale,
            delta_scale=delta_scale,
        )
        self.bandits[group].update(operator, reward, len(scores))
        if parts["frontier_gain"] > 0.0:
            self.frontier_history.append(float(parts["frontier_gain"]))
            del self.frontier_history[:-256]
        for score, parent in zip(scores, parent_scores):
            if parent is not None:
                self.delta_history.append(abs(float(score) - float(parent)))
        del self.delta_history[:-2048]
        return reward, parts

    def update_constrained_batch(
        self,
        *,
        group,
        operator,
        transitions,
        requested=None,
    ):
        transitions = list(transitions)
        reward, parts = self.adapter.batch_reward(transitions)
        requested = len(transitions) if requested is None else max(0, int(requested))
        yield_rate = len(transitions) / max(1, requested)
        yield_rate = max(0.0, min(1.0, yield_rate))
        reward *= math.sqrt(yield_rate)
        parts["yield_rate"] = yield_rate
        self.bandits[group].update(operator, reward, len(transitions))
        return reward, parts

    def delayed_credit(self, group, operator, reward, alpha=0.05):
        bandit = self.bandits.get(group)
        if bandit is not None:
            bandit.delayed_credit(operator, reward, alpha=alpha)

    def snapshot(self):
        return {
            "adapter": type(self.adapter).__name__,
            "state": self.last_state,
            "frontier_scale": max(
                self.frontier_min_scale,
                _median(self.frontier_history[-64:]),
            ),
            "delta_scale": max(
                self.delta_min_scale,
                _median(self.delta_history[-512:]),
            ),
            "bandits": {
                group: bandit.snapshot()
                for group, bandit in sorted(self.bandits.items())
            },
        }

    def state_dict(self):
        return {
            "version": 2,
            "last_state": self.last_state,
            "last_context": self.last_context,
            "frontier_history": [float(value) for value in self.frontier_history],
            "delta_history": [float(value) for value in self.delta_history],
            "bandits": {
                group: bandit.snapshot()
                for group, bandit in sorted(self.bandits.items())
            },
        }

    def load_state_dict(self, state):
        state = dict(state or {})
        if int(state.get("version", 1)) not in {1, 2}:
            raise ValueError("Unsupported UnifiedFrontierEngine state version")
        self.last_state = str(state.get("last_state", "warmup"))
        self.last_context = dict(state.get("last_context", {}))
        self.frontier_history = [
            float(value) for value in state.get("frontier_history", [])
        ][-256:]
        self.delta_history = [
            float(value) for value in state.get("delta_history", [])
        ][-2048:]
        for group, snapshot in dict(state.get("bandits", {})).items():
            if group in self.bandits:
                self.bandits[group].load_snapshot(snapshot)


def _median(values):
    values = sorted(float(value) for value in values)
    if not values:
        return 0.0
    middle = len(values) // 2
    if len(values) % 2:
        return values[middle]
    return 0.5 * (values[middle - 1] + values[middle])
