"""Baseline-preserving policy composition for constrained molecular search.

The shared engine deliberately does less than :class:`UnifiedFrontierEngine`:
it never replaces a validated policy.  A task head may reserve a bounded part
of the budget, while the remainder is allocated by the historical baseline.
This makes the fallback path explicit and keeps task semantics in small heads.
"""

from __future__ import annotations

from dataclasses import dataclass
import math
from typing import Callable, Mapping

from CSDNet.optim.frontier import allocate_weighted_counts


def _merge_nested_counts(*allocations):
    merged = {}
    for allocation in allocations:
        for group, counts in dict(allocation or {}).items():
            target = merged.setdefault(str(group), {})
            for operator, count in dict(counts or {}).items():
                target[str(operator)] = target.get(str(operator), 0) + int(count)
    return {
        group: {operator: count for operator, count in counts.items() if count > 0}
        for group, counts in merged.items()
    }


class BaselineProtectedFrontierEngine:
    """Compose a proven baseline allocator with a bounded task-head reserve."""

    def __init__(self, task_head):
        self.task_head = task_head
        self.last_state = "baseline"
        self.last_context = {}
        self.last_reserve = 0

    def classify(self, **context):
        self.last_context = dict(context)
        self.last_state = str(self.task_head.classify(**context))
        return self.last_state

    def reserve_fraction(self, state=None, context=None):
        state = self.last_state if state is None else str(state)
        context = self.last_context if context is None else dict(context)
        value = float(self.task_head.reserve_fraction(state, context))
        return max(0.0, min(1.0, value))

    def observe_batch(self, **evidence):
        """Forward optional online evidence without changing allocation policy."""
        observer = getattr(self.task_head, "observe_batch", None)
        if observer is None:
            return None
        return observer(**evidence)

    def allocate(
        self,
        total: int,
        baseline_allocator: Callable[[int], Mapping[str, Mapping[str, int]]],
        *,
        state=None,
        context=None,
        available=None,
    ):
        """Allocate ``total`` slots while preserving the configured base floor.

        ``baseline_allocator`` remains the authority for the protected share.
        The task head only allocates its reserve among currently available
        operators.  If no reserve operator is available, every slot returns to
        the baseline.
        """
        total = max(0, int(total))
        state = self.last_state if state is None else str(state)
        context = self.last_context if context is None else dict(context)
        reserve = int(round(total * self.reserve_fraction(state, context)))

        group_weights = dict(self.task_head.group_fractions(state, context))
        allowed = {
            str(group): set(operators)
            for group, operators in dict(available or {}).items()
        }
        reserve_allocation = {}
        if reserve > 0 and group_weights:
            group_counts = allocate_weighted_counts(reserve, group_weights.items())
            for group, group_count in group_counts.items():
                priors = dict(self.task_head.operator_priors(group, state, context))
                if group in allowed:
                    priors = {
                        operator: weight
                        for operator, weight in priors.items()
                        if operator in allowed[group]
                    }
                operator_counts = allocate_weighted_counts(
                    group_count,
                    priors.items(),
                )
                if operator_counts:
                    reserve_allocation[group] = operator_counts

        reserve_used = sum(
            count
            for counts in reserve_allocation.values()
            for count in counts.values()
        )
        self.last_reserve = reserve_used
        baseline_allocation = baseline_allocator(total - reserve_used)
        allocation = _merge_nested_counts(baseline_allocation, reserve_allocation)
        allocated = sum(
            count for counts in allocation.values() for count in counts.values()
        )
        if allocated != total:
            raise RuntimeError(
                f"Protected allocation lost budget: requested={total}, allocated={allocated}"
            )
        return allocation


@dataclass(frozen=True)
class SafeLeadFrontierHead:
    """Late recovery head for the SAFE-GPT universal-frontier lead policy."""

    rescue_start_iteration: int = 6
    reserve_share: float = 0.75

    def classify(
        self,
        *,
        iteration,
        official_feasible,
        has_generated_similarity,
        has_stage_three,
        pair_presence,
        **_,
    ):
        if bool(official_feasible):
            return "locked"
        if int(iteration) <= int(self.rescue_start_iteration):
            return "baseline"
        if not bool(has_generated_similarity):
            return "seed_anchor"
        if any(bool(value) for value in dict(pair_presence).values()):
            return "completion"
        if bool(has_stage_three):
            return "bridge"
        return "explore"

    def reserve_fraction(self, state, context=None):
        if state in {"baseline", "locked"}:
            return 0.0
        return float(self.reserve_share)

    @staticmethod
    def group_fractions(state, context=None):
        return {"proposal": 1.0}

    @staticmethod
    def operator_priors(group, state, context=None):
        if group != "proposal":
            return {}
        context = dict(context or {})
        if state == "seed_anchor":
            return {
                "start_repair": 0.55,
                "joint_repair": 0.30,
                "similarity_repair": 0.15,
            }
        if state == "completion":
            pair_presence = dict(context.get("pair_presence") or {})
            completion = {
                "dock_refine": 1.0 if pair_presence.get("sq") else 0.0,
                "similarity_repair": 1.0 if pair_presence.get("qd") else 0.0,
                "quality_repair": 1.0 if pair_presence.get("sd") else 0.0,
            }
            active = sum(completion.values())
            priors = {
                operator: 0.65 * weight / max(1.0, active)
                for operator, weight in completion.items()
                if weight > 0.0
            }
            priors.update({"joint_repair": 0.20, "start_repair": 0.15})
            return priors
        if state == "bridge":
            return {
                "joint_repair": 0.45,
                "start_repair": 0.25,
                "dock_refine": 0.10,
                "similarity_repair": 0.10,
                "quality_repair": 0.10,
            }
        return {
            "joint_repair": 0.45,
            "start_repair": 0.35,
            "dock_refine": 0.08,
            "similarity_repair": 0.06,
            "quality_repair": 0.06,
        }


@dataclass(frozen=True)
class SafeLeadBridgeHead:
    """A small late bridge reserve that never replaces the lead baseline.

    The bridge is available only when two distinct pairwise-feasible frontiers
    have actually been observed. Tasks without that evidence receive the exact
    historical universal-frontier allocation.
    """

    start_iteration: int = 6
    reserve_share: float = 0.12

    def classify(
        self,
        *,
        iteration,
        official_feasible,
        pair_presence,
        **_,
    ):
        if bool(official_feasible):
            return "locked"
        if int(iteration) <= int(self.start_iteration):
            return "baseline"
        pair_count = sum(bool(value) for value in dict(pair_presence).values())
        return "bridge" if pair_count >= 2 else "baseline"

    def reserve_fraction(self, state, context=None):
        return float(self.reserve_share) if state == "bridge" else 0.0

    @staticmethod
    def group_fractions(state, context=None):
        return {"proposal": 1.0}

    @staticmethod
    def operator_priors(group, state, context=None):
        if group != "proposal" or state != "bridge":
            return {}
        return {"pair_bridge": 1.0}


@dataclass(frozen=True)
class SafePMOFrontierHead:
    """Small exploitation reserve layered on the SAFE-GPT PMO V9 policy."""

    warmup_calls: int = 1000
    search_reserve: float = 0.12
    saturated_reserve: float = 0.08

    @staticmethod
    def classify(*, state, **_):
        return str(state)

    def reserve_fraction(self, state, context=None):
        context = dict(context or {})
        if int(context.get("calls", 0)) < int(self.warmup_calls):
            return 0.0
        if state == "search":
            return float(self.search_reserve)
        if state == "saturated":
            return float(self.saturated_reserve)
        return 0.0

    @staticmethod
    def group_fractions(state, context=None):
        return {"local": 1.0}

    @staticmethod
    def operator_priors(group, state, context=None):
        if group != "local":
            return {}
        return {
            "elite_tiny": 0.42,
            "elite_small": 0.42,
            "elite_medium": 0.10,
            "graph_shrink": 0.06,
        }


class EvidenceGatedPMOHead:
    """Probe local exploitation, then expand it only with online evidence.

    Decisions use the same scalar batch-frontier reward for every PMO oracle.
    Oracle names and hand-written task classes are deliberately absent.
    """

    target_operators = frozenset({"elite_tiny", "elite_small"})

    def __init__(
        self,
        *,
        warmup_calls=1000,
        probe_fraction=0.04,
        maximum_fraction=0.12,
        window_calls=500,
        positive_margin=0.02,
        negative_margin=0.01,
        reprobe_calls=1000,
    ):
        self.warmup_calls = max(0, int(warmup_calls))
        self.probe_fraction = max(0.0, min(1.0, float(probe_fraction)))
        self.maximum_fraction = max(
            self.probe_fraction,
            min(1.0, float(maximum_fraction)),
        )
        self.window_calls = max(1, int(window_calls))
        self.positive_margin = max(0.0, float(positive_margin))
        self.negative_margin = max(0.0, float(negative_margin))
        self.reprobe_calls = max(1, int(reprobe_calls))
        self.current_reserve = 0.0
        self.next_probe_call = self.warmup_calls
        self.phase = "warmup"
        self.last_advantage = 0.0
        self.last_uncertainty = 0.0
        self.decisions = []
        self._reset_window()

    def _reset_window(self):
        self.window = {
            "target_reward": 0.0,
            "target_reward_sq": 0.0,
            "target_evaluated": 0,
            "target_batches": 0,
            "reference_reward": 0.0,
            "reference_reward_sq": 0.0,
            "reference_evaluated": 0,
            "reference_batches": 0,
        }

    @staticmethod
    def classify(*, state, **_):
        return str(state)

    def _maybe_start_probe(self, calls):
        calls = int(calls)
        if calls < self.warmup_calls:
            self.phase = "warmup"
            self.current_reserve = 0.0
            return
        if self.current_reserve <= 0.0 and calls >= self.next_probe_call:
            self.current_reserve = self.probe_fraction
            self.phase = "probe"
            self._reset_window()

    def reserve_fraction(self, state, context=None):
        context = dict(context or {})
        calls = int(context.get("calls", 0))
        self._maybe_start_probe(calls)
        if state != "search":
            return 0.0
        return float(self.current_reserve)

    @staticmethod
    def group_fractions(state, context=None):
        return {"local": 1.0}

    @staticmethod
    def operator_priors(group, state, context=None):
        if group != "local":
            return {}
        return {"elite_tiny": 0.50, "elite_small": 0.50}

    def observe_batch(self, *, operator, reward, evaluated, calls, state, **_):
        """Update one evidence window after a scored V9 operator batch."""
        calls = int(calls)
        evaluated = max(0, int(evaluated))
        if calls <= self.warmup_calls or evaluated <= 0 or state != "search":
            return
        self._maybe_start_probe(calls)
        prefix = (
            "target" if str(operator) in self.target_operators else "reference"
        )
        reward = float(reward)
        self.window[f"{prefix}_reward"] += reward
        self.window[f"{prefix}_reward_sq"] += reward * reward
        self.window[f"{prefix}_evaluated"] += evaluated
        self.window[f"{prefix}_batches"] += 1
        total = (
            self.window["target_evaluated"]
            + self.window["reference_evaluated"]
        )
        if total < self.window_calls:
            return

        target_n = self.window["target_batches"]
        reference_n = self.window["reference_batches"]
        if target_n < 2 or reference_n < 2:
            return
        target_mean = self.window["target_reward"] / target_n
        reference_mean = self.window["reference_reward"] / reference_n
        advantage = target_mean - reference_mean

        def sample_variance(prefix, count, mean):
            if count <= 1:
                return 0.0
            centered = self.window[f"{prefix}_reward_sq"] - count * mean * mean
            return max(0.0, centered / (count - 1))

        target_variance = sample_variance("target", target_n, target_mean)
        reference_variance = sample_variance(
            "reference", reference_n, reference_mean
        )
        uncertainty = max(
            0.01,
            math.sqrt(
                target_variance / target_n
                + reference_variance / reference_n
            ),
        )
        self.last_advantage = float(advantage)
        self.last_uncertainty = float(uncertainty)

        if advantage > self.positive_margin + uncertainty:
            self.current_reserve = min(
                self.maximum_fraction,
                max(self.probe_fraction, self.current_reserve + 0.04),
            )
            self.phase = "expanded"
        elif advantage < -self.negative_margin - uncertainty:
            self.current_reserve = 0.0
            self.next_probe_call = calls + self.reprobe_calls
            self.phase = "baseline"
        else:
            self.current_reserve = self.probe_fraction
            self.phase = "probe"
        self.decisions.append(
            {
                "calls": calls,
                "target_mean": target_mean,
                "reference_mean": reference_mean,
                "target_batches": target_n,
                "reference_batches": reference_n,
                "advantage": advantage,
                "uncertainty": uncertainty,
                "reserve": self.current_reserve,
                "phase": self.phase,
            }
        )
        self.decisions = self.decisions[-20:]
        self._reset_window()

    def snapshot(self):
        return {
            "phase": self.phase,
            "reserve": float(self.current_reserve),
            "next_probe_call": int(self.next_probe_call),
            "last_advantage": float(self.last_advantage),
            "last_uncertainty": float(self.last_uncertainty),
            "window": dict(self.window),
            "decisions": list(self.decisions),
        }

    def state_dict(self):
        return self.snapshot()

    def load_state_dict(self, state):
        state = dict(state or {})
        self.phase = str(state.get("phase", self.phase))
        self.current_reserve = max(
            0.0,
            min(self.maximum_fraction, float(state.get("reserve", 0.0))),
        )
        self.next_probe_call = int(
            state.get("next_probe_call", self.next_probe_call)
        )
        self.last_advantage = float(
            state.get("last_advantage", self.last_advantage)
        )
        self.last_uncertainty = float(
            state.get("last_uncertainty", self.last_uncertainty)
        )
        restored_window = dict(state.get("window") or {})
        self._reset_window()
        for key in self.window:
            if key in restored_window:
                self.window[key] = restored_window[key]
        self.decisions = list(state.get("decisions") or [])[-20:]


class ReversibleEvidencePMOHead:
    """Conservative PMO reserve controlled by call-normalized evidence.

    The head is oracle agnostic.  It probes the same two local operators as the
    original evidence gate, but promotion requires repeated evidence that they
    move the actual top-10 frontier efficiently per oracle call.  Weak or stale
    evidence returns the complete budget to the historical V9 allocator.
    """

    target_operators = frozenset({"elite_tiny", "elite_small"})

    def __init__(
        self,
        *,
        warmup_calls=1000,
        probe_fraction=0.04,
        maximum_fraction=0.12,
        window_calls=500,
        promotion_windows=2,
        neutral_patience=2,
        confidence_z=1.2816,
        frontier_margin=0.05,
        entry_tolerance=0.002,
        minimum_target_calls=40,
        minimum_reference_calls=160,
        reprobe_calls=1000,
        state_reprobe_calls=250,
    ):
        self.warmup_calls = max(0, int(warmup_calls))
        self.probe_fraction = max(0.0, min(1.0, float(probe_fraction)))
        self.maximum_fraction = max(
            self.probe_fraction,
            min(1.0, float(maximum_fraction)),
        )
        self.window_calls = max(1, int(window_calls))
        self.promotion_windows = max(1, int(promotion_windows))
        self.neutral_patience = max(1, int(neutral_patience))
        self.confidence_z = max(0.0, float(confidence_z))
        self.frontier_margin = max(0.0, float(frontier_margin))
        self.entry_tolerance = max(0.0, float(entry_tolerance))
        self.minimum_target_calls = max(1, int(minimum_target_calls))
        self.minimum_reference_calls = max(1, int(minimum_reference_calls))
        self.reprobe_calls = max(1, int(reprobe_calls))
        self.state_reprobe_calls = max(0, int(state_reprobe_calls))
        self.current_reserve = 0.0
        self.effective_reserve = 0.0
        self.next_probe_call = self.warmup_calls
        self.phase = "warmup"
        self.last_state = "warmup"
        self.positive_streak = 0
        self.neutral_streak = 0
        self.last_advantage = 0.0
        self.last_uncertainty = 0.0
        self.last_frontier_advantage = 0.0
        self.decisions = []
        self._reset_window()

    def _reset_window(self):
        self.window = {
            "target_evaluated": 0,
            "target_batches": 0,
            "target_entries": 0.0,
            "target_frontier_gain": 0.0,
            "target_reward": 0.0,
            "reference_evaluated": 0,
            "reference_batches": 0,
            "reference_entries": 0.0,
            "reference_frontier_gain": 0.0,
            "reference_reward": 0.0,
        }

    @staticmethod
    def classify(*, state, **_):
        return str(state)

    def _suspend(self, calls, state):
        if self.last_state == "search" or self.current_reserve > 0.0:
            self.next_probe_call = max(
                self.next_probe_call,
                int(calls) + self.state_reprobe_calls,
            )
        self.current_reserve = 0.0
        self.effective_reserve = 0.0
        self.phase = f"suspended_{state}"
        self.positive_streak = 0
        self.neutral_streak = 0
        self._reset_window()

    def _maybe_start_probe(self, calls):
        calls = int(calls)
        if calls < self.warmup_calls:
            self.current_reserve = 0.0
            self.effective_reserve = 0.0
            self.phase = "warmup"
            return
        if self.current_reserve <= 0.0 and calls >= self.next_probe_call:
            self.current_reserve = self.probe_fraction
            self.phase = "probe"
            self.positive_streak = 0
            self.neutral_streak = 0
            self._reset_window()

    def reserve_fraction(self, state, context=None):
        context = dict(context or {})
        calls = int(context.get("calls", 0))
        state = str(state)
        if state != "search":
            self._suspend(calls, state)
            self.last_state = state
            return 0.0
        self._maybe_start_probe(calls)
        self.last_state = state
        self.effective_reserve = float(self.current_reserve)
        return self.effective_reserve

    @staticmethod
    def group_fractions(state, context=None):
        return {"local": 1.0}

    @staticmethod
    def operator_priors(group, state, context=None):
        if group != "local":
            return {}
        return {"elite_tiny": 0.50, "elite_small": 0.50}

    @staticmethod
    def _posterior_entry_rate(entries, evaluated):
        return (float(entries) + 1.0) / (int(evaluated) + 2.0)

    def _entry_advantage(self):
        target_n = self.window["target_evaluated"]
        reference_n = self.window["reference_evaluated"]
        target_rate = self._posterior_entry_rate(
            self.window["target_entries"], target_n
        )
        reference_rate = self._posterior_entry_rate(
            self.window["reference_entries"], reference_n
        )
        uncertainty = math.sqrt(
            target_rate * (1.0 - target_rate) / max(1, target_n + 3)
            + reference_rate
            * (1.0 - reference_rate)
            / max(1, reference_n + 3)
        )
        advantage = target_rate - reference_rate
        lower_bound = advantage - self.confidence_z * uncertainty
        return target_rate, reference_rate, advantage, uncertainty, lower_bound

    def _classify_window(self):
        target_n = self.window["target_evaluated"]
        reference_n = self.window["reference_evaluated"]
        target_frontier = self.window["target_frontier_gain"] / max(1, target_n)
        reference_frontier = self.window["reference_frontier_gain"] / max(
            1, reference_n
        )
        (
            target_entry,
            reference_entry,
            entry_advantage,
            uncertainty,
            entry_lower_bound,
        ) = self._entry_advantage()

        frontier_floor = 1e-12
        frontier_positive = (
            self.window["target_frontier_gain"] > 0.0
            and target_frontier
            >= reference_frontier * (1.0 + self.frontier_margin)
        )
        entry_support = entry_lower_bound >= -self.entry_tolerance
        entry_positive = entry_lower_bound > self.entry_tolerance
        frontier_not_harmful = target_frontier >= (
            reference_frontier * (1.0 - self.frontier_margin)
        )
        positive = frontier_positive and entry_support
        positive = positive or (entry_positive and frontier_not_harmful)

        no_target_frontier = (
            self.window["target_frontier_gain"] <= frontier_floor
            and self.window["reference_frontier_gain"] > frontier_floor
        )
        jointly_worse = (
            target_frontier
            < reference_frontier * (1.0 - self.frontier_margin)
            and entry_lower_bound < -self.entry_tolerance
        )
        if positive:
            verdict = "positive"
        elif no_target_frontier or jointly_worse:
            verdict = "negative"
        else:
            verdict = "neutral"
        return {
            "verdict": verdict,
            "target_entry_rate": target_entry,
            "reference_entry_rate": reference_entry,
            "entry_advantage": entry_advantage,
            "entry_uncertainty": uncertainty,
            "entry_lower_bound": entry_lower_bound,
            "target_frontier_per_call": target_frontier,
            "reference_frontier_per_call": reference_frontier,
            "frontier_advantage": target_frontier - reference_frontier,
        }

    def _return_to_baseline(self, calls, reason):
        self.current_reserve = 0.0
        self.effective_reserve = 0.0
        self.next_probe_call = int(calls) + self.reprobe_calls
        self.phase = f"baseline_{reason}"
        self.positive_streak = 0
        self.neutral_streak = 0

    def observe_batch(
        self,
        *,
        operator,
        reward,
        evaluated,
        calls,
        state,
        frontier_gain=0.0,
        top10_entries=0.0,
        **_,
    ):
        calls = int(calls)
        evaluated = max(0, int(evaluated))
        state = str(state)
        if state != "search" or evaluated <= 0:
            return
        self._maybe_start_probe(calls)
        if self.current_reserve <= 0.0:
            return

        prefix = (
            "target" if str(operator) in self.target_operators else "reference"
        )
        self.window[f"{prefix}_evaluated"] += evaluated
        self.window[f"{prefix}_batches"] += 1
        self.window[f"{prefix}_entries"] += max(0.0, float(top10_entries))
        self.window[f"{prefix}_frontier_gain"] += max(
            0.0, float(frontier_gain)
        )
        self.window[f"{prefix}_reward"] += float(reward)

        total = (
            self.window["target_evaluated"]
            + self.window["reference_evaluated"]
        )
        enough_support = (
            self.window["target_evaluated"] >= self.minimum_target_calls
            and self.window["reference_evaluated"]
            >= self.minimum_reference_calls
        )
        if total < self.window_calls or not enough_support:
            return

        metrics = self._classify_window()
        verdict = metrics["verdict"]
        if verdict == "positive":
            self.positive_streak += 1
            self.neutral_streak = 0
            if self.positive_streak >= self.promotion_windows:
                self.current_reserve = min(
                    self.maximum_fraction,
                    max(
                        self.probe_fraction,
                        self.current_reserve + self.probe_fraction,
                    ),
                )
                self.phase = "expanded"
                self.positive_streak = 0
            else:
                self.phase = "confirm_positive"
        elif verdict == "negative":
            self.positive_streak = 0
            self.neutral_streak = 0
            self.current_reserve = max(
                0.0, self.current_reserve - self.probe_fraction
            )
            if self.current_reserve <= 0.0:
                self._return_to_baseline(calls, "negative")
            else:
                self.phase = "deescalated"
        else:
            self.positive_streak = 0
            self.neutral_streak += 1
            if self.neutral_streak >= self.neutral_patience:
                self.current_reserve = max(
                    0.0, self.current_reserve - self.probe_fraction
                )
                self.neutral_streak = 0
                if self.current_reserve <= 0.0:
                    self._return_to_baseline(calls, "neutral")
                else:
                    self.phase = "deescalated"
            else:
                self.phase = "confirm_neutral"

        self.effective_reserve = float(self.current_reserve)
        self.last_advantage = float(metrics["entry_advantage"])
        self.last_uncertainty = float(metrics["entry_uncertainty"])
        self.last_frontier_advantage = float(metrics["frontier_advantage"])
        self.decisions.append(
            {
                "calls": calls,
                **metrics,
                "target_evaluated": self.window["target_evaluated"],
                "reference_evaluated": self.window["reference_evaluated"],
                "target_entries": self.window["target_entries"],
                "reference_entries": self.window["reference_entries"],
                "reserve": self.current_reserve,
                "phase": self.phase,
            }
        )
        self.decisions = self.decisions[-40:]
        self._reset_window()

    def snapshot(self):
        return {
            "phase": self.phase,
            "reserve": float(self.current_reserve),
            "effective_reserve": float(self.effective_reserve),
            "next_probe_call": int(self.next_probe_call),
            "last_state": self.last_state,
            "positive_streak": int(self.positive_streak),
            "neutral_streak": int(self.neutral_streak),
            "last_advantage": float(self.last_advantage),
            "last_uncertainty": float(self.last_uncertainty),
            "last_frontier_advantage": float(self.last_frontier_advantage),
            "window": dict(self.window),
            "decisions": list(self.decisions),
        }

    def state_dict(self):
        return self.snapshot()

    def load_state_dict(self, state):
        state = dict(state or {})
        self.phase = str(state.get("phase", self.phase))
        self.current_reserve = max(
            0.0,
            min(self.maximum_fraction, float(state.get("reserve", 0.0))),
        )
        self.effective_reserve = max(
            0.0,
            min(
                self.maximum_fraction,
                float(state.get("effective_reserve", self.current_reserve)),
            ),
        )
        self.next_probe_call = max(
            self.warmup_calls,
            int(state.get("next_probe_call", self.next_probe_call)),
        )
        self.last_state = str(state.get("last_state", self.last_state))
        self.positive_streak = max(
            0, int(state.get("positive_streak", self.positive_streak))
        )
        self.neutral_streak = max(
            0, int(state.get("neutral_streak", self.neutral_streak))
        )
        self.last_advantage = float(
            state.get("last_advantage", self.last_advantage)
        )
        self.last_uncertainty = float(
            state.get("last_uncertainty", self.last_uncertainty)
        )
        self.last_frontier_advantage = float(
            state.get(
                "last_frontier_advantage", self.last_frontier_advantage
            )
        )
        saved_window = dict(state.get("window") or {})
        self._reset_window()
        for key in self.window:
            if key in saved_window:
                self.window[key] = saved_window[key]
        self.decisions = list(state.get("decisions") or [])[-40:]
