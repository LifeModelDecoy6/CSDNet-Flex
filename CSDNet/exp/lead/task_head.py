"""Lead-specific task heads layered over the protected frontier engine."""

from __future__ import annotations

from dataclasses import dataclass, field
import math


_ROBUST_PARENT_SCHEDULE = (0, 0, 1, 0, 2)


def _normalized_margin(item, keys):
    normalized = item.get("normalized", {})
    return min(
        (float(normalized.get(key, 0.0)) - 1.0 for key in keys),
        default=-float("inf"),
    )


def robust_completion_parent_rank(
    item,
    operator,
    *,
    minimum_constraint_slack=0.02,
):
    """Rank completion parents without using a target or oracle identity.

    The source archive already determines which three constraints are
    satisfied.  This rank keeps a small safety margin around those constraints
    and then prioritizes the single missing objective.
    """
    operator = {
        "completion_dock_refine": "feasible_dock_polish",
        "completion_similarity_repair": "boundary_similarity_polish",
        "completion_quality_repair": "boundary_quality_polish",
    }.get(str(operator), str(operator))
    minimum_constraint_slack = max(0.0, float(minimum_constraint_slack))
    deficits = item.get("deficits", {})
    residual = float(item.get("residual", float("inf")))
    dock = float(item.get("dock", 0.0))

    if operator == "feasible_dock_polish":
        slack = _normalized_margin(item, ("qed", "sa", "sim"))
        return (
            int(slack >= minimum_constraint_slack),
            dock,
            slack,
            -residual,
        )
    if operator == "boundary_similarity_polish":
        protected_slack = _normalized_margin(item, ("dock", "qed", "sa"))
        return (
            -float(deficits.get("sim", float("inf"))),
            protected_slack,
            dock,
            -residual,
        )
    if operator == "boundary_quality_polish":
        protected_slack = _normalized_margin(item, ("dock", "sim"))
        quality_deficit = max(
            float(deficits.get("qed", 0.0)),
            float(deficits.get("sa", 0.0)),
        )
        return (
            -quality_deficit,
            protected_slack,
            dock,
            -residual,
        )
    return (-residual, dock)


def choose_robust_completion_parent(
    archive,
    operator,
    selection_index,
    *,
    top_k=3,
    minimum_constraint_slack=0.02,
):
    """Select from a tiny robust elite with a deterministic exploit schedule."""
    if not archive:
        return None, None
    ranked = sorted(
        archive,
        key=lambda item: robust_completion_parent_rank(
            item,
            operator,
            minimum_constraint_slack=minimum_constraint_slack,
        ),
        reverse=True,
    )[: max(1, int(top_k))]
    scheduled_rank = _ROBUST_PARENT_SCHEDULE[
        int(selection_index) % len(_ROBUST_PARENT_SCHEDULE)
    ]
    selected_rank = min(scheduled_rank, len(ranked) - 1)
    return ranked[selected_rank], selected_rank


@dataclass
class ProtectedCompletionLeadHead:
    """Late completion policy for constrained lead optimization.

    The shared frontier engine remains task agnostic.  This head interprets
    Lead's explicit docking, similarity, QED, and SA constraints and reserves
    only a bounded late-search budget.  The fragment baseline owns every other
    proposal slot.
    """

    start_iteration: int = 6
    bridge_share: float = 0.12
    dock_polish_share: float = 0.18
    boundary_polish_share: float = 0.12
    boundary_tolerance: float = 0.02
    _bridge_probe_used: bool = field(default=False, init=False, repr=False)

    def classify(
        self,
        *,
        iteration,
        official_feasible,
        pair_presence,
        has_loose_feasible=False,
        boundary_mode=None,
        boundary_deficit=None,
        **_,
    ):
        if bool(official_feasible):
            return "locked"
        if int(iteration) <= int(self.start_iteration):
            return "baseline"

        pair_count = sum(bool(value) for value in dict(pair_presence).values())
        if pair_count >= 2 and not self._bridge_probe_used:
            self._bridge_probe_used = True
            return "bridge_probe"

        if bool(has_loose_feasible):
            return "dock_polish"

        deficit = float("inf") if boundary_deficit is None else float(boundary_deficit)
        if deficit <= max(0.0, float(self.boundary_tolerance)):
            if boundary_mode == "similarity":
                return "similarity_boundary"
            if boundary_mode == "quality":
                return "quality_boundary"
        return "baseline"

    def reserve_fraction(self, state, context=None):
        return {
            "bridge_probe": self.bridge_share,
            "dock_polish": self.dock_polish_share,
            "similarity_boundary": self.boundary_polish_share,
            "quality_boundary": self.boundary_polish_share,
        }.get(str(state), 0.0)

    @staticmethod
    def group_fractions(state, context=None):
        return {"proposal": 1.0}

    @staticmethod
    def operator_priors(group, state, context=None):
        if group != "proposal":
            return {}
        return {
            "bridge_probe": {"pair_bridge": 1.0},
            "dock_polish": {"feasible_dock_polish": 1.0},
            "similarity_boundary": {"boundary_similarity_polish": 1.0},
            "quality_boundary": {"boundary_quality_polish": 1.0},
        }.get(str(state), {})


@dataclass
class ReversibleRouteCompletionLeadHead(ProtectedCompletionLeadHead):
    """Preserve V1, then test one evidence-selected completion route.

    The V1 policy remains authoritative through ``late_start_iteration``.  For
    an unresolved task, the final two proposal rounds may reserve a bounded
    budget for the existing operator associated with the closest three-of-four
    constraint frontier.  The reserve expands only if the best normalized
    completion gap improves; otherwise allocation returns to V1.
    """

    late_start_iteration: int = 8
    probe_share: float = 0.20
    commit_share: float = 0.30
    max_route_deficit: float = 0.16
    route_tie_tolerance: float = 0.015
    min_absolute_improvement: float = 0.001
    min_relative_improvement: float = 0.05
    _probe_best_deficit: float | None = field(default=None, init=False, repr=False)
    _selected_routes: tuple[str, ...] = field(default=(), init=False, repr=False)

    @staticmethod
    def _finite_routes(route_deficits, max_deficit):
        routes = {}
        for name, value in dict(route_deficits or {}).items():
            value = float(value)
            if value >= 0.0 and value <= float(max_deficit):
                routes[str(name)] = value
        return routes

    def _select_routes(self, routes):
        if not routes:
            return ()
        best = min(routes.values())
        tolerance = max(0.0, float(self.route_tie_tolerance))
        return tuple(
            sorted(name for name, value in routes.items() if value <= best + tolerance)
        )

    def classify(
        self,
        *,
        iteration,
        official_feasible,
        route_deficits=None,
        **context,
    ):
        if bool(official_feasible):
            return "locked"

        iteration = int(iteration)
        if iteration < int(self.late_start_iteration):
            return super().classify(
                iteration=iteration,
                official_feasible=False,
                **context,
            )

        routes = self._finite_routes(route_deficits, self.max_route_deficit)
        if self._probe_best_deficit is None:
            if not routes:
                return super().classify(
                    iteration=iteration,
                    official_feasible=False,
                    **context,
                )
            self._probe_best_deficit = min(routes.values())
            self._selected_routes = self._select_routes(routes)
            return "late_route_probe"

        if routes:
            current_best = min(routes.values())
            required = max(
                max(0.0, float(self.min_absolute_improvement)),
                max(0.0, float(self.min_relative_improvement))
                * self._probe_best_deficit,
            )
            if current_best <= self._probe_best_deficit - required:
                self._selected_routes = self._select_routes(routes)
                return "late_route_commit"

        return super().classify(
            iteration=iteration,
            official_feasible=False,
            **context,
        )

    def reserve_fraction(self, state, context=None):
        if state == "late_route_probe":
            return float(self.probe_share)
        if state == "late_route_commit":
            return float(self.commit_share)
        return super().reserve_fraction(state, context)

    def operator_priors(self, group, state, context=None):
        if group != "proposal":
            return {}
        if state in {"late_route_probe", "late_route_commit"}:
            mapping = {
                "dock": "dock_refine",
                "similarity": "similarity_repair",
                "quality": "quality_repair",
            }
            return {
                mapping[route]: 1.0
                for route in self._selected_routes
                if route in mapping
            }
        return super().operator_priors(group, state, context)


@dataclass
class AnchoredRestartCompletionLeadHead(ReversibleRouteCompletionLeadHead):
    """Extend V3 only for late searches that never reached a pair frontier.

    Existing V3 behavior remains authoritative whenever an ``sq``, ``qd`` or
    ``sd`` frontier already exists.  A task with no pair frontier at the late
    anchor boundary receives one small, seed-anchored probe.  If that probe
    creates a pair frontier, normal V3 completion handles a close route; a
    bounded relaxed-route probe is available only when the new route is still
    outside V3's standard completion radius.
    """

    anchor_start_iteration: int = 8
    anchor_probe_share: float = 0.08
    anchor_route_probe_share: float = 0.12
    anchor_route_commit_share: float = 0.20
    anchor_route_max_deficit: float = 0.35
    _anchor_probe_started: bool = field(default=False, init=False, repr=False)
    _anchor_route_best_deficit: float | None = field(
        default=None,
        init=False,
        repr=False,
    )
    _anchor_selected_routes: tuple[str, ...] = field(
        default=(),
        init=False,
        repr=False,
    )

    def classify(
        self,
        *,
        iteration,
        official_feasible,
        pair_presence,
        route_deficits=None,
        **context,
    ):
        if bool(official_feasible):
            return "locked"

        iteration = int(iteration)
        pair_count = sum(bool(value) for value in dict(pair_presence).values())
        routes = {
            str(name): float(value)
            for name, value in dict(route_deficits or {}).items()
            if float(value) >= 0.0
        }

        # This is the only V4-only entry point.  It is one shot and leaves all
        # tasks that already have a completion frontier exactly on V3.
        if (
            iteration >= int(self.anchor_start_iteration)
            and not self._anchor_probe_started
            and pair_count == 0
        ):
            self._anchor_probe_started = True
            return "late_anchor_probe"

        if self._anchor_probe_started and pair_count > 0:
            standard_routes = self._finite_routes(routes, self.max_route_deficit)
            if standard_routes:
                return super().classify(
                    iteration=iteration,
                    official_feasible=False,
                    pair_presence=pair_presence,
                    route_deficits=routes,
                    **context,
                )

            relaxed_routes = self._finite_routes(
                routes,
                self.anchor_route_max_deficit,
            )
            if self._anchor_route_best_deficit is None and relaxed_routes:
                self._anchor_route_best_deficit = min(relaxed_routes.values())
                self._anchor_selected_routes = self._select_routes(relaxed_routes)
                return "late_anchor_route_probe"

            if relaxed_routes and self._anchor_route_best_deficit is not None:
                current_best = min(relaxed_routes.values())
                required = max(
                    max(0.0, float(self.min_absolute_improvement)),
                    max(0.0, float(self.min_relative_improvement))
                    * self._anchor_route_best_deficit,
                )
                if current_best <= self._anchor_route_best_deficit - required:
                    self._anchor_selected_routes = self._select_routes(relaxed_routes)
                    return "late_anchor_route_commit"

        return super().classify(
            iteration=iteration,
            official_feasible=False,
            pair_presence=pair_presence,
            route_deficits=routes,
            **context,
        )

    def reserve_fraction(self, state, context=None):
        return {
            "late_anchor_probe": self.anchor_probe_share,
            "late_anchor_route_probe": self.anchor_route_probe_share,
            "late_anchor_route_commit": self.anchor_route_commit_share,
        }.get(str(state), super().reserve_fraction(state, context))

    def operator_priors(self, group, state, context=None):
        if group != "proposal":
            return {}
        if state == "late_anchor_probe":
            return {"no_pair_anchor_restart": 1.0}
        if state in {"late_anchor_route_probe", "late_anchor_route_commit"}:
            mapping = {
                "dock": "dock_refine",
                "similarity": "similarity_repair",
                "quality": "quality_repair",
            }
            return {
                mapping[route]: 1.0
                for route in self._anchor_selected_routes
                if route in mapping
            }
        return super().operator_priors(group, state, context)


@dataclass
class ProtectedRoutePortfolioLeadHead(ProtectedCompletionLeadHead):
    """Preserve V1 and diversify only the final unresolved completion routes.

    V3 committed its whole reserve to one closest route, although several
    three-of-four frontiers were often simultaneously useful.  This head keeps
    every eligible route alive with a small probability floor.  It is purely
    constraint driven: target and oracle identities are never inspected.
    """

    late_start_iteration: int = 8
    portfolio_share: float = 0.20
    max_route_deficit: float = 0.21
    route_temperature: float = 0.08
    minimum_route_weight: float = 0.25
    seed_probe_share: float = 0.08
    _route_weights: dict[str, float] = field(default_factory=dict, init=False, repr=False)
    _seed_probe_used: bool = field(default=False, init=False, repr=False)

    def _eligible_route_weights(self, route_deficits):
        routes = {
            str(name): float(value)
            for name, value in dict(route_deficits or {}).items()
            if 0.0 <= float(value) <= float(self.max_route_deficit)
        }
        if not routes:
            return {}
        best = min(routes.values())
        temperature = max(1e-6, float(self.route_temperature))
        floor = max(0.0, float(self.minimum_route_weight))
        return {
            name: max(floor, math.exp(-(value - best) / temperature))
            for name, value in routes.items()
        }

    def classify(
        self,
        *,
        iteration,
        official_feasible,
        pair_presence,
        route_deficits=None,
        **context,
    ):
        if bool(official_feasible):
            self._route_weights = {}
            return "locked"

        iteration = int(iteration)
        if iteration < int(self.late_start_iteration):
            self._route_weights = {}
            return super().classify(
                iteration=iteration,
                official_feasible=False,
                pair_presence=pair_presence,
                **context,
            )

        self._route_weights = self._eligible_route_weights(route_deficits)
        if self._route_weights:
            return "late_route_portfolio"

        pair_count = sum(bool(value) for value in dict(pair_presence).values())
        if pair_count == 0 and not self._seed_probe_used:
            self._seed_probe_used = True
            return "late_seed_probe"

        return super().classify(
            iteration=iteration,
            official_feasible=False,
            pair_presence=pair_presence,
            **context,
        )

    def reserve_fraction(self, state, context=None):
        if state == "late_route_portfolio":
            return float(self.portfolio_share)
        if state == "late_seed_probe":
            return float(self.seed_probe_share)
        return super().reserve_fraction(state, context)

    def operator_priors(self, group, state, context=None):
        if group != "proposal":
            return {}
        if state == "late_route_portfolio":
            mapping = {
                "dock": "completion_dock_refine",
                "similarity": "completion_similarity_repair",
                "quality": "completion_quality_repair",
            }
            return {
                mapping[route]: weight
                for route, weight in self._route_weights.items()
                if route in mapping
            }
        if state == "late_seed_probe":
            return {"completion_seed_restart": 1.0}
        return super().operator_priors(group, state, context)
