import random
import unittest

from CSDNet.optim.protected_frontier import (
    BaselineProtectedFrontierEngine,
    EvidenceGatedPMOHead,
    ReversibleEvidencePMOHead,
    SafeLeadBridgeHead,
    SafeLeadFrontierHead,
    SafePMOFrontierHead,
)
from CSDNet.exp.lead.frontier import (
    constraint_state,
    seed_directed_atom_edit_plan,
)
from CSDNet.exp.lead.task_head import (
    AnchoredRestartCompletionLeadHead,
    ProtectedCompletionLeadHead,
    ProtectedRoutePortfolioLeadHead,
    ReversibleRouteCompletionLeadHead,
    choose_robust_completion_parent,
)
from CSDNet.util.tokenizer import tokenize_smiles


def completion_item(dock, qed, sa, similarity):
    state = constraint_state(
        dock=dock,
        qed=qed,
        sa=sa,
        similarity=similarity,
        start_dock=6.4,
        similarity_threshold=0.6,
        docking_margin=0.05,
        residual_l1_weight=0.10,
    )
    return {
        "smiles": f"item-{dock}-{similarity}",
        "dock": dock,
        "qed": qed,
        "sa": sa,
        "sim": similarity,
        **state,
    }


class ProtectedFrontierTest(unittest.TestCase):
    def test_completion_success_is_locked_before_any_reserved_operator(self):
        head = ProtectedCompletionLeadHead()
        engine = BaselineProtectedFrontierEngine(head)
        state = engine.classify(
            iteration=8,
            official_feasible=True,
            pair_presence={"sq": True, "qd": True, "sd": True},
            has_loose_feasible=True,
        )
        self.assertEqual(state, "locked")
        self.assertEqual(engine.reserve_fraction(), 0.0)

    def test_robust_similarity_parent_uses_the_nearest_boundary(self):
        near = completion_item(6.8, 0.63, 0.85, 0.5965)
        distant = completion_item(7.4, 0.72, 0.86, 0.55)
        parent, rank = choose_robust_completion_parent(
            [distant, near],
            "boundary_similarity_polish",
            0,
        )
        self.assertIs(parent, near)
        self.assertEqual(rank, 0)

    def test_robust_dock_parent_requires_constraint_slack(self):
        fragile = completion_item(9.5, 0.601, 0.80, 0.601)
        robust = completion_item(9.0, 0.70, 0.82, 0.68)
        parent, _ = choose_robust_completion_parent(
            [fragile, robust],
            "feasible_dock_polish",
            0,
            minimum_constraint_slack=0.02,
        )
        self.assertIs(parent, robust)

    def test_seed_directed_plan_masks_a_non_common_atom(self):
        plan = seed_directed_atom_edit_plan(
            "CCCO",
            "CCCN",
            random.Random(0),
        )
        self.assertIsNotNone(plan)
        tokens = tokenize_smiles("CCCO")
        self.assertEqual(tokens[plan["start"] : plan["stop"]], ["O"])
        self.assertTrue(plan["seed_directed"])

    def test_completion_head_uses_one_shot_bridge_then_dock_polish(self):
        head = ProtectedCompletionLeadHead()
        engine = BaselineProtectedFrontierEngine(head)
        context = {
            "official_feasible": False,
            "pair_presence": {"sq": True, "qd": True, "sd": False},
            "has_loose_feasible": True,
            "boundary_mode": None,
            "boundary_deficit": None,
        }
        state = engine.classify(iteration=7, **context)
        self.assertEqual(state, "bridge_probe")
        self.assertEqual(engine.reserve_fraction(), 0.12)
        state = engine.classify(iteration=8, **context)
        self.assertEqual(state, "dock_polish")
        self.assertEqual(engine.reserve_fraction(), 0.18)

    def test_completion_head_boundary_polish_is_task_agnostic(self):
        head = ProtectedCompletionLeadHead(boundary_tolerance=0.02)
        state = head.classify(
            iteration=8,
            official_feasible=False,
            pair_presence={"sq": False, "qd": True, "sd": False},
            has_loose_feasible=False,
            boundary_mode="similarity",
            boundary_deficit=0.006,
        )
        self.assertEqual(state, "similarity_boundary")
        self.assertEqual(
            head.operator_priors("proposal", state),
            {"boundary_similarity_polish": 1.0},
        )

    def test_completion_head_rejects_distant_boundary(self):
        head = ProtectedCompletionLeadHead(boundary_tolerance=0.02)
        state = head.classify(
            iteration=8,
            official_feasible=False,
            pair_presence={"sq": False, "qd": True, "sd": False},
            has_loose_feasible=False,
            boundary_mode="similarity",
            boundary_deficit=0.10,
        )
        self.assertEqual(state, "baseline")

    def test_v3_preserves_v1_bridge_through_round_eight(self):
        head = ReversibleRouteCompletionLeadHead(late_start_iteration=8)
        context = {
            "official_feasible": False,
            "pair_presence": {"sq": True, "qd": True, "sd": False},
            "has_loose_feasible": True,
            "boundary_mode": None,
            "boundary_deficit": None,
            "route_deficits": {"dock": 0.08, "similarity": 0.12},
        }
        self.assertEqual(head.classify(iteration=7, **context), "bridge_probe")
        self.assertEqual(head.reserve_fraction("bridge_probe"), 0.12)
        self.assertEqual(head.classify(iteration=8, **context), "late_route_probe")

    def test_v3_splits_probe_between_near_tied_existing_routes(self):
        head = ReversibleRouteCompletionLeadHead(
            late_start_iteration=8,
            route_tie_tolerance=0.015,
        )
        engine = BaselineProtectedFrontierEngine(head)
        state = engine.classify(
            iteration=8,
            official_feasible=False,
            pair_presence={"sq": True, "qd": False, "sd": True},
            has_loose_feasible=True,
            boundary_mode=None,
            boundary_deficit=None,
            route_deficits={"dock": 0.082, "quality": 0.081},
        )
        allocation = engine.allocate(
            100,
            lambda total: {"proposal": {"legacy": total}},
            state=state,
            context=engine.last_context,
            available={"proposal": {"legacy", "dock_refine", "quality_repair"}},
        )
        self.assertEqual(allocation["proposal"]["legacy"], 80)
        self.assertEqual(allocation["proposal"]["dock_refine"], 10)
        self.assertEqual(allocation["proposal"]["quality_repair"], 10)

    def test_v3_commits_only_after_route_gap_improves(self):
        head = ReversibleRouteCompletionLeadHead(late_start_iteration=8)
        context = {
            "official_feasible": False,
            "pair_presence": {"sq": False, "qd": True, "sd": False},
            "has_loose_feasible": False,
            "boundary_mode": "similarity",
            "boundary_deficit": 0.006,
        }
        state = head.classify(
            iteration=8,
            route_deficits={"similarity": 0.006},
            **context,
        )
        self.assertEqual(state, "late_route_probe")
        state = head.classify(
            iteration=9,
            route_deficits={"similarity": 0.004},
            **context,
        )
        self.assertEqual(state, "late_route_commit")
        self.assertEqual(head.reserve_fraction(state), 0.30)

    def test_v3_reverts_to_v1_when_probe_does_not_improve(self):
        head = ReversibleRouteCompletionLeadHead(late_start_iteration=8)
        context = {
            "official_feasible": False,
            "pair_presence": {"sq": False, "qd": True, "sd": False},
            "has_loose_feasible": False,
            "boundary_mode": "similarity",
            "boundary_deficit": 0.08,
        }
        self.assertEqual(
            head.classify(
                iteration=8,
                route_deficits={"similarity": 0.08},
                **context,
            ),
            "late_route_probe",
        )
        self.assertEqual(
            head.classify(
                iteration=9,
                route_deficits={"similarity": 0.08},
                **context,
            ),
            "baseline",
        )

    def test_v4_is_identical_to_v3_when_a_pair_frontier_exists(self):
        head = AnchoredRestartCompletionLeadHead(
            anchor_start_iteration=8,
            late_start_iteration=8,
        )
        context = {
            "official_feasible": False,
            "pair_presence": {"sq": True, "qd": True, "sd": False},
            "has_loose_feasible": True,
            "boundary_mode": None,
            "boundary_deficit": None,
            "route_deficits": {"dock": 0.08, "similarity": 0.12},
        }
        self.assertEqual(head.classify(iteration=7, **context), "bridge_probe")
        self.assertEqual(head.classify(iteration=8, **context), "late_route_probe")

    def test_v4_uses_one_small_anchor_probe_only_without_pair_frontiers(self):
        head = AnchoredRestartCompletionLeadHead(
            anchor_start_iteration=8,
            anchor_probe_share=0.08,
        )
        context = {
            "official_feasible": False,
            "pair_presence": {"sq": False, "qd": False, "sd": False},
            "has_loose_feasible": False,
            "boundary_mode": None,
            "boundary_deficit": None,
            "route_deficits": {},
        }
        self.assertEqual(head.classify(iteration=7, **context), "baseline")
        state = head.classify(iteration=8, **context)
        self.assertEqual(state, "late_anchor_probe")
        self.assertEqual(head.reserve_fraction(state), 0.08)
        self.assertEqual(
            head.operator_priors("proposal", state),
            {"no_pair_anchor_restart": 1.0},
        )
        self.assertEqual(head.classify(iteration=9, **context), "baseline")

    def test_v4_returns_to_standard_v3_after_anchor_creates_close_route(self):
        head = AnchoredRestartCompletionLeadHead(
            anchor_start_iteration=8,
            late_start_iteration=8,
        )
        no_pair = {
            "official_feasible": False,
            "pair_presence": {"sq": False, "qd": False, "sd": False},
            "has_loose_feasible": False,
            "boundary_mode": None,
            "boundary_deficit": None,
            "route_deficits": {},
        }
        self.assertEqual(head.classify(iteration=8, **no_pair), "late_anchor_probe")
        state = head.classify(
            iteration=9,
            official_feasible=False,
            pair_presence={"sq": False, "qd": True, "sd": False},
            has_loose_feasible=False,
            boundary_mode="similarity",
            boundary_deficit=0.10,
            route_deficits={"similarity": 0.10},
        )
        self.assertEqual(state, "late_route_probe")
        self.assertEqual(head.reserve_fraction(state), 0.20)

    def test_v4_relaxed_anchor_route_is_reversible(self):
        head = AnchoredRestartCompletionLeadHead(
            anchor_start_iteration=8,
            late_start_iteration=8,
            anchor_route_max_deficit=0.35,
            anchor_route_probe_share=0.12,
            anchor_route_commit_share=0.20,
        )
        no_pair = {
            "official_feasible": False,
            "pair_presence": {"sq": False, "qd": False, "sd": False},
            "has_loose_feasible": False,
            "boundary_mode": None,
            "boundary_deficit": None,
            "route_deficits": {},
        }
        self.assertEqual(head.classify(iteration=8, **no_pair), "late_anchor_probe")
        route_context = {
            "official_feasible": False,
            "pair_presence": {"sq": False, "qd": True, "sd": False},
            "has_loose_feasible": False,
            "boundary_mode": "similarity",
            "boundary_deficit": 0.28,
        }
        state = head.classify(
            iteration=9,
            route_deficits={"similarity": 0.28},
            **route_context,
        )
        self.assertEqual(state, "late_anchor_route_probe")
        self.assertEqual(head.reserve_fraction(state), 0.12)
        state = head.classify(
            iteration=10,
            route_deficits={"similarity": 0.24},
            **route_context,
        )
        self.assertEqual(state, "late_anchor_route_commit")
        self.assertEqual(head.reserve_fraction(state), 0.20)

    def test_v5_preserves_v1_before_the_late_portfolio(self):
        head = ProtectedRoutePortfolioLeadHead(late_start_iteration=8)
        context = {
            "official_feasible": False,
            "pair_presence": {"sq": True, "qd": True, "sd": False},
            "has_loose_feasible": True,
            "boundary_mode": None,
            "boundary_deficit": None,
            "route_deficits": {"dock": 0.08, "similarity": 0.12},
        }
        self.assertEqual(head.classify(iteration=7, **context), "bridge_probe")
        self.assertEqual(
            head.classify(iteration=8, **context),
            "late_route_portfolio",
        )

    def test_v5_keeps_every_eligible_completion_route(self):
        head = ProtectedRoutePortfolioLeadHead(
            late_start_iteration=8,
            portfolio_share=0.20,
            max_route_deficit=0.21,
        )
        engine = BaselineProtectedFrontierEngine(head)
        state = engine.classify(
            iteration=8,
            official_feasible=False,
            pair_presence={"sq": True, "qd": True, "sd": True},
            has_loose_feasible=True,
            boundary_mode=None,
            boundary_deficit=None,
            route_deficits={"dock": 0.05, "similarity": 0.12, "quality": 0.20},
        )
        allocation = engine.allocate(
            100,
            lambda total: {"proposal": {"legacy": total}},
            state=state,
            context=engine.last_context,
            available={
                "proposal": {
                    "legacy",
                    "completion_dock_refine",
                    "completion_similarity_repair",
                    "completion_quality_repair",
                }
            },
        )["proposal"]

        self.assertEqual(allocation["legacy"], 80)
        self.assertEqual(
            sum(
                allocation[name]
                for name in (
                    "completion_dock_refine",
                    "completion_similarity_repair",
                    "completion_quality_repair",
                )
            ),
            20,
        )
        self.assertGreater(allocation["completion_dock_refine"], 0)
        self.assertGreater(allocation["completion_similarity_repair"], 0)
        self.assertGreater(allocation["completion_quality_repair"], 0)

    def test_v5_uses_only_one_seed_probe_without_pair_frontiers(self):
        head = ProtectedRoutePortfolioLeadHead(
            late_start_iteration=8,
            seed_probe_share=0.08,
        )
        context = {
            "official_feasible": False,
            "pair_presence": {"sq": False, "qd": False, "sd": False},
            "has_loose_feasible": False,
            "boundary_mode": None,
            "boundary_deficit": None,
            "route_deficits": {},
        }
        state = head.classify(iteration=8, **context)
        self.assertEqual(state, "late_seed_probe")
        self.assertEqual(head.reserve_fraction(state), 0.08)
        self.assertEqual(
            head.operator_priors("proposal", state),
            {"completion_seed_restart": 1.0},
        )
        self.assertEqual(head.classify(iteration=9, **context), "baseline")

    def test_lead_keeps_baseline_through_iteration_six(self):
        engine = BaselineProtectedFrontierEngine(SafeLeadFrontierHead())
        state = engine.classify(
            iteration=6,
            official_feasible=False,
            has_generated_similarity=False,
            has_stage_three=False,
            pair_presence={},
        )
        self.assertEqual(state, "baseline")
        self.assertEqual(engine.reserve_fraction(), 0.0)

    def test_lead_success_is_absorbing(self):
        engine = BaselineProtectedFrontierEngine(SafeLeadFrontierHead())
        state = engine.classify(
            iteration=10,
            official_feasible=True,
            has_generated_similarity=True,
            has_stage_three=True,
            pair_presence={"sq": True},
        )
        self.assertEqual(state, "locked")
        self.assertEqual(engine.reserve_fraction(), 0.0)

    def test_lead_completion_uses_only_observed_pair_frontiers(self):
        head = SafeLeadFrontierHead()
        context = {"pair_presence": {"sq": True, "qd": False, "sd": True}}
        priors = head.operator_priors("proposal", "completion", context)
        self.assertIn("dock_refine", priors)
        self.assertIn("quality_repair", priors)
        self.assertNotIn("similarity_repair", priors)

    def test_pmo_warmup_is_exact_baseline(self):
        engine = BaselineProtectedFrontierEngine(SafePMOFrontierHead())
        state = engine.classify(state="warmup", calls=500)
        allocation = engine.allocate(
            40,
            lambda total: {"root": {"attach_only": total}},
            state=state,
            context={"calls": 500},
            available={"root": {"attach_only"}, "local": {"elite_tiny"}},
        )
        self.assertEqual(allocation, {"root": {"attach_only": 40}})
        self.assertEqual(engine.last_reserve, 0)

    def test_pmo_search_reserve_is_bounded_and_budget_preserving(self):
        engine = BaselineProtectedFrontierEngine(SafePMOFrontierHead())
        state = engine.classify(state="search", calls=2000)

        def baseline(total):
            return {
                "root": {"attach_only": total // 4},
                "local": {"graph_shrink": total - total // 4},
            }

        allocation = engine.allocate(
            100,
            baseline,
            state=state,
            context={"calls": 2000},
            available={
                "root": {"attach_only"},
                "local": {
                    "elite_tiny",
                    "elite_small",
                    "elite_medium",
                    "graph_shrink",
                },
            },
        )
        total = sum(sum(group.values()) for group in allocation.values())
        self.assertEqual(total, 100)
        self.assertEqual(engine.last_reserve, 12)
        self.assertGreaterEqual(allocation["local"].get("elite_tiny", 0), 4)
        self.assertGreaterEqual(allocation["local"].get("elite_small", 0), 4)

    def test_pmo_recovery_states_do_not_override_v9(self):
        head = SafePMOFrontierHead()
        for state in ("plateau", "collapsed", "sparse", "fallback"):
            self.assertEqual(head.reserve_fraction(state, {"calls": 5000}), 0.0)

    def test_lead_bridge_requires_two_complementary_frontiers(self):
        engine = BaselineProtectedFrontierEngine(SafeLeadBridgeHead())
        state = engine.classify(
            iteration=7,
            official_feasible=False,
            pair_presence={"sq": True, "qd": False, "sd": False},
        )
        self.assertEqual(state, "baseline")
        self.assertEqual(engine.reserve_fraction(), 0.0)

        state = engine.classify(
            iteration=7,
            official_feasible=False,
            pair_presence={"sq": True, "qd": True, "sd": False},
        )
        self.assertEqual(state, "bridge")
        self.assertEqual(engine.reserve_fraction(), 0.12)

    def test_lead_bridge_preserves_budget_and_baseline_floor(self):
        engine = BaselineProtectedFrontierEngine(SafeLeadBridgeHead())
        state = engine.classify(
            iteration=8,
            official_feasible=False,
            pair_presence={"sq": True, "qd": False, "sd": True},
        )
        allocation = engine.allocate(
            100,
            lambda total: {"proposal": {"legacy": total}},
            state=state,
            context=engine.last_context,
            available={"proposal": {"legacy", "pair_bridge"}},
        )
        self.assertEqual(allocation["proposal"]["legacy"], 88)
        self.assertEqual(allocation["proposal"]["pair_bridge"], 12)
        self.assertEqual(engine.last_reserve, 12)

    def test_pmo_evidence_gate_expands_and_shuts_off(self):
        head = EvidenceGatedPMOHead(window_calls=500)
        engine = BaselineProtectedFrontierEngine(head)

        state = engine.classify(state="warmup", calls=999)
        self.assertEqual(engine.reserve_fraction(state, {"calls": 999}), 0.0)
        state = engine.classify(state="search", calls=1000)
        self.assertEqual(engine.reserve_fraction(state, {"calls": 1000}), 0.04)

        head.observe_batch(
            operator="elite_tiny",
            reward=0.60,
            evaluated=125,
            calls=1125,
            state="search",
        )
        head.observe_batch(
            operator="elite_small",
            reward=0.58,
            evaluated=125,
            calls=1250,
            state="search",
        )
        head.observe_batch(
            operator="graph_swap",
            reward=0.20,
            evaluated=125,
            calls=1375,
            state="search",
        )
        head.observe_batch(
            operator="graph_shrink",
            reward=0.22,
            evaluated=125,
            calls=1500,
            state="search",
        )
        self.assertEqual(head.phase, "expanded")
        self.assertAlmostEqual(head.current_reserve, 0.08)

        head.observe_batch(
            operator="elite_small",
            reward=0.10,
            evaluated=125,
            calls=1625,
            state="search",
        )
        head.observe_batch(
            operator="elite_tiny",
            reward=0.12,
            evaluated=125,
            calls=1750,
            state="search",
        )
        head.observe_batch(
            operator="motif_restart",
            reward=0.50,
            evaluated=125,
            calls=1875,
            state="search",
        )
        head.observe_batch(
            operator="graph_swap",
            reward=0.48,
            evaluated=125,
            calls=2000,
            state="search",
        )
        self.assertEqual(head.phase, "baseline")
        self.assertEqual(head.current_reserve, 0.0)
        self.assertEqual(head.next_probe_call, 3000)

        restored = EvidenceGatedPMOHead(window_calls=500)
        restored.load_state_dict(head.state_dict())
        self.assertEqual(restored.phase, "baseline")
        self.assertEqual(restored.next_probe_call, 3000)

    def test_reversible_gate_requires_repeated_frontier_evidence(self):
        head = ReversibleEvidencePMOHead(window_calls=500)
        engine = BaselineProtectedFrontierEngine(head)
        state = engine.classify(state="search", calls=1000)
        self.assertEqual(engine.reserve_fraction(state, {"calls": 1000}), 0.04)

        for calls in (1500, 2000):
            engine.observe_batch(
                operator="elite_tiny",
                reward=0.90,
                evaluated=100,
                calls=calls - 400,
                state="search",
                frontier_gain=0.02,
                top10_entries=20,
            )
            engine.observe_batch(
                operator="graph_shrink",
                reward=0.20,
                evaluated=400,
                calls=calls,
                state="search",
                frontier_gain=0.01,
                top10_entries=20,
            )
            if calls == 1500:
                self.assertEqual(head.phase, "confirm_positive")
                self.assertEqual(head.current_reserve, 0.04)

        self.assertEqual(head.phase, "expanded")
        self.assertEqual(head.current_reserve, 0.08)

    def test_reversible_gate_ignores_reward_without_frontier_evidence(self):
        head = ReversibleEvidencePMOHead(window_calls=500)
        engine = BaselineProtectedFrontierEngine(head)
        state = engine.classify(state="search", calls=1000)
        engine.reserve_fraction(state, {"calls": 1000})
        engine.observe_batch(
            operator="elite_small",
            reward=1.0,
            evaluated=100,
            calls=1100,
            state="search",
            frontier_gain=0.0,
            top10_entries=0,
        )
        engine.observe_batch(
            operator="attach_only",
            reward=0.0,
            evaluated=400,
            calls=1500,
            state="search",
            frontier_gain=0.01,
            top10_entries=10,
        )
        self.assertEqual(head.phase, "baseline_negative")
        self.assertEqual(head.current_reserve, 0.0)
        self.assertEqual(head.next_probe_call, 2500)

    def test_reversible_gate_discards_evidence_outside_search(self):
        head = ReversibleEvidencePMOHead(window_calls=500)
        engine = BaselineProtectedFrontierEngine(head)
        state = engine.classify(state="search", calls=1000)
        engine.reserve_fraction(state, {"calls": 1000})
        engine.observe_batch(
            operator="elite_tiny",
            reward=0.8,
            evaluated=100,
            calls=1100,
            state="search",
            frontier_gain=0.01,
            top10_entries=10,
        )
        state = engine.classify(state="plateau", calls=1200)
        self.assertEqual(engine.reserve_fraction(state, {"calls": 1200}), 0.0)
        self.assertEqual(head.current_reserve, 0.0)
        self.assertEqual(head.window["target_evaluated"], 0)


if __name__ == "__main__":
    unittest.main()
