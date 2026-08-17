import random
import unittest

from CSDNet.optim.frontier import (
    AdaptiveOperatorBandit,
    LeadBestUnionAdapter,
    LeadFrontierAdapter,
    LeadFrontierAdapterV21,
    LineageFrontierArchive,
    PMOFrontierAdapter,
    RestoredLeadFrontierAdapter,
    RestoredPMOFrontierAdapter,
    ScalarFrontierAdapter,
    UnifiedFrontierEngine,
    allocate_insertion_flags,
    constrained_batch_frontier_reward,
    scalar_batch_frontier_reward,
)


ROOT_OPERATORS = ("attach_only", "motif_restart", "fragment_anchor")
LOCAL_OPERATORS = (
    "elite_tiny",
    "elite_small",
    "elite_medium",
    "diverse_medium",
    "graph_shrink",
    "graph_swap",
    "graph_expand",
    "rescue_large",
)


def make_engine():
    return UnifiedFrontierEngine(
        adapter=ScalarFrontierAdapter(warmup_calls=100),
        operator_groups={
            "root": ROOT_OPERATORS,
            "local": LOCAL_OPERATORS,
        },
    )


class UnifiedFrontierTest(unittest.TestCase):
    def test_insertion_allocation_is_exact_and_reproducible(self):
        flags_a = allocate_insertion_flags(17, 0.25, random.Random(9))
        flags_b = allocate_insertion_flags(17, 0.25, random.Random(9))
        self.assertEqual(sum(flags_a), 4)
        self.assertEqual(flags_a, flags_b)

    def test_lead_insertion_budget_protects_locked_and_polish_states(self):
        self.assertEqual(LeadFrontierAdapter.insertion_fraction("locked"), 0.0)
        self.assertLess(
            LeadFrontierAdapter.insertion_fraction("dock_polish"),
            LeadFrontierAdapter.insertion_fraction("explore"),
        )

    def test_scalar_state_uses_trajectory_only(self):
        adapter = ScalarFrontierAdapter(
            warmup_calls=100,
            saturation_threshold=0.90,
            sparse_threshold=0.25,
            stagnation_patience=80,
            collapse_threshold=0.60,
        )
        base = {
            "calls": 1000,
            "avg_top10": 0.50,
            "nonzero_rate": 0.50,
            "stagnant_calls": 0,
            "largest_root_fraction": 0.20,
        }
        self.assertEqual(adapter.classify(**base), "search")
        self.assertEqual(
            adapter.classify(**{**base, "avg_top10": 0.95}),
            "saturated",
        )
        self.assertEqual(
            adapter.classify(**{**base, "largest_root_fraction": 0.75}),
            "collapsed",
        )
        self.assertEqual(
            adapter.classify(**{**base, "stagnant_calls": 100}),
            "plateau",
        )
        self.assertEqual(
            adapter.classify(**{**base, "nonzero_rate": 0.10}),
            "sparse",
        )

    def test_allocation_preserves_budget_in_every_state(self):
        engine = make_engine()
        for state in (
            "warmup",
            "saturated",
            "collapsed",
            "plateau",
            "sparse",
            "search",
        ):
            allocation = engine.allocate(97, state=state)
            allocated = sum(
                sum(group.values()) for group in allocation.values()
            )
            self.assertEqual(allocated, 97, state)
        self.assertEqual(
            engine.allocate(97, state="warmup"),
            {"root": {"attach_only": 97}},
        )

    def test_scalar_reward_is_order_invariant(self):
        kwargs = {
            "before_scores": [0.88, 0.84, 0.80, 0.77, 0.70, 0.62],
            "before_top10": 0.7683,
            "after_top10": 0.7733,
            "frontier_scale": 0.01,
            "delta_scale": 0.08,
        }
        scores = [0.91, 0.66, 0.42, 0.35]
        parents = [0.82, 0.70, None, 0.30]
        forward = scalar_batch_frontier_reward(
            scores=scores,
            parent_scores=parents,
            **kwargs,
        )
        reverse = scalar_batch_frontier_reward(
            scores=list(reversed(scores)),
            parent_scores=list(reversed(parents)),
            **kwargs,
        )
        self.assertEqual(forward, reverse)

    def test_batch_credit_updates_only_selected_group(self):
        engine = make_engine()
        before_root = engine.bandits["root"].snapshot()
        reward, parts = engine.update_scalar_batch(
            group="local",
            operator="elite_small",
            scores=[0.92, 0.84],
            before_scores=[0.80, 0.75, 0.70],
            before_top10=0.75,
            after_top10=0.80,
            parent_scores=[0.82, 0.80],
        )
        self.assertGreater(reward, 0.0)
        self.assertGreater(parts["frontier_gain"], 0.0)
        self.assertEqual(engine.bandits["root"].snapshot(), before_root)
        row = engine.bandits["local"].snapshot()["elite_small"]
        self.assertEqual(row["batches"], 1)
        self.assertEqual(row["evaluated"], 2)

    def test_state_round_trip_preserves_policy_learning(self):
        engine = make_engine()
        engine.classify(
            calls=1000,
            avg_top10=0.50,
            nonzero_rate=0.50,
            stagnant_calls=0,
            largest_root_fraction=0.20,
        )
        engine.update_scalar_batch(
            group="root",
            operator="attach_only",
            scores=[0.90],
            before_scores=[0.70],
            before_top10=0.70,
            after_top10=0.80,
            parent_scores=[None],
        )

        restored = make_engine()
        restored.load_state_dict(engine.state_dict())

        self.assertEqual(restored.state_dict(), engine.state_dict())
        self.assertEqual(restored.snapshot(), engine.snapshot())

    def test_low_floor_bandit_suppresses_repeatedly_weak_operator(self):
        bandit = AdaptiveOperatorBandit(
            ["good", "weak"],
            alpha=0.5,
            ucb_weight=0.0,
            min_multiplier=0.01,
            base_floor=0.02,
        )
        for _ in range(8):
            bandit.update("good", reward=1.0, evaluated=20)
            bandit.update("weak", reward=0.0, evaluated=20)
        weighted = dict(bandit.weighted({"good": 1.0, "weak": 1.0}))
        self.assertGreater(weighted["good"], 5.0 * weighted["weak"])

    def test_pmo_v2_moves_budget_from_shrink_to_local_refinement(self):
        old = ScalarFrontierAdapter.operator_priors("local", "search")
        new = PMOFrontierAdapter.operator_priors("local", "search")
        self.assertLess(new["graph_shrink"], old["graph_shrink"])
        self.assertGreater(
            new["elite_tiny"] + new["elite_small"],
            old["elite_tiny"] + old["elite_small"],
        )

    def test_restored_pmo_adapter_preserves_v9_allocation(self):
        engine = UnifiedFrontierEngine(
            adapter=RestoredPMOFrontierAdapter(warmup_calls=1000),
            operator_groups={
                "root": ROOT_OPERATORS,
                "local": LOCAL_OPERATORS,
            },
            bandit_configs={
                "root": {"base_floor": 0.30},
                "local": {"base_floor": 0.30},
            },
        )
        expected_root = {
            "saturated": 6,
            "collapsed": 31,
            "plateau": 27,
            "sparse": 29,
            "search": 17,
        }
        for state, root_count in expected_root.items():
            allocation = engine.allocate(96, state=state)
            self.assertEqual(sum(allocation["root"].values()), root_count, state)
            self.assertEqual(
                sum(sum(group.values()) for group in allocation.values()),
                96,
                state,
            )

    def test_shared_lineage_archive_retains_independent_roots(self):
        archive = LineageFrontierArchive(score_slots=2, lineage_slots=3)
        root = archive.add_root("a", 0.9, "a", "attach_only", 1)
        archive.add_root("b", 0.7, "b", "motif_restart", 2)
        archive.add_root("c", 0.6, "c", "fragment_anchor", 3)
        for index in range(5):
            archive.add_child(
                f"a{index}",
                0.89 - 0.01 * index,
                root,
                "elite_small",
                4 + index,
            )
        score_pool, lineage_pool = archive.parent_pools()
        self.assertEqual({row.root_id for row in score_pool}, {"a"})
        self.assertEqual({row.root_id for row in lineage_pool}, {"a", "b", "c"})

    def test_constrained_reward_penalizes_constraint_regression(self):
        base = {
            "rank_improved": True,
            "strict": False,
            "crossed": 1,
            "pair_gain": 1,
            "admitted": True,
            "residual_gain": 0.10,
        }
        clean, _ = constrained_batch_frontier_reward(
            [{**base, "regressed": 0}]
        )
        harmful, parts = constrained_batch_frontier_reward(
            [{**base, "regressed": 1}]
        )
        self.assertGreater(clean, harmful)
        self.assertEqual(parts["regression_rate"], 1.0)

    def test_lead_adapter_uses_observed_bridge_state_and_availability(self):
        adapter = LeadFrontierAdapter(warmup_iterations=1)
        state = adapter.classify(
            iteration=2,
            has_feasible=False,
            has_pair_feasible=True,
            best_max_deficit=0.30,
            stagnant_iterations=0,
            largest_root_fraction=0.20,
            available_operators=("legacy", "dock_refine"),
            constraint_needs={"dock_refine": 0.8},
        )
        self.assertEqual(state, "bridge")
        priors = adapter.operator_priors(
            "proposal",
            state,
            {
                "available_operators": ("legacy", "dock_refine"),
                "constraint_needs": {"dock_refine": 0.8},
            },
        )
        self.assertEqual(set(priors), {"legacy", "dock_refine"})
        self.assertGreater(priors["dock_refine"], priors["legacy"])

    def test_lead_engine_allocation_preserves_complete_budget(self):
        operators = (
            "legacy",
            "start_repair",
            "dock_refine",
            "similarity_repair",
            "quality_repair",
            "joint_repair",
            "lineage_restart",
        )
        engine = UnifiedFrontierEngine(
            adapter=LeadFrontierAdapter(warmup_iterations=1),
            operator_groups={"proposal": operators},
            bandit_configs={"proposal": {"base_floor": 0.02}},
        )
        engine.classify(
            iteration=2,
            has_feasible=False,
            has_pair_feasible=True,
            best_max_deficit=0.10,
            stagnant_iterations=0,
            largest_root_fraction=0.20,
            available_operators=operators,
            constraint_needs={},
        )
        allocation = engine.allocate(100)
        self.assertEqual(sum(allocation["proposal"].values()), 100)

    def test_lead_v21_bridge_precedes_collapse_and_plateau(self):
        adapter = LeadFrontierAdapterV21(warmup_iterations=1)
        state = adapter.classify(
            iteration=3,
            has_feasible=False,
            has_pair_feasible=True,
            best_max_deficit=0.30,
            stagnant_iterations=8,
            largest_root_fraction=0.95,
            available_operators=("legacy", "dock_refine"),
            constraint_needs={"dock_refine": 0.2},
            completion_operators=("dock_refine",),
            has_generated_similarity_feasible=True,
        )
        self.assertEqual(state, "bridge")

    def test_lead_v21_warmup_keeps_half_legacy_budget(self):
        operators = (
            "legacy",
            "start_repair",
            "dock_refine",
            "similarity_repair",
            "quality_repair",
            "joint_repair",
            "lineage_restart",
            "scaffold_rescue",
        )
        engine = UnifiedFrontierEngine(
            adapter=LeadFrontierAdapterV21(warmup_iterations=2),
            operator_groups={"proposal": operators},
            bandit_configs={"proposal": {"base_floor": 0.02}},
        )
        context = {
            "iteration": 1,
            "has_feasible": False,
            "has_pair_feasible": False,
            "best_max_deficit": 0.40,
            "stagnant_iterations": 0,
            "largest_root_fraction": 0.20,
            "available_operators": operators,
            "constraint_needs": {},
            "completion_operators": (),
            "has_generated_similarity_feasible": False,
        }
        engine.classify(**context)
        allocation = engine.allocate(100)["proposal"]
        self.assertEqual(allocation["legacy"], 50)
        self.assertEqual(sum(allocation.values()), 100)

    def test_lead_v21_bridge_completes_observed_missing_constraint(self):
        operators = (
            "legacy",
            "start_repair",
            "dock_refine",
            "similarity_repair",
            "quality_repair",
            "joint_repair",
            "lineage_restart",
            "scaffold_rescue",
        )
        engine = UnifiedFrontierEngine(
            adapter=LeadFrontierAdapterV21(warmup_iterations=1),
            operator_groups={"proposal": operators},
            bandit_configs={"proposal": {"base_floor": 0.02}},
        )
        engine.classify(
            iteration=2,
            has_feasible=False,
            has_pair_feasible=True,
            best_max_deficit=0.03,
            stagnant_iterations=4,
            largest_root_fraction=0.90,
            available_operators=operators,
            constraint_needs={"dock_refine": 0.5},
            completion_operators=("dock_refine",),
            has_generated_similarity_feasible=True,
        )
        allocation = engine.allocate(100)["proposal"]
        self.assertGreaterEqual(allocation["dock_refine"], 50)
        self.assertGreaterEqual(allocation["legacy"], 10)
        self.assertEqual(sum(allocation.values()), 100)

    def test_lead_v21_activates_generic_scaffold_rescue(self):
        adapter = LeadFrontierAdapterV21()
        context = {
            "available_operators": (
                "legacy",
                "start_repair",
                "joint_repair",
                "lineage_restart",
                "scaffold_rescue",
            ),
            "constraint_needs": {},
            "has_generated_similarity_feasible": False,
        }
        priors = adapter.operator_priors("proposal", "search", context)
        floors = adapter.operator_floors("proposal", "search", context)
        self.assertGreater(priors["scaffold_rescue"], priors["start_repair"])
        self.assertEqual(floors["scaffold_rescue"], 0.15)

    def test_restored_lead_adapter_cannot_erase_legacy_sampler(self):
        operators = (
            "legacy",
            "start_repair",
            "dock_refine",
            "similarity_repair",
            "quality_repair",
            "joint_repair",
            "lineage_restart",
        )
        engine = UnifiedFrontierEngine(
            adapter=RestoredLeadFrontierAdapter(warmup_iterations=2),
            operator_groups={"proposal": operators},
            bandit_configs={"proposal": {"base_floor": 0.02}},
        )
        common = {
            "has_feasible": False,
            "best_max_deficit": 0.40,
            "stagnant_iterations": 0,
            "largest_root_fraction": 0.20,
            "available_operators": operators,
            "constraint_needs": {},
            "completion_operators": (),
            "has_generated_similarity_feasible": True,
        }
        engine.classify(iteration=1, has_pair_feasible=False, **common)
        warmup = engine.allocate(100)["proposal"]
        self.assertEqual(warmup, {"legacy": 100})

        engine.classify(iteration=3, has_pair_feasible=False, **common)
        search = engine.allocate(100)["proposal"]
        self.assertGreaterEqual(search["legacy"], 85)
        self.assertEqual(sum(search.values()), 100)

    def test_restored_lead_bridge_keeps_legacy_and_completion_floors(self):
        operators = (
            "legacy",
            "start_repair",
            "dock_refine",
            "similarity_repair",
            "quality_repair",
            "joint_repair",
            "lineage_restart",
        )
        engine = UnifiedFrontierEngine(
            adapter=RestoredLeadFrontierAdapter(warmup_iterations=1),
            operator_groups={"proposal": operators},
            bandit_configs={"proposal": {"base_floor": 0.02}},
        )
        engine.classify(
            iteration=3,
            has_feasible=False,
            has_pair_feasible=True,
            best_max_deficit=0.05,
            stagnant_iterations=0,
            largest_root_fraction=0.20,
            available_operators=operators,
            constraint_needs={"dock_refine": 1.0},
            completion_operators=("dock_refine",),
            has_generated_similarity_feasible=True,
        )
        allocation = engine.allocate(100)["proposal"]
        self.assertGreaterEqual(allocation["legacy"], 70)
        self.assertGreaterEqual(allocation["dock_refine"], 15)
        self.assertEqual(sum(allocation.values()), 100)

    def test_best_union_lead_adapter_keeps_both_proposal_scales(self):
        operators = (
            "legacy",
            "legacy_local",
            "start_repair",
            "dock_refine",
            "similarity_repair",
            "quality_repair",
            "joint_repair",
            "lineage_restart",
        )
        engine = UnifiedFrontierEngine(
            adapter=LeadBestUnionAdapter(warmup_iterations=2),
            operator_groups={"proposal": operators},
            bandit_configs={"proposal": {"base_floor": 0.02}},
        )
        common = {
            "has_feasible": False,
            "best_max_deficit": 0.40,
            "stagnant_iterations": 0,
            "largest_root_fraction": 0.20,
            "available_operators": operators,
            "constraint_needs": {},
            "completion_operators": (),
            "has_generated_similarity_feasible": True,
            "similarity_threshold": 0.6,
        }
        engine.classify(iteration=1, has_pair_feasible=False, **common)
        allocation = engine.allocate(100)["proposal"]
        self.assertGreaterEqual(allocation["legacy"], 39)
        self.assertGreaterEqual(allocation["legacy_local"], 20)
        self.assertEqual(sum(allocation.values()), 100)

    def test_best_union_bridge_reserves_completion_without_target_routing(self):
        operators = (
            "legacy",
            "legacy_local",
            "start_repair",
            "dock_refine",
            "similarity_repair",
            "quality_repair",
            "joint_repair",
            "lineage_restart",
        )
        engine = UnifiedFrontierEngine(
            adapter=LeadBestUnionAdapter(warmup_iterations=1),
            operator_groups={"proposal": operators},
            bandit_configs={"proposal": {"base_floor": 0.02}},
        )
        context = {
            "iteration": 3,
            "has_feasible": False,
            "has_pair_feasible": True,
            "best_max_deficit": 0.05,
            "stagnant_iterations": 0,
            "largest_root_fraction": 0.20,
            "available_operators": operators,
            "constraint_needs": {"dock_refine": 1.0},
            "completion_operators": ("dock_refine",),
            "has_generated_similarity_feasible": True,
            "similarity_threshold": 0.6,
        }
        engine.classify(**context)
        allocation = engine.allocate(100)["proposal"]
        self.assertGreaterEqual(
            allocation["legacy"] + allocation["legacy_local"],
            34,
        )
        self.assertGreaterEqual(allocation["dock_refine"], 30)
        self.assertEqual(sum(allocation.values()), 100)

    def test_constrained_engine_charges_unproductive_proposal_budget(self):
        engine = UnifiedFrontierEngine(
            adapter=LeadFrontierAdapter(),
            operator_groups={"proposal": ("legacy",)},
            bandit_configs={"proposal": {"alpha": 0.5, "base_floor": 0.02}},
        )
        before = engine.bandits["proposal"].stats["legacy"]["ema"]
        reward, parts = engine.update_constrained_batch(
            group="proposal",
            operator="legacy",
            transitions=[],
            requested=20,
        )
        self.assertEqual(reward, 0.0)
        self.assertEqual(parts["yield_rate"], 0.0)
        self.assertLess(
            engine.bandits["proposal"].stats["legacy"]["ema"],
            before,
        )


if __name__ == "__main__":
    unittest.main()
