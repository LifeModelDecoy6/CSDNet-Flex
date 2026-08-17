import random
import unittest
from types import SimpleNamespace
from unittest.mock import Mock

from CSDNet.exp.lead.frontier import adaptive_peripheral_edit_plan, constraint_state
from CSDNet.exp.lead.run import CSDNetLeadOptimizer
from CSDNet.optim.frontier import (
    classify_frontier_state,
    constraint_rank,
    integrated_operator_weights,
    lineage_metrics,
    rank_improved,
    trust_region_fraction,
)


def make_item(dock, qed, sa, similarity, root_id="root"):
    item = {
        "dock": dock,
        "root_id": root_id,
        **constraint_state(
            dock=dock,
            qed=qed,
            sa=sa,
            similarity=similarity,
            start_dock=10.0,
            similarity_threshold=0.6,
            docking_margin=0.05,
            residual_l1_weight=0.10,
        ),
    }
    return item


class IntegratedFrontierTest(unittest.TestCase):
    def test_constraint_domination_prefers_feasibility_before_docking(self):
        infeasible_high_dock = make_item(14.0, 0.50, 0.80, 0.70)
        feasible_lower_dock = make_item(10.2, 0.70, 0.80, 0.70)

        self.assertLess(
            constraint_rank(feasible_lower_dock),
            constraint_rank(infeasible_high_dock),
        )
        self.assertTrue(rank_improved(feasible_lower_dock, infeasible_high_dock))

    def test_search_state_switches_without_target_specific_rules(self):
        self.assertEqual(
            classify_frontier_state(1, 2, False, 0, 2, 0.2, 0.7),
            "warmup",
        )
        self.assertEqual(
            classify_frontier_state(2, 2, False, 0, 2, 0.2, 0.7),
            "search",
        )
        self.assertEqual(
            classify_frontier_state(4, 2, False, 2, 2, 0.2, 0.7),
            "plateau",
        )
        self.assertEqual(
            classify_frontier_state(4, 2, False, 0, 2, 0.8, 0.7),
            "collapsed",
        )
        self.assertEqual(
            classify_frontier_state(4, 2, True, 9, 2, 0.9, 0.7),
            "refine",
        )

    def test_lineage_metric_detects_population_collapse(self):
        items = [make_item(9.0, 0.6, 0.8, 0.6, "a") for _ in range(8)]
        items += [make_item(8.0, 0.6, 0.8, 0.6, "b") for _ in range(2)]
        result = lineage_metrics(items)

        self.assertEqual(result["root_count"], 2)
        self.assertAlmostEqual(result["largest_root_fraction"], 0.8)

    def test_trust_region_expands_for_larger_target_deficit(self):
        near = make_item(9.8, 0.70, 0.80, 0.70)
        far = make_item(6.0, 0.70, 0.80, 0.70)
        near_radius = trust_region_fraction(near, ("dock",), 0.07)
        far_radius = trust_region_fraction(far, ("dock",), 0.07)

        self.assertGreater(far_radius, near_radius)
        self.assertLessEqual(far_radius, 0.30)

    def test_refine_state_prioritizes_docking_but_keeps_restart_alive(self):
        weights = integrated_operator_weights("refine")
        self.assertGreater(weights["dock_refine"], weights["joint_repair"])
        self.assertGreater(weights["lineage_restart"], 0.0)

    def test_adaptive_peripheral_plan_respects_component_limit(self):
        plan = adaptive_peripheral_edit_plan(
            "CCOc1ccccc1",
            random.Random(0),
            target_atom_fraction=0.30,
            max_atom_fraction=0.55,
            max_span_tokens=6,
        )
        self.assertIsNotNone(plan)
        self.assertTrue(plan["peripheral"])
        self.assertLessEqual(plan["component_atoms"], 5)

    def test_docking_cache_reuses_first_canonical_evaluation(self):
        optimizer = CSDNetLeadOptimizer.__new__(CSDNetLeadOptimizer)
        optimizer.args = SimpleNamespace(sampler_profile="integrated_frontier")
        optimizer.evaluation_cache = {}
        optimizer.reward_vina = Mock(return_value=[7.5])
        optimizer.reward_qed = Mock(return_value=[0.7])
        optimizer.reward_sa = Mock(return_value=[0.8])
        optimizer.reward_sim = Mock(return_value=[0.9])

        first = optimizer.reward_cached(["CC", "C(C)"])
        second = optimizer.reward_cached(["CC"])

        self.assertEqual(first[0], [7.5, 7.5])
        self.assertEqual(second[0], [7.5])
        optimizer.reward_vina.assert_called_once_with(["CC"])


if __name__ == "__main__":
    unittest.main()
