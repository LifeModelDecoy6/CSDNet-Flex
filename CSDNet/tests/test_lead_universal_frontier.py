import random
import tempfile
import unittest
from pathlib import Path
from types import SimpleNamespace

from rdkit import Chem, DataStructs
from rdkit.Chem import AllChem

from CSDNet.exp.lead.frontier import (
    archive_constraint_need,
    completion_recovery_multipliers,
    constraint_state,
    recovery_v2_operator_multipliers,
    recovery_v2_state,
    transition_reward,
)
from CSDNet.exp.lead.run import (
    CSDNetLeadOptimizer,
    is_learned_length_mode,
    token_safe_model_smiles,
)
from CSDNet.exp.lead.verify_feasible_result import verify_result
from CSDNet.util.tokenizer import tokenize_smiles


def make_item(dock, qed, sa, similarity):
    return constraint_state(
        dock=dock,
        qed=qed,
        sa=sa,
        similarity=similarity,
        start_dock=10.0,
        similarity_threshold=0.6,
        docking_margin=0.05,
        residual_l1_weight=0.10,
    )


def make_frontier_item(smiles, dock, qed, sa, similarity):
    return {
        "smiles": smiles,
        "dock": float(dock),
        "qed": float(qed),
        "sa": float(sa),
        "sim": float(similarity),
        "root_id": smiles,
        "depth": 0,
        **make_item(dock, qed, sa, similarity),
    }


class UniversalFrontierTest(unittest.TestCase):
    def test_token_safe_model_smiles_removes_only_unscored_stereo(self):
        isomeric = "N[C@@H](C)C(=O)O"
        nonisomeric = Chem.MolToSmiles(
            Chem.MolFromSmiles(isomeric),
            canonical=True,
            isomericSmiles=False,
        )
        tokenizer = SimpleNamespace(vocab=set(tokenize_smiles(nonisomeric)))

        safe, changed = token_safe_model_smiles(isomeric, tokenizer)

        original_fp = AllChem.GetMorganFingerprintAsBitVect(
            Chem.MolFromSmiles(isomeric),
            2,
            2048,
        )
        safe_fp = AllChem.GetMorganFingerprintAsBitVect(
            Chem.MolFromSmiles(safe),
            2,
            2048,
        )
        self.assertTrue(changed)
        self.assertEqual(safe, nonisomeric)
        self.assertEqual(
            DataStructs.TanimotoSimilarity(original_fp, safe_fp),
            1.0,
        )

    def test_feasible_result_verifier_rejects_public_constraint_failure(self):
        with tempfile.TemporaryDirectory() as tmp_dir:
            result = Path(tmp_dir) / "lead.csv"
            result.write_text(
                "CCN,10.0,0.70,0.75,0.65,\n"
                "CCC,11.0,0.59,0.75,0.70,\n",
                encoding="utf-8",
            )

            with self.assertRaisesRegex(RuntimeError, "Infeasible oracle row"):
                verify_result(result, 0.6, min_calls=1, oracle_budget=1000)

    def test_resume_restores_prior_oracle_rows_without_requerying(self):
        with tempfile.TemporaryDirectory() as tmp_dir:
            result_path = Path(tmp_dir) / "result.csv"
            result_path.write_text(
                "CCN,9.0,0.70,0.80,0.70,\nCCC,11.0,0.70,0.80,0.70,\n",
                encoding="utf-8",
            )

            optimizer = CSDNetLeadOptimizer.__new__(CSDNetLeadOptimizer)
            optimizer.args = SimpleNamespace(
                global_oracle_dedup=True,
                oracle_budget=1000,
                num_iter=10,
                num_gen=100,
                sim_thr=0.6,
            )
            optimizer.fname = str(result_path)
            optimizer.frontier_diagnostic_path = str(Path(tmp_dir) / "diagnostics.csv")
            optimizer.start_smiles = "CCO"
            optimizer.start_prop = 10.0
            optimizer.oracle_evaluated_smiles = {"CCO"}
            optimizer.generated_seen = set()
            optimizer.evaluation_cache = {}
            optimizer.current_candidate_ops = []
            optimizer.current_candidate_meta = []
            optimizer.frontier_iteration = 0
            restored = {}

            def restore_frontier(smiles, props):
                restored["smiles"] = smiles
                restored["props"] = props

            optimizer.frontier_update_state = restore_frontier
            optimizer.update_population = lambda smiles, props: None

            optimizer._load_oracle_resume_history()

            self.assertEqual(optimizer.resume_total, 2)
            self.assertEqual(optimizer.resume_raw_best, 11.0)
            self.assertEqual(optimizer.resume_feasible_best, 11.0)
            self.assertEqual(restored["smiles"], ["CCN", "CCC"])
            self.assertEqual(
                optimizer.oracle_evaluated_smiles,
                {"CCO", "CCN", "CCC"},
            )
            self.assertEqual(optimizer.resume_iteration_offset, 10)

    def test_budget_completion_restarts_without_similarity_quality_frontier(self):
        optimizer = CSDNetLeadOptimizer.__new__(CSDNetLeadOptimizer)
        optimizer.args = SimpleNamespace(budget_close_dock_ratio=0.90)
        optimizer.start_smiles = "CCO"
        optimizer.start_prop = 10.0
        optimizer.population = [(1.0, "C"), (1.0, "N")]
        optimizer.initial_population_size = 2
        optimizer.frontier_archives = {"sq": [], "strict": []}

        decision = optimizer.budget_completion_decision()

        self.assertEqual(decision["route"], "restart")
        self.assertEqual(
            decision["reason"],
            "no_similarity_quality_frontier",
        )

    def test_budget_completion_continues_near_docking_boundary(self):
        optimizer = CSDNetLeadOptimizer.__new__(CSDNetLeadOptimizer)
        optimizer.args = SimpleNamespace(budget_close_dock_ratio=0.90)
        optimizer.start_smiles = "CCO"
        optimizer.start_prop = 10.0
        optimizer.population = [(1.0, "C"), (1.0, "N")]
        optimizer.initial_population_size = 2
        near = {
            "smiles": "CCN",
            "dock": 9.2,
        }
        optimizer.frontier_archives = {"sq": [near], "strict": []}

        decision = optimizer.budget_completion_decision()

        self.assertEqual(decision["route"], "frontier")
        self.assertEqual(decision["reason"], "near_docking_boundary")
        self.assertAlmostEqual(decision["best_dock_ratio"], 0.92)

    def test_global_oracle_dedup_preserves_metadata_alignment(self):
        optimizer = CSDNetLeadOptimizer.__new__(CSDNetLeadOptimizer)
        optimizer.args = SimpleNamespace(global_oracle_dedup=True)
        optimizer.oracle_evaluated_smiles = {"CCO"}
        optimizer.current_candidate_meta = [
            {"operator": "old"},
            {"operator": "new"},
            {"operator": "duplicate"},
        ]
        optimizer.current_candidate_ops = ["old", "new", "duplicate"]

        kept = optimizer.filter_oracle_novel_candidates(["OCC", "CCN", "NCC"])

        self.assertEqual(kept, ["CCN"])
        self.assertEqual(
            optimizer.current_candidate_meta,
            [{"operator": "new"}],
        )
        self.assertEqual(optimizer.current_candidate_ops, ["new"])

    def test_normal_iterations_trim_to_official_oracle_budget(self):
        optimizer = CSDNetLeadOptimizer.__new__(CSDNetLeadOptimizer)
        optimizer.args = SimpleNamespace(oracle_budget=1000)
        optimizer.current_candidate_meta = [
            {"candidate": index} for index in range(100)
        ]
        optimizer.current_candidate_ops = ["legacy"] * 100

        kept = optimizer.limit_oracle_budget_candidates(
            [f"candidate-{index}" for index in range(100)],
            total=975,
        )

        self.assertEqual(len(kept), 25)
        self.assertEqual(len(optimizer.current_candidate_meta), 25)
        self.assertEqual(len(optimizer.current_candidate_ops), 25)

    def test_budget_completion_uses_exact_remaining_oracle_calls(self):
        optimizer = CSDNetLeadOptimizer.__new__(CSDNetLeadOptimizer)
        optimizer.args = SimpleNamespace(
            budget_completion=True,
            global_oracle_dedup=True,
            oracle_budget=1000,
            budget_completion_empty_patience=3,
            budget_completion_max_iterations=10,
            budget_completion_until_budget=False,
            num_gen=100,
            num_iter=10,
        )
        optimizer.frontier_iteration = 10
        requested = []
        optimizer.budget_completion_decision = lambda: {
            "route": "frontier",
            "reason": "near_docking_boundary",
            "best_dock_ratio": 0.95,
            "sim_quality_count": 1,
            "strict_count": 0,
            "population_growth": 1,
        }

        def generate(route, target_n):
            self.assertEqual(route, "frontier")
            requested.append(target_n)
            return [f"candidate-{index}" for index in range(target_n)]

        def evaluate(
            smiles_list,
            iter_idx,
            total,
            raw_best,
            feasible_best,
            phase,
        ):
            self.assertEqual(phase, "Budget")
            return (
                total + len(smiles_list),
                raw_best,
                feasible_best,
                bool(smiles_list),
            )

        optimizer.generate_budget_completion_batch = generate
        optimizer.evaluate_lead_batch = evaluate

        total, _, _ = optimizer.run_budget_completion(735, 10.0, 10.0)

        self.assertEqual(total, 1000)
        self.assertEqual(requested, [100, 100, 65])

    def test_hard_budget_completion_restarts_instead_of_stopping(self):
        optimizer = CSDNetLeadOptimizer.__new__(CSDNetLeadOptimizer)
        optimizer.args = SimpleNamespace(
            budget_completion=True,
            global_oracle_dedup=True,
            oracle_budget=3,
            budget_completion_empty_patience=2,
            budget_completion_max_iterations=2,
            budget_completion_until_budget=True,
            num_gen=1,
            num_iter=10,
        )
        optimizer.frontier_iteration = 10
        routes = []
        attempts = 0
        optimizer.budget_completion_decision = lambda: {
            "route": "frontier",
            "reason": "near_docking_boundary",
            "best_dock_ratio": 0.95,
            "sim_quality_count": 1,
            "strict_count": 0,
            "population_growth": 1,
        }

        def generate(route, target_n):
            nonlocal attempts
            attempts += 1
            routes.append(route)
            if attempts <= 3:
                return []
            return [f"candidate-{attempts}"]

        def evaluate(
            smiles_list,
            iter_idx,
            total,
            raw_best,
            feasible_best,
            phase,
        ):
            return (
                total + len(smiles_list),
                raw_best,
                feasible_best,
                bool(smiles_list),
            )

        optimizer.generate_budget_completion_batch = generate
        optimizer.evaluate_lead_batch = evaluate

        total, _, _ = optimizer.run_budget_completion(0, 10.0, 10.0)

        self.assertEqual(total, 3)
        self.assertGreater(attempts, optimizer.args.budget_completion_max_iterations)
        self.assertIn("restart", routes)

    def test_joint_feasibility_is_rewarded(self):
        parent = make_item(9.8, 0.70, 0.80, 0.70)
        strict_child = make_item(10.2, 0.70, 0.80, 0.70)
        result = transition_reward(parent, strict_child)

        self.assertTrue(strict_child["strict"])
        self.assertEqual(result["crossed"], 1)
        self.assertEqual(result["regressed"], 0)
        self.assertGreater(result["reward"], 0.45)

    def test_losing_a_satisfied_constraint_is_penalized(self):
        parent = make_item(9.8, 0.70, 0.80, 0.70)
        preserved = make_item(10.2, 0.70, 0.80, 0.70)
        regressed = make_item(10.2, 0.70, 0.80, 0.50)

        preserved_reward = transition_reward(parent, preserved)["reward"]
        regressed_result = transition_reward(parent, regressed)
        self.assertEqual(regressed_result["regressed"], 1)
        self.assertLess(regressed_result["reward"], preserved_reward)

    def test_joint_operator_needs_two_unresolved_constraints(self):
        one_deficit = [make_item(8.0, 0.70, 0.80, 0.70)]
        two_deficits = [make_item(8.0, 0.45, 0.80, 0.70)]

        one_need = archive_constraint_need(
            one_deficit,
            ("dock", "qed", "sa", "sim"),
            joint=True,
        )
        two_need = archive_constraint_need(
            two_deficits,
            ("dock", "qed", "sa", "sim"),
            joint=True,
        )
        self.assertGreater(two_need, one_need)

    def test_recovery_completes_the_available_pair_frontiers(self):
        multipliers = completion_recovery_multipliers(
            {"sq": [object()], "qd": [], "sd": [object()]},
            completion_boost=4.0,
            joint_boost=2.0,
        )

        self.assertEqual(multipliers["dock_refine"], 4.0)
        self.assertEqual(multipliers["quality_repair"], 4.0)
        self.assertEqual(multipliers["similarity_repair"], 1.0)
        self.assertEqual(multipliers["joint_repair"], 2.0)
        self.assertEqual(multipliers["start_repair"], 1.0)

    def test_recovery_restarts_when_no_pair_frontier_exists(self):
        multipliers = completion_recovery_multipliers(
            {"sq": [], "qd": [], "sd": []},
            start_boost=3.0,
            joint_boost=2.0,
        )

        self.assertEqual(multipliers["start_repair"], 3.0)
        self.assertEqual(multipliers["joint_repair"], 2.0)
        self.assertEqual(multipliers["dock_refine"], 1.0)

    def test_recovery_v2_state_uses_observed_frontier(self):
        common = {
            "iteration": 7,
            "warmup_iterations": 2,
            "has_strict": False,
        }
        self.assertEqual(
            recovery_v2_state(
                **common,
                has_stage_three=True,
                has_generated_similarity=True,
                pair_frontier_count=2,
            ),
            "complete",
        )
        self.assertEqual(
            recovery_v2_state(
                **common,
                has_stage_three=False,
                has_generated_similarity=False,
                pair_frontier_count=0,
            ),
            "seed_anchor",
        )
        self.assertEqual(
            recovery_v2_state(
                **common,
                has_stage_three=False,
                has_generated_similarity=True,
                pair_frontier_count=2,
            ),
            "bridge",
        )

    def test_recovery_v2_completion_prioritizes_observed_missing_constraint(self):
        multipliers = recovery_v2_operator_multipliers(
            "complete",
            {"sq": [object()], "qd": [], "sd": [object()]},
        )
        self.assertEqual(multipliers["dock_refine"], 7.0)
        self.assertEqual(multipliers["quality_repair"], 7.0)
        self.assertEqual(multipliers["similarity_repair"], 1.0)

    def test_recovery_v2_seed_anchor_uses_original_start(self):
        optimizer = CSDNetLeadOptimizer.__new__(CSDNetLeadOptimizer)
        optimizer.args = SimpleNamespace(
            sampler_profile="universal_frontier_recovery_v2",
            frontier_start_parent_prob=0.0,
            frontier_parent_top_k=10,
        )
        optimizer.integrated_state = "seed_anchor"
        optimizer.start_smiles = "CCO"
        start = {"smiles": "CCO"}
        optimizer.frontier_items = {"CCO": start}
        optimizer.frontier_archives = {
            "s": [{"smiles": "CCC"}],
            "near": [{"smiles": "CCCC"}],
        }

        self.assertIs(optimizer.frontier_choose_parent("start_repair"), start)
        self.assertIs(optimizer.frontier_choose_parent("joint_repair"), start)

    def test_recovery_v2_uses_first_docking_cache(self):
        optimizer = CSDNetLeadOptimizer.__new__(CSDNetLeadOptimizer)
        optimizer.args = SimpleNamespace(
            sampler_profile="universal_frontier_recovery_v2"
        )
        sentinel = object()
        optimizer.reward_cached = lambda smiles: sentinel
        self.assertIs(optimizer.reward(["CC"]), sentinel)

    def test_recovery_v3_waits_until_late_search(self):
        optimizer = CSDNetLeadOptimizer.__new__(CSDNetLeadOptimizer)
        optimizer.args = SimpleNamespace(
            sampler_profile="universal_frontier_recovery_v3",
            frontier_recovery_start_iter=8,
        )
        optimizer.integrated_state = "complete"
        optimizer.frontier_archives = {"strict": []}

        optimizer.frontier_iteration = 8
        self.assertFalse(optimizer.frontier_recovery_active())
        optimizer.frontier_iteration = 9
        self.assertTrue(optimizer.frontier_recovery_active())
        optimizer.frontier_archives["strict"] = [object()]
        self.assertFalse(optimizer.frontier_recovery_active())

    def test_recovery_v3_keeps_baseline_operator_params_before_activation(self):
        optimizer = CSDNetLeadOptimizer.__new__(CSDNetLeadOptimizer)
        optimizer.args = SimpleNamespace(
            sampler_profile="universal_frontier_recovery_v3",
            frontier_recovery_start_iter=8,
            frontier_start_remask=0.08,
            frontier_start_temperature=1.02,
            frontier_start_span_prob=0.84,
        )
        optimizer.frontier_iteration = 6
        optimizer.integrated_state = "complete"
        optimizer.frontier_archives = {"strict": []}

        self.assertEqual(
            optimizer.frontier_operator_params("start_repair"),
            (0.08, 1.02, 0.84, 0.90),
        )

    def test_completion_head_operators_never_leak_into_baseline_allocation(self):
        optimizer = CSDNetLeadOptimizer.__new__(CSDNetLeadOptimizer)
        optimizer.args = SimpleNamespace(
            frontier_legacy_fraction=0.60,
            frontier_legacy_fraction_late=0.40,
            frontier_legacy_warmup_iters=2,
            frontier_min_operator_fraction=0.06,
        )
        optimizer.frontier_iteration = 8
        optimizer.frontier_task_head_operators = (
            "pair_bridge",
            "feasible_dock_polish",
            "boundary_similarity_polish",
            "boundary_quality_polish",
        )
        optimizer.frontier_available_operator_scores = lambda: {
            "start_repair": 1.0,
            "dock_refine": 1.0,
            "feasible_dock_polish": 100.0,
            "boundary_similarity_polish": 100.0,
        }

        allocation = optimizer.frontier_universal_baseline_allocation(100)

        self.assertEqual(sum(allocation.values()), 100)
        self.assertEqual(allocation["legacy"], 40)
        self.assertNotIn("feasible_dock_polish", allocation)
        self.assertNotIn("boundary_similarity_polish", allocation)

    def test_v4_anchor_restart_uses_start_and_independent_plan_rng(self):
        optimizer = CSDNetLeadOptimizer.__new__(CSDNetLeadOptimizer)
        optimizer.args = SimpleNamespace(
            sampler_profile="lead_protected_completion_v4",
            completion_v4_anchor_remask=0.06,
            completion_v4_anchor_temperature=0.94,
            completion_v4_anchor_span_prob=0.90,
            completion_v4_anchor_max_span_tokens=3,
            completion_v4_anchor_max_atom_fraction=0.25,
        )
        optimizer.start_smiles = "CCOc1ccccc1"
        optimizer.frontier_items = {
            optimizer.start_smiles: {
                "smiles": optimizer.start_smiles,
                "residual": 0.8,
                "stage": 1,
                "root_id": "start",
                "depth": 0,
            }
        }
        optimizer.frontier_rng = random.Random(11)
        optimizer.completion_v4_rng = random.Random(29)
        optimizer.integrated_state = "late_anchor_probe"
        baseline_rng_state = optimizer.frontier_rng.getstate()

        proposal = optimizer.frontier_make_guided_proposal("no_pair_anchor_restart")

        self.assertIsNotNone(proposal)
        self.assertEqual(proposal["seed"], optimizer.start_smiles)
        self.assertTrue(proposal["peripheral"])
        self.assertEqual(proposal["planned_delta"], 0)
        self.assertEqual(proposal["rng_stream"], "completion_v4")
        self.assertEqual(optimizer.frontier_rng.getstate(), baseline_rng_state)

    def test_v5_seed_restart_falls_back_to_a_fixed_atom_span(self):
        optimizer = CSDNetLeadOptimizer.__new__(CSDNetLeadOptimizer)
        optimizer.args = SimpleNamespace(
            sampler_profile="lead_protected_completion_v5",
            completion_v4_anchor_remask=0.06,
            completion_v4_anchor_temperature=0.94,
            completion_v4_anchor_span_prob=0.90,
            completion_v4_anchor_max_span_tokens=3,
            completion_v4_anchor_max_atom_fraction=0.25,
        )
        optimizer.start_smiles = "c1ccccc1"
        optimizer.frontier_items = {
            optimizer.start_smiles: {
                "smiles": optimizer.start_smiles,
                "residual": 0.8,
                "stage": 1,
                "root_id": "start",
                "depth": 0,
            }
        }
        optimizer.frontier_rng = random.Random(11)
        optimizer.completion_v5_rng = random.Random(31)
        optimizer.integrated_state = "late_seed_probe"
        baseline_rng_state = optimizer.frontier_rng.getstate()

        proposal = optimizer.frontier_make_guided_proposal("completion_seed_restart")

        self.assertIsNotNone(proposal)
        self.assertEqual(proposal["seed"], optimizer.start_smiles)
        self.assertEqual(proposal["planned_delta"], 0)
        self.assertEqual(proposal["edit_strategy"], "fixed_atom_span")
        self.assertEqual(proposal["rng_stream"], "completion_v5")
        self.assertEqual(optimizer.frontier_rng.getstate(), baseline_rng_state)

    def test_v5_refills_unproductive_reserve_from_the_v1_policy(self):
        optimizer = CSDNetLeadOptimizer.__new__(CSDNetLeadOptimizer)
        optimizer.frontier_base_operators = ("legacy",)
        optimizer.frontier_operators = ("completion_seed_restart",)
        optimizer.frontier_task_head_operators = ("completion_seed_restart",)
        optimizer.frontier_universal_baseline_allocation = lambda total: {
            "legacy": total
        }
        optimizer.frontier_generate_operator = lambda operator, count, seen: (
            [f"mol-{len(seen) + idx}" for idx in range(count)],
            [{"operator": operator} for _ in range(count)],
        )

        generated, metadata = optimizer.frontier_refill_v5_reserve(7, set())

        self.assertEqual(len(generated), 7)
        self.assertEqual(len(metadata), 7)
        self.assertTrue(all(row["v5_refill"] for row in metadata))
        self.assertTrue(all(row["operator"] == "legacy" for row in metadata))

    def test_elastic_joint_selects_the_closest_pair_completion(self):
        optimizer = CSDNetLeadOptimizer.__new__(CSDNetLeadOptimizer)
        optimizer.start_smiles = "CCO"
        start = make_frontier_item("CCO", 10.0, 0.45, 0.80, 1.0)
        needs_dock = make_frontier_item("CCN", 9.0, 0.70, 0.80, 0.70)
        needs_similarity = make_frontier_item("CCC", 11.0, 0.70, 0.80, 0.58)
        optimizer.frontier_items = {
            row["smiles"]: row for row in (start, needs_dock, needs_similarity)
        }
        optimizer.frontier_archives = {
            "near": [needs_similarity, needs_dock, start],
            "s": [start, needs_dock],
            "sq": [needs_dock],
            "qd": [needs_similarity],
            "sd": [],
            "strict": [],
        }

        state = optimizer._elastic_joint_state()

        self.assertEqual(state, "similarity_completion")
        self.assertEqual(optimizer.elastic_joint_source, "qd")

    def test_elastic_joint_strict_archive_switches_to_docking_polish(self):
        optimizer = CSDNetLeadOptimizer.__new__(CSDNetLeadOptimizer)
        optimizer.start_smiles = "CCO"
        start = make_frontier_item("CCO", 10.0, 0.45, 0.80, 1.0)
        strict = make_frontier_item("CCN", 11.0, 0.70, 0.80, 0.70)
        optimizer.frontier_items = {"CCO": start, "CCN": strict}
        optimizer.frontier_archives = {
            "near": [strict, start],
            "s": [strict, start],
            "sq": [strict],
            "qd": [strict],
            "sd": [strict],
            "strict": [strict],
        }

        self.assertEqual(optimizer._elastic_joint_state(), "polish")
        self.assertEqual(optimizer.elastic_joint_source, "strict")

    def test_elastic_joint_similarity_plan_uses_a_one_token_trust_region(self):
        optimizer = CSDNetLeadOptimizer.__new__(CSDNetLeadOptimizer)
        optimizer.args = SimpleNamespace(
            direct_peripheral_probability=0.0,
            max_len=256,
        )
        optimizer.frontier_rng = random.Random(17)
        optimizer.start_smiles = "CCOc1ccccc1"
        optimizer.elastic_length_support = (4, 80)

        plan = optimizer._elastic_joint_plan(
            optimizer.start_smiles,
            "similarity_completion",
            learned_insertion=True,
        )

        self.assertEqual(plan["stop"] - plan["start"], 1)
        self.assertEqual(plan["length_mode"], "learned_insertion")
        self.assertEqual(plan["min_replacement_len"], 1)
        self.assertEqual(plan["max_replacement_len"], 2)

    def test_elastic_joint_profile_dispatches_before_generic_frontier(self):
        optimizer = CSDNetLeadOptimizer.__new__(CSDNetLeadOptimizer)
        optimizer.args = SimpleNamespace(sampler_profile="elastic_joint_frontier")
        optimizer.generate_batch_elastic_joint_frontier = lambda: ["CCN"]

        self.assertEqual(optimizer.generate_batch(), ["CCN"])

    def test_recursive_elastic_diagnostics_count_as_learned(self):
        self.assertTrue(is_learned_length_mode("learned_insertion"))
        self.assertTrue(is_learned_length_mode("learned_recursive_insertion"))
        self.assertFalse(is_learned_length_mode("fixed_fallback"))

    def test_elastic_joint_v2_stagnation_increases_escape_share(self):
        optimizer = CSDNetLeadOptimizer.__new__(CSDNetLeadOptimizer)
        optimizer.no_improve_iters = 0
        initial = optimizer._elastic_joint_v2_route_weights("similarity_completion")
        optimizer.no_improve_iters = 4
        stagnant = optimizer._elastic_joint_v2_route_weights("similarity_completion")

        self.assertAlmostEqual(sum(initial.values()), 1.0)
        self.assertAlmostEqual(sum(stagnant.values()), 1.0)
        self.assertGreater(stagnant["escape"], initial["escape"])
        self.assertLess(stagnant["route"], initial["route"])

    def test_elastic_joint_v2_routes_use_learned_direct_plan_states(self):
        self.assertEqual(
            CSDNetLeadOptimizer._elastic_joint_v2_plan_state(
                "similarity_completion",
                "route",
            ),
            "anchor",
        )
        self.assertEqual(
            CSDNetLeadOptimizer._elastic_joint_v2_plan_state(
                "dock_completion",
                "start",
            ),
            "polish",
        )
        self.assertEqual(
            CSDNetLeadOptimizer._elastic_joint_v2_plan_state(
                "quality_completion",
                "escape",
            ),
            "explore",
        )

    def test_elastic_joint_v2_cheap_rank_prefers_joint_feasibility(self):
        optimizer = CSDNetLeadOptimizer.__new__(CSDNetLeadOptimizer)
        optimizer.args = SimpleNamespace(sim_thr=0.6)
        feasible = {
            "smiles": "CCN",
            "cheap_qed": 0.61,
            "cheap_sa": 0.68,
            "cheap_similarity": 0.61,
        }
        high_similarity_but_low_quality = {
            "smiles": "CCO",
            "cheap_qed": 0.30,
            "cheap_sa": 0.80,
            "cheap_similarity": 0.95,
        }

        self.assertLess(
            optimizer._elastic_joint_v2_cheap_key(feasible, "anchor"),
            optimizer._elastic_joint_v2_cheap_key(
                high_similarity_but_low_quality,
                "anchor",
            ),
        )

    def test_elastic_joint_v2_dispatches_before_generic_frontier(self):
        optimizer = CSDNetLeadOptimizer.__new__(CSDNetLeadOptimizer)
        optimizer.args = SimpleNamespace(sampler_profile="elastic_joint_frontier_v2")
        optimizer.generate_batch_elastic_joint_frontier_v2 = lambda: ["CCN"]

        self.assertEqual(optimizer.generate_batch(), ["CCN"])

    def test_elastic_joint_v3_caps_high_similarity_escape_pressure(self):
        optimizer = CSDNetLeadOptimizer.__new__(CSDNetLeadOptimizer)
        optimizer.args = SimpleNamespace(sim_thr=0.6)
        optimizer.no_improve_iters = 20

        weights = optimizer._elastic_joint_v3_route_weights("similarity_completion")

        self.assertAlmostEqual(sum(weights.values()), 1.0)
        self.assertLessEqual(weights["escape"], 0.20)
        self.assertGreater(weights["start"], weights["escape"])

    def test_elastic_joint_v3_keeps_escape_edits_conservative_at_delta06(self):
        optimizer = CSDNetLeadOptimizer.__new__(CSDNetLeadOptimizer)
        optimizer.args = SimpleNamespace(sim_thr=0.6)

        self.assertEqual(
            optimizer._elastic_joint_v3_plan_state(
                "similarity_completion",
                "escape",
            ),
            "warmup",
        )
        self.assertEqual(
            optimizer._elastic_joint_v3_plan_state(
                "dock_completion",
                "route",
            ),
            "polish",
        )

    def test_elastic_joint_v3_preselection_reserves_frontier_buckets(self):
        optimizer = CSDNetLeadOptimizer.__new__(CSDNetLeadOptimizer)
        optimizer.args = SimpleNamespace(sim_thr=0.6)
        smiles = [f"row-{index}" for index in range(12)]
        optimizer.reward_qed = lambda _mols: [
            0.70,
            0.69,
            0.68,
            0.67,
            0.66,
            0.40,
            0.42,
            0.45,
            0.65,
            0.64,
            0.30,
            0.25,
        ]
        optimizer.reward_sa = lambda _mols: [0.75] * 12
        optimizer.reward_sim = lambda _mols: [
            0.70,
            0.68,
            0.66,
            0.64,
            0.62,
            0.72,
            0.69,
            0.65,
            0.57,
            0.56,
            0.30,
            0.20,
        ]

        _, metadata = optimizer._elastic_joint_v3_preselect(
            smiles,
            [{} for _ in smiles],
            "dock_completion",
            8,
        )

        buckets = [row["cheap_bucket"] for row in metadata]
        self.assertEqual(len(metadata), 8)
        self.assertEqual(buckets.count("joint_feasible"), 5)
        self.assertIn("similarity_safe", buckets)
        self.assertIn("joint_boundary", buckets)

    def test_elastic_joint_v3_dispatches_before_generic_frontier(self):
        optimizer = CSDNetLeadOptimizer.__new__(CSDNetLeadOptimizer)
        optimizer.args = SimpleNamespace(sampler_profile="elastic_joint_frontier_v3")
        optimizer.generate_batch_elastic_joint_frontier_v3 = lambda: ["CCN"]

        self.assertEqual(optimizer.generate_batch(), ["CCN"])

    def test_elastic_joint_v4_caps_escape_and_preserves_fixed_reserve(self):
        optimizer = CSDNetLeadOptimizer.__new__(CSDNetLeadOptimizer)
        optimizer.args = SimpleNamespace(sim_thr=0.6)
        optimizer.no_improve_iters = 20

        weights = optimizer._elastic_joint_v4_route_weights("quality_completion")

        self.assertAlmostEqual(sum(weights.values()), 1.0)
        self.assertLessEqual(weights["escape"], 0.03)
        self.assertGreater(
            optimizer._elastic_joint_v4_legacy_fraction("quality_completion"),
            optimizer._elastic_joint_v4_legacy_fraction("polish"),
        )

    def test_elastic_joint_v4_preselection_never_discards_feasible_first(self):
        optimizer = CSDNetLeadOptimizer.__new__(CSDNetLeadOptimizer)
        optimizer.args = SimpleNamespace(sim_thr=0.6)
        smiles = ["joint-a", "near", "joint-b", "far"]
        optimizer.reward_qed = lambda _mols: [0.70, 0.59, 0.65, 0.20]
        optimizer.reward_sa = lambda _mols: [0.75, 0.75, 0.70, 0.75]
        optimizer.reward_sim = lambda _mols: [0.62, 0.90, 0.61, 0.95]

        selected, metadata = optimizer._elastic_joint_v4_preselect(
            smiles,
            [{} for _ in smiles],
            "quality_completion",
            2,
        )

        self.assertEqual(set(selected), {"joint-a", "joint-b"})
        self.assertTrue(all(row["cheap_feasible"] for row in metadata))

    def test_elastic_joint_v4_dispatches_before_generic_frontier(self):
        optimizer = CSDNetLeadOptimizer.__new__(CSDNetLeadOptimizer)
        optimizer.args = SimpleNamespace(sampler_profile="elastic_joint_frontier_v4")
        optimizer.generate_batch_elastic_joint_frontier_v4 = lambda: ["CCN"]

        self.assertEqual(optimizer.generate_batch(), ["CCN"])

    def test_elastic_joint_v5_preselection_docks_only_joint_feasible(self):
        optimizer = CSDNetLeadOptimizer.__new__(CSDNetLeadOptimizer)
        optimizer.args = SimpleNamespace(
            sim_thr=0.6,
            frontier_archive_size=10,
        )
        optimizer.elastic_public_frontier = {}
        smiles = ["CCN", "CCO", "CCC"]
        optimizer.reward_qed = lambda _mols: [0.70, 0.55, 0.70]
        optimizer.reward_sa = lambda _mols: [0.75, 0.75, 0.60]
        optimizer.reward_sim = lambda _mols: [0.65, 0.90, 0.80]

        selected, metadata = optimizer._elastic_joint_v5_preselect(
            smiles,
            [{"root_id": value} for value in smiles],
            "warmup",
            100,
        )

        self.assertEqual(selected, ["CCN"])
        self.assertTrue(all(row["cheap_feasible"] for row in metadata))
        self.assertEqual(set(optimizer.elastic_public_frontier), {"CCO", "CCC"})

    def test_elastic_joint_v5_public_frontier_selects_missing_constraint(self):
        optimizer = CSDNetLeadOptimizer.__new__(CSDNetLeadOptimizer)
        optimizer.args = SimpleNamespace(
            sampler_profile="elastic_joint_frontier_v5",
            sim_thr=0.6,
        )
        optimizer.start_smiles = "CCO"
        optimizer.frontier_items = {"CCO": {"smiles": "CCO"}}
        optimizer.frontier_archives = {"strict": []}
        optimizer.elastic_public_frontier = {
            "CCN": {
                "smiles": "CCN",
                "cheap_qed": 0.45,
                "cheap_sa": 0.75,
                "cheap_similarity": 0.70,
                "cheap_sim_deficit": 0.0,
                "cheap_quality_deficit": 0.25,
            }
        }

        state = optimizer._elastic_joint_state()

        self.assertEqual(state, "quality")
        self.assertEqual(optimizer.elastic_joint_source, "public")

    def test_elastic_joint_v5_polish_stagnation_broadens_sources(self):
        optimizer = CSDNetLeadOptimizer.__new__(CSDNetLeadOptimizer)
        optimizer.args = SimpleNamespace(sim_thr=0.6)
        optimizer.no_improve_iters = 0
        initial = optimizer._elastic_joint_v5_route_weights("polish")
        optimizer.no_improve_iters = 10
        stagnant = optimizer._elastic_joint_v5_route_weights("polish")

        self.assertAlmostEqual(sum(initial.values()), 1.0)
        self.assertAlmostEqual(sum(stagnant.values()), 1.0)
        self.assertLess(stagnant["route"], initial["route"])
        self.assertGreater(
            stagnant["start"] + stagnant["escape"],
            initial["start"] + initial["escape"],
        )

    def test_elastic_joint_v5_restores_progressive_adaptive_span_plan(self):
        optimizer = CSDNetLeadOptimizer.__new__(CSDNetLeadOptimizer)
        optimizer.args = SimpleNamespace(sim_thr=0.6)
        optimizer.elastic_joint_source = "strict"
        optimizer.start_fp = AllChem.GetMorganFingerprintAsBitVect(
            Chem.MolFromSmiles("CCCO"),
            2,
            2048,
        )
        observed = {}

        def direct_plan(_smiles, state):
            observed["state"] = state
            return {
                "start": 1,
                "stop": 4,
                "length_mode": "learned_insertion",
            }

        optimizer._elastic_direct_plan = direct_plan

        plan = optimizer._elastic_joint_v5_plan("CCCO", "polish", "route")

        self.assertEqual(observed["state"], "polish")
        self.assertEqual(plan["start"], 1)
        self.assertEqual(plan["stop"], 4)
        self.assertEqual(plan["v5_trust_region_atoms"], 3)
        self.assertEqual(plan["v5_edit_arm"], "progressive_adaptive_span")

    def test_oracle_feasibility_gate_preserves_metadata_alignment(self):
        optimizer = CSDNetLeadOptimizer.__new__(CSDNetLeadOptimizer)
        optimizer.args = SimpleNamespace(
            oracle_feasible_only=True,
            sim_thr=0.6,
        )
        optimizer.current_candidate_meta = [
            {"candidate": "keep"},
            {"candidate": "low_qed"},
            {"candidate": "low_sa"},
            {"candidate": "low_sim"},
        ]
        optimizer.current_candidate_ops = ["a", "b", "c", "d"]
        optimizer.reward_qed = lambda _mols: [0.70, 0.59, 0.70, 0.70]
        optimizer.reward_sa = lambda _mols: [0.75, 0.75, 0.65, 0.75]
        optimizer.reward_sim = lambda _mols: [0.65, 0.65, 0.65, 0.59]

        kept = optimizer.filter_oracle_feasible_candidates(
            ["CCN", "CCO", "CCC", "CCCl"]
        )

        self.assertEqual(kept, ["CCN"])
        self.assertEqual(
            optimizer.current_candidate_meta[0]["candidate"],
            "keep",
        )
        self.assertTrue(optimizer.current_candidate_meta[0]["cheap_feasible"])
        self.assertEqual(optimizer.current_candidate_ops, ["a"])

    def test_minimum_oracle_calls_extend_beyond_nominal_iterations(self):
        optimizer = CSDNetLeadOptimizer.__new__(CSDNetLeadOptimizer)
        optimizer.args = SimpleNamespace(
            sampler_profile="elastic_joint_frontier_v5",
            oracle_budget=1000,
            min_oracle_calls=500,
            num_iter=15,
            budget_completion=False,
        )
        optimizer.resume_raw_best = 10.0
        optimizer.resume_feasible_best = 10.0
        optimizer.resume_total = 0
        optimizer.resume_iteration_offset = 0
        optimizer.frontier_iteration = 0
        iterations = []
        optimizer.generate_batch = lambda: [f"mol-{index}" for index in range(20)]

        def evaluate(smiles, iteration, total, raw_best, feasible_best):
            iterations.append(iteration)
            return total + len(smiles), raw_best, feasible_best, True

        optimizer.evaluate_lead_batch = evaluate

        optimizer.run()

        self.assertEqual(len(iterations), 25)
        self.assertEqual(iterations[-1], 25)

    def test_elastic_joint_v5_dispatches_before_generic_frontier(self):
        optimizer = CSDNetLeadOptimizer.__new__(CSDNetLeadOptimizer)
        optimizer.args = SimpleNamespace(sampler_profile="elastic_joint_frontier_v5")
        optimizer.generate_batch_elastic_joint_frontier_v5 = lambda: ["CCN"]

        self.assertEqual(optimizer.generate_batch(), ["CCN"])


if __name__ == "__main__":
    unittest.main()
