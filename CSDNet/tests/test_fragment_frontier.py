import pickle
import json
import random
import sys
import tempfile
import types
import unittest
from collections import Counter
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import patch

import pandas as pd
import torch

import CSDNet.exp.frag.run_native_projected_infill as native_runner

from CSDNet.exp.frag.direct_infill import (
    _sample_added_length,
    apply_native_gap_constraint_policy,
    build_masked_template,
    build_native_projected_template,
    load_length_prior,
    native_gap_insertion_rate_scale,
    native_nucleus_support,
    native_sampler_arm,
)
from CSDNet.exp.frag.length_search import NonParametricLengthController
from CSDNet.exp.frag.fragment_length_prior import (
    FRAGMENT_GAP_PRIOR_SCHEMA,
    FragmentGapLengthPrior,
    apply_prefill_lengths,
)
from CSDNet.exp.frag.build_zinc_geometry_length_prior import PriorAccumulator
from CSDNet.exp.frag.run_unified_frontier import stable_case_seed
from CSDNet.exp.frag.run_direct_infill_v2 import (
    GEOMETRY_LENGTH_PROFILES,
    QUALITY_FRONTIER_LENGTH_PROFILES,
    allocate_confidence_profiles,
    allocate_profiles,
    allocate_stratified_quantiles,
)
from CSDNet.exp.frag.task_head import (
    CANONICAL_TASKS,
    FragmentConstraintAdapter,
    FragmentConstraintAdapterV2,
    OPERATOR_PROFILES_V2,
    assess_candidate,
    available_operators,
    build_constraint_spec,
    build_seed_pool,
    build_seed_pool_v2,
    make_edit_plan,
    prepare_model_seed,
)
from CSDNet.optim.frontier import UnifiedFrontierEngine
from CSDNet.optim.length_policy import ProtectedLengthAllocator
from CSDNet.util.elastic_sampling import (
    _apply_local_gap_insertions,
    _build_local_infill_state,
    _filter_logits,
    _fit_gap_insertions_to_capacity,
    _position_constraint_sequences,
    sample_elastic_local_infill,
)
from CSDNet.util.tokenizer import SMILESTokenizer, tokenize_smiles


class FragmentFrontierTest(unittest.TestCase):
    ATOMIC_LENGTH_PRIOR = "data/zinc250k_csdnet_atomic_lengths_max256.json"

    @classmethod
    def setUpClass(cls):
        cls.data = pd.read_csv("data/fragments.csv")
        with open("csdnet_vocab.pkl", "rb") as handle:
            cls.tokenizer = SMILESTokenizer(pickle.load(handle))

    def test_every_benchmark_condition_has_valid_seeds_and_edit_plans(self):
        for task in CANONICAL_TASKS:
            for _, row in self.data.iterrows():
                case_seed = stable_case_seed(0, task, row["name"])
                random.seed(case_seed)
                spec = build_constraint_spec(task, row)
                seeds = build_seed_pool(spec, limit=16, rng=random.Random(case_seed))
                self.assertTrue(seeds, f"{task}:{spec.name}")
                self.assertTrue(
                    all(
                        assess_candidate(seed, spec).structural_success
                        for seed in seeds
                    ),
                    f"{task}:{spec.name}",
                )
                prepared = [
                    prepare_model_seed(seed, self.tokenizer, 128) for seed in seeds
                ]
                self.assertTrue(all(seed is not None for seed in prepared))
                operators = available_operators(spec)
                for operator in operators:
                    seed, plan = make_edit_plan(
                        seeds[0], spec.queries, operator, random.Random(case_seed)
                    )
                    self.assertIsNotNone(seed, f"{task}:{spec.name}:{operator}")
                    self.assertIsNotNone(plan, f"{task}:{spec.name}:{operator}")
                    self.assertLess(plan["start"], plan["stop"])

    def test_target_molecule_is_not_used_to_construct_seed_pool(self):
        row = self.data.iloc[0].copy()
        changed = row.copy()
        changed["smiles"] = "CC"
        spec_a = build_constraint_spec("motif_extension", row)
        spec_b = build_constraint_spec("motif_extension", changed)
        random.seed(19)
        seeds_a = build_seed_pool(spec_a, limit=10, rng=random.Random(19))
        random.seed(19)
        seeds_b = build_seed_pool(spec_b, limit=10, rng=random.Random(19))
        self.assertEqual(seeds_a, seeds_b)

    def test_v2_seed_grammar_expands_small_structural_seed_pools(self):
        row = self.data.iloc[0]
        for task in ("linker_design", "motif_extension"):
            spec = build_constraint_spec(task, row)
            v1 = build_seed_pool(spec, limit=48, rng=random.Random(31))
            v2 = build_seed_pool_v2(spec, limit=48, rng=random.Random(31))
            self.assertGreater(len(v2), len(v1), task)
            self.assertTrue(
                all(assess_candidate(seed, spec).structural_success for seed in v2)
            )

    def test_v2_exploration_profiles_delegate_length_to_insertion_head(self):
        for operator in (
            "anchor_growth",
            "structural_restart",
            "bridge_closure",
            "decoration_fill",
            "superstructure_expand",
        ):
            profile = OPERATOR_PROFILES_V2[operator]
            self.assertTrue(profile.learned_insertion)
            self.assertGreater(
                profile.max_growth_tokens + profile.max_shrink_tokens,
                0,
            )

    def test_linker_and_scaffold_morphing_share_random_stream(self):
        self.assertEqual(
            stable_case_seed(2, "linker_design", "BARICITINIB"),
            stable_case_seed(2, "scaffold_morphing", "BARICITINIB"),
        )

    def test_v1_engine_preserves_fixed_raw_budget(self):
        row = self.data.iloc[0]
        spec = build_constraint_spec("scaffold_decoration", row)
        operators = available_operators(spec)
        adapter = FragmentConstraintAdapter(warmup_attempts=25)
        engine = UnifiedFrontierEngine(
            adapter=adapter,
            operator_groups={"proposal": operators},
        )
        context = {
            "attempts": 25,
            "structural_success_rate": 0.10,
            "valid_rate": 0.80,
            "stagnant_rounds": 0,
            "largest_root_fraction": 0.20,
            "available_operators": operators,
            "geometry": spec.geometry,
        }
        state = engine.classify(**context)
        allocation = engine.allocate(25, state=state, context=context)
        self.assertEqual(sum(allocation["proposal"].values()), 25)
        self.assertGreaterEqual(allocation["proposal"]["legacy_completion"], 3)

    def test_structural_reward_ignores_property_metrics(self):
        adapter = FragmentConstraintAdapter()
        base = {
            "valid": True,
            "connected": True,
            "no_dummies": True,
            "preserved_fraction": 1.0,
            "strict": True,
            "structural_score": 1.0,
        }
        reward_a, parts_a = adapter.batch_reward([base])
        reward_b, parts_b = adapter.batch_reward(
            [{**base, "qed": 0.0, "sa": 10.0, "distance": 0.0, "diversity": 0.0}]
        )
        self.assertEqual(reward_a, reward_b)
        self.assertEqual(parts_a, parts_b)

    def test_v2_adapter_diversifies_and_rewards_incremental_novelty(self):
        adapter = FragmentConstraintAdapterV2(
            warmup_attempts=25,
            unique_target=0.70,
        )
        state = adapter.classify(
            attempts=25,
            structural_success_rate=0.95,
            unique_success_rate=0.15,
            valid_rate=0.99,
            stagnant_rounds=0,
            largest_root_fraction=0.20,
        )
        self.assertEqual(state, "diversify")

        base = {
            "valid": True,
            "connected": True,
            "no_dummies": True,
            "preserved_fraction": 1.0,
            "strict": True,
            "structural_score": 1.0,
            "lineage_credit": 1.0,
        }
        duplicate_reward, _ = adapter.batch_reward([{**base, "novel": False}])
        novel_reward, _ = adapter.batch_reward(
            [{**base, "novel": True, "qed": 0.0, "sa": 10.0}]
        )
        self.assertGreater(novel_reward, duplicate_reward)

    def test_direct_infill_templates_cover_every_constraint_without_target(self):
        lengths = load_length_prior(self.ATOMIC_LENGTH_PRIOR)
        for task in CANONICAL_TASKS:
            for _, row in self.data.iterrows():
                spec = build_constraint_spec(task, row)
                template = build_masked_template(
                    spec,
                    max_len=128,
                    length_prior=lengths,
                    min_added_tokens=4,
                    rng=random.Random(7),
                )
                self.assertTrue(template.edit_plans, f"{task}:{spec.name}")
                self.assertGreaterEqual(template.added_tokens, 1)
                self.assertLessEqual(
                    len(template.seed_smiles) + template.added_tokens,
                    512,
                )

    def test_direct_infill_does_not_use_target_molecule(self):
        row = self.data.iloc[0].copy()
        changed = row.copy()
        changed["smiles"] = "CC"
        lengths = [40]
        for task in CANONICAL_TASKS:
            first = build_masked_template(
                build_constraint_spec(task, row),
                max_len=128,
                length_prior=lengths,
                min_added_tokens=4,
                rng=random.Random(23),
            )
            second = build_masked_template(
                build_constraint_spec(task, changed),
                max_len=128,
                length_prior=lengths,
                min_added_tokens=4,
                rng=random.Random(23),
            )
            self.assertEqual(first, second, task)

    def test_native_projected_templates_open_only_required_gaps(self):
        expected_gaps = {
            "linker_design": 1,
            "scaffold_morphing": 1,
            "motif_extension": 1,
            "scaffold_decoration": 2,
            "superstructure_generation": 1,
        }
        for task, gap_count in expected_gaps.items():
            spec = build_constraint_spec(task, self.data.iloc[0])
            template = build_native_projected_template(
                spec,
                max_len=256,
                rng=random.Random(73),
            )
            self.assertEqual(len(template.edit_plans), gap_count, task)
            expected_constraint = (
                "chain_atom"
                if task in {"linker_design", "scaffold_morphing"}
                else None
            )
            self.assertTrue(
                all(
                    plan["length_mode"] == "learned_insertion"
                    and plan["initial_replacement_len"] == 1
                    and plan["min_replacement_len"] == 1
                    and "replacement_len" not in plan
                    and plan.get("token_constraint") == expected_constraint
                    for plan in template.edit_plans
                ),
                task,
            )
            _validate = [
                token
                for index, token in enumerate(
                    tokenize_smiles(template.seed_smiles)
                )
                if not any(
                    int(plan["start"]) <= index < int(plan["stop"])
                    for plan in template.edit_plans
                )
                and token not in self.tokenizer.vocab
            ]
            self.assertFalse(_validate, f"{task}: {_validate}")

            state = _build_local_infill_state(
                template.seed_smiles,
                list(template.edit_plans),
                tk=self.tokenizer,
                max_len=256,
                recursive_gap_insertions=True,
            )
            self.assertIsNotNone(state, task)
            self.assertEqual(len(state["gaps"]), gap_count, task)
            self.assertTrue(
                all(
                    gap["minimum"] == 1 and gap["inserted"] == 1
                    for gap in state["gaps"]
                ),
                task,
            )
            self.assertEqual(
                sum(token == self.tokenizer.mask_id for token in state["tokens"]),
                gap_count,
                task,
            )
            self.assertEqual(
                sum(anchor >= 0 for anchor in state["anchors"]),
                2 * gap_count,
                task,
            )
            self.assertEqual(
                sum(value == "chain_atom" for value in state["constraints"]),
                gap_count if expected_constraint else 0,
                task,
            )

    def test_geometry_adaptive_gap_constraints_do_not_use_task_names(self):
        row = self.data.iloc[0]
        expected = {
            "linker_design": 25,
            "scaffold_morphing": 25,
            "motif_extension": 0,
            "scaffold_decoration": 25,
            "superstructure_generation": 100,
        }
        for task, expected_count in expected.items():
            spec = build_constraint_spec(task, row)
            template = build_native_projected_template(
                spec,
                max_len=256,
                rng=random.Random(91),
            )
            constrained = []
            for attempt_index in range(100):
                plans, applied = apply_native_gap_constraint_policy(
                    template,
                    geometry=spec.geometry,
                    attempt_index=attempt_index,
                    case_seed=0,
                    policy="geometry_adaptive",
                )
                constrained.append(applied)
                self.assertTrue(
                    all(
                        plan.get("token_constraint") == "chain_atom"
                        for plan in plans
                    )
                    if applied
                    else all(
                        "token_constraint" not in plan for plan in plans
                    )
                )
            self.assertEqual(sum(constrained), expected_count, task)

    def test_geometry_calibrated_constraints_follow_only_attachment_geometry(self):
        row = self.data.iloc[0]
        expected = {
            "linker_design": 0,
            "scaffold_morphing": 0,
            "motif_extension": 0,
            "scaffold_decoration": 75,
            "superstructure_generation": 100,
        }
        for task, expected_count in expected.items():
            spec = build_constraint_spec(task, row)
            template = build_native_projected_template(
                spec,
                max_len=256,
                rng=random.Random(93),
            )
            constrained = [
                apply_native_gap_constraint_policy(
                    template,
                    geometry=spec.geometry,
                    attempt_index=index,
                    case_seed=0,
                    policy="geometry_calibrated",
                )[1]
                for index in range(100)
            ]
            self.assertEqual(sum(constrained), expected_count, task)

    def test_geometry_adaptive_insertion_rate_uses_attachment_geometry(self):
        expected = {
            "multi_anchor": 1.0,
            "single_attachment": 1.4,
            "multi_attachment": 1.0,
            "substructure_expand": 1.0,
        }
        for geometry, scale in expected.items():
            self.assertAlmostEqual(
                native_gap_insertion_rate_scale(
                    geometry=geometry,
                    base_scale=1.0,
                    policy="geometry_adaptive",
                ),
                scale,
            )
            self.assertAlmostEqual(
                native_gap_insertion_rate_scale(
                    geometry=geometry,
                    base_scale=0.5,
                    policy="uniform",
                ),
                0.5,
            )

    def test_nucleus_support_anneals_only_multi_anchor_geometry(self):
        expected = {
            "multi_anchor": (4, 1),
            "single_attachment": (1, 1),
            "multi_attachment": (1, 1),
            "substructure_expand": (1, 1),
        }
        for geometry, support in expected.items():
            self.assertEqual(
                native_nucleus_support(
                    geometry=geometry,
                    start=4,
                    end=1,
                    policy="multi_anchor_annealed",
                ),
                support,
            )
        self.assertEqual(
            native_nucleus_support(
                geometry="single_attachment",
                start=4,
                end=1,
                policy="uniform",
            ),
            (4, 1),
        )

    def test_fixed_diversity_portfolio_has_exact_predeclared_budget(self):
        for case_seed in (0, 17, 2026):
            arms = [
                native_sampler_arm(
                    attempt_index=index,
                    case_seed=case_seed,
                    exploration_fraction=0.2,
                    policy="fixed_diversity",
                )
                for index in range(100)
            ]
            self.assertEqual(arms.count("exploration"), 20)
            self.assertEqual(arms.count("core"), 80)
        self.assertEqual(
            native_sampler_arm(
                attempt_index=0,
                case_seed=0,
                exploration_fraction=1.0,
                policy="fixed_diversity",
            ),
            "exploration",
        )
        self.assertEqual(
            native_sampler_arm(
                attempt_index=0,
                case_seed=0,
                exploration_fraction=0.2,
                policy="uniform",
            ),
            "core",
        )

    def test_prefill_guarded_portfolio_only_opens_upper_tail(self):
        for source, expected in (
            ("native_one_mask", "core"),
            ("zinc_lower_fixed_bin_24_gaps_1", "core"),
            ("zinc_middle_fixed_bin_24_gaps_1", "core"),
            ("zinc_upper_fixed_bin_24_gaps_1", "exploration"),
        ):
            self.assertEqual(
                native_sampler_arm(
                    attempt_index=0,
                    case_seed=17,
                    exploration_fraction=0.3,
                    policy="prefill_guarded",
                    prefill_source=source,
                ),
                expected,
            )
        with self.assertRaisesRegex(ValueError, "prefill source"):
            native_sampler_arm(
                attempt_index=0,
                case_seed=17,
                exploration_fraction=0.3,
                policy="prefill_guarded",
            )

    def test_zinc_gap_prior_stratifies_without_target_lengths(self):
        group = {
            "count": 80,
            "token_histogram": {"3": 20, "5": 20, "8": 20, "12": 20},
            "atom_histogram": {"2": 20, "3": 20, "5": 20, "7": 20},
        }
        geometry = {
            "groups": {"16:1": group},
            "gap_groups": {"1": group},
            "global": group,
        }
        prior = FragmentGapLengthPrior(
            {
                "schema": FRAGMENT_GAP_PRIOR_SCHEMA,
                "tokenizer": "csdnet_atomic_smiles",
                "fixed_bin_width": 8,
                "minimum_group_count": 32,
                "geometries": {
                    name: geometry
                    for name in (
                        "multi_anchor",
                        "single_attachment",
                        "multi_attachment",
                        "substructure_expand",
                    )
                },
            }
        )
        proposals = [
            prior.propose(
                geometry="multi_anchor",
                fixed_tokens=19,
                gap_count=1,
                attempt_index=index,
                constrained_atoms=False,
                maximum_total=20,
                maximum_per_gap=16,
                rng=random.Random(index),
            )
            for index in range(100)
        ]
        self.assertEqual(
            sum(item.source == "native_one_mask" for item in proposals),
            25,
        )
        self.assertEqual(
            sum(item.source.startswith("zinc_lower_") for item in proposals),
            50,
        )
        self.assertEqual(
            sum(item.source.startswith("zinc_middle_") for item in proposals),
            25,
        )
        self.assertTrue(all(1 <= item.lengths[0] <= 16 for item in proposals))
        for constraint_residue in range(4):
            sources = {
                (
                    "native"
                    if proposals[index].source == "native_one_mask"
                    else "lower"
                    if proposals[index].source.startswith("zinc_lower_")
                    else "middle"
                )
                for index in range(100)
                if index % 4 == constraint_residue
            }
            self.assertEqual(sources, {"native", "lower", "middle"})
        atom_proposal = prior.propose(
            geometry="multi_anchor",
            fixed_tokens=19,
            gap_count=1,
            attempt_index=1,
            constrained_atoms=True,
            maximum_total=20,
            maximum_per_gap=16,
            rng=random.Random(1),
        )
        self.assertEqual(atom_proposal.measure, "atoms")
        distant = prior.propose(
            geometry="multi_anchor",
            fixed_tokens=96,
            gap_count=1,
            attempt_index=1,
            constrained_atoms=False,
            maximum_total=20,
            maximum_per_gap=16,
            rng=random.Random(1),
        )
        self.assertIn("geometry_gaps_1", distant.source)

        calibrated_counts = {
            "multi_anchor": (5, 25, 70),
            "single_attachment": (5, 75, 20),
            "multi_attachment": (5, 25, 70),
            "substructure_expand": (100, 0, 0),
        }
        for geometry_name, expected_counts in calibrated_counts.items():
            calibrated = [
                prior.propose(
                    geometry=geometry_name,
                    fixed_tokens=19,
                    gap_count=1,
                    attempt_index=index,
                    constrained_atoms=False,
                    maximum_total=20,
                    maximum_per_gap=16,
                    rng=random.Random(index),
                    allocation_profile="geometry_calibrated",
                )
                for index in range(100)
            ]
            actual_counts = (
                sum(
                    item.source == "native_one_mask"
                    for item in calibrated
                ),
                sum(
                    item.source.startswith("zinc_lower_")
                    for item in calibrated
                ),
                sum(
                    item.source.startswith("zinc_middle_")
                    for item in calibrated
                ),
            )
            self.assertEqual(actual_counts, expected_counts, geometry_name)

        guarded_counts = {
            "multi_anchor": (10, 30, 30, 30),
            "single_attachment": (5, 75, 20, 0),
            "multi_attachment": (10, 30, 30, 30),
            "substructure_expand": (100, 0, 0, 0),
        }
        for geometry_name, expected_counts in guarded_counts.items():
            guarded = [
                prior.propose(
                    geometry=geometry_name,
                    fixed_tokens=19,
                    gap_count=1,
                    attempt_index=index,
                    constrained_atoms=False,
                    maximum_total=20,
                    maximum_per_gap=16,
                    rng=random.Random(index),
                    allocation_profile="diversity_guarded",
                )
                for index in range(100)
            ]
            actual_counts = (
                sum(item.source == "native_one_mask" for item in guarded),
                sum(item.source.startswith("zinc_lower_") for item in guarded),
                sum(item.source.startswith("zinc_middle_") for item in guarded),
                sum(item.source.startswith("zinc_upper_") for item in guarded),
            )
            self.assertEqual(actual_counts, expected_counts, geometry_name)
            upper_quantiles = [
                item.quantile
                for item in guarded
                if item.source.startswith("zinc_upper_")
            ]
            if expected_counts[-1]:
                self.assertGreaterEqual(min(upper_quantiles), 0.55)
                self.assertLess(max(upper_quantiles), 0.90)
                self.assertGreater(
                    max(upper_quantiles) - min(upper_quantiles),
                    0.30,
                )

    def test_prefill_lengths_respect_each_gap_hard_limit(self):
        plans = (
            {"initial_replacement_len": 1, "max_replacement_len": 4},
            {"initial_replacement_len": 1, "max_replacement_len": 9},
        )
        updated = apply_prefill_lengths(plans, (7, 3))
        self.assertEqual(updated[0]["initial_replacement_len"], 4)
        self.assertEqual(updated[1]["initial_replacement_len"], 3)
        self.assertEqual(plans[0]["initial_replacement_len"], 1)

    def test_prior_accumulator_keeps_atomic_and_token_lengths_distinct(self):
        accumulator = PriorAccumulator(bin_width=8)
        accumulator.add(
            "multi_anchor",
            fixed_tokens=19,
            gap_count=1,
            missing_tokens=8,
            missing_atoms=5,
        )
        group = accumulator.serialise()["multi_anchor"]["groups"]["16:1"]
        self.assertEqual(group["token_histogram"], {"8": 1})
        self.assertEqual(group["atom_histogram"], {"5": 1})

    def test_native_runner_persists_geometry_metadata_for_first_case(self):
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            fragments = root / "fragments.csv"
            self.data.iloc[[0]].to_csv(fragments, index=False)
            output = root / "results"
            args = SimpleNamespace(
                task="motif_extension",
                output_dir=str(output),
                fragments_csv=str(fragments),
                resume=False,
                seed=0,
                num_samples=1,
                n_steps=2,
                top_p=0.5,
                nucleus_min_tokens_start=4,
                nucleus_min_tokens_end=1,
                nucleus_support_policy="multi_anchor_annealed",
                sampler_portfolio_policy="uniform",
                diversity_fraction=0.0,
                diversity_temperature=1.08,
                diversity_top_p=0.8,
                diversity_nucleus_min_tokens_start=8,
                diversity_nucleus_min_tokens_end=2,
                diversity_unmask_selection="top_prob",
                gap_insertion_mode="recursive",
                trajectory_mode="coupled",
                planning_fraction=0.5,
                fill_mode="absorbing",
                fill_remask_power=0.8,
                fill_gumbel_scale=0.65,
                initial_gap_tokens=1,
                insertion_rate_scale=1.0,
                insertion_rate_policy="geometry_adaptive",
                gap_constraint_policy="geometry_adaptive",
                length_prefill_policy="native",
                temperature=1.0,
                max_len=256,
            )
            spec = build_constraint_spec(args.task, self.data.iloc[0])
            attempts = [
                {
                    "gap_constraint_applied": True,
                    "model_output": True,
                    "inserted_tokens": 3,
                    "actual_delta": 2,
                    "learned_inserted_tokens": 2,
                    "max_open_sites": 3,
                    "mean_open_site_rate": 0.25,
                    "forced_final_unmasks": 0,
                    "chain_atom_constrained_tokens": 3,
                    "initial_mask_tokens": 1,
                    "prefill_prior_total": 1,
                    "prefill_source": "native_one_mask",
                }
            ]
            metrics = {
                "validity": 1.0,
                "uniqueness": 1.0,
                "quality": 1.0,
                "diversity": 0.0,
                "distance": 0.0,
                "mean_qed": 0.7,
                "mean_sa": 3.0,
            }
            fake_tdc = types.ModuleType("tdc")
            fake_tdc.Evaluator = lambda *_args, **_kwargs: object()
            fake_tdc.Oracle = lambda *_args, **_kwargs: object()
            fake_model = SimpleNamespace(
                is_elastic=True,
                fragment_corruption_prob=0.15,
                kuma_shape_a=1.0,
            )
            with (
                patch.dict(sys.modules, {"tdc": fake_tdc}),
                patch.object(
                    native_runner,
                    "load_csdnet_model",
                    return_value=(fake_model, None, "cpu"),
                ),
                patch.object(
                    native_runner,
                    "run_case",
                    return_value=(spec, [spec.original], attempts, 1.4),
                ),
                patch.object(
                    native_runner,
                    "evaluate_samples",
                    return_value=(metrics, []),
                ),
            ):
                native_runner.run_task(args)

            frame = pd.read_csv(output / "metrics_motif_extension_seed0.csv")
            self.assertEqual(len(frame), 1)
            self.assertAlmostEqual(
                float(frame.loc[0, "effective_insertion_rate_scale"]),
                1.4,
            )
            self.assertAlmostEqual(
                float(frame.loc[0, "gap_constraint_fraction"]),
                1.0,
            )

    def test_nucleus_support_floor_prevents_atomic_top_p_collapse(self):
        logits = torch.tensor([[[8.0, 2.0, 1.0, 0.0, -1.0]]])
        collapsed = _filter_logits(logits, top_p=0.5)
        supported = _filter_logits(
            logits,
            top_p=0.5,
            min_tokens_to_keep=4,
        )
        self.assertEqual(int(collapsed.gt(-1e8).sum().item()), 1)
        self.assertEqual(int(supported.gt(-1e8).sum().item()), 4)

    def test_recursive_local_insertions_keep_new_subgaps_open(self):
        state = {
            "tokens": [self.tokenizer.bos_id, self.tokenizer.eos_id],
            "editable": [False, False],
            "anchors": [-1, 0],
            "gaps": [
                {
                    "minimum": 0,
                    "maximum": 8,
                    "removed": 0,
                    "initial": 0,
                    "inserted": 0,
                }
            ],
        }
        inserted = _apply_local_gap_insertions(
            state,
            [(1, 0, 3)],
            self.tokenizer.mask_id,
            recursive_gap_insertions=True,
        )
        self.assertEqual(inserted, 3)
        self.assertEqual(state["anchors"], [-1, 0, 0, 0, 0])
        self.assertEqual(state["gaps"][0]["inserted"], 3)

    def test_recursive_insertions_inherit_gap_token_constraint(self):
        state = {
            "tokens": [self.tokenizer.bos_id, self.tokenizer.eos_id],
            "editable": [False, False],
            "constraints": [None, None],
            "anchors": [-1, 0],
            "gaps": [
                {
                    "minimum": 0,
                    "maximum": 8,
                    "removed": 0,
                    "initial": 0,
                    "inserted": 0,
                    "constraint": "chain_atom",
                }
            ],
        }
        _apply_local_gap_insertions(
            state,
            [(1, 0, 3)],
            self.tokenizer.mask_id,
            recursive_gap_insertions=True,
        )
        self.assertEqual(
            state["constraints"],
            [None, "chain_atom", "chain_atom", "chain_atom", None],
        )

    def test_chain_atom_role_blocks_smiles_syntax_tokens(self):
        tokenizer = self.tokenizer

        class SyntaxBiasedModel(torch.nn.Module):
            is_elastic = True
            kuma_shape_a = 1.0

            def forward(
                self,
                input_ids,
                attention_mask,
                t=None,
                return_aux=False,
                rate_family=None,
            ):
                logits = torch.zeros(
                    *input_ids.shape,
                    tokenizer.vocab_size,
                    device=input_ids.device,
                )
                logits[..., tokenizer.vocab["="]] = 20.0
                logits[..., tokenizer.vocab["C"]] = 10.0
                if not return_aux:
                    return logits
                active = attention_mask.float()
                return {
                    "logits": logits,
                    "b_ins": torch.zeros_like(active),
                    "b_unmask": 100.0 * active,
                }

        generated = sample_elastic_local_infill(
            model=SyntaxBiasedModel(),
            tk=tokenizer,
            seed_smiles=["CO"],
            edit_plans=[[
                {
                    "start": 1,
                    "stop": 2,
                    "initial_replacement_len": 1,
                    "min_replacement_len": 1,
                    "max_replacement_len": 1,
                    "token_constraint": "chain_atom",
                }
            ]],
            max_len=8,
            device="cpu",
            batch_size=1,
            n_steps=2,
            use_fsm_check=False,
            use_rdkit_kekulize_check=False,
            temperature_start=1.0,
            temperature_end=1.0,
            top_p=1.0,
            recursive_gap_insertions=True,
            deterministic_final_unmask=True,
            return_diagnostics=True,
        )
        self.assertEqual(generated[0][0], "CC")
        self.assertEqual(
            generated[0][1]["chain_atom_constrained_tokens"],
            1,
        )
        self.assertEqual(generated[0][1]["fsm_constraint_mode"], "disabled")
        self.assertEqual(
            generated[0][1]["fsm_repair_progressive_steps"],
            8,
        )

    def test_simultaneous_recursive_insertions_share_each_gap_budget(self):
        bounded = _fit_gap_insertions_to_capacity(
            [(1, 0, 3), (2, 0, 3), (3, 1, 2)],
            capacity=10,
            gap_capacities={0: 4, 1: 1},
        )
        self.assertEqual(bounded, [(1, 0, 3), (2, 0, 1), (3, 1, 1)])

    def test_recursive_local_sampler_respects_gap_and_sequence_limits(self):
        tokenizer = self.tokenizer

        class HighInsertionModel(torch.nn.Module):
            is_elastic = True
            kuma_shape_a = 1.0

            def forward(
                self,
                input_ids,
                attention_mask,
                t=None,
                return_aux=False,
                rate_family=None,
            ):
                logits = torch.zeros(
                    *input_ids.shape,
                    tokenizer.vocab_size,
                    device=input_ids.device,
                )
                logits[..., tokenizer.vocab["C"]] = 10.0
                if not return_aux:
                    return logits
                active = attention_mask.float()
                return {
                    "logits": logits,
                    "b_ins": 100.0 * active,
                    "b_unmask": 100.0 * active,
                }

        torch.manual_seed(11)
        generated = sample_elastic_local_infill(
            model=HighInsertionModel(),
            tk=tokenizer,
            seed_smiles=["CC"],
            edit_plans=[
                [
                    {
                        "start": 1,
                        "stop": 2,
                        "initial_replacement_len": 1,
                        "min_replacement_len": 1,
                        "max_replacement_len": 6,
                    }
                ]
            ],
            max_len=10,
            device="cpu",
            batch_size=1,
            n_steps=8,
            use_fsm_check=False,
            use_rdkit_kekulize_check=False,
            temperature_start=1.0,
            temperature_end=1.0,
            top_p=0.5,
            max_insertions_per_step=4,
            recursive_gap_insertions=True,
            return_diagnostics=True,
        )
        self.assertEqual(len(generated), 1)
        smiles, diagnostics = generated[0]
        self.assertTrue(smiles)
        self.assertLessEqual(diagnostics["inserted_tokens"], 6)
        self.assertGreater(diagnostics["learned_inserted_tokens"], 0)
        self.assertGreater(diagnostics["max_open_sites"], 2)
        self.assertLessEqual(diagnostics["max_sequence_tokens"], 10)

    def test_plan_then_fill_separates_length_and_content_events(self):
        tokenizer = self.tokenizer

        class RecordingModel(torch.nn.Module):
            is_elastic = True
            kuma_shape_a = 1.0

            def __init__(self):
                super().__init__()
                self.mask_counts = []

            def forward(
                self,
                input_ids,
                attention_mask,
                t=None,
                return_aux=False,
                rate_family=None,
            ):
                self.mask_counts.append(
                    input_ids.eq(tokenizer.mask_id).sum(dim=1).tolist()
                )
                logits = torch.zeros(
                    *input_ids.shape,
                    tokenizer.vocab_size,
                    device=input_ids.device,
                )
                logits[..., tokenizer.vocab["C"]] = 10.0
                if not return_aux:
                    return logits
                active = attention_mask.float()
                return {
                    "logits": logits,
                    "b_ins": 4.0 * active,
                    "b_unmask": 100.0 * active,
                }

        model = RecordingModel()
        torch.manual_seed(17)
        generated = sample_elastic_local_infill(
            model=model,
            tk=tokenizer,
            seed_smiles=["CC"],
            edit_plans=[
                [
                    {
                        "start": 1,
                        "stop": 2,
                        "initial_replacement_len": 1,
                        "min_replacement_len": 1,
                        "max_replacement_len": 6,
                    }
                ]
            ],
            max_len=10,
            device="cpu",
            batch_size=1,
            n_steps=8,
            use_fsm_check=False,
            use_rdkit_kekulize_check=False,
            temperature_start=1.0,
            temperature_end=1.0,
            top_p=0.5,
            max_insertions_per_step=2,
            recursive_gap_insertions=True,
            trajectory_mode="plan_then_fill",
            planning_fraction=0.5,
            fill_mode="progressive_remask",
            return_diagnostics=True,
        )
        self.assertEqual(len(generated), 1)
        smiles, diagnostics = generated[0]
        self.assertTrue(smiles)
        self.assertEqual(diagnostics["trajectory_mode"], "plan_then_fill")
        self.assertEqual(diagnostics["fill_mode"], "progressive_remask")
        self.assertEqual(diagnostics["planning_steps"], 4)
        self.assertGreater(diagnostics["learned_inserted_tokens"], 0)
        self.assertEqual(diagnostics["forced_final_unmasks"], 0)
        # Every planning call sees all materialized editable slots still masked.
        self.assertTrue(all(count[0] >= 1 for count in model.mask_counts[:4]))

    def test_direct_infill_v2_respects_geometry_length_profiles(self):
        lengths = load_length_prior(self.ATOMIC_LENGTH_PRIOR)
        for task in CANONICAL_TASKS:
            spec = build_constraint_spec(task, self.data.iloc[0])
            bounds = GEOMETRY_LENGTH_PROFILES[spec.geometry]["quality"]
            template = build_masked_template(
                spec,
                max_len=128,
                length_prior=lengths,
                min_added_tokens=4,
                rng=random.Random(41),
                added_token_range=bounds,
            )
            self.assertGreaterEqual(template.added_tokens, bounds[0], task)
            self.assertLessEqual(template.added_tokens, bounds[1], task)

    def test_direct_infill_v2_profile_allocation_preserves_raw_budget(self):
        profiles = allocate_profiles(100, 0.80, random.Random(5))
        self.assertEqual(len(profiles), 100)
        self.assertEqual(profiles.count("quality"), 80)
        self.assertEqual(profiles.count("explore"), 20)

    def test_atomic_length_prior_is_converted_to_body_coordinates(self):
        from CSDNet.util.length_prior import ATOMIC_LENGTH_PRIOR_SCHEMA

        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "atomic_lengths.json"
            path.write_text(
                json.dumps(
                    {
                        "schema": ATOMIC_LENGTH_PRIOR_SCHEMA,
                        "tokenizer": "csdnet_atomic_smiles",
                        "include_special_tokens": True,
                        "lengths": [12, 14, 18],
                    }
                )
            )
            self.assertEqual(load_length_prior(path, max_len=18), [10, 12, 16])

    def test_empirical_quantiles_cover_feasible_conditional_prior(self):
        quantiles = allocate_stratified_quantiles(4, random.Random(3))
        targets = []
        for quantile in quantiles:
            target, added = _sample_added_length(
                fixed_tokens=10,
                attachment_count=1,
                max_body_tokens=30,
                length_prior=[8, 12, 16, 20, 24],
                min_added_tokens=4,
                rng=random.Random(5),
                length_quantile=quantile,
            )
            targets.append(target)
            self.assertEqual(added, target - 10)
        self.assertEqual(sorted(targets), [16, 20, 20, 24])

    def test_confidence_policy_keeps_baseline_and_exploration_floors(self):
        profiles = allocate_confidence_profiles(
            100,
            confidence_fraction=0.60,
            baseline_fraction=0.20,
            rng=random.Random(5),
        )
        self.assertEqual(len(profiles), 100)
        self.assertEqual(profiles.count("confidence_quality"), 60)
        self.assertEqual(profiles.count("quality"), 20)
        self.assertEqual(profiles.count("explore"), 20)

    def test_quality_frontier_profiles_keep_a_short_arm_and_exploration_floor(self):
        for geometry, profiles in QUALITY_FRONTIER_LENGTH_PROFILES.items():
            quality = profiles["quality"]
            explore = profiles["explore"]
            self.assertLessEqual(quality[0], quality[1], geometry)
            self.assertLess(quality[1], explore[0], geometry)
            self.assertLessEqual(explore[0], explore[1], geometry)

    def test_protected_total_allocator_preserves_all_three_floors(self):
        allocator = ProtectedLengthAllocator(
            total_quality_range=(32, 47),
            global_fraction=0.40,
            local_fraction=0.40,
            explore_fraction=0.20,
        )
        proposals = allocator.allocate(
            100,
            local_added_range=(9, 24),
            explore_added_range=(25, 48),
            rng=random.Random(11),
        )
        counts = Counter(proposal.arm for proposal in proposals)
        self.assertEqual(len(proposals), 100)
        self.assertEqual(counts["global_quality"], 40)
        self.assertEqual(counts["local_quality"], 40)
        self.assertEqual(counts["explore"], 20)
        self.assertTrue(
            all(0.0 <= proposal.quantile < 1.0 for proposal in proposals)
        )

    def test_total_length_support_shifts_above_a_large_fixed_fragment(self):
        low_target, low_added = _sample_added_length(
            fixed_tokens=50,
            attachment_count=2,
            max_body_tokens=126,
            length_prior=[40],
            min_added_tokens=4,
            rng=random.Random(7),
            target_length_range=(32, 47),
            length_quantile=0.0,
        )
        high_target, high_added = _sample_added_length(
            fixed_tokens=50,
            attachment_count=2,
            max_body_tokens=126,
            length_prior=[40],
            min_added_tokens=4,
            rng=random.Random(7),
            target_length_range=(32, 47),
            length_quantile=0.999,
        )
        self.assertEqual((low_target, low_added), (54, 4))
        self.assertEqual((high_target, high_added), (69, 19))

    def test_total_length_template_uses_complete_sequence_support(self):
        lengths = load_length_prior(self.ATOMIC_LENGTH_PRIOR)
        spec = build_constraint_spec("motif_extension", self.data.iloc[0])
        template = build_masked_template(
            spec,
            max_len=128,
            length_prior=lengths,
            min_added_tokens=4,
            rng=random.Random(41),
            target_length_range=(32, 47),
            length_quantile=0.5,
        )
        self.assertGreaterEqual(template.target_length, 32)
        self.assertLessEqual(template.target_length, 47)

    def test_nonparametric_length_controller_preserves_budget_and_bounds(self):
        controller = NonParametricLengthController(
            [6] * 20 + [8] * 40 + [10] * 20,
            minimum=4,
            maximum=14,
            warmup_attempts=20,
            prior_floor=0.65,
            exploration_fraction=0.10,
        )
        proposals = controller.allocate(20, random.Random(17))
        self.assertEqual(len(proposals), 20)
        self.assertTrue(
            all(4 <= proposal.added_tokens <= 14 for proposal in proposals)
        )
        self.assertTrue(all(proposal.source == "warmup" for proposal in proposals))
        counts = Counter(proposal.added_tokens for proposal in proposals)
        self.assertLessEqual(max(counts.values()) - min(counts.values()), 1)

    def test_nonparametric_length_controller_refines_successful_neighbours(self):
        controller = NonParametricLengthController(
            [8] * 100,
            minimum=4,
            maximum=14,
            warmup_attempts=0,
            prior_floor=0.0,
            exploration_fraction=0.0,
            ucb_scale=0.0,
            softmax_temperature=0.05,
            refinement_radius=2,
        )
        successful = controller.allocate(1, random.Random(3))[0]
        successful = type(successful)(8, "adaptive")
        failed = type(successful)(12, "adaptive")
        controller.update(
            [(successful, True, True)] * 20
            + [(failed, False, False)] * 20
        )
        self.assertTrue({6, 7, 8, 9, 10}.issubset(controller.active))
        proposals = controller.allocate(200, random.Random(9))
        near_success = sum(
            abs(proposal.added_tokens - 8) <= 2
            for proposal in proposals
        )
        near_failure = sum(
            abs(proposal.added_tokens - 12) <= 1
            for proposal in proposals
        )
        self.assertGreater(near_success, near_failure)

    def test_nonparametric_length_controller_uses_only_feasible_lengths(self):
        controller = NonParametricLengthController(
            [4, 5, 6, 10, 11, 12],
            minimum=4,
            maximum=12,
            warmup_attempts=0,
            prior_floor=0.50,
            exploration_fraction=0.25,
            feasible_lengths=[4, 6, 10, 12],
        )
        proposals = controller.allocate(200, random.Random(29))
        self.assertTrue(
            all(
                proposal.added_tokens in {4, 6, 10, 12}
                for proposal in proposals
            )
        )

    def test_structural_feasible_policy_matches_attachment_boundaries(self):
        row = self.data.iloc[0]
        expected = {
            "linker_design": "atom_bounded",
            "scaffold_morphing": "atom_bounded",
            "motif_extension": None,
            "scaffold_decoration": "atom_bounded",
            "superstructure_generation": "chain_atom",
        }
        for task, token_constraint in expected.items():
            spec = build_constraint_spec(task, row)
            template = build_native_projected_template(
                spec,
                max_len=256,
                rng=random.Random(101),
            )
            plans, constrained = apply_native_gap_constraint_policy(
                template,
                geometry=spec.geometry,
                attempt_index=0,
                case_seed=0,
                policy="structural_feasible",
            )
            self.assertEqual(constrained, token_constraint is not None, task)
            self.assertTrue(
                all(plan.get("token_constraint") == token_constraint for plan in plans),
                task,
            )

    def test_atom_bounded_constraint_tracks_dynamic_gap_boundaries(self):
        state = {
            "tokens": [
                self.tokenizer.bos_id,
                self.tokenizer.mask_id,
                self.tokenizer.mask_id,
                self.tokenizer.mask_id,
                self.tokenizer.eos_id,
            ],
            "constraints": [None] * 5,
            "gap_ids": [-1, 0, 0, 0, -1],
            "editable": [False, True, True, True, False],
            "gaps": [{"constraint": "atom_bounded"}],
        }
        resolved = _position_constraint_sequences([state])[0]
        self.assertEqual(
            resolved,
            [None, "chain_atom", None, "chain_atom", None],
        )

    def test_condition_repair_reopens_only_the_editable_gap(self):
        tokenizer = self.tokenizer

        class ConditionRepairModel(torch.nn.Module):
            is_elastic = True
            kuma_shape_a = 1.0

            def forward(
                self,
                input_ids,
                attention_mask,
                t=None,
                return_aux=False,
                rate_family=None,
            ):
                logits = torch.zeros(
                    *input_ids.shape,
                    tokenizer.vocab_size,
                    device=input_ids.device,
                )
                preferred = "O" if return_aux else "C"
                logits[..., tokenizer.vocab[preferred]] = 30.0
                if not return_aux:
                    return logits
                active = attention_mask.float()
                return {
                    "logits": logits,
                    "b_ins": -100.0 * active,
                    "b_unmask": 100.0 * active,
                }

        generated = sample_elastic_local_infill(
            model=ConditionRepairModel(),
            tk=tokenizer,
            seed_smiles=["CO"],
            edit_plans=[[
                {
                    "start": 1,
                    "stop": 2,
                    "initial_replacement_len": 1,
                    "min_replacement_len": 1,
                    "max_replacement_len": 1,
                }
            ]],
            max_len=8,
            device="cpu",
            batch_size=1,
            n_steps=2,
            use_fsm_check=False,
            use_rdkit_kekulize_check=False,
            max_sample_retries=1,
            temperature_start=1.0,
            temperature_end=1.0,
            sequence_validators=[lambda smiles: smiles == "CC"],
            return_diagnostics=True,
        )
        smiles, diagnostics = generated[0]
        self.assertEqual(smiles, "CC")
        self.assertEqual(
            diagnostics["condition_repair_initial_constraint_invalid_rows"],
            1,
        )
        self.assertEqual(
            diagnostics["condition_repair_final_constraint_invalid_rows"],
            0,
        )


if __name__ == "__main__":
    unittest.main()
