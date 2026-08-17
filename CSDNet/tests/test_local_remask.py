import random
import unittest
from types import SimpleNamespace
from unittest.mock import patch

import torch

from CSDNet.exp.denovo.sampler_profiles import SAMPLER_PROFILES
from CSDNet.exp.lead.run import CSDNetLeadOptimizer
from CSDNet.exp.lead.frontier import (
    allocate_counts,
    constraint_state,
    peripheral_edit_plan,
    upper_tail_reward,
)
from CSDNet.exp.pmo.optimizer import (
    apply_token_edit_plans,
    progressive_global_sampler_kwargs,
    resolve_local_sampler_profile,
    sample_csdnet_local_remask,
)
from CSDNet.util.tokenizer import SMILESTokenizer


class CountingCarbonModel(torch.nn.Module):
    def __init__(self, tokenizer):
        super().__init__()
        self.tokenizer = tokenizer
        self.calls = 0

    def forward(self, input_ids, attention_mask):
        self.calls += 1
        batch_size, seq_len = input_ids.shape
        logits = torch.full(
            (batch_size, seq_len, self.tokenizer.vocab_size),
            -100.0,
            device=input_ids.device,
        )
        logits[:, :, self.tokenizer.vocab["C"]] = 100.0
        return logits


class ElasticCountingCarbonModel(CountingCarbonModel):
    is_elastic = True
    kuma_shape_a = 2.0

    def forward(
        self,
        input_ids,
        attention_mask,
        t=None,
        return_aux=False,
        rate_family=None,
    ):
        del t, rate_family
        logits = super().forward(input_ids, attention_mask)
        if not return_aux:
            return logits
        rates = torch.ones_like(input_ids, dtype=torch.float)
        return {
            "logits": logits,
            "b_ins": rates,
            "b_unmask": rates,
        }


class VisibleTokenRefinementModel(CountingCarbonModel):
    corruption_level_conditioning = True

    def forward(
        self,
        input_ids,
        attention_mask,
        corruption_level=None,
    ):
        del attention_mask, corruption_level
        self.calls += 1
        batch_size, seq_len = input_ids.shape
        logits = torch.full(
            (batch_size, seq_len, self.tokenizer.vocab_size),
            -100.0,
            device=input_ids.device,
        )
        if input_ids.eq(self.tokenizer.mask_id).any():
            logits[:, :, self.tokenizer.vocab["C"]] = 100.0
        else:
            logits[:, :, self.tokenizer.vocab["O"]] = 100.0
        return logits


class MaskedProposalRepairModel(CountingCarbonModel):
    corruption_level_conditioning = True

    def forward(
        self,
        input_ids,
        attention_mask,
        corruption_level=None,
    ):
        del attention_mask
        self.calls += 1
        batch_size, seq_len = input_ids.shape
        logits = torch.full(
            (batch_size, seq_len, self.tokenizer.vocab_size),
            -100.0,
            device=input_ids.device,
        )
        token = "O" if corruption_level is None else "C"
        logits[:, :, self.tokenizer.vocab[token]] = 100.0
        return logits


class LocalRemaskRegressionTest(unittest.TestCase):
    def test_fixed_length_remask_creates_masks_and_commits_tokens(self):
        tokenizer = SMILESTokenizer(
            ["<pad>", "<mask>", "<bos>", "<eos>", "<unk>", "C", "O"]
        )
        model = CountingCarbonModel(tokenizer)
        random.seed(0)
        torch.manual_seed(0)

        generated = sample_csdnet_local_remask(
            model=model,
            tk=tokenizer,
            seed_smiles=["CC"],
            max_len=8,
            device=torch.device("cpu"),
            batch_size=1,
            n_steps=4,
            remask_fraction=0.5,
            min_remask_tokens=1,
            span_prob=1.0,
            use_fsm_check=False,
            use_rdkit_kekulize_check=False,
            temperature_start=0.1,
            temperature_end=0.1,
            length_delta_choices="0",
            length_edit_prob=0.0,
            return_seed_indices=True,
        )

        self.assertEqual(generated, [("CC", 0)])
        self.assertEqual(
            model.calls,
            1,
            "A committed token must not be sampled again unless it is remasked.",
        )

    def test_explicit_edit_plan_can_change_sequence_length(self):
        tokenizer = SMILESTokenizer(
            ["<pad>", "<mask>", "<bos>", "<eos>", "<unk>", "C", "O"]
        )
        model = CountingCarbonModel(tokenizer)
        generated = sample_csdnet_local_remask(
            model=model,
            tk=tokenizer,
            seed_smiles=["CCO"],
            max_len=8,
            device=torch.device("cpu"),
            batch_size=1,
            n_steps=2,
            use_fsm_check=False,
            use_rdkit_kekulize_check=False,
            edit_plans=[{"start": 1, "stop": 2, "delta": 1}],
            return_seed_indices=True,
        )
        self.assertEqual(generated, [("CCCO", 0)])

    def test_local_sampler_accepts_confidence_controls(self):
        tokenizer = SMILESTokenizer(
            ["<pad>", "<mask>", "<bos>", "<eos>", "<unk>", "C", "O"]
        )
        model = CountingCarbonModel(tokenizer)
        generated = sample_csdnet_local_remask(
            model=model,
            tk=tokenizer,
            seed_smiles=["CCO"],
            max_len=8,
            device=torch.device("cpu"),
            batch_size=1,
            n_steps=3,
            use_fsm_check=False,
            use_rdkit_kekulize_check=False,
            edit_plans=[{"start": 1, "stop": 2, "replacement_len": 1}],
            top_p=0.97,
            gumbel_scale=0.25,
            remask_power=1.2,
            return_seed_indices=True,
        )
        self.assertEqual(generated, [("CCO", 0)])

    def test_progressive_length_coupled_local_profile_is_explicit(self):
        tokenizer = SMILESTokenizer(
            ["<pad>", "<mask>", "<bos>", "<eos>", "<unk>", "C", "O"]
        )
        model = CountingCarbonModel(tokenizer)
        generated = sample_csdnet_local_remask(
            model=model,
            tk=tokenizer,
            seed_smiles=["CCO"],
            max_len=8,
            device=torch.device("cpu"),
            batch_size=1,
            n_steps=3,
            use_fsm_check=False,
            use_rdkit_kekulize_check=False,
            edit_plans=[{"start": 1, "stop": 2, "replacement_len": 1}],
            local_sampler_profile="progressive_length_coupled",
            return_seed_indices=True,
            return_diagnostics=True,
        )
        self.assertEqual(generated[0][0], "CCO")
        self.assertEqual(
            generated[0][2]["local_sampler_profile"],
            "progressive_length_coupled",
        )
        global_kwargs = progressive_global_sampler_kwargs(
            "progressive_length_coupled"
        )
        self.assertTrue(global_kwargs["progressive_commit"])
        self.assertTrue(global_kwargs["confidence_length_adaptive"])
        self.assertEqual(global_kwargs["temperature_start"], 1.2)
        self.assertEqual(global_kwargs["temperature_end"], 0.15)
        self.assertEqual(global_kwargs["gumbel_scale"], 0.65)
        self.assertEqual(global_kwargs["remask_power"], 1.35)
        self.assertEqual(global_kwargs["length_batching"], "sorted")
        expected_global = {
            key: value
            for key, value in SAMPLER_PROFILES[
                "promax_progressive_length_coupled"
            ].items()
            if key not in {
                "local_confidence_uses_editable_length",
                "local_sampling_uses_editable_length",
                "local_adaptive_length_low",
                "local_adaptive_length_high",
                "all_position_proposal_masked",
                "local_temperature_mode",
            }
        }
        self.assertEqual(global_kwargs, expected_global)

    def test_conditional_refinement_only_changes_the_editable_region(self):
        tokenizer = SMILESTokenizer(
            ["<pad>", "<mask>", "<bos>", "<eos>", "<unk>", "C", "O"]
        )
        model = VisibleTokenRefinementModel(tokenizer)
        generated = sample_csdnet_local_remask(
            model=model,
            tk=tokenizer,
            seed_smiles=["CCO"],
            max_len=8,
            device=torch.device("cpu"),
            batch_size=1,
            n_steps=2,
            use_fsm_check=False,
            use_rdkit_kekulize_check=False,
            edit_plans=[{"start": 1, "stop": 2, "replacement_len": 1}],
            local_sampler_profile="conditional_progressive_refine",
            return_seed_indices=True,
            return_diagnostics=True,
        )
        self.assertEqual(generated[0][0], "COO")
        self.assertEqual(generated[0][1], 0)
        self.assertEqual(generated[0][2]["refinement_edits"], 1)
        self.assertEqual(
            generated[0][2]["local_sampler_profile"],
            "conditional_progressive_refine",
        )

    def test_masked_fragment_refinement_uses_editable_length_and_repairs(self):
        tokenizer = SMILESTokenizer(
            ["<pad>", "<mask>", "<bos>", "<eos>", "<unk>", "C", "O"]
        )
        model = MaskedProposalRepairModel(tokenizer)
        generated = sample_csdnet_local_remask(
            model=model,
            tk=tokenizer,
            seed_smiles=["CCO"],
            max_len=8,
            device=torch.device("cpu"),
            batch_size=1,
            n_steps=2,
            use_fsm_check=False,
            use_rdkit_kekulize_check=False,
            edit_plans=[{"start": 1, "stop": 2, "replacement_len": 1}],
            local_sampler_profile="conditional_masked_refine",
            return_seed_indices=True,
            return_diagnostics=True,
        )
        self.assertEqual(generated[0][0], "CCO")
        diagnostics = generated[0][2]
        self.assertEqual(diagnostics["refinement_edits"], 1)
        self.assertTrue(diagnostics["sampling_uses_editable_length"])
        self.assertTrue(diagnostics["confidence_uses_editable_length"])

    def test_task_local_profile_preserves_operator_temperature(self):
        tokenizer = SMILESTokenizer(
            ["<pad>", "<mask>", "<bos>", "<eos>", "<unk>", "C", "O"]
        )
        model = CountingCarbonModel(tokenizer)
        generated = sample_csdnet_local_remask(
            model=model,
            tk=tokenizer,
            seed_smiles=["CCO"],
            max_len=8,
            device=torch.device("cpu"),
            batch_size=1,
            n_steps=2,
            use_fsm_check=False,
            use_rdkit_kekulize_check=False,
            edit_plans=[{"start": 1, "stop": 2, "replacement_len": 1}],
            temperature_start=0.83,
            temperature_end=0.17,
            local_sampler_profile="task_adaptive_local",
            return_seed_indices=True,
            return_diagnostics=True,
        )
        diagnostics = generated[0][2]
        self.assertEqual(diagnostics["local_temperature_mode"], "operator_scaled")
        self.assertTrue(diagnostics["confidence_uses_editable_length"])
        self.assertAlmostEqual(diagnostics["temperature_start_long"], 0.83)
        self.assertAlmostEqual(diagnostics["temperature_end_long"], 0.17)
        self.assertEqual(diagnostics["refinement_edits"], 0)

    def test_task_profile_aliases_resolve_explicitly(self):
        self.assertEqual(
            resolve_local_sampler_profile("promax_task_adaptive_local"),
            "task_adaptive_local",
        )
        self.assertEqual(
            resolve_local_sampler_profile("task_refine"),
            "task_adaptive_refine",
        )
        self.assertEqual(
            resolve_local_sampler_profile("promax_fragment_masked_refine"),
            "conditional_masked_refine",
        )

    def test_lead_routes_refinement_only_to_conservative_repair_arms(self):
        with patch(
            "CSDNet.exp.lead.run.resolve_local_sampler_profile",
            return_value="task_adaptive_local",
        ):
            self.assertEqual(
                CSDNetLeadOptimizer.frontier_local_sampler_profile(
                    "quality_repair"
                ),
                "task_adaptive_refine",
            )
            self.assertEqual(
                CSDNetLeadOptimizer.frontier_local_sampler_profile(
                    "dock_refine"
                ),
                "task_adaptive_local",
            )

    def test_learned_local_insertion_can_shrink_without_manual_delta(self):
        tokenizer = SMILESTokenizer(
            ["<pad>", "<mask>", "<bos>", "<eos>", "<unk>", "C", "O"]
        )
        model = ElasticCountingCarbonModel(tokenizer)
        with patch(
            "CSDNet.util.elastic_sampling.torch.poisson",
            side_effect=lambda value: torch.zeros_like(value),
        ):
            generated = sample_csdnet_local_remask(
                model=model,
                tk=tokenizer,
                seed_smiles=["CCCO"],
                max_len=8,
                device=torch.device("cpu"),
                batch_size=1,
                n_steps=2,
                use_fsm_check=False,
                use_rdkit_kekulize_check=False,
                edit_plans=[
                    {
                        "start": 1,
                        "stop": 3,
                        "length_mode": "learned_insertion",
                        "min_replacement_len": 0,
                        "max_replacement_len": 3,
                    }
                ],
                return_seed_indices=True,
                return_diagnostics=True,
            )

        self.assertEqual(generated[0][0], "CO")
        self.assertEqual(generated[0][1], 0)
        self.assertEqual(generated[0][2]["actual_delta"], -2)

    def test_multiple_edit_plans_share_original_token_coordinates(self):
        edited, positions = apply_token_edit_plans(
            ["C", "O", "C", "N"],
            [
                {"start": 1, "stop": 2, "replacement_len": 2},
                {"start": 3, "stop": 4, "replacement_len": 1},
            ],
        )
        self.assertEqual(edited, ["C", "<mask>", "<mask>", "C", "<mask>"])
        self.assertEqual(positions, [2, 3, 5])

        tokenizer = SMILESTokenizer(
            ["<pad>", "<mask>", "<bos>", "<eos>", "<unk>", "C", "O", "N"]
        )
        model = CountingCarbonModel(tokenizer)
        generated = sample_csdnet_local_remask(
            model=model,
            tk=tokenizer,
            seed_smiles=["COCN"],
            max_len=10,
            device=torch.device("cpu"),
            batch_size=1,
            n_steps=2,
            use_fsm_check=False,
            use_rdkit_kekulize_check=False,
            edit_plans=[[
                {"start": 1, "stop": 2, "replacement_len": 2},
                {"start": 3, "stop": 4, "replacement_len": 1},
            ]],
            return_seed_indices=True,
        )
        self.assertEqual(generated, [("CCCCC", 0)])

    def test_peripheral_plan_keeps_ring_core_outside_edit_span(self):
        plan = peripheral_edit_plan(
            "CCOc1ccccc1",
            random.Random(0),
            delta=1,
            max_atom_fraction=0.4,
            max_span_tokens=8,
        )
        self.assertIsNotNone(plan)
        self.assertTrue(plan["peripheral"])
        self.assertEqual((plan["start"], plan["stop"]), (0, 3))

    def test_frontier_count_allocation_preserves_budget(self):
        counts = allocate_counts(
            40,
            {"start": 1.0, "dock": 2.0, "quality": 1.0},
            minimum_each=3,
        )
        self.assertEqual(sum(counts.values()), 40)
        self.assertTrue(all(value >= 3 for value in counts.values()))

    def test_frontier_search_margin_matches_strict_docking_resolution(self):
        tied = constraint_state(
            dock=9.8,
            qed=0.7,
            sa=0.8,
            similarity=0.7,
            start_dock=9.8,
            similarity_threshold=0.6,
            docking_margin=0.05,
            residual_l1_weight=0.10,
        )
        improved = constraint_state(
            dock=9.9,
            qed=0.7,
            sa=0.8,
            similarity=0.7,
            start_dock=9.8,
            similarity_threshold=0.6,
            docking_margin=0.05,
            residual_l1_weight=0.10,
        )
        self.assertFalse(tied["strict"])
        self.assertGreater(tied["residual"], 0.0)
        self.assertTrue(improved["strict"])
        self.assertEqual(improved["residual"], 0.0)

    def test_augmented_residual_credits_non_bottleneck_progress(self):
        parent = constraint_state(
            dock=5.0,
            qed=0.30,
            sa=0.8,
            similarity=0.7,
            start_dock=10.0,
            similarity_threshold=0.6,
            residual_l1_weight=0.20,
        )
        child = constraint_state(
            dock=5.0,
            qed=0.36,
            sa=0.8,
            similarity=0.7,
            start_dock=10.0,
            similarity_threshold=0.6,
            residual_l1_weight=0.20,
        )
        self.assertEqual(parent["max_deficit"], child["max_deficit"])
        self.assertLess(child["residual"], parent["residual"])

    def test_upper_tail_reward_is_order_invariant_and_keeps_rare_hits(self):
        rewards = [0.8, 0.6] + [-0.2] * 8
        forward = upper_tail_reward(rewards, 0.20, 2, 0.20)
        reverse = upper_tail_reward(list(reversed(rewards)), 0.20, 2, 0.20)
        self.assertAlmostEqual(forward, reverse)
        self.assertGreater(forward, 0.5)

    def test_frontier_bandit_updates_once_per_operator_batch(self):
        optimizer = CSDNetLeadOptimizer.__new__(CSDNetLeadOptimizer)
        optimizer.args = SimpleNamespace(
            sampler_profile="multi_frontier",
            sim_thr=0.6,
            frontier_docking_margin=0.05,
            frontier_residual_l1_weight=0.10,
            frontier_similarity_slack=0.08,
            frontier_archive_size=10,
            frontier_crossing_bonus=0.08,
            frontier_strict_bonus=0.15,
            frontier_reward_alpha=0.50,
            frontier_reward_tail_fraction=0.20,
            frontier_reward_tail_min=2,
            frontier_reward_mean_weight=0.20,
        )
        optimizer.start_prop = 10.0
        optimizer.frontier_archives = {
            label: [] for label in ("s", "sq", "qd", "sd", "strict")
        }
        optimizer.frontier_items = {}
        optimizer.frontier_operator_stats = {
            "dock_refine": {
                "attempted": 2.0,
                "accepted": 2.0,
                "evaluated": 0.0,
                "updates": 0.0,
                "reward_ema": 0.0,
                "last_batch_reward": 0.0,
                "strict": 0.0,
            }
        }
        optimizer.frontier_operators = ("dock_refine",)
        optimizer.current_candidate_meta = [
            {
                "operator": "dock_refine",
                "parent_residual": 0.5,
                "parent_stage": 1,
            },
            {
                "operator": "dock_refine",
                "parent_residual": 0.5,
                "parent_stage": 1,
            },
        ]

        optimizer.frontier_update_state(
            ["CC", "CCC"],
            ([10.1, 9.0], [0.7, 0.7], [0.8, 0.8], [0.7, 0.7]),
        )
        stats = optimizer.frontier_operator_stats["dock_refine"]
        self.assertEqual(stats["evaluated"], 2.0)
        self.assertEqual(stats["updates"], 1.0)
        self.assertGreater(stats["reward_ema"], 0.0)
        self.assertEqual(
            optimizer.current_candidate_meta[0]["operator_batch_reward"],
            optimizer.current_candidate_meta[1]["operator_batch_reward"],
        )

    def test_zero_output_proposals_receive_zero_bandit_rewards(self):
        optimizer = CSDNetLeadOptimizer.__new__(CSDNetLeadOptimizer)
        optimizer.args = SimpleNamespace(
            sampler_profile="transition_feasible",
            tf_operator_overgenerate_factor=1.0,
            tf_max_generation_rounds=1,
            tf_proposal_batch_size=2,
            max_len=8,
            batch_size=2,
            n_steps=2,
            min_remask_tokens=1,
            disable_fsm_check=True,
            disable_rdkit_kekulize_check=True,
            rdkit_check_interval=1,
            max_sample_retries=0,
            violation_neighborhood=1,
            temperature_end=0.2,
            temperature_power=1.0,
            min_atoms=1,
            max_atoms=20,
            tf_bandit_alpha=0.5,
            tf_positive_reward_threshold=0.6,
        )
        optimizer.model = object()
        optimizer.tk = object()
        optimizer.device = torch.device("cpu")
        optimizer.tf_seen_smiles = set()
        optimizer.tf_operators = ("local_micro",)
        optimizer.tf_context_stats = {}
        optimizer.tf_choose_operator = lambda context: "local_micro"
        optimizer.tf_operator_params = lambda operator: (0.1, 1.0, 0.7)
        optimizer.tf_make_proposal = lambda operator, context: {
            "seed": "CC",
            "parent_smiles": "CC",
            "parent_dock": 1.0,
            "operator": operator,
            "context": context,
        }

        with patch("CSDNet.exp.lead.run.sample_csdnet_local_remask", return_value=[]):
            generated, lineage, failed = optimizer.tf_generate_unique_batch(2, "warmup")

        self.assertEqual(generated, [])
        self.assertEqual(lineage, [])
        self.assertEqual(failed, ["local_micro", "local_micro"])

        rewards = {"local_micro": [0.0 for _ in failed]}
        optimizer.tf_update_arm_stats("warmup", rewards)
        stats = optimizer.tf_context_stats["warmup"]["local_micro"]
        self.assertEqual(stats["pulls"], 2.0)
        self.assertEqual(stats["ema"], 0.125)


if __name__ == "__main__":
    unittest.main()
