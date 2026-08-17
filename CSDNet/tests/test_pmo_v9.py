import inspect
import random
import tempfile
import unittest
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import patch

import yaml

from CSDNet.exp.pmo.optimizer import (
    CSDNetOptimizer,
    initialize_v9_prior,
    load_ref_lengths,
    progressive_global_sampler_kwargs,
    task_size_bounds,
)
from CSDNet.util.sampling import sample_csdnet
from CSDNet.exp.pmo.run import load_resume_buffer, result_path
from CSDNet.exp.pmo.v9 import (
    V9BatchBandit,
    V9LineageArchive,
    allocate_weighted_counts,
    batch_frontier_reward,
    classify_v9_state,
    v9_local_weights,
    v9_root_fraction,
)


class PMOV9PolicyTest(unittest.TestCase):
    def test_global_restart_filters_local_only_profile_arguments(self):
        values = progressive_global_sampler_kwargs("task_adaptive_local")
        accepted = set(inspect.signature(sample_csdnet).parameters)
        self.assertFalse(set(values) - accepted)
        self.assertTrue(values["progressive_commit"])
        self.assertTrue(values["confidence_length_adaptive"])
        self.assertNotIn("local_confidence_uses_editable_length", values)
        self.assertNotIn("local_temperature_mode", values)

    def test_pmo_accepts_validated_atomic_length_prior(self):
        from CSDNet.util.length_prior import ATOMIC_LENGTH_PRIOR_SCHEMA
        import json

        with tempfile.TemporaryDirectory() as tmpdir:
            path = Path(tmpdir) / "prior.json"
            path.write_text(
                json.dumps(
                    {
                        "schema": ATOMIC_LENGTH_PRIOR_SCHEMA,
                        "tokenizer": "csdnet_atomic_smiles",
                        "include_special_tokens": True,
                        "lengths": [12, 18, 18, 31],
                    }
                )
            )
            lengths = load_ref_lengths(
                data_dir=None,
                tk=object(),
                max_len=128,
                atomic_length_prior=path,
            )
        self.assertEqual(lengths, [12, 18, 18, 31])

    def test_task_reference_lengths_preserve_empirical_multiplicity(self):
        optimizer = CSDNetOptimizer.__new__(CSDNetOptimizer)
        optimizer.args = SimpleNamespace(max_len=32)
        optimizer.ref_lengths = [3, 7, 7, 9, 17, 31]
        self.assertEqual(
            optimizer._v9_task_reference_lengths(min_size=3, max_size=7),
            [7, 7, 9],
        )

    def test_no_prescreen_prior_never_loads_oracle_vocab(self):
        with patch(
            "CSDNet.exp.pmo.optimizer.load_pmo_fragments",
            side_effect=AssertionError("oracle fragment vocab must not be read"),
        ), patch(
            "CSDNet.exp.pmo.optimizer.load_pmo_motifs",
            side_effect=AssertionError("oracle motif vocab must not be read"),
        ):
            population, motifs, source = initialize_v9_prior(
                mode="iterative_remask_v9_no_prescreen",
                oracle_name="drd2",
                tk=object(),
                max_len=128,
                population_size=100,
                motif_limit=240,
            )

        self.assertEqual(population, [])
        self.assertEqual(motifs, [])
        self.assertEqual(source, "online_budgeted")

    def test_weighted_allocation_preserves_budget(self):
        counts = allocate_weighted_counts(
            97,
            [("tiny", 0.51), ("small", 0.31), ("shrink", 0.18)],
        )
        self.assertEqual(sum(counts.values()), 97)
        self.assertTrue(all(value > 0 for value in counts.values()))

    def test_batch_reward_is_order_invariant(self):
        scores = [0.91, 0.66, 0.42, 0.35]
        parents = [0.82, 0.70, None, 0.30]
        kwargs = {
            "before_scores": [0.88, 0.84, 0.80, 0.77, 0.70, 0.62],
            "before_top10": 0.7683,
            "after_top10": 0.7733,
            "frontier_scale": 0.01,
            "delta_scale": 0.08,
        }
        forward, forward_parts = batch_frontier_reward(
            scores=scores,
            parent_scores=parents,
            **kwargs,
        )
        reverse, reverse_parts = batch_frontier_reward(
            scores=list(reversed(scores)),
            parent_scores=list(reversed(parents)),
            **kwargs,
        )
        self.assertAlmostEqual(forward, reverse)
        self.assertEqual(forward_parts, reverse_parts)

    def test_lineage_slots_keep_independent_roots(self):
        archive = V9LineageArchive(score_slots=3, lineage_slots=3)
        root_a = archive.add_root("root-a", 0.90, "a", "attach_only", 1)
        archive.add_root("root-b", 0.72, "b", "motif_restart", 2)
        archive.add_root("root-c", 0.68, "c", "fragment_anchor", 3)
        for index in range(12):
            archive.add_child(
                f"a-child-{index}",
                0.89 - 0.001 * index,
                root_a,
                "elite_small",
                10 + index,
            )

        score_pool, lineage_pool = archive.parent_pools()
        self.assertEqual({row.root_id for row in score_pool}, {"a"})
        self.assertEqual({row.root_id for row in lineage_pool}, {"a", "b", "c"})
        chosen = {
            archive.choose_parent(random.Random(seed), lineage_probability=1.0).root_id
            for seed in range(20)
        }
        self.assertGreaterEqual(len(chosen), 2)

    def test_saturated_state_has_priority_and_disables_risky_edits(self):
        state = classify_v9_state(
            calls=5000,
            warmup_calls=1000,
            avg_top10=0.95,
            nonzero_rate=0.01,
            stagnant_calls=5000,
            largest_root_fraction=0.95,
            saturation_threshold=0.90,
            sparse_threshold=0.25,
            stagnation_patience=800,
            collapse_threshold=0.60,
        )
        self.assertEqual(state, "saturated")
        self.assertEqual(v9_root_fraction(state), 0.06)
        weights = v9_local_weights(state)
        self.assertNotIn("graph_swap", weights)
        self.assertNotIn("graph_expand", weights)
        self.assertNotIn("rescue_large", weights)

    def test_bandit_updates_once_for_an_operator_batch(self):
        bandit = V9BatchBandit(["elite_small"], alpha=0.5)
        bandit.update("elite_small", reward=0.8, evaluated=24)
        row = bandit.stats["elite_small"]
        self.assertEqual(row["batches"], 1.0)
        self.assertEqual(row["evaluated"], 24.0)
        self.assertAlmostEqual(row["ema"], 0.65)

    def test_public_benchmark_size_ranges(self):
        self.assertEqual(task_size_bounds("median1"), (10, 30))
        self.assertEqual(task_size_bounds("jnk3"), (30, 80))
        self.assertEqual(task_size_bounds("amlodipine_mpo"), (20, 40))

    def test_group_evaluation_filters_before_oracle_and_updates_once(self):
        class FakeOracle:
            def __init__(self):
                self.mol_buffer = {}

            @property
            def finish(self):
                return len(self.mol_buffer) >= 10

        optimizer = CSDNetOptimizer.__new__(CSDNetOptimizer)
        optimizer.oracle = FakeOracle()
        optimizer.tk = object()
        optimizer.args = SimpleNamespace(
            max_len=128,
            v9_frontier_min_scale=0.01,
            v9_delta_min_scale=0.03,
            v9_delayed_credit_alpha=0.05,
            v9_archive_percentile=0.90,
        )
        optimizer._v9_rejections = {
            "invalid": 0,
            "duplicate": 0,
            "out_of_bounds": 0,
            "untokenizable": 0,
        }
        scores = {"CC": 0.70, "CCC": 0.90}

        def score_and_record(smiles, _):
            score = scores[smiles]
            optimizer.oracle.mol_buffer[smiles] = [
                score,
                len(optimizer.oracle.mol_buffer) + 1,
            ]
            return smiles, score

        optimizer._score_and_record = score_and_record
        optimizer._update_fragment_population_v2 = lambda *args: None
        optimizer._update_v5_motif_archive = lambda *args: None
        archive = V9LineageArchive(score_slots=4, lineage_slots=2)
        root_bandit = V9BatchBandit(
            ["attach_only", "motif_restart", "fragment_anchor"]
        )
        local_bandit = V9BatchBandit(["elite_small"])
        lineage = [
            ("CC", {"seed": "CC", "parent_record": None, "root_operator": "attach_only"}),
            ("CCC", {"seed": "CCC", "parent_record": None, "root_operator": "attach_only"}),
            ("CCCC", {"seed": "CCCC", "parent_record": None, "root_operator": "attach_only"}),
        ]

        with tempfile.TemporaryDirectory() as tmp, patch(
            "CSDNet.exp.pmo.optimizer.tokenizable", return_value=True
        ):
            evaluated, gain = optimizer._v9_evaluate_group(
                oracle_name="test",
                state="warmup",
                operator="attach_only",
                lineage=lineage,
                archive=archive,
                adaptive_population=[],
                motifs=[],
                min_size=2,
                max_size=3,
                csv_path=str(Path(tmp) / "scores.csv"),
                transition_path=str(Path(tmp) / "transitions.csv"),
                root_bandit=root_bandit,
                local_bandit=local_bandit,
                frontier_history=[],
                delta_history=[],
            )
            transition_lines = (Path(tmp) / "transitions.csv").read_text().splitlines()

        self.assertEqual(evaluated, 2)
        self.assertGreater(gain, 0.0)
        self.assertEqual(len(optimizer.oracle.mol_buffer), 2)
        self.assertEqual(optimizer._v9_rejections["out_of_bounds"], 1)
        self.assertEqual(len(archive.roots), 2)
        self.assertEqual(root_bandit.stats["attach_only"]["batches"], 1.0)
        self.assertEqual(len(transition_lines), 3)

    def test_resume_buffer_restores_original_call_order(self):
        with tempfile.TemporaryDirectory() as tmp:
            path = result_path(tmp, "iterative_remask_v9", "median2", 0)
            with open(path, "w") as handle:
                yaml.safe_dump(
                    {
                        "CCC": [0.90, 3],
                        "C": [0.20, 1],
                        "CC": [0.50, 2],
                    },
                    handle,
                    sort_keys=False,
                )
            restored = load_resume_buffer(
                tmp,
                "iterative_remask_v9",
                "median2",
                0,
                10000,
            )

        self.assertEqual(list(restored), ["C", "CC", "CCC"])
        self.assertEqual([row[1] for row in restored.values()], [1, 2, 3])

    def test_resume_buffer_rejects_noncontiguous_history(self):
        with tempfile.TemporaryDirectory() as tmp:
            path = result_path(tmp, "iterative_remask_v9", "median2", 0)
            with open(path, "w") as handle:
                yaml.safe_dump({"C": [0.20, 1], "CCC": [0.90, 3]}, handle)
            with self.assertRaisesRegex(RuntimeError, "Non-contiguous"):
                load_resume_buffer(
                    tmp,
                    "iterative_remask_v9",
                    "median2",
                    0,
                    10000,
                )

    def test_resume_buffer_uses_longer_matching_score_csv(self):
        with tempfile.TemporaryDirectory() as tmp:
            path = result_path(tmp, "iterative_remask_v9", "median2", 0)
            with open(path, "w") as handle:
                yaml.safe_dump(
                    {
                        "CCC": [0.90, 3],
                        "C": [0.20, 1],
                        "CC": [0.50, 2],
                    },
                    handle,
                    sort_keys=False,
                )
            (Path(tmp) / "median2_0.csv").write_text(
                "C,0.2\nCC,0.5\nCCC,0.9\nCCCC,0.95\n"
            )
            restored = load_resume_buffer(
                tmp,
                "iterative_remask_v9",
                "median2",
                0,
                10000,
            )

        self.assertEqual(len(restored), 4)
        self.assertEqual(restored["CCCC"], [0.95, 4])

    def test_global_restart_keeps_hard_atom_range_and_unseen_only(self):
        optimizer = CSDNetOptimizer.__new__(CSDNetOptimizer)
        optimizer.model = object()
        optimizer.tk = object()
        optimizer.device = "cpu"
        optimizer.oracle = SimpleNamespace(mol_buffer={"CC": [0.4, 1]})
        optimizer.args = SimpleNamespace(
            max_len=32,
            batch_size=8,
            n_steps=5,
            disable_fsm_check=False,
            disable_rdkit_kekulize_check=False,
            rdkit_check_interval=2,
            max_sample_retries=1,
            violation_neighborhood=1,
            v5_rescue_temperature=1.5,
            temperature_start=1.1,
            temperature_end=0.2,
            temperature_power=1.5,
        )
        generated = ["CC", "CCC", "CCCC", "not-smiles"]
        with patch(
            "CSDNet.exp.pmo.optimizer.sample_csdnet",
            return_value=generated,
        ), patch("CSDNet.exp.pmo.optimizer.tokenizable", return_value=True):
            lineage = optimizer._v9_make_global_restart_lineage(
                count=4,
                min_size=2,
                max_size=3,
                attempt=1,
            )

        self.assertEqual([smiles for smiles, _ in lineage], ["CCC"])
        self.assertTrue(
            all(row["root_operator"] == "global_restart" for _, row in lineage)
        )


if __name__ == "__main__":
    unittest.main()
