import random
import unittest

from CSDNet.exp.pmo.v10 import BetaEventBandit, KnnUCBScreen
from CSDNet.optim.structure import (
    adaptive_peripheral_edit_plan,
    preserves_murcko_scaffold,
)


class PMOV10PolicyTest(unittest.TestCase):
    def test_event_bandit_prefers_observed_success(self):
        bandit = BetaEventBandit(("winner", "loser"), ucb_weight=0.0)
        bandit.update("winner", successes=8, trials=20)
        bandit.update("loser", successes=0, trials=20)
        weights = dict(bandit.weighted({"winner": 1.0, "loser": 1.0}))
        self.assertGreater(weights["winner"], weights["loser"])

    def test_knn_screen_uses_only_supplied_history(self):
        history = {
            "CCO": [0.9, 1],
            "CCCO": [0.8, 2],
            "c1ccccc1": [0.1, 3],
            "c1ccncc1": [0.2, 4],
        }
        proposals = [
            {"smiles": "CCOC", "operator": "a"},
            {"smiles": "CCCCO", "operator": "a"},
            {"smiles": "c1ccoc1", "operator": "b"},
        ]
        screen = KnnUCBScreen(
            k=2,
            history_limit=10,
            min_history=2,
            exploration_floor=0.0,
            beta_start=0.0,
            beta_end=0.0,
        )
        selected = screen.select(
            proposals,
            history=history,
            n_select=1,
            calls=100,
            max_calls=1000,
            rng=random.Random(0),
        )
        self.assertEqual(len(selected), 1)
        self.assertIn("screen_mean", selected[0])
        self.assertTrue(selected[0]["smiles"].startswith("C"))

    def test_peripheral_plan_freezes_ring_core(self):
        parent = "CCOc1ccccc1"
        plan = adaptive_peripheral_edit_plan(
            parent,
            random.Random(0),
            delta=1,
            target_atom_fraction=0.20,
        )
        self.assertIsNotNone(plan)
        self.assertTrue(plan["peripheral"])
        self.assertTrue(preserves_murcko_scaffold(parent, "COc1ccccc1"))
        self.assertFalse(preserves_murcko_scaffold(parent, "CCCC"))


if __name__ == "__main__":
    unittest.main()
