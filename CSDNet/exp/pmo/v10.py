"""Event-driven policy and online candidate screening for PMO V10."""

from __future__ import annotations

import math

import numpy as np
from rdkit import Chem, DataStructs
from rdkit.Chem import AllChem


class BetaEventBandit:
    """A reproducible beta-UCB bandit updated by observable success events."""

    def __init__(
        self,
        operators,
        prior_success=1.0,
        prior_failure=3.0,
        ucb_weight=0.30,
        min_multiplier=0.08,
    ):
        self.prior_success = max(1e-6, float(prior_success))
        self.prior_failure = max(1e-6, float(prior_failure))
        self.ucb_weight = max(0.0, float(ucb_weight))
        self.min_multiplier = max(0.0, float(min_multiplier))
        self.stats = {
            str(name): {"successes": 0.0, "trials": 0.0}
            for name in operators
        }

    def weighted(self, priors):
        total_trials = sum(row["trials"] for row in self.stats.values())
        weighted = []
        for name, prior in priors.items():
            prior = max(0.0, float(prior))
            if prior <= 0.0:
                continue
            row = self.stats.setdefault(
                str(name), {"successes": 0.0, "trials": 0.0}
            )
            alpha = self.prior_success + row["successes"]
            beta = self.prior_failure + row["trials"] - row["successes"]
            mean = alpha / max(alpha + beta, 1e-8)
            explore = self.ucb_weight * math.sqrt(
                math.log(total_trials + 2.0) / (row["trials"] + 1.0)
            )
            multiplier = max(self.min_multiplier, mean + explore)
            weighted.append((str(name), prior * multiplier))
        return weighted

    def update(self, name, successes, trials):
        row = self.stats.setdefault(
            str(name), {"successes": 0.0, "trials": 0.0}
        )
        trials = max(0, int(trials))
        successes = max(0, min(trials, int(successes)))
        row["successes"] += float(successes)
        row["trials"] += float(trials)

    def state_dict(self):
        return {
            "prior_success": self.prior_success,
            "prior_failure": self.prior_failure,
            "ucb_weight": self.ucb_weight,
            "min_multiplier": self.min_multiplier,
            "stats": self.stats,
        }

    def load_state_dict(self, state):
        for name, saved in dict((state or {}).get("stats", {})).items():
            trials = max(0.0, float(saved.get("trials", 0.0)))
            successes = max(
                0.0, min(trials, float(saved.get("successes", 0.0)))
            )
            self.stats[str(name)] = {
                "successes": successes,
                "trials": trials,
            }

    def snapshot(self):
        result = {}
        for name, row in sorted(self.stats.items()):
            alpha = self.prior_success + row["successes"]
            beta = self.prior_failure + row["trials"] - row["successes"]
            result[name] = {
                "successes": int(row["successes"]),
                "trials": int(row["trials"]),
                "posterior_mean": alpha / max(alpha + beta, 1e-8),
            }
        return result


class KnnUCBScreen:
    """Rank proposals using only molecules scored earlier in the same run."""

    def __init__(
        self,
        k=16,
        history_limit=2000,
        beta_start=0.85,
        beta_end=0.20,
        exploration_floor=0.15,
        min_history=200,
    ):
        self.k = max(1, int(k))
        self.history_limit = max(self.k, int(history_limit))
        self.beta_start = max(0.0, float(beta_start))
        self.beta_end = max(0.0, float(beta_end))
        self.exploration_floor = max(
            0.0, min(0.50, float(exploration_floor))
        )
        self.min_history = max(self.k, int(min_history))
        self._fp_cache = {}

    def _fingerprint(self, smiles):
        if smiles in self._fp_cache:
            return self._fp_cache[smiles]
        mol = Chem.MolFromSmiles(smiles)
        fp = None
        if mol is not None:
            fp = AllChem.GetMorganFingerprintAsBitVect(
                mol, radius=2, nBits=2048
            )
        self._fp_cache[smiles] = fp
        return fp

    def _history_rows(self, history):
        rows = [
            (smiles, float(value[0]), int(value[1]))
            for smiles, value in history.items()
            if self._fingerprint(smiles) is not None
        ]
        if len(rows) <= self.history_limit:
            return rows
        keep_top = self.history_limit // 2
        selected = sorted(rows, key=lambda row: row[1], reverse=True)[:keep_top]
        selected_smiles = {row[0] for row in selected}
        recent = sorted(rows, key=lambda row: row[2], reverse=True)
        for row in recent:
            if row[0] in selected_smiles:
                continue
            selected.append(row)
            selected_smiles.add(row[0])
            if len(selected) >= self.history_limit:
                break
        return selected

    @staticmethod
    def _rank01(values):
        values = np.asarray(values, dtype=float)
        if len(values) <= 1:
            return np.ones(len(values), dtype=float)
        order = np.argsort(np.argsort(values, kind="mergesort"), kind="mergesort")
        return order.astype(float) / float(len(values) - 1)

    def select(self, proposals, history, n_select, calls, max_calls, rng):
        proposals = list(proposals)
        n_select = min(max(0, int(n_select)), len(proposals))
        if n_select == 0:
            return []
        history_rows = self._history_rows(history)
        if len(history_rows) < self.min_history or len(proposals) <= n_select:
            selected = list(proposals)
            rng.shuffle(selected)
            for row in selected[:n_select]:
                row["screen_mean"] = ""
                row["screen_uncertainty"] = ""
                row["screen_acquisition"] = ""
            return selected[:n_select]

        history_fps = [self._fingerprint(row[0]) for row in history_rows]
        history_scores = np.asarray([row[1] for row in history_rows], dtype=float)
        valid = []
        means = []
        uncertainties = []
        for proposal in proposals:
            fp = self._fingerprint(proposal["smiles"])
            if fp is None:
                continue
            similarities = np.asarray(
                DataStructs.BulkTanimotoSimilarity(fp, history_fps),
                dtype=float,
            )
            k = min(self.k, len(similarities))
            if k < len(similarities):
                indices = np.argpartition(similarities, -k)[-k:]
            else:
                indices = np.arange(len(similarities))
            local_sim = similarities[indices]
            local_scores = history_scores[indices]
            weights = np.maximum(local_sim, 1e-3) ** 3
            mean = float(np.average(local_scores, weights=weights))
            variance = float(
                np.average((local_scores - mean) ** 2, weights=weights)
            )
            max_similarity = float(local_sim.max(initial=0.0))
            uncertainty = 0.5 * math.sqrt(max(0.0, variance)) + 0.5 * (
                1.0 - max_similarity
            )
            valid.append(proposal)
            means.append(mean)
            uncertainties.append(uncertainty)

        if len(valid) <= n_select:
            return valid
        progress = max(0.0, min(1.0, float(calls) / max(1, int(max_calls))))
        beta = self.beta_end + (self.beta_start - self.beta_end) * (
            1.0 - progress
        )
        acquisition = self._rank01(means) + beta * self._rank01(uncertainties)
        for idx, proposal in enumerate(valid):
            proposal["screen_mean"] = means[idx]
            proposal["screen_uncertainty"] = uncertainties[idx]
            proposal["screen_acquisition"] = float(acquisition[idx])

        explore_n = 0
        if self.exploration_floor > 0.0:
            explore_n = min(
                n_select,
                max(1, int(round(n_select * self.exploration_floor))),
            )
        exploit_n = n_select - explore_n
        ranked = list(np.argsort(-acquisition, kind="mergesort"))
        chosen_indices = ranked[:exploit_n]
        remaining = ranked[exploit_n:]
        rng.shuffle(remaining)
        chosen_indices.extend(remaining[:explore_n])
        return [valid[idx] for idx in chosen_indices]
