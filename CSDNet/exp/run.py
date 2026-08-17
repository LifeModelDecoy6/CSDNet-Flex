#!/usr/bin/env python
import argparse
import csv
import math
import os
import pickle
import random
import sys
from pathlib import Path
from time import time

import numpy as np
import pandas as pd
import torch
from rdkit import Chem, DataStructs, RDConfig, RDLogger
from rdkit.Chem import AllChem, QED

from CSDNet.exp.pmo.optimizer import (
    attach_fragments,
    canonical_smiles,
    clean_dummy_fragment,
    fragment_heavy_atom_count,
    load_csdnet_model,
    local_genmol_cut,
    sample_csdnet_local_remask,
    tokenizable,
)
from CSDNet.util.tokenizer import SMILESTokenizer


RDLogger.DisableLog("rdApp.*")
ROOT_DIR = Path(__file__).resolve().parent


def load_sa_scorer():
    try:
        from datamol.descriptors import sas
        return sas
    except Exception:
        pass

    contrib = os.path.join(RDConfig.RDContribDir, "SA_Score")
    if contrib not in sys.path:
        sys.path.append(contrib)
    try:
        import sascorer
        return sascorer.calculateScore
    except Exception as exc:
        raise SystemExit(
            "Could not load an SA scorer. Install datamol or make sure RDKit "
            "Contrib/SA_Score is available."
        ) from exc


def load_tokenizer(vocab_path):
    with open(vocab_path, "rb") as f:
        return SMILESTokenizer(pickle.load(f))


def atom_count(smiles):
    mol = Chem.MolFromSmiles(smiles)
    return mol.GetNumAtoms() if mol is not None else 0


def pareto_dominates(a, b, keys):
    return all(a[key] >= b[key] for key in keys) and any(a[key] > b[key] for key in keys)


def pareto_rank(items, keys):
    remaining = list(items)
    ranked = []
    rank = 0
    while remaining:
        front = []
        rest = []
        for item in remaining:
            dominated = any(
                pareto_dominates(other, item, keys)
                for other in remaining
                if other is not item
            )
            if dominated:
                rest.append(item)
            else:
                front.append(item)
        for item in front:
            item["pareto_rank"] = rank
        ranked.extend(front)
        remaining = rest
        rank += 1
    return ranked


def clamp(value, low, high):
    return max(low, min(high, value))


class CSDNetLeadOptimizer:
    """GenMol-style lead optimization with CSDNet local remasking.

    The protocol mirrors scripts/exps/lead/run.py from GenMol:
    seed from one known active, cut non-ring fragments, randomly reattach two
    fragments, locally remask the assembled molecule, then keep fragments from
    molecules with better docking score while satisfying QED/SA/similarity.
    """

    def __init__(self, args):
        self.args = args
        random.seed(args.seed)
        np.random.seed(args.seed)
        torch.manual_seed(args.seed)

        self.sa_scorer = load_sa_scorer()
        self.model, self.tk, self.device = load_csdnet_model(args)

        active_path = ROOT_DIR / "docking" / "actives.csv"
        df = pd.read_csv(active_path)
        df = df[df["target"] == args.oracle_name].reset_index(drop=True)
        if args.start_mol_idx >= len(df):
            raise SystemExit(
                f"start_mol_idx={args.start_mol_idx} is invalid for {args.oracle_name}; "
                f"available indices: 0..{len(df) - 1}"
            )

        row = df.iloc[args.start_mol_idx]
        self.start_smiles = canonical_smiles(row["smiles"])
        self.start_prop = float(row["DS"])
        start_mol = Chem.MolFromSmiles(self.start_smiles)
        self.start_fp = AllChem.GetMorganFingerprintAsBitVect(start_mol, 2, 2048)
        try:
            from CSDNet.exp.lead.docking.docking import DockingVina
        except ModuleNotFoundError as exc:
            raise SystemExit(
                "Lead optimization docking requires Open Babel Python bindings "
                "(`from openbabel import pybel`) and the `obabel` command. "
                "Install/load Open Babel before running this experiment."
            ) from exc
        self.predictor = DockingVina(args.oracle_name)

        fragments = sorted(local_genmol_cut(self.start_smiles))
        fragments = [frag for frag in fragments if Chem.MolFromSmiles(frag) is not None]
        if len(fragments) < 2:
            raise SystemExit(f"Too few initial fragments from {self.start_smiles}: {len(fragments)}")
        self.population = [(self.start_prop, frag) for frag in fragments]
        self.initial_fragments = list(fragments)
        self.tf_fragment_meta = {
            frag: {"prior": 0.50, "credit": 0.50, "updates": 0.0}
            for frag in fragments
        }
        self.elite_smiles = [(self.start_prop, 1.0, self.start_smiles)]
        self.dock_elite = []
        self.quality_elite = []
        self.near_miss_elite = []
        self.diverse_elite = []
        self.rescue_mode = False
        self.no_improve_iters = 0
        self.generated_seen = set()
        self.current_candidate_ops = []
        self.residual_length_ops = ("keep", "micro", "shrink", "expand", "symmetric", "rescue")
        self.residual_length_weights = {
            "keep": args.residual_keep_weight,
            "micro": args.residual_micro_weight,
            "shrink": args.residual_shrink_weight,
            "expand": args.residual_expand_weight,
            "symmetric": args.residual_symmetric_weight,
            "rescue": args.residual_rescue_weight,
        }
        self.residual_length_reward_ema = {op: 0.0 for op in self.residual_length_ops}
        self.residual_length_counts = {op: 0 for op in self.residual_length_ops}
        self.best_residual_score = -float("inf")
        self.last_length_bandit_summary = ""
        self.last_adaptive_mode = "fixed"
        self.tf_operators = (
            "local_micro",
            "local_small",
            "local_medium",
            "graph_swap",
            "graph_shrink",
            "graph_expand",
            "fragment_restart",
        )
        self.tf_context_stats = {}
        self.tf_items = {}
        self.tf_feasible_archive = []
        self.tf_near_archive = []
        self.tf_bottleneck_archives = {
            "similarity": [],
            "qed": [],
            "sa": [],
            "docking": [],
        }
        self.tf_seen_smiles = {self.start_smiles}
        self.tf_evaluated_smiles = set()
        self.tf_no_improve_batches = 0
        self.tf_best_key = None
        self.tf_last_context = "warmup"
        self.adaptive_base = {
            "remask_fraction": args.remask_fraction,
            "temperature_start": args.temperature_start,
            "span_prob": args.span_prob,
            "lead_elite_seed_prob": args.lead_elite_seed_prob,
            "lead_start_seed_prob": args.lead_start_seed_prob,
            "lead_elite_similarity_slack": args.lead_elite_similarity_slack,
            "pareto_similarity_slack": args.pareto_similarity_slack,
            "pareto_docking_weight": args.pareto_docking_weight,
            "pareto_similarity_weight": args.pareto_similarity_weight,
            "pareto_qed_weight": args.pareto_qed_weight,
            "pareto_sa_weight": args.pareto_sa_weight,
            "lead_elite_docking_weight": args.lead_elite_docking_weight,
            "lead_elite_similarity_weight": args.lead_elite_similarity_weight,
        }

        os.makedirs(args.output_dir, exist_ok=True)
        self.fname = os.path.join(
            args.output_dir,
            f"{args.oracle_name}_id{args.start_mol_idx}_thr{args.sim_thr}_{args.seed}.csv",
        )
        self.tf_transition_path = os.path.join(
            args.output_dir,
            f"transitions_{args.oracle_name}_id{args.start_mol_idx}_thr{args.sim_thr}_{args.seed}.csv",
        )
        self.tf_diagnostic_path = os.path.join(
            args.output_dir,
            f"diagnostics_{args.oracle_name}_id{args.start_mol_idx}_thr{args.sim_thr}_{args.seed}.csv",
        )
        if os.path.exists(self.fname) and not args.resume:
            os.remove(self.fname)
        if args.sampler_profile == "transition_feasible" and not args.resume:
            for path in (self.tf_transition_path, self.tf_diagnostic_path):
                if os.path.exists(path):
                    os.remove(path)

        start_qed = float(QED.qed(start_mol))
        start_sa = float((10.0 - self.sa_scorer(start_mol)) / 9.0)
        start_item = self.tf_make_item(
            smiles=self.start_smiles,
            dock=self.start_prop,
            qed=start_qed,
            sa=start_sa,
            sim=1.0,
            operator="start",
            parent_smiles=None,
        )
        self.tf_items[self.start_smiles] = start_item
        self.tf_near_archive = [start_item]
        if args.sampler_profile == "transition_feasible" and args.resume:
            self._load_transition_feasible_resume()

        print(f"Start SMILES:\t{self.start_smiles}")
        print(f"Start DS:\t{self.start_prop}")
        print(f"Initial population:\t{len(self.population)} fragments")
        print(f"Sampler profile:\t{args.sampler_profile}")
        print(f"Pareto population update:\t{args.pareto_population_update}")
        print(f"Remask fraction:\t{args.remask_fraction}")
        print(f"Output:\t{self.fname}")

    def reward_vina(self, smiles_list):
        reward = -np.array(self.predictor.predict(smiles_list), dtype=float)
        return np.clip(reward, 0.0, None).tolist()

    def reward_qed(self, mols):
        out = []
        for mol in mols:
            out.append(float(QED.qed(mol)) if mol is not None else 0.0)
        return out

    def reward_sa(self, mols):
        out = []
        for mol in mols:
            if mol is None:
                out.append(0.0)
                continue
            out.append(float((10.0 - self.sa_scorer(mol)) / 9.0))
        return out

    def reward_sim(self, mols):
        fps = []
        valid_idx = []
        for idx, mol in enumerate(mols):
            if mol is None:
                continue
            fps.append(AllChem.GetMorganFingerprintAsBitVect(mol, 2, 2048))
            valid_idx.append(idx)
        sims = [0.0] * len(mols)
        if fps:
            vals = DataStructs.BulkTanimotoSimilarity(self.start_fp, fps)
            for idx, val in zip(valid_idx, vals):
                sims[idx] = float(val)
        return sims

    def reward(self, smiles_list):
        mols = [Chem.MolFromSmiles(s) for s in smiles_list]
        return (
            self.reward_vina(smiles_list),
            self.reward_qed(mols),
            self.reward_sa(mols),
            self.reward_sim(mols),
        )

    def make_seed(self):
        if (
            self.args.sampler_profile in {"similarity_aware", "adaptive_similarity"}
            and self.elite_smiles
            and random.random() < self.args.lead_elite_seed_prob
        ):
            return self.make_elite_seed()
        return self.make_fragment_seed()

    def make_fragment_seed(self, fragments=None):
        if fragments is None:
            fragments = [frag for _, frag in self.population]
        if len(fragments) < 2:
            return None
        for _ in range(200):
            frag1, frag2 = random.sample(fragments, 2)
            smiles = attach_fragments(frag1, frag2)
            can = canonical_smiles(smiles) if smiles else None
            if can is None:
                continue
            atoms = atom_count(can)
            if atoms < self.args.min_atoms or atoms > self.args.max_atoms:
                continue
            if tokenizable(can, self.tk, self.args.max_len):
                return can
        return None

    def make_elite_seed(self):
        top_n = max(1, min(len(self.elite_smiles), self.args.lead_elite_size))
        if random.random() < self.args.lead_start_seed_prob:
            return self.start_smiles
        return random.choice(self.elite_smiles[:top_n])[2]

    def make_start_fragment_seed(self):
        if random.random() < self.args.lead_start_seed_prob:
            return self.start_smiles
        return self.make_fragment_seed(self.initial_fragments)

    def length_edit_kwargs(self):
        return {
            "length_delta_choices": self.args.length_delta_choices,
            "length_edit_prob": self.args.length_edit_prob,
            "length_edit_min_span": self.args.length_edit_min_span,
            "length_edit_max_span": self.args.length_edit_max_span,
        }

    def residual_length_operator_params(self, operator):
        """Task-agnostic length-edit operators for residual-guided lead search."""
        if operator == "keep":
            return (
                self.args.residual_keep_remask,
                self.args.residual_keep_temperature,
                self.args.residual_keep_span_prob,
                {
                    "length_delta_choices": "0",
                    "length_edit_prob": 0.0,
                    "length_edit_min_span": self.args.length_edit_min_span,
                    "length_edit_max_span": self.args.length_edit_max_span,
                },
            )
        if operator == "micro":
            return (
                self.args.residual_micro_remask,
                self.args.residual_micro_temperature,
                self.args.residual_micro_span_prob,
                {
                    "length_delta_choices": "-1:0:1",
                    "length_edit_prob": self.args.residual_micro_length_prob,
                    "length_edit_min_span": 1,
                    "length_edit_max_span": self.args.residual_micro_max_span,
                },
            )
        if operator == "shrink":
            return (
                self.args.residual_shrink_remask,
                self.args.residual_shrink_temperature,
                self.args.residual_shrink_span_prob,
                {
                    "length_delta_choices": self.args.residual_shrink_deltas,
                    "length_edit_prob": self.args.residual_shrink_length_prob,
                    "length_edit_min_span": self.args.residual_shrink_min_span,
                    "length_edit_max_span": self.args.residual_shrink_max_span,
                },
            )
        if operator == "expand":
            return (
                self.args.residual_expand_remask,
                self.args.residual_expand_temperature,
                self.args.residual_expand_span_prob,
                {
                    "length_delta_choices": self.args.residual_expand_deltas,
                    "length_edit_prob": self.args.residual_expand_length_prob,
                    "length_edit_min_span": self.args.residual_expand_min_span,
                    "length_edit_max_span": self.args.residual_expand_max_span,
                },
            )
        if operator == "rescue":
            return (
                self.args.residual_rescue_remask,
                self.args.residual_rescue_temperature,
                self.args.residual_rescue_span_prob,
                {
                    "length_delta_choices": self.args.residual_rescue_deltas,
                    "length_edit_prob": self.args.residual_rescue_length_prob,
                    "length_edit_min_span": self.args.residual_rescue_min_span,
                    "length_edit_max_span": self.args.residual_rescue_max_span,
                },
            )
        return (
            self.args.residual_symmetric_remask,
            self.args.residual_symmetric_temperature,
            self.args.residual_symmetric_span_prob,
            {
                "length_delta_choices": self.args.residual_symmetric_deltas,
                "length_edit_prob": self.args.residual_symmetric_length_prob,
                "length_edit_min_span": self.args.residual_symmetric_min_span,
                "length_edit_max_span": self.args.residual_symmetric_max_span,
            },
        )

    def residual_length_operator_probs(self):
        weights = {
            op: max(self.args.residual_min_operator_weight, self.residual_length_weights.get(op, 0.0))
            for op in self.residual_length_ops
        }
        if self.no_improve_iters >= self.args.residual_rescue_after_iters:
            weights["rescue"] *= self.args.residual_rescue_boost
            weights["symmetric"] *= 1.25
        if self.best_residual_score >= self.args.residual_refine_score:
            weights["keep"] *= 1.35
            weights["micro"] *= 1.25
            weights["rescue"] *= 0.60
        total = sum(weights.values())
        if total <= 0:
            return {op: 1.0 / len(self.residual_length_ops) for op in self.residual_length_ops}
        return {op: weight / total for op, weight in weights.items()}

    def choose_residual_length_operator(self):
        probs = self.residual_length_operator_probs()
        draw = random.random()
        cumulative = 0.0
        for op in self.residual_length_ops:
            cumulative += probs.get(op, 0.0)
            if draw <= cumulative:
                return op
        return self.residual_length_ops[-1]

    def make_residual_length_seed(self, operator):
        if operator in {"keep", "micro"}:
            return (
                self._archive_choice(self.near_miss_elite)
                or self._archive_choice(self.quality_elite)
                or self.make_elite_seed()
            )
        if operator == "shrink":
            return (
                self._archive_choice(self.dock_elite)
                or self._archive_choice(self.near_miss_elite)
                or self.make_elite_seed()
            )
        if operator == "expand":
            return (
                self._archive_choice(self.quality_elite)
                or self._archive_choice(self.near_miss_elite)
                or self.make_fragment_seed()
            )
        if operator == "rescue":
            return (
                self._archive_choice(self.diverse_elite)
                or self.make_start_fragment_seed()
                or self.make_fragment_seed()
            )
        return (
            self._archive_choice(self.near_miss_elite)
            or self._archive_choice(self.diverse_elite)
            or self.make_fragment_seed()
        )

    def residual_item(self, rv, rq, rs, rsim, smiles, operator=None):
        can = canonical_smiles(smiles)
        if can is None:
            return None
        sim_res = max(0.0, self.args.sim_thr - float(rsim)) / max(self.args.sim_thr, 1e-6)
        qed_res = max(0.0, 0.6 - float(rq)) / 0.6
        sa_thr = 6 / 9
        sa_res = max(0.0, sa_thr - float(rs)) / sa_thr
        dock_res = max(0.0, self.start_prop - float(rv)) / max(self.start_prop, 1e-6)
        residual_loss = (
            self.args.residual_sim_weight * sim_res
            + self.args.residual_qed_weight * qed_res
            + self.args.residual_sa_weight * sa_res
            + self.args.residual_dock_weight * dock_res
        )
        residual_score = 1.0 - residual_loss
        strict_ok = (
            float(rsim) >= self.args.sim_thr
            and float(rq) >= 0.6
            and float(rs) >= sa_thr
            and float(rv) > self.start_prop
        )
        if strict_ok:
            residual_score += self.args.residual_strict_score_bonus
        sim_norm = min(1.0, float(rsim) / max(self.args.sim_thr, 1e-6))
        qed_norm = min(1.0, float(rq) / 0.6)
        sa_norm = min(1.0, float(rs) / sa_thr)
        dock_ratio = float(rv) / max(self.start_prop, 1e-6)
        dock_norm = min(1.5, max(0.0, dock_ratio)) / 1.5
        return {
            "smiles": can,
            "operator": operator,
            "dock": float(rv),
            "qed": float(rq),
            "sa": float(rs),
            "sim": float(rsim),
            "dock_norm": float(dock_norm),
            "quality_score": float(0.55 * qed_norm + 0.45 * sa_norm),
            "near_score": float(max(0.0, min(1.0, 1.0 - residual_loss))),
            "residual_score": float(residual_score),
            "residual_loss": float(residual_loss),
            "sim_residual": float(sim_res),
            "qed_residual": float(qed_res),
            "sa_residual": float(sa_res),
            "dock_residual": float(dock_res),
            "sim_ok": float(rsim) >= self.args.sim_thr,
            "quality_ok": float(rq) >= 0.6 and float(rs) >= sa_thr,
            "dock_ok": float(rv) > self.start_prop,
            "strict_ok": bool(strict_ok),
            "soft_sim_ok": sim_res <= self.args.residual_soft_residual,
            "soft_quality_ok": max(qed_res, sa_res) <= self.args.residual_soft_residual,
            "pareto_score": float(
                self.args.residual_dock_weight * dock_norm
                + self.args.residual_sim_weight * sim_norm
                + self.args.residual_qed_weight * qed_norm
                + self.args.residual_sa_weight * sa_norm
            ),
        }

    def residual_items(self, smiles_list, prop_list, operators=None):
        rv_list, rq_list, rs_list, rsim_list = prop_list
        if operators is None:
            operators = [None] * len(smiles_list)
        items = []
        for rv, rq, rs, rsim, smiles, op in zip(rv_list, rq_list, rs_list, rsim_list, smiles_list, operators):
            item = self.residual_item(rv, rq, rs, rsim, smiles, op)
            if item is not None:
                items.append(item)
        return items

    def tf_make_item(
        self,
        smiles,
        dock,
        qed,
        sa,
        sim,
        operator=None,
        parent_smiles=None,
    ):
        can = canonical_smiles(smiles)
        if can is None:
            return None
        dock = float(dock)
        qed = float(qed)
        sa = float(sa)
        sim = float(sim)
        sim_res = max(0.0, self.args.sim_thr - sim) / max(self.args.sim_thr, 1e-8)
        qed_res = max(0.0, 0.6 - qed) / 0.6
        sa_thr = 6 / 9
        sa_res = max(0.0, sa_thr - sa) / sa_thr
        dock_res = max(0.0, self.start_prop - dock) / max(self.start_prop, 1e-8)
        dock_ok = dock > self.start_prop + 1e-9
        if not dock_ok:
            dock_res = max(dock_res, self.args.tf_boundary_epsilon)
        violations = {
            "similarity": float(sim_res),
            "qed": float(qed_res),
            "sa": float(sa_res),
            "docking": float(dock_res),
        }
        feasible = (
            sim >= self.args.sim_thr
            and qed >= 0.6
            and sa >= sa_thr
            and dock_ok
        )
        satisfied = sum(
            (
                sim >= self.args.sim_thr,
                qed >= 0.6,
                sa >= sa_thr,
                dock_ok,
            )
        )
        max_violation = max(violations.values())
        sum_violation = sum(violations.values())
        bottleneck = max(violations, key=violations.get)
        return {
            "smiles": can,
            "dock": dock,
            "qed": qed,
            "sa": sa,
            "sim": sim,
            "operator": operator,
            "parent_smiles": parent_smiles,
            "violations": violations,
            "max_violation": float(max_violation),
            "sum_violation": float(sum_violation),
            "constraints_satisfied": int(satisfied),
            "bottleneck": bottleneck,
            "feasible": bool(feasible),
        }

    @staticmethod
    def tf_rank_key(item):
        if item["feasible"]:
            margin = min(
                item["sim"],
                item["qed"],
                item["sa"],
                item["dock"],
            )
            return (0, -item["dock"], -margin)
        return (
            1,
            item["max_violation"],
            item["sum_violation"],
            -item["constraints_satisfied"],
            -item["dock"],
        )

    def tf_refresh_archives(self):
        items = list(self.tf_items.values())
        cap = self.args.tf_archive_size
        self.tf_feasible_archive = sorted(
            (item for item in items if item["feasible"]),
            key=lambda item: (-item["dock"], -item["sim"], -item["qed"]),
        )[:cap]
        self.tf_near_archive = sorted(
            (item for item in items if not item["feasible"]),
            key=self.tf_rank_key,
        )[:cap]
        for name in self.tf_bottleneck_archives:
            candidates = [
                item
                for item in self.tf_near_archive
                if item["bottleneck"] == name
            ]
            self.tf_bottleneck_archives[name] = candidates[:cap]

    def tf_context(self, evaluated_count):
        if evaluated_count < self.args.tf_warmup_evaluations:
            return "warmup"
        if self.tf_no_improve_batches >= self.args.tf_stagnation_batches:
            return "rescue"
        if self.tf_feasible_archive:
            return "feasible_refine"
        if not self.tf_near_archive:
            return "explore"
        bottleneck = self.tf_near_archive[0]["bottleneck"]
        return f"repair_{bottleneck}"

    def tf_select_parent(self, context):
        pool = []
        if context == "feasible_refine" and self.tf_feasible_archive:
            pool = self.tf_feasible_archive
        elif context.startswith("repair_"):
            name = context.removeprefix("repair_")
            pool = self.tf_bottleneck_archives.get(name, [])
        if not pool:
            pool = self.tf_near_archive or self.tf_feasible_archive
        if not pool:
            return self.tf_items[self.start_smiles]
        top_n = max(1, min(len(pool), self.args.tf_parent_top_k))
        return random.choice(pool[:top_n])

    @staticmethod
    def tf_base_operator_weights(context):
        weights = {
            "local_micro": 0.18,
            "local_small": 0.20,
            "local_medium": 0.12,
            "graph_swap": 0.18,
            "graph_shrink": 0.08,
            "graph_expand": 0.08,
            "fragment_restart": 0.16,
        }
        if context == "warmup":
            weights.update(
                {
                    "local_micro": 0.08,
                    "local_small": 0.12,
                    "local_medium": 0.15,
                    "graph_swap": 0.16,
                    "graph_shrink": 0.08,
                    "graph_expand": 0.10,
                    "fragment_restart": 0.31,
                }
            )
        elif context == "feasible_refine":
            weights.update(
                {
                    "local_micro": 0.38,
                    "local_small": 0.27,
                    "local_medium": 0.08,
                    "graph_swap": 0.13,
                    "graph_shrink": 0.05,
                    "graph_expand": 0.03,
                    "fragment_restart": 0.06,
                }
            )
        elif context == "repair_similarity":
            weights.update(
                {
                    "local_micro": 0.34,
                    "local_small": 0.29,
                    "local_medium": 0.08,
                    "graph_swap": 0.13,
                    "graph_shrink": 0.06,
                    "graph_expand": 0.03,
                    "fragment_restart": 0.07,
                }
            )
        elif context in {"repair_qed", "repair_sa"}:
            weights.update(
                {
                    "local_micro": 0.16,
                    "local_small": 0.22,
                    "local_medium": 0.12,
                    "graph_swap": 0.20,
                    "graph_shrink": 0.16,
                    "graph_expand": 0.05,
                    "fragment_restart": 0.09,
                }
            )
        elif context == "repair_docking":
            weights.update(
                {
                    "local_micro": 0.10,
                    "local_small": 0.17,
                    "local_medium": 0.20,
                    "graph_swap": 0.24,
                    "graph_shrink": 0.07,
                    "graph_expand": 0.14,
                    "fragment_restart": 0.08,
                }
            )
        elif context in {"rescue", "explore"}:
            weights.update(
                {
                    "local_micro": 0.05,
                    "local_small": 0.08,
                    "local_medium": 0.18,
                    "graph_swap": 0.22,
                    "graph_shrink": 0.10,
                    "graph_expand": 0.12,
                    "fragment_restart": 0.25,
                }
            )
        return weights

    def tf_operator_multipliers(self, context):
        stats_by_op = self.tf_context_stats.setdefault(
            context,
            {
                op: {"ema": 0.50, "pulls": 0.0, "positive": 0.0}
                for op in self.tf_operators
            },
        )
        total_pulls = max(
            1.0,
            sum(float(stats["pulls"]) for stats in stats_by_op.values()),
        )
        out = {}
        for op, stats in stats_by_op.items():
            exploit = math.exp(
                self.args.tf_bandit_temperature * (float(stats["ema"]) - 0.50)
            )
            explore = self.args.tf_ucb_weight * math.sqrt(
                math.log(total_pulls + 1.0) / (float(stats["pulls"]) + 1.0)
            )
            out[op] = max(self.args.tf_min_operator_weight, exploit + explore)
        return out

    @staticmethod
    def tf_weighted_choice(weighted_items):
        total = sum(max(0.0, weight) for _, weight in weighted_items)
        if total <= 0:
            return random.choice([item for item, _ in weighted_items])
        draw = random.random() * total
        cumulative = 0.0
        for item, weight in weighted_items:
            cumulative += max(0.0, weight)
            if draw <= cumulative:
                return item
        return weighted_items[-1][0]

    def tf_choose_operator(self, context):
        base = self.tf_base_operator_weights(context)
        multipliers = self.tf_operator_multipliers(context)
        weighted = [
            (op, weight * multipliers.get(op, 1.0))
            for op, weight in base.items()
            if weight > 0
        ]
        return self.tf_weighted_choice(weighted)

    def tf_operator_params(self, operator):
        if operator == "local_micro":
            return self.args.tf_micro_remask, self.args.tf_micro_temperature, self.args.tf_micro_span_prob
        if operator == "local_small":
            return self.args.tf_small_remask, self.args.tf_small_temperature, self.args.tf_small_span_prob
        if operator == "local_medium":
            return self.args.tf_medium_remask, self.args.tf_medium_temperature, self.args.tf_medium_span_prob
        if operator.startswith("graph_"):
            return self.args.tf_graph_remask, self.args.tf_graph_temperature, self.args.tf_graph_span_prob
        return self.args.tf_restart_remask, self.args.tf_restart_temperature, self.args.tf_restart_span_prob

    def tf_graph_fragment_edit(self, parent_smiles, direction):
        parent = canonical_smiles(parent_smiles)
        if parent is None:
            return None
        parent_atoms = atom_count(parent)
        parent_fragments = set()
        try:
            for _ in range(max(1, self.args.tf_graph_cut_rounds)):
                parent_fragments.update(local_genmol_cut(parent))
        except Exception:
            return None
        valid_parent_fragments = []
        for frag in parent_fragments:
            mol = Chem.MolFromSmiles(frag)
            size = fragment_heavy_atom_count(frag)
            if mol is None or size <= 0:
                continue
            if any(atom.GetAtomicNum() == 0 for atom in mol.GetAtoms()):
                valid_parent_fragments.append(frag)
        if not valid_parent_fragments:
            return None
        valid_parent_fragments.sort(key=fragment_heavy_atom_count, reverse=True)
        core_pool = valid_parent_fragments[: max(1, (len(valid_parent_fragments) + 1) // 2)]

        replacements = []
        for score, frag in self.population:
            mol = Chem.MolFromSmiles(frag)
            size = fragment_heavy_atom_count(frag)
            if mol is None or size <= 0:
                continue
            if any(atom.GetAtomicNum() == 0 for atom in mol.GetAtoms()):
                replacements.append((float(score), size, frag))
        if direction == "shrink":
            replacements.sort(key=lambda row: (row[1], -row[0]))
        elif direction == "expand":
            replacements.sort(key=lambda row: (-row[1], -row[0]))
        else:
            replacements.sort(key=lambda row: row[0], reverse=True)
        pool_size = max(10, min(len(replacements), self.args.tf_replacement_pool_size))
        replacement_pool = [frag for _, _, frag in replacements[:pool_size]]
        if not replacement_pool:
            return None

        start_atoms = atom_count(self.start_smiles)
        min_size = max(
            self.args.min_atoms,
            int(math.floor(start_atoms * self.args.tf_min_size_ratio)),
        )
        max_size = min(
            self.args.max_atoms,
            int(math.ceil(start_atoms * self.args.tf_max_size_ratio)),
        )
        for _ in range(self.args.tf_graph_edit_attempts):
            core = random.choice(core_pool)
            replacement = random.choice(replacement_pool)
            edited = attach_fragments(core, replacement)
            can = canonical_smiles(edited) if edited else None
            if can is None or can == parent:
                continue
            atoms = atom_count(can)
            if atoms < min_size or atoms > max_size:
                continue
            if direction == "shrink" and atoms >= parent_atoms:
                continue
            if direction == "expand" and atoms <= parent_atoms:
                continue
            if not tokenizable(can, self.tk, self.args.max_len):
                continue
            return can
        return None

    def tf_make_proposal(self, operator, context):
        parent_item = None
        if operator == "fragment_restart":
            seed = self.make_fragment_seed()
        else:
            parent_item = self.tf_select_parent(context)
            if operator.startswith("graph_"):
                direction = operator.removeprefix("graph_")
                seed = self.tf_graph_fragment_edit(parent_item["smiles"], direction)
            else:
                seed = parent_item["smiles"]
        can = canonical_smiles(seed) if seed else None
        if can is None or not tokenizable(can, self.tk, self.args.max_len):
            return None
        return {
            "seed": can,
            "parent_smiles": None if parent_item is None else parent_item["smiles"],
            "parent_dock": None if parent_item is None else parent_item["dock"],
            "operator": operator,
            "context": context,
        }

    def tf_generate_unique_batch(self, target_n, context):
        out = []
        lineage = []
        batch_seen = set()
        max_model_samples = max(
            target_n,
            int(math.ceil(target_n * self.args.tf_operator_overgenerate_factor)),
        )
        model_samples = 0
        rounds = 0
        while (
            len(out) < target_n
            and model_samples < max_model_samples
            and rounds < self.args.tf_max_generation_rounds
        ):
            rounds += 1
            request_n = min(
                self.args.tf_proposal_batch_size,
                max_model_samples - model_samples,
            )
            proposals = []
            attempts = 0
            while len(proposals) < request_n and attempts < request_n * 100:
                attempts += 1
                operator = self.tf_choose_operator(context)
                proposal = self.tf_make_proposal(operator, context)
                if proposal is not None:
                    proposals.append(proposal)
            if not proposals:
                continue

            grouped = {}
            for proposal in proposals:
                grouped.setdefault(proposal["operator"], []).append(proposal)
            group_items = list(grouped.items())
            random.shuffle(group_items)
            for operator, operator_proposals in group_items:
                remask, temperature, span_prob = self.tf_operator_params(operator)
                candidates = sample_csdnet_local_remask(
                    model=self.model,
                    tk=self.tk,
                    seed_smiles=[proposal["seed"] for proposal in operator_proposals],
                    max_len=self.args.max_len,
                    device=self.device,
                    batch_size=self.args.batch_size,
                    n_steps=self.args.n_steps,
                    remask_fraction=remask,
                    min_remask_tokens=self.args.min_remask_tokens,
                    span_prob=span_prob,
                    use_fsm_check=not self.args.disable_fsm_check,
                    use_rdkit_kekulize_check=not self.args.disable_rdkit_kekulize_check,
                    rdkit_check_interval=self.args.rdkit_check_interval,
                    max_sample_retries=self.args.max_sample_retries,
                    violation_neighborhood=self.args.violation_neighborhood,
                    temperature_start=temperature,
                    temperature_end=self.args.temperature_end,
                    temperature_power=self.args.temperature_power,
                    length_delta_choices="0",
                    length_edit_prob=0.0,
                    length_edit_min_span=1,
                    length_edit_max_span=1,
                    return_seed_indices=True,
                )
                model_samples += len(operator_proposals)
                for smiles, proposal_idx in candidates:
                    if not (0 <= proposal_idx < len(operator_proposals)):
                        continue
                    can = canonical_smiles(smiles)
                    if can is None:
                        continue
                    if can in batch_seen or can in self.tf_seen_smiles:
                        continue
                    atoms = atom_count(can)
                    if atoms < self.args.min_atoms or atoms > self.args.max_atoms:
                        continue
                    if not tokenizable(can, self.tk, self.args.max_len):
                        continue
                    batch_seen.add(can)
                    self.tf_seen_smiles.add(can)
                    out.append(can)
                    lineage.append(operator_proposals[proposal_idx])
                    if len(out) >= target_n:
                        break
                if len(out) >= target_n:
                    break
        return out, lineage

    def tf_transition_reward(self, parent_item, child_item):
        if parent_item is None:
            feasibility = 1.0 - min(
                1.0,
                0.65 * child_item["max_violation"]
                + 0.35 * child_item["sum_violation"] / 4.0,
            )
            reward = 1.0 if child_item["feasible"] else 0.15 + 0.70 * feasibility
            return float(clamp(reward, 0.0, 1.0)), {
                "parent_feasible": "",
                "child_feasible": int(child_item["feasible"]),
                "max_violation_gain": "",
                "sum_violation_gain": "",
                "dock_gain": "",
            }

        if child_item["feasible"] and not parent_item["feasible"]:
            reward = 1.0
        elif parent_item["feasible"] and not child_item["feasible"]:
            reward = 0.0
        elif child_item["feasible"] and parent_item["feasible"]:
            dock_gain = child_item["dock"] - parent_item["dock"]
            reward = 0.5 + 0.5 * math.tanh(
                dock_gain / max(self.args.tf_docking_gain_scale, 1e-8)
            )
        else:
            max_gain = parent_item["max_violation"] - child_item["max_violation"]
            sum_gain = (
                parent_item["sum_violation"] - child_item["sum_violation"]
            ) / 4.0
            progress = 0.70 * max_gain + 0.30 * sum_gain
            reward = 0.5 + 0.5 * math.tanh(
                progress / max(self.args.tf_violation_gain_scale, 1e-8)
            )
            if child_item["constraints_satisfied"] > parent_item["constraints_satisfied"]:
                reward += self.args.tf_constraint_crossing_bonus
        return float(clamp(reward, 0.0, 1.0)), {
            "parent_feasible": int(parent_item["feasible"]),
            "child_feasible": int(child_item["feasible"]),
            "max_violation_gain": float(
                parent_item["max_violation"] - child_item["max_violation"]
            ),
            "sum_violation_gain": float(
                parent_item["sum_violation"] - child_item["sum_violation"]
            ),
            "dock_gain": float(child_item["dock"] - parent_item["dock"]),
        }

    def tf_update_arm_stats(self, context, rewards_by_operator):
        stats_by_op = self.tf_context_stats.setdefault(
            context,
            {
                op: {"ema": 0.50, "pulls": 0.0, "positive": 0.0}
                for op in self.tf_operators
            },
        )
        alpha = clamp(self.args.tf_bandit_alpha, 0.0, 1.0)
        for operator, rewards in rewards_by_operator.items():
            stats = stats_by_op[operator]
            for reward in rewards:
                stats["ema"] = (1.0 - alpha) * stats["ema"] + alpha * float(reward)
                stats["pulls"] += 1.0
                if reward >= self.args.tf_positive_reward_threshold:
                    stats["positive"] += 1.0

    def tf_update_fragment_credit(self, parent_smiles, child_smiles, reward):
        try:
            child_fragments = local_genmol_cut(child_smiles)
        except Exception:
            return
        parent_mol = Chem.MolFromSmiles(parent_smiles) if parent_smiles else None
        alpha = clamp(self.args.tf_credit_alpha, 0.0, 1.0)
        updated = False
        for frag in child_fragments:
            clean = clean_dummy_fragment(frag)
            clean_mol = Chem.MolFromSmiles(clean) if clean else None
            if clean_mol is None:
                continue
            if parent_mol is not None and parent_mol.HasSubstructMatch(clean_mol):
                continue
            meta = self.tf_fragment_meta.setdefault(
                frag,
                {"prior": 0.50, "credit": 0.50, "updates": 0.0},
            )
            meta["credit"] = (1.0 - alpha) * meta["credit"] + alpha * float(reward)
            meta["updates"] += 1.0
            updated = True
        if not updated:
            return
        ranked = []
        for frag, meta in self.tf_fragment_meta.items():
            score = (
                self.args.tf_fragment_prior_weight * meta["prior"]
                + (1.0 - self.args.tf_fragment_prior_weight) * meta["credit"]
            )
            ranked.append((float(score), frag))
        ranked.sort(key=lambda item: item[0], reverse=True)
        cap = self.args.population_cap if self.args.population_cap > 0 else len(ranked)
        self.population[:] = ranked[:cap]

    @staticmethod
    def tf_append_transition(path, context, proposal, child_item, reward, reward_parts):
        fields = [
            "context",
            "operator",
            "parent_smiles",
            "seed_smiles",
            "child_smiles",
            "parent_feasible",
            "child_feasible",
            "parent_dock",
            "child_dock",
            "child_qed",
            "child_sa",
            "child_sim",
            "child_max_violation",
            "child_sum_violation",
            "child_bottleneck",
            "max_violation_gain",
            "sum_violation_gain",
            "dock_gain",
            "reward",
        ]
        exists = os.path.exists(path)
        parent_smiles = proposal.get("parent_smiles")
        with open(path, "a", newline="") as f:
            writer = csv.DictWriter(f, fieldnames=fields)
            if not exists:
                writer.writeheader()
            writer.writerow(
                {
                    "context": context,
                    "operator": proposal["operator"],
                    "parent_smiles": parent_smiles or "",
                    "seed_smiles": proposal["seed"],
                    "child_smiles": child_item["smiles"],
                    "parent_feasible": reward_parts["parent_feasible"],
                    "child_feasible": reward_parts["child_feasible"],
                    "parent_dock": (
                        "" if proposal.get("parent_dock") is None else proposal["parent_dock"]
                    ),
                    "child_dock": child_item["dock"],
                    "child_qed": child_item["qed"],
                    "child_sa": child_item["sa"],
                    "child_sim": child_item["sim"],
                    "child_max_violation": child_item["max_violation"],
                    "child_sum_violation": child_item["sum_violation"],
                    "child_bottleneck": child_item["bottleneck"],
                    "max_violation_gain": reward_parts["max_violation_gain"],
                    "sum_violation_gain": reward_parts["sum_violation_gain"],
                    "dock_gain": reward_parts["dock_gain"],
                    "reward": reward,
                }
            )

    def tf_append_diagnostic(self, evaluated_count, context):
        fields = [
            "evaluated",
            "context",
            "feasible_count",
            "near_count",
            "best_dock",
            "best_max_violation",
            "best_sum_violation",
            "no_improve_batches",
            "operator_stats",
        ]
        exists = os.path.exists(self.tf_diagnostic_path)
        best_feasible = self.tf_feasible_archive[0] if self.tf_feasible_archive else None
        best_near = self.tf_near_archive[0] if self.tf_near_archive else None
        stats = self.tf_context_stats.get(context, {})
        operator_stats = ";".join(
            f"{op}:ema={values['ema']:.3f},n={int(values['pulls'])},"
            f"pos={(values['positive'] / values['pulls'] if values['pulls'] else 0.0):.3f}"
            for op, values in sorted(stats.items())
        )
        with open(self.tf_diagnostic_path, "a", newline="") as f:
            writer = csv.DictWriter(f, fieldnames=fields)
            if not exists:
                writer.writeheader()
            writer.writerow(
                {
                    "evaluated": evaluated_count,
                    "context": context,
                    "feasible_count": len(self.tf_feasible_archive),
                    "near_count": len(self.tf_near_archive),
                    "best_dock": "" if best_feasible is None else best_feasible["dock"],
                    "best_max_violation": "" if best_near is None else best_near["max_violation"],
                    "best_sum_violation": "" if best_near is None else best_near["sum_violation"],
                    "no_improve_batches": self.tf_no_improve_batches,
                    "operator_stats": operator_stats,
                }
            )

    def _load_transition_feasible_resume(self):
        if not os.path.exists(self.fname):
            return
        try:
            df = pd.read_csv(
                self.fname,
                names=["smiles", "DS", "QED", "SA", "SIM", "unused"],
            )
        except Exception:
            return
        for row in df.itertuples(index=False):
            can = canonical_smiles(row.smiles)
            if can is None:
                continue
            item = self.tf_make_item(
                smiles=can,
                dock=row.DS,
                qed=row.QED,
                sa=row.SA,
                sim=row.SIM,
                operator="resume",
                parent_smiles=None,
            )
            if item is None:
                continue
            self.tf_items[can] = item
            self.tf_seen_smiles.add(can)
            self.tf_evaluated_smiles.add(can)
        self.tf_refresh_archives()
        if self.tf_items:
            self.tf_best_key = min(self.tf_rank_key(item) for item in self.tf_items.values())

    def run_transition_feasible(self):
        t_start = time()
        self.tf_refresh_archives()
        if self.tf_best_key is None:
            self.tf_best_key = min(self.tf_rank_key(item) for item in self.tf_items.values())
        evaluated_count = len(self.tf_evaluated_smiles)
        empty_batches = 0
        while evaluated_count < self.args.evaluation_budget:
            context = self.tf_context(evaluated_count)
            self.tf_last_context = context
            target_n = min(
                self.args.tf_feedback_batch_size,
                self.args.evaluation_budget - evaluated_count,
            )
            smiles_list, lineage = self.tf_generate_unique_batch(target_n, context)
            if not smiles_list:
                empty_batches += 1
                self.tf_no_improve_batches += 1
                print(
                    f"[TF] no unique candidates context={context} "
                    f"empty_batches={empty_batches}/{self.args.tf_empty_batch_patience}"
                )
                if empty_batches >= self.args.tf_empty_batch_patience:
                    break
                continue
            empty_batches = 0

            prop_list = self.reward(smiles_list)
            rewards_by_operator = {op: [] for op in self.tf_operators}
            batch_items = []
            for smiles, proposal, dock, qed, sa, sim in zip(
                smiles_list,
                lineage,
                *prop_list,
            ):
                child_item = self.tf_make_item(
                    smiles=smiles,
                    dock=dock,
                    qed=qed,
                    sa=sa,
                    sim=sim,
                    operator=proposal["operator"],
                    parent_smiles=proposal.get("parent_smiles"),
                )
                if child_item is None:
                    continue
                parent_item = self.tf_items.get(proposal.get("parent_smiles"))
                reward, reward_parts = self.tf_transition_reward(parent_item, child_item)
                rewards_by_operator[proposal["operator"]].append(reward)
                self.tf_items[child_item["smiles"]] = child_item
                self.tf_evaluated_smiles.add(child_item["smiles"])
                self.tf_update_fragment_credit(
                    parent_smiles=proposal.get("parent_smiles"),
                    child_smiles=child_item["smiles"],
                    reward=reward,
                )
                self.tf_append_transition(
                    self.tf_transition_path,
                    context=context,
                    proposal=proposal,
                    child_item=child_item,
                    reward=reward,
                    reward_parts=reward_parts,
                )
                batch_items.append(child_item)

            self.record(smiles_list, prop_list)
            evaluated_count = len(self.tf_evaluated_smiles)
            self.tf_refresh_archives()
            current_best_key = min(
                self.tf_rank_key(item) for item in self.tf_items.values()
            )
            if current_best_key < self.tf_best_key:
                self.tf_best_key = current_best_key
                self.tf_no_improve_batches = 0
            else:
                self.tf_no_improve_batches += 1
            self.tf_update_arm_stats(context, rewards_by_operator)
            self.tf_append_diagnostic(evaluated_count, context)

            best_feasible = self.tf_feasible_archive[0] if self.tf_feasible_archive else None
            best_near = self.tf_near_archive[0] if self.tf_near_archive else None
            print(
                f"[TF {evaluated_count:04d}/{self.args.evaluation_budget}] "
                f"context={context} batch={len(smiles_list)} "
                f"feasible={len(self.tf_feasible_archive)} "
                f"best_DS={(best_feasible['dock'] if best_feasible else self.start_prop):.3f} "
                f"best_violation={(best_near['max_violation'] if best_near else 0.0):.4f} "
                f"stagnant={self.tf_no_improve_batches}"
            )

        print(
            f"Transition-feasible lead optimization evaluated "
            f"{evaluated_count}/{self.args.evaluation_budget} unique candidates "
            f"in {time() - t_start:.2f} sec"
        )

    @staticmethod
    def _archive_choice(archive):
        if not archive:
            return None
        top_n = max(1, min(len(archive), 40))
        return random.choice(archive[:top_n])["smiles"]

    def make_ladder_seed(self, operator):
        if operator in {"start_small", "restart_medium"}:
            return self.make_start_fragment_seed()
        if operator in {"near_miss_small", "near_miss_medium"}:
            return (
                self._archive_choice(self.near_miss_elite)
                or self._archive_choice(self.quality_elite)
                or self.make_elite_seed()
            )
        if operator in {"quality_small", "quality_medium"}:
            return (
                self._archive_choice(self.quality_elite)
                or self._archive_choice(self.near_miss_elite)
                or self.make_elite_seed()
            )
        if operator in {"dock_quality_small", "dock_quality_medium"}:
            return (
                self._archive_choice(self.dock_elite)
                or self._archive_choice(self.near_miss_elite)
                or self.make_elite_seed()
            )
        if operator == "diverse_medium":
            return (
                self._archive_choice(self.diverse_elite)
                or self.make_fragment_seed()
            )
        return self.make_fragment_seed()

    def constraint_ladder_mix(self):
        mode = self.last_adaptive_mode
        has_near = bool(self.near_miss_elite)
        has_quality = bool(self.quality_elite)
        has_dock = bool(self.dock_elite)
        has_diverse = bool(self.diverse_elite)
        if mode == "similarity_repair":
            return [
                ("start_small", 0.28),
                ("near_miss_small", 0.26 if has_near else 0.0),
                ("quality_small", 0.24 if has_quality else 0.0),
                ("fragment_medium", 0.14),
                ("restart_medium", 0.08),
            ]
        if mode == "quality_repair":
            return [
                ("dock_quality_small", 0.36 if has_dock else 0.0),
                ("near_miss_small", 0.26 if has_near else 0.0),
                ("quality_small", 0.14 if has_quality else 0.0),
                ("start_small", 0.10),
                ("fragment_medium", 0.14),
            ]
        if mode == "docking_repair":
            return [
                ("quality_medium", 0.38 if has_quality else 0.0),
                ("near_miss_medium", 0.28 if has_near else 0.0),
                ("dock_quality_medium", 0.14 if has_dock else 0.0),
                ("fragment_medium", 0.14),
                ("diverse_medium", 0.06 if has_diverse else 0.0),
            ]
        if mode == "diversity_repair":
            return [
                ("fragment_medium", 0.34),
                ("diverse_medium", 0.24 if has_diverse else 0.0),
                ("restart_medium", 0.18),
                ("near_miss_medium", 0.14 if has_near else 0.0),
                ("quality_medium", 0.10 if has_quality else 0.0),
            ]
        if mode == "feasible_refine":
            return [
                ("quality_small", 0.36 if has_quality else 0.0),
                ("near_miss_small", 0.28 if has_near else 0.0),
                ("start_small", 0.14),
                ("dock_quality_small", 0.12 if has_dock else 0.0),
                ("fragment_medium", 0.10),
            ]
        return [
            ("near_miss_small", 0.24 if has_near else 0.0),
            ("quality_small", 0.20 if has_quality else 0.0),
            ("dock_quality_small", 0.18 if has_dock else 0.0),
            ("fragment_medium", 0.24),
            ("start_small", 0.14),
        ]

    def choose_ladder_operator(self):
        mix = [(name, weight) for name, weight in self.constraint_ladder_mix() if weight > 0]
        if not mix:
            mix = [("fragment_medium", 1.0)]
        total = sum(weight for _, weight in mix)
        draw = random.random() * total
        cumulative = 0.0
        for name, weight in mix:
            cumulative += weight
            if draw <= cumulative:
                return name
        return mix[-1][0]

    def ladder_operator_params(self, operator):
        if operator in {"start_small", "near_miss_small"}:
            return (
                self.args.constraint_small_remask,
                self.args.constraint_low_temperature_start,
                self.args.constraint_small_span_prob,
            )
        if operator in {"quality_small", "dock_quality_small"}:
            return (
                self.args.constraint_quality_remask,
                self.args.constraint_quality_temperature_start,
                self.args.constraint_small_span_prob,
            )
        if operator in {"quality_medium", "near_miss_medium", "dock_quality_medium"}:
            return (
                self.args.constraint_medium_remask,
                self.args.constraint_medium_temperature_start,
                self.args.constraint_medium_span_prob,
            )
        if operator in {"restart_medium", "diverse_medium"}:
            return (
                self.args.constraint_large_remask,
                self.args.constraint_high_temperature_start,
                self.args.constraint_large_span_prob,
            )
        return (
            self.args.constraint_medium_remask,
            self.args.temperature_start,
            self.args.span_prob,
        )

    def v4_operator_mix(self):
        if self.rescue_mode:
            return [
                ("fragment_large", 0.40),
                ("fragment_medium", 0.35),
                ("restart_medium", 0.15),
                ("similarity_small", 0.10),
            ]
        if self.args.sim_thr >= 0.55:
            return [
                ("similarity_small", 0.45),
                ("fragment_medium", 0.25),
                ("fragment_old", 0.20),
                ("restart_small", 0.10),
            ]
        return [
            ("fragment_old", 0.60),
            ("fragment_medium", 0.25),
            ("similarity_small", 0.15),
        ]

    def choose_v4_operator(self):
        mix = self.v4_operator_mix()
        total = sum(weight for _, weight in mix)
        draw = random.random() * total
        cumulative = 0.0
        for name, weight in mix:
            cumulative += weight
            if draw <= cumulative:
                return name
        return mix[-1][0]

    def make_v4_seed(self, operator):
        if operator == "similarity_small":
            return self.make_elite_seed()
        if operator in {"restart_small", "restart_medium"}:
            return self.make_start_fragment_seed()
        return self.make_fragment_seed()

    def v4_operator_params(self, operator):
        if operator in {"similarity_small", "restart_small"}:
            return (
                self.args.v4_small_remask,
                self.args.v4_similarity_temperature_start,
                self.args.v4_similarity_span_prob,
            )
        if operator in {"fragment_medium", "restart_medium"}:
            return (
                self.args.v4_medium_remask,
                self.args.temperature_start,
                self.args.span_prob,
            )
        if operator == "fragment_large":
            return (
                self.args.v4_large_remask,
                self.args.v4_rescue_temperature_start,
                self.args.v4_rescue_span_prob,
            )
        return (
            self.args.remask_fraction,
            self.args.temperature_start,
            self.args.span_prob,
        )

    def generate_batch(self):
        self.current_candidate_ops = []
        if self.args.sampler_profile == "residual_length":
            return self.generate_batch_residual_length()
        if self.args.sampler_profile == "constraint_ladder":
            return self.generate_batch_constraint_ladder()
        if self.args.sampler_profile == "hybrid_v4":
            return self.generate_batch_v4()

        seen = set()
        out = []
        rounds = 0
        while len(out) < self.args.num_gen and rounds < self.args.max_generation_rounds:
            rounds += 1
            remaining = self.args.num_gen - len(out)
            seeds = []
            attempts = 0
            while len(seeds) < remaining and attempts < remaining * 100:
                attempts += 1
                seed = self.make_seed()
                if seed is not None:
                    seeds.append(seed)
            if not seeds:
                break

            candidates = sample_csdnet_local_remask(
                model=self.model,
                tk=self.tk,
                seed_smiles=seeds,
                max_len=self.args.max_len,
                device=self.device,
                batch_size=self.args.batch_size,
                n_steps=self.args.n_steps,
                remask_fraction=self.args.remask_fraction,
                min_remask_tokens=self.args.min_remask_tokens,
                span_prob=self.args.span_prob,
                use_fsm_check=not self.args.disable_fsm_check,
                use_rdkit_kekulize_check=not self.args.disable_rdkit_kekulize_check,
                rdkit_check_interval=self.args.rdkit_check_interval,
                max_sample_retries=self.args.max_sample_retries,
                violation_neighborhood=self.args.violation_neighborhood,
                temperature_start=self.args.temperature_start,
                temperature_end=self.args.temperature_end,
                temperature_power=self.args.temperature_power,
                **self.length_edit_kwargs(),
            )
            for smi in candidates:
                can = canonical_smiles(smi)
                if can is None or can in seen:
                    continue
                if not tokenizable(can, self.tk, self.args.max_len):
                    continue
                seen.add(can)
                out.append(can)
                self.current_candidate_ops.append("fixed")
                if len(out) >= self.args.num_gen:
                    break
        return out

    def generate_batch_residual_length(self):
        seen = set()
        out = []
        ops_out = []
        rounds = 0
        max_model_samples = max(
            self.args.num_gen,
            int(self.args.num_gen * self.args.residual_overgenerate_factor),
        )
        model_samples = 0

        while (
            len(out) < self.args.num_gen
            and rounds < self.args.max_generation_rounds
            and model_samples < max_model_samples
        ):
            rounds += 1
            operator = self.choose_residual_length_operator()
            remask_fraction, temp_start, span_prob, length_kwargs = self.residual_length_operator_params(operator)
            request_n = min(
                self.args.residual_operator_batch_size,
                self.args.num_gen - len(out),
                max_model_samples - model_samples,
            )
            seeds = []
            attempts = 0
            while len(seeds) < request_n and attempts < request_n * 140:
                attempts += 1
                seed = self.make_residual_length_seed(operator)
                if seed is not None:
                    seeds.append(seed)
            if not seeds:
                self.residual_length_counts[operator] += 1
                continue

            candidates = sample_csdnet_local_remask(
                model=self.model,
                tk=self.tk,
                seed_smiles=seeds,
                max_len=self.args.max_len,
                device=self.device,
                batch_size=self.args.batch_size,
                n_steps=self.args.n_steps,
                remask_fraction=remask_fraction,
                min_remask_tokens=self.args.min_remask_tokens,
                span_prob=span_prob,
                use_fsm_check=not self.args.disable_fsm_check,
                use_rdkit_kekulize_check=not self.args.disable_rdkit_kekulize_check,
                rdkit_check_interval=self.args.rdkit_check_interval,
                max_sample_retries=self.args.max_sample_retries,
                violation_neighborhood=self.args.violation_neighborhood,
                temperature_start=temp_start,
                temperature_end=self.args.temperature_end,
                temperature_power=self.args.temperature_power,
                **length_kwargs,
            )
            model_samples += len(seeds)
            for smi in candidates:
                can = canonical_smiles(smi)
                if can is None or can in seen:
                    continue
                atoms = atom_count(can)
                if atoms < self.args.min_atoms or atoms > self.args.max_atoms:
                    continue
                if not tokenizable(can, self.tk, self.args.max_len):
                    continue
                seen.add(can)
                out.append(can)
                ops_out.append(operator)
                if len(out) >= self.args.num_gen:
                    break
        self.current_candidate_ops = ops_out
        return out

    def generate_batch_constraint_ladder(self):
        seen = set()
        out = []
        rounds = 0
        max_model_samples = max(
            self.args.num_gen,
            int(self.args.num_gen * self.args.constraint_overgenerate_factor),
        )
        model_samples = 0

        while (
            len(out) < self.args.num_gen
            and rounds < self.args.max_generation_rounds
            and model_samples < max_model_samples
        ):
            rounds += 1
            operator = self.choose_ladder_operator()
            remask_fraction, temp_start, span_prob = self.ladder_operator_params(operator)
            request_n = min(
                self.args.constraint_operator_batch_size,
                self.args.num_gen - len(out),
                max_model_samples - model_samples,
            )
            seeds = []
            attempts = 0
            while len(seeds) < request_n and attempts < request_n * 120:
                attempts += 1
                seed = self.make_ladder_seed(operator)
                if seed is not None:
                    seeds.append(seed)
            if not seeds:
                continue

            candidates = sample_csdnet_local_remask(
                model=self.model,
                tk=self.tk,
                seed_smiles=seeds,
                max_len=self.args.max_len,
                device=self.device,
                batch_size=self.args.batch_size,
                n_steps=self.args.n_steps,
                remask_fraction=remask_fraction,
                min_remask_tokens=self.args.min_remask_tokens,
                span_prob=span_prob,
                use_fsm_check=not self.args.disable_fsm_check,
                use_rdkit_kekulize_check=not self.args.disable_rdkit_kekulize_check,
                rdkit_check_interval=self.args.rdkit_check_interval,
                max_sample_retries=self.args.max_sample_retries,
                violation_neighborhood=self.args.violation_neighborhood,
                temperature_start=temp_start,
                temperature_end=self.args.temperature_end,
                temperature_power=self.args.temperature_power,
                **self.length_edit_kwargs(),
            )
            model_samples += len(seeds)
            for smi in candidates:
                can = canonical_smiles(smi)
                if can is None or can in seen:
                    continue
                atoms = atom_count(can)
                if atoms < self.args.min_atoms or atoms > self.args.max_atoms:
                    continue
                if not tokenizable(can, self.tk, self.args.max_len):
                    continue
                seen.add(can)
                out.append(can)
                self.current_candidate_ops.append(operator)
                if len(out) >= self.args.num_gen:
                    break
        return out

    def generate_batch_v4(self):
        seen = set()
        out = []
        rounds = 0
        max_model_samples = max(
            self.args.num_gen,
            int(self.args.num_gen * self.args.v4_overgenerate_factor),
        )
        model_samples = 0

        while (
            len(out) < self.args.num_gen
            and rounds < self.args.max_generation_rounds
            and model_samples < max_model_samples
        ):
            rounds += 1
            operator = self.choose_v4_operator()
            remask_fraction, temp_start, span_prob = self.v4_operator_params(operator)
            request_n = min(
                self.args.v4_operator_batch_size,
                self.args.num_gen - len(out),
                max_model_samples - model_samples,
            )
            seeds = []
            attempts = 0
            while len(seeds) < request_n and attempts < request_n * 100:
                attempts += 1
                seed = self.make_v4_seed(operator)
                if seed is not None:
                    seeds.append(seed)
            if not seeds:
                continue

            candidates = sample_csdnet_local_remask(
                model=self.model,
                tk=self.tk,
                seed_smiles=seeds,
                max_len=self.args.max_len,
                device=self.device,
                batch_size=self.args.batch_size,
                n_steps=self.args.n_steps,
                remask_fraction=remask_fraction,
                min_remask_tokens=self.args.min_remask_tokens,
                span_prob=span_prob,
                use_fsm_check=not self.args.disable_fsm_check,
                use_rdkit_kekulize_check=not self.args.disable_rdkit_kekulize_check,
                rdkit_check_interval=self.args.rdkit_check_interval,
                max_sample_retries=self.args.max_sample_retries,
                violation_neighborhood=self.args.violation_neighborhood,
                temperature_start=temp_start,
                temperature_end=self.args.temperature_end,
                temperature_power=self.args.temperature_power,
                **self.length_edit_kwargs(),
            )
            model_samples += len(seeds)
            for smi in candidates:
                can = canonical_smiles(smi)
                if can is None or can in seen:
                    continue
                atoms = atom_count(can)
                if atoms < self.args.min_atoms or atoms > self.args.max_atoms:
                    continue
                if not tokenizable(can, self.tk, self.args.max_len):
                    continue
                seen.add(can)
                out.append(can)
                self.current_candidate_ops.append(operator)
                if len(out) >= self.args.num_gen:
                    break
        return out

    def _fragment_scores_from_selected(self, selected):
        out = []
        for item in selected:
            for frag in local_genmol_cut(item["smiles"]):
                if Chem.MolFromSmiles(frag) is None:
                    continue
                out.append((float(item["pareto_score"]), frag))
        return out

    def _pareto_candidates(self, smiles_list, prop_list):
        rv_list, rq_list, rs_list, rsim_list = prop_list
        items = []
        sim_floor = max(0.0, self.args.sim_thr - self.args.pareto_similarity_slack)
        for rv, rq, rs, rsim, smiles in zip(rv_list, rq_list, rs_list, rsim_list, smiles_list):
            if rv < self.start_prop * self.args.pareto_min_docking_ratio:
                continue
            if rq < self.args.pareto_min_qed or rs < self.args.pareto_min_sa or rsim < sim_floor:
                continue
            dock_gain = max(0.0, (rv - self.start_prop) / max(self.start_prop, 1e-6))
            dock_norm = min(1.5, dock_gain) / 1.5
            pareto_score = (
                self.args.pareto_docking_weight * dock_norm
                + self.args.pareto_similarity_weight * rsim
                + self.args.pareto_qed_weight * rq
                + self.args.pareto_sa_weight * rs
            )
            items.append(
                {
                    "smiles": smiles,
                    "dock": float(rv),
                    "dock_norm": float(dock_norm),
                    "sim": float(rsim),
                    "qed": float(rq),
                    "sa": float(rs),
                    "pareto_score": float(pareto_score),
                }
            )
        if not items:
            return []
        items = pareto_rank(items, keys=("dock_norm", "sim", "qed", "sa"))
        items.sort(key=lambda item: (item["pareto_rank"], -item["pareto_score"]))
        return items[: self.args.pareto_top_k]

    def _v4_exploration_candidates(self, smiles_list, prop_list):
        rv_list, rq_list, rs_list, rsim_list = prop_list
        sim_floor = max(0.0, self.args.sim_thr - self.args.v4_explore_similarity_slack)
        docking_items = []
        sim_items = []
        for rv, rq, rs, rsim, smiles in zip(rv_list, rq_list, rs_list, rsim_list, smiles_list):
            if rq < self.args.pareto_min_qed or rs < self.args.pareto_min_sa:
                continue
            dock_gain = max(0.0, (rv - self.start_prop) / max(self.start_prop, 1e-6))
            dock_norm = min(1.5, dock_gain) / 1.5
            item = {
                "smiles": smiles,
                "dock": float(rv),
                "dock_norm": float(dock_norm),
                "sim": float(rsim),
                "qed": float(rq),
                "sa": float(rs),
                "pareto_score": float(0.65 * dock_norm + 0.25 * rsim + 0.05 * rq + 0.05 * rs),
            }
            if rv >= self.start_prop * self.args.v4_explore_min_docking_ratio and rsim >= sim_floor:
                docking_items.append(item)
            if rsim >= self.args.sim_thr and rv >= self.start_prop * self.args.v4_high_sim_min_docking_ratio:
                sim_items.append(item)

        docking_items.sort(key=lambda item: (item["dock_norm"], item["sim"]), reverse=True)
        sim_items.sort(key=lambda item: (item["sim"], item["dock_norm"]), reverse=True)
        selected = docking_items[: self.args.v4_explore_top_k] + sim_items[: self.args.v4_explore_top_k]
        dedup = {}
        for item in selected:
            dedup.setdefault(item["smiles"], item)
        return list(dedup.values())

    def _constraint_items(self, smiles_list, prop_list):
        rv_list, rq_list, rs_list, rsim_list = prop_list
        items = []
        sim_soft_floor = max(0.0, self.args.sim_thr - self.args.constraint_similarity_slack)
        for rv, rq, rs, rsim, smiles in zip(rv_list, rq_list, rs_list, rsim_list, smiles_list):
            can = canonical_smiles(smiles)
            if can is None:
                continue
            sim_norm = min(1.0, float(rsim) / max(self.args.sim_thr, 1e-6))
            qed_norm = min(1.0, float(rq) / 0.6)
            sa_norm = min(1.0, float(rs) / (6 / 9))
            dock_ratio = float(rv) / max(self.start_prop, 1e-6)
            dock_norm = min(1.5, max(0.0, dock_ratio)) / 1.5
            quality_score = 0.55 * qed_norm + 0.45 * sa_norm
            near_score = (
                self.args.constraint_near_docking_weight * dock_norm
                + self.args.constraint_near_similarity_weight * sim_norm
                + self.args.constraint_near_quality_weight * quality_score
            )
            items.append(
                {
                    "smiles": can,
                    "dock": float(rv),
                    "qed": float(rq),
                    "sa": float(rs),
                    "sim": float(rsim),
                    "dock_norm": float(dock_norm),
                    "quality_score": float(quality_score),
                    "near_score": float(near_score),
                    "sim_ok": float(rsim) >= self.args.sim_thr,
                    "soft_sim_ok": float(rsim) >= sim_soft_floor,
                    "quality_ok": float(rq) >= 0.6 and float(rs) >= 6 / 9,
                    "soft_quality_ok": (
                        float(rq) >= self.args.constraint_min_qed
                        and float(rs) >= self.args.constraint_min_sa
                    ),
                    "dock_ok": float(rv) > self.start_prop,
                    "pareto_score": float(
                        0.35 * dock_norm + 0.25 * sim_norm + 0.25 * qed_norm + 0.15 * sa_norm
                    ),
                }
            )
        return items

    @staticmethod
    def _merge_archive(archive, items, score_key, cap):
        dedup = {}
        for item in list(archive) + list(items):
            smiles = item["smiles"]
            if smiles not in dedup or item[score_key] > dedup[smiles][score_key]:
                dedup[smiles] = item
        archive[:] = sorted(
            dedup.values(),
            key=lambda item: (item[score_key], item["dock"], item["sim"]),
            reverse=True,
        )[:cap]

    def _merge_diverse_archive(self, items):
        cap = self.args.constraint_diverse_size
        threshold = self.args.constraint_diverse_max_sim
        pool = {}
        for item in list(self.diverse_elite) + list(items):
            smiles = item["smiles"]
            if smiles not in pool or item["near_score"] > pool[smiles]["near_score"]:
                pool[smiles] = item
        ordered = sorted(
            pool.values(),
            key=lambda item: (item["near_score"], item["quality_score"], item["dock_norm"]),
            reverse=True,
        )
        selected = []
        fps = []
        for item in ordered:
            mol = Chem.MolFromSmiles(item["smiles"])
            if mol is None:
                continue
            fp = AllChem.GetMorganFingerprintAsBitVect(mol, 2, 2048)
            if fps:
                max_sim = max(DataStructs.BulkTanimotoSimilarity(fp, fps))
                if max_sim > threshold and len(selected) >= max(10, cap // 5):
                    continue
            selected.append(item)
            fps.append(fp)
            if len(selected) >= cap:
                break
        self.diverse_elite[:] = selected

    def update_constraint_archives(self, smiles_list, prop_list):
        if self.args.sampler_profile not in {"constraint_ladder", "residual_length"}:
            return
        if self.args.sampler_profile == "residual_length":
            items = self.residual_items(smiles_list, prop_list, self.current_candidate_ops)
        else:
            items = self._constraint_items(smiles_list, prop_list)
        if not items:
            return
        archive_size = self.args.constraint_archive_size
        dock_items = [
            item for item in items
            if item["soft_sim_ok"]
            and item["dock"] >= self.start_prop * self.args.constraint_dock_archive_min_ratio
        ]
        quality_items = [
            item for item in items
            if item["soft_sim_ok"] and item["soft_quality_ok"]
        ]
        near_items = [
            item for item in items
            if (
                item["near_score"] >= self.args.constraint_near_miss_min_score
                or item.get("residual_score", -float("inf")) >= self.args.residual_archive_min_score
            )
        ]
        diverse_items = [
            item for item in items
            if (
                item["near_score"] >= self.args.constraint_near_miss_min_score * 0.85
                or item.get("residual_score", -float("inf")) >= self.args.residual_archive_min_score * 0.92
            )
        ]
        self._merge_archive(self.dock_elite, dock_items, "dock", archive_size)
        self._merge_archive(self.quality_elite, quality_items, "pareto_score", archive_size)
        near_key = "residual_score" if self.args.sampler_profile == "residual_length" else "near_score"
        self._merge_archive(self.near_miss_elite, near_items, near_key, archive_size)
        self._merge_diverse_archive(diverse_items)

    def _constraint_ladder_candidates(self, smiles_list, prop_list):
        if self.args.sampler_profile == "residual_length":
            items = self.residual_items(smiles_list, prop_list, self.current_candidate_ops)
        else:
            items = self._constraint_items(smiles_list, prop_list)
        if not items:
            return []
        selected = []
        if self.args.sampler_profile == "residual_length":
            selected.extend(
                item for item in items
                if item["residual_score"] >= self.args.residual_archive_min_score
            )
            selected.extend(
                item for item in items
                if item["soft_sim_ok"] and item["soft_quality_ok"]
            )
        else:
            selected.extend(
                item for item in items
                if item["soft_sim_ok"] and item["soft_quality_ok"]
            )
            selected.extend(
                item for item in items
                if item["near_score"] >= self.args.constraint_near_miss_min_score
            )
        dedup = {}
        for item in selected:
            old = dedup.get(item["smiles"])
            if old is None or item["pareto_score"] > old["pareto_score"]:
                dedup[item["smiles"]] = item
        out = sorted(
            dedup.values(),
            key=lambda item: (item.get("residual_score", item["near_score"]), item["pareto_score"]),
            reverse=True,
        )
        return out[: self.args.pareto_top_k]

    def update_population(self, smiles_list, prop_list):
        rv_list, rq_list, rs_list, rsim_list = prop_list
        known = {frag for _, frag in self.population}
        self.update_elite_smiles(smiles_list, prop_list)
        self.update_constraint_archives(smiles_list, prop_list)
        if self.args.pareto_population_update:
            selected = self._pareto_candidates(smiles_list, prop_list)
            if self.args.sampler_profile == "hybrid_v4":
                selected += self._v4_exploration_candidates(smiles_list, prop_list)
            if self.args.sampler_profile == "constraint_ladder":
                selected += self._constraint_ladder_candidates(smiles_list, prop_list)
            if self.args.sampler_profile == "residual_length":
                selected += self._constraint_ladder_candidates(smiles_list, prop_list)
            updates = self._fragment_scores_from_selected(selected)
        else:
            updates = []
            for rv, rq, rs, rsim, smiles in zip(rv_list, rq_list, rs_list, rsim_list, smiles_list):
                if rv <= self.start_prop or rq < 0.6 or rs < 6 / 9 or rsim < self.args.sim_thr:
                    continue
                updates.extend((float(rv), frag) for frag in local_genmol_cut(smiles))

        for score, frag in updates:
            if frag in known or Chem.MolFromSmiles(frag) is None:
                continue
            known.add(frag)
            self.population.append((float(score), frag))

        self.population.sort(key=lambda item: item[0], reverse=True)
        if self.args.population_cap > 0:
            del self.population[self.args.population_cap:]

    def update_elite_smiles(self, smiles_list, prop_list):
        rv_list, rq_list, rs_list, rsim_list = prop_list
        existing = {smi for _, _, smi in self.elite_smiles}
        sim_floor = max(0.0, self.args.sim_thr - self.args.lead_elite_similarity_slack)
        for rv, rq, rs, rsim, smiles in zip(rv_list, rq_list, rs_list, rsim_list, smiles_list):
            can = canonical_smiles(smiles)
            if can is None or can in existing:
                continue
            if rsim < sim_floor or rq < self.args.pareto_min_qed or rs < self.args.pareto_min_sa:
                continue
            dock_gain = max(0.0, (rv - self.start_prop) / max(self.start_prop, 1e-6))
            score = (
                self.args.lead_elite_docking_weight * min(1.5, dock_gain) / 1.5
                + self.args.lead_elite_similarity_weight * rsim
                + 0.10 * rq
            )
            existing.add(can)
            self.elite_smiles.append((float(score), float(rsim), can))
        self.elite_smiles.sort(key=lambda item: (item[0], item[1]), reverse=True)
        del self.elite_smiles[self.args.lead_elite_size:]

    def record(self, smiles_list, prop_list):
        rv_list, rq_list, rs_list, rsim_list = prop_list
        with open(self.fname, "a") as f:
            for smiles, rv, rq, rs, rsim in zip(smiles_list, rv_list, rq_list, rs_list, rsim_list):
                f.write(f"{smiles},{rv},{rq},{rs},{rsim},\n")

    def adaptive_iter_stats(self, smiles_list, prop_list):
        rv_list, rq_list, rs_list, rsim_list = prop_list
        n = max(1, len(smiles_list))
        new_n = sum(1 for smiles in smiles_list if smiles not in self.generated_seen)
        self.generated_seen.update(smiles_list)
        sim_ok = [rsim >= self.args.sim_thr for rsim in rsim_list]
        qed_ok = [rq >= 0.6 for rq in rq_list]
        sa_ok = [rs >= 6 / 9 for rs in rs_list]
        dock_ok = [rv > self.start_prop for rv in rv_list]
        strict_ok = [
            bool(s and q and a and d)
            for s, q, a, d in zip(sim_ok, qed_ok, sa_ok, dock_ok)
        ]
        sim_quality_ok = [
            bool(s and q and a)
            for s, q, a in zip(sim_ok, qed_ok, sa_ok)
        ]
        soft_quality_ok = [
            bool(q >= self.args.constraint_min_qed and a >= self.args.constraint_min_sa)
            for q, a in zip(rq_list, rs_list)
        ]
        return {
            "sim_rate": sum(sim_ok) / n,
            "quality_rate": sum(q and a for q, a in zip(qed_ok, sa_ok)) / n,
            "soft_quality_rate": sum(soft_quality_ok) / n,
            "sim_quality_rate": sum(sim_quality_ok) / n,
            "dock_rate": sum(dock_ok) / n,
            "strict_rate": sum(strict_ok) / n,
            "new_rate": new_n / n,
            "mean_sim": float(np.mean(rsim_list)) if rsim_list else 0.0,
            "mean_dock": float(np.mean(rv_list)) if rv_list else 0.0,
            "mean_qed": float(np.mean(rq_list)) if rq_list else 0.0,
            "mean_sa": float(np.mean(rs_list)) if rs_list else 0.0,
        }

    def update_residual_length_bandit(self, smiles_list, prop_list):
        if self.args.sampler_profile != "residual_length":
            return
        items = self.residual_items(smiles_list, prop_list, self.current_candidate_ops)
        if not items:
            self.last_length_bandit_summary = "no_items"
            return

        scores = [item["residual_score"] for item in items]
        baseline = float(np.mean(scores))
        prev_best = self.best_residual_score
        self.best_residual_score = max(self.best_residual_score, max(scores))
        updated = []

        for op in self.residual_length_ops:
            op_items = [item for item in items if item["operator"] == op]
            if not op_items:
                continue
            self.residual_length_counts[op] += len(op_items)
            op_scores = sorted((item["residual_score"] for item in op_items), reverse=True)
            op_best = float(op_scores[0])
            op_top_mean = float(np.mean(op_scores[: min(5, len(op_scores))]))
            strict_rate = sum(item["strict_ok"] for item in op_items) / len(op_items)
            sim_quality_rate = sum(item["sim_ok"] and item["quality_ok"] for item in op_items) / len(op_items)
            sim_dock_rate = sum(item["sim_ok"] and item["dock_ok"] for item in op_items) / len(op_items)
            improvement = 0.0 if prev_best == -float("inf") else max(0.0, op_best - prev_best)
            reward = (
                0.58 * (op_best - baseline)
                + 0.28 * (op_top_mean - baseline)
                + self.args.residual_strict_reward * strict_rate
                + self.args.residual_sim_quality_reward * sim_quality_rate
                + self.args.residual_sim_dock_reward * sim_dock_rate
                + self.args.residual_improvement_reward * improvement
            )
            reward = clamp(
                reward,
                -self.args.residual_reward_clip,
                self.args.residual_reward_clip,
            )
            old_ema = self.residual_length_reward_ema[op]
            ema = self.args.residual_reward_decay * old_ema + (1.0 - self.args.residual_reward_decay) * reward
            self.residual_length_reward_ema[op] = ema
            self.residual_length_weights[op] *= float(np.exp(self.args.residual_bandit_eta * ema))
            self.residual_length_weights[op] = clamp(
                self.residual_length_weights[op],
                self.args.residual_min_operator_weight,
                self.args.residual_max_operator_weight,
            )
            updated.append((op, op_best, reward, self.residual_length_weights[op]))

        probs = self.residual_length_operator_probs()
        top_probs = sorted(probs.items(), key=lambda item: item[1], reverse=True)[:3]
        self.last_length_bandit_summary = ",".join(
            f"{op}:{prob:.2f}" for op, prob in top_probs
        )
        self.last_adaptive_mode = "residual_bandit"

    def _blend_arg(self, name, target):
        current = getattr(self.args, name)
        alpha = self.args.adaptive_step_size
        setattr(self.args, name, (1.0 - alpha) * current + alpha * target)

    def _set_adaptive_weights(self, docking_weight, similarity_weight):
        self._blend_arg("pareto_docking_weight", docking_weight)
        self._blend_arg("pareto_similarity_weight", similarity_weight)
        self.args.pareto_qed_weight = self.adaptive_base["pareto_qed_weight"]
        self.args.pareto_sa_weight = self.adaptive_base["pareto_sa_weight"]
        self._blend_arg("lead_elite_docking_weight", docking_weight)
        self._blend_arg("lead_elite_similarity_weight", similarity_weight)

    def update_adaptive_profile(self, iter_idx, stats):
        if self.args.sampler_profile != "adaptive_similarity":
            return

        base = self.adaptive_base
        mode = "balanced"
        remask = base["remask_fraction"]
        temp = base["temperature_start"]
        elite_prob = base["lead_elite_seed_prob"]
        start_prob = base["lead_start_seed_prob"]
        sim_slack = base["pareto_similarity_slack"]
        dock_weight = base["pareto_docking_weight"]
        sim_weight = base["pareto_similarity_weight"]

        if iter_idx <= self.args.adaptive_warmup_iters:
            mode = "warmup"
        elif stats["strict_rate"] > 0.0:
            mode = "feasible_refine"
            remask -= self.args.adaptive_less_remask_delta * 0.5
            temp -= 0.04
            elite_prob += 0.08
            start_prob += 0.04
            dock_weight = 0.45
            sim_weight = 0.35
        elif stats["sim_rate"] < self.args.adaptive_low_sim_rate:
            mode = "similarity_repair"
            remask -= self.args.adaptive_less_remask_delta
            temp -= 0.06
            elite_prob += 0.15
            start_prob += 0.08
            sim_slack += 0.02
            dock_weight = 0.30
            sim_weight = 0.50
        elif (
            stats["sim_rate"] >= self.args.adaptive_high_sim_rate
            and stats["dock_rate"] < self.args.adaptive_low_dock_rate
        ) or self.no_improve_iters >= self.args.adaptive_rescue_after_iters:
            mode = "docking_expand"
            remask += self.args.adaptive_more_remask_delta
            temp += 0.06
            elite_prob -= 0.08
            start_prob -= 0.04
            sim_slack += 0.02
            dock_weight = 0.50
            sim_weight = 0.30
        elif stats["new_rate"] < self.args.adaptive_low_new_rate:
            mode = "diversity_repair"
            remask += self.args.adaptive_more_remask_delta * 0.75
            temp += 0.08
            elite_prob -= 0.12
            start_prob -= 0.04
            dock_weight = 0.45
            sim_weight = 0.35

        remask = clamp(remask, self.args.adaptive_min_remask, self.args.adaptive_max_remask)
        temp = clamp(temp, self.args.adaptive_min_temp, self.args.adaptive_max_temp)
        elite_prob = clamp(elite_prob, 0.20, 0.88)
        start_prob = clamp(start_prob, 0.05, 0.55)
        sim_slack = clamp(sim_slack, 0.04, self.args.adaptive_max_similarity_slack)

        self._blend_arg("remask_fraction", remask)
        self._blend_arg("temperature_start", temp)
        self._blend_arg("lead_elite_seed_prob", elite_prob)
        self._blend_arg("lead_start_seed_prob", start_prob)
        self._blend_arg("lead_elite_similarity_slack", sim_slack)
        self._blend_arg("pareto_similarity_slack", sim_slack)
        self._set_adaptive_weights(dock_weight, sim_weight)
        self.last_adaptive_mode = mode

    def update_constraint_ladder_profile(self, iter_idx, stats):
        if self.args.sampler_profile != "constraint_ladder":
            return
        if iter_idx <= self.args.constraint_warmup_iters:
            mode = "warmup"
        elif stats["strict_rate"] > 0.0:
            mode = "feasible_refine"
        elif stats["sim_rate"] < self.args.constraint_low_sim_rate:
            mode = "similarity_repair"
        elif stats["sim_quality_rate"] < self.args.constraint_low_sim_quality_rate:
            mode = "quality_repair"
        elif (
            self.no_improve_iters >= self.args.constraint_rescue_after_iters
            or (
                stats["sim_quality_rate"] >= self.args.constraint_low_sim_quality_rate
                and stats["dock_rate"] < self.args.constraint_low_dock_rate
            )
        ):
            mode = "docking_repair"
        elif stats["new_rate"] < self.args.constraint_low_new_rate:
            mode = "diversity_repair"
        else:
            mode = "balanced"
        self.last_adaptive_mode = mode

    def run(self):
        if self.args.sampler_profile == "transition_feasible":
            self.run_transition_feasible()
            return
        t_start = time()
        raw_best = self.start_prop
        feasible_best = self.start_prop
        total = 0
        for i in range(self.args.num_iter):
            smiles_list = self.generate_batch()
            if not smiles_list:
                print(f"[Iter {i + 1:03d}] no candidates generated")
                if self.args.sampler_profile == "hybrid_v4":
                    self.no_improve_iters += 1
                    self.rescue_mode = self.no_improve_iters >= self.args.v4_rescue_after_iters
                if self.args.sampler_profile == "constraint_ladder":
                    self.no_improve_iters += 1
                    self.last_adaptive_mode = "diversity_repair"
                if self.args.sampler_profile == "residual_length":
                    self.no_improve_iters += 1
                    self.last_adaptive_mode = "residual_bandit"
                    self.last_length_bandit_summary = "no_candidates"
                continue
            prop_list = self.reward(smiles_list)
            self.update_population(smiles_list, prop_list)
            self.record(smiles_list, prop_list)
            adaptive_stats = self.adaptive_iter_stats(smiles_list, prop_list)
            self.update_residual_length_bandit(smiles_list, prop_list)
            total += len(smiles_list)
            raw_best = max(raw_best, max(prop_list[0], default=raw_best))
            feasible_scores = [
                rv
                for rv, rq, rs, rsim in zip(*prop_list)
                if rv > self.start_prop and rq >= 0.6 and rs >= 6 / 9 and rsim >= self.args.sim_thr
            ]
            previous_feasible_best = feasible_best
            feasible_best = max(feasible_best, max(feasible_scores, default=feasible_best))
            if feasible_best > previous_feasible_best + 1e-6:
                self.no_improve_iters = 0
            else:
                self.no_improve_iters += 1
            self.rescue_mode = (
                self.args.sampler_profile == "hybrid_v4"
                and self.no_improve_iters >= self.args.v4_rescue_after_iters
            )
            if self.args.sampler_profile == "constraint_ladder":
                self.update_constraint_ladder_profile(i + 1, adaptive_stats)
            else:
                self.update_adaptive_profile(i + 1, adaptive_stats)
            print(
                f"[Iter {i + 1:03d}] generated={len(smiles_list)} total={total} "
                f"raw_top_DS={raw_best:.3f} constrained_top_DS={feasible_best:.3f} "
                f"population={len(self.population)} rescue={self.rescue_mode} "
                f"adaptive={self.last_adaptive_mode} "
                f"sim_rate={adaptive_stats['sim_rate']:.3f} "
                f"sim_quality_rate={adaptive_stats['sim_quality_rate']:.3f} "
                f"dock_rate={adaptive_stats['dock_rate']:.3f} "
                f"strict_rate={adaptive_stats['strict_rate']:.3f} "
                f"new_rate={adaptive_stats['new_rate']:.3f} "
                f"archives=d{len(self.dock_elite)}/q{len(self.quality_elite)}/n{len(self.near_miss_elite)} "
                f"remask={self.args.remask_fraction:.3f} "
                f"temp={self.args.temperature_start:.3f} "
                f"w_dock={self.args.pareto_docking_weight:.3f} "
                f"w_sim={self.args.pareto_similarity_weight:.3f} "
                f"length_ops={self.last_length_bandit_summary}"
            )
        print(f"{time() - t_start:.2f} sec elapsed")


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("-o", "--oracle_name", type=str, default="parp1",
                        choices=["parp1", "fa7", "5ht1b", "braf", "jak2"])
    parser.add_argument("-i", "--start_mol_idx", type=int, default=0, choices=[0, 1, 2])
    parser.add_argument("-d", "--sim_thr", type=float, default=0.4)
    parser.add_argument("-s", "--seed", type=int, default=0)

    parser.add_argument("--ckpt_path", type=str, required=True)
    parser.add_argument("--vocab", type=str, default="csdnet_vocab.pkl")
    parser.add_argument("--cpu", action="store_true")
    parser.add_argument("--output_dir", type=str, default=os.path.join("CSDNet", "exp", "lead", "results"))
    parser.add_argument("--resume", action="store_true")

    parser.add_argument("--num_gen", type=int, default=100)
    parser.add_argument("--num_iter", type=int, default=10)
    parser.add_argument("--max_generation_rounds", type=int, default=3)
    parser.add_argument("--min_atoms", type=int, default=10)
    parser.add_argument("--max_atoms", type=int, default=80)
    parser.add_argument("--population_cap", type=int, default=0)
    parser.add_argument(
        "--sampler_profile",
        choices=[
            "fragment",
            "similarity_aware",
            "adaptive_similarity",
            "hybrid_v4",
            "constraint_ladder",
            "residual_length",
            "transition_feasible",
        ],
        default="fragment",
    )
    parser.add_argument("--lead_elite_seed_prob", type=float, default=0.55)
    parser.add_argument("--lead_start_seed_prob", type=float, default=0.25)
    parser.add_argument("--lead_elite_size", type=int, default=80)
    parser.add_argument("--lead_elite_similarity_slack", type=float, default=0.10)
    parser.add_argument("--lead_elite_docking_weight", type=float, default=0.45)
    parser.add_argument("--lead_elite_similarity_weight", type=float, default=0.45)
    parser.add_argument("--pareto_population_update", action="store_true")
    parser.add_argument("--pareto_top_k", type=int, default=30)
    parser.add_argument("--pareto_similarity_slack", type=float, default=0.15)
    parser.add_argument("--pareto_min_docking_ratio", type=float, default=0.85)
    parser.add_argument("--pareto_min_qed", type=float, default=0.45)
    parser.add_argument("--pareto_min_sa", type=float, default=0.45)
    parser.add_argument("--pareto_docking_weight", type=float, default=0.40)
    parser.add_argument("--pareto_similarity_weight", type=float, default=0.35)
    parser.add_argument("--pareto_qed_weight", type=float, default=0.125)
    parser.add_argument("--pareto_sa_weight", type=float, default=0.125)
    parser.add_argument("--v4_small_remask", type=float, default=0.12)
    parser.add_argument("--v4_medium_remask", type=float, default=0.24)
    parser.add_argument("--v4_large_remask", type=float, default=0.48)
    parser.add_argument("--v4_operator_batch_size", type=int, default=64)
    parser.add_argument("--v4_overgenerate_factor", type=float, default=2.5)
    parser.add_argument("--v4_rescue_after_iters", type=int, default=2)
    parser.add_argument("--v4_similarity_temperature_start", type=float, default=0.95)
    parser.add_argument("--v4_rescue_temperature_start", type=float, default=1.45)
    parser.add_argument("--v4_similarity_span_prob", type=float, default=0.80)
    parser.add_argument("--v4_rescue_span_prob", type=float, default=0.90)
    parser.add_argument("--v4_explore_similarity_slack", type=float, default=0.20)
    parser.add_argument("--v4_explore_min_docking_ratio", type=float, default=0.70)
    parser.add_argument("--v4_high_sim_min_docking_ratio", type=float, default=0.55)
    parser.add_argument("--v4_explore_top_k", type=int, default=20)
    parser.add_argument("--adaptive_warmup_iters", type=int, default=2)
    parser.add_argument("--adaptive_rescue_after_iters", type=int, default=2)
    parser.add_argument("--adaptive_low_sim_rate", type=float, default=0.25)
    parser.add_argument("--adaptive_high_sim_rate", type=float, default=0.65)
    parser.add_argument("--adaptive_low_dock_rate", type=float, default=0.08)
    parser.add_argument("--adaptive_low_new_rate", type=float, default=0.65)
    parser.add_argument("--adaptive_min_remask", type=float, default=0.08)
    parser.add_argument("--adaptive_max_remask", type=float, default=0.34)
    parser.add_argument("--adaptive_more_remask_delta", type=float, default=0.04)
    parser.add_argument("--adaptive_less_remask_delta", type=float, default=0.04)
    parser.add_argument("--adaptive_min_temp", type=float, default=0.88)
    parser.add_argument("--adaptive_max_temp", type=float, default=1.25)
    parser.add_argument("--adaptive_max_similarity_slack", type=float, default=0.16)
    parser.add_argument("--adaptive_step_size", type=float, default=0.70)
    parser.add_argument("--constraint_small_remask", type=float, default=0.10)
    parser.add_argument("--constraint_quality_remask", type=float, default=0.14)
    parser.add_argument("--constraint_medium_remask", type=float, default=0.24)
    parser.add_argument("--constraint_large_remask", type=float, default=0.34)
    parser.add_argument("--constraint_operator_batch_size", type=int, default=48)
    parser.add_argument("--constraint_overgenerate_factor", type=float, default=2.0)
    parser.add_argument("--constraint_low_temperature_start", type=float, default=0.92)
    parser.add_argument("--constraint_quality_temperature_start", type=float, default=1.02)
    parser.add_argument("--constraint_medium_temperature_start", type=float, default=1.12)
    parser.add_argument("--constraint_high_temperature_start", type=float, default=1.28)
    parser.add_argument("--constraint_small_span_prob", type=float, default=0.72)
    parser.add_argument("--constraint_medium_span_prob", type=float, default=0.82)
    parser.add_argument("--constraint_large_span_prob", type=float, default=0.90)
    parser.add_argument("--constraint_archive_size", type=int, default=140)
    parser.add_argument("--constraint_diverse_size", type=int, default=100)
    parser.add_argument("--constraint_diverse_max_sim", type=float, default=0.92)
    parser.add_argument("--constraint_similarity_slack", type=float, default=0.10)
    parser.add_argument("--constraint_min_qed", type=float, default=0.42)
    parser.add_argument("--constraint_min_sa", type=float, default=0.42)
    parser.add_argument("--constraint_dock_archive_min_ratio", type=float, default=0.72)
    parser.add_argument("--constraint_near_miss_min_score", type=float, default=0.58)
    parser.add_argument("--constraint_near_docking_weight", type=float, default=0.34)
    parser.add_argument("--constraint_near_similarity_weight", type=float, default=0.28)
    parser.add_argument("--constraint_near_quality_weight", type=float, default=0.38)
    parser.add_argument("--constraint_warmup_iters", type=int, default=1)
    parser.add_argument("--constraint_rescue_after_iters", type=int, default=2)
    parser.add_argument("--constraint_low_sim_rate", type=float, default=0.18)
    parser.add_argument("--constraint_low_sim_quality_rate", type=float, default=0.06)
    parser.add_argument("--constraint_low_dock_rate", type=float, default=0.08)
    parser.add_argument("--constraint_low_new_rate", type=float, default=0.58)
    parser.add_argument("--evaluation_budget", type=int, default=1000)
    parser.add_argument("--tf_feedback_batch_size", type=int, default=25)
    parser.add_argument("--tf_proposal_batch_size", type=int, default=24)
    parser.add_argument("--tf_operator_overgenerate_factor", type=float, default=6.0)
    parser.add_argument("--tf_max_generation_rounds", type=int, default=20)
    parser.add_argument("--tf_empty_batch_patience", type=int, default=20)
    parser.add_argument("--tf_warmup_evaluations", type=int, default=100)
    parser.add_argument("--tf_stagnation_batches", type=int, default=4)
    parser.add_argument("--tf_archive_size", type=int, default=200)
    parser.add_argument("--tf_parent_top_k", type=int, default=40)
    parser.add_argument("--tf_bandit_alpha", type=float, default=0.08)
    parser.add_argument("--tf_bandit_temperature", type=float, default=2.0)
    parser.add_argument("--tf_ucb_weight", type=float, default=0.40)
    parser.add_argument("--tf_min_operator_weight", type=float, default=0.04)
    parser.add_argument("--tf_positive_reward_threshold", type=float, default=0.60)
    parser.add_argument("--tf_boundary_epsilon", type=float, default=0.01)
    parser.add_argument("--tf_violation_gain_scale", type=float, default=0.10)
    parser.add_argument("--tf_docking_gain_scale", type=float, default=1.0)
    parser.add_argument("--tf_constraint_crossing_bonus", type=float, default=0.08)
    parser.add_argument("--tf_credit_alpha", type=float, default=0.15)
    parser.add_argument("--tf_fragment_prior_weight", type=float, default=0.35)
    parser.add_argument("--tf_graph_cut_rounds", type=int, default=6)
    parser.add_argument("--tf_graph_edit_attempts", type=int, default=120)
    parser.add_argument("--tf_replacement_pool_size", type=int, default=80)
    parser.add_argument("--tf_min_size_ratio", type=float, default=0.55)
    parser.add_argument("--tf_max_size_ratio", type=float, default=1.60)
    parser.add_argument("--tf_micro_remask", type=float, default=0.06)
    parser.add_argument("--tf_micro_temperature", type=float, default=0.92)
    parser.add_argument("--tf_micro_span_prob", type=float, default=0.70)
    parser.add_argument("--tf_small_remask", type=float, default=0.12)
    parser.add_argument("--tf_small_temperature", type=float, default=0.98)
    parser.add_argument("--tf_small_span_prob", type=float, default=0.76)
    parser.add_argument("--tf_medium_remask", type=float, default=0.22)
    parser.add_argument("--tf_medium_temperature", type=float, default=1.10)
    parser.add_argument("--tf_medium_span_prob", type=float, default=0.84)
    parser.add_argument("--tf_graph_remask", type=float, default=0.10)
    parser.add_argument("--tf_graph_temperature", type=float, default=1.02)
    parser.add_argument("--tf_graph_span_prob", type=float, default=0.80)
    parser.add_argument("--tf_restart_remask", type=float, default=0.30)
    parser.add_argument("--tf_restart_temperature", type=float, default=1.22)
    parser.add_argument("--tf_restart_span_prob", type=float, default=0.88)
    parser.add_argument("--residual_operator_batch_size", type=int, default=36)
    parser.add_argument("--residual_overgenerate_factor", type=float, default=2.4)
    parser.add_argument("--residual_keep_weight", type=float, default=1.10)
    parser.add_argument("--residual_micro_weight", type=float, default=1.20)
    parser.add_argument("--residual_shrink_weight", type=float, default=0.90)
    parser.add_argument("--residual_expand_weight", type=float, default=0.80)
    parser.add_argument("--residual_symmetric_weight", type=float, default=1.00)
    parser.add_argument("--residual_rescue_weight", type=float, default=0.35)
    parser.add_argument("--residual_min_operator_weight", type=float, default=0.04)
    parser.add_argument("--residual_max_operator_weight", type=float, default=8.0)
    parser.add_argument("--residual_bandit_eta", type=float, default=0.75)
    parser.add_argument("--residual_reward_decay", type=float, default=0.55)
    parser.add_argument("--residual_reward_clip", type=float, default=0.45)
    parser.add_argument("--residual_strict_reward", type=float, default=0.22)
    parser.add_argument("--residual_sim_quality_reward", type=float, default=0.08)
    parser.add_argument("--residual_sim_dock_reward", type=float, default=0.06)
    parser.add_argument("--residual_improvement_reward", type=float, default=0.45)
    parser.add_argument("--residual_rescue_after_iters", type=int, default=3)
    parser.add_argument("--residual_rescue_boost", type=float, default=2.2)
    parser.add_argument("--residual_refine_score", type=float, default=0.96)
    parser.add_argument("--residual_archive_min_score", type=float, default=0.72)
    parser.add_argument("--residual_soft_residual", type=float, default=0.18)
    parser.add_argument("--residual_strict_score_bonus", type=float, default=0.10)
    parser.add_argument("--residual_dock_weight", type=float, default=0.30)
    parser.add_argument("--residual_sim_weight", type=float, default=0.30)
    parser.add_argument("--residual_qed_weight", type=float, default=0.27)
    parser.add_argument("--residual_sa_weight", type=float, default=0.13)
    parser.add_argument("--residual_keep_remask", type=float, default=0.08)
    parser.add_argument("--residual_keep_temperature", type=float, default=0.92)
    parser.add_argument("--residual_keep_span_prob", type=float, default=0.74)
    parser.add_argument("--residual_micro_remask", type=float, default=0.10)
    parser.add_argument("--residual_micro_temperature", type=float, default=0.96)
    parser.add_argument("--residual_micro_span_prob", type=float, default=0.76)
    parser.add_argument("--residual_micro_length_prob", type=float, default=0.55)
    parser.add_argument("--residual_micro_max_span", type=int, default=4)
    parser.add_argument("--residual_shrink_remask", type=float, default=0.13)
    parser.add_argument("--residual_shrink_temperature", type=float, default=1.00)
    parser.add_argument("--residual_shrink_span_prob", type=float, default=0.80)
    parser.add_argument("--residual_shrink_deltas", type=str, default="-6:-4:-2:0")
    parser.add_argument("--residual_shrink_length_prob", type=float, default=0.82)
    parser.add_argument("--residual_shrink_min_span", type=int, default=3)
    parser.add_argument("--residual_shrink_max_span", type=int, default=10)
    parser.add_argument("--residual_expand_remask", type=float, default=0.15)
    parser.add_argument("--residual_expand_temperature", type=float, default=1.08)
    parser.add_argument("--residual_expand_span_prob", type=float, default=0.82)
    parser.add_argument("--residual_expand_deltas", type=str, default="0:1:2:4")
    parser.add_argument("--residual_expand_length_prob", type=float, default=0.76)
    parser.add_argument("--residual_expand_min_span", type=int, default=2)
    parser.add_argument("--residual_expand_max_span", type=int, default=8)
    parser.add_argument("--residual_symmetric_remask", type=float, default=0.18)
    parser.add_argument("--residual_symmetric_temperature", type=float, default=1.12)
    parser.add_argument("--residual_symmetric_span_prob", type=float, default=0.84)
    parser.add_argument("--residual_symmetric_deltas", type=str, default="-3:-1:0:1:3")
    parser.add_argument("--residual_symmetric_length_prob", type=float, default=0.74)
    parser.add_argument("--residual_symmetric_min_span", type=int, default=2)
    parser.add_argument("--residual_symmetric_max_span", type=int, default=8)
    parser.add_argument("--residual_rescue_remask", type=float, default=0.28)
    parser.add_argument("--residual_rescue_temperature", type=float, default=1.24)
    parser.add_argument("--residual_rescue_span_prob", type=float, default=0.90)
    parser.add_argument("--residual_rescue_deltas", type=str, default="-8:-4:0:4:8")
    parser.add_argument("--residual_rescue_length_prob", type=float, default=0.90)
    parser.add_argument("--residual_rescue_min_span", type=int, default=3)
    parser.add_argument("--residual_rescue_max_span", type=int, default=12)

    parser.add_argument("--max_len", type=int, default=128)
    parser.add_argument("--batch_size", type=int, default=64)
    parser.add_argument("--n_steps", type=int, default=250)
    parser.add_argument("--remask_fraction", type=float, default=0.35)
    parser.add_argument("--min_remask_tokens", type=int, default=2)
    parser.add_argument("--span_prob", type=float, default=0.7)
    parser.add_argument("--length_delta_choices", type=str, default="0")
    parser.add_argument("--length_edit_prob", type=float, default=0.0)
    parser.add_argument("--length_edit_min_span", type=int, default=1)
    parser.add_argument("--length_edit_max_span", type=int, default=8)

    parser.add_argument("--disable_fsm_check", action="store_true")
    parser.add_argument("--disable_rdkit_kekulize_check", action="store_true")
    parser.add_argument("--rdkit_check_interval", type=int, default=25)
    parser.add_argument("--max_sample_retries", type=int, default=2)
    parser.add_argument("--violation_neighborhood", type=int, default=2)
    parser.add_argument("--temperature_start", type=float, default=1.2)
    parser.add_argument("--temperature_end", type=float, default=0.2)
    parser.add_argument("--temperature_power", type=float, default=1.5)
    return parser.parse_args()


def main():
    args = parse_args()
    CSDNetLeadOptimizer(args).run()


if __name__ == "__main__":
    main()
