#!/usr/bin/env python
import argparse
import csv
import hashlib
import math
import os
import pickle
import random
import sys
from collections import defaultdict
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
    resolve_local_sampler_profile,
    sample_csdnet_local_remask,
    tokenizable,
)
from CSDNet.util.elastic_sampling import sample_elastic_local_infill
from CSDNet.util.length_prior import load_atomic_length_prior
from CSDNet.exp.lead.frontier import (
    adaptive_peripheral_edit_plan,
    allocate_counts,
    archive_constraint_need,
    completion_recovery_multipliers,
    atom_span_edit_plan,
    constraint_state,
    frontier_labels,
    merge_archive,
    peripheral_edit_plan,
    transition_reward,
    upper_tail_reward,
    recovery_v2_operator_multipliers,
    recovery_v2_state,
    seed_directed_atom_edit_plan,
)
from CSDNet.optim.frontier import (
    LeadBestUnionAdapter,
    LeadFrontierAdapter,
    LeadFrontierAdapterV21,
    RestoredLeadFrontierAdapter,
    UnifiedFrontierEngine,
    allocate_insertion_flags,
    classify_frontier_state,
    constraint_rank,
    integrated_operator_weights,
    lineage_metrics,
    rank_improved,
    trust_region_fraction,
)
from CSDNet.optim.protected_frontier import (
    BaselineProtectedFrontierEngine,
    SafeLeadBridgeHead,
    SafeLeadFrontierHead,
)
from CSDNet.exp.lead.task_head import (
    AnchoredRestartCompletionLeadHead,
    ProtectedCompletionLeadHead,
    ProtectedRoutePortfolioLeadHead,
    ReversibleRouteCompletionLeadHead,
    choose_robust_completion_parent,
)
from CSDNet.util.tokenizer import SMILESTokenizer, tokenize_smiles


RDLogger.DisableLog("rdApp.*")
ROOT_DIR = Path(__file__).resolve().parent
UNIFIED_FRONTIER_PROFILES = {
    "lead_best_union",
    "unified_frontier_v2",
    "unified_frontier_v2_1",
    "unified_frontier_restored",
}
INTEGRATED_FRONTIER_PROFILES = {"integrated_frontier", *UNIFIED_FRONTIER_PROFILES}
UNIVERSAL_FRONTIER_PROFILES = {
    "universal_frontier",
    "universal_frontier_recovery",
    "universal_frontier_recovery_v2",
    "universal_frontier_recovery_v3",
    "safe_frontier_final",
    "universal_frontier_bridge",
    "lead_protected_completion",
    "lead_protected_completion_v2",
    "lead_protected_completion_v3",
    "lead_protected_completion_v4",
    "lead_protected_completion_v5",
}
PROTECTED_LEAD_PROFILES = {
    "safe_frontier_final",
    "universal_frontier_bridge",
    "lead_protected_completion",
    "lead_protected_completion_v2",
    "lead_protected_completion_v3",
    "lead_protected_completion_v4",
    "lead_protected_completion_v5",
}
ELASTIC_LEAD_PROFILES = {
    "elastic_direct",
    "elastic_joint_frontier",
    "elastic_joint_frontier_v2",
    "elastic_joint_frontier_v3",
    "elastic_joint_frontier_v4",
    "elastic_joint_frontier_v5",
}
ELASTIC_FRONTIER_PROFILES = {
    "elastic_joint_frontier",
    "elastic_joint_frontier_v2",
    "elastic_joint_frontier_v3",
    "elastic_joint_frontier_v4",
    "elastic_joint_frontier_v5",
}
FRONTIER_PROFILES = {
    "multi_frontier",
    *UNIVERSAL_FRONTIER_PROFILES,
    *INTEGRATED_FRONTIER_PROFILES,
    *ELASTIC_FRONTIER_PROFILES,
}


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
    return all(a[key] >= b[key] for key in keys) and any(
        a[key] > b[key] for key in keys
    )


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


def is_learned_length_mode(mode):
    """Recognize both flat and recursive learned-insertion diagnostics."""
    return str(mode or "").startswith("learned")


def token_safe_model_smiles(smiles, tk):
    """Return a graph-equivalent canonical SMILES covered by the model vocab.

    The benchmark actives contain directional bonds and tetrahedral stereo
    tokens that are absent from the Safe-GPT vocabulary. Morgan similarity in
    this benchmark is achiral, so removing only stereochemical annotations
    preserves the scored molecular graph while preventing frozen ``<unk>``
    tokens from making every local edit undecodable.
    """
    mol = Chem.MolFromSmiles(smiles)
    if mol is None:
        raise ValueError(f"Invalid Lead start SMILES: {smiles!r}")

    canonical = Chem.MolToSmiles(mol, canonical=True)
    if tokenizable(canonical, tk, 10**9):
        return canonical, False

    nonisomeric = Chem.MolToSmiles(
        mol,
        canonical=True,
        isomericSmiles=False,
    )
    if tokenizable(nonisomeric, tk, 10**9):
        return nonisomeric, True

    unsupported = sorted(
        {token for token in tokenize_smiles(nonisomeric) if token not in tk.vocab}
    )
    raise ValueError(
        "Lead start SMILES remains outside the model vocabulary after "
        f"stereochemistry normalization: {unsupported}"
    )


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
        self.elastic_length_support = None
        self.elastic_atomic_body_lengths = None
        if args.sampler_profile in ELASTIC_LEAD_PROFILES:
            if not getattr(self.model, "is_elastic", False):
                raise RuntimeError(
                    "elastic Lead optimization requires an ElasticCSDNet "
                    "checkpoint with learned insertion rates"
                )
            if not args.atomic_length_prior:
                raise RuntimeError(
                    "elastic Lead optimization requires --atomic_length_prior"
                )
            if not (
                0.0
                <= args.direct_length_quantile_low
                < args.direct_length_quantile_high
                <= 1.0
            ):
                raise ValueError(
                    "direct length quantiles must satisfy 0 <= low < high <= 1"
                )
            reference_lengths, metadata = load_atomic_length_prior(
                args.atomic_length_prior,
                max_len=args.max_len,
            )
            self.elastic_atomic_body_lengths = np.asarray(
                [max(1, int(length) - 2) for length in reference_lengths],
                dtype=float,
            )
            lower = int(
                math.floor(
                    np.quantile(
                        self.elastic_atomic_body_lengths,
                        args.direct_length_quantile_low,
                    )
                )
            )
            upper = int(
                math.ceil(
                    np.quantile(
                        self.elastic_atomic_body_lengths,
                        args.direct_length_quantile_high,
                    )
                )
            )
            self.elastic_length_support = (
                max(1, lower),
                min(args.max_len - 2, upper),
            )
            print(
                "Elastic Lead atomic-length support: "
                f"body={self.elastic_length_support[0]}-"
                f"{self.elastic_length_support[1]} tokens "
                f"source={metadata['path']}"
            )
        if resolve_local_sampler_profile() in {
            "task_adaptive_local",
            "task_adaptive_refine",
        } and not bool(getattr(self.model, "corruption_level_conditioning", False)):
            raise RuntimeError(
                "task-adaptive Lead sampling requires the trajectory-refinement "
                "checkpoint; corruption-level conditioning is absent"
            )

        active_path = ROOT_DIR / "docking" / "actives.csv"
        df = pd.read_csv(active_path)
        df = df[df["target"] == args.oracle_name].reset_index(drop=True)
        if args.start_mol_idx >= len(df):
            raise SystemExit(
                f"start_mol_idx={args.start_mol_idx} is invalid for {args.oracle_name}; "
                f"available indices: 0..{len(df) - 1}"
            )

        row = df.iloc[args.start_mol_idx]
        benchmark_start_smiles = canonical_smiles(row["smiles"])
        self.start_smiles, stereo_normalized = token_safe_model_smiles(
            benchmark_start_smiles,
            self.tk,
        )
        self.benchmark_start_smiles = benchmark_start_smiles
        self.start_prop = float(row["DS"])
        start_mol = Chem.MolFromSmiles(benchmark_start_smiles)
        self.start_fp = AllChem.GetMorganFingerprintAsBitVect(start_mol, 2, 2048)
        if stereo_normalized:
            print(
                "Lead model seed stereochemistry normalized for vocabulary "
                f"coverage: {benchmark_start_smiles} -> {self.start_smiles}"
            )
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
            raise SystemExit(
                f"Too few initial fragments from {self.start_smiles}: {len(fragments)}"
            )
        self.population = [(self.start_prop, frag) for frag in fragments]
        self.initial_fragments = list(fragments)
        self.tf_fragment_meta = {
            frag: {"prior": 0.50, "credit": 0.50, "updates": 0.0} for frag in fragments
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
        self.current_candidate_meta = []
        self.frontier_rng = random.Random((args.seed + 1) * 1_000_003 + 97)
        self.completion_v4_rng = random.Random((args.seed + 1) * 9_999_991 + 409)
        self.completion_v5_rng = random.Random((args.seed + 1) * 15_485_863 + 521)
        self.frontier_base_operators = ("legacy",)
        if args.sampler_profile in {
            "universal_frontier_recovery_v2",
            "universal_frontier_recovery_v3",
        }:
            self.frontier_base_operators += ("legacy_local",)
        if args.sampler_profile == "lead_best_union":
            self.frontier_base_operators += ("legacy_local",)
        self.frontier_operators = (
            "start_repair",
            "dock_refine",
            "similarity_repair",
            "quality_repair",
        )
        if (
            args.sampler_profile in UNIVERSAL_FRONTIER_PROFILES
            or args.sampler_profile in INTEGRATED_FRONTIER_PROFILES
            or args.sampler_profile in ELASTIC_FRONTIER_PROFILES
        ):
            self.frontier_operators += ("joint_repair",)
        if args.sampler_profile == "universal_frontier_bridge":
            self.frontier_operators += ("pair_bridge",)
        self.frontier_task_head_operators = ()
        if args.sampler_profile == "universal_frontier_bridge":
            self.frontier_task_head_operators = ("pair_bridge",)
        if args.sampler_profile in {
            "lead_protected_completion",
            "lead_protected_completion_v2",
            "lead_protected_completion_v3",
            "lead_protected_completion_v4",
            "lead_protected_completion_v5",
        }:
            self.frontier_task_head_operators = (
                "pair_bridge",
                "feasible_dock_polish",
                "boundary_similarity_polish",
                "boundary_quality_polish",
            )
            if args.sampler_profile == "lead_protected_completion_v4":
                self.frontier_task_head_operators += ("no_pair_anchor_restart",)
            if args.sampler_profile == "lead_protected_completion_v5":
                self.frontier_task_head_operators += (
                    "completion_dock_refine",
                    "completion_similarity_repair",
                    "completion_quality_repair",
                    "completion_seed_restart",
                )
            self.frontier_operators += self.frontier_task_head_operators
        if args.sampler_profile in INTEGRATED_FRONTIER_PROFILES:
            self.frontier_operators += ("lineage_restart",)
        if args.sampler_profile == "unified_frontier_v2_1":
            self.frontier_operators += ("scaffold_rescue",)
        self.frontier_archives = {
            label: [] for label in ("near", "s", "sq", "qd", "sd", "strict")
        }
        self.frontier_items = {}
        # V5 keeps public-constraint near misses available as proposal parents
        # without assigning them a docking score or spending oracle budget.
        self.elastic_public_frontier = {}
        self.frontier_operator_stats = {
            operator: {
                "attempted": 0.0,
                "accepted": 0.0,
                "evaluated": 0.0,
                "updates": 0.0,
                "reward_ema": 0.0,
                "last_batch_reward": 0.0,
                "strict": 0.0,
            }
            for operator in self.frontier_base_operators + self.frontier_operators
        }
        self.frontier_iteration = 0
        self.last_frontier_summary = "uninitialized"
        self.frontier_last_needs = {}
        self.frontier_last_allocation = {}
        self.completion_parent_counters = defaultdict(int)
        self.completion_span_counters = defaultdict(int)
        self.evaluation_cache = {}
        self.integrated_best_item = None
        self.integrated_stagnant_iters = 0
        self.integrated_state = "warmup"
        self.integrated_lineage = {
            "root_count": 0,
            "largest_root_fraction": 0.0,
            "lineage_entropy": 0.0,
        }
        self.integrated_root_credit = defaultdict(lambda: 0.5)
        self.frontier_engine = None
        self.protected_frontier_engine = None
        self.safe_final_seen_smiles = set()
        self.safe_final_official_feasible = False
        self.oracle_evaluated_smiles = set()
        if args.global_oracle_dedup:
            # The seed's docking score is supplied by the benchmark, so querying
            # the unchanged seed again cannot add information.
            self.oracle_evaluated_smiles.add(self.start_smiles)
        self.initial_population_size = len(self.population)
        if args.sampler_profile == "safe_frontier_final":
            self.protected_frontier_engine = BaselineProtectedFrontierEngine(
                SafeLeadFrontierHead(
                    rescue_start_iteration=args.frontier_recovery_start_iter,
                    reserve_share=1.0 - args.frontier_recovery_legacy_fraction,
                )
            )
        elif args.sampler_profile == "universal_frontier_bridge":
            self.protected_frontier_engine = BaselineProtectedFrontierEngine(
                SafeLeadBridgeHead(
                    start_iteration=args.frontier_bridge_start_iter,
                    reserve_share=args.frontier_bridge_fraction,
                )
            )
        elif args.sampler_profile in {
            "lead_protected_completion",
            "lead_protected_completion_v2",
            "lead_protected_completion_v3",
            "lead_protected_completion_v4",
            "lead_protected_completion_v5",
        }:
            head_cls = {
                "lead_protected_completion": ProtectedCompletionLeadHead,
                "lead_protected_completion_v2": ProtectedCompletionLeadHead,
                "lead_protected_completion_v3": ReversibleRouteCompletionLeadHead,
                "lead_protected_completion_v4": AnchoredRestartCompletionLeadHead,
                "lead_protected_completion_v5": ProtectedRoutePortfolioLeadHead,
            }[args.sampler_profile]
            reversible_kwargs = (
                {
                    "late_start_iteration": args.completion_v3_late_start_iter,
                    "probe_share": args.completion_v3_probe_fraction,
                    "commit_share": args.completion_v3_commit_fraction,
                    "max_route_deficit": args.completion_v3_max_route_deficit,
                    "route_tie_tolerance": args.completion_v3_route_tie_tolerance,
                    "min_absolute_improvement": args.completion_v3_min_abs_improvement,
                    "min_relative_improvement": args.completion_v3_min_rel_improvement,
                }
                if args.sampler_profile
                in {"lead_protected_completion_v3", "lead_protected_completion_v4"}
                else {}
            )
            if args.sampler_profile == "lead_protected_completion_v5":
                reversible_kwargs.update(
                    {
                        "late_start_iteration": args.completion_v5_late_start_iter,
                        "portfolio_share": args.completion_v5_portfolio_fraction,
                        "max_route_deficit": args.completion_v5_max_route_deficit,
                        "route_temperature": args.completion_v5_route_temperature,
                        "minimum_route_weight": args.completion_v5_min_route_weight,
                        "seed_probe_share": args.completion_v5_seed_probe_fraction,
                    }
                )
            if args.sampler_profile == "lead_protected_completion_v4":
                reversible_kwargs.update(
                    {
                        "anchor_start_iteration": args.completion_v4_anchor_start_iter,
                        "anchor_probe_share": args.completion_v4_anchor_probe_fraction,
                        "anchor_route_probe_share": args.completion_v4_route_probe_fraction,
                        "anchor_route_commit_share": args.completion_v4_route_commit_fraction,
                        "anchor_route_max_deficit": args.completion_v4_route_max_deficit,
                    }
                )
            self.protected_frontier_engine = BaselineProtectedFrontierEngine(
                head_cls(
                    start_iteration=args.completion_start_iter,
                    bridge_share=args.completion_bridge_fraction,
                    dock_polish_share=args.completion_dock_fraction,
                    boundary_polish_share=args.completion_boundary_fraction,
                    boundary_tolerance=args.completion_boundary_tolerance,
                    **reversible_kwargs,
                )
            )
        if args.sampler_profile in UNIFIED_FRONTIER_PROFILES:
            adapter_cls = {
                "lead_best_union": LeadBestUnionAdapter,
                "unified_frontier_v2": LeadFrontierAdapter,
                "unified_frontier_v2_1": LeadFrontierAdapterV21,
                "unified_frontier_restored": RestoredLeadFrontierAdapter,
            }[args.sampler_profile]
            self.frontier_engine = UnifiedFrontierEngine(
                adapter=adapter_cls(
                    warmup_iterations=args.integrated_warmup_iterations,
                    plateau_patience=args.integrated_plateau_patience,
                    collapse_threshold=args.integrated_collapse_threshold,
                    need_weight=args.frontier_need_weight,
                    tail_fraction=args.frontier_reward_tail_fraction,
                    mean_weight=args.frontier_reward_mean_weight,
                    regression_penalty=max(
                        0.30,
                        args.frontier_regression_penalty,
                    ),
                ),
                operator_groups={
                    "proposal": self.frontier_base_operators + self.frontier_operators,
                },
                bandit_configs={
                    "proposal": {
                        "alpha": args.frontier_reward_alpha,
                        "temperature": args.frontier_bandit_eta,
                        "ucb_weight": args.frontier_bandit_ucb,
                        "min_multiplier": 0.01,
                        "base_floor": 0.02,
                    },
                },
            )
        self.last_cache_hits = 0
        self.last_cache_misses = 0
        self.residual_length_ops = (
            "keep",
            "micro",
            "shrink",
            "expand",
            "symmetric",
            "rescue",
        )
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
            "legacy_fragment_floor",
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
        self.frontier_diagnostic_path = os.path.join(
            args.output_dir,
            f"frontier_diagnostics_{args.oracle_name}_id{args.start_mol_idx}_thr{args.sim_thr}_{args.seed}.csv",
        )
        if os.path.exists(self.fname) and not args.resume:
            os.remove(self.fname)
        transition_profiles = {"transition_feasible", "transition_feasible_hybrid"}
        if args.sampler_profile in transition_profiles and not args.resume:
            for path in (self.tf_transition_path, self.tf_diagnostic_path):
                if os.path.exists(path):
                    os.remove(path)
        if args.sampler_profile in FRONTIER_PROFILES and not args.resume:
            if os.path.exists(self.frontier_diagnostic_path):
                os.remove(self.frontier_diagnostic_path)

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
        frontier_start = self.frontier_make_item(
            self.start_smiles,
            self.start_prop,
            start_qed,
            start_sa,
            1.0,
        )
        frontier_start.update(
            {
                "root_id": "start",
                "depth": 0,
                "parent_smiles": None,
            }
        )
        self.frontier_items[self.start_smiles] = frontier_start
        self.frontier_archives["near"] = [frontier_start]
        self.frontier_archives["s"] = [frontier_start]
        self.integrated_best_item = frontier_start
        if (
            args.sampler_profile in INTEGRATED_FRONTIER_PROFILES
            or args.sampler_profile
            in {
                "universal_frontier_recovery_v2",
                "universal_frontier_recovery_v3",
                *ELASTIC_LEAD_PROFILES,
            }
        ):
            self.evaluation_cache[self.start_smiles] = (
                self.start_prop,
                start_qed,
                start_sa,
                1.0,
            )
        if args.sampler_profile in transition_profiles and args.resume:
            self._load_transition_feasible_resume()
        self.resume_total = 0
        self.resume_raw_best = self.start_prop
        self.resume_feasible_best = self.start_prop
        self.resume_iteration_offset = 0
        if args.resume and os.path.exists(self.fname):
            self._load_oracle_resume_history()

        print(f"Start SMILES:\t{self.start_smiles}")
        print(f"Start DS:\t{self.start_prop}")
        print(f"Initial population:\t{len(self.population)} fragments")
        print(f"Sampler profile:\t{args.sampler_profile}")
        print(f"Pareto population update:\t{args.pareto_population_update}")
        print(f"Remask fraction:\t{args.remask_fraction}")
        print(f"Output:\t{self.fname}")

    def _load_oracle_resume_history(self):
        """Restore prior unique oracle calls and their searchable frontier."""
        if not self.args.global_oracle_dedup:
            raise RuntimeError(
                "--resume for Lead oracle completion requires --global_oracle_dedup."
            )

        smiles_list = []
        docks = []
        qeds = []
        sas = []
        similarities = []
        seen = set()
        with open(self.fname, newline="") as handle:
            for line_number, row in enumerate(csv.reader(handle), start=1):
                if not row or all(not value.strip() for value in row):
                    continue
                if len(row) < 5:
                    raise RuntimeError(
                        f"Malformed Lead resume row {line_number} in "
                        f"{self.fname}: expected at least 5 columns."
                    )
                can = canonical_smiles(row[0])
                if can is None:
                    raise RuntimeError(
                        f"Invalid SMILES in Lead resume row {line_number}: {row[0]!r}"
                    )
                if can in seen:
                    raise RuntimeError(
                        "Lead resume file contains duplicate canonical oracle "
                        f"calls at row {line_number}: {can}"
                    )
                try:
                    dock, qed, sa, similarity = (
                        float(row[1]),
                        float(row[2]),
                        float(row[3]),
                        float(row[4]),
                    )
                except ValueError as exc:
                    raise RuntimeError(
                        f"Non-numeric Lead resume row {line_number} in {self.fname}."
                    ) from exc
                seen.add(can)
                smiles_list.append(can)
                docks.append(dock)
                qeds.append(qed)
                sas.append(sa)
                similarities.append(similarity)

        if len(smiles_list) > int(self.args.oracle_budget):
            raise RuntimeError(
                f"Lead resume file already exceeds oracle budget: "
                f"{len(smiles_list)}/{self.args.oracle_budget}"
            )
        if not smiles_list:
            print(f"[Resume] no prior oracle rows found in {self.fname}")
            return

        prop_list = (docks, qeds, sas, similarities)
        self.oracle_evaluated_smiles.update(smiles_list)
        self.generated_seen.update(smiles_list)
        for smiles, values in zip(smiles_list, zip(*prop_list)):
            self.evaluation_cache[smiles] = tuple(float(value) for value in values)

        self.resume_total = len(smiles_list)
        # Standardized jobs write frontier diagnostics after every evaluated
        # batch. If they are unavailable, conservatively assume that the
        # baseline phase finished and resume only budget completion.
        self.resume_iteration_offset = self.args.num_iter
        if os.path.exists(self.frontier_diagnostic_path):
            try:
                diagnostics = pd.read_csv(
                    self.frontier_diagnostic_path,
                    usecols=["iteration"],
                )
                if not diagnostics.empty:
                    diagnostic_iteration = int(
                        pd.to_numeric(
                            diagnostics["iteration"],
                            errors="coerce",
                        ).max()
                    )
                    self.resume_iteration_offset = max(
                        0,
                        diagnostic_iteration,
                    )
            except (ValueError, TypeError):
                pass
        self.frontier_iteration = self.resume_iteration_offset

        self.current_candidate_ops = ["legacy"] * len(smiles_list)
        self.current_candidate_meta = [
            {
                "operator": "legacy",
                "parent_smiles": self.start_smiles,
                "root_id": hashlib.sha1(smiles.encode("utf-8")).hexdigest()[:12],
                "depth": 1,
                "search_state": "resume",
            }
            for smiles in smiles_list
        ]
        self.frontier_update_state(smiles_list, prop_list)
        self.update_population(smiles_list, prop_list)

        feasible_docks = [
            dock
            for dock, qed, sa, similarity in zip(*prop_list)
            if (
                dock > self.start_prop
                and qed >= 0.6
                and sa >= 6 / 9
                and similarity >= self.args.sim_thr
            )
        ]
        self.resume_raw_best = max([self.start_prop, *docks])
        self.resume_feasible_best = max([self.start_prop, *feasible_docks])

        self.current_candidate_ops = []
        self.current_candidate_meta = []
        print(
            f"[Resume] restored calls={self.resume_total}/"
            f"{self.args.oracle_budget}, "
            f"raw_best={self.resume_raw_best:.3f}, "
            f"feasible_best={self.resume_feasible_best:.3f}, "
            f"iteration_offset={self.resume_iteration_offset}, "
            f"oracle_seen={len(self.oracle_evaluated_smiles)}"
        )

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
        if (
            self.args.sampler_profile in INTEGRATED_FRONTIER_PROFILES
            or self.args.sampler_profile
            in {
                "universal_frontier_recovery_v2",
                "universal_frontier_recovery_v3",
                *ELASTIC_LEAD_PROFILES,
            }
        ):
            return self.reward_cached(smiles_list)
        mols = [Chem.MolFromSmiles(s) for s in smiles_list]
        return (
            self.reward_vina(smiles_list),
            self.reward_qed(mols),
            self.reward_sa(mols),
            self.reward_sim(mols),
        )

    def reward_cached(self, smiles_list):
        """Evaluate each canonical molecule once and reuse its first docking result."""
        canonical = [canonical_smiles(smiles) or smiles for smiles in smiles_list]
        missing = []
        missing_seen = set()
        for smiles in canonical:
            if smiles not in self.evaluation_cache and smiles not in missing_seen:
                missing.append(smiles)
                missing_seen.add(smiles)

        if missing:
            mols = [Chem.MolFromSmiles(smiles) for smiles in missing]
            values = zip(
                self.reward_vina(missing),
                self.reward_qed(mols),
                self.reward_sa(mols),
                self.reward_sim(mols),
            )
            for smiles, value in zip(missing, values):
                self.evaluation_cache[smiles] = tuple(float(item) for item in value)

        self.last_cache_misses = len(missing)
        self.last_cache_hits = len(canonical) - len(missing)
        rows = [self.evaluation_cache[smiles] for smiles in canonical]
        return tuple([row[index] for row in rows] for index in range(4))

    def make_seed(self):
        if (
            self.args.sampler_profile
            in {
                "similarity_aware",
                "adaptive_similarity",
                "multi_frontier",
                "lead_best_union",
            }
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
            op: max(
                self.args.residual_min_operator_weight,
                self.residual_length_weights.get(op, 0.0),
            )
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
            return {
                op: 1.0 / len(self.residual_length_ops)
                for op in self.residual_length_ops
            }
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
        sim_res = max(0.0, self.args.sim_thr - float(rsim)) / max(
            self.args.sim_thr, 1e-6
        )
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
        for rv, rq, rs, rsim, smiles, op in zip(
            rv_list, rq_list, rs_list, rsim_list, smiles_list, operators
        ):
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
        feasible = sim >= self.args.sim_thr and qed >= 0.6 and sa >= sa_thr and dock_ok
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
                item for item in self.tf_near_archive if item["bottleneck"] == name
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
        if operator == "legacy_fragment_floor":
            return (
                self.args.remask_fraction,
                self.args.temperature_start,
                self.args.span_prob,
            )
        if operator == "local_micro":
            return (
                self.args.tf_micro_remask,
                self.args.tf_micro_temperature,
                self.args.tf_micro_span_prob,
            )
        if operator == "local_small":
            return (
                self.args.tf_small_remask,
                self.args.tf_small_temperature,
                self.args.tf_small_span_prob,
            )
        if operator == "local_medium":
            return (
                self.args.tf_medium_remask,
                self.args.tf_medium_temperature,
                self.args.tf_medium_span_prob,
            )
        if operator.startswith("graph_"):
            return (
                self.args.tf_graph_remask,
                self.args.tf_graph_temperature,
                self.args.tf_graph_span_prob,
            )
        return (
            self.args.tf_restart_remask,
            self.args.tf_restart_temperature,
            self.args.tf_restart_span_prob,
        )

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
        core_pool = valid_parent_fragments[
            : max(1, (len(valid_parent_fragments) + 1) // 2)
        ]

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
        if operator in {"fragment_restart", "legacy_fragment_floor"}:
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
        failed_operators = []
        batch_seen = set()
        stats = defaultdict(int)
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
            if self.args.sampler_profile == "transition_feasible_hybrid":
                floor_fraction = clamp(self.args.tf_legacy_floor_fraction, 0.0, 1.0)
                floor_n = min(request_n, int(math.ceil(request_n * floor_fraction)))
                floor_attempts = 0
                while len(proposals) < floor_n and floor_attempts < floor_n * 100:
                    floor_attempts += 1
                    proposal = self.tf_make_proposal("legacy_fragment_floor", context)
                    if proposal is not None:
                        proposals.append(proposal)
                stats["floor_requested"] += floor_n
                stats["floor_proposals"] += len(proposals)
            attempts = 0
            while len(proposals) < request_n and attempts < request_n * 100:
                attempts += 1
                operator = self.tf_choose_operator(context)
                proposal = self.tf_make_proposal(operator, context)
                if proposal is not None:
                    proposals.append(proposal)
            if not proposals:
                continue
            stats["proposals"] += len(proposals)

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
                stats["model_outputs"] += len(candidates)
                accepted_proposals = set()

                def consider_candidate(smiles, proposal_idx, source):
                    stats[f"seen_{source}"] += 1
                    if not (0 <= proposal_idx < len(operator_proposals)):
                        stats["bad_lineage"] += 1
                        return False
                    can = canonical_smiles(smiles)
                    if can is None:
                        stats["invalid"] += 1
                        return False
                    if can in batch_seen or can in self.tf_seen_smiles:
                        stats["duplicate"] += 1
                        return False
                    atoms = atom_count(can)
                    if atoms < self.args.min_atoms or atoms > self.args.max_atoms:
                        stats["size"] += 1
                        return False
                    if not tokenizable(can, self.tk, self.args.max_len):
                        stats["token"] += 1
                        return False
                    batch_seen.add(can)
                    self.tf_seen_smiles.add(can)
                    out.append(can)
                    lineage.append(operator_proposals[proposal_idx])
                    accepted_proposals.add(proposal_idx)
                    stats[f"accepted_{source}"] += 1
                    return True

                for smiles, proposal_idx in candidates:
                    consider_candidate(smiles, proposal_idx, "model")
                    if len(out) >= target_n:
                        break

                # Graph edits and fragment restarts are valid proposal operators
                # in their own right. If denoising returns only the parent or an
                # invalid string, evaluate the chemically valid edited seed rather
                # than discarding the whole proposal batch.
                if len(out) < target_n and operator in {
                    "graph_shrink",
                    "graph_expand",
                    "graph_swap",
                    "fragment_restart",
                    "legacy_fragment_floor",
                }:
                    for proposal_idx, proposal in enumerate(operator_proposals):
                        if proposal_idx in accepted_proposals:
                            continue
                        consider_candidate(proposal["seed"], proposal_idx, "fallback")
                        if len(out) >= target_n:
                            break
                if len(out) < target_n:
                    for proposal_idx in range(len(operator_proposals)):
                        if proposal_idx in accepted_proposals:
                            continue
                        failed_operators.append(operator)
                        stats["failed_proposals"] += 1
                        stats[f"failed_{operator}"] += 1
                if len(out) >= target_n:
                    break
        print(
            "[TF-gen] "
            + " ".join(f"{key}={value}" for key, value in sorted(stats.items()))
            + f" accepted={len(out)}/{target_n} rounds={rounds}"
        )
        return out, lineage, failed_operators

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
            if (
                child_item["constraints_satisfied"]
                > parent_item["constraints_satisfied"]
            ):
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
                        ""
                        if proposal.get("parent_dock") is None
                        else proposal["parent_dock"]
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
        best_feasible = (
            self.tf_feasible_archive[0] if self.tf_feasible_archive else None
        )
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
                    "best_max_violation": ""
                    if best_near is None
                    else best_near["max_violation"],
                    "best_sum_violation": ""
                    if best_near is None
                    else best_near["sum_violation"],
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
            self.tf_best_key = min(
                self.tf_rank_key(item) for item in self.tf_items.values()
            )

    def run_transition_feasible(self):
        t_start = time()
        self.tf_refresh_archives()
        if self.tf_best_key is None:
            self.tf_best_key = min(
                self.tf_rank_key(item) for item in self.tf_items.values()
            )
        evaluated_count = len(self.tf_evaluated_smiles)
        empty_batches = 0
        while evaluated_count < self.args.evaluation_budget:
            context = self.tf_context(evaluated_count)
            self.tf_last_context = context
            target_n = min(
                self.args.tf_feedback_batch_size,
                self.args.evaluation_budget - evaluated_count,
            )
            smiles_list, lineage, failed_operators = self.tf_generate_unique_batch(
                target_n, context
            )
            rewards_by_operator = {op: [] for op in self.tf_operators}
            for operator in failed_operators:
                rewards_by_operator[operator].append(0.0)
            if not smiles_list:
                empty_batches += 1
                self.tf_no_improve_batches += 1
                self.tf_update_arm_stats(context, rewards_by_operator)
                print(
                    f"[TF] no unique candidates context={context} "
                    f"empty_batches={empty_batches}/{self.args.tf_empty_batch_patience}"
                )
                if empty_batches >= self.args.tf_empty_batch_patience:
                    break
                continue
            empty_batches = 0

            prop_list = self.reward(smiles_list)
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
                    rewards_by_operator[proposal["operator"]].append(0.0)
                    continue
                parent_item = self.tf_items.get(proposal.get("parent_smiles"))
                reward, reward_parts = self.tf_transition_reward(
                    parent_item, child_item
                )
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

            best_feasible = (
                self.tf_feasible_archive[0] if self.tf_feasible_archive else None
            )
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
            return self._archive_choice(self.diverse_elite) or self.make_fragment_seed()
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
        mix = [
            (name, weight)
            for name, weight in self.constraint_ladder_mix()
            if weight > 0
        ]
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

    def frontier_make_item(self, smiles, dock, qed, sa, similarity):
        can = canonical_smiles(smiles)
        state = constraint_state(
            dock=dock,
            qed=qed,
            sa=sa,
            similarity=similarity,
            start_dock=self.start_prop,
            similarity_threshold=self.args.sim_thr,
            docking_margin=self.args.frontier_docking_margin,
            residual_l1_weight=self.args.frontier_residual_l1_weight,
        )
        return {
            "smiles": can or smiles,
            "dock": float(dock),
            "qed": float(qed),
            "sa": float(sa),
            "sim": float(similarity),
            **state,
        }

    def frontier_update_state(self, smiles_list, prop_list):
        if self.args.sampler_profile not in FRONTIER_PROFILES:
            return
        rv_list, rq_list, rs_list, rsim_list = prop_list
        self.frontier_archives.setdefault("near", [])
        if not hasattr(self, "frontier_last_needs"):
            self.frontier_last_needs = {}
        previous_items = dict(self.frontier_items)
        new_items = []
        grouped = {label: [] for label in self.frontier_archives}
        for idx, (smiles, rv, rq, rs, rsim) in enumerate(
            zip(smiles_list, rv_list, rq_list, rs_list, rsim_list)
        ):
            meta = (
                self.current_candidate_meta[idx]
                if idx < len(self.current_candidate_meta)
                else {}
            )
            item = self.frontier_make_item(smiles, rv, rq, rs, rsim)
            if self.args.sampler_profile in PROTECTED_LEAD_PROFILES:
                first_official_evaluation = (
                    item["smiles"] not in self.safe_final_seen_smiles
                )
                self.safe_final_seen_smiles.add(item["smiles"])
                if first_official_evaluation and item["strict"]:
                    self.safe_final_official_feasible = True
            previous = previous_items.get(item["smiles"])
            root_id = meta.get("root_id")
            if not root_id and previous is not None:
                root_id = previous.get("root_id")
            if not root_id:
                root_id = hashlib.sha1(item["smiles"].encode("utf-8")).hexdigest()[:12]
            item.update(
                {
                    "root_id": root_id,
                    "depth": int(meta.get("depth", 0)),
                    "parent_smiles": meta.get("parent_smiles"),
                }
            )
            new_items.append(item)
            self.frontier_items[item["smiles"]] = item
            grouped["near"].append(item)
            for label in frontier_labels(
                item,
                self.args.sim_thr,
                self.args.frontier_similarity_slack,
            ):
                grouped[label].append(item)

        for label, candidates in grouped.items():
            self.frontier_archives[label] = merge_archive(
                self.frontier_archives[label],
                candidates,
                label,
                self.args.frontier_archive_size,
            )

        admitted_smiles = set()
        for label in ("near", "sq", "qd", "sd", "strict"):
            limit = (
                getattr(self.args, "frontier_parent_top_k", 30)
                if label == "near"
                else None
            )
            archive = self.frontier_archives[label]
            selected = archive if limit is None else archive[:limit]
            admitted_smiles.update(item["smiles"] for item in selected)

        rewards_by_operator = defaultdict(list)
        metadata_by_operator = defaultdict(list)
        transitions_by_operator = defaultdict(list)
        for idx, item in enumerate(new_items):
            meta = (
                self.current_candidate_meta[idx]
                if idx < len(self.current_candidate_meta)
                else {}
            )
            operator = meta.get("operator", "legacy")
            stats = self.frontier_operator_stats[operator]
            parent_residual = meta.get("parent_residual")
            parent_stage = meta.get("parent_stage")
            parent = previous_items.get(meta.get("parent_smiles"))
            if parent is not None and parent_residual is None:
                parent_residual = parent["residual"]
                parent_stage = parent["stage"]
            if parent is None and self.args.sampler_profile in {
                *UNIVERSAL_FRONTIER_PROFILES,
                *INTEGRATED_FRONTIER_PROFILES,
            }:
                parent = previous_items[self.start_smiles]
                parent_residual = parent["residual"]
                parent_stage = parent["stage"]

            reward = 0.0
            reward_parts = {
                "crossed": 0,
                "regressed": 0,
                "pair_gain": 0,
            }
            if (
                self.args.sampler_profile
                in {
                    *UNIVERSAL_FRONTIER_PROFILES,
                    *INTEGRATED_FRONTIER_PROFILES,
                }
                and parent is not None
            ):
                reward_parts = transition_reward(
                    parent,
                    item,
                    crossing_bonus=self.args.frontier_crossing_bonus,
                    regression_penalty=(
                        max(0.45, self.args.frontier_regression_penalty)
                        if self.args.sampler_profile == "universal_frontier_recovery_v2"
                        else self.args.frontier_regression_penalty
                    ),
                    pair_bonus=self.args.frontier_pair_bonus,
                    strict_bonus=self.args.frontier_strict_bonus,
                    mean_deficit_weight=self.args.frontier_mean_deficit_weight,
                )
                if self.args.sampler_profile == "universal_frontier_recovery_v2":
                    protected = {
                        "dock_refine": ("qed", "sa", "sim"),
                        "similarity_repair": ("dock", "qed", "sa"),
                        "quality_repair": ("dock", "sim"),
                    }.get(operator, ())
                    targets = {
                        "dock_refine": ("dock",),
                        "similarity_repair": ("sim",),
                        "quality_repair": ("qed", "sa"),
                    }.get(operator, ())
                    protected_regressions = sum(
                        parent["checks"][key] and not item["checks"][key]
                        for key in protected
                    )
                    target_gain = sum(
                        max(
                            0.0,
                            float(parent["deficits"][key])
                            - float(item["deficits"][key]),
                        )
                        for key in targets
                    )
                    if protected_regressions:
                        reward_parts["reward"] -= 0.50 * protected_regressions
                    else:
                        reward_parts["reward"] += 0.75 * target_gain
                if self.args.sampler_profile == "integrated_frontier":
                    novel_admission = (
                        item["smiles"] not in previous_items
                        and item["smiles"] in admitted_smiles
                    )
                    reward = float(
                        rank_improved(
                            item,
                            parent,
                            tolerance=self.args.integrated_rank_tolerance,
                        )
                        or novel_admission
                    )
                else:
                    reward = reward_parts["reward"]
            else:
                if parent_residual is not None:
                    reward += float(parent_residual) - item["residual"]
                if parent_stage is not None:
                    reward += self.args.frontier_crossing_bonus * max(
                        0, item["stage"] - int(parent_stage)
                    )
                if item["strict"]:
                    reward += self.args.frontier_strict_bonus
            if item["strict"]:
                stats["strict"] += 1.0
            reward = clamp(reward, -1.0, 1.0)
            if self.args.sampler_profile in INTEGRATED_FRONTIER_PROFILES:
                root_id = item.get("root_id", "unknown")
                root_credit = self.integrated_root_credit[root_id]
                credit_alpha = self.args.integrated_root_credit_alpha
                credit_reward = reward
                if self.args.sampler_profile in UNIFIED_FRONTIER_PROFILES:
                    credit_reward = 0.5 * (reward + 1.0)
                self.integrated_root_credit[root_id] = (
                    1.0 - credit_alpha
                ) * root_credit + credit_alpha * credit_reward
                self.integrated_root_credit[root_id] = clamp(
                    self.integrated_root_credit[root_id],
                    0.0,
                    1.0,
                )
                meta["root_credit"] = self.integrated_root_credit[root_id]
            stats["evaluated"] += 1.0
            rewards_by_operator[operator].append(reward)
            metadata_by_operator[operator].append(meta)
            if parent is not None:
                transitions_by_operator[operator].append(
                    {
                        "rank_improved": rank_improved(
                            item,
                            parent,
                            tolerance=self.args.integrated_rank_tolerance,
                        ),
                        "admitted": item["smiles"] in admitted_smiles,
                        "strict": bool(item["strict"]),
                        "crossed": reward_parts.get("crossed", 0),
                        "regressed": reward_parts.get("regressed", 0),
                        "pair_gain": reward_parts.get("pair_gain", 0),
                        "residual_gain": float(parent["residual"])
                        - float(item["residual"]),
                    }
                )
            meta.update(
                {
                    "operator_reward": reward,
                    "output_residual": item["residual"],
                    "output_max_deficit": item["max_deficit"],
                    "output_mean_deficit": item["mean_deficit"],
                    "output_bottleneck": item["bottleneck"],
                    "output_stage": item["stage"],
                    "output_strict": item["strict"],
                    "constraints_crossed": reward_parts.get("crossed", 0),
                    "constraints_regressed": reward_parts.get("regressed", 0),
                    "pair_frontiers_gained": reward_parts.get("pair_gain", 0),
                    "output_frontiers": ";".join(
                        frontier_labels(
                            item,
                            self.args.sim_thr,
                            self.args.frontier_similarity_slack,
                        )
                    ),
                }
            )

        alpha = self.args.frontier_reward_alpha
        operators_to_update = set(rewards_by_operator)
        if self.args.sampler_profile in UNIFIED_FRONTIER_PROFILES:
            operators_to_update.update(self.frontier_last_allocation)
        for operator in sorted(operators_to_update):
            rewards = rewards_by_operator.get(operator, [])
            stats = self.frontier_operator_stats[operator]
            if self.args.sampler_profile in UNIFIED_FRONTIER_PROFILES:
                batch_reward, _ = self.frontier_engine.update_constrained_batch(
                    group="proposal",
                    operator=operator,
                    transitions=transitions_by_operator[operator],
                    requested=self.frontier_last_allocation.get(operator),
                )
            elif self.args.sampler_profile == "integrated_frontier":
                batch_reward = sum(rewards) / max(1, len(rewards))
            else:
                batch_reward = upper_tail_reward(
                    rewards,
                    tail_fraction=self.args.frontier_reward_tail_fraction,
                    min_tail=self.args.frontier_reward_tail_min,
                    mean_weight=self.args.frontier_reward_mean_weight,
                )
            batch_reward = clamp(batch_reward, -1.0, 1.0)
            stats["updates"] += 1.0
            stats["last_batch_reward"] = batch_reward
            if self.args.sampler_profile in UNIFIED_FRONTIER_PROFILES:
                stats["reward_ema"] = self.frontier_engine.bandits["proposal"].stats[
                    operator
                ]["ema"]
            else:
                stats["reward_ema"] = (1.0 - alpha) * stats[
                    "reward_ema"
                ] + alpha * batch_reward
            for meta in metadata_by_operator[operator]:
                meta["operator_batch_reward"] = batch_reward

        if self.args.sampler_profile in INTEGRATED_FRONTIER_PROFILES:
            state_pool = (
                self.frontier_archives["strict"] or self.frontier_archives["near"]
            )
            current_best = min(state_pool, key=constraint_rank)
            if rank_improved(
                current_best,
                self.integrated_best_item,
                tolerance=self.args.integrated_rank_tolerance,
            ):
                self.integrated_best_item = current_best
                self.integrated_stagnant_iters = 0
            else:
                self.integrated_stagnant_iters += 1
            lineage_pool = (
                self.frontier_archives["strict"] or self.frontier_archives["near"]
            )
            self.integrated_lineage = lineage_metrics(
                lineage_pool,
                limit=self.args.integrated_lineage_top_k,
            )
            if self.args.sampler_profile in UNIFIED_FRONTIER_PROFILES:
                available_operators = list(self.frontier_base_operators)
                operator_needs = {}
                for operator in self.frontier_operators:
                    archive_name = self.frontier_archive_for_operator(operator)
                    if not self.frontier_archives[archive_name]:
                        continue
                    available_operators.append(operator)
                    operator_needs[operator] = self.frontier_operator_need(operator)
                self.frontier_last_needs = operator_needs
                pair_feasible = any(
                    self.frontier_archives[label] for label in ("sq", "qd", "sd")
                )
                completion_candidates = []
                for label, operator in (
                    ("sq", "dock_refine"),
                    ("qd", "similarity_repair"),
                    ("sd", "quality_repair"),
                ):
                    archive = self.frontier_archives[label]
                    if archive:
                        best_pair = min(archive, key=constraint_rank)
                        completion_candidates.append(
                            (float(best_pair["max_deficit"]), operator)
                        )
                completion_operators = []
                if completion_candidates:
                    best_pair_deficit = min(row[0] for row in completion_candidates)
                    completion_operators = [
                        operator
                        for deficit, operator in completion_candidates
                        if deficit <= best_pair_deficit + 0.03
                    ]
                has_generated_similarity_feasible = any(
                    item["smiles"] != self.start_smiles and item["checks"]["sim"]
                    for item in self.frontier_items.values()
                )
                self.integrated_state = self.frontier_engine.classify(
                    iteration=self.frontier_iteration,
                    has_feasible=bool(self.frontier_archives["strict"]),
                    has_pair_feasible=pair_feasible,
                    best_max_deficit=float(current_best["max_deficit"]),
                    stagnant_iterations=self.integrated_stagnant_iters,
                    largest_root_fraction=self.integrated_lineage[
                        "largest_root_fraction"
                    ],
                    available_operators=available_operators,
                    constraint_needs=operator_needs,
                    completion_operators=completion_operators,
                    has_generated_similarity_feasible=(
                        has_generated_similarity_feasible
                    ),
                    similarity_threshold=self.args.sim_thr,
                )
            else:
                self.integrated_state = classify_frontier_state(
                    iteration=self.frontier_iteration,
                    warmup_iterations=self.args.integrated_warmup_iterations,
                    has_feasible=bool(self.frontier_archives["strict"]),
                    stagnant_iterations=self.integrated_stagnant_iters,
                    plateau_patience=self.args.integrated_plateau_patience,
                    largest_root_fraction=self.integrated_lineage[
                        "largest_root_fraction"
                    ],
                    collapse_threshold=self.args.integrated_collapse_threshold,
                )
        elif self.args.sampler_profile == "safe_frontier_final":
            generated_items = [
                item
                for smiles, item in self.frontier_items.items()
                if smiles != self.start_smiles
            ]
            pair_presence = {
                label: bool(self.frontier_archives[label])
                for label in ("sq", "qd", "sd")
            }
            self.integrated_state = self.protected_frontier_engine.classify(
                iteration=self.frontier_iteration,
                official_feasible=self.safe_final_official_feasible,
                has_generated_similarity=any(
                    item["checks"]["sim"] for item in generated_items
                ),
                has_stage_three=any(item["stage"] >= 3 for item in generated_items),
                pair_presence=pair_presence,
            )
        elif self.args.sampler_profile == "universal_frontier_bridge":
            pair_presence = {
                label: bool(self.frontier_archives[label])
                for label in ("sq", "qd", "sd")
            }
            self.integrated_state = self.protected_frontier_engine.classify(
                iteration=self.frontier_iteration,
                official_feasible=self.safe_final_official_feasible,
                pair_presence=pair_presence,
            )
        elif self.args.sampler_profile in {
            "lead_protected_completion",
            "lead_protected_completion_v2",
            "lead_protected_completion_v3",
            "lead_protected_completion_v4",
            "lead_protected_completion_v5",
        }:
            pair_presence = {
                label: bool(self.frontier_archives[label])
                for label in ("sq", "qd", "sd")
            }
            boundary_candidates = []
            if self.frontier_archives["qd"]:
                boundary_candidates.append(
                    (
                        min(
                            item["deficits"]["sim"]
                            for item in self.frontier_archives["qd"]
                        ),
                        "similarity",
                    )
                )
            if self.frontier_archives["sd"]:
                boundary_candidates.append(
                    (
                        min(
                            max(item["deficits"]["qed"], item["deficits"]["sa"])
                            for item in self.frontier_archives["sd"]
                        ),
                        "quality",
                    )
                )
            boundary_deficit, boundary_mode = (
                min(boundary_candidates) if boundary_candidates else (None, None)
            )
            route_deficits = {}
            if self.frontier_archives["sq"]:
                route_deficits["dock"] = min(
                    item["deficits"]["dock"] for item in self.frontier_archives["sq"]
                )
            if self.frontier_archives["qd"]:
                route_deficits["similarity"] = min(
                    item["deficits"]["sim"] for item in self.frontier_archives["qd"]
                )
            if self.frontier_archives["sd"]:
                route_deficits["quality"] = min(
                    max(item["deficits"]["qed"], item["deficits"]["sa"])
                    for item in self.frontier_archives["sd"]
                )
            self.integrated_state = self.protected_frontier_engine.classify(
                iteration=self.frontier_iteration,
                official_feasible=self.safe_final_official_feasible,
                pair_presence=pair_presence,
                has_loose_feasible=bool(self.frontier_archives["sq"]),
                boundary_mode=boundary_mode,
                boundary_deficit=boundary_deficit,
                route_deficits=route_deficits,
            )
        elif self.args.sampler_profile in {
            "universal_frontier_recovery_v2",
            "universal_frontier_recovery_v3",
        }:
            generated_items = [
                item
                for smiles, item in self.frontier_items.items()
                if smiles != self.start_smiles
            ]
            pair_frontier_count = sum(
                bool(self.frontier_archives[label]) for label in ("sq", "qd", "sd")
            )
            self.integrated_state = recovery_v2_state(
                iteration=self.frontier_iteration,
                warmup_iterations=self.args.frontier_legacy_warmup_iters,
                has_strict=bool(self.frontier_archives["strict"]),
                has_stage_three=any(item["stage"] >= 3 for item in generated_items),
                has_generated_similarity=any(
                    item["checks"]["sim"] for item in generated_items
                ),
                pair_frontier_count=pair_frontier_count,
            )

        archive_counts = "/".join(
            f"{label}:{len(self.frontier_archives[label])}"
            for label in ("near", "s", "sq", "qd", "sd", "strict")
        )
        arm_scores = "/".join(
            f"{name}:{self.frontier_operator_stats[name]['reward_ema']:.3f}"
            for name in self.frontier_operators
        )
        needs = "/".join(
            f"{name}:{value:.3f}" for name, value in self.frontier_last_needs.items()
        )
        integrated_state = getattr(self, "integrated_state", "n/a")
        integrated_lineage = getattr(
            self,
            "integrated_lineage",
            {"root_count": 0, "largest_root_fraction": 0.0},
        )
        self.last_frontier_summary = (
            f"archives={archive_counts} arm_ema={arm_scores} needs={needs or 'n/a'} "
            f"state={integrated_state} "
            f"roots={integrated_lineage['root_count']} "
            f"root_max={integrated_lineage['largest_root_fraction']:.2f} "
            f"cache={getattr(self, 'last_cache_hits', 0)}/"
            f"{getattr(self, 'last_cache_misses', 0)}"
        )

    def frontier_operator_params(self, operator):
        if operator in {"no_pair_anchor_restart", "completion_seed_restart"}:
            return (
                self.args.completion_v4_anchor_remask,
                self.args.completion_v4_anchor_temperature,
                self.args.completion_v4_anchor_span_prob,
                1.0,
            )
        if operator in {"feasible_dock_polish", "completion_dock_refine"}:
            return (
                self.args.completion_dock_remask,
                self.args.completion_dock_temperature,
                self.args.completion_dock_span_prob,
                1.0,
            )
        if operator in {
            "boundary_similarity_polish",
            "completion_similarity_repair",
        }:
            return (
                self.args.completion_similarity_remask,
                self.args.completion_similarity_temperature,
                self.args.completion_similarity_span_prob,
                1.0,
            )
        if operator in {"boundary_quality_polish", "completion_quality_repair"}:
            return (
                self.args.completion_quality_remask,
                self.args.completion_quality_temperature,
                self.args.completion_quality_span_prob,
                1.0,
            )
        if operator == "pair_bridge":
            return (
                self.args.frontier_bridge_remask,
                self.args.frontier_bridge_temperature,
                self.args.frontier_bridge_span_prob,
                1.0,
            )
        if (
            self.args.sampler_profile == "safe_frontier_final"
            and self.frontier_recovery_active()
        ):
            if self.integrated_state == "seed_anchor":
                if operator == "start_repair":
                    return (0.06, 0.92, 0.88, 1.00)
                if operator == "joint_repair":
                    return (0.08, 0.96, 0.84, 0.95)
            if self.integrated_state == "completion":
                completion_params = {
                    "dock_refine": (0.06, 0.92, 0.84, 0.95),
                    "similarity_repair": (0.045, 0.88, 0.88, 1.00),
                    "quality_repair": (0.09, 0.96, 0.82, 0.95),
                    "joint_repair": (0.08, 0.96, 0.82, 0.90),
                }
                if operator in completion_params:
                    return completion_params[operator]
            if self.integrated_state in {"bridge", "explore"}:
                if operator == "joint_repair":
                    return (0.10, 0.99, 0.80, 0.90)
                if operator == "start_repair":
                    return (0.08, 0.96, 0.84, 0.95)
        if self.args.sampler_profile == "universal_frontier_recovery_v2":
            state = self.integrated_state
            if state == "seed_anchor":
                if operator == "start_repair":
                    return (0.03, 0.88, 0.90, 1.00)
                if operator == "joint_repair":
                    return (0.05, 0.94, 0.86, 1.00)
            if state in {"complete", "refine"}:
                completion_params = {
                    "dock_refine": (0.05, 0.90, 0.86, 0.95),
                    "similarity_repair": (0.04, 0.86, 0.88, 1.00),
                    "quality_repair": (0.07, 0.94, 0.86, 0.95),
                    "joint_repair": (0.07, 0.94, 0.84, 0.90),
                }
                if operator in completion_params:
                    return completion_params[operator]
            if state == "bridge" and operator == "joint_repair":
                return (0.08, 0.98, 0.84, 0.90)
        if (
            self.args.sampler_profile == "universal_frontier_recovery_v3"
            and self.frontier_recovery_active()
        ):
            state = self.integrated_state
            if state == "seed_anchor":
                if operator == "start_repair":
                    return (0.08, 0.94, 0.84, 0.95)
                if operator == "joint_repair":
                    return (0.10, 0.98, 0.80, 0.90)
            if state == "complete":
                completion_params = {
                    "dock_refine": (0.09, 0.94, 0.82, 0.80),
                    "similarity_repair": (0.06, 0.90, 0.86, 0.95),
                    "quality_repair": (0.12, 0.98, 0.78, 0.85),
                    "joint_repair": (0.10, 0.98, 0.78, 0.85),
                }
                if operator in completion_params:
                    return completion_params[operator]
            if state in {"bridge", "explore"} and operator == "joint_repair":
                return (0.11, 1.00, 0.76, 0.85)
        if operator == "scaffold_rescue":
            return (0.06, 0.90, 0.72, 1.00)
        if operator == "dock_refine":
            return (
                self.args.frontier_dock_remask,
                self.args.frontier_dock_temperature,
                self.args.frontier_dock_span_prob,
                0.65,
            )
        if operator == "similarity_repair":
            return (
                self.args.frontier_similarity_remask,
                self.args.frontier_similarity_temperature,
                self.args.frontier_similarity_span_prob,
                0.90,
            )
        if operator == "quality_repair":
            return (
                self.args.frontier_quality_remask,
                self.args.frontier_quality_temperature,
                self.args.frontier_quality_span_prob,
                0.85,
            )
        if operator == "joint_repair":
            return (
                self.args.frontier_joint_remask,
                self.args.frontier_joint_temperature,
                self.args.frontier_joint_span_prob,
                0.85,
            )
        if operator == "lineage_restart":
            return (
                self.args.integrated_lineage_remask,
                self.args.integrated_lineage_temperature,
                self.args.integrated_lineage_span_prob,
                0.70,
            )
        return (
            self.args.frontier_start_remask,
            self.args.frontier_start_temperature,
            self.args.frontier_start_span_prob,
            0.90,
        )

    @staticmethod
    def frontier_local_sampler_profile(operator):
        """Expose trajectory refinement only to quality-seeking operators."""
        base_profile = resolve_local_sampler_profile()
        if base_profile not in {"task_adaptive_local", "task_adaptive_refine"}:
            return base_profile
        if operator in {
            "quality_repair",
            "joint_repair",
            "boundary_quality_polish",
            "completion_quality_repair",
        }:
            return "task_adaptive_refine"
        return "task_adaptive_local"

    def frontier_archive_for_operator(self, operator):
        return {
            "start_repair": "s",
            "dock_refine": "sq",
            "similarity_repair": "qd",
            "quality_repair": "sd",
            "joint_repair": "near",
            "lineage_restart": "near",
            "scaffold_rescue": "s",
            "pair_bridge": "near",
            "feasible_dock_polish": "sq",
            "boundary_similarity_polish": "qd",
            "boundary_quality_polish": "sd",
            "no_pair_anchor_restart": "s",
            "completion_dock_refine": "sq",
            "completion_similarity_repair": "qd",
            "completion_quality_repair": "sd",
            "completion_seed_restart": "s",
        }[operator]

    def frontier_operator_need(self, operator):
        archive = self.frontier_archives[self.frontier_archive_for_operator(operator)]
        keys = {
            "start_repair": ("dock", "qed", "sa", "sim"),
            "dock_refine": ("dock",),
            "similarity_repair": ("sim",),
            "quality_repair": ("qed", "sa"),
            "joint_repair": ("dock", "qed", "sa", "sim"),
            "lineage_restart": ("dock", "qed", "sa", "sim"),
            "scaffold_rescue": ("sim",),
            "pair_bridge": ("dock", "qed", "sa", "sim"),
            "feasible_dock_polish": ("dock",),
            "boundary_similarity_polish": ("sim",),
            "boundary_quality_polish": ("qed", "sa"),
            "no_pair_anchor_restart": ("dock", "qed", "sa", "sim"),
            "completion_dock_refine": ("dock",),
            "completion_similarity_repair": ("sim",),
            "completion_quality_repair": ("qed", "sa"),
            "completion_seed_restart": ("dock", "qed", "sa", "sim"),
        }[operator]
        need = archive_constraint_need(
            archive,
            keys,
            joint=operator in {"joint_repair", "lineage_restart", "pair_bridge"},
            top_k=self.args.frontier_need_top_k,
            readiness_weight=self.args.frontier_readiness_weight,
            residual_scale=self.args.frontier_readiness_scale,
        )
        if operator == "start_repair":
            need *= self.args.frontier_start_need_scale
        return need

    def frontier_choose_bridge_parents(self):
        """Choose complementary pair-frontier parents without task identities."""
        combinations = []
        for anchor_label, donor_label in (
            ("sq", "qd"),
            ("sq", "sd"),
            ("sd", "qd"),
        ):
            anchors = self.frontier_archives[anchor_label][
                : max(1, self.args.frontier_bridge_parent_top_k)
            ]
            donors = self.frontier_archives[donor_label][
                : max(1, self.args.frontier_bridge_parent_top_k)
            ]
            if anchors and donors:
                combinations.append((anchor_label, donor_label, anchors, donors))
        if not combinations:
            return None

        for _ in range(40):
            anchor_label, donor_label, anchors, donors = self.frontier_rng.choice(
                combinations
            )
            anchor_weights = [
                1.0 / math.sqrt(rank + 1.0) for rank in range(len(anchors))
            ]
            donor_weights = [1.0 / math.sqrt(rank + 1.0) for rank in range(len(donors))]
            anchor = self.frontier_rng.choices(anchors, weights=anchor_weights, k=1)[0]
            donor = self.frontier_rng.choices(donors, weights=donor_weights, k=1)[0]
            if anchor["smiles"] != donor["smiles"]:
                return anchor_label, donor_label, anchor, donor
        return None

    def frontier_make_pair_bridge_proposal(self):
        """Recombine a similarity-preserving core with a complementary R group."""
        selected = self.frontier_choose_bridge_parents()
        if selected is None:
            return None
        anchor_label, donor_label, anchor, donor = selected
        anchor_fragments = [
            fragment
            for fragment in set(local_genmol_cut(anchor["smiles"]))
            if Chem.MolFromSmiles(fragment) is not None
        ]
        donor_fragments = [
            fragment
            for fragment in set(local_genmol_cut(donor["smiles"]))
            if Chem.MolFromSmiles(fragment) is not None
        ]
        if not anchor_fragments or not donor_fragments:
            return None

        anchor_fragments.sort(key=fragment_heavy_atom_count, reverse=True)
        anchor_fragments = anchor_fragments[: min(4, len(anchor_fragments))]
        donor_limit = max(3, min(12, atom_count(anchor["smiles"]) // 3))
        donor_fragments = [
            fragment
            for fragment in donor_fragments
            if 1 <= fragment_heavy_atom_count(fragment) <= donor_limit
        ]
        if not donor_fragments:
            return None
        donor_weights = [
            math.exp(-abs(fragment_heavy_atom_count(fragment) - 4) / 4.0)
            for fragment in donor_fragments
        ]

        bridge_seed = None
        for _ in range(60):
            core = self.frontier_rng.choice(anchor_fragments)
            substituent = self.frontier_rng.choices(
                donor_fragments, weights=donor_weights, k=1
            )[0]
            try:
                candidate = attach_fragments(core, substituent)
            except Exception:
                continue
            candidate = canonical_smiles(candidate) if candidate else None
            if candidate is None or candidate == anchor["smiles"]:
                continue
            atoms = atom_count(candidate)
            if not self.args.min_atoms <= atoms <= self.args.max_atoms:
                continue
            if not tokenizable(candidate, self.tk, self.args.max_len):
                continue
            mol = Chem.MolFromSmiles(candidate)
            if mol is None:
                continue
            similarity = self.reward_sim([mol])[0]
            similarity_floor = max(
                0.0,
                self.args.sim_thr - self.args.frontier_similarity_slack,
            )
            if similarity < similarity_floor:
                continue
            bridge_seed = candidate
            break
        if bridge_seed is None:
            return None

        plan = adaptive_peripheral_edit_plan(
            bridge_seed,
            self.frontier_rng,
            delta=0,
            target_atom_fraction=0.08,
            max_atom_fraction=0.28,
            max_span_tokens=6,
        )
        if plan is None:
            plan = atom_span_edit_plan(
                bridge_seed,
                self.frontier_rng,
                delta=0,
                span_tokens=2,
            )
        if plan is None:
            return None
        parent_tokens = tokenize_smiles(bridge_seed)
        remask, temperature, span_prob, _ = self.frontier_operator_params("pair_bridge")
        return {
            "seed": bridge_seed,
            "plan": plan,
            "operator": "pair_bridge",
            "source_frontier": f"{anchor_label}+{donor_label}",
            "parent_smiles": anchor["smiles"],
            "donor_smiles": donor["smiles"],
            "bridge_pair": f"{anchor_label}+{donor_label}",
            "parent_residual": anchor["residual"],
            "parent_stage": anchor["stage"],
            "parent_token_len": len(parent_tokens),
            "planned_delta": int(plan.get("delta", 0)),
            "peripheral": bool(plan.get("peripheral", False)),
            "remask": remask,
            "temperature": temperature,
            "span_prob": span_prob,
            "root_id": anchor.get("root_id", "start"),
            "depth": int(anchor.get("depth", 0)) + 1,
            "search_state": self.integrated_state,
        }

    def frontier_available_operator_scores(self):
        total_updates = sum(
            stats.get("updates", 0.0) for stats in self.frontier_operator_stats.values()
        )
        scores = {}
        self.frontier_last_needs = {}
        integrated_priors = (
            integrated_operator_weights(self.integrated_state)
            if self.args.sampler_profile == "integrated_frontier"
            else None
        )
        for operator in self.frontier_operators:
            archive_name = self.frontier_archive_for_operator(operator)
            if not self.frontier_archives[archive_name]:
                continue
            if (
                integrated_priors is not None
                and integrated_priors.get(operator, 0.0) <= 0.0
            ):
                continue
            stats = self.frontier_operator_stats[operator]
            acceptance = (stats["accepted"] + 1.0) / (stats["attempted"] + 2.0)
            ucb = math.sqrt(
                math.log(total_updates + 2.0) / (stats.get("updates", 0.0) + 1.0)
            )
            reward_center = stats["reward_ema"]
            if integrated_priors is not None:
                reward_center = reward_center if stats.get("updates", 0.0) > 0 else 0.5
                reward_center -= 0.5
            learned = math.exp(
                clamp(self.args.frontier_bandit_eta * reward_center, -2.0, 2.0)
            )
            score = (
                learned * (0.5 + 0.5 * acceptance) + self.args.frontier_bandit_ucb * ucb
            )
            if integrated_priors is not None:
                prior = integrated_priors[operator]
                self.frontier_last_needs[operator] = prior
                score *= prior
            if self.args.sampler_profile in UNIVERSAL_FRONTIER_PROFILES:
                need = self.frontier_operator_need(operator)
                self.frontier_last_needs[operator] = need
                score *= 1.0 + self.args.frontier_need_weight * need
            scores[operator] = score
        if self.frontier_recovery_active():
            archive_presence = {
                label: self.frontier_archives[label] for label in ("sq", "qd", "sd")
            }
            if self.args.sampler_profile == "safe_frontier_final":
                head_context = {
                    **self.protected_frontier_engine.last_context,
                    "pair_presence": archive_presence,
                }
                priors = self.protected_frontier_engine.task_head.operator_priors(
                    "proposal",
                    self.integrated_state,
                    head_context,
                )
                scores = {
                    operator: score * max(0.0, float(priors.get(operator, 0.0)))
                    for operator, score in scores.items()
                    if float(priors.get(operator, 0.0)) > 0.0
                }
                return scores
            if self.args.sampler_profile in {
                "universal_frontier_recovery_v2",
                "universal_frontier_recovery_v3",
            }:
                multipliers = recovery_v2_operator_multipliers(
                    self.integrated_state,
                    archive_presence,
                )
                if self.args.sampler_profile == "universal_frontier_recovery_v3":
                    multipliers = {
                        operator: 1.0 + 0.35 * (value - 1.0)
                        for operator, value in multipliers.items()
                    }
            else:
                multipliers = completion_recovery_multipliers(
                    archive_presence,
                    completion_boost=self.args.frontier_recovery_completion_boost,
                    start_boost=self.args.frontier_recovery_start_boost,
                    joint_boost=self.args.frontier_recovery_joint_boost,
                )
            scores = {
                operator: score * multipliers.get(operator, 1.0)
                for operator, score in scores.items()
            }
        return scores

    def frontier_recovery_active(self):
        if self.args.sampler_profile == "safe_frontier_final":
            return self.integrated_state not in {"baseline", "locked"}
        if self.args.sampler_profile == "universal_frontier_bridge":
            return self.integrated_state == "bridge"
        if self.args.sampler_profile in {
            "universal_frontier_recovery_v2",
            "universal_frontier_recovery_v3",
        }:
            return (
                self.frontier_iteration > self.args.frontier_recovery_start_iter
                and self.integrated_state
                in {"seed_anchor", "explore", "bridge", "complete"}
                and not self.frontier_archives["strict"]
            )
        return (
            self.args.sampler_profile == "universal_frontier_recovery"
            and self.frontier_iteration > self.args.frontier_recovery_start_iter
            and not self.frontier_archives["strict"]
        )

    def frontier_choose_robust_completion_parent(self, operator):
        archive = self.frontier_archives[self.frontier_archive_for_operator(operator)]
        selection_index = self.completion_parent_counters[operator]
        parent, selected_rank = choose_robust_completion_parent(
            archive,
            operator,
            selection_index,
            top_k=self.args.completion_parent_top_k,
            minimum_constraint_slack=self.args.completion_parent_min_slack,
        )
        self.completion_parent_counters[operator] += 1
        if parent is None:
            return None

        parent = dict(parent)
        protected_keys = {
            "feasible_dock_polish": ("qed", "sa", "sim"),
            "boundary_similarity_polish": ("dock", "qed", "sa"),
            "boundary_quality_polish": ("dock", "sim"),
            "completion_dock_refine": ("qed", "sa", "sim"),
            "completion_similarity_repair": ("dock", "qed", "sa"),
            "completion_quality_repair": ("dock", "sim"),
        }[operator]
        parent["_completion_parent_rank"] = int(selected_rank)
        parent["_completion_constraint_slack"] = min(
            float(parent["normalized"].get(key, 0.0)) - 1.0 for key in protected_keys
        )
        return parent

    def frontier_completion_span_tokens(self, operator, parent):
        """Use one-token edits near a boundary and widen only with slack."""
        if operator in {
            "boundary_similarity_polish",
            "boundary_quality_polish",
            "completion_similarity_repair",
            "completion_quality_repair",
        }:
            return 1

        slack = float(parent.get("_completion_constraint_slack", 0.0))
        if slack < 0.04:
            schedule = (1,)
        elif slack < 0.12:
            schedule = (1, 1, 2)
        elif slack < 0.25:
            schedule = (1, 2, 2)
        else:
            schedule = (1, 2, 3)
        index = self.completion_span_counters[operator]
        self.completion_span_counters[operator] += 1
        return min(
            schedule[index % len(schedule)],
            max(1, int(self.args.completion_v2_max_span_tokens)),
        )

    def frontier_choose_parent(self, operator):
        if operator in {"no_pair_anchor_restart", "completion_seed_restart"}:
            return self.frontier_items[self.start_smiles]
        if self.args.sampler_profile == "lead_protected_completion_v2" and operator in {
            "feasible_dock_polish",
            "boundary_similarity_polish",
            "boundary_quality_polish",
        }:
            parent = self.frontier_choose_robust_completion_parent(operator)
            if parent is not None:
                return parent
        if self.args.sampler_profile == "lead_protected_completion_v5" and operator in {
            "completion_dock_refine",
            "completion_similarity_repair",
            "completion_quality_repair",
        }:
            parent = self.frontier_choose_robust_completion_parent(operator)
            if parent is not None:
                return parent
        archive = self.frontier_archives[self.frontier_archive_for_operator(operator)]
        if operator == "scaffold_rescue":
            return self.frontier_items[self.start_smiles]
        if (
            self.args.sampler_profile == "safe_frontier_final"
            and self.frontier_recovery_active()
            and self.integrated_state == "seed_anchor"
            and operator in {"start_repair", "joint_repair", "similarity_repair"}
        ):
            return self.frontier_items[self.start_smiles]
        if (
            self.args.sampler_profile == "safe_frontier_final"
            and self.frontier_recovery_active()
            and self.integrated_state == "bridge"
            and operator == "joint_repair"
        ):
            complementary = []
            for label in ("sq", "qd", "sd"):
                complementary.extend(
                    self.frontier_archives[label][
                        : max(1, self.args.frontier_parent_top_k // 3)
                    ]
                )
            if complementary:
                deduplicated = {item["smiles"]: item for item in complementary}
                archive = sorted(deduplicated.values(), key=constraint_rank)
        if (
            (
                self.args.sampler_profile == "universal_frontier_recovery_v2"
                or (
                    self.args.sampler_profile == "universal_frontier_recovery_v3"
                    and self.frontier_recovery_active()
                )
            )
            and self.integrated_state == "seed_anchor"
            and operator in {"start_repair", "joint_repair"}
        ):
            return self.frontier_items[self.start_smiles]
        if (
            (
                self.args.sampler_profile == "universal_frontier_recovery_v2"
                or (
                    self.args.sampler_profile == "universal_frontier_recovery_v3"
                    and self.frontier_recovery_active()
                )
            )
            and self.integrated_state == "bridge"
            and operator == "joint_repair"
        ):
            complementary = []
            for label in ("sq", "qd", "sd"):
                complementary.extend(
                    self.frontier_archives[label][
                        : max(1, self.args.frontier_parent_top_k // 3)
                    ]
                )
            if complementary:
                deduplicated = {item["smiles"]: item for item in complementary}
                archive = sorted(deduplicated.values(), key=constraint_rank)
        if (
            operator == "start_repair"
            and self.frontier_rng.random() < self.args.frontier_start_parent_prob
        ):
            return self.frontier_items[self.start_smiles]
        top = archive[: max(1, self.args.frontier_parent_top_k)]
        root_counts = defaultdict(int)
        if self.args.sampler_profile in INTEGRATED_FRONTIER_PROFILES:
            for item in top:
                root_counts[item.get("root_id", "unknown")] += 1
        weights = []
        for rank, item in enumerate(top):
            weight = 1.0 / math.sqrt(rank + 1.0)
            if root_counts:
                root_id = item.get("root_id", "unknown")
                weight *= 0.5 + self.integrated_root_credit[root_id]
                weight /= math.sqrt(root_counts[root_id])
            weights.append(weight)
        return self.frontier_rng.choices(top, weights=weights, k=1)[0]

    def frontier_choose_length_delta(self, operator):
        if operator in {
            "pair_bridge",
            "feasible_dock_polish",
            "boundary_similarity_polish",
            "boundary_quality_polish",
            "no_pair_anchor_restart",
            "completion_dock_refine",
            "completion_similarity_repair",
            "completion_quality_repair",
            "completion_seed_restart",
        }:
            return 0
        probability = self.args.frontier_length_prob
        if (
            self.args.sampler_profile == "safe_frontier_final"
            and self.frontier_recovery_active()
        ):
            if self.integrated_state == "completion" and operator in {
                "dock_refine",
                "similarity_repair",
                "quality_repair",
            }:
                return 0
            probability *= {
                "seed_anchor": 0.35,
                "completion": 0.20,
                "bridge": 0.50,
                "explore": 0.75,
            }.get(self.integrated_state, 1.0)
        if self.args.sampler_profile == "universal_frontier_recovery_v2" or (
            self.args.sampler_profile == "universal_frontier_recovery_v3"
            and self.frontier_recovery_active()
        ):
            if self.integrated_state in {"complete", "refine"} and operator in {
                "dock_refine",
                "similarity_repair",
                "quality_repair",
            }:
                return 0
            probability *= {
                "seed_anchor": 0.35,
                "complete": 0.20,
                "refine": 0.10,
                "bridge": 0.50,
                "explore": 0.75,
                "warmup": 1.00,
            }.get(self.integrated_state, 1.0)
        if self.args.sampler_profile in INTEGRATED_FRONTIER_PROFILES:
            probability *= {
                "start_repair": 0.35,
                "dock_refine": 0.35,
                "similarity_repair": 0.0,
                "quality_repair": 0.65,
                "joint_repair": 0.75,
                "lineage_restart": 1.00,
                "scaffold_rescue": 0.00,
            }.get(operator, 1.0)
            if self.args.sampler_profile in UNIFIED_FRONTIER_PROFILES:
                probability *= self.frontier_engine.adapter.length_edit_scale(
                    self.integrated_state,
                    self.frontier_engine.last_context,
                )
        elif self.args.sampler_profile in UNIVERSAL_FRONTIER_PROFILES:
            probability *= {
                "start_repair": 0.40,
                "dock_refine": 1.00,
                "similarity_repair": 1.00,
                "quality_repair": 0.25,
                "joint_repair": 0.80,
            }.get(operator, 1.0)
        elif operator == "similarity_repair":
            probability *= 0.5
        probability = clamp(probability, 0.0, 1.0)
        if self.frontier_rng.random() >= probability:
            return 0
        deltas = [
            int(value)
            for value in self.args.frontier_length_deltas.replace(",", ":").split(":")
            if value.strip()
        ]
        deltas = [value for value in deltas if value != 0]
        if (
            (
                self.args.sampler_profile == "safe_frontier_final"
                and self.frontier_recovery_active()
            )
            or (
                self.args.sampler_profile == "universal_frontier_recovery_v2"
                or (
                    self.args.sampler_profile == "universal_frontier_recovery_v3"
                    and self.frontier_recovery_active()
                )
            )
            and self.integrated_state
            in {"seed_anchor", "bridge", "completion", "complete", "refine"}
        ):
            deltas = [value for value in deltas if abs(value) == 1]
        if (
            self.args.sampler_profile == "unified_frontier_v2_1"
            and self.integrated_state == "bridge"
        ):
            deltas = [value for value in deltas if abs(value) == 1]
        return self.frontier_rng.choice(deltas) if deltas else 0

    def frontier_make_guided_proposal(self, operator):
        if operator == "pair_bridge":
            return self.frontier_make_pair_bridge_proposal()
        v5_route = operator in {
            "completion_dock_refine",
            "completion_similarity_repair",
            "completion_quality_repair",
        }
        v5_seed_restart = operator == "completion_seed_restart"
        if v5_route or v5_seed_restart:
            proposal_rng = self.completion_v5_rng
            rng_stream = "completion_v5"
        elif operator == "no_pair_anchor_restart":
            proposal_rng = self.completion_v4_rng
            rng_stream = "completion_v4"
        else:
            proposal_rng = self.frontier_rng
            rng_stream = "frontier"
        parent = self.frontier_choose_parent(operator)
        remask, temperature, span_prob, peripheral_probability = (
            self.frontier_operator_params(operator)
        )
        if self.args.sampler_profile in INTEGRATED_FRONTIER_PROFILES:
            target_keys = {
                "start_repair": ("dock", "qed", "sa", "sim"),
                "dock_refine": ("dock",),
                "similarity_repair": ("sim",),
                "quality_repair": ("qed", "sa"),
                "joint_repair": ("dock", "qed", "sa", "sim"),
                "lineage_restart": ("dock", "qed", "sa", "sim"),
                "scaffold_rescue": ("sim",),
            }[operator]
            trust_min = self.args.integrated_trust_min
            trust_max = self.args.integrated_trust_max
            if self.args.sampler_profile in UNIFIED_FRONTIER_PROFILES:
                trust_min, trust_max = self.frontier_engine.adapter.trust_region_bounds(
                    self.integrated_state,
                    trust_min,
                    trust_max,
                )
            remask = trust_region_fraction(
                parent,
                target_keys=target_keys,
                base_fraction=remask,
                min_fraction=trust_min,
                max_fraction=trust_max,
                deficit_scale=self.args.integrated_trust_deficit_scale,
                slack_scale=self.args.integrated_trust_slack_scale,
            )
        # Length is no longer selected by a hand-written delta. A bounded
        # subset of these plans is upgraded to learned insertion immediately
        # before sampling; the remainder stays equal-length.
        delta = 0
        parent_tokens = tokenize_smiles(parent["smiles"])
        minimum_span_tokens = 2
        protected_micro = operator in {
            "feasible_dock_polish",
            "boundary_similarity_polish",
            "boundary_quality_polish",
            "completion_dock_refine",
            "completion_similarity_repair",
            "completion_quality_repair",
        }
        anchor_restart = operator in {
            "no_pair_anchor_restart",
            "completion_seed_restart",
        }
        if protected_micro or anchor_restart:
            minimum_span_tokens = 1
        if (
            (
                self.args.sampler_profile == "safe_frontier_final"
                and self.frontier_recovery_active()
            )
            or (
                self.args.sampler_profile == "universal_frontier_recovery_v2"
                or (
                    self.args.sampler_profile == "universal_frontier_recovery_v3"
                    and self.frontier_recovery_active()
                )
            )
            and self.integrated_state
            in {"seed_anchor", "completion", "complete", "refine"}
        ):
            minimum_span_tokens = 1
        max_span_tokens = max(
            minimum_span_tokens,
            int(round(len(parent_tokens) * remask)),
        )
        if protected_micro:
            max_span_tokens = min(
                max_span_tokens,
                max(1, self.args.completion_max_span_tokens),
            )
        if anchor_restart:
            max_span_tokens = min(
                max_span_tokens,
                max(1, self.args.completion_v4_anchor_max_span_tokens),
            )
        if (
            self.args.sampler_profile
            in {"lead_protected_completion_v2", "lead_protected_completion_v5"}
            and protected_micro
        ):
            max_span_tokens = self.frontier_completion_span_tokens(
                operator,
                parent,
            )
        plan = None
        edit_strategy = "peripheral"
        if (
            self.args.sampler_profile == "lead_protected_completion_v2"
            and operator == "boundary_similarity_polish"
        ):
            plan = seed_directed_atom_edit_plan(
                parent["smiles"],
                self.start_smiles,
                self.frontier_rng,
            )
            if plan is not None:
                edit_strategy = "seed_directed_atom"
        if plan is None and proposal_rng.random() < peripheral_probability:
            if anchor_restart:
                plan = peripheral_edit_plan(
                    parent["smiles"],
                    proposal_rng,
                    delta=0,
                    max_atom_fraction=self.args.completion_v4_anchor_max_atom_fraction,
                    max_span_tokens=max_span_tokens,
                )
                edit_strategy = "anchor_peripheral"
            elif self.args.sampler_profile in INTEGRATED_FRONTIER_PROFILES:
                plan = adaptive_peripheral_edit_plan(
                    parent["smiles"],
                    proposal_rng,
                    delta=delta,
                    target_atom_fraction=remask,
                    max_atom_fraction=self.args.integrated_max_component_fraction,
                    max_span_tokens=max_span_tokens,
                )
            else:
                plan = peripheral_edit_plan(
                    parent["smiles"],
                    proposal_rng,
                    delta=delta,
                    max_atom_fraction=self.args.frontier_max_peripheral_fraction,
                    max_span_tokens=max_span_tokens,
                )
        if (
            self.args.sampler_profile == "lead_protected_completion_v5"
            and (v5_route or v5_seed_restart)
            and plan is None
        ):
            plan = atom_span_edit_plan(
                parent["smiles"],
                proposal_rng,
                delta=0,
                span_tokens=max_span_tokens,
            )
            edit_strategy = "fixed_atom_span"
        if (protected_micro or anchor_restart) and plan is None:
            return None
        if plan is None and operator == "scaffold_rescue":
            return None
        if plan is None:
            plan = atom_span_edit_plan(
                parent["smiles"],
                proposal_rng,
                delta=delta,
                span_tokens=max_span_tokens,
            )
            edit_strategy = "atom_span"
        if plan is None:
            return None
        plan["delta"] = 0
        return {
            "seed": parent["smiles"],
            "plan": plan,
            "operator": operator,
            "source_frontier": self.frontier_archive_for_operator(operator),
            "parent_smiles": parent["smiles"],
            "parent_residual": parent["residual"],
            "parent_stage": parent["stage"],
            "parent_token_len": len(parent_tokens),
            "planned_delta": int(plan["delta"]),
            "peripheral": bool(plan.get("peripheral", False)),
            "edit_strategy": edit_strategy,
            "completion_parent_rank": parent.get("_completion_parent_rank"),
            "completion_constraint_slack": parent.get("_completion_constraint_slack"),
            "remask": remask,
            "temperature": temperature,
            "span_prob": span_prob,
            "root_id": parent.get("root_id", "start"),
            "depth": int(parent.get("depth", 0)) + 1,
            "search_state": self.integrated_state,
            "rng_stream": rng_stream,
        }

    def frontier_generate_operator(self, operator, target_n, seen):
        if target_n <= 0:
            return [], []
        out = []
        out_meta = []
        rounds = 0
        while len(out) < target_n and rounds < self.args.max_generation_rounds:
            rounds += 1
            remaining = target_n - len(out)
            requested = max(
                remaining,
                int(math.ceil(remaining * self.args.frontier_overgenerate_factor)),
            )
            proposals = []
            attempts = 0
            while len(proposals) < requested and attempts < requested * 100:
                attempts += 1
                if operator in self.frontier_base_operators:
                    if (
                        self.args.sampler_profile
                        in {
                            "lead_best_union",
                            "universal_frontier_recovery_v2",
                            "universal_frontier_recovery_v3",
                        }
                        and operator == "legacy"
                    ):
                        seed = self.make_fragment_seed()
                    elif (
                        self.args.sampler_profile
                        in {
                            "universal_frontier_recovery_v2",
                            "universal_frontier_recovery_v3",
                        }
                        and operator == "legacy_local"
                    ):
                        seed = self.make_elite_seed()
                    else:
                        seed = self.make_seed()
                    if seed is None:
                        continue
                    if operator == "legacy_local":
                        remask = self.args.frontier_legacy_local_remask
                        temperature = self.args.frontier_legacy_local_temperature
                        span_prob = self.args.frontier_legacy_local_span_prob
                    else:
                        remask = self.args.remask_fraction
                        temperature = self.args.temperature_start
                        span_prob = self.args.span_prob
                    root_id = hashlib.sha1(seed.encode("utf-8")).hexdigest()[:12]
                    proposals.append(
                        {
                            "seed": seed,
                            "plan": None,
                            "operator": operator,
                            "source_frontier": "stable",
                            "parent_smiles": seed,
                            "parent_residual": None,
                            "parent_stage": None,
                            "parent_token_len": len(tokenize_smiles(seed)),
                            "planned_delta": 0,
                            "peripheral": False,
                            "remask": remask,
                            "temperature": temperature,
                            "span_prob": span_prob,
                            "root_id": root_id,
                            "depth": 1,
                            "search_state": self.integrated_state,
                        }
                    )
                else:
                    proposal = self.frontier_make_guided_proposal(operator)
                    if proposal is not None:
                        proposals.append(proposal)
            if not proposals:
                break

            eligible_indices = [
                index
                for index, proposal in enumerate(proposals)
                if proposal.get("plan") is not None
            ]
            if (
                eligible_indices
                and bool(getattr(self.model, "is_elastic", False))
                and not getattr(self.args, "disable_learned_insertion", False)
            ):
                if self.args.sampler_profile in UNIFIED_FRONTIER_PROFILES:
                    insertion_fraction = (
                        self.frontier_engine.adapter.insertion_fraction(
                            self.integrated_state,
                            self.frontier_engine.last_context,
                        )
                    )
                else:
                    insertion_fraction = LeadFrontierAdapter.insertion_fraction(
                        self.integrated_state,
                        None,
                    )
                insertion_fraction = clamp(
                    insertion_fraction
                    * getattr(
                        self.args,
                        "learned_insertion_fraction_scale",
                        1.0,
                    ),
                    0.0,
                    1.0,
                )
                insertion_flags = allocate_insertion_flags(
                    len(eligible_indices),
                    insertion_fraction,
                    rng=self.frontier_rng,
                )
                state_cap = {
                    "locked": 0,
                    "baseline": 2,
                    "complete": 1,
                    "completion": 1,
                    "refine": 1,
                    "dock_polish": 1,
                    "similarity_boundary": 1,
                    "quality_boundary": 1,
                    "late_route_commit": 1,
                    "late_anchor_route_commit": 1,
                    "bridge": 2,
                    "bridge_probe": 2,
                    "seed_anchor": 2,
                    "warmup": 2,
                    "search": 2,
                    "explore": 3,
                    "plateau": 3,
                    "collapsed": 3,
                }.get(self.integrated_state, 2)
                max_growth = min(
                    max(
                        0,
                        getattr(self.args, "learned_insertion_max_growth", 3),
                    ),
                    state_cap,
                )
                max_shrink = min(
                    max(
                        0,
                        getattr(self.args, "learned_insertion_max_shrink", 3),
                    ),
                    state_cap,
                )
                for proposal_index, use_insertion in zip(
                    eligible_indices,
                    insertion_flags,
                ):
                    proposal = proposals[proposal_index]
                    plan = proposal["plan"]
                    span_len = max(
                        1,
                        int(plan["stop"]) - int(plan["start"]),
                    )
                    if use_insertion:
                        plan["length_mode"] = "learned_insertion"
                        plan["min_replacement_len"] = max(
                            0,
                            span_len - max_shrink,
                        )
                        plan["max_replacement_len"] = max(
                            plan["min_replacement_len"],
                            span_len + max_growth,
                        )
                        proposal["length_mode"] = "learned_insertion"
                        proposal["planned_delta"] = None
                    else:
                        plan["delta"] = 0
                        plan.pop("length_mode", None)
                        proposal["length_mode"] = "fixed"

            stats = self.frontier_operator_stats.setdefault(
                operator,
                {
                    "attempted": 0.0,
                    "accepted": 0.0,
                    "evaluated": 0.0,
                    "updates": 0.0,
                    "reward_ema": 0.0,
                    "last_batch_reward": 0.0,
                    "strict": 0.0,
                },
            )
            stats["attempted"] += len(proposals)
            remask = proposals[0]["remask"]
            temperature = proposals[0]["temperature"]
            span_prob = proposals[0]["span_prob"]
            isolated_rng = operator == "no_pair_anchor_restart" or operator.startswith(
                "completion_"
            )
            if isolated_rng:
                python_rng_state = random.getstate()
                numpy_rng_state = np.random.get_state()
                torch_rng_state = torch.random.get_rng_state()
                cuda_rng_states = (
                    torch.cuda.get_rng_state_all()
                    if torch.cuda.is_available()
                    else None
                )
                rescue_seed = (
                    (self.args.seed + 1) * 104_729
                    + self.frontier_iteration * 10_007
                    + rounds * 101
                    + int.from_bytes(
                        hashlib.sha1(operator.encode("utf-8")).digest()[:4],
                        "big",
                    )
                ) % (2**31 - 1)
                random.seed(rescue_seed)
                np.random.seed(rescue_seed)
                torch.manual_seed(rescue_seed)
                if torch.cuda.is_available():
                    torch.cuda.manual_seed_all(rescue_seed)
            try:
                sampled = sample_csdnet_local_remask(
                    model=self.model,
                    tk=self.tk,
                    seed_smiles=[proposal["seed"] for proposal in proposals],
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
                    edit_plans=(
                        None
                        if operator in self.frontier_base_operators
                        else [proposal["plan"] for proposal in proposals]
                    ),
                    return_seed_indices=True,
                    return_diagnostics=True,
                    learned_insertion_max_growth=getattr(
                        self.args,
                        "learned_insertion_max_growth",
                        3,
                    ),
                    learned_insertion_max_shrink=getattr(
                        self.args,
                        "learned_insertion_max_shrink",
                        3,
                    ),
                    learned_insertion_max_per_step=getattr(
                        self.args,
                        "learned_insertion_max_per_step",
                        4,
                    ),
                    local_sampler_profile=self.frontier_local_sampler_profile(operator),
                )
            finally:
                if isolated_rng:
                    random.setstate(python_rng_state)
                    np.random.set_state(numpy_rng_state)
                    torch.random.set_rng_state(torch_rng_state)
                    if cuda_rng_states is not None:
                        torch.cuda.set_rng_state_all(cuda_rng_states)
            for sampled_item in sampled:
                smiles, proposal_idx = sampled_item[:2]
                sample_diagnostics = sampled_item[2] if len(sampled_item) > 2 else {}
                if not (0 <= proposal_idx < len(proposals)):
                    continue
                can = canonical_smiles(smiles)
                proposal = proposals[proposal_idx]
                if (
                    self.args.sampler_profile
                    in {"lead_protected_completion_v2", "lead_protected_completion_v5"}
                    and operator
                    in {
                        "feasible_dock_polish",
                        "boundary_similarity_polish",
                        "boundary_quality_polish",
                        "completion_dock_refine",
                        "completion_similarity_repair",
                        "completion_quality_repair",
                        "completion_seed_restart",
                    }
                    and can == canonical_smiles(proposal.get("parent_smiles"))
                ):
                    continue
                if can is None or can in seen:
                    continue
                if not tokenizable(can, self.tk, self.args.max_len):
                    continue
                seen.add(can)
                meta = dict(proposal)
                meta.pop("plan", None)
                meta["output_token_len"] = len(tokenize_smiles(can))
                meta["actual_delta"] = (
                    meta["output_token_len"] - meta["parent_token_len"]
                )
                meta.update(sample_diagnostics)
                out.append(can)
                out_meta.append(meta)
                stats["accepted"] += 1.0
                if len(out) >= target_n:
                    break
        return out, out_meta

    def frontier_universal_baseline_allocation(self, target_total):
        """Return the historical universal-frontier allocation at any budget."""
        target_total = max(0, int(target_total))
        if target_total == 0:
            return {}
        legacy_fraction = self.args.frontier_legacy_fraction
        if self.frontier_iteration > self.args.frontier_legacy_warmup_iters:
            legacy_fraction = self.args.frontier_legacy_fraction_late
        legacy_target = max(
            1,
            min(target_total, int(round(target_total * legacy_fraction))),
        )
        allocation = {"legacy": legacy_target}
        guided_budget = max(0, target_total - legacy_target)
        scores = {
            operator: score
            for operator, score in self.frontier_available_operator_scores().items()
            if operator not in set(self.frontier_task_head_operators)
        }
        minimum_each = max(
            1,
            int(round(guided_budget * self.args.frontier_min_operator_fraction)),
        )
        allocation.update(
            allocate_counts(guided_budget, scores, minimum_each=minimum_each)
        )
        return allocation

    def frontier_refill_v5_reserve(self, target_total, seen):
        """Return unproductive V5 reserve slots to the untouched V1 policy."""
        target_total = max(0, int(target_total))
        out = []
        metadata = []
        attempts = 0
        while len(out) < target_total and attempts < 3:
            attempts += 1
            remaining = target_total - len(out)
            allocation = self.frontier_universal_baseline_allocation(remaining)
            before = len(out)
            for operator in self.frontier_base_operators + self.frontier_operators:
                if operator in set(self.frontier_task_head_operators):
                    continue
                generated, generated_meta = self.frontier_generate_operator(
                    operator,
                    allocation.get(operator, 0),
                    seen,
                )
                for row in generated_meta:
                    row["v5_refill"] = True
                out.extend(generated)
                metadata.extend(generated_meta)
            if len(out) == before:
                break
        return out[:target_total], metadata[:target_total]

    def generate_batch_multi_frontier(self):
        self.current_candidate_ops = []
        self.current_candidate_meta = []
        seen = (
            set(self.oracle_evaluated_smiles)
            if self.args.global_oracle_dedup
            else set()
        )
        target_total = self.args.num_gen
        if self.args.sampler_profile in UNIFIED_FRONTIER_PROFILES:
            allocation = self.frontier_engine.allocate(
                target_total,
                state=self.integrated_state,
            ).get("proposal", {})
            self.frontier_last_allocation = dict(allocation)
            out = []
            metadata = []
            for operator in self.frontier_base_operators + self.frontier_operators:
                count = allocation.get(operator, 0)
                generated, generated_meta = self.frontier_generate_operator(
                    operator,
                    count,
                    seen,
                )
                out.extend(generated)
                metadata.extend(generated_meta)
            legacy_fraction = sum(
                allocation.get(operator, 0) for operator in self.frontier_base_operators
            ) / max(1, target_total)
        elif self.args.sampler_profile in {
            "universal_frontier_bridge",
            "lead_protected_completion",
            "lead_protected_completion_v2",
            "lead_protected_completion_v3",
            "lead_protected_completion_v4",
            "lead_protected_completion_v5",
        }:
            nested = self.protected_frontier_engine.allocate(
                target_total,
                lambda total: {
                    "proposal": self.frontier_universal_baseline_allocation(total)
                },
                state=self.integrated_state,
                context=self.protected_frontier_engine.last_context,
                available={
                    "proposal": set(
                        self.frontier_base_operators + self.frontier_operators
                    )
                },
            )
            allocation = nested.get("proposal", {})
            self.frontier_last_allocation = dict(allocation)
            out = []
            metadata = []
            task_head_shortfall = 0
            task_head_operators = set(self.frontier_task_head_operators)
            for operator in self.frontier_base_operators + self.frontier_operators:
                requested = allocation.get(operator, 0)
                generated, generated_meta = self.frontier_generate_operator(
                    operator,
                    requested,
                    seen,
                )
                out.extend(generated)
                metadata.extend(generated_meta)
                if (
                    self.args.sampler_profile == "lead_protected_completion_v5"
                    and operator in task_head_operators
                ):
                    task_head_shortfall += max(0, requested - len(generated))
            if (
                self.args.sampler_profile == "lead_protected_completion_v5"
                and task_head_shortfall > 0
                and len(out) < target_total
            ):
                refill_target = min(task_head_shortfall, target_total - len(out))
                refill, refill_meta = self.frontier_refill_v5_reserve(
                    refill_target,
                    seen,
                )
                out.extend(refill)
                metadata.extend(refill_meta)
                print(
                    "[V5-refill] "
                    f"reserved_shortfall={task_head_shortfall} "
                    f"requested={refill_target} accepted={len(refill)}"
                )
            legacy_fraction = sum(
                allocation.get(operator, 0) for operator in self.frontier_base_operators
            ) / max(1, target_total)
        elif self.args.sampler_profile == "integrated_frontier":
            legacy_fraction = {
                "warmup": self.args.integrated_legacy_fraction_warmup,
                "search": self.args.integrated_legacy_fraction_search,
                "refine": self.args.integrated_legacy_fraction_refine,
                "plateau": self.args.integrated_legacy_fraction_rescue,
                "collapsed": self.args.integrated_legacy_fraction_rescue,
            }[self.integrated_state]
        elif self.args.sampler_profile in {
            "universal_frontier_recovery_v2",
            "universal_frontier_recovery_v3",
        }:
            if not self.frontier_recovery_active():
                fragment_fraction = self.args.frontier_legacy_fraction
                if self.frontier_iteration > self.args.frontier_legacy_warmup_iters:
                    fragment_fraction = self.args.frontier_legacy_fraction_late
                local_fraction = 0.0
            elif self.args.sampler_profile == "universal_frontier_recovery_v3":
                fragment_fraction, local_fraction = (0.30, 0.15)
            else:
                fragment_fraction, local_fraction = {
                    "seed_anchor": (0.15, 0.25),
                    "explore": (0.30, 0.15),
                    "bridge": (0.20, 0.20),
                    "complete": (0.15, 0.15),
                    "refine": (0.40, 0.20),
                    "warmup": (0.60, 0.00),
                }[self.integrated_state]
            legacy_fraction = fragment_fraction + local_fraction
        else:
            legacy_fraction = self.args.frontier_legacy_fraction
            if self.frontier_iteration > self.args.frontier_legacy_warmup_iters:
                legacy_fraction = self.args.frontier_legacy_fraction_late
            if (
                self.args.sampler_profile == "safe_frontier_final"
                and self.frontier_recovery_active()
            ):
                reserve = self.protected_frontier_engine.reserve_fraction(
                    self.integrated_state,
                    self.protected_frontier_engine.last_context,
                )
                legacy_fraction = 1.0 - reserve
            elif self.frontier_recovery_active():
                legacy_fraction = self.args.frontier_recovery_legacy_fraction
        if (
            self.args.sampler_profile not in UNIFIED_FRONTIER_PROFILES
            and self.args.sampler_profile
            not in {
                "universal_frontier_bridge",
                "lead_protected_completion",
                "lead_protected_completion_v2",
                "lead_protected_completion_v3",
                "lead_protected_completion_v4",
                "lead_protected_completion_v5",
            }
        ):
            self.frontier_last_allocation = {}
            legacy_fraction = clamp(legacy_fraction, 0.0, 1.0)
            if self.args.sampler_profile in {
                "universal_frontier_recovery_v2",
                "universal_frontier_recovery_v3",
            }:
                fragment_target = int(round(target_total * fragment_fraction))
                local_target = int(round(target_total * local_fraction))
                fragment_target = max(1, min(target_total, fragment_target))
                local_target = max(
                    0,
                    min(target_total - fragment_target, local_target),
                )
                out, metadata = self.frontier_generate_operator(
                    "legacy", fragment_target, seen
                )
                local_out, local_meta = self.frontier_generate_operator(
                    "legacy_local", local_target, seen
                )
                out.extend(local_out)
                metadata.extend(local_meta)
                base_target = fragment_target + local_target
                allocation = {
                    "legacy": fragment_target,
                    "legacy_local": local_target,
                }
            else:
                legacy_target = int(round(target_total * legacy_fraction))
                legacy_target = max(1, min(target_total, legacy_target))
                out, metadata = self.frontier_generate_operator(
                    "legacy", legacy_target, seen
                )
                base_target = legacy_target
                allocation = {"legacy": legacy_target}

            guided_budget = max(0, target_total - base_target)
            scores = self.frontier_available_operator_scores()
            minimum_each = max(
                1,
                int(round(guided_budget * self.args.frontier_min_operator_fraction)),
            )
            guided_allocation = allocate_counts(
                guided_budget,
                scores,
                minimum_each=minimum_each,
            )
            allocation.update(guided_allocation)
            for operator in self.frontier_operators:
                count = allocation.get(operator, 0)
                generated, generated_meta = self.frontier_generate_operator(
                    operator, count, seen
                )
                out.extend(generated)
                metadata.extend(generated_meta)

        if len(out) < target_total:
            generated, generated_meta = self.frontier_generate_operator(
                "legacy", target_total - len(out), seen
            )
            out.extend(generated)
            metadata.extend(generated_meta)

        self.current_candidate_meta = metadata
        self.current_candidate_ops = [meta["operator"] for meta in metadata]
        accepted = defaultdict(int)
        for meta in metadata:
            accepted[meta["operator"]] += 1
        requested_summary = ",".join(
            f"{operator}:{allocation.get(operator, 0)}"
            for operator in self.frontier_base_operators + self.frontier_operators
        )
        accepted_summary = ",".join(
            f"{operator}:{accepted.get(operator, 0)}"
            for operator in self.frontier_base_operators + self.frontier_operators
        )
        print(
            f"[Frontier-gen] state={self.integrated_state} "
            f"recovery={'on' if self.frontier_recovery_active() else 'off'} "
            f"legacy_fraction={legacy_fraction:.2f} "
            f"requested={requested_summary} "
            f"accepted={accepted_summary} total={len(out)}/{target_total}"
        )
        return out

    def record_frontier_diagnostics(self, smiles_list, prop_list):
        if self.args.sampler_profile not in FRONTIER_PROFILES:
            return
        rv_list, rq_list, rs_list, rsim_list = prop_list
        fields = [
            "iteration",
            "smiles",
            "parent_smiles",
            "donor_smiles",
            "bridge_pair",
            "operator",
            "source_frontier",
            "planned_delta",
            "actual_delta",
            "length_mode",
            "removed_tokens",
            "inserted_tokens",
            "edit_strategy",
            "completion_parent_rank",
            "completion_constraint_slack",
            "parent_token_len",
            "output_token_len",
            "peripheral",
            "parent_residual",
            "parent_stage",
            "dock",
            "qed",
            "sa",
            "similarity",
            "output_residual",
            "output_max_deficit",
            "output_mean_deficit",
            "output_bottleneck",
            "output_stage",
            "output_frontiers",
            "strict",
            "constraints_crossed",
            "constraints_regressed",
            "pair_frontiers_gained",
            "operator_reward",
            "operator_batch_reward",
            "root_id",
            "depth",
            "root_credit",
            "search_state",
            "rng_stream",
            "proposal_route",
            "plan_state",
            "v5_edit_arm",
            "v5_parent_similarity",
            "v5_similarity_slack_ratio",
            "v5_trust_region_atoms",
            "planned_learned_insertion",
            "initial_mask_tokens",
            "learned_inserted_tokens",
            "insertion_steps",
            "fsm_repair_progressive_steps",
            "fsm_repair_prefer_localization",
            "fsm_constraint_mode",
            "fsm_check_enabled",
            "rdkit_kekulize_check_enabled",
            "online_fsm_repair_events",
            "online_fsm_remasked_tokens",
            "cheap_qed",
            "cheap_sa",
            "cheap_similarity",
            "cheap_feasible",
            "cheap_sim_deficit",
            "cheap_quality_deficit",
            "cheap_joint_deficit",
            "cheap_bucket",
        ]
        write_header = not os.path.exists(self.frontier_diagnostic_path)
        with open(self.frontier_diagnostic_path, "a", newline="") as handle:
            writer = csv.DictWriter(handle, fieldnames=fields)
            if write_header:
                writer.writeheader()
            for idx, (smiles, dock, qed, sa, similarity) in enumerate(
                zip(smiles_list, rv_list, rq_list, rs_list, rsim_list)
            ):
                meta = (
                    self.current_candidate_meta[idx]
                    if idx < len(self.current_candidate_meta)
                    else {}
                )
                writer.writerow(
                    {
                        "iteration": self.frontier_iteration,
                        "smiles": smiles,
                        "parent_smiles": meta.get("parent_smiles"),
                        "donor_smiles": meta.get("donor_smiles"),
                        "bridge_pair": meta.get("bridge_pair"),
                        "operator": meta.get("operator", "legacy"),
                        "source_frontier": meta.get("source_frontier"),
                        "planned_delta": meta.get("planned_delta", 0),
                        "actual_delta": meta.get("actual_delta"),
                        "length_mode": meta.get("length_mode", "fixed"),
                        "removed_tokens": meta.get("removed_tokens"),
                        "inserted_tokens": meta.get("inserted_tokens"),
                        "edit_strategy": meta.get("edit_strategy"),
                        "completion_parent_rank": meta.get("completion_parent_rank"),
                        "completion_constraint_slack": meta.get(
                            "completion_constraint_slack"
                        ),
                        "parent_token_len": meta.get("parent_token_len"),
                        "output_token_len": meta.get("output_token_len"),
                        "peripheral": meta.get("peripheral", False),
                        "parent_residual": meta.get("parent_residual"),
                        "parent_stage": meta.get("parent_stage"),
                        "dock": dock,
                        "qed": qed,
                        "sa": sa,
                        "similarity": similarity,
                        "output_residual": meta.get("output_residual"),
                        "output_max_deficit": meta.get("output_max_deficit"),
                        "output_mean_deficit": meta.get("output_mean_deficit"),
                        "output_bottleneck": meta.get("output_bottleneck"),
                        "output_stage": meta.get("output_stage"),
                        "output_frontiers": meta.get("output_frontiers"),
                        "strict": meta.get("output_strict", False),
                        "constraints_crossed": meta.get("constraints_crossed", 0),
                        "constraints_regressed": meta.get("constraints_regressed", 0),
                        "pair_frontiers_gained": meta.get("pair_frontiers_gained", 0),
                        "operator_reward": meta.get("operator_reward"),
                        "operator_batch_reward": meta.get("operator_batch_reward"),
                        "root_id": meta.get("root_id"),
                        "depth": meta.get("depth"),
                        "root_credit": meta.get("root_credit"),
                        "search_state": meta.get("search_state", self.integrated_state),
                        "rng_stream": meta.get("rng_stream", "frontier"),
                        "proposal_route": meta.get("proposal_route"),
                        "plan_state": meta.get("plan_state"),
                        "v5_edit_arm": meta.get("v5_edit_arm"),
                        "v5_parent_similarity": meta.get(
                            "v5_parent_similarity"
                        ),
                        "v5_similarity_slack_ratio": meta.get(
                            "v5_similarity_slack_ratio"
                        ),
                        "v5_trust_region_atoms": meta.get(
                            "v5_trust_region_atoms"
                        ),
                        "planned_learned_insertion": meta.get(
                            "planned_learned_insertion"
                        ),
                        "initial_mask_tokens": meta.get("initial_mask_tokens"),
                        "learned_inserted_tokens": meta.get("learned_inserted_tokens"),
                        "insertion_steps": meta.get("insertion_steps"),
                        "fsm_repair_progressive_steps": meta.get(
                            "fsm_repair_progressive_steps"
                        ),
                        "fsm_repair_prefer_localization": meta.get(
                            "fsm_repair_prefer_localization"
                        ),
                        "fsm_constraint_mode": meta.get("fsm_constraint_mode"),
                        "fsm_check_enabled": meta.get("fsm_check_enabled"),
                        "rdkit_kekulize_check_enabled": meta.get(
                            "rdkit_kekulize_check_enabled"
                        ),
                        "online_fsm_repair_events": meta.get(
                            "online_fsm_repair_events"
                        ),
                        "online_fsm_remasked_tokens": meta.get(
                            "online_fsm_remasked_tokens"
                        ),
                        "cheap_qed": meta.get("cheap_qed"),
                        "cheap_sa": meta.get("cheap_sa"),
                        "cheap_similarity": meta.get("cheap_similarity"),
                        "cheap_feasible": meta.get("cheap_feasible"),
                        "cheap_sim_deficit": meta.get("cheap_sim_deficit"),
                        "cheap_quality_deficit": meta.get("cheap_quality_deficit"),
                        "cheap_joint_deficit": meta.get("cheap_joint_deficit"),
                        "cheap_bucket": meta.get("cheap_bucket"),
                    }
                )

    @staticmethod
    def _elastic_joint_unmet(item, keys):
        """Measure only the constraints missing from one pair frontier."""
        values = []
        for key in keys:
            deficit = max(0.0, float(item["deficits"].get(key, 0.0)))
            if not item["checks"].get(key, False):
                deficit = max(deficit, 1e-6)
            values.append(deficit)
        return max(values, default=0.0)

    def _elastic_joint_state(self):
        """Choose the smallest observed joint-constraint completion problem.

        The route depends only on benchmark constraints. In particular, an SQ
        parent needs docking, QD needs similarity, and SD needs molecular
        quality. This avoids the global pass-rate heuristic used by
        ``elastic_direct``, which can ignore a rare but nearly feasible parent.
        """
        self.elastic_joint_source = "start"
        self.elastic_joint_route_score = None
        generated_items = [
            item
            for smiles, item in self.frontier_items.items()
            if smiles != self.start_smiles
        ]
        if self.frontier_archives["strict"]:
            self.elastic_joint_source = "strict"
            self.elastic_joint_route_score = 0.0
            return "polish"
        if not generated_items:
            if (
                self.args.sampler_profile == "elastic_joint_frontier_v5"
                and self.elastic_public_frontier
            ):
                best = min(
                    self.elastic_public_frontier.values(),
                    key=lambda meta: self._elastic_joint_v2_cheap_key(
                        meta,
                        "warmup",
                    ),
                )
                sim_deficit = float(best["cheap_sim_deficit"])
                quality_deficit = float(best["cheap_quality_deficit"])
                self.elastic_joint_source = "public"
                self.elastic_joint_route_score = max(
                    sim_deficit,
                    quality_deficit,
                )
                if sim_deficit <= 0.0 < quality_deficit:
                    return "quality"
                if quality_deficit <= 0.0 < sim_deficit:
                    return "anchor"
                if max(sim_deficit, quality_deficit) <= 0.14:
                    return "joint_bridge"
                return "anchor" if sim_deficit >= quality_deficit else "quality"
            return "warmup"

        route_specs = (
            ("dock_completion", "sq", ("dock",)),
            ("similarity_completion", "qd", ("sim",)),
            ("quality_completion", "sd", ("qed", "sa")),
        )
        routes = []
        for state, label, missing_keys in route_specs:
            archive = self.frontier_archives[label]
            if not archive:
                continue
            route_score = min(
                self._elastic_joint_unmet(item, missing_keys)
                + 0.02 * float(item["mean_deficit"])
                for item in archive
            )
            routes.append((route_score, state, label))
        if routes:
            route_score, state, label = min(routes)
            self.elastic_joint_source = label
            self.elastic_joint_route_score = float(route_score)
            return state

        best = min(generated_items, key=constraint_rank)
        self.elastic_joint_route_score = float(best["max_deficit"])
        if float(best["max_deficit"]) <= 0.14 or int(best["stage"]) >= 3:
            self.elastic_joint_source = "near"
            return "joint_bridge"

        sim_feasible = any(item["checks"]["sim"] for item in generated_items)
        quality_feasible = any(
            item["checks"]["qed"] and item["checks"]["sa"] for item in generated_items
        )
        if not sim_feasible and not quality_feasible:
            best_sim = min(item["deficits"]["sim"] for item in generated_items)
            best_quality = min(
                max(item["deficits"]["qed"], item["deficits"]["sa"])
                for item in generated_items
            )
            self.elastic_joint_source = "near"
            return "anchor" if best_sim >= best_quality else "quality"
        if not sim_feasible:
            self.elastic_joint_source = "s"
            return "anchor"
        if not quality_feasible:
            self.elastic_joint_source = "s"
            return "quality"
        self.elastic_joint_source = "near"
        return "joint_bridge"

    def _elastic_joint_parent_key(self, item, state):
        deficits = item["deficits"]
        normalized = item["normalized"]
        if state == "polish":
            return (-item["dock"], -item["sim"], -item["qed"], -item["sa"])
        if state == "dock_completion":
            protected = min(normalized[key] for key in ("qed", "sa", "sim"))
            return (
                self._elastic_joint_unmet(item, ("dock",)),
                -protected,
                -item["dock"],
            )
        if state == "similarity_completion":
            protected = min(normalized[key] for key in ("dock", "qed", "sa"))
            return (
                self._elastic_joint_unmet(item, ("sim",)),
                -protected,
                item["mean_deficit"],
            )
        if state == "quality_completion":
            protected = min(normalized[key] for key in ("dock", "sim"))
            return (
                self._elastic_joint_unmet(item, ("qed", "sa")),
                -protected,
                item["mean_deficit"],
            )
        if state == "anchor":
            joint_gap = max(
                deficits["sim"],
                deficits["qed"],
                deficits["sa"],
                deficits["dock"],
            )
            return (
                joint_gap,
                deficits["sim"],
                max(deficits["qed"], deficits["sa"], deficits["dock"]),
                item["mean_deficit"],
            )
        if state == "quality":
            joint_gap = max(
                deficits["sim"],
                deficits["qed"],
                deficits["sa"],
                deficits["dock"],
            )
            return (
                joint_gap,
                max(deficits["qed"], deficits["sa"]),
                deficits["sim"],
                item["mean_deficit"],
            )
        return constraint_rank(item)

    def _elastic_joint_parent_pool(self, state):
        source_labels = {
            "polish": ("strict",),
            "dock_completion": ("sq",),
            "similarity_completion": ("qd",),
            "quality_completion": ("sd",),
            "joint_bridge": ("sq", "qd", "sd", "near"),
            "anchor": ("s", "near"),
            "quality": ("s", "near"),
            "explore": ("near",),
        }.get(state, ("near",))
        deduplicated = {}
        for label in source_labels:
            for item in self.frontier_archives[label]:
                deduplicated[item["smiles"]] = item
        if state in {"warmup", "anchor", "quality", "explore"}:
            start = self.frontier_items.get(self.start_smiles)
            if start is not None:
                deduplicated[start["smiles"]] = start
        if (
            self.args.sampler_profile == "elastic_joint_frontier_v5"
            and state not in {"polish", "dock_completion"}
        ):
            for item in self._elastic_joint_v5_public_parent_items():
                deduplicated.setdefault(item["smiles"], item)
        items = sorted(
            deduplicated.values(),
            key=lambda item: self._elastic_joint_parent_key(item, state),
        )
        return items[: max(1, int(self.args.direct_parent_pool_size))]

    def _elastic_joint_plan(self, smiles, state, learned_insertion):
        tokens = tokenize_smiles(smiles)
        mol = Chem.MolFromSmiles(smiles)
        if not tokens or mol is None:
            return None
        span_ranges = {
            "warmup": (2, 4),
            "anchor": (1, 2),
            "quality": (1, 3),
            "joint_bridge": (1, 2),
            "dock_completion": (1, 3),
            "similarity_completion": (1, 1),
            "quality_completion": (1, 2),
            "polish": (1, 2),
            "explore": (2, 5),
        }
        low, high = span_ranges[state]
        span_tokens = self.frontier_rng.randint(low, high)

        plan = None
        if (
            state in {"anchor", "similarity_completion"}
            and smiles != self.start_smiles
            and self.frontier_rng.random() < 0.70
        ):
            plan = seed_directed_atom_edit_plan(
                smiles,
                self.start_smiles,
                self.frontier_rng,
            )
        peripheral_probability = {
            "anchor": 0.85,
            "similarity_completion": 0.90,
            "quality": 0.85,
            "quality_completion": 0.88,
            "dock_completion": 0.75,
            "polish": 0.82,
        }.get(state, self.args.direct_peripheral_probability)
        if plan is None and self.frontier_rng.random() < peripheral_probability:
            target_fraction = span_tokens / max(1, mol.GetNumAtoms())
            plan = adaptive_peripheral_edit_plan(
                smiles,
                self.frontier_rng,
                delta=0,
                target_atom_fraction=target_fraction,
                max_atom_fraction=max(0.12, target_fraction + 0.08),
                max_span_tokens=span_tokens,
            )
        if plan is None:
            plan = atom_span_edit_plan(
                smiles,
                self.frontier_rng,
                delta=0,
                span_tokens=span_tokens,
            )
        if plan is None or not learned_insertion:
            return plan

        removed = max(1, int(plan["stop"]) - int(plan["start"]))
        parent_length = len(tokens)
        shrink, growth = {
            "warmup": (1, 2),
            "anchor": (0, 1),
            "quality": (1, 2),
            "joint_bridge": (1, 1),
            "dock_completion": (1, 2),
            "similarity_completion": (0, 1),
            "quality_completion": (1, 1),
            "polish": (0, 1),
            "explore": (2, 3),
        }[state]
        prior_low, prior_high = self.elastic_length_support
        start_length = len(tokenize_smiles(self.start_smiles))
        support_low = max(3, min(prior_low, start_length, parent_length) - 4)
        support_high = min(
            self.args.max_len - 2,
            max(prior_high, start_length, parent_length) + 4,
        )
        final_low = max(support_low, parent_length - shrink)
        final_high = min(support_high, parent_length + growth)
        fixed_length = parent_length - removed
        minimum = max(0, final_low - fixed_length)
        maximum = max(minimum, final_high - fixed_length)
        if minimum == maximum:
            return plan
        learned = dict(plan)
        learned.update(
            {
                "length_mode": "learned_insertion",
                "min_replacement_len": int(minimum),
                "max_replacement_len": int(maximum),
                # Starting at the lower trust-region boundary leaves the learned
                # insertion head, rather than a hand-written delta, in control.
                "initial_replacement_len": int(minimum),
                "prior_guidance": "atomic_trust_region",
            }
        )
        return learned

    def generate_batch_elastic_joint_frontier(self):
        state = self._elastic_joint_state()
        settings = {
            "warmup": ("start_repair", 0.55, 1.00, 0.65, 0.12),
            "anchor": ("similarity_repair", 0.35, 0.82, 0.50, 0.06),
            "quality": ("quality_repair", 0.20, 0.88, 0.55, 0.12),
            "joint_bridge": ("joint_repair", 0.12, 0.86, 0.55, 0.10),
            "dock_completion": ("dock_refine", 0.05, 0.92, 0.65, 0.08),
            "similarity_completion": (
                "similarity_repair",
                0.04,
                0.78,
                0.45,
                0.04,
            ),
            "quality_completion": ("quality_repair", 0.05, 0.80, 0.48, 0.08),
            "polish": ("dock_refine", 0.03, 0.82, 0.52, 0.04),
            "explore": ("joint_repair", 0.20, 1.05, 0.70, 0.15),
        }
        operator, start_probability, temperature, top_p, insertion_fraction = settings[
            state
        ]
        parents = self._elastic_joint_parent_pool(state)
        if not parents:
            parents = [self.frontier_items[self.start_smiles]]

        seen = set()
        output = []
        metadata = []
        rounds = 0
        while (
            len(output) < self.args.num_gen and rounds < self.args.max_generation_rounds
        ):
            rounds += 1
            remaining = self.args.num_gen - len(output)
            proposal_count = max(
                remaining,
                int(math.ceil(remaining * self.args.direct_overgenerate_factor)),
            )
            insertion_flags = allocate_insertion_flags(
                proposal_count,
                insertion_fraction,
                rng=self.frontier_rng,
            )
            seeds = []
            plans = []
            proposal_meta = []
            for learned_insertion in insertion_flags:
                if self.frontier_rng.random() < start_probability:
                    parent = self.frontier_items[self.start_smiles]
                else:
                    rank = min(
                        len(parents) - 1,
                        int(self.frontier_rng.expovariate(0.38)),
                    )
                    parent = parents[rank]
                plan = self._elastic_joint_plan(
                    parent["smiles"],
                    state,
                    learned_insertion,
                )
                if plan is None:
                    continue
                seeds.append(parent["smiles"])
                plans.append(plan)
                proposal_meta.append(
                    {
                        "operator": operator,
                        "parent_smiles": parent["smiles"],
                        "parent_residual": parent["residual"],
                        "parent_stage": parent["stage"],
                        "parent_token_len": len(tokenize_smiles(parent["smiles"])),
                        "root_id": parent.get("root_id", "start"),
                        "depth": int(parent.get("depth", 0)) + 1,
                        "search_state": state,
                        "source_frontier": self.elastic_joint_source,
                        "planned_learned_insertion": bool(
                            plan.get("length_mode") == "learned_insertion"
                        ),
                        "peripheral": bool(plan.get("peripheral", False)),
                    }
                )
            if not seeds:
                break

            generated = sample_csdnet_local_remask(
                model=self.model,
                tk=self.tk,
                seed_smiles=seeds,
                edit_plans=plans,
                max_len=self.args.max_len,
                device=self.device,
                batch_size=self.args.batch_size,
                n_steps=self.args.n_steps,
                use_fsm_check=not self.args.disable_fsm_check,
                use_rdkit_kekulize_check=not self.args.disable_rdkit_kekulize_check,
                max_sample_retries=self.args.max_sample_retries,
                violation_neighborhood=self.args.violation_neighborhood,
                temperature_start=temperature,
                temperature_end=max(0.20, min(0.35, temperature * 0.35)),
                temperature_power=1.2,
                top_k=0,
                top_p=top_p,
                gumbel_scale=0.55,
                remask_power=1.0,
                learned_insertion_max_growth=3,
                learned_insertion_max_shrink=2,
                learned_insertion_max_per_step=self.args.learned_insertion_max_per_step,
                learned_insertion_fallback=True,
                learned_insertion_recursive_gap_insertions=True,
                learned_insertion_trajectory_mode="plan_then_fill",
                learned_insertion_planning_fraction=0.30,
                learned_insertion_fill_mode="progressive_remask",
                learned_insertion_fill_remask_power=0.85,
                learned_insertion_fill_gumbel_scale=0.35,
                local_sampler_profile="legacy",
                return_seed_indices=True,
                return_diagnostics=True,
            )
            for smiles, seed_index, diagnostics in generated:
                can = canonical_smiles(smiles)
                if (
                    can is None
                    or can in seen
                    or can in self.oracle_evaluated_smiles
                    or not tokenizable(can, self.tk, self.args.max_len)
                ):
                    continue
                if not self.args.min_atoms <= atom_count(can) <= self.args.max_atoms:
                    continue
                seed_index = int(seed_index)
                if not 0 <= seed_index < len(proposal_meta):
                    continue
                seen.add(can)
                output.append(can)
                meta = dict(proposal_meta[seed_index])
                meta.update(
                    {
                        "length_mode": diagnostics.get("length_mode", "fixed"),
                        "removed_tokens": diagnostics.get("removed_tokens"),
                        "inserted_tokens": diagnostics.get("inserted_tokens"),
                        "actual_delta": diagnostics.get("actual_delta"),
                        "initial_mask_tokens": diagnostics.get(
                            "initial_inserted_tokens"
                        ),
                    }
                )
                metadata.append(meta)
                if len(output) >= self.args.num_gen:
                    break

        self.current_candidate_meta = metadata
        self.current_candidate_ops = [item["operator"] for item in metadata]
        learned_count = sum(
            is_learned_length_mode(item.get("length_mode")) for item in metadata
        )
        print(
            f"[Elastic-joint] state={state} source={self.elastic_joint_source} "
            f"route_gap={self.elastic_joint_route_score} generated={len(output)}/"
            f"{self.args.num_gen} rounds={rounds} parents={len(parents)} "
            f"start_prob={start_probability:.2f} top_p={top_p:.2f} "
            f"learned={learned_count}/{len(metadata)}"
        )
        return output

    @staticmethod
    def _elastic_direct_sampling_settings(state):
        return {
            "warmup": (0.60, 1.00, 0.60),
            "anchor": (0.72, 0.88, 0.50),
            "quality": (0.40, 0.94, 0.50),
            "dock": (0.22, 1.05, 0.70),
            "polish": (0.12, 0.86, 0.50),
            "explore": (0.28, 1.10, 0.75),
        }[state]

    def _elastic_joint_v2_route_weights(self, state):
        """Allocate proposals from observed frontier state, never target ID."""
        base = {
            "warmup": (0.45, 0.30, 0.25),
            "anchor": (0.35, 0.50, 0.15),
            "quality": (0.55, 0.20, 0.25),
            "joint_bridge": (0.60, 0.15, 0.25),
            "dock_completion": (0.60, 0.15, 0.25),
            "similarity_completion": (0.55, 0.35, 0.10),
            "quality_completion": (0.65, 0.15, 0.20),
            "polish": (0.70, 0.10, 0.20),
            "explore": (0.40, 0.15, 0.45),
        }[state]
        route, start, escape = base
        # Repeated batches without a strict improvement broaden the proposal
        # source, but do not change benchmark thresholds or inspect target ID.
        pressure = min(0.30, 0.05 * max(0, int(self.no_improve_iters)))
        updated_escape = min(0.60, escape + pressure)
        retained = max(0.0, 1.0 - updated_escape)
        protected_total = route + start
        if protected_total > 0.0:
            route = retained * route / protected_total
            start = retained * start / protected_total
        return {
            "route": float(route),
            "start": float(start),
            "escape": float(updated_escape),
        }

    def _elastic_joint_v3_route_weights(self, state):
        """Protect the narrow delta=0.6 trust region without target priors."""
        if float(self.args.sim_thr) < 0.55:
            return self._elastic_joint_v2_route_weights(state)

        route, start, escape = {
            "warmup": (0.30, 0.60, 0.10),
            "anchor": (0.38, 0.57, 0.05),
            "quality": (0.52, 0.38, 0.10),
            "joint_bridge": (0.62, 0.28, 0.10),
            "dock_completion": (0.72, 0.23, 0.05),
            "similarity_completion": (0.45, 0.52, 0.03),
            "quality_completion": (0.65, 0.25, 0.10),
            "polish": (0.75, 0.20, 0.05),
            "explore": (0.50, 0.35, 0.15),
        }[state]
        # Stagnation still broadens the search, but a high-similarity benchmark
        # must not turn most of its docking budget into unconstrained restarts.
        updated_escape = min(
            0.20,
            escape + 0.02 * max(0, int(self.no_improve_iters)),
        )
        retained = 1.0 - updated_escape
        protected_total = route + start
        route = retained * route / protected_total
        start = retained * start / protected_total
        return {
            "route": float(route),
            "start": float(start),
            "escape": float(updated_escape),
        }

    def _elastic_joint_v4_route_weights(self, state):
        """Threshold-specific source allocation for the final Lead policy."""
        if float(self.args.sim_thr) < 0.55:
            return self._elastic_joint_v2_route_weights(state)
        route, start, escape = {
            "warmup": (0.25, 0.72, 0.03),
            "anchor": (0.45, 0.53, 0.02),
            "quality": (0.58, 0.40, 0.02),
            "joint_bridge": (0.68, 0.30, 0.02),
            "dock_completion": (0.78, 0.21, 0.01),
            "similarity_completion": (0.55, 0.44, 0.01),
            "quality_completion": (0.68, 0.30, 0.02),
            "polish": (0.84, 0.15, 0.01),
            "explore": (0.55, 0.42, 0.03),
        }[state]
        # At delta=0.6, broad restarts empirically leave the joint feasible
        # region. Stagnation increases only the local route share; it never
        # converts this benchmark into an unconstrained global search.
        pressure = min(0.12, 0.015 * max(0, int(self.no_improve_iters)))
        route = min(0.90, route + pressure)
        retained = max(0.0, 1.0 - route - escape)
        start = retained
        return {"route": route, "start": start, "escape": escape}

    def _elastic_joint_v4_legacy_fraction(self, state):
        """Reserve high-similarity proposals for the proven fixed kernel.

        Historical replay gives the fixed protected-completion route 11/15
        strict delta=0.6 cells versus 8/14 for the purely local elastic route.
        The kernels are complementary: fixed fragment proposals can make the
        larger chemical redesign needed by low-QED supplied actives, while
        elastic micro-edits protect an already feasible similarity lineage.
        Cheap public constraints arbitrate the union before docking calls.
        """
        if float(self.args.sim_thr) < 0.55:
            return 0.0
        return {
            "warmup": 0.45,
            "anchor": 0.40,
            "quality": 0.50,
            "joint_bridge": 0.38,
            "dock_completion": 0.20,
            "similarity_completion": 0.35,
            "quality_completion": 0.45,
            "polish": 0.15,
            "explore": 0.45,
        }[state]

    def _elastic_joint_threshold_tightness(self):
        """Map the benchmark threshold onto one shared trust-region scale."""
        return float(clamp((float(self.args.sim_thr) - 0.4) / 0.2, 0.0, 1.0))

    def _elastic_joint_v5_route_weights(self, state):
        """Use one threshold-adaptive proposal framework for both deltas.

        The similarity threshold changes only continuous search parameters.
        It does not select a task-specific model, operator, or scoring rule.
        Public-constraint gating makes moderate exploration affordable because
        an infeasible proposal can become a parent but never an oracle call.
        """
        loose = {
            "warmup": (0.45, 0.30, 0.25),
            "anchor": (0.35, 0.50, 0.15),
            "quality": (0.55, 0.20, 0.25),
            "joint_bridge": (0.60, 0.15, 0.25),
            "dock_completion": (0.60, 0.15, 0.25),
            "similarity_completion": (0.55, 0.35, 0.10),
            "quality_completion": (0.65, 0.15, 0.20),
            "polish": (0.55, 0.20, 0.25),
            "explore": (0.40, 0.15, 0.45),
        }[state]
        tight = {
            "warmup": (0.30, 0.58, 0.12),
            "anchor": (0.45, 0.48, 0.07),
            "quality": (0.58, 0.30, 0.12),
            "joint_bridge": (0.65, 0.25, 0.10),
            "dock_completion": (0.78, 0.17, 0.05),
            "similarity_completion": (0.55, 0.40, 0.05),
            "quality_completion": (0.68, 0.22, 0.10),
            "polish": (0.55, 0.35, 0.10),
            "explore": (0.50, 0.35, 0.15),
        }[state]
        tightness = self._elastic_joint_threshold_tightness()
        route, start, escape = (
            (1.0 - tightness) * low + tightness * high
            for low, high in zip(loose, tight)
        )

        # Once a strict hit exists, repeatedly editing only that lineage causes
        # canonical mode collapse. Stagnation therefore broadens away from the
        # strict route; the public gate still prevents infeasible proposals from
        # consuming docking calls.
        if state == "polish":
            pressure = min(0.25, 0.025 * max(0, int(self.no_improve_iters)))
            previous_route = route
            route = max(0.30, route - pressure)
            released = previous_route - route
            start += 0.65 * released
            escape += 0.35 * released
            total = max(route + start + escape, 1e-8)
            return {
                "route": float(route / total),
                "start": float(start / total),
                "escape": float(escape / total),
            }

        # Before a strict hit, stagnation shifts seed restarts towards learned
        # frontier repair while retaining a smaller broadening arm.
        pressure = min(0.20, 0.02 * max(0, int(self.no_improve_iters)))
        route_share = 0.25 + 0.50 * tightness
        route_gain = route_share * pressure
        escape_gain = (1.0 - route_share) * pressure
        available = max(0.0, start - 0.05)
        scale = min(1.0, available / max(route_gain + escape_gain, 1e-8))
        route += route_gain * scale
        escape += escape_gain * scale
        start = 1.0 - route - escape
        return {
            "route": float(route),
            "start": float(start),
            "escape": float(escape),
        }

    def _elastic_joint_v5_legacy_fraction(self, state):
        """Keep a small fixed-length diversity arm at the strict threshold.

        Downloaded V3 trajectories show that learned progressive edits are
        roughly twice as likely as the fixed fallback to satisfy all public
        constraints at delta=0.6. The previous V5 allocation gave the weaker
        arm up to half of the proposal budget and compounded mode collapse.
        """
        loose = {
            "warmup": 0.0,
            "anchor": 0.0,
            "quality": 0.0,
            "joint_bridge": 0.0,
            "dock_completion": 0.0,
            "similarity_completion": 0.0,
            "quality_completion": 0.0,
            "polish": 0.0,
            "explore": 0.0,
        }[state]
        tight = {
            "warmup": 0.12,
            "anchor": 0.10,
            "quality": 0.12,
            "joint_bridge": 0.10,
            "dock_completion": 0.06,
            "similarity_completion": 0.08,
            "quality_completion": 0.10,
            "polish": 0.05,
            "explore": 0.12,
        }[state]
        tightness = self._elastic_joint_threshold_tightness()
        return float((1.0 - tightness) * loose + tightness * tight)

    def _elastic_joint_v5_plan_state(self, state, proposal_route):
        """Map all thresholds through one state machine."""
        tightness = self._elastic_joint_threshold_tightness()
        if self.elastic_joint_source == "public" and proposal_route != "route":
            # A single-axis near miss is useful as a route parent, but seed and
            # escape proposals must continue making joint, wider edits. The
            # old policy mapped almost every restart to the same micro-edit and
            # generated millions of canonical duplicates.
            return "warmup"
        if proposal_route == "escape":
            if state in {"quality", "quality_completion"}:
                return "quality"
            return "warmup" if tightness >= 0.5 else "explore"
        if state == "polish" and proposal_route == "start":
            return "anchor" if tightness >= 0.5 else "warmup"
        route_state = {
            "warmup": "warmup",
            "anchor": "anchor",
            "quality": "quality",
            "joint_bridge": "anchor" if tightness >= 0.5 else "explore",
            "dock_completion": "polish",
            "similarity_completion": "anchor",
            "quality_completion": "quality",
            "polish": "polish",
            "explore": "warmup" if tightness >= 0.5 else "explore",
        }[state]
        if (
            proposal_route == "start"
            and tightness >= 0.5
            and state not in {
                "quality",
                "quality_completion",
            }
        ):
            return "anchor"
        return route_state

    def _elastic_joint_v5_plan(self, smiles, state, proposal_route):
        """Create a broad learned edit, with the threshold acting as a prior.

        This restores the empirically successful V3 proposal family. Public
        feasibility remains a hard pre-docking gate, so broad proposals spend
        model compute but can never consume oracle budget unless they satisfy
        QED, SA and similarity simultaneously.
        """
        mol = Chem.MolFromSmiles(smiles)
        if mol is None:
            return None

        threshold = float(self.args.sim_thr)
        parent_fp = AllChem.GetMorganFingerprintAsBitVect(mol, 2, 2048)
        parent_similarity = float(
            DataStructs.TanimotoSimilarity(self.start_fp, parent_fp)
        )
        slack_ratio = clamp(
            (parent_similarity - threshold) / max(1.0 - threshold, 1e-8),
            0.0,
            1.0,
        )
        plan_state = self._elastic_joint_v5_plan_state(state, proposal_route)
        plan = self._elastic_direct_plan(smiles, plan_state)
        if plan is None:
            return None
        removed = max(1, int(plan["stop"]) - int(plan["start"]))
        plan.update(
            {
                "v5_parent_similarity": parent_similarity,
                "v5_similarity_slack_ratio": float(slack_ratio),
                "v5_trust_region_atoms": removed,
                "v5_edit_arm": "progressive_adaptive_span",
            }
        )
        return plan

    def _elastic_joint_v5_sampling_settings(self, plan_state):
        _, temperature, top_p = self._elastic_direct_sampling_settings(plan_state)
        tightness = self._elastic_joint_threshold_tightness()
        temperature_cap = (1.0 - tightness) * 1.08 + tightness * 0.98
        top_p_cap = (1.0 - tightness) * 0.80 + tightness * 0.72
        return min(float(temperature), temperature_cap), min(
            float(top_p),
            top_p_cap,
        )

    @staticmethod
    def _elastic_joint_v2_plan_state(state, proposal_route):
        if proposal_route == "escape":
            return "explore"
        route_state = {
            "warmup": "warmup",
            "anchor": "anchor",
            "quality": "quality",
            "joint_bridge": "explore",
            "dock_completion": "dock",
            "similarity_completion": "anchor",
            "quality_completion": "quality",
            "polish": "polish",
            "explore": "explore",
        }[state]
        if proposal_route != "start":
            return route_state
        # The supplied active is a similarity-safe restart. Keep its edits
        # conservative except during the initial warmup or explicit exploration.
        return {
            "joint_bridge": "warmup",
            "dock_completion": "polish",
            "explore": "warmup",
        }.get(state, route_state)

    def _elastic_joint_v3_plan_state(self, state, proposal_route):
        if float(self.args.sim_thr) < 0.55:
            return self._elastic_joint_v2_plan_state(state, proposal_route)
        if proposal_route == "escape":
            return "quality" if "quality" in state else "warmup"
        route_state = {
            "warmup": "warmup",
            "anchor": "anchor",
            "quality": "quality",
            "joint_bridge": "anchor",
            "dock_completion": "polish",
            "similarity_completion": "anchor",
            "quality_completion": "quality",
            "polish": "polish",
            "explore": "warmup",
        }[state]
        if proposal_route == "start" and state not in {
            "quality",
            "quality_completion",
        }:
            return "anchor"
        return route_state

    def _elastic_joint_v4_plan(self, smiles, state, proposal_route):
        """Make similarity-slack-aware trust-region edits for delta=0.6.

        The old high-similarity route reused the broad elastic-direct span
        planner and grew molecules by roughly 2.6 tokens on average. That is a
        poor proposal kernel when Tanimoto similarity itself is a hard
        constraint. Conversely, forcing every proposal to a single-token edit
        wastes the large trust-region slack near the supplied active and cannot
        repair very low-QED starts quickly enough. This planner spends at most
        three peripheral atoms according to the *observed* similarity slack,
        then contracts to one atom as a lineage approaches the boundary.
        """
        mol = Chem.MolFromSmiles(smiles)
        tokens = tokenize_smiles(smiles)
        if mol is None or not tokens:
            return None

        parent_fp = AllChem.GetMorganFingerprintAsBitVect(mol, 2, 2048)
        parent_similarity = float(
            DataStructs.TanimotoSimilarity(self.start_fp, parent_fp)
        )
        similarity_slack = max(
            0.0,
            parent_similarity - float(self.args.sim_thr),
        )
        edit_atoms = min(
            3,
            max(
                1,
                int(
                    math.floor(
                        similarity_slack * max(1, mol.GetNumAtoms()) * 0.25
                    )
                )
                + 1,
            ),
        )

        plan = None
        if (
            proposal_route == "route"
            and smiles != self.start_smiles
            and self.frontier_rng.random() < 0.35
        ):
            plan = seed_directed_atom_edit_plan(
                smiles,
                self.start_smiles,
                self.frontier_rng,
            )
        if plan is None:
            target_fraction = edit_atoms / max(1, mol.GetNumAtoms())
            plan = adaptive_peripheral_edit_plan(
                smiles,
                self.frontier_rng,
                delta=0,
                target_atom_fraction=target_fraction,
                max_atom_fraction=max(0.08, target_fraction),
                max_span_tokens=max(1, edit_atoms * 2),
            )
        if plan is None:
            plan = atom_span_edit_plan(
                smiles,
                self.frontier_rng,
                delta=0,
                span_tokens=edit_atoms,
            )
        if plan is None:
            return None

        learned_fraction = {
            "warmup": 0.20,
            "anchor": 0.08,
            "quality": 0.16,
            "joint_bridge": 0.14,
            "dock_completion": 0.08,
            "similarity_completion": 0.05,
            "quality_completion": 0.12,
            "polish": 0.05,
            "explore": 0.16,
        }[state]
        if self.frontier_rng.random() >= learned_fraction:
            fixed = dict(plan)
            fixed["v4_edit_arm"] = "fixed_micro"
            fixed["v4_parent_similarity"] = parent_similarity
            fixed["v4_trust_region_atoms"] = edit_atoms
            return fixed

        removed = max(1, int(plan["stop"]) - int(plan["start"]))
        learned = dict(plan)
        learned.update(
            {
                "length_mode": "learned_insertion",
                "min_replacement_len": max(0, removed - 1),
                "max_replacement_len": removed + 1,
                "initial_replacement_len": removed,
                "prior_guidance": "high_similarity_micro_trust_region",
                "v4_edit_arm": "learned_micro_resize",
                "v4_parent_similarity": parent_similarity,
                "v4_trust_region_atoms": edit_atoms,
            }
        )
        return learned

    def _elastic_joint_v2_draw_route(self, weights):
        draw = self.frontier_rng.random()
        cumulative = 0.0
        for route in ("route", "start", "escape"):
            cumulative += float(weights[route])
            if draw <= cumulative:
                return route
        return "escape"

    def _elastic_joint_v2_cheap_key(self, meta, state):
        """Rank without docking calls using only public Lead constraints."""
        qed = float(meta["cheap_qed"])
        sa = float(meta["cheap_sa"])
        similarity = float(meta["cheap_similarity"])
        sim_threshold = max(float(self.args.sim_thr), 1e-8)
        sim_deficit = max(0.0, sim_threshold - similarity) / sim_threshold
        qed_deficit = max(0.0, 0.6 - qed) / 0.6
        sa_threshold = 6.0 / 9.0
        sa_deficit = max(0.0, sa_threshold - sa) / sa_threshold
        quality_deficit = max(qed_deficit, sa_deficit)
        joint_deficit = max(sim_deficit, quality_deficit)
        mean_deficit = (sim_deficit + qed_deficit + sa_deficit) / 3.0
        feasible = joint_deficit <= 0.0
        tie = str(meta.get("smiles", ""))
        if state in {"anchor", "similarity_completion"}:
            return (
                int(not feasible),
                sim_deficit,
                quality_deficit,
                -similarity,
                -(qed + sa),
                tie,
            )
        if state in {"quality", "quality_completion"}:
            return (
                int(not feasible),
                quality_deficit,
                sim_deficit,
                -min(qed / 0.6, sa / sa_threshold),
                -similarity,
                tie,
            )
        return (
            int(not feasible),
            joint_deficit,
            mean_deficit,
            -min(qed / 0.6, sa / sa_threshold, similarity / sim_threshold),
            tie,
        )

    def _elastic_joint_v2_preselect(self, smiles_list, metadata, state, limit):
        if not smiles_list or limit <= 0:
            return [], []
        mols = [Chem.MolFromSmiles(smiles) for smiles in smiles_list]
        qeds = self.reward_qed(mols)
        sas = self.reward_sa(mols)
        similarities = self.reward_sim(mols)
        rows = []
        for smiles, meta, qed, sa, similarity in zip(
            smiles_list,
            metadata,
            qeds,
            sas,
            similarities,
        ):
            enriched = dict(meta)
            enriched.update(
                {
                    "smiles": smiles,
                    "cheap_qed": float(qed),
                    "cheap_sa": float(sa),
                    "cheap_similarity": float(similarity),
                    "cheap_feasible": bool(
                        qed >= 0.6
                        and sa >= 6.0 / 9.0
                        and similarity >= self.args.sim_thr
                    ),
                }
            )
            rows.append((smiles, enriched))
        rows.sort(key=lambda row: self._elastic_joint_v2_cheap_key(row[1], state))
        selected = rows[: int(limit)]
        return (
            [smiles for smiles, _ in selected],
            [meta for _, meta in selected],
        )

    def _elastic_joint_v3_preselect(self, smiles_list, metadata, state, limit):
        """Reserve oracle slots across the public Lead constraint frontier."""
        if float(self.args.sim_thr) < 0.55:
            return self._elastic_joint_v2_preselect(
                smiles_list,
                metadata,
                state,
                limit,
            )
        if not smiles_list or limit <= 0:
            return [], []

        mols = [Chem.MolFromSmiles(smiles) for smiles in smiles_list]
        qeds = self.reward_qed(mols)
        sas = self.reward_sa(mols)
        similarities = self.reward_sim(mols)
        sim_threshold = float(self.args.sim_thr)
        sa_threshold = 6.0 / 9.0
        sim_boundary = sim_threshold - max(0.04, 0.10 * sim_threshold)
        buckets = {
            "joint_feasible": [],
            "similarity_safe": [],
            "joint_boundary": [],
            "explore": [],
        }
        all_rows = []
        for smiles, meta, qed, sa, similarity in zip(
            smiles_list,
            metadata,
            qeds,
            sas,
            similarities,
        ):
            qed = float(qed)
            sa = float(sa)
            similarity = float(similarity)
            sim_deficit = max(0.0, sim_threshold - similarity) / max(
                sim_threshold,
                1e-8,
            )
            qed_deficit = max(0.0, 0.6 - qed) / 0.6
            sa_deficit = max(0.0, sa_threshold - sa) / sa_threshold
            quality_deficit = max(qed_deficit, sa_deficit)
            quality_safe = qed_deficit <= 0.0 and sa_deficit <= 0.0
            similarity_safe = sim_deficit <= 0.0
            feasible = quality_safe and similarity_safe
            if feasible:
                bucket = "joint_feasible"
            elif similarity_safe:
                bucket = "similarity_safe"
            elif similarity >= sim_boundary and (
                quality_safe or quality_deficit <= 0.15
            ):
                bucket = "joint_boundary"
            else:
                bucket = "explore"
            enriched = dict(meta)
            enriched.update(
                {
                    "smiles": smiles,
                    "cheap_qed": qed,
                    "cheap_sa": sa,
                    "cheap_similarity": similarity,
                    "cheap_sim_deficit": sim_deficit,
                    "cheap_quality_deficit": quality_deficit,
                    "cheap_feasible": feasible,
                    "cheap_bucket": bucket,
                }
            )
            row = (smiles, enriched)
            buckets[bucket].append(row)
            all_rows.append(row)

        def bucket_key(row):
            meta = row[1]
            bucket = meta["cheap_bucket"]
            if bucket == "similarity_safe":
                return (
                    meta["cheap_quality_deficit"],
                    -meta["cheap_similarity"],
                    self._elastic_joint_v2_cheap_key(meta, state),
                )
            if bucket == "joint_boundary":
                return (
                    max(
                        meta["cheap_sim_deficit"],
                        meta["cheap_quality_deficit"],
                    ),
                    meta["cheap_sim_deficit"] + meta["cheap_quality_deficit"],
                    self._elastic_joint_v2_cheap_key(meta, state),
                )
            return self._elastic_joint_v2_cheap_key(meta, state)

        shares = {
            "dock_completion": (0.75, 0.15, 0.08, 0.02),
            "polish": (0.75, 0.15, 0.08, 0.02),
            "similarity_completion": (0.45, 0.38, 0.12, 0.05),
            "anchor": (0.45, 0.38, 0.12, 0.05),
            "quality_completion": (0.55, 0.15, 0.25, 0.05),
            "quality": (0.45, 0.20, 0.28, 0.07),
            "warmup": (0.50, 0.25, 0.18, 0.07),
            "joint_bridge": (0.55, 0.22, 0.18, 0.05),
            "explore": (0.45, 0.25, 0.20, 0.10),
        }[state]
        names = (
            "joint_feasible",
            "similarity_safe",
            "joint_boundary",
            "explore",
        )
        raw_quotas = [float(limit) * share for share in shares]
        quotas = [int(math.floor(value)) for value in raw_quotas]
        for index in sorted(
            range(len(names)),
            key=lambda idx: raw_quotas[idx] - quotas[idx],
            reverse=True,
        )[: int(limit) - sum(quotas)]:
            quotas[index] += 1

        selected = []
        selected_smiles = set()
        for name, quota in zip(names, quotas):
            for row in sorted(buckets[name], key=bucket_key)[:quota]:
                selected.append(row)
                selected_smiles.add(row[0])
        if len(selected) < int(limit):
            remaining = [row for row in all_rows if row[0] not in selected_smiles]
            remaining.sort(
                key=lambda row: self._elastic_joint_v2_cheap_key(
                    row[1],
                    state,
                )
            )
            selected.extend(remaining[: int(limit) - len(selected)])
        return (
            [smiles for smiles, _ in selected],
            [meta for _, meta in selected],
        )

    def _elastic_joint_v4_preselect(self, smiles_list, metadata, state, limit):
        """Dock every joint-feasible proposal before spending on near misses."""
        if float(self.args.sim_thr) < 0.55:
            return self._elastic_joint_v2_preselect(
                smiles_list,
                metadata,
                state,
                limit,
            )
        _, enriched = self._elastic_joint_v3_preselect(
            smiles_list,
            metadata,
            state,
            len(smiles_list),
        )
        rows = [(row["smiles"], row) for row in enriched]
        feasible = [row for row in rows if row[1]["cheap_feasible"]]
        remainder = [row for row in rows if not row[1]["cheap_feasible"]]
        feasible.sort(key=lambda row: self._elastic_joint_v2_cheap_key(row[1], state))
        remainder.sort(key=lambda row: self._elastic_joint_v2_cheap_key(row[1], state))
        selected = (feasible + remainder)[: int(limit)]
        return (
            [smiles for smiles, _ in selected],
            [meta for _, meta in selected],
        )

    def _elastic_joint_v5_enrich(self, smiles_list, metadata):
        """Evaluate only public, inexpensive Lead constraints."""
        if not smiles_list:
            return []
        mols = [Chem.MolFromSmiles(smiles) for smiles in smiles_list]
        qeds = self.reward_qed(mols)
        sas = self.reward_sa(mols)
        similarities = self.reward_sim(mols)
        threshold = float(self.args.sim_thr)
        sa_threshold = 6.0 / 9.0
        rows = []
        for smiles, meta, qed, sa, similarity in zip(
            smiles_list,
            metadata,
            qeds,
            sas,
            similarities,
        ):
            qed = float(qed)
            sa = float(sa)
            similarity = float(similarity)
            sim_deficit = max(0.0, threshold - similarity) / max(threshold, 1e-8)
            qed_deficit = max(0.0, 0.6 - qed) / 0.6
            sa_deficit = max(0.0, sa_threshold - sa) / sa_threshold
            quality_deficit = max(qed_deficit, sa_deficit)
            feasible = (
                qed_deficit <= 0.0
                and sa_deficit <= 0.0
                and sim_deficit <= 0.0
            )
            enriched = dict(meta)
            enriched.update(
                {
                    "smiles": smiles,
                    "cheap_qed": qed,
                    "cheap_sa": sa,
                    "cheap_similarity": similarity,
                    "cheap_sim_deficit": sim_deficit,
                    "cheap_qed_deficit": qed_deficit,
                    "cheap_sa_deficit": sa_deficit,
                    "cheap_quality_deficit": quality_deficit,
                    "cheap_joint_deficit": max(sim_deficit, quality_deficit),
                    "cheap_feasible": feasible,
                    "cheap_bucket": (
                        "joint_feasible"
                        if feasible
                        else "similarity_safe"
                        if sim_deficit <= 0.0
                        else "quality_safe"
                        if quality_deficit <= 0.0
                        else "joint_boundary"
                    ),
                }
            )
            rows.append((smiles, enriched))
        return rows

    def _elastic_joint_v5_update_public_frontier(self, rows, state):
        """Retain near misses as free proposal parents, never oracle results."""
        candidates = dict(self.elastic_public_frontier)
        for smiles, meta in rows:
            if meta["cheap_feasible"]:
                continue
            previous = candidates.get(smiles)
            if previous is None or self._elastic_joint_v2_cheap_key(
                meta,
                state,
            ) < self._elastic_joint_v2_cheap_key(previous, state):
                candidates[smiles] = dict(meta)

        archive_size = max(1, int(getattr(self.args, "frontier_archive_size", 100)))
        ordered = sorted(
            candidates.values(),
            key=lambda meta: self._elastic_joint_v2_cheap_key(meta, state),
        )
        # Preserve several independent lineages before filling by global rank.
        # This is proposal diversity, not post-hoc selection on docking scores.
        root_cap = max(4, archive_size // 8)
        root_counts = defaultdict(int)
        selected = []
        deferred = []
        for meta in ordered:
            root = str(meta.get("root_id") or meta.get("parent_smiles") or "start")
            if root_counts[root] < root_cap:
                selected.append(meta)
                root_counts[root] += 1
            else:
                deferred.append(meta)
            if len(selected) >= archive_size:
                break
        if len(selected) < archive_size:
            selected.extend(deferred[: archive_size - len(selected)])
        self.elastic_public_frontier = {
            meta["smiles"]: meta for meta in selected[:archive_size]
        }

    def _elastic_joint_v5_public_parent_items(self):
        items = []
        for meta in self.elastic_public_frontier.values():
            item = self.frontier_make_item(
                meta["smiles"],
                self.start_prop,
                meta["cheap_qed"],
                meta["cheap_sa"],
                meta["cheap_similarity"],
            )
            item.update(
                {
                    "root_id": meta.get("root_id", "public"),
                    "depth": int(meta.get("depth", 0)),
                    "parent_smiles": meta.get("parent_smiles"),
                    "public_constraint_only": True,
                }
            )
            items.append(item)
        return items

    def _elastic_joint_v5_preselect(self, smiles_list, metadata, state, limit):
        """Return only jointly feasible proposals for docking."""
        rows = self._elastic_joint_v5_enrich(smiles_list, metadata)
        self._elastic_joint_v5_update_public_frontier(rows, state)
        feasible = [row for row in rows if row[1]["cheap_feasible"]]
        feasible.sort(key=lambda row: self._elastic_joint_v2_cheap_key(row[1], state))
        selected = feasible[: int(limit)]
        return (
            [smiles for smiles, _ in selected],
            [meta for _, meta in selected],
        )

    def generate_batch_elastic_joint_frontier_v2(
        self,
        constraint_guarded=False,
        final_v4=False,
        feasible_v5=False,
    ):
        """Constraint-state search over the proven elastic local trajectory."""
        state = self._elastic_joint_state()
        operator = {
            "warmup": "start_repair",
            "anchor": "similarity_repair",
            "quality": "quality_repair",
            "joint_bridge": "joint_repair",
            "dock_completion": "dock_refine",
            "similarity_completion": "similarity_repair",
            "quality_completion": "quality_repair",
            "polish": "dock_refine",
            "explore": "joint_repair",
        }[state]
        route_parents = self._elastic_joint_parent_pool(state)
        start_parent = self.frontier_items[self.start_smiles]
        if not route_parents:
            route_parents = [start_parent]
        escape_parents = [
            self.frontier_items[smiles]
            for smiles in self._elastic_direct_parent_pool("explore")
            if smiles in self.frontier_items
        ] or route_parents
        if feasible_v5:
            weights = self._elastic_joint_v5_route_weights(state)
        elif final_v4:
            weights = self._elastic_joint_v4_route_weights(state)
        elif constraint_guarded:
            weights = self._elastic_joint_v3_route_weights(state)
        else:
            weights = self._elastic_joint_v2_route_weights(state)

        seen = {self.start_smiles, *self.oracle_evaluated_smiles}
        if feasible_v5:
            seen.update(self.elastic_public_frontier)
        pool = []
        pool_meta = []
        proposal_inputs = 0
        raw_outputs = 0
        learned_outputs = 0
        fallback_outputs = 0
        route_draws = defaultdict(int)
        route_accepts = defaultdict(int)
        zero_rounds = 0
        rounds = 0
        proposal_input_cap = max(
            0,
            int(getattr(self.args, "max_proposal_inputs_per_iteration", 0)),
        )
        factor = max(1.0, float(self.args.direct_overgenerate_factor))
        pool_target = max(
            int(self.args.num_gen),
            int(math.ceil(self.args.num_gen * factor)),
        )
        legacy_outputs = 0
        legacy_fraction = (
            self._elastic_joint_v5_legacy_fraction(state)
            if feasible_v5
            else self._elastic_joint_v4_legacy_fraction(state)
            if final_v4
            else 0.0
        )
        if legacy_fraction > 0.0:
            legacy_target = min(
                pool_target,
                max(1, int(round(pool_target * legacy_fraction))),
            )
            legacy_pool, legacy_meta = self.frontier_generate_operator(
                "legacy",
                legacy_target,
                seen,
            )
            for meta in legacy_meta:
                meta.update(
                    {
                        "proposal_route": "legacy_fixed",
                        "plan_state": "legacy_fixed",
                        "planned_learned_insertion": False,
                        "v4_edit_arm": "legacy_fixed",
                        "v5_edit_arm": "legacy_fixed",
                    }
                )
            pool.extend(legacy_pool)
            pool_meta.extend(legacy_meta)
            legacy_outputs = len(legacy_pool)

        zero_round_patience = (
            int(self.args.max_generation_rounds)
            if final_v4 or feasible_v5
            else 4
            if constraint_guarded
            else 2
        )
        while (
            len(pool) < pool_target
            and rounds < self.args.max_generation_rounds
            and zero_rounds < zero_round_patience
            and (
                proposal_input_cap <= 0
                or proposal_inputs < proposal_input_cap
            )
        ):
            rounds += 1
            proposal_count = pool_target - len(pool)
            if proposal_input_cap > 0:
                proposal_count = min(
                    proposal_count,
                    proposal_input_cap - proposal_inputs,
                )
            if proposal_count <= 0:
                break
            grouped = defaultdict(lambda: {"seeds": [], "plans": [], "meta": []})
            for _ in range(proposal_count):
                proposal_route = self._elastic_joint_v2_draw_route(weights)
                if proposal_route == "start":
                    parent = start_parent
                else:
                    parents = (
                        escape_parents if proposal_route == "escape" else route_parents
                    )
                    rate = 0.18 if proposal_route == "escape" else 0.38
                    rank = min(
                        len(parents) - 1,
                        int(self.frontier_rng.expovariate(rate)),
                    )
                    parent = parents[rank]
                plan_state = (
                    self._elastic_joint_v5_plan_state(state, proposal_route)
                    if feasible_v5
                    else self._elastic_joint_v3_plan_state(state, proposal_route)
                    if constraint_guarded
                    else self._elastic_joint_v2_plan_state(
                        state,
                        proposal_route,
                    )
                )
                plan = (
                    self._elastic_joint_v5_plan(
                        parent["smiles"],
                        state,
                        proposal_route,
                    )
                    if feasible_v5
                    else self._elastic_joint_v4_plan(
                        parent["smiles"],
                        state,
                        proposal_route,
                    )
                    if final_v4 and float(self.args.sim_thr) >= 0.55
                    else self._elastic_direct_plan(parent["smiles"], plan_state)
                )
                if plan is None:
                    continue
                route_draws[proposal_route] += 1
                group = grouped[plan_state]
                group["seeds"].append(parent["smiles"])
                group["plans"].append(plan)
                group["meta"].append(
                    {
                        "operator": operator,
                        "parent_smiles": parent["smiles"],
                        "parent_residual": parent["residual"],
                        "parent_stage": parent["stage"],
                        "parent_token_len": len(tokenize_smiles(parent["smiles"])),
                        "root_id": parent.get("root_id", "start"),
                        "depth": int(parent.get("depth", 0)) + 1,
                        "search_state": state,
                        "source_frontier": self.elastic_joint_source,
                        "proposal_route": proposal_route,
                        "plan_state": plan_state,
                        "planned_learned_insertion": bool(
                            plan.get("length_mode") == "learned_insertion"
                        ),
                        "v4_edit_arm": plan.get("v4_edit_arm"),
                        "v4_parent_similarity": plan.get("v4_parent_similarity"),
                        "v4_trust_region_atoms": plan.get(
                            "v4_trust_region_atoms"
                        ),
                        "v5_edit_arm": plan.get("v5_edit_arm"),
                        "v5_parent_similarity": plan.get(
                            "v5_parent_similarity"
                        ),
                        "v5_similarity_slack_ratio": plan.get(
                            "v5_similarity_slack_ratio"
                        ),
                        "v5_trust_region_atoms": plan.get(
                            "v5_trust_region_atoms"
                        ),
                        "peripheral": bool(plan.get("peripheral", False)),
                        "edit_strategy": plan.get("edit_strategy"),
                    }
                )

            accepted_before = len(pool)
            for plan_state, group in grouped.items():
                if not group["seeds"]:
                    continue
                if feasible_v5:
                    temperature, top_p = self._elastic_joint_v5_sampling_settings(
                        plan_state
                    )
                else:
                    _, temperature, top_p = self._elastic_direct_sampling_settings(
                        plan_state
                    )
                if (
                    constraint_guarded
                    and not feasible_v5
                    and float(self.args.sim_thr) >= 0.55
                ):
                    temperature = min(float(temperature), 0.95)
                    top_p = min(float(top_p), 0.60)
                proposal_inputs += len(group["seeds"])
                generated = sample_csdnet_local_remask(
                    model=self.model,
                    tk=self.tk,
                    seed_smiles=group["seeds"],
                    edit_plans=group["plans"],
                    max_len=self.args.max_len,
                    device=self.device,
                    batch_size=self.args.batch_size,
                    n_steps=self.args.n_steps,
                    use_fsm_check=not self.args.disable_fsm_check,
                    use_rdkit_kekulize_check=(
                        not self.args.disable_rdkit_kekulize_check
                    ),
                    max_sample_retries=self.args.max_sample_retries,
                    violation_neighborhood=self.args.violation_neighborhood,
                    temperature_start=temperature,
                    temperature_end=max(
                        0.20,
                        min(0.35, temperature * 0.35),
                    ),
                    temperature_power=1.2,
                    top_k=0,
                    top_p=top_p,
                    gumbel_scale=0.55,
                    remask_power=1.0,
                    learned_insertion_max_per_step=(
                        self.args.learned_insertion_max_per_step
                    ),
                    learned_insertion_fallback=True,
                    learned_insertion_fallback_fraction=(
                        1.0 if final_v4 or feasible_v5 else 0.20
                    ),
                    learned_insertion_recursive_gap_insertions=True,
                    learned_insertion_trajectory_mode="plan_then_fill",
                    learned_insertion_planning_fraction=0.30,
                    learned_insertion_fill_mode="progressive_remask",
                    learned_insertion_fill_remask_power=0.85,
                    learned_insertion_fill_gumbel_scale=0.35,
                    learned_insertion_nucleus_min_tokens_start=3,
                    learned_insertion_nucleus_min_tokens_end=1,
                    learned_insertion_unmask_selection="top_prob",
                    learned_insertion_deterministic_final_unmask=True,
                    learned_insertion_fsm_repair_steps=8,
                    learned_insertion_fsm_prefer_localization=True,
                    local_sampler_profile="legacy",
                    return_seed_indices=True,
                    return_diagnostics=True,
                )
                raw_outputs += len(generated)
                for smiles, seed_index, diagnostics in generated:
                    can = canonical_smiles(smiles)
                    if (
                        can is None
                        or can in seen
                        or not tokenizable(can, self.tk, self.args.max_len)
                        or not self.args.min_atoms
                        <= atom_count(can)
                        <= self.args.max_atoms
                    ):
                        continue
                    seed_index = int(seed_index)
                    if not 0 <= seed_index < len(group["meta"]):
                        continue
                    seen.add(can)
                    pool.append(can)
                    meta = dict(group["meta"][seed_index])
                    mode = diagnostics.get("length_mode", "fixed")
                    meta.update(
                        {
                            "length_mode": mode,
                            "removed_tokens": diagnostics.get("removed_tokens"),
                            "inserted_tokens": diagnostics.get("inserted_tokens"),
                            "actual_delta": diagnostics.get("actual_delta"),
                            "initial_mask_tokens": diagnostics.get(
                                "initial_inserted_tokens"
                            ),
                            "learned_inserted_tokens": diagnostics.get(
                                "learned_inserted_tokens", 0
                            ),
                            "insertion_steps": diagnostics.get("insertion_steps", 0),
                            "fsm_repair_progressive_steps": diagnostics.get(
                                "fsm_repair_progressive_steps"
                            ),
                            "fsm_repair_prefer_localization": diagnostics.get(
                                "fsm_repair_prefer_localization"
                            ),
                            "fsm_constraint_mode": diagnostics.get(
                                "fsm_constraint_mode"
                            ),
                            "fsm_check_enabled": diagnostics.get("fsm_check_enabled"),
                            "rdkit_kekulize_check_enabled": diagnostics.get(
                                "rdkit_kekulize_check_enabled"
                            ),
                            "online_fsm_repair_events": diagnostics.get(
                                "online_fsm_repair_events", 0
                            ),
                            "online_fsm_remasked_tokens": diagnostics.get(
                                "online_fsm_remasked_tokens", 0
                            ),
                        }
                    )
                    pool_meta.append(meta)
                    route_accepts[meta["proposal_route"]] += 1
                    if is_learned_length_mode(mode):
                        learned_outputs += 1
                    elif mode == "fixed_fallback":
                        fallback_outputs += 1
                    if len(pool) >= pool_target:
                        break
                if len(pool) >= pool_target:
                    break
            if len(pool) == accepted_before:
                zero_rounds += 1
            else:
                zero_rounds = 0

        output, metadata = (
            self._elastic_joint_v5_preselect(
                pool,
                pool_meta,
                state,
                self.args.num_gen,
            )
            if feasible_v5
            else self._elastic_joint_v4_preselect(
                pool,
                pool_meta,
                state,
                self.args.num_gen,
            )
            if final_v4
            else self._elastic_joint_v3_preselect(
                pool,
                pool_meta,
                state,
                self.args.num_gen,
            )
            if constraint_guarded
            else self._elastic_joint_v2_preselect(
                pool,
                pool_meta,
                state,
                self.args.num_gen,
            )
        )
        self.current_candidate_meta = metadata
        self.current_candidate_ops = [item["operator"] for item in metadata]
        cheap_feasible = sum(item["cheap_feasible"] for item in metadata)
        weights_summary = ",".join(
            f"{name}:{weights[name]:.2f}" for name in ("route", "start", "escape")
        )
        draws_summary = ",".join(
            f"{name}:{route_draws[name]}" for name in ("route", "start", "escape")
        )
        accepts_summary = ",".join(
            f"{name}:{route_accepts[name]}" for name in ("route", "start", "escape")
        )
        bucket_counts = defaultdict(int)
        for item in metadata:
            bucket_counts[item.get("cheap_bucket", "ranked")] += 1
        buckets_summary = ",".join(
            f"{name}:{bucket_counts[name]}"
            for name in (
                "joint_feasible",
                "similarity_safe",
                "joint_boundary",
                "explore",
                "ranked",
            )
            if bucket_counts[name]
        )
        version = (
            "v5"
            if feasible_v5
            else "v4"
            if final_v4
            else "v3"
            if constraint_guarded
            else "v2"
        )
        print(
            f"[Elastic-joint-{version}] state={state} "
            f"source={self.elastic_joint_source} "
            f"route_gap={self.elastic_joint_route_score} "
            f"selected={len(output)}/{self.args.num_gen} "
            f"pool={len(pool)}/{pool_target} rounds={rounds} "
            f"proposal_inputs={proposal_inputs}/"
            f"{proposal_input_cap or 'unlimited'} raw_outputs={raw_outputs} "
            f"legacy_fixed={legacy_outputs}/{legacy_fraction:.2f} "
            f"learned={learned_outputs} fallback={fallback_outputs} "
            "fsm_repair=online-syntax+valence/localized/8step+projection "
            f"cheap_feasible={cheap_feasible}/{len(output)} "
            f"public_frontier={len(self.elastic_public_frontier)} "
            f"buckets={buckets_summary or 'none'} "
            f"weights={weights_summary} draws={draws_summary} "
            f"accepted={accepts_summary}"
        )
        return output

    def generate_batch_elastic_joint_frontier_v3(self):
        """High-similarity guarded frontier with the complete FSM pipeline."""
        return self.generate_batch_elastic_joint_frontier_v2(constraint_guarded=True)

    def generate_batch_elastic_joint_frontier_v4(self):
        """Final threshold-specific Lead kernel with exhaustive fixed fallback."""
        return self.generate_batch_elastic_joint_frontier_v2(
            constraint_guarded=True,
            final_v4=True,
        )

    def generate_batch_elastic_joint_frontier_v5(self):
        """Public-frontier search with a strictly feasible docking queue."""
        return self.generate_batch_elastic_joint_frontier_v2(
            constraint_guarded=True,
            feasible_v5=True,
        )

    def _elastic_direct_state(self):
        rows = [
            values
            for smiles, values in self.evaluation_cache.items()
            if smiles != self.start_smiles
        ]
        if not rows:
            return "warmup"
        sim_rate = np.mean([row[3] >= self.args.sim_thr for row in rows])
        quality_rate = np.mean([row[1] >= 0.6 and row[2] >= 6.0 / 9.0 for row in rows])
        sim_quality_rate = np.mean(
            [
                row[3] >= self.args.sim_thr and row[1] >= 0.6 and row[2] >= 6.0 / 9.0
                for row in rows
            ]
        )
        strict = any(
            row[0] > self.start_prop
            and row[1] >= 0.6
            and row[2] >= 6.0 / 9.0
            and row[3] >= self.args.sim_thr
            for row in rows
        )
        if strict:
            return "polish"
        if sim_rate < 0.18:
            return "anchor"
        if sim_quality_rate >= 0.08:
            return "dock"
        if quality_rate < 0.20:
            return "quality"
        return "explore"

    def _elastic_direct_parent_pool(self, state):
        items = []
        for smiles, (dock, qed, sa, sim) in self.evaluation_cache.items():
            sim_deficit = max(0.0, self.args.sim_thr - sim) / max(
                self.args.sim_thr, 1e-8
            )
            quality_deficit = max(0.0, 0.6 - qed) / 0.6 + max(0.0, 6.0 / 9.0 - sa) / (
                6.0 / 9.0
            )
            feasible = sim >= self.args.sim_thr and qed >= 0.6 and sa >= 6.0 / 9.0
            strict = feasible and dock > self.start_prop
            dock_ratio = dock / max(self.start_prop, 1e-8)
            score = (
                4.0 * strict
                + 2.0 * feasible
                + min(2.0, dock_ratio)
                + 0.35 * sim
                - 1.4 * sim_deficit
                - 0.8 * quality_deficit
            )
            if state == "anchor":
                score += 1.5 * sim
            elif state == "quality":
                score += 0.8 * (qed + sa)
            items.append((float(score), smiles))
        items.sort(reverse=True)
        return [smiles for _, smiles in items[: self.args.direct_parent_pool_size]]

    def _elastic_direct_prefill(
        self,
        *,
        removed,
        fixed_length,
        minimum,
        maximum,
        state,
    ):
        """Blend span retention with a soft atomic-length prior.

        Initial masks cannot be deleted by the reverse process, so prefill is
        deliberately below the intended replacement length. Learned insertion
        remains responsible for deciding whether to stop short, restore the
        removed span, or grow beyond it.
        """
        settings = {
            "warmup": (0.25, (0.55, 0.85), (0.25, 0.60)),
            "anchor": (0.10, (0.75, 0.98), (0.30, 0.62)),
            "quality": (0.25, (0.50, 0.82), (0.20, 0.55)),
            "dock": (0.30, (0.38, 0.72), (0.20, 0.62)),
            "polish": (0.08, (0.82, 1.00), (0.35, 0.65)),
            "explore": (0.40, (0.25, 0.65), (0.15, 0.62)),
        }
        prior_weight, retention_range, quantile_range = settings[state]
        allowed = self.elastic_atomic_body_lengths[
            (self.elastic_atomic_body_lengths >= fixed_length + minimum)
            & (self.elastic_atomic_body_lengths <= fixed_length + maximum)
        ]
        if allowed.size:
            quantile = self.frontier_rng.uniform(*quantile_range)
            prior_gap = float(np.quantile(allowed - fixed_length, quantile))
        else:
            prior_gap = float(clamp(removed, minimum, maximum))
        target_gap = clamp(
            (1.0 - prior_weight) * removed + prior_weight * prior_gap,
            minimum,
            maximum,
        )
        retention = self.frontier_rng.uniform(*retention_range)
        initial = minimum + retention * max(0.0, target_gap - minimum)
        return int(clamp(round(initial), minimum, maximum))

    def _elastic_direct_plan(self, smiles, state):
        tokens = tokenize_smiles(smiles)
        if not tokens:
            return None
        settings = {
            "warmup": (0.08, 0.18, 0.18),
            "anchor": (0.05, 0.11, 0.12),
            "quality": (0.07, 0.16, 0.18),
            "dock": (0.11, 0.24, 0.25),
            "polish": (0.04, 0.10, 0.12),
            "explore": (0.15, 0.30, 0.30),
        }
        fraction_low, fraction_high, flex_fraction = settings[state]
        target_fraction = self.frontier_rng.uniform(fraction_low, fraction_high)
        plan = None
        if self.frontier_rng.random() < self.args.direct_peripheral_probability:
            plan = adaptive_peripheral_edit_plan(
                smiles,
                self.frontier_rng,
                target_atom_fraction=target_fraction,
                max_atom_fraction=min(0.45, fraction_high + 0.12),
                max_span_tokens=max(2, int(math.ceil(len(tokens) * 0.40))),
            )
        if plan is None:
            span_tokens = max(1, int(round(len(tokens) * target_fraction)))
            plan = atom_span_edit_plan(
                smiles,
                self.frontier_rng,
                span_tokens=span_tokens,
            )
        if plan is None:
            return None

        start = int(plan["start"])
        stop = int(plan["stop"])
        removed = max(1, stop - start)
        parent_length = len(tokens)
        prior_low, prior_high = self.elastic_length_support
        # The empirical prior defines a trusted support interval. The current
        # lead is always admitted and the interval has a small margin, so an
        # unusual supplied active is not forced abruptly towards ZINC.
        start_length = len(tokenize_smiles(self.start_smiles))
        support_low = max(3, min(prior_low, start_length) - 4)
        support_high = min(
            self.args.max_len - 2,
            max(prior_high, start_length) + 4,
        )
        max_change = max(2, int(math.ceil(parent_length * flex_fraction)))
        final_low = max(3, support_low, parent_length - max_change)
        final_high = min(
            self.args.max_len - 2,
            support_high,
            parent_length + max_change,
        )
        fixed_length = parent_length - removed
        minimum = max(0, final_low - fixed_length)
        maximum = max(minimum, final_high - fixed_length)
        initial = self._elastic_direct_prefill(
            removed=removed,
            fixed_length=fixed_length,
            minimum=minimum,
            maximum=maximum,
            state=state,
        )
        return {
            "start": start,
            "stop": stop,
            "length_mode": "learned_insertion",
            "min_replacement_len": int(minimum),
            "max_replacement_len": int(maximum),
            "initial_replacement_len": int(min(initial, maximum)),
            "prior_guidance": "atomic_soft_span_retention",
            "peripheral": bool(plan.get("peripheral", False)),
            "edit_strategy": plan.get("edit_strategy"),
        }

    def generate_batch_elastic_direct(self):
        state = self._elastic_direct_state()
        start_probability, temperature, top_p = self._elastic_direct_sampling_settings(
            state
        )
        parents = self._elastic_direct_parent_pool(state)
        if not parents:
            parents = [self.start_smiles]

        seen = set()
        output = []
        metadata = []
        rounds = 0
        while (
            len(output) < self.args.num_gen and rounds < self.args.max_generation_rounds
        ):
            rounds += 1
            remaining = self.args.num_gen - len(output)
            proposal_count = max(
                remaining,
                int(math.ceil(remaining * self.args.direct_overgenerate_factor)),
            )
            seeds = []
            plans = []
            for _ in range(proposal_count):
                if self.frontier_rng.random() < start_probability:
                    parent = self.start_smiles
                else:
                    rank = min(
                        len(parents) - 1,
                        int(self.frontier_rng.expovariate(0.32)),
                    )
                    parent = parents[rank]
                plan = self._elastic_direct_plan(parent, state)
                if plan is None:
                    continue
                seeds.append(parent)
                plans.append(plan)
            if not seeds:
                break

            generated = sample_elastic_local_infill(
                model=self.model,
                tk=self.tk,
                seed_smiles=seeds,
                edit_plans=plans,
                max_len=self.args.max_len,
                device=self.device,
                batch_size=self.args.batch_size,
                n_steps=self.args.n_steps,
                use_fsm_check=not self.args.disable_fsm_check,
                use_rdkit_kekulize_check=not self.args.disable_rdkit_kekulize_check,
                max_sample_retries=self.args.max_sample_retries,
                violation_neighborhood=self.args.violation_neighborhood,
                temperature_start=temperature,
                temperature_end=max(0.50, temperature * 0.70),
                temperature_power=1.0,
                top_k=0,
                top_p=top_p,
                nucleus_min_tokens_start=3,
                nucleus_min_tokens_end=1,
                max_insertions_per_step=self.args.learned_insertion_max_per_step,
                insertion_rate_scale=1.0,
                unmask_selection="top_prob",
                deterministic_final_unmask=True,
                recursive_gap_insertions=True,
                trajectory_mode="coupled",
                fill_mode="absorbing",
                return_seed_indices=True,
                return_diagnostics=True,
            )
            for smiles, seed_index, diagnostics in generated:
                can = canonical_smiles(smiles)
                if (
                    can is None
                    or can in seen
                    or can in self.oracle_evaluated_smiles
                    or not tokenizable(can, self.tk, self.args.max_len)
                ):
                    continue
                if not self.args.min_atoms <= atom_count(can) <= self.args.max_atoms:
                    continue
                seen.add(can)
                output.append(can)
                metadata.append(
                    {
                        "operator": f"elastic_{state}",
                        "parent_smiles": seeds[int(seed_index)],
                        "length_mode": diagnostics.get("length_mode"),
                        "removed_tokens": diagnostics.get("removed_tokens"),
                        "inserted_tokens": diagnostics.get("inserted_tokens"),
                        "actual_delta": diagnostics.get("actual_delta"),
                        "initial_mask_tokens": diagnostics.get(
                            "initial_inserted_tokens"
                        ),
                    }
                )
                if len(output) >= self.args.num_gen:
                    break
        self.current_candidate_meta = metadata
        self.current_candidate_ops = [item["operator"] for item in metadata]
        print(
            f"[Elastic-direct] state={state} generated={len(output)}/"
            f"{self.args.num_gen} rounds={rounds} parents={len(parents)} "
            f"start_prob={start_probability:.2f} top_p={top_p:.2f} "
            f"mean_initial_masks={np.mean([row['initial_mask_tokens'] for row in metadata]):.2f}"
            if metadata
            else (
                f"[Elastic-direct] state={state} generated=0/"
                f"{self.args.num_gen} rounds={rounds} parents={len(parents)}"
            )
        )
        return output

    def generate_batch(self):
        self.current_candidate_ops = []
        self.current_candidate_meta = []
        if self.args.sampler_profile == "elastic_joint_frontier_v5":
            return self.generate_batch_elastic_joint_frontier_v5()
        if self.args.sampler_profile == "elastic_joint_frontier_v4":
            return self.generate_batch_elastic_joint_frontier_v4()
        if self.args.sampler_profile == "elastic_joint_frontier_v3":
            return self.generate_batch_elastic_joint_frontier_v3()
        if self.args.sampler_profile == "elastic_joint_frontier_v2":
            return self.generate_batch_elastic_joint_frontier_v2()
        if self.args.sampler_profile == "elastic_joint_frontier":
            return self.generate_batch_elastic_joint_frontier()
        if self.args.sampler_profile == "elastic_direct":
            return self.generate_batch_elastic_direct()
        if self.args.sampler_profile in FRONTIER_PROFILES:
            return self.generate_batch_multi_frontier()
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

    def filter_oracle_novel_candidates(self, smiles_list):
        """Remove canonical molecules already sent to the docking oracle."""
        if not self.args.global_oracle_dedup:
            return smiles_list

        kept = []
        kept_meta = []
        kept_ops = []
        batch_seen = set()
        skipped = 0
        for index, smiles in enumerate(smiles_list):
            can = canonical_smiles(smiles)
            if can is None or can in self.oracle_evaluated_smiles or can in batch_seen:
                skipped += 1
                continue
            batch_seen.add(can)
            kept.append(can)
            if index < len(self.current_candidate_meta):
                kept_meta.append(self.current_candidate_meta[index])
            if index < len(self.current_candidate_ops):
                kept_ops.append(self.current_candidate_ops[index])

        if self.current_candidate_meta:
            self.current_candidate_meta = kept_meta
        if self.current_candidate_ops:
            self.current_candidate_ops = kept_ops
        if skipped:
            print(
                f"[Oracle-dedup] skipped={skipped} novel={len(kept)}/"
                f"{len(smiles_list)} previously_evaluated="
                f"{len(self.oracle_evaluated_smiles)}"
            )
        return kept

    def filter_oracle_feasible_candidates(self, smiles_list):
        """Keep only molecules satisfying all public Lead constraints.

        QED, SA and similarity are deterministic public filters, not docking
        oracle calls. This final guard is intentionally repeated after sampler
        preselection so no alternative proposal route can leak an infeasible
        molecule into the expensive oracle queue.
        """
        if not getattr(self.args, "oracle_feasible_only", False) or not smiles_list:
            return smiles_list

        mols = [Chem.MolFromSmiles(smiles) for smiles in smiles_list]
        qeds = self.reward_qed(mols)
        sas = self.reward_sa(mols)
        similarities = self.reward_sim(mols)
        kept = []
        kept_meta = []
        kept_ops = []
        rejected_qed = 0
        rejected_sa = 0
        rejected_similarity = 0
        for index, (smiles, qed, sa, similarity) in enumerate(
            zip(smiles_list, qeds, sas, similarities)
        ):
            qed_ok = float(qed) >= 0.6
            sa_ok = float(sa) >= 6.0 / 9.0
            similarity_ok = float(similarity) >= float(self.args.sim_thr)
            rejected_qed += int(not qed_ok)
            rejected_sa += int(not sa_ok)
            rejected_similarity += int(not similarity_ok)
            if not (qed_ok and sa_ok and similarity_ok):
                continue
            kept.append(smiles)
            if index < len(self.current_candidate_meta):
                meta = dict(self.current_candidate_meta[index])
                meta.update(
                    {
                        "cheap_qed": float(qed),
                        "cheap_sa": float(sa),
                        "cheap_similarity": float(similarity),
                        "cheap_feasible": True,
                    }
                )
                kept_meta.append(meta)
            if index < len(self.current_candidate_ops):
                kept_ops.append(self.current_candidate_ops[index])

        if self.current_candidate_meta:
            self.current_candidate_meta = kept_meta
        if self.current_candidate_ops:
            self.current_candidate_ops = kept_ops
        rejected = len(smiles_list) - len(kept)
        if rejected:
            print(
                f"[Oracle-feasibility-gate] rejected={rejected} "
                f"kept={len(kept)}/{len(smiles_list)} "
                f"qed_fail={rejected_qed} sa_fail={rejected_sa} "
                f"similarity_fail={rejected_similarity}"
            )
        return kept

    def limit_oracle_budget_candidates(self, smiles_list, total):
        """Trim a batch so every Lead run respects the official call cap."""
        remaining = max(0, int(self.args.oracle_budget) - int(total))
        if len(smiles_list) <= remaining:
            return smiles_list

        kept = smiles_list[:remaining]
        if self.current_candidate_meta:
            self.current_candidate_meta = self.current_candidate_meta[:remaining]
        if self.current_candidate_ops:
            self.current_candidate_ops = self.current_candidate_ops[:remaining]
        print(
            f"[Oracle-budget] trimmed={len(smiles_list) - len(kept)} "
            f"kept={len(kept)} remaining={remaining} "
            f"limit={self.args.oracle_budget}"
        )
        return kept

    def budget_completion_decision(self):
        """Choose continuation or a seed-anchored restart from observed state."""
        strict = [
            item
            for item in self.frontier_archives.get("strict", [])
            if item["smiles"] != self.start_smiles
        ]
        sim_quality = [
            item
            for item in self.frontier_archives.get("sq", [])
            if item["smiles"] != self.start_smiles
        ]
        best_dock_ratio = (
            max(item["dock"] for item in sim_quality)
            / max(float(self.start_prop), 1e-8)
            if sim_quality
            else 0.0
        )
        population_growth = max(
            0,
            len(self.population) - int(self.initial_population_size),
        )
        if strict:
            route = "frontier"
            reason = "strict_candidate_available"
        elif sim_quality and best_dock_ratio >= self.args.budget_close_dock_ratio:
            route = "frontier"
            reason = "near_docking_boundary"
        else:
            route = "restart"
            reason = (
                "no_similarity_quality_frontier"
                if not sim_quality
                else "docking_response_outside_near_domain"
            )
        return {
            "route": route,
            "reason": reason,
            "best_dock_ratio": float(best_dock_ratio),
            "sim_quality_count": len(sim_quality),
            "strict_count": len(strict),
            "population_growth": population_growth,
        }

    def generate_budget_completion_batch(self, route, target_n):
        """Generate one bounded batch without changing the baseline policy."""
        target_n = max(0, int(target_n))
        if target_n == 0:
            return []

        if route == "frontier":
            original_num_gen = self.args.num_gen
            self.args.num_gen = target_n
            try:
                return self.generate_batch()
            finally:
                self.args.num_gen = original_num_gen

        operator = "no_pair_anchor_restart"
        self.frontier_operator_stats.setdefault(
            operator,
            {
                "attempted": 0.0,
                "accepted": 0.0,
                "evaluated": 0.0,
                "updates": 0.0,
                "reward_ema": 0.0,
                "last_batch_reward": 0.0,
                "strict": 0.0,
            },
        )
        seen = set(self.oracle_evaluated_smiles)
        generated, metadata = self.frontier_generate_operator(
            operator,
            target_n,
            seen,
        )
        self.current_candidate_meta = metadata
        self.current_candidate_ops = [operator] * len(generated)
        self.frontier_last_allocation = {operator: target_n}
        print(
            f"[Budget-restart] source=start_anchor accepted={len(generated)}/{target_n}"
        )
        return generated

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
            remask_fraction, temp_start, span_prob, length_kwargs = (
                self.residual_length_operator_params(operator)
            )
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
            remask_fraction, temp_start, span_prob = self.ladder_operator_params(
                operator
            )
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
        for rv, rq, rs, rsim, smiles in zip(
            rv_list, rq_list, rs_list, rsim_list, smiles_list
        ):
            if rv < self.start_prop * self.args.pareto_min_docking_ratio:
                continue
            if (
                rq < self.args.pareto_min_qed
                or rs < self.args.pareto_min_sa
                or rsim < sim_floor
            ):
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
        for rv, rq, rs, rsim, smiles in zip(
            rv_list, rq_list, rs_list, rsim_list, smiles_list
        ):
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
                "pareto_score": float(
                    0.65 * dock_norm + 0.25 * rsim + 0.05 * rq + 0.05 * rs
                ),
            }
            if (
                rv >= self.start_prop * self.args.v4_explore_min_docking_ratio
                and rsim >= sim_floor
            ):
                docking_items.append(item)
            if (
                rsim >= self.args.sim_thr
                and rv >= self.start_prop * self.args.v4_high_sim_min_docking_ratio
            ):
                sim_items.append(item)

        docking_items.sort(
            key=lambda item: (item["dock_norm"], item["sim"]), reverse=True
        )
        sim_items.sort(key=lambda item: (item["sim"], item["dock_norm"]), reverse=True)
        selected = (
            docking_items[: self.args.v4_explore_top_k]
            + sim_items[: self.args.v4_explore_top_k]
        )
        dedup = {}
        for item in selected:
            dedup.setdefault(item["smiles"], item)
        return list(dedup.values())

    def _constraint_items(self, smiles_list, prop_list):
        rv_list, rq_list, rs_list, rsim_list = prop_list
        items = []
        sim_soft_floor = max(
            0.0, self.args.sim_thr - self.args.constraint_similarity_slack
        )
        for rv, rq, rs, rsim, smiles in zip(
            rv_list, rq_list, rs_list, rsim_list, smiles_list
        ):
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
                        0.35 * dock_norm
                        + 0.25 * sim_norm
                        + 0.25 * qed_norm
                        + 0.15 * sa_norm
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
            key=lambda item: (
                item["near_score"],
                item["quality_score"],
                item["dock_norm"],
            ),
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
            items = self.residual_items(
                smiles_list, prop_list, self.current_candidate_ops
            )
        else:
            items = self._constraint_items(smiles_list, prop_list)
        if not items:
            return
        archive_size = self.args.constraint_archive_size
        dock_items = [
            item
            for item in items
            if item["soft_sim_ok"]
            and item["dock"]
            >= self.start_prop * self.args.constraint_dock_archive_min_ratio
        ]
        quality_items = [
            item for item in items if item["soft_sim_ok"] and item["soft_quality_ok"]
        ]
        near_items = [
            item
            for item in items
            if (
                item["near_score"] >= self.args.constraint_near_miss_min_score
                or item.get("residual_score", -float("inf"))
                >= self.args.residual_archive_min_score
            )
        ]
        diverse_items = [
            item
            for item in items
            if (
                item["near_score"] >= self.args.constraint_near_miss_min_score * 0.85
                or item.get("residual_score", -float("inf"))
                >= self.args.residual_archive_min_score * 0.92
            )
        ]
        self._merge_archive(self.dock_elite, dock_items, "dock", archive_size)
        self._merge_archive(
            self.quality_elite, quality_items, "pareto_score", archive_size
        )
        near_key = (
            "residual_score"
            if self.args.sampler_profile == "residual_length"
            else "near_score"
        )
        self._merge_archive(self.near_miss_elite, near_items, near_key, archive_size)
        self._merge_diverse_archive(diverse_items)

    def _constraint_ladder_candidates(self, smiles_list, prop_list):
        if self.args.sampler_profile == "residual_length":
            items = self.residual_items(
                smiles_list, prop_list, self.current_candidate_ops
            )
        else:
            items = self._constraint_items(smiles_list, prop_list)
        if not items:
            return []
        selected = []
        if self.args.sampler_profile == "residual_length":
            selected.extend(
                item
                for item in items
                if item["residual_score"] >= self.args.residual_archive_min_score
            )
            selected.extend(
                item
                for item in items
                if item["soft_sim_ok"] and item["soft_quality_ok"]
            )
        else:
            selected.extend(
                item
                for item in items
                if item["soft_sim_ok"] and item["soft_quality_ok"]
            )
            selected.extend(
                item
                for item in items
                if item["near_score"] >= self.args.constraint_near_miss_min_score
            )
        dedup = {}
        for item in selected:
            old = dedup.get(item["smiles"])
            if old is None or item["pareto_score"] > old["pareto_score"]:
                dedup[item["smiles"]] = item
        out = sorted(
            dedup.values(),
            key=lambda item: (
                item.get("residual_score", item["near_score"]),
                item["pareto_score"],
            ),
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
            for rv, rq, rs, rsim, smiles in zip(
                rv_list, rq_list, rs_list, rsim_list, smiles_list
            ):
                if (
                    rv <= self.start_prop
                    or rq < 0.6
                    or rs < 6 / 9
                    or rsim < self.args.sim_thr
                ):
                    continue
                updates.extend((float(rv), frag) for frag in local_genmol_cut(smiles))

        for score, frag in updates:
            if frag in known or Chem.MolFromSmiles(frag) is None:
                continue
            known.add(frag)
            self.population.append((float(score), frag))

        self.population.sort(key=lambda item: item[0], reverse=True)
        if self.args.population_cap > 0:
            del self.population[self.args.population_cap :]

    def update_elite_smiles(self, smiles_list, prop_list):
        rv_list, rq_list, rs_list, rsim_list = prop_list
        existing = {smi for _, _, smi in self.elite_smiles}
        sim_floor = max(0.0, self.args.sim_thr - self.args.lead_elite_similarity_slack)
        for rv, rq, rs, rsim, smiles in zip(
            rv_list, rq_list, rs_list, rsim_list, smiles_list
        ):
            can = canonical_smiles(smiles)
            if can is None or can in existing:
                continue
            if (
                rsim < sim_floor
                or rq < self.args.pareto_min_qed
                or rs < self.args.pareto_min_sa
            ):
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
        del self.elite_smiles[self.args.lead_elite_size :]

    def record(self, smiles_list, prop_list):
        rv_list, rq_list, rs_list, rsim_list = prop_list
        with open(self.fname, "a") as f:
            for smiles, rv, rq, rs, rsim in zip(
                smiles_list, rv_list, rq_list, rs_list, rsim_list
            ):
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
        sim_quality_ok = [bool(s and q and a) for s, q, a in zip(sim_ok, qed_ok, sa_ok)]
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
            op_scores = sorted(
                (item["residual_score"] for item in op_items), reverse=True
            )
            op_best = float(op_scores[0])
            op_top_mean = float(np.mean(op_scores[: min(5, len(op_scores))]))
            strict_rate = sum(item["strict_ok"] for item in op_items) / len(op_items)
            sim_quality_rate = sum(
                item["sim_ok"] and item["quality_ok"] for item in op_items
            ) / len(op_items)
            sim_dock_rate = sum(
                item["sim_ok"] and item["dock_ok"] for item in op_items
            ) / len(op_items)
            improvement = (
                0.0 if prev_best == -float("inf") else max(0.0, op_best - prev_best)
            )
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
            ema = (
                self.args.residual_reward_decay * old_ema
                + (1.0 - self.args.residual_reward_decay) * reward
            )
            self.residual_length_reward_ema[op] = ema
            self.residual_length_weights[op] *= float(
                np.exp(self.args.residual_bandit_eta * ema)
            )
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

        remask = clamp(
            remask, self.args.adaptive_min_remask, self.args.adaptive_max_remask
        )
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
        elif self.no_improve_iters >= self.args.constraint_rescue_after_iters or (
            stats["sim_quality_rate"] >= self.args.constraint_low_sim_quality_rate
            and stats["dock_rate"] < self.args.constraint_low_dock_rate
        ):
            mode = "docking_repair"
        elif stats["new_rate"] < self.args.constraint_low_new_rate:
            mode = "diversity_repair"
        else:
            mode = "balanced"
        self.last_adaptive_mode = mode

    def record_empty_iteration(self, iter_idx, phase="Iter"):
        print(f"[{phase} {iter_idx:03d}] no novel candidates generated")
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
        if self.args.sampler_profile in FRONTIER_PROFILES:
            self.no_improve_iters += 1
            self.last_frontier_summary = "no_candidates"

    def evaluate_lead_batch(
        self,
        smiles_list,
        iter_idx,
        total,
        raw_best,
        feasible_best,
        phase="Iter",
    ):
        smiles_list = self.filter_oracle_novel_candidates(smiles_list)
        smiles_list = self.filter_oracle_feasible_candidates(smiles_list)
        smiles_list = self.limit_oracle_budget_candidates(smiles_list, total)
        if not smiles_list:
            self.record_empty_iteration(iter_idx, phase=phase)
            return total, raw_best, feasible_best, False

        prop_list = self.reward(smiles_list)
        self.frontier_update_state(smiles_list, prop_list)
        self.update_population(smiles_list, prop_list)
        self.record(smiles_list, prop_list)
        self.record_frontier_diagnostics(smiles_list, prop_list)
        if self.args.global_oracle_dedup:
            self.oracle_evaluated_smiles.update(smiles_list)
        adaptive_stats = self.adaptive_iter_stats(smiles_list, prop_list)
        self.update_residual_length_bandit(smiles_list, prop_list)
        total += len(smiles_list)
        raw_best = max(raw_best, max(prop_list[0], default=raw_best))
        feasible_scores = [
            rv
            for rv, rq, rs, rsim in zip(*prop_list)
            if (
                rv > self.start_prop
                and rq >= 0.6
                and rs >= 6 / 9
                and rsim >= self.args.sim_thr
            )
        ]
        previous_feasible_best = feasible_best
        feasible_best = max(
            feasible_best,
            max(feasible_scores, default=feasible_best),
        )
        if feasible_best > previous_feasible_best + 1e-6:
            self.no_improve_iters = 0
        else:
            self.no_improve_iters += 1
        self.rescue_mode = (
            self.args.sampler_profile == "hybrid_v4"
            and self.no_improve_iters >= self.args.v4_rescue_after_iters
        )
        if self.args.sampler_profile == "constraint_ladder":
            self.update_constraint_ladder_profile(iter_idx, adaptive_stats)
        else:
            self.update_adaptive_profile(iter_idx, adaptive_stats)
        print(
            f"[{phase} {iter_idx:03d}] generated={len(smiles_list)} "
            f"total={total} raw_top_DS={raw_best:.3f} "
            f"constrained_top_DS={feasible_best:.3f} "
            f"population={len(self.population)} rescue={self.rescue_mode} "
            f"adaptive={self.last_adaptive_mode} "
            f"sim_rate={adaptive_stats['sim_rate']:.3f} "
            f"sim_quality_rate={adaptive_stats['sim_quality_rate']:.3f} "
            f"dock_rate={adaptive_stats['dock_rate']:.3f} "
            f"strict_rate={adaptive_stats['strict_rate']:.3f} "
            f"new_rate={adaptive_stats['new_rate']:.3f} "
            f"archives=d{len(self.dock_elite)}/q{len(self.quality_elite)}/"
            f"n{len(self.near_miss_elite)} "
            f"remask={self.args.remask_fraction:.3f} "
            f"temp={self.args.temperature_start:.3f} "
            f"w_dock={self.args.pareto_docking_weight:.3f} "
            f"w_sim={self.args.pareto_similarity_weight:.3f} "
            f"length_ops={self.last_length_bandit_summary} "
            f"frontier={self.last_frontier_summary}"
        )
        return total, raw_best, feasible_best, True

    def run_budget_completion(self, total, raw_best, feasible_best):
        if not self.args.budget_completion:
            return total, raw_best, feasible_best
        if not self.args.global_oracle_dedup:
            raise RuntimeError(
                "--budget_completion requires --global_oracle_dedup so that "
                "every additional docking call contributes a new molecule."
            )

        budget = int(self.args.oracle_budget)
        if budget <= 0:
            raise RuntimeError("--oracle_budget must be positive.")
        if total >= budget:
            print(f"[Budget-completion] no extension required: calls={total}/{budget}")
            return total, raw_best, feasible_best

        decision = self.budget_completion_decision()
        route = decision["route"]
        print(
            "[Budget-decision] "
            f"route={route} reason={decision['reason']} "
            f"calls={total}/{budget} "
            f"best_dock_ratio={decision['best_dock_ratio']:.3f} "
            f"sim_quality={decision['sim_quality_count']} "
            f"strict={decision['strict_count']} "
            f"population_growth={decision['population_growth']}"
        )

        hard_completion = bool(self.args.budget_completion_until_budget)
        empty_batches = 0
        completion_iterations = 0
        restart_cycles = 0
        base_iteration = max(self.args.num_iter, self.frontier_iteration)
        while total < budget:
            if not hard_completion and (
                empty_batches >= self.args.budget_completion_empty_patience
                or completion_iterations >= self.args.budget_completion_max_iterations
            ):
                break
            completion_iterations += 1
            self.frontier_iteration = base_iteration + completion_iterations
            target_n = min(self.args.num_gen, budget - total)
            smiles_list = self.generate_budget_completion_batch(route, target_n)
            previous_total = total
            total, raw_best, feasible_best, evaluated = self.evaluate_lead_batch(
                smiles_list,
                self.frontier_iteration,
                total,
                raw_best,
                feasible_best,
                phase="Budget",
            )
            if not evaluated or total == previous_total:
                empty_batches += 1
                if (
                    hard_completion
                    and empty_batches >= self.args.budget_completion_empty_patience
                ):
                    restart_cycles += 1
                    route = "restart"
                    empty_batches = 0
                    print(
                        "[Budget-restart-cycle] "
                        f"cycle={restart_cycles} calls={total}/{budget} "
                        f"next_iteration={self.frontier_iteration + 1}"
                    )
                continue
            empty_batches = 0

            if route == "restart":
                updated = self.budget_completion_decision()
                if updated["route"] == "frontier":
                    route = "frontier"
                    print(
                        "[Budget-route-switch] restart->frontier "
                        f"reason={updated['reason']} "
                        f"best_dock_ratio={updated['best_dock_ratio']:.3f}"
                    )

        if total < budget:
            raise RuntimeError(
                "Lead budget completion exhausted proposal patience before "
                f"using the oracle budget: calls={total}/{budget}, "
                f"empty_batches={empty_batches}, "
                f"completion_iterations={completion_iterations}."
            )
        print(
            f"[Budget-completion] complete calls={total}/{budget} "
            f"extension_iterations={completion_iterations} "
            f"restart_cycles={restart_cycles} final_route={route}"
        )
        return total, raw_best, feasible_best

    def run(self):
        if self.args.sampler_profile in {
            "transition_feasible",
            "transition_feasible_hybrid",
        }:
            self.run_transition_feasible()
            return
        t_start = time()
        raw_best = self.resume_raw_best
        feasible_best = self.resume_feasible_best
        total = self.resume_total
        minimum_calls = int(getattr(self.args, "min_oracle_calls", 0))
        oracle_budget = int(self.args.oracle_budget)
        if minimum_calls < 0 or minimum_calls > oracle_budget:
            raise ValueError(
                "--min_oracle_calls must satisfy 0 <= min_oracle_calls <= "
                f"oracle_budget; got {minimum_calls} and {oracle_budget}."
            )
        start_iteration = 0
        if total:
            start_iteration = min(
                self.args.num_iter,
                self.resume_iteration_offset,
            )
            print(
                f"[Resume] continuing from {total}/"
                f"{self.args.oracle_budget} oracle calls at "
                f"iteration={self.resume_iteration_offset}."
            )
        iteration = start_iteration
        while iteration < int(self.args.num_iter) or total < minimum_calls:
            if total >= oracle_budget:
                print(
                    f"[Oracle-budget] stopping before iteration {iteration + 1}: "
                    f"calls={total}/{self.args.oracle_budget}"
                )
                break
            iteration += 1
            self.frontier_iteration = iteration
            smiles_list = self.generate_batch()
            total, raw_best, feasible_best, _ = self.evaluate_lead_batch(
                smiles_list,
                iteration,
                total,
                raw_best,
                feasible_best,
            )
            if iteration >= int(self.args.num_iter) and total < minimum_calls:
                print(
                    "[Minimum-oracle-extension] "
                    f"iteration={iteration} calls={total}/{minimum_calls} "
                    "continuing beyond the nominal iteration limit"
                )
        total, raw_best, feasible_best = self.run_budget_completion(
            total,
            raw_best,
            feasible_best,
        )
        print(f"{time() - t_start:.2f} sec elapsed")


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "-o",
        "--oracle_name",
        type=str,
        default="parp1",
        choices=["parp1", "fa7", "5ht1b", "braf", "jak2"],
    )
    parser.add_argument("-i", "--start_mol_idx", type=int, default=0, choices=[0, 1, 2])
    parser.add_argument("-d", "--sim_thr", type=float, default=0.4)
    parser.add_argument("-s", "--seed", type=int, default=0)

    parser.add_argument("--ckpt_path", type=str, required=True)
    parser.add_argument("--vocab", type=str, default="csdnet_vocab.pkl")
    parser.add_argument("--cpu", action="store_true")
    parser.add_argument(
        "--output_dir",
        type=str,
        default=os.path.join("CSDNet", "exp", "lead", "results"),
    )
    parser.add_argument("--resume", action="store_true")

    parser.add_argument("--num_gen", type=int, default=100)
    parser.add_argument("--num_iter", type=int, default=10)
    parser.add_argument("--max_generation_rounds", type=int, default=3)
    parser.add_argument(
        "--max_proposal_inputs_per_iteration",
        type=int,
        default=0,
        help=(
            "Hard cap on model proposal inputs in one outer Lead iteration; "
            "0 leaves the cap disabled. This bounds refill cost when canonical "
            "novelty collapses."
        ),
    )
    parser.add_argument(
        "--global_oracle_dedup",
        action="store_true",
        help=(
            "Canonicalize proposals and never send the same molecule to docking "
            "more than once."
        ),
    )
    parser.add_argument(
        "--budget_completion",
        action="store_true",
        help=(
            "After the baseline iterations, complete the actual docking-call "
            "budget using state-based continuation or seed-anchored restart."
        ),
    )
    parser.add_argument(
        "--budget_completion_until_budget",
        action="store_true",
        help=(
            "Do not stop after empty proposal batches. Use each patience window "
            "as a fresh seed-anchored restart and continue until oracle_budget."
        ),
    )
    parser.add_argument("--oracle_budget", type=int, default=1000)
    parser.add_argument(
        "--min_oracle_calls",
        type=int,
        default=0,
        help=(
            "Do not stop at num_iter until at least this many actual docking "
            "calls have been completed. The oracle_budget remains a hard cap."
        ),
    )
    parser.add_argument(
        "--oracle_feasible_only",
        action="store_true",
        help=(
            "Require QED >= 0.6, normalized SA >= 6/9 and SIM >= sim_thr "
            "before a proposal is sent to docking."
        ),
    )
    parser.add_argument(
        "--budget_close_dock_ratio",
        type=float,
        default=0.90,
        help=(
            "Continue the current frontier when a similarity-and-quality feasible "
            "candidate reaches this fraction of the starting docking score."
        ),
    )
    parser.add_argument(
        "--budget_completion_empty_patience",
        type=int,
        default=12,
    )
    parser.add_argument(
        "--budget_completion_max_iterations",
        type=int,
        default=200,
    )
    parser.add_argument("--min_atoms", type=int, default=10)
    parser.add_argument("--max_atoms", type=int, default=80)
    parser.add_argument("--population_cap", type=int, default=0)
    parser.add_argument(
        "--sampler_profile",
        choices=[
            "fragment",
            "similarity_aware",
            "adaptive_similarity",
            "multi_frontier",
            "universal_frontier",
            "universal_frontier_recovery",
            "universal_frontier_recovery_v2",
            "universal_frontier_recovery_v3",
            "safe_frontier_final",
            "universal_frontier_bridge",
            "lead_protected_completion",
            "lead_protected_completion_v2",
            "lead_protected_completion_v3",
            "lead_protected_completion_v4",
            "lead_protected_completion_v5",
            "integrated_frontier",
            "unified_frontier_v2",
            "unified_frontier_v2_1",
            "unified_frontier_restored",
            "lead_best_union",
            "hybrid_v4",
            "constraint_ladder",
            "residual_length",
            "transition_feasible",
            "transition_feasible_hybrid",
            "elastic_direct",
            "elastic_joint_frontier",
            "elastic_joint_frontier_v2",
            "elastic_joint_frontier_v3",
            "elastic_joint_frontier_v4",
            "elastic_joint_frontier_v5",
        ],
        default="fragment",
    )
    parser.add_argument("--lead_elite_seed_prob", type=float, default=0.55)
    parser.add_argument(
        "--atomic_length_prior",
        type=str,
        default=None,
        help="Validated CSDNet atomic-token length prior for elastic Lead profiles.",
    )
    parser.add_argument("--direct_length_quantile_low", type=float, default=0.02)
    parser.add_argument("--direct_length_quantile_high", type=float, default=0.98)
    parser.add_argument("--direct_parent_pool_size", type=int, default=48)
    parser.add_argument("--direct_peripheral_probability", type=float, default=0.72)
    parser.add_argument("--direct_overgenerate_factor", type=float, default=1.35)
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
    parser.add_argument("--frontier_legacy_fraction", type=float, default=0.60)
    parser.add_argument("--frontier_legacy_fraction_late", type=float, default=0.60)
    parser.add_argument("--frontier_legacy_warmup_iters", type=int, default=2)
    parser.add_argument("--frontier_recovery_start_iter", type=int, default=6)
    parser.add_argument("--frontier_recovery_legacy_fraction", type=float, default=0.25)
    parser.add_argument("--frontier_recovery_completion_boost", type=float, default=4.0)
    parser.add_argument("--frontier_recovery_start_boost", type=float, default=3.0)
    parser.add_argument("--frontier_recovery_joint_boost", type=float, default=2.0)
    parser.add_argument("--frontier_bridge_start_iter", type=int, default=6)
    parser.add_argument("--frontier_bridge_fraction", type=float, default=0.12)
    parser.add_argument("--frontier_bridge_parent_top_k", type=int, default=12)
    parser.add_argument("--frontier_bridge_remask", type=float, default=0.05)
    parser.add_argument("--frontier_bridge_temperature", type=float, default=0.92)
    parser.add_argument("--frontier_bridge_span_prob", type=float, default=0.85)
    parser.add_argument("--completion_start_iter", type=int, default=6)
    parser.add_argument("--completion_bridge_fraction", type=float, default=0.12)
    parser.add_argument("--completion_dock_fraction", type=float, default=0.18)
    parser.add_argument("--completion_boundary_fraction", type=float, default=0.12)
    parser.add_argument("--completion_boundary_tolerance", type=float, default=0.02)
    parser.add_argument("--completion_max_span_tokens", type=int, default=2)
    parser.add_argument("--completion_parent_top_k", type=int, default=3)
    parser.add_argument("--completion_parent_min_slack", type=float, default=0.02)
    parser.add_argument("--completion_v2_max_span_tokens", type=int, default=3)
    parser.add_argument("--completion_v3_late_start_iter", type=int, default=8)
    parser.add_argument("--completion_v3_probe_fraction", type=float, default=0.20)
    parser.add_argument("--completion_v3_commit_fraction", type=float, default=0.30)
    parser.add_argument("--completion_v3_max_route_deficit", type=float, default=0.16)
    parser.add_argument(
        "--completion_v3_route_tie_tolerance", type=float, default=0.015
    )
    parser.add_argument(
        "--completion_v3_min_abs_improvement", type=float, default=0.001
    )
    parser.add_argument("--completion_v3_min_rel_improvement", type=float, default=0.05)
    parser.add_argument("--completion_v4_anchor_start_iter", type=int, default=8)
    parser.add_argument(
        "--completion_v4_anchor_probe_fraction", type=float, default=0.08
    )
    parser.add_argument(
        "--completion_v4_route_probe_fraction", type=float, default=0.12
    )
    parser.add_argument(
        "--completion_v4_route_commit_fraction", type=float, default=0.20
    )
    parser.add_argument("--completion_v4_route_max_deficit", type=float, default=0.35)
    parser.add_argument("--completion_v4_anchor_remask", type=float, default=0.06)
    parser.add_argument("--completion_v4_anchor_temperature", type=float, default=0.94)
    parser.add_argument("--completion_v4_anchor_span_prob", type=float, default=0.90)
    parser.add_argument("--completion_v4_anchor_max_span_tokens", type=int, default=3)
    parser.add_argument(
        "--completion_v4_anchor_max_atom_fraction", type=float, default=0.25
    )
    parser.add_argument("--completion_v5_late_start_iter", type=int, default=8)
    parser.add_argument("--completion_v5_portfolio_fraction", type=float, default=0.20)
    parser.add_argument("--completion_v5_max_route_deficit", type=float, default=0.21)
    parser.add_argument("--completion_v5_route_temperature", type=float, default=0.08)
    parser.add_argument("--completion_v5_min_route_weight", type=float, default=0.25)
    parser.add_argument("--completion_v5_seed_probe_fraction", type=float, default=0.08)
    parser.add_argument("--completion_dock_remask", type=float, default=0.035)
    parser.add_argument("--completion_dock_temperature", type=float, default=0.90)
    parser.add_argument("--completion_dock_span_prob", type=float, default=0.92)
    parser.add_argument("--completion_similarity_remask", type=float, default=0.025)
    parser.add_argument("--completion_similarity_temperature", type=float, default=0.88)
    parser.add_argument("--completion_similarity_span_prob", type=float, default=0.95)
    parser.add_argument("--completion_quality_remask", type=float, default=0.035)
    parser.add_argument("--completion_quality_temperature", type=float, default=0.92)
    parser.add_argument("--completion_quality_span_prob", type=float, default=0.90)
    parser.add_argument("--frontier_legacy_local_remask", type=float, default=0.14)
    parser.add_argument("--frontier_legacy_local_temperature", type=float, default=0.95)
    parser.add_argument("--frontier_legacy_local_span_prob", type=float, default=0.85)
    parser.add_argument("--frontier_archive_size", type=int, default=100)
    parser.add_argument("--frontier_parent_top_k", type=int, default=30)
    parser.add_argument("--frontier_similarity_slack", type=float, default=0.08)
    parser.add_argument("--frontier_min_operator_fraction", type=float, default=0.08)
    parser.add_argument("--frontier_overgenerate_factor", type=float, default=2.0)
    parser.add_argument("--frontier_bandit_eta", type=float, default=2.0)
    parser.add_argument("--frontier_bandit_ucb", type=float, default=0.25)
    parser.add_argument("--frontier_reward_alpha", type=float, default=0.35)
    parser.add_argument("--frontier_reward_tail_fraction", type=float, default=0.20)
    parser.add_argument("--frontier_reward_tail_min", type=int, default=2)
    parser.add_argument("--frontier_reward_mean_weight", type=float, default=0.20)
    parser.add_argument("--frontier_crossing_bonus", type=float, default=0.08)
    parser.add_argument("--frontier_regression_penalty", type=float, default=0.14)
    parser.add_argument("--frontier_pair_bonus", type=float, default=0.12)
    parser.add_argument("--frontier_strict_bonus", type=float, default=0.15)
    parser.add_argument("--frontier_mean_deficit_weight", type=float, default=0.25)
    parser.add_argument("--frontier_docking_margin", type=float, default=0.05)
    parser.add_argument("--frontier_residual_l1_weight", type=float, default=0.10)
    parser.add_argument("--frontier_need_weight", type=float, default=1.50)
    parser.add_argument("--frontier_need_top_k", type=int, default=20)
    parser.add_argument("--frontier_readiness_weight", type=float, default=0.35)
    parser.add_argument("--frontier_readiness_scale", type=float, default=4.0)
    parser.add_argument("--frontier_start_need_scale", type=float, default=0.75)
    parser.add_argument("--frontier_start_parent_prob", type=float, default=0.35)
    parser.add_argument("--frontier_length_prob", type=float, default=0.25)
    parser.add_argument("--frontier_length_deltas", type=str, default="-2:-1:1:2")
    parser.add_argument("--disable_learned_insertion", action="store_true")
    parser.add_argument("--learned_insertion_fraction_scale", type=float, default=1.0)
    parser.add_argument("--learned_insertion_max_growth", type=int, default=3)
    parser.add_argument("--learned_insertion_max_shrink", type=int, default=3)
    parser.add_argument("--learned_insertion_max_per_step", type=int, default=4)
    parser.add_argument("--frontier_max_peripheral_fraction", type=float, default=0.35)
    parser.add_argument("--frontier_start_remask", type=float, default=0.08)
    parser.add_argument("--frontier_start_temperature", type=float, default=0.96)
    parser.add_argument("--frontier_start_span_prob", type=float, default=0.78)
    parser.add_argument("--frontier_dock_remask", type=float, default=0.09)
    parser.add_argument("--frontier_dock_temperature", type=float, default=0.96)
    parser.add_argument("--frontier_dock_span_prob", type=float, default=0.76)
    parser.add_argument("--frontier_similarity_remask", type=float, default=0.06)
    parser.add_argument("--frontier_similarity_temperature", type=float, default=0.90)
    parser.add_argument("--frontier_similarity_span_prob", type=float, default=0.72)
    parser.add_argument("--frontier_quality_remask", type=float, default=0.12)
    parser.add_argument("--frontier_quality_temperature", type=float, default=1.02)
    parser.add_argument("--frontier_quality_span_prob", type=float, default=0.80)
    parser.add_argument("--frontier_joint_remask", type=float, default=0.10)
    parser.add_argument("--frontier_joint_temperature", type=float, default=0.98)
    parser.add_argument("--frontier_joint_span_prob", type=float, default=0.78)
    parser.add_argument("--integrated_warmup_iterations", type=int, default=2)
    parser.add_argument("--integrated_plateau_patience", type=int, default=2)
    parser.add_argument("--integrated_collapse_threshold", type=float, default=0.70)
    parser.add_argument("--integrated_lineage_top_k", type=int, default=80)
    parser.add_argument("--integrated_rank_tolerance", type=float, default=1e-4)
    parser.add_argument("--integrated_root_credit_alpha", type=float, default=0.20)
    parser.add_argument("--integrated_legacy_fraction_warmup", type=float, default=0.60)
    parser.add_argument("--integrated_legacy_fraction_search", type=float, default=0.35)
    parser.add_argument("--integrated_legacy_fraction_refine", type=float, default=0.35)
    parser.add_argument("--integrated_legacy_fraction_rescue", type=float, default=0.25)
    parser.add_argument("--integrated_trust_min", type=float, default=0.04)
    parser.add_argument("--integrated_trust_max", type=float, default=0.30)
    parser.add_argument("--integrated_trust_deficit_scale", type=float, default=0.18)
    parser.add_argument("--integrated_trust_slack_scale", type=float, default=0.25)
    parser.add_argument("--integrated_max_component_fraction", type=float, default=0.55)
    parser.add_argument("--integrated_lineage_remask", type=float, default=0.22)
    parser.add_argument("--integrated_lineage_temperature", type=float, default=1.05)
    parser.add_argument("--integrated_lineage_span_prob", type=float, default=0.78)
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
    parser.add_argument(
        "--constraint_quality_temperature_start", type=float, default=1.02
    )
    parser.add_argument(
        "--constraint_medium_temperature_start", type=float, default=1.12
    )
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
    parser.add_argument("--tf_legacy_floor_fraction", type=float, default=0.35)
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
