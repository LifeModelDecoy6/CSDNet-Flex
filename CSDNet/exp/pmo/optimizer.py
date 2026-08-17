#!/usr/bin/env python
import csv
import hashlib
import inspect
import json
import math
import os
import pickle
import random
from pathlib import Path
from time import sleep, time

import numpy as np
import pandas as pd
import torch
import torch.nn.functional as F
from rdkit import Chem, DataStructs, RDLogger
from rdkit.Chem import AllChem, QED

from CSDNet.exp.pmo.base_optimizer import BaseOptimizer, top_auc
from CSDNet.exp.denovo.sampler_profiles import SAMPLER_PROFILES
from CSDNet.util.checkpoint import load_backbone_from_checkpoint
from CSDNet.util.fsm import (
    ValenceFSMTracker,
    compute_rdkit_kekulize_penalties,
    expand_violation_mask,
    prepare_rdkit_kekulize_checker,
)
from CSDNet.util.motif import extract_motifs, sample_csdnet_with_frozen_motifs
from CSDNet.util.elastic_sampling import (
    _repair_final_sequences,
    sample_elastic_local_infill,
)
from CSDNet.util.unified_sampling import sample_unified_local_infill
from CSDNet.util.length_prior import load_atomic_length_prior
from CSDNet.util.sampling import (
    _all_position_refine_tokens,
    _cosine_remask_rates,
    _filter_sampling_logits,
    _length_conditioned_confidence_temperatures,
    _smooth_length_mix,
    sample_csdnet,
)
from CSDNet.util.tokenizer import SMILESTokenizer, tokenize_smiles
from CSDNet.exp.pmo.v9 import (
    V9BatchBandit,
    V9LineageArchive,
    allocate_weighted_counts,
    batch_frontier_reward,
    classify_v9_state,
    v9_local_weights,
    v9_root_fraction,
    v9_root_weights,
)
from CSDNet.exp.pmo.v10 import BetaEventBandit, KnnUCBScreen
from CSDNet.optim.frontier import (
    PMOFrontierAdapter,
    RestoredPMOFrontierAdapter,
    ScalarFrontierAdapter,
    UnifiedFrontierEngine,
    allocate_insertion_flags,
)
from CSDNet.optim.protected_frontier import (
    BaselineProtectedFrontierEngine,
    EvidenceGatedPMOHead,
    ReversibleEvidencePMOHead,
    SafePMOFrontierHead,
)
from CSDNet.optim.structure import (
    adaptive_peripheral_edit_plan,
    atom_span_edit_plan,
    preserves_murcko_scaffold,
)


RDLogger.DisableLog("rdApp.*")

ROOT_DIR = os.fspath(Path(__file__).resolve().parent)
PMO_TASKS = [
    "albuterol_similarity",
    "amlodipine_mpo",
    "celecoxib_rediscovery",
    "deco_hop",
    "drd2",
    "fexofenadine_mpo",
    "gsk3b",
    "isomers_c7h8n2o2",
    "isomers_c9h10n2o2pf2cl",
    "jnk3",
    "median1",
    "median2",
    "mestranol_similarity",
    "osimertinib_mpo",
    "perindopril_mpo",
    "qed",
    "ranolazine_mpo",
    "scaffold_hop",
    "sitagliptin_mpo",
    "thiothixene_rediscovery",
    "troglitazone_rediscovery",
    "valsartan_smarts",
    "zaleplon_mpo",
]


def task_size_bounds(oracle_name, default_min=20, default_max=40):
    """Return the public PMO benchmark atom-count range for an oracle."""
    if oracle_name in {
        "albuterol_similarity",
        "isomers_c7h8n2o2",
        "isomers_c9h10n2o2pf2cl",
        "median1",
        "qed",
        "sitagliptin_mpo",
        "zaleplon_mpo",
    }:
        return 10, 30
    if oracle_name in {"gsk3b", "jnk3"}:
        return 30, 80
    return default_min, default_max


def canonical_smiles(smiles):
    mol = Chem.MolFromSmiles(smiles)
    if mol is None:
        return None
    return Chem.MolToSmiles(mol, canonical=True)


def atom_count(smiles):
    mol = Chem.MolFromSmiles(smiles)
    return mol.GetNumAtoms() if mol is not None else 0


def tokenizable(smiles, tk, max_len):
    toks = tokenize_smiles(smiles)
    if len(toks) + 2 > max_len:
        return False
    return all(tok in tk.vocab for tok in toks)


def clean_dummy_fragment(fragment):
    mol = Chem.MolFromSmiles(fragment)
    if mol is None:
        return None
    rw = Chem.RWMol(mol)
    dummy = [a.GetIdx() for a in rw.GetAtoms() if a.GetAtomicNum() == 0]
    if not dummy:
        return Chem.MolToSmiles(mol, canonical=True)
    for idx in sorted(dummy, reverse=True):
        rw.RemoveAtom(idx)
    mol = rw.GetMol()
    try:
        Chem.SanitizeMol(mol)
    except Exception:
        return None
    if mol.GetNumAtoms() == 0:
        return None
    return Chem.MolToSmiles(mol, canonical=True)


def load_ref_lengths(
    data_dir,
    tk,
    max_len,
    sample_n=50000,
    atomic_length_prior=None,
):
    if atomic_length_prior:
        lengths, metadata = load_atomic_length_prior(
            atomic_length_prior,
            max_len=max_len,
        )
        print(
            "Loaded atomic reference lengths: "
            f"path={metadata['path']} count={len(lengths)} "
            f"range={min(lengths)}-{max(lengths)}"
        )
        return lengths
    if not data_dir:
        return list(range(16, min(max_len, 96) + 1))
    try:
        from datasets import load_from_disk

        ds = load_from_disk(data_dir)
    except Exception as exc:
        print(
            f"Could not load data_dir={data_dir}; using uniform reference lengths. {exc}"
        )
        return list(range(16, min(max_len, 96) + 1))

    col = (
        "text"
        if "text" in ds.column_names
        else "smiles"
        if "smiles" in ds.column_names
        else None
    )
    if col is None:
        print(f"No text/smiles column in {data_dir}; using uniform reference lengths.")
        return list(range(16, min(max_len, 96) + 1))

    n = min(len(ds), sample_n)
    lengths = []
    for row in ds.select(range(n)):
        smi = row.get(col)
        if not isinstance(smi, str):
            continue
        L = min(max_len, max(3, tk.token_length(smi, include_special=True)))
        lengths.append(L)
    return lengths or list(range(16, min(max_len, 96) + 1))


def load_csdnet_model(args):
    with open(args.vocab, "rb") as f:
        tk = SMILESTokenizer(pickle.load(f))
    device = "cuda" if torch.cuda.is_available() and not args.cpu else "cpu"
    print(f"Loading CSDNet checkpoint: {args.ckpt_path}")
    model = load_backbone_from_checkpoint(args.ckpt_path, tk, device="cpu")
    if device == "cuda":
        last_error = None
        for attempt in range(1, 4):
            try:
                torch.cuda.empty_cache()
                model = model.to(device)
                last_error = None
                break
            except RuntimeError as exc:
                last_error = exc
                print(f"CUDA model transfer failed ({attempt}/3): {exc}")
                if attempt < 3:
                    sleep(20)
        if last_error is not None:
            raise RuntimeError(
                "Could not move CSDNet model to CUDA after 3 attempts"
            ) from last_error
    else:
        model = model.to(device)
    model.eval()
    return model, tk, device


def pmo_vocab_path(oracle_name):
    path = os.path.join(ROOT_DIR, "vocab", f"{oracle_name}.csv")
    if os.path.exists(path):
        return path
    if oracle_name.endswith("_current"):
        fallback = os.path.join(
            ROOT_DIR, "vocab", f"{oracle_name[: -len('_current')]}.csv"
        )
        if os.path.exists(fallback):
            return fallback
    return path


def load_pmo_motifs(oracle_name, tk, max_len, limit, min_atoms=4, max_atoms=36):
    path = pmo_vocab_path(oracle_name)
    df = pd.read_csv(path)
    motifs = []
    seen = set()
    for _, row in df.iterrows():
        motif = clean_dummy_fragment(str(row["frag"]))
        if motif is None or motif in seen:
            continue
        mol = Chem.MolFromSmiles(motif)
        if mol is None:
            continue
        atoms = mol.GetNumAtoms()
        if atoms < min_atoms or atoms > max_atoms:
            continue
        if not tokenizable(motif, tk, max_len=max_len - 4):
            continue
        seen.add(motif)
        motifs.append(
            {
                "motif": motif,
                "score": float(row["score"]),
                "support": 1.0,
                "quality_rate": float(row["score"]),
                "enrichment": float(row["score"]),
                "mean_qed": 0.0,
                "mean_sa": 0.0,
                "motif_type": "pmo_vocab",
            }
        )
        if len(motifs) >= limit:
            break
    if len(motifs) < 2:
        raise RuntimeError(
            f"Too few tokenizable PMO motifs for {oracle_name}: {len(motifs)}"
        )
    return motifs


def load_pmo_fragments(oracle_name, population_size):
    path = pmo_vocab_path(oracle_name)
    df = pd.read_csv(path).head(max(population_size * 4, population_size))
    fragments = []
    seen = set()
    for _, row in df.iterrows():
        frag = str(row["frag"])
        if frag in seen or Chem.MolFromSmiles(frag) is None:
            continue
        seen.add(frag)
        fragments.append((float(row["score"]), frag))
        if len(fragments) >= population_size:
            break
    if len(fragments) < 2:
        raise RuntimeError(f"Too few PMO fragments for {oracle_name}: {len(fragments)}")
    return fragments


ORACLE_PRESCREEN_FREE_MODES = frozenset({"iterative_remask_v9_no_prescreen"})
ELASTIC_PRESCREEN_MODES = frozenset(
    {"elastic_frontier_prescreen", "elastic_frontier_prescreen_v2"}
)


def elastic_prescreen_rank_tier_weights(state):
    """Allocate a deep ranked prior without diluting its score elite.

    Rank 1--100 is the historical prior used by the successful V9 runs.
    Ranks 101--500 and 501+ widen structural recall, but receive substantial
    probability only when score progress is sparse, plateaued, or collapsed.
    The policy depends only on online search state, never on oracle identity.
    """
    return {
        "warmup": (0.85, 0.12, 0.03),
        "saturated": (0.85, 0.12, 0.03),
        "search": (0.65, 0.25, 0.10),
        "sparse": (0.50, 0.30, 0.20),
        "plateau": (0.40, 0.35, 0.25),
        "collapsed": (0.35, 0.35, 0.30),
        "fallback": (0.30, 0.35, 0.35),
    }.get(state, (0.65, 0.25, 0.10))


def _rank_tiers(rows):
    rows = list(rows)
    return rows[:100], rows[100:500], rows[500:]


def _weighted_rank_tier(rows, state, rng=random):
    tiers = _rank_tiers(rows)
    weights = elastic_prescreen_rank_tier_weights(state)
    available = [
        (tier, weight)
        for tier, weight in zip(tiers, weights)
        if tier and weight > 0.0
    ]
    if not available:
        return []
    threshold = rng.random() * sum(weight for _, weight in available)
    cumulative = 0.0
    for tier, weight in available:
        cumulative += weight
        if threshold <= cumulative:
            return tier
    return available[-1][0]


def sample_rank_stratified_fragments(population, state, count=2, rng=random):
    """Sample distinct fragments from a state-adaptive ranked prior."""
    rows = list(population)
    if len(rows) < count:
        return []
    selected = []
    selected_fragments = set()
    for _ in range(max(20, count * 20)):
        tier = _weighted_rank_tier(rows, state, rng=rng)
        if not tier:
            break
        row = rng.choice(tier)
        fragment = row[1]
        if fragment in selected_fragments:
            continue
        selected.append(row)
        selected_fragments.add(fragment)
        if len(selected) >= count:
            return selected
    remaining = [row for row in rows if row[1] not in selected_fragments]
    rng.shuffle(remaining)
    return (selected + remaining)[:count]


def select_rank_stratified_motifs(motifs, state, limit, rng=random):
    """Build a bounded active motif pool from a much deeper ranked list."""
    rows = list(motifs)
    limit = max(0, min(int(limit), len(rows)))
    if limit == 0 or limit == len(rows):
        return rows[:limit]
    tiers = _rank_tiers(rows)
    weights = elastic_prescreen_rank_tier_weights(state)
    tier_names = ("elite", "middle", "tail")
    counts = allocate_weighted_counts(
        limit,
        [
            (name, weight)
            for name, tier, weight in zip(tier_names, tiers, weights)
            if tier
        ],
    )
    selected = []
    for index, (name, tier) in enumerate(zip(tier_names, tiers)):
        count = min(len(tier), int(counts.get(name, 0)))
        if count <= 0:
            continue
        if index == 0:
            selected.extend(tier[:count])
        else:
            selected.extend(rng.sample(tier, count))
    if len(selected) < limit:
        selected_ids = {id(row) for row in selected}
        remaining = [row for row in rows if id(row) not in selected_ids]
        selected.extend(remaining[: limit - len(selected)])
    selected = selected[:limit]
    tier_by_id = {
        id(row): tier_index
        for tier_index, tier in enumerate(tiers)
        for row in tier
    }
    selected_by_tier = {
        tier_index: [
            row for row in selected if tier_by_id.get(id(row)) == tier_index
        ]
        for tier_index in range(3)
    }
    weighted_rows = []
    for tier_index, tier_rows in selected_by_tier.items():
        if not tier_rows:
            continue
        raw_scores = [max(0.0, float(row["score"])) for row in tier_rows]
        score_sum = sum(raw_scores)
        if score_sum <= 0.0:
            within_weights = [1.0 / len(tier_rows)] * len(tier_rows)
        else:
            within_weights = [score / score_sum for score in raw_scores]
        for row, within_weight in zip(tier_rows, within_weights):
            copied = dict(row)
            copied["sampling_weight"] = weights[tier_index] * within_weight
            weighted_rows.append(copied)
    return weighted_rows


def elastic_prescreen_root_policy(mode, oracle_name, state):
    """Return the root budget for the declared oracle-specific PMO protocol.

    The default is exactly the historical V9 policy. Scaffold hopping is the
    sole root-budget exception in V2: downloaded trajectories show that its
    successful route entered through a low-ranked oracle-specific ZINC
    fragment. All V2 tasks may read the deeper rank-stratified prior, but only
    scaffold hopping spends a larger fraction of oracle calls on roots. This
    is valid only for the explicitly labelled oracle-prescreen benchmark,
    never the direct no-prescreen mode.
    """
    fraction = float(v9_root_fraction(state))
    weights = dict(v9_root_weights(state))
    if mode != "elastic_frontier_prescreen_v2" or oracle_name != "scaffold_hop":
        return fraction, weights
    if state == "warmup":
        return 1.0, {"attach_only": 1.0}
    fraction = max(
        fraction,
        {
            "search": 0.42,
            "sparse": 0.55,
            "plateau": 0.55,
            "collapsed": 0.60,
            "saturated": 0.18,
            "fallback": 1.00,
        }.get(state, 0.42),
    )
    return fraction, {
        "attach_only": 0.20,
        "motif_restart": 0.25,
        "fragment_anchor": 0.55,
    }


def initialize_v9_prior(
    mode,
    oracle_name,
    tk,
    max_len,
    population_size,
    motif_limit,
    motif_min_atoms=4,
    motif_max_atoms=36,
):
    """Load the V9 prior, or leave it empty for budgeted online bootstrap."""
    if mode in ORACLE_PRESCREEN_FREE_MODES:
        return [], [], "online_budgeted"
    population = load_pmo_fragments(oracle_name, population_size)
    motifs = load_pmo_motifs(
        oracle_name,
        tk,
        max_len=max_len,
        limit=motif_limit,
        min_atoms=motif_min_atoms,
        max_atoms=motif_max_atoms,
    )
    return population, motifs, "zinc250k_oracle_scored"


def append_csv(path, smiles, score):
    os.makedirs(os.path.dirname(path), exist_ok=True)
    with open(path, "a") as f:
        f.write(f"{smiles},{float(score)}\n")


def append_summary(path, row):
    os.makedirs(os.path.dirname(path), exist_ok=True)
    fields = [
        "mode",
        "oracle",
        "seed",
        "calls",
        "avg_top1",
        "avg_top10",
        "avg_top100",
        "auc_top1",
        "auc_top10",
        "auc_top100",
        "elapsed_sec",
        "nonzero_scores",
        "best_score",
        "unique_recorded",
    ]
    lock_path = path + ".lock"
    with open(lock_path, "w") as lock:
        try:
            import fcntl

            fcntl.flock(lock, fcntl.LOCK_EX)
        except Exception:
            pass
        exists = os.path.exists(path)
        with open(path, "a", newline="") as f:
            writer = csv.DictWriter(f, fieldnames=fields, extrasaction="ignore")
            if not exists:
                writer.writeheader()
            writer.writerow(row)
        try:
            import fcntl

            fcntl.flock(lock, fcntl.LOCK_UN)
        except Exception:
            pass


def summarize_buffer(buffer, max_oracle_calls, freq_log):
    if not buffer:
        return {
            "calls": 0,
            "avg_top1": 0.0,
            "avg_top10": 0.0,
            "avg_top100": 0.0,
            "auc_top1": 0.0,
            "auc_top10": 0.0,
            "auc_top100": 0.0,
        }
    ordered_by_time = dict(
        sorted(buffer.items(), key=lambda kv: kv[1][1], reverse=False)
    )
    top = sorted(buffer.items(), key=lambda kv: kv[1][0], reverse=True)[:100]
    scores = [x[1][0] for x in top]
    return {
        "calls": len(buffer),
        "avg_top1": float(np.max(scores)),
        "avg_top10": float(np.mean(sorted(scores, reverse=True)[:10])),
        "avg_top100": float(np.mean(scores)),
        "auc_top1": float(
            top_auc(ordered_by_time, 1, True, freq_log, max_oracle_calls)
        ),
        "auc_top10": float(
            top_auc(ordered_by_time, 10, True, freq_log, max_oracle_calls)
        ),
        "auc_top100": float(
            top_auc(ordered_by_time, 100, True, freq_log, max_oracle_calls)
        ),
    }


def parse_float_list(text, default):
    if text is None:
        return list(default)
    vals = []
    normalized = str(text).replace(";", ",").replace(":", ",")
    for item in normalized.split(","):
        item = item.strip()
        if not item:
            continue
        vals.append(float(item))
    return vals or list(default)


def mol_fp_from_smiles(smiles):
    mol = Chem.MolFromSmiles(smiles)
    if mol is None:
        return None
    return AllChem.GetMorganFingerprintAsBitVect(mol, 2, 2048)


def max_tanimoto(fp, fps):
    if fp is None or not fps:
        return 0.0
    return float(max(DataStructs.BulkTanimotoSimilarity(fp, fps)))


def attach_fragments(frag1, frag2):
    rxn = AllChem.ReactionFromSmarts("[*:1]-[1*].[1*]-[*:2]>>[*:1]-[*:2]")
    mol1 = Chem.MolFromSmiles(frag1)
    mol2 = Chem.MolFromSmiles(frag2)
    if mol1 is None or mol2 is None:
        return None
    try:
        products = rxn.RunReactants((mol1, mol2))
    except Exception:
        return None
    if not products:
        return None
    product = random.choice(products)[0]
    try:
        Chem.SanitizeMol(product)
    except Exception:
        return None
    return Chem.MolToSmiles(product, canonical=True)


def local_genmol_cut(smiles):
    def cut_nonring(mol):
        query = Chem.MolFromSmarts("[*]-;!@[*]")
        if not mol.HasSubstructMatch(query):
            return None
        bis = random.choice(mol.GetSubstructMatches(query))
        bond = mol.GetBondBetweenAtoms(bis[0], bis[1]).GetIdx()
        fragments_mol = Chem.FragmentOnBonds(
            mol,
            [bond],
            addDummies=True,
            dummyLabels=[(1, 1)],
        )
        try:
            return Chem.GetMolFrags(fragments_mol, asMols=True, sanitizeFrags=True)
        except ValueError:
            return None

    mol = Chem.MolFromSmiles(smiles)
    if mol is None:
        return set()
    frags = set()
    for _ in range(3):
        frags_nonring = cut_nonring(mol)
        if frags_nonring is not None:
            frags |= {Chem.MolToSmiles(f) for f in frags_nonring}
    return frags


def fragment_heavy_atom_count(fragment):
    mol = Chem.MolFromSmiles(fragment)
    if mol is None:
        return 0
    return sum(atom.GetAtomicNum() > 1 for atom in mol.GetAtoms())


def infer_fragment_pair_size_bounds(
    population,
    tk,
    max_len,
    sample_n=600,
    low_quantile=0.02,
    high_quantile=0.98,
    margin=6,
):
    """Infer broad molecule-size bounds from the active fragment vocabulary."""
    fragments = [frag for _, frag in population if Chem.MolFromSmiles(frag) is not None]
    counts = []
    if len(fragments) >= 2:
        for _ in range(max(1, int(sample_n))):
            frag1, frag2 = random.sample(fragments, 2)
            smiles = attach_fragments(frag1, frag2)
            can = canonical_smiles(smiles) if smiles else None
            if can is None or not tokenizable(can, tk, max_len):
                continue
            counts.append(atom_count(can))

    if len(counts) < 20:
        frag_sizes = [fragment_heavy_atom_count(frag) for frag in fragments]
        frag_sizes = [size for size in frag_sizes if size > 0]
        for _ in range(max(20, int(sample_n) // 2)):
            if len(frag_sizes) < 2:
                break
            size1, size2 = random.sample(frag_sizes, 2)
            counts.append(size1 + size2)

    if not counts:
        # This is a universal tokenizer-level fallback, not an oracle-specific rule.
        return 4, max(12, min(max_len - 2, 80))

    low_q = float(np.clip(low_quantile, 0.0, 0.49))
    high_q = float(np.clip(high_quantile, 0.51, 1.0))
    low = int(math.floor(np.quantile(counts, low_q))) - int(margin)
    high = int(math.ceil(np.quantile(counts, high_q))) + int(margin)
    low = max(2, low)
    high = max(low + 4, high)
    return low, high


def choose_remask_positions(body_len, fraction, min_tokens, span_prob):
    if body_len <= 0:
        return []
    n_mask = max(min_tokens, int(round(body_len * fraction)))
    n_mask = max(1, min(body_len, n_mask))
    positions = list(range(1, body_len + 1))
    if random.random() < span_prob:
        start = random.randint(1, body_len - n_mask + 1)
        return list(range(start, start + n_mask))
    return random.sample(positions, n_mask)


def resolve_local_sampler_profile(profile=None):
    profile = (
        str(profile or os.environ.get("CSDNET_LOCAL_SAMPLER_PROFILE", "legacy"))
        .strip()
        .lower()
    )
    aliases = {
        "promax_progressive_length_coupled": "progressive_length_coupled",
        "ztrajlc": "progressive_length_coupled",
        "promax_fragment_conditional_refine": "conditional_progressive_refine",
        "fragment_conditional_refine": "conditional_progressive_refine",
        "promax_fragment_editable_refine": "conditional_editable_refine",
        "fragment_editable_refine": "conditional_editable_refine",
        "promax_fragment_masked_refine": "conditional_masked_refine",
        "fragment_masked_refine": "conditional_masked_refine",
        "promax_task_adaptive_local": "task_adaptive_local",
        "task_local": "task_adaptive_local",
        "promax_task_adaptive_refine": "task_adaptive_refine",
        "task_refine": "task_adaptive_refine",
    }
    profile = aliases.get(profile, profile)
    if profile not in {
        "legacy",
        "progressive_length_coupled",
        "conditional_progressive_refine",
        "conditional_editable_refine",
        "conditional_masked_refine",
        "task_adaptive_local",
        "task_adaptive_refine",
    }:
        raise ValueError(
            "local_sampler_profile must be legacy, progressive_length_coupled, "
            "conditional_progressive_refine, conditional_editable_refine, "
            "conditional_masked_refine, task_adaptive_local, or "
            f"task_adaptive_refine; got {profile!r}"
        )
    return profile


def progressive_global_sampler_kwargs(profile=None):
    """Transfer ProMax trajectory controls to task-level global restarts."""
    if resolve_local_sampler_profile(profile) == "legacy":
        return {}
    accepted = inspect.signature(sample_csdnet).parameters
    return {
        key: value
        for key, value in SAMPLER_PROFILES["promax_progressive_length_coupled"].items()
        if key in accepted
    }


def _is_length_edit_atom_token(tok):
    if not tok:
        return False
    if tok in {"(", ")", ".", "=", "#", "-", "+", "\\", "/", ":", "~", "@", "?"}:
        return False
    if tok.isdigit() or (tok.startswith("%") and tok[1:].isdigit()):
        return False
    return True


def choose_length_edit_tokens(
    tokens,
    fraction,
    min_tokens,
    span_prob,
    length_delta_choices=None,
    length_edit_prob=0.0,
    length_edit_min_span=1,
    length_edit_max_span=8,
):
    tokens = list(tokens)
    body_len = len(tokens)
    if body_len <= 0:
        return tokens, []

    deltas = [int(round(x)) for x in parse_float_list(length_delta_choices, [0])]
    if not deltas or random.random() >= max(0.0, length_edit_prob):
        return tokens, choose_remask_positions(
            body_len, fraction, min_tokens, span_prob
        )

    delta = random.choice(deltas)
    max_span = max(1, min(body_len, int(length_edit_max_span)))
    min_span = max(1, int(length_edit_min_span))
    if delta < 0:
        min_span = max(min_span, abs(delta) + 1)
    min_span = min(min_span, max_span)
    span_len = random.randint(min_span, max_span)
    new_span_len = max(1, span_len + delta)
    max_body = max(1, len(tokens) + delta)
    new_span_len = min(new_span_len, max_body)

    starts = []
    for start in range(0, body_len - span_len + 1):
        span = tokens[start : start + span_len]
        if "." in span:
            continue
        if any(_is_length_edit_atom_token(tok) for tok in span):
            starts.append(start)
    if starts:
        start = random.choice(starts)
    else:
        start = random.randint(0, body_len - span_len)

    edited = tokens[:start] + ["<mask>"] * new_span_len + tokens[start + span_len :]
    max_body_len = max(1, body_len + max(deltas + [0]))
    edited = edited[:max_body_len]
    mask_positions = [idx + 1 for idx, tok in enumerate(edited) if tok == "<mask>"]
    return edited, mask_positions


def apply_token_edit_plan(tokens, plan):
    """Apply an explicit zero-based token span replacement plan."""
    return apply_token_edit_plans(tokens, [plan] if plan else [])


def apply_token_edit_plans(tokens, plans, return_constraints=False):
    """Apply non-overlapping token replacements in one coordinate system.

    Each plan may provide either ``replacement_len`` directly or the legacy
    ``delta`` relative to the replaced span.  Supporting several spans lets a
    constrained sampler fill every attachment point while leaving the supplied
    molecular context frozen.
    """
    tokens = list(tokens)
    if not tokens or not plans:
        return tokens, []

    normalized = []
    for plan in plans:
        if not plan:
            continue
        start = max(0, min(len(tokens) - 1, int(plan.get("start", 0))))
        stop = max(start + 1, min(len(tokens), int(plan.get("stop", start + 1))))
        if "replacement_len" in plan:
            replacement_len = max(1, int(plan["replacement_len"]))
        else:
            replacement_len = max(1, stop - start + int(plan.get("delta", 0)))
        constraint = plan.get("token_constraint")
        if constraint not in (None, "chain_atom"):
            raise ValueError(f"Unsupported token constraint: {constraint!r}")
        normalized.append((start, stop, replacement_len, constraint))

    normalized.sort(key=lambda row: (row[0], row[1]))
    for previous, current in zip(normalized, normalized[1:]):
        if previous[1] > current[0]:
            raise ValueError(f"Overlapping token edit plans: {previous} and {current}")

    edited = []
    cursor = 0
    mask_positions = []
    position_constraints = {}
    for start, stop, replacement_len, constraint in normalized:
        edited.extend(tokens[cursor:start])
        mask_start = len(edited)
        edited.extend(["<mask>"] * replacement_len)
        new_positions = list(range(mask_start + 1, mask_start + replacement_len + 1))
        mask_positions.extend(new_positions)
        if constraint is not None:
            position_constraints.update(
                {position: constraint for position in new_positions}
            )
        cursor = stop
    edited.extend(tokens[cursor:])
    if return_constraints:
        return edited, mask_positions, position_constraints
    return edited, mask_positions


@torch.no_grad()
def _sample_csdnet_fixed_local_remask(
    model,
    tk,
    seed_smiles,
    max_len,
    device,
    batch_size=64,
    n_steps=120,
    remask_fraction=0.35,
    min_remask_tokens=2,
    span_prob=0.7,
    use_fsm_check=True,
    use_rdkit_kekulize_check=True,
    rdkit_check_interval=25,
    max_sample_retries=2,
    violation_neighborhood=2,
    fsm_repair_progressive_steps=8,
    fsm_repair_prefer_localization=False,
    temperature_start=1.2,
    temperature_end=0.2,
    temperature_power=1.5,
    top_k=0,
    top_p=1.0,
    gumbel_scale=1.0,
    remask_power=1.0,
    length_delta_choices=None,
    length_edit_prob=0.0,
    length_edit_min_span=1,
    length_edit_max_span=8,
    edit_plans=None,
    local_sampler_profile=None,
    return_seed_indices=False,
    return_diagnostics=False,
):
    local_sampler_profile = resolve_local_sampler_profile(local_sampler_profile)
    progressive_length_coupled = local_sampler_profile in {
        "progressive_length_coupled",
        "conditional_progressive_refine",
        "conditional_editable_refine",
        "conditional_masked_refine",
        "task_adaptive_local",
        "task_adaptive_refine",
    }
    if progressive_length_coupled:
        profile_name = {
            "conditional_progressive_refine": "promax_fragment_conditional_refine",
            "conditional_editable_refine": "promax_fragment_editable_refine",
            "conditional_masked_refine": "promax_fragment_masked_refine",
            "task_adaptive_local": "promax_task_adaptive_local",
            "task_adaptive_refine": "promax_task_adaptive_refine",
        }.get(local_sampler_profile, "promax_progressive_length_coupled")
        profile = SAMPLER_PROFILES[profile_name]
        temperature_mode = profile.get("local_temperature_mode", "profile_absolute")
        if temperature_mode == "operator_scaled":

            def scale_short(operator_value, profile_key, base_key):
                base_value = max(abs(float(profile[base_key])), 1e-8)
                return float(operator_value) * float(profile[profile_key]) / base_value

            adaptive_temperature_start_short = scale_short(
                temperature_start,
                "adaptive_temperature_start_short",
                "temperature_start",
            )
            adaptive_temperature_end_short = scale_short(
                temperature_end,
                "adaptive_temperature_end_short",
                "temperature_end",
            )
            adaptive_temperature_power_short = scale_short(
                temperature_power,
                "adaptive_temperature_power_short",
                "temperature_power",
            )
        elif temperature_mode == "profile_absolute":
            temperature_start = profile["temperature_start"]
            temperature_end = profile["temperature_end"]
            temperature_power = profile["temperature_power"]
            adaptive_temperature_start_short = profile[
                "adaptive_temperature_start_short"
            ]
            adaptive_temperature_end_short = profile["adaptive_temperature_end_short"]
            adaptive_temperature_power_short = profile[
                "adaptive_temperature_power_short"
            ]
        else:
            raise ValueError(
                f"Unsupported local_temperature_mode: {temperature_mode!r}"
            )
        top_k = profile["top_k"]
        top_p = profile["top_p"]
        gumbel_scale = profile["gumbel_scale"]
        remask_power = profile["remask_power"]
    effective_long_temperature_start = float(temperature_start)
    effective_long_temperature_end = float(temperature_end)
    if float(gumbel_scale) < 0.0:
        raise ValueError("gumbel_scale must be non-negative")
    if float(remask_power) <= 0.0:
        raise ValueError("remask_power must be positive")
    if not 0.0 < float(top_p) <= 1.0:
        raise ValueError("top_p must be in (0, 1]")
    model.eval()
    if not seed_smiles:
        return []
    fsm_start_step = int(n_steps * 0.8)
    retry_step = int(n_steps * 0.6)
    unk_id = getattr(tk, "unk_id", tk.vocab.get("<unk>", -1))

    fsm_tracker = None
    if use_fsm_check or use_rdkit_kekulize_check:
        fsm_tracker = ValenceFSMTracker(tk)
    rdkit_checker = None
    if use_rdkit_kekulize_check:
        rdkit_checker = prepare_rdkit_kekulize_checker(tk, fsm_tracker)

    generated = []
    ordered_indices = list(range(len(seed_smiles)))
    if progressive_length_coupled:
        # The de novo profile sorts requested lengths to reduce padding. For
        # conditional sampling, seed token length is the deterministic proxy.
        ordered_indices.sort(key=lambda index: len(tokenize_smiles(seed_smiles[index])))
    for offset in range(0, len(ordered_indices), batch_size):
        original_indices = ordered_indices[offset : offset + batch_size]
        seeds = [seed_smiles[index] for index in original_indices]
        token_lists = []
        lengths = []
        mask_positions = []
        token_constraints = []
        for local_idx, smi in enumerate(seeds):
            toks = tokenize_smiles(smi)
            toks = toks[: max_len - 2]
            plan_idx = original_indices[local_idx]
            plan = (
                edit_plans[plan_idx]
                if edit_plans is not None and plan_idx < len(edit_plans)
                else None
            )
            if plan is not None:
                if isinstance(plan, (list, tuple)):
                    toks, mask_pos, constraints = apply_token_edit_plans(
                        toks,
                        plan,
                        return_constraints=True,
                    )
                else:
                    toks, mask_pos, constraints = apply_token_edit_plans(
                        toks,
                        [plan],
                        return_constraints=True,
                    )
            else:
                toks, mask_pos = choose_length_edit_tokens(
                    toks,
                    fraction=remask_fraction,
                    min_tokens=min_remask_tokens,
                    span_prob=span_prob,
                    length_delta_choices=length_delta_choices,
                    length_edit_prob=length_edit_prob,
                    length_edit_min_span=length_edit_min_span,
                    length_edit_max_span=length_edit_max_span,
                )
                constraints = {}
            toks = toks[: max_len - 2]
            mask_pos = [pos for pos in mask_pos if 0 < pos <= len(toks)]
            token_lists.append(toks)
            lengths.append(len(toks) + 2)
            mask_positions.append(mask_pos)
            token_constraints.append(constraints)

        maxL = max(lengths)
        bsz = len(seeds)
        x = torch.full((bsz, maxL), tk.pad_id, device=device, dtype=torch.long)
        frozen = torch.ones((bsz, maxL), device=device, dtype=torch.bool)
        for b, toks in enumerate(token_lists):
            body_ids = [
                tk.mask_id if tok == "<mask>" else tk.vocab.get(tok, unk_id)
                for tok in toks
            ]
            ids = [tk.bos_id] + body_ids + [tk.eos_id]
            L = len(ids)
            x[b, :L] = torch.tensor(ids, device=device, dtype=torch.long)
            mask_pos = [p for p in mask_positions[b] if 0 < p < L - 1]
            if not mask_pos:
                continue
            frozen[b, mask_pos] = False
            x[b, mask_pos] = tk.mask_id

        fillable = ~frozen & (x != tk.pad_id)
        if not fillable.any():
            continue
        output_scores = torch.zeros_like(x, dtype=torch.float)
        output_scores.masked_fill_(fillable, -math.inf)
        non_special = (x != tk.pad_id) & (x != tk.bos_id) & (x != tk.eos_id)
        valid_lens = fillable.sum(dim=1, keepdim=True).float().clamp(min=1)
        ranks_template = torch.arange(maxL, device=device).unsqueeze(0).expand(bsz, -1)
        chain_atom_positions = torch.zeros_like(x, dtype=torch.bool)
        for row, constraints in enumerate(token_constraints):
            for position, constraint in constraints.items():
                if constraint == "chain_atom" and 0 <= int(position) < maxL:
                    chain_atom_positions[row, int(position)] = True
        chain_atom_ids = [
            tk.vocab[token]
            for token in ("B", "C", "N", "O", "P", "S")
            if token in tk.vocab
        ]
        if chain_atom_positions.any() and not chain_atom_ids:
            raise RuntimeError(
                "chain_atom constraints require at least one compatible token"
            )
        chain_atom_allowed = torch.zeros(
            tk.vocab_size,
            device=device,
            dtype=torch.bool,
        )
        if chain_atom_ids:
            chain_atom_allowed[chain_atom_ids] = True

        row_lengths = torch.as_tensor(lengths, device=device, dtype=torch.float32)
        if progressive_length_coupled:
            sampling_lengths = (
                valid_lens.squeeze(1)
                if profile.get("local_sampling_uses_editable_length", False)
                else row_lengths
            )
            sampling_length_low = (
                profile.get("local_adaptive_length_low")
                if profile.get("local_sampling_uses_editable_length", False)
                else profile["adaptive_length_low"]
            )
            sampling_length_high = (
                profile.get("local_adaptive_length_high")
                if profile.get("local_sampling_uses_editable_length", False)
                else profile["adaptive_length_high"]
            )
            length_mix = _smooth_length_mix(
                sampling_lengths,
                sampling_length_low,
                sampling_length_high,
            )
            confidence_lengths = (
                valid_lens.squeeze(1)
                if profile.get("local_confidence_uses_editable_length", False)
                else row_lengths
            )
            confidence_length_mix = _smooth_length_mix(
                confidence_lengths,
                profile["adaptive_confidence_length_low"],
                profile["adaptive_confidence_length_high"],
            )

            def interpolate(short_value, long_value):
                return (
                    float(short_value)
                    + (float(long_value) - float(short_value)) * length_mix
                )

            row_temperature_start = interpolate(
                adaptive_temperature_start_short,
                temperature_start,
            )
            row_temperature_end = interpolate(
                adaptive_temperature_end_short,
                temperature_end,
            )
            row_temperature_power = interpolate(
                adaptive_temperature_power_short,
                temperature_power,
            )
            row_gumbel_scale = interpolate(
                profile["adaptive_gumbel_scale_short"],
                gumbel_scale,
            ).unsqueeze(1)
            row_remask_power = interpolate(
                profile["adaptive_remask_power_short"],
                remask_power,
            )
        else:
            confidence_length_mix = None
            row_temperature_start = torch.full_like(
                row_lengths, float(temperature_start)
            )
            row_temperature_end = torch.full_like(row_lengths, float(temperature_end))
            row_temperature_power = torch.full_like(
                row_lengths, float(temperature_power)
            )
            row_gumbel_scale = torch.full(
                (bsz, 1),
                float(gumbel_scale),
                device=device,
                dtype=torch.float32,
            )
            row_remask_power = torch.full_like(row_lengths, float(remask_power))

        step = 0
        active_rows = torch.ones(bsz, device=device, dtype=torch.bool)
        retries = torch.zeros(bsz, device=device, dtype=torch.long)
        while step < n_steps:
            if not active_rows.any():
                break
            editable_positions = fillable & active_rows.unsqueeze(1)
            sample_positions = editable_positions & (x == tk.mask_id)
            if sample_positions.any():
                amask = (x != tk.pad_id).long()
                learned_order_rate = None
                if getattr(model, "is_elastic", False):
                    diffusion_time = torch.full(
                        (x.size(0),),
                        (step + 1) / max(n_steps, 1),
                        device=x.device,
                        dtype=torch.float32,
                    )
                    elastic_output = model(
                        x,
                        amask,
                        t=diffusion_time,
                        return_aux=True,
                        rate_family="theta",
                    )
                    logits = elastic_output["logits"]
                    learned_order_rate = elastic_output["b_unmask"].float()
                else:
                    logits = model(x, amask)
                logits[:, :, tk.bos_id] = -1e9
                logits[:, :, tk.eos_id] = -1e9
                logits[:, :, tk.mask_id] = -1e9
                logits[:, :, tk.pad_id] = -1e9
                if unk_id != -1:
                    logits[:, :, unk_id] = -1e9
                if chain_atom_positions.any():
                    logits.masked_fill_(
                        chain_atom_positions.unsqueeze(-1)
                        & ~chain_atom_allowed.view(1, 1, -1),
                        -1e9,
                    )

                progress = step / max(n_steps - 1, 1)
                temperature = (
                    row_temperature_end
                    + (row_temperature_start - row_temperature_end)
                    * (1.0 - progress) ** row_temperature_power
                )
                sample_logits = _filter_sampling_logits(
                    logits / temperature[:, None, None],
                    top_k=top_k,
                    top_p=top_p,
                )
                cur_tokens = torch.distributions.Categorical(
                    logits=sample_logits
                ).sample()
                confidence_logits = sample_logits
                if confidence_length_mix is not None:
                    confidence_temperatures = (
                        _length_conditioned_confidence_temperatures(
                            sampling_temperatures=temperature,
                            length_mix=confidence_length_mix,
                            short_temperature=profile[
                                "adaptive_confidence_temperature_short"
                            ],
                        )
                    )
                    confidence_logits = logits / confidence_temperatures[:, None, None]
                lm_log_probs = F.log_softmax(confidence_logits, dim=-1)
                cur_scores = torch.gather(
                    lm_log_probs, 2, cur_tokens.unsqueeze(-1)
                ).squeeze(-1)
                if learned_order_rate is not None:
                    cur_scores = cur_scores + 0.10 * torch.log(
                        learned_order_rate.clamp_min(1e-8)
                    )
                x.masked_scatter_(sample_positions, cur_tokens[sample_positions])
                output_scores.masked_scatter_(
                    sample_positions, cur_scores[sample_positions]
                )

            should_check_fsm = (
                use_fsm_check
                and step >= fsm_start_step
                and (step % 5 == 0 or step == n_steps - 1)
            )
            should_check_rdkit = (
                rdkit_checker is not None
                and step >= fsm_start_step
                and (step % max(1, rdkit_check_interval) == 0 or step == n_steps - 1)
            )
            if should_check_fsm or should_check_rdkit:
                penalties = torch.zeros_like(output_scores)
                fsm_penalties = None
                if should_check_fsm:
                    fsm_penalties = fsm_tracker.compute_penalties(x)
                    penalties += fsm_penalties
                if should_check_rdkit:
                    chem, rdkit_focus_ids = rdkit_checker
                    rdkit_penalties = compute_rdkit_kekulize_penalties(
                        x, tk, chem, rdkit_focus_ids
                    )
                    if fsm_repair_prefer_localization and fsm_penalties is not None:
                        localized_rows = fsm_penalties.lt(0).any(dim=1)
                        rdkit_penalties.masked_fill_(
                            localized_rows.unsqueeze(1),
                            0.0,
                        )
                    penalties += rdkit_penalties
                check_positions = non_special & active_rows.unsqueeze(1)
                penalties = penalties.masked_fill(~check_positions, 0.0)
                output_scores += penalties.masked_fill(frozen, 0.0)

                violation_positions = (penalties < 0) & check_positions
                if violation_positions.any() and step != n_steps - 1:
                    repair_mask = expand_violation_mask(
                        violation_positions,
                        editable_positions,
                        radius=violation_neighborhood,
                    )
                    x.masked_fill_(repair_mask, tk.mask_id)
                    output_scores.masked_fill_(repair_mask, -math.inf)

                if step == n_steps - 1:
                    bad_rows = violation_positions.any(dim=1) & active_rows
                    retry_rows = bad_rows & (retries < max_sample_retries)
                    if retry_rows.any():
                        retries[retry_rows] += 1
                        active_rows = retry_rows
                        retry_positions = fillable & retry_rows.unsqueeze(1)
                        retry_violation_positions = (
                            violation_positions & retry_rows.unsqueeze(1)
                        )
                        retry_repair_mask = expand_violation_mask(
                            retry_violation_positions,
                            retry_positions,
                            radius=violation_neighborhood,
                        )
                        x.masked_fill_(retry_repair_mask, tk.mask_id)
                        output_scores.masked_fill_(retry_repair_mask, -math.inf)

                        retry_rate = _cosine_remask_rates(
                            retry_step + 1,
                            n_steps,
                            row_remask_power,
                        )
                        retry_cutoff_len = (valid_lens * retry_rate.unsqueeze(1)).long()
                        retry_scores = output_scores.masked_fill(
                            ~retry_positions, 1000.0
                        )
                        retry_sorted_idx = torch.argsort(retry_scores, dim=1)
                        retry_ranks = torch.zeros_like(retry_sorted_idx).scatter_(
                            1, retry_sorted_idx, ranks_template
                        )
                        retry_mask = (retry_ranks < retry_cutoff_len) & retry_positions
                        x.masked_fill_(retry_mask, tk.mask_id)
                        output_scores.masked_fill_(retry_mask, -math.inf)
                        step = retry_step
                        continue
                    active_rows = torch.zeros_like(active_rows)
                    break

            t = step + 1
            remask_rate = _cosine_remask_rates(t, n_steps, row_remask_power)
            cutoff_len = (valid_lens * remask_rate.unsqueeze(1)).long()
            scores = output_scores.masked_fill(~editable_positions, 1000.0)
            gumbel = -torch.log(-torch.log(torch.rand_like(scores) + 1e-10) + 1e-10)
            scores = scores + gumbel * row_gumbel_scale * remask_rate.unsqueeze(1)
            sorted_idx = torch.argsort(scores, dim=1)
            ranks = torch.zeros_like(sorted_idx).scatter_(1, sorted_idx, ranks_template)
            bottom_mask = (ranks < cutoff_len) & editable_positions
            x.masked_fill_(bottom_mask, tk.mask_id)
            output_scores.masked_fill_(bottom_mask, -math.inf)
            step += 1

        refinement_edits = torch.zeros(bsz, device=device, dtype=torch.long)
        refinement_diagnostics = {
            "steps": 0,
            "accepted_rows": 0,
            "accepted_edits": 0,
        }
        if (
            progressive_length_coupled
            and int(profile.get("all_position_refine_steps", 0)) > 0
        ):
            if getattr(model, "is_elastic", False):
                raise ValueError(
                    "conditional all-position refinement is not supported by "
                    "elastic checkpoints"
                )
            before_refinement = x.clone()
            x, output_scores, refinement_diagnostics = _all_position_refine_tokens(
                model=model,
                x=x,
                output_scores=output_scores,
                non_special=fillable,
                tk=tk,
                steps=profile["all_position_refine_steps"],
                corruption_start=profile["all_position_corruption_start"],
                corruption_end=profile["all_position_corruption_end"],
                corruption_power=profile["all_position_corruption_power"],
                max_edits=profile["all_position_max_edits"],
                max_total_edits=profile["all_position_max_total_edits"],
                min_logprob_gain=profile["all_position_min_logprob_gain"],
                proposal_masked=profile.get("all_position_proposal_masked", False),
                verify_masked=profile["all_position_verify_masked"],
                verify_min_logprob_gain=profile["all_position_verify_min_logprob_gain"],
                prevent_revisit=profile["all_position_prevent_revisit"],
                patience=profile["all_position_patience"],
                rdkit_each_step=profile["all_position_rdkit_each_step"],
                fsm_tracker=fsm_tracker if use_fsm_check else None,
                rdkit_checker=rdkit_checker,
                context_non_special=non_special,
            )
            refinement_edits = ((x != before_refinement) & fillable).sum(dim=1)

        sequence_rows = [
            x[row, : int(lengths[row])].detach().cpu().tolist() for row in range(bsz)
        ]
        editable_rows = [
            fillable[row, : int(lengths[row])].detach().cpu().tolist()
            for row in range(bsz)
        ]
        constraint_rows = []
        for row, length in enumerate(lengths):
            constraints = [None] * int(length)
            for position, constraint in token_constraints[row].items():
                position = int(position)
                if 0 <= position < int(length):
                    constraints[position] = constraint
            constraint_rows.append(constraints)
        repaired_sequences, fsm_repair_diagnostics = _repair_final_sequences(
            model=model,
            tk=tk,
            sequences=sequence_rows,
            device=device,
            use_fsm_check=use_fsm_check,
            use_rdkit_kekulize_check=use_rdkit_kekulize_check,
            max_sample_retries=max_sample_retries,
            violation_neighborhood=violation_neighborhood,
            temperature=temperature_end,
            top_k=top_k,
            top_p=top_p,
            editable_sequences=editable_rows,
            token_constraint_sequences=constraint_rows,
            progressive_steps=fsm_repair_progressive_steps,
            prefer_fsm_localization=fsm_repair_prefer_localization,
            hard_syntax_projection=True,
            repair_time_start=0.65,
            repair_time_end=0.95,
            return_diagnostics=True,
        )

        for i in range(bsz):
            seq = repaired_sequences[i]
            if tk.eos_id in seq:
                seq = seq[: seq.index(tk.eos_id) + 1]
            smi = tk.decode(seq).strip("'\"")
            can = canonical_smiles(smi)
            if can is not None:
                score_mask = fillable[i] & torch.isfinite(output_scores[i])
                diagnostics = {
                    "mean_log_prob": (
                        float(output_scores[i][score_mask].mean().item())
                        if score_mask.any()
                        else float("-inf")
                    ),
                    "editable_tokens": int(score_mask.sum().item()),
                    "local_sampler_profile": local_sampler_profile,
                    "local_temperature_mode": (
                        profile.get("local_temperature_mode", "profile_absolute")
                        if progressive_length_coupled
                        else "legacy"
                    ),
                    "temperature_start_long": float(effective_long_temperature_start),
                    "temperature_end_long": float(effective_long_temperature_end),
                    "confidence_uses_editable_length": bool(
                        progressive_length_coupled
                        and profile.get("local_confidence_uses_editable_length", False)
                    ),
                    "sampling_uses_editable_length": bool(
                        progressive_length_coupled
                        and profile.get("local_sampling_uses_editable_length", False)
                    ),
                    "refinement_edits": int(refinement_edits[i].item()),
                    "chain_atom_constrained_tokens": int(
                        chain_atom_positions[i].sum().item()
                    ),
                    "refinement_batch_accepted_edits": int(
                        refinement_diagnostics.get("accepted_edits", 0)
                    ),
                    "fsm_repair_prefer_localization": bool(
                        fsm_repair_prefer_localization
                    ),
                    "fsm_repair_progressive_steps": int(fsm_repair_progressive_steps),
                    "fsm_constraint_mode": (
                        "online_penalty_then_neural_repair_then_projection"
                        if use_fsm_check
                        else "disabled"
                    ),
                    "fsm_check_enabled": bool(use_fsm_check),
                    "rdkit_kekulize_check_enabled": bool(use_rdkit_kekulize_check),
                    **{
                        f"fsm_{key}": value
                        for key, value in sorted(fsm_repair_diagnostics.items())
                    },
                }
                if return_seed_indices and return_diagnostics:
                    generated.append((can, original_indices[i], diagnostics))
                elif return_seed_indices:
                    generated.append((can, original_indices[i]))
                elif return_diagnostics:
                    generated.append((can, diagnostics))
                else:
                    generated.append(can)
    return generated


def _uses_learned_insertion(plan):
    rows = plan if isinstance(plan, (list, tuple)) else [plan]
    return any(
        isinstance(row, dict)
        and row.get("length_mode") in {"learned", "learned_insertion"}
        for row in rows
    )


def _fixed_fallback_plan(plan):
    rows = plan if isinstance(plan, (list, tuple)) else [plan]
    fallback = []
    for row in rows:
        if not isinstance(row, dict):
            continue
        fixed = dict(row)
        fixed.pop("length_mode", None)
        fixed.pop("min_replacement_len", None)
        fixed.pop("max_replacement_len", None)
        fixed["replacement_len"] = max(
            1,
            int(fixed.get("stop", 1)) - int(fixed.get("start", 0)),
        )
        fixed.pop("delta", None)
        fallback.append(fixed)
    if isinstance(plan, (list, tuple)):
        return fallback
    return fallback[0] if fallback else None


@torch.no_grad()
def sample_csdnet_local_remask(
    model,
    tk,
    seed_smiles,
    max_len,
    device,
    batch_size=64,
    n_steps=120,
    remask_fraction=0.35,
    min_remask_tokens=2,
    span_prob=0.7,
    use_fsm_check=True,
    use_rdkit_kekulize_check=True,
    rdkit_check_interval=25,
    max_sample_retries=2,
    violation_neighborhood=2,
    temperature_start=1.2,
    temperature_end=0.2,
    temperature_power=1.5,
    top_k=0,
    top_p=1.0,
    gumbel_scale=1.0,
    remask_power=1.0,
    length_delta_choices=None,
    length_edit_prob=0.0,
    length_edit_min_span=1,
    length_edit_max_span=8,
    edit_plans=None,
    learned_insertion_max_growth=3,
    learned_insertion_max_shrink=3,
    learned_insertion_max_per_step=4,
    learned_insertion_fallback=True,
    learned_insertion_fallback_fraction=1.0,
    learned_insertion_recursive_gap_insertions=False,
    learned_insertion_trajectory_mode="coupled",
    learned_insertion_planning_fraction=0.5,
    learned_insertion_fill_mode="absorbing",
    learned_insertion_fill_remask_power=0.8,
    learned_insertion_fill_gumbel_scale=0.65,
    learned_insertion_nucleus_min_tokens_start=1,
    learned_insertion_nucleus_min_tokens_end=1,
    learned_insertion_rate_scale=1.0,
    learned_insertion_unmask_selection="top_prob",
    learned_insertion_deterministic_final_unmask=True,
    learned_insertion_fsm_repair_steps=8,
    learned_insertion_fsm_prefer_localization=True,
    local_sampler_profile=None,
    return_seed_indices=False,
    return_diagnostics=False,
):
    """Dispatch local edits between fixed filling and learned gap insertion.

    Existing callers are unchanged.  A plan opts into learned length only by
    setting ``length_mode=learned_insertion``; non-elastic checkpoints and
    failed learned proposals fall back to the historical equal-length fill.
    """
    seed_smiles = list(seed_smiles or [])
    if not seed_smiles:
        return []
    if not 0.0 <= float(learned_insertion_fallback_fraction) <= 1.0:
        raise ValueError("learned_insertion_fallback_fraction must be in [0, 1]")
    plans = list(edit_plans) if edit_plans is not None else [None] * len(seed_smiles)
    if len(plans) < len(seed_smiles):
        plans.extend([None] * (len(seed_smiles) - len(plans)))

    if (
        getattr(model, "is_unified", False)
        and edit_plans is not None
        and all(plan is not None for plan in plans[: len(seed_smiles)])
    ):
        return sample_unified_local_infill(
            model=model,
            tk=tk,
            seed_smiles=seed_smiles,
            edit_plans=plans,
            max_len=max_len,
            device=device,
            batch_size=batch_size,
            n_steps=n_steps,
            use_fsm_check=use_fsm_check,
            use_rdkit_kekulize_check=use_rdkit_kekulize_check,
            max_sample_retries=max_sample_retries,
            violation_neighborhood=violation_neighborhood,
            temperature_start=temperature_start,
            temperature_end=temperature_end,
            max_growth=learned_insertion_max_growth,
            max_shrink=learned_insertion_max_shrink,
            return_seed_indices=return_seed_indices,
            return_diagnostics=return_diagnostics,
        )

    learned_indices = [
        index
        for index, plan in enumerate(plans[: len(seed_smiles)])
        if _uses_learned_insertion(plan) and getattr(model, "is_elastic", False)
    ]
    fixed_indices = [
        index for index in range(len(seed_smiles)) if index not in set(learned_indices)
    ]
    collected = []

    def collect(rows, subset_indices, length_mode):
        accepted = set()
        for row in rows:
            if return_diagnostics:
                smiles, subset_index, diagnostics = row
            else:
                smiles, subset_index = row
                diagnostics = {}
            subset_index = int(subset_index)
            if not 0 <= subset_index < len(subset_indices):
                continue
            original_index = subset_indices[subset_index]
            can = canonical_smiles(smiles)
            if can is None:
                continue
            diagnostics = dict(diagnostics)
            diagnostics.setdefault("length_mode", length_mode)
            collected.append((original_index, can, diagnostics))
            accepted.add(original_index)
        return accepted

    if fixed_indices:
        fixed_rows = _sample_csdnet_fixed_local_remask(
            model=model,
            tk=tk,
            seed_smiles=[seed_smiles[index] for index in fixed_indices],
            max_len=max_len,
            device=device,
            batch_size=batch_size,
            n_steps=n_steps,
            remask_fraction=remask_fraction,
            min_remask_tokens=min_remask_tokens,
            span_prob=span_prob,
            use_fsm_check=use_fsm_check,
            use_rdkit_kekulize_check=use_rdkit_kekulize_check,
            rdkit_check_interval=rdkit_check_interval,
            max_sample_retries=max_sample_retries,
            violation_neighborhood=violation_neighborhood,
            fsm_repair_progressive_steps=(learned_insertion_fsm_repair_steps),
            fsm_repair_prefer_localization=(learned_insertion_fsm_prefer_localization),
            temperature_start=temperature_start,
            temperature_end=temperature_end,
            temperature_power=temperature_power,
            top_k=top_k,
            top_p=top_p,
            gumbel_scale=gumbel_scale,
            remask_power=remask_power,
            length_delta_choices=length_delta_choices,
            length_edit_prob=length_edit_prob,
            length_edit_min_span=length_edit_min_span,
            length_edit_max_span=length_edit_max_span,
            edit_plans=[plans[index] for index in fixed_indices],
            local_sampler_profile=local_sampler_profile,
            return_seed_indices=True,
            return_diagnostics=return_diagnostics,
        )
        collect(fixed_rows, fixed_indices, "fixed")

    learned_accepted = set()
    if learned_indices:
        learned_rows = sample_elastic_local_infill(
            model=model,
            tk=tk,
            seed_smiles=[seed_smiles[index] for index in learned_indices],
            edit_plans=[plans[index] for index in learned_indices],
            max_len=max_len,
            device=device,
            batch_size=batch_size,
            n_steps=n_steps,
            use_fsm_check=use_fsm_check,
            use_rdkit_kekulize_check=use_rdkit_kekulize_check,
            max_sample_retries=max_sample_retries,
            violation_neighborhood=violation_neighborhood,
            temperature_start=temperature_start,
            temperature_end=temperature_end,
            temperature_power=temperature_power,
            top_k=top_k,
            top_p=top_p,
            nucleus_min_tokens_start=(learned_insertion_nucleus_min_tokens_start),
            nucleus_min_tokens_end=learned_insertion_nucleus_min_tokens_end,
            max_insertions_per_step=learned_insertion_max_per_step,
            insertion_rate_scale=learned_insertion_rate_scale,
            unmask_selection=learned_insertion_unmask_selection,
            deterministic_final_unmask=(learned_insertion_deterministic_final_unmask),
            fsm_repair_progressive_steps=(learned_insertion_fsm_repair_steps),
            fsm_repair_prefer_localization=(learned_insertion_fsm_prefer_localization),
            recursive_gap_insertions=learned_insertion_recursive_gap_insertions,
            trajectory_mode=learned_insertion_trajectory_mode,
            planning_fraction=learned_insertion_planning_fraction,
            fill_mode=learned_insertion_fill_mode,
            fill_remask_power=learned_insertion_fill_remask_power,
            fill_gumbel_scale=learned_insertion_fill_gumbel_scale,
            return_seed_indices=True,
            return_diagnostics=return_diagnostics,
        )
        learned_accepted = collect(
            learned_rows,
            learned_indices,
            "learned_insertion",
        )

    fallback_indices = [
        index for index in learned_indices if index not in learned_accepted
    ]
    fallback_limit = int(
        math.ceil(len(learned_indices) * float(learned_insertion_fallback_fraction))
    )
    fallback_indices = fallback_indices[:fallback_limit]
    if learned_insertion_fallback and fallback_indices:
        fallback_rows = _sample_csdnet_fixed_local_remask(
            model=model,
            tk=tk,
            seed_smiles=[seed_smiles[index] for index in fallback_indices],
            max_len=max_len,
            device=device,
            batch_size=batch_size,
            n_steps=n_steps,
            remask_fraction=remask_fraction,
            min_remask_tokens=min_remask_tokens,
            span_prob=span_prob,
            use_fsm_check=use_fsm_check,
            use_rdkit_kekulize_check=use_rdkit_kekulize_check,
            rdkit_check_interval=rdkit_check_interval,
            max_sample_retries=max_sample_retries,
            violation_neighborhood=violation_neighborhood,
            fsm_repair_progressive_steps=(learned_insertion_fsm_repair_steps),
            fsm_repair_prefer_localization=(learned_insertion_fsm_prefer_localization),
            temperature_start=temperature_start,
            temperature_end=temperature_end,
            temperature_power=temperature_power,
            top_k=top_k,
            top_p=top_p,
            gumbel_scale=gumbel_scale,
            remask_power=remask_power,
            edit_plans=[
                _fixed_fallback_plan(plans[index]) for index in fallback_indices
            ],
            local_sampler_profile=local_sampler_profile,
            return_seed_indices=True,
            return_diagnostics=return_diagnostics,
        )
        collect(fallback_rows, fallback_indices, "fixed_fallback")

    output = []
    for original_index, smiles, diagnostics in sorted(
        collected,
        key=lambda row: row[0],
    ):
        if return_seed_indices and return_diagnostics:
            output.append((smiles, original_index, diagnostics))
        elif return_seed_indices:
            output.append((smiles, original_index))
        elif return_diagnostics:
            output.append((smiles, diagnostics))
        else:
            output.append(smiles)
    return output


class CSDNetOptimizer(BaseOptimizer):
    def __init__(self, args=None, model_bundle=None):
        super().__init__(args)
        self.model_name = f"CSDNet_{args.mode}"
        self.mode = args.mode
        self.model, self.tk, self.device = model_bundle or load_csdnet_model(args)
        if hasattr(args, "_csdnet_ref_lengths"):
            self.ref_lengths = args._csdnet_ref_lengths
        else:
            self.ref_lengths = load_ref_lengths(
                args.data_dir,
                self.tk,
                max_len=args.max_len,
                sample_n=args.ref_sample_n,
                atomic_length_prior=getattr(args, "atomic_length_prior", None),
            )
            args._csdnet_ref_lengths = self.ref_lengths
        self.summary_path = os.path.join(args.output_dir, f"summary_{args.mode}.csv")

    def _length_edit_kwargs(self):
        return {
            "length_delta_choices": getattr(self.args, "length_delta_choices", "0"),
            "length_edit_prob": getattr(self.args, "length_edit_prob", 0.0),
            "length_edit_min_span": getattr(self.args, "length_edit_min_span", 1),
            "length_edit_max_span": getattr(self.args, "length_edit_max_span", 8),
        }

    def optimize(self, oracle, config, seed=0, project="test"):
        np.random.seed(seed)
        torch.manual_seed(seed)
        random.seed(seed)
        self.seed = seed
        task_name = getattr(self.args, "oracle", None) or oracle.name
        self.oracle.task_label = f"{self.model_name}_{task_name}_{seed}"
        self._optimize(oracle, config)
        self.save_result(self.oracle.task_label)
        self.reset()

    def _optimize(self, oracle, config):
        self.oracle.assign_evaluator(oracle)
        task_name = getattr(self.args, "oracle", None) or oracle.name
        t_start = time()
        if self.mode == "motif_seeded":
            self._run_motif_seeded(task_name)
        elif self.mode == "iterative_remask":
            self._run_iterative_remask(task_name)
        elif self.mode == "iterative_remask_v2":
            self._run_iterative_remask_v2(task_name)
        elif self.mode == "iterative_remask_v3":
            self._run_iterative_remask_v3(task_name)
        elif self.mode == "iterative_remask_v4":
            self._run_iterative_remask_v4(task_name)
        elif self.mode == "iterative_remask_v5":
            self._run_iterative_remask_v5(task_name)
        elif self.mode == "iterative_remask_v6":
            self._run_iterative_remask_v6(task_name)
        elif self.mode == "iterative_remask_v7":
            self._run_iterative_remask_v7(task_name)
        elif self.mode == "iterative_remask_v8":
            self._run_iterative_remask_v8(task_name)
        elif self.mode == "iterative_remask_v9":
            self._run_iterative_remask_v9(task_name)
        elif self.mode == "iterative_remask_v9_no_prescreen":
            self._run_iterative_remask_v9(task_name)
        elif self.mode == "iterative_remask_v10":
            self._run_iterative_remask_v10(task_name)
        elif self.mode == "elastic_direct":
            self._run_elastic_direct(task_name)
        elif self.mode == "elastic_frontier":
            self._run_elastic_frontier(task_name)
        elif self.mode in ELASTIC_PRESCREEN_MODES:
            if not getattr(self.model, "is_elastic", False):
                raise RuntimeError(
                    f"{self.mode} requires an ElasticCSDNet checkpoint"
                )
            self._run_iterative_remask_v9(task_name)
        elif self.mode == "safe_frontier_final":
            self._run_iterative_remask_v9(task_name)
        elif self.mode == "iterative_remask_v9_gated":
            self._run_iterative_remask_v9(task_name)
        elif self.mode == "iterative_remask_v9_reversible":
            self._run_iterative_remask_v9(task_name)
        elif self.mode == "unified_frontier":
            self._run_iterative_remask_v9(task_name)
        elif self.mode == "unified_frontier_v2":
            self._run_iterative_remask_v9(task_name)
        elif self.mode == "unified_frontier_restored":
            self._run_iterative_remask_v9(task_name)
        else:
            raise ValueError(f"Unsupported mode: {self.mode}")
        elapsed = time() - t_start
        metrics = summarize_buffer(
            self.oracle.mol_buffer,
            max_oracle_calls=self.args.max_oracle_calls,
            freq_log=self.args.freq_log,
        )
        metrics.update(
            {
                "mode": self.mode,
                "oracle": task_name,
                "seed": self.args.seed,
                "elapsed_sec": elapsed,
                "nonzero_scores": sum(
                    1
                    for score, _ in self.oracle.mol_buffer.values()
                    if float(score) > self._nonzero_threshold()
                ),
                "best_score": max(
                    [float(score) for score, _ in self.oracle.mol_buffer.values()],
                    default=0.0,
                ),
                "unique_recorded": len(self.oracle.mol_buffer),
            }
        )
        append_summary(self.summary_path, metrics)

    def _score_and_record(self, smiles, csv_path):
        if self.oracle.finish:
            return
        can = canonical_smiles(smiles)
        if can is None:
            return
        before = len(self.oracle.mol_buffer)
        score = self.oracle(can)
        if len(self.oracle.mol_buffer) > before:
            append_csv(csv_path, can, score)
        return can, float(score)

    def _direct_length_support(self):
        lengths = np.asarray(self.ref_lengths, dtype=float)
        if lengths.size == 0:
            raise RuntimeError("elastic_direct PMO received an empty length prior")
        if not (
            0.0
            <= self.args.direct_length_quantile_low
            < self.args.direct_length_quantile_high
            <= 1.0
        ):
            raise ValueError(
                "direct length quantiles must satisfy 0 <= low < high <= 1"
            )
        self._direct_atomic_body_lengths = np.asarray(
            [max(1, int(length) - 2) for length in lengths],
            dtype=float,
        )
        lower = int(
            math.floor(
                np.quantile(
                    self._direct_atomic_body_lengths,
                    self.args.direct_length_quantile_low,
                )
            )
        )
        upper = int(
            math.ceil(
                np.quantile(
                    self._direct_atomic_body_lengths,
                    self.args.direct_length_quantile_high,
                )
            )
        )
        return max(1, lower), min(self.args.max_len - 2, upper)

    def _direct_parent_smiles(self):
        ranked = sorted(
            self.oracle.mol_buffer.items(),
            key=lambda item: (float(item[1][0]), -int(item[1][1])),
            reverse=True,
        )
        if not ranked:
            return []
        pool_size = max(1, int(self.args.direct_parent_pool_size))
        top_count = max(1, int(round(pool_size * 0.60)))
        recent_count = max(1, int(round(pool_size * 0.20)))
        selected = []
        selected_set = set()

        def add(rows, limit):
            added = 0
            for smiles, _ in rows:
                if smiles in selected_set:
                    continue
                selected_set.add(smiles)
                selected.append(smiles)
                added += 1
                if added >= limit or len(selected) >= pool_size:
                    break

        add(ranked, top_count)
        by_time = sorted(
            ranked,
            key=lambda item: int(item[1][1]),
            reverse=True,
        )
        add(by_time, recent_count)
        remaining = [item for item in ranked if item[0] not in selected_set]
        random.shuffle(remaining)
        add(remaining, max(0, pool_size - len(selected)))
        output = []
        seen = set()
        for smiles in selected:
            if smiles in seen or not tokenizable(smiles, self.tk, self.args.max_len):
                continue
            seen.add(smiles)
            output.append(smiles)
            if len(output) >= pool_size:
                break
        return output

    def _direct_prefill(
        self,
        *,
        removed,
        fixed_length,
        minimum,
        maximum,
        phase,
    ):
        """Initialize below a span/prior target and let insertion finish it."""
        settings = {
            "explore": (0.45, (0.20, 0.60), (0.12, 0.62)),
            "balanced": (0.25, (0.45, 0.80), (0.18, 0.60)),
            "exploit": (0.10, (0.70, 0.95), (0.28, 0.65)),
        }
        prior_weight, retention_range, quantile_range = settings[phase]
        allowed = self._direct_atomic_body_lengths[
            (self._direct_atomic_body_lengths >= fixed_length + minimum)
            & (self._direct_atomic_body_lengths <= fixed_length + maximum)
        ]
        if allowed.size:
            quantile = random.uniform(*quantile_range)
            prior_gap = float(np.quantile(allowed - fixed_length, quantile))
        else:
            prior_gap = float(min(max(removed, minimum), maximum))
        target_gap = min(
            max(
                (1.0 - prior_weight) * removed + prior_weight * prior_gap,
                minimum,
            ),
            maximum,
        )
        retention = random.uniform(*retention_range)
        initial = minimum + retention * max(0.0, target_gap - minimum)
        return int(min(max(round(initial), minimum), maximum))

    def _direct_edit_plan(self, smiles, phase, length_support):
        tokens = tokenize_smiles(smiles)
        if not tokens:
            return None
        settings = {
            "explore": ((0.14, 0.34), 0.34, 0.45),
            "balanced": ((0.08, 0.22), 0.25, 0.62),
            "exploit": ((0.04, 0.13), 0.15, 0.78),
        }
        (low_fraction, high_fraction), flex_fraction, peripheral_probability = settings[
            phase
        ]
        target_fraction = random.uniform(low_fraction, high_fraction)
        plan = None
        if random.random() < peripheral_probability:
            plan = adaptive_peripheral_edit_plan(
                smiles,
                random,
                target_atom_fraction=target_fraction,
                max_atom_fraction=min(0.50, high_fraction + 0.12),
                max_span_tokens=max(2, int(math.ceil(len(tokens) * 0.45))),
            )
        if plan is None:
            plan = atom_span_edit_plan(
                smiles,
                random,
                span_tokens=max(1, int(round(len(tokens) * target_fraction))),
            )
        if plan is None:
            return None

        start = int(plan["start"])
        stop = int(plan["stop"])
        removed = max(1, stop - start)
        parent_length = len(tokens)
        prior_low, prior_high = length_support
        # A parent discovered online is never invalidated merely because it is
        # near an empirical tail. The prior bounds the next move rather than
        # replacing the insertion head with a fixed requested length.
        support_low = max(3, min(prior_low, parent_length) - 4)
        support_high = min(
            self.args.max_len - 2,
            max(prior_high, parent_length) + 8,
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
        initial = self._direct_prefill(
            removed=removed,
            fixed_length=fixed_length,
            minimum=minimum,
            maximum=maximum,
            phase=phase,
        )
        return {
            "start": start,
            "stop": stop,
            "length_mode": "learned_insertion",
            "min_replacement_len": int(minimum),
            "max_replacement_len": int(maximum),
            "initial_replacement_len": int(min(initial, maximum)),
            "prior_guidance": "atomic_soft_span_retention",
        }

    def _direct_local_candidates(
        self,
        count,
        phase,
        min_size,
        max_size,
        length_support,
    ):
        parents = self._direct_parent_smiles()
        if not parents or count <= 0:
            return [], []
        proposal_count = max(
            count,
            int(math.ceil(count * self.args.direct_overgenerate_factor)),
        )
        seeds = []
        plans = []
        for _ in range(proposal_count):
            draw = random.random()
            if draw < 0.70:
                limit = min(24, len(parents))
            elif draw < 0.92:
                limit = min(80, len(parents))
            else:
                limit = len(parents)
            parent = parents[random.randrange(max(1, limit))]
            plan = self._direct_edit_plan(parent, phase, length_support)
            if plan is None:
                continue
            seeds.append(parent)
            plans.append(plan)
        if not seeds:
            return [], []

        phase_parameters = {
            "explore": (1.10, 0.78, 4),
            "balanced": (1.00, 0.68, 3),
            "exploit": (0.88, 0.55, 2),
        }
        temperature, top_p, nucleus_start = phase_parameters[phase]
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
            nucleus_min_tokens_start=nucleus_start,
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
        candidates = []
        diagnostics = []
        seen = set()
        for smiles, seed_index, row in generated:
            can = canonical_smiles(smiles)
            if (
                can is None
                or can in seen
                or can in self.oracle.mol_buffer
                or not min_size <= atom_count(can) <= max_size
                or not tokenizable(can, self.tk, self.args.max_len)
            ):
                continue
            seen.add(can)
            candidates.append(can)
            diagnostics.append(
                {
                    "parent": seeds[int(seed_index)],
                    "removed_tokens": row.get("removed_tokens"),
                    "inserted_tokens": row.get("inserted_tokens"),
                    "actual_delta": row.get("actual_delta"),
                    "initial_mask_tokens": row.get("initial_inserted_tokens"),
                }
            )
            if len(candidates) >= count:
                break
        return candidates, diagnostics

    def _direct_global_candidates(
        self,
        count,
        min_size,
        max_size,
        length_support,
    ):
        if count <= 0:
            return []
        profile = dict(SAMPLER_PROFILES["elastic_loflex"])
        profile["length_min"] = 3
        # The learned generator owns length. The empirical upper tail is only
        # a guardrail, with enough slack for PMO objectives that favour larger
        # molecules than the centre of ZINC250K.
        profile["length_max"] = min(
            self.args.max_len,
            max(
                int(length_support[1]) + 18,
                int(math.ceil(length_support[1] * 1.25)) + 2,
            ),
        )
        accepted = inspect.signature(sample_csdnet).parameters
        profile = {key: value for key, value in profile.items() if key in accepted}
        requested = max(
            count,
            int(math.ceil(count * self.args.direct_overgenerate_factor)),
        )
        generated = sample_csdnet(
            model=self.model,
            tk=self.tk,
            ref_lengths=self.ref_lengths,
            n_mol=requested,
            device=self.device,
            batch_size=self.args.batch_size,
            n_steps=self.args.direct_global_steps,
            use_fsm_check=not self.args.disable_fsm_check,
            use_rdkit_kekulize_check=not self.args.disable_rdkit_kekulize_check,
            max_sample_retries=self.args.max_sample_retries,
            violation_neighborhood=self.args.violation_neighborhood,
            **profile,
        )
        output = []
        seen = set()
        for smiles in generated:
            can = canonical_smiles(smiles)
            if (
                can is None
                or can in seen
                or can in self.oracle.mol_buffer
                or not min_size <= atom_count(can) <= max_size
                or not tokenizable(can, self.tk, self.args.max_len)
            ):
                continue
            seen.add(can)
            output.append(can)
            if len(output) >= count:
                break
        return output

    @staticmethod
    def _append_direct_diagnostic(path, row):
        fields = [
            "round",
            "calls_before",
            "calls_after",
            "phase",
            "local_requested",
            "local_generated",
            "global_requested",
            "global_generated",
            "best_before",
            "best_after",
            "empty_rounds",
            "mean_actual_delta",
            "mean_initial_mask_tokens",
        ]
        exists = os.path.exists(path)
        os.makedirs(os.path.dirname(path), exist_ok=True)
        with open(path, "a", newline="") as handle:
            writer = csv.DictWriter(handle, fieldnames=fields)
            if not exists:
                writer.writeheader()
            writer.writerow({key: row.get(key) for key in fields})

    def _run_elastic_direct(self, oracle_name):
        if not getattr(self.model, "is_elastic", False):
            raise RuntimeError(
                "elastic_direct PMO requires an ElasticCSDNet checkpoint"
            )
        if not getattr(self.args, "atomic_length_prior", None):
            raise RuntimeError(
                "elastic_direct PMO requires --atomic_length_prior; "
                "SAFE/fragment lengths are not interchangeable with atomic tokens"
            )
        length_support = self._direct_length_support()
        min_size, max_size = task_size_bounds(oracle_name)
        csv_path = os.path.join(
            self.args.output_dir,
            f"{oracle_name}_{self.args.seed}.csv",
        )
        diagnostic_path = os.path.join(
            self.args.output_dir,
            f"diagnostics_{self.mode}_{oracle_name}_{self.args.seed}.csv",
        )
        print(
            "Elastic-direct PMO: oracle-only online adaptation, "
            "no oracle-specific ZINC prescreen"
        )
        print(
            f"Atomic-token support={length_support[0]}-{length_support[1]} "
            f"body tokens; atom bounds={min_size}-{max_size}"
        )

        round_index = 0
        no_improve_rounds = 0
        empty_rounds = 0
        while not self.oracle.finish:
            round_index += 1
            calls_before = len(self.oracle.mol_buffer)
            remaining = self.args.max_oracle_calls - calls_before
            target = min(self.args.candidate_batch_size, remaining)
            before_scores = [
                float(score) for score, _ in self.oracle.mol_buffer.values()
            ]
            best_before = max(before_scores, default=0.0)
            has_positive_feedback = any(score > 1e-8 for score in before_scores)

            if calls_before < self.args.direct_bootstrap_calls:
                phase = "explore"
                local_requested = 0
                global_requested = target
            else:
                progress = calls_before / max(1, self.args.max_oracle_calls)
                if not has_positive_feedback:
                    phase = "explore"
                    global_fraction = max(0.50, self.args.direct_global_fraction)
                elif no_improve_rounds >= self.args.direct_stagnation_rounds:
                    phase = "explore"
                    global_fraction = max(0.25, self.args.direct_global_fraction)
                elif progress >= 0.72:
                    phase = "exploit"
                    global_fraction = min(0.05, self.args.direct_global_fraction)
                else:
                    phase = "balanced"
                    global_fraction = self.args.direct_global_fraction
                global_requested = int(round(target * global_fraction))
                local_requested = max(0, target - global_requested)

            local, local_diagnostics = self._direct_local_candidates(
                local_requested,
                phase,
                min_size,
                max_size,
                length_support,
            )
            global_candidates = self._direct_global_candidates(
                global_requested,
                min_size,
                max_size,
                length_support,
            )
            candidates = local + global_candidates
            seen = set()
            unique_candidates = []
            for smiles in candidates:
                if smiles in seen:
                    continue
                seen.add(smiles)
                unique_candidates.append(smiles)
            candidates = unique_candidates

            if len(candidates) < target:
                refill = self._direct_global_candidates(
                    target - len(candidates),
                    min_size,
                    max_size,
                    length_support,
                )
                for smiles in refill:
                    if smiles not in seen:
                        seen.add(smiles)
                        candidates.append(smiles)
            for smiles in candidates[:target]:
                if self.oracle.finish:
                    break
                self._score_and_record(smiles, csv_path)

            calls_after = len(self.oracle.mol_buffer)
            after_scores = [
                float(score) for score, _ in self.oracle.mol_buffer.values()
            ]
            best_after = max(after_scores, default=best_before)
            if best_after > best_before + 1e-12:
                no_improve_rounds = 0
            else:
                no_improve_rounds += 1
            if calls_after == calls_before:
                empty_rounds += 1
            else:
                empty_rounds = 0

            deltas = [
                float(row["actual_delta"])
                for row in local_diagnostics
                if row.get("actual_delta") is not None
            ]
            initial_masks = [
                float(row["initial_mask_tokens"])
                for row in local_diagnostics
                if row.get("initial_mask_tokens") is not None
            ]
            mean_delta = float(np.mean(deltas)) if deltas else 0.0
            mean_initial_masks = float(np.mean(initial_masks)) if initial_masks else 0.0
            self._append_direct_diagnostic(
                diagnostic_path,
                {
                    "round": round_index,
                    "calls_before": calls_before,
                    "calls_after": calls_after,
                    "phase": phase,
                    "local_requested": local_requested,
                    "local_generated": len(local),
                    "global_requested": global_requested,
                    "global_generated": len(global_candidates),
                    "best_before": best_before,
                    "best_after": best_after,
                    "empty_rounds": empty_rounds,
                    "mean_actual_delta": mean_delta,
                    "mean_initial_mask_tokens": mean_initial_masks,
                },
            )
            print(
                f"[Elastic-direct {round_index:03d}] phase={phase} "
                f"calls={calls_after}/{self.args.max_oracle_calls} "
                f"local={len(local)}/{local_requested} "
                f"global={len(global_candidates)}/{global_requested} "
                f"mean_initial={mean_initial_masks:.2f} "
                f"mean_delta={mean_delta:.2f} best={best_after:.4f}"
            )
            if empty_rounds >= self.args.direct_max_empty_rounds:
                raise RuntimeError(
                    "elastic_direct PMO could not produce a novel in-range "
                    f"candidate for {empty_rounds} consecutive rounds"
                )

    @staticmethod
    def _direct_top_mean(scores, top_n=10):
        values = sorted((float(value) for value in scores), reverse=True)
        if not values:
            return 0.0
        return float(np.mean(values[: max(1, min(int(top_n), len(values)))]))

    def _direct_molecular_qed(self, smiles):
        cache = getattr(self, "_direct_qed_cache", None)
        if cache is None:
            cache = {}
            self._direct_qed_cache = cache
        if smiles not in cache:
            mol = Chem.MolFromSmiles(smiles)
            cache[smiles] = float(QED.qed(mol)) if mol is not None else 0.0
        return cache[smiles]

    def _direct_frontier_parent_pools(self):
        """Build score, chemistry-aware and structural parent pools.

        QED is used only as a generic trust signal, never as an oracle-specific
        prescreen. The score elite remains the largest pool, while the other
        pools keep the search from collapsing onto one oversized lineage.
        """
        ranked = sorted(
            self.oracle.mol_buffer.items(),
            key=lambda item: (float(item[1][0]), -int(item[1][1])),
            reverse=True,
        )
        limit = max(1, int(self.args.direct_parent_pool_size))
        candidates = [
            (smiles, float(values[0]))
            for smiles, values in ranked[: max(limit * 2, 64)]
            if tokenizable(smiles, self.tk, self.args.max_len)
        ]
        if not candidates:
            return {"elite": [], "quality": [], "diverse": []}

        elite = [smiles for smiles, _ in candidates[: min(limit, 32)]]
        prior_high = int(
            math.ceil(
                np.quantile(
                    self._direct_atomic_body_lengths,
                    self.args.direct_length_quantile_high,
                )
            )
        )
        slack = max(1, int(self.args.direct_absolute_length_slack))

        def trust_score(row):
            smiles, score = row
            excess = max(0, len(tokenize_smiles(smiles)) - prior_high)
            length_penalty = min(1.0, excess / slack)
            return (
                score
                + 0.20 * self._direct_molecular_qed(smiles)
                - 0.12 * length_penalty
            )

        quality = [
            smiles
            for smiles, _ in sorted(candidates, key=trust_score, reverse=True)[:limit]
        ]

        # Greedy max-min diversity is applied only inside a score-screened
        # candidate set. It preserves independent roots without spending an
        # oracle call on a separate prescreen.
        fp_rows = []
        for smiles, score in candidates[: max(limit, 96)]:
            mol = Chem.MolFromSmiles(smiles)
            if mol is None:
                continue
            fp_rows.append(
                (
                    smiles,
                    score,
                    AllChem.GetMorganFingerprintAsBitVect(mol, 2, 2048),
                )
            )
        diverse = []
        selected_fps = []
        remaining = list(fp_rows)
        if remaining:
            first = remaining.pop(0)
            diverse.append(first[0])
            selected_fps.append(first[2])
        while remaining and len(diverse) < min(limit, 32):
            best_index = max(
                range(len(remaining)),
                key=lambda index: (
                    min(
                        1.0
                        - DataStructs.TanimotoSimilarity(remaining[index][2], selected)
                        for selected in selected_fps
                    ),
                    remaining[index][1],
                ),
            )
            smiles, _, fingerprint = remaining.pop(best_index)
            diverse.append(smiles)
            selected_fps.append(fingerprint)
        return {"elite": elite, "quality": quality, "diverse": diverse}

    def _direct_frontier_choose_parent(self, pools):
        quality_fraction = min(
            0.45,
            max(0.0, float(self.args.direct_quality_parent_fraction)),
        )
        draw = random.random()
        if draw < 0.60 and pools["elite"]:
            pool = pools["elite"]
        elif draw < 0.60 + quality_fraction and pools["quality"]:
            pool = pools["quality"]
        elif pools["diverse"]:
            pool = pools["diverse"]
        else:
            pool = pools["elite"] or pools["quality"]
        if not pool:
            return None
        # Geometric rank bias exploits the frontier without deterministically
        # cloning only its current best molecule.
        rank = min(len(pool) - 1, int(random.expovariate(0.24)))
        return pool[rank]

    def _direct_frontier_prefill(
        self,
        *,
        removed,
        minimum,
        maximum,
        phase,
    ):
        retention = {
            "explore": (0.55, 0.86),
            "balanced": (0.68, 0.92),
            "exploit": (0.82, 0.98),
        }[phase]
        target = removed * random.uniform(*retention)
        return int(min(max(round(target), minimum), maximum))

    def _direct_frontier_edit_plan(self, smiles, phase, length_support):
        tokens = tokenize_smiles(smiles)
        if not tokens:
            return None
        settings = {
            "explore": ((0.06, 0.16), 0.12, 0.62),
            "balanced": ((0.04, 0.11), 0.08, 0.76),
            "exploit": ((0.02, 0.07), 0.05, 0.88),
        }
        (low_fraction, high_fraction), flex_fraction, peripheral_probability = settings[
            phase
        ]
        target_fraction = random.uniform(low_fraction, high_fraction)
        plan = None
        if random.random() < peripheral_probability:
            plan = adaptive_peripheral_edit_plan(
                smiles,
                random,
                target_atom_fraction=target_fraction,
                max_atom_fraction=min(0.24, high_fraction + 0.06),
                max_span_tokens=max(2, int(math.ceil(len(tokens) * 0.20))),
            )
        if plan is None:
            plan = atom_span_edit_plan(
                smiles,
                random,
                span_tokens=max(1, int(round(len(tokens) * target_fraction))),
            )
        if plan is None:
            return None

        start = int(plan["start"])
        stop = int(plan["stop"])
        removed = max(1, stop - start)
        parent_length = len(tokens)
        prior_low, prior_high = length_support
        absolute_high = min(
            self.args.max_len - 2,
            int(prior_high) + max(1, int(self.args.direct_absolute_length_slack)),
        )
        max_change = max(1, int(math.ceil(parent_length * flex_fraction)))
        final_low = max(3, int(prior_low) - 4, parent_length - max_change)
        final_high = min(absolute_high, parent_length + max_change)
        fixed_length = parent_length - removed
        minimum = max(0, final_low - fixed_length)
        maximum = max(minimum, final_high - fixed_length)
        initial = self._direct_frontier_prefill(
            removed=removed,
            minimum=minimum,
            maximum=maximum,
            phase=phase,
        )
        return {
            "start": start,
            "stop": stop,
            "length_mode": "learned_insertion",
            "min_replacement_len": int(minimum),
            "max_replacement_len": int(maximum),
            "initial_replacement_len": int(initial),
            "prior_guidance": "atomic_frontier_trust_region",
        }

    def _direct_frontier_local_candidates(
        self,
        count,
        phase,
        min_size,
        max_size,
        length_support,
    ):
        if count <= 0:
            return [], []
        pools = self._direct_frontier_parent_pools()
        if not any(pools.values()):
            return [], []
        output = []
        metadata = []
        accepted = set()
        phase_mixes = {
            "explore": (("explore", 0.30), ("balanced", 0.50), ("exploit", 0.20)),
            "balanced": (("explore", 0.12), ("balanced", 0.63), ("exploit", 0.25)),
            "exploit": (("explore", 0.04), ("balanced", 0.21), ("exploit", 0.75)),
        }

        def draw_plan_phase():
            draw = random.random()
            cumulative = 0.0
            for name, weight in phase_mixes[phase]:
                cumulative += weight
                if draw <= cumulative:
                    return name
            return phase_mixes[phase][-1][0]

        generation_settings = {
            "explore": (1.00, 0.68, 4),
            "balanced": (0.90, 0.58, 3),
            "exploit": (0.80, 0.50, 2),
        }
        temperature, top_p, nucleus_start = generation_settings[phase]
        rounds = max(1, int(self.args.direct_local_rounds))
        for attempt in range(rounds):
            remaining = count - len(output)
            if remaining <= 0:
                break
            factor = max(1.0, float(self.args.direct_overgenerate_factor))
            factor = max(1.8, factor) * (1.0 + 0.35 * attempt)
            proposal_count = max(remaining, int(math.ceil(remaining * factor)))
            seeds = []
            plans = []
            plan_phases = []
            for _ in range(proposal_count):
                parent = self._direct_frontier_choose_parent(pools)
                if parent is None:
                    continue
                plan_phase = draw_plan_phase()
                plan = self._direct_frontier_edit_plan(
                    parent,
                    plan_phase,
                    length_support,
                )
                if plan is None:
                    continue
                seeds.append(parent)
                plans.append(plan)
                plan_phases.append(plan_phase)
            if not seeds:
                continue
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
                temperature_end=max(0.50, temperature * 0.72),
                temperature_power=1.0,
                top_k=0,
                top_p=top_p,
                nucleus_min_tokens_start=nucleus_start,
                nucleus_min_tokens_end=1,
                max_insertions_per_step=self.args.learned_insertion_max_per_step,
                insertion_rate_scale=1.0,
                unmask_selection="top_prob",
                deterministic_final_unmask=True,
                recursive_gap_insertions=True,
                trajectory_mode="plan_then_fill",
                planning_fraction=0.30,
                fill_mode="progressive_remask",
                fill_remask_power=1.0,
                fill_gumbel_scale=0.35,
                return_seed_indices=True,
                return_diagnostics=True,
            )
            for smiles, seed_index, row in generated:
                can = canonical_smiles(smiles)
                if (
                    can is None
                    or can in accepted
                    or can in self.oracle.mol_buffer
                    or not min_size <= atom_count(can) <= max_size
                    or not tokenizable(can, self.tk, self.args.max_len)
                ):
                    continue
                index = int(seed_index)
                accepted.add(can)
                output.append(can)
                metadata.append(
                    {
                        "source": "local",
                        "parent": seeds[index],
                        "parent_score": float(self.oracle.mol_buffer[seeds[index]][0]),
                        "plan_phase": plan_phases[index],
                        "removed_tokens": row.get("removed_tokens"),
                        "inserted_tokens": row.get("inserted_tokens"),
                        "actual_delta": row.get("actual_delta"),
                        "initial_mask_tokens": row.get("initial_inserted_tokens"),
                    }
                )
                if len(output) >= count:
                    break
        return output, metadata

    def _direct_frontier_global_candidates(
        self,
        count,
        min_size,
        max_size,
        length_support,
    ):
        if count <= 0:
            return []
        profile = dict(SAMPLER_PROFILES["elastic_loflex"])
        absolute_high = min(
            self.args.max_len - 2,
            int(length_support[1])
            + max(1, int(self.args.direct_absolute_length_slack)),
        )
        # The minimum is a public benchmark atom-count bound translated only
        # into a loose token floor. It prevents the elastic sampler from
        # repeatedly proposing tiny molecules that are rejected before the
        # oracle, especially for JNK3/GSK3B.
        profile["length_min"] = min(absolute_high + 2, max(3, min_size + 2))
        profile["length_max"] = absolute_high + 2
        accepted_parameters = inspect.signature(sample_csdnet).parameters
        profile = {
            key: value for key, value in profile.items() if key in accepted_parameters
        }
        output = []
        seen = set()
        for attempt in range(max(1, int(self.args.direct_local_rounds))):
            remaining = count - len(output)
            if remaining <= 0:
                break
            requested = max(
                remaining,
                int(
                    math.ceil(
                        remaining
                        * max(1.5, float(self.args.direct_overgenerate_factor))
                        * (1.0 + 0.25 * attempt)
                    )
                ),
            )
            generated = sample_csdnet(
                model=self.model,
                tk=self.tk,
                ref_lengths=self.ref_lengths,
                n_mol=requested,
                device=self.device,
                batch_size=self.args.batch_size,
                n_steps=self.args.direct_global_steps,
                use_fsm_check=not self.args.disable_fsm_check,
                use_rdkit_kekulize_check=not self.args.disable_rdkit_kekulize_check,
                max_sample_retries=self.args.max_sample_retries,
                violation_neighborhood=self.args.violation_neighborhood,
                **profile,
            )
            for smiles in generated:
                can = canonical_smiles(smiles)
                if (
                    can is None
                    or can in seen
                    or can in self.oracle.mol_buffer
                    or not min_size <= atom_count(can) <= max_size
                    or not tokenizable(can, self.tk, self.args.max_len)
                ):
                    continue
                seen.add(can)
                output.append(can)
                if len(output) >= count:
                    break
        return output

    @staticmethod
    def _append_direct_frontier_rows(path, rows):
        if not rows:
            return
        os.makedirs(os.path.dirname(path), exist_ok=True)
        fields = list(rows[0])
        exists = os.path.exists(path)
        with open(path, "a", newline="") as handle:
            writer = csv.DictWriter(handle, fieldnames=fields, extrasaction="ignore")
            if not exists:
                writer.writeheader()
            writer.writerows(rows)

    def _run_elastic_frontier(self, oracle_name):
        """Prescreen-free PMO search aligned to the official top-10 AUC."""
        if not getattr(self.model, "is_elastic", False):
            raise RuntimeError(
                "elastic_frontier PMO requires an ElasticCSDNet checkpoint"
            )
        if not getattr(self.args, "atomic_length_prior", None):
            raise RuntimeError("elastic_frontier PMO requires --atomic_length_prior")
        length_support = self._direct_length_support()
        min_size, max_size = task_size_bounds(oracle_name)
        csv_path = os.path.join(
            self.args.output_dir,
            f"{oracle_name}_{self.args.seed}.csv",
        )
        diagnostic_path = os.path.join(
            self.args.output_dir,
            f"diagnostics_{self.mode}_{oracle_name}_{self.args.seed}.csv",
        )
        transition_path = os.path.join(
            self.args.output_dir,
            f"transitions_{self.mode}_{oracle_name}_{self.args.seed}.csv",
        )
        print(
            "Elastic-frontier PMO: top-10 feedback, online budgeted roots, "
            "no oracle-specific molecular prescreen"
        )
        print(
            f"Atomic-token support={length_support[0]}-{length_support[1]} "
            f"body tokens; atom bounds={min_size}-{max_size}"
        )

        round_index = 0
        stagnant_rounds = 0
        empty_rounds = 0
        while not self.oracle.finish:
            round_index += 1
            calls_before = len(self.oracle.mol_buffer)
            remaining = self.args.max_oracle_calls - calls_before
            target = min(self.args.candidate_batch_size, remaining)
            before_scores = [
                float(score) for score, _ in self.oracle.mol_buffer.values()
            ]
            top10_before = self._direct_top_mean(before_scores)
            positive_rate = (
                np.mean([score > 1e-8 for score in before_scores])
                if before_scores
                else 0.0
            )

            if calls_before < self.args.direct_bootstrap_calls:
                target = min(
                    target,
                    self.args.direct_bootstrap_calls - calls_before,
                )
                phase = "explore"
                global_requested = target
                local_requested = 0
            else:
                progress = calls_before / max(1, self.args.max_oracle_calls)
                if positive_rate <= 1e-8:
                    phase = "explore"
                    global_fraction = 0.30
                elif (
                    progress < 0.20
                    or stagnant_rounds >= self.args.direct_frontier_stagnation_rounds
                ):
                    phase = "balanced"
                    global_fraction = (
                        self.args.direct_rescue_global_fraction
                        if stagnant_rounds
                        >= self.args.direct_frontier_stagnation_rounds
                        else 0.0
                    )
                else:
                    phase = "exploit"
                    global_fraction = 0.0
                global_requested = int(round(target * global_fraction))
                local_requested = max(0, target - global_requested)

            local, local_metadata = self._direct_frontier_local_candidates(
                local_requested,
                phase,
                min_size,
                max_size,
                length_support,
            )
            global_candidates = self._direct_frontier_global_candidates(
                global_requested,
                min_size,
                max_size,
                length_support,
            )
            metadata_by_smiles = {
                smiles: row for smiles, row in zip(local, local_metadata)
            }
            for smiles in global_candidates:
                metadata_by_smiles[smiles] = {
                    "source": "global",
                    "parent": "",
                    "parent_score": None,
                    "plan_phase": "root",
                    "removed_tokens": None,
                    "inserted_tokens": None,
                    "actual_delta": None,
                    "initial_mask_tokens": None,
                }
            candidates = []
            seen = set()
            for smiles in local + global_candidates:
                if smiles in seen:
                    continue
                seen.add(smiles)
                candidates.append(smiles)

            # A local shortfall is deliberately not refilled globally. Global
            # fallback is reserved for a truly empty round, while local retries
            # are free of oracle cost and preserve top-10 exploitation.
            if not candidates:
                emergency = self._direct_frontier_global_candidates(
                    target,
                    min_size,
                    max_size,
                    length_support,
                )
                for smiles in emergency:
                    if smiles in seen:
                        continue
                    seen.add(smiles)
                    candidates.append(smiles)
                    metadata_by_smiles[smiles] = {
                        "source": "emergency_global",
                        "parent": "",
                        "parent_score": None,
                        "plan_phase": "root",
                        "removed_tokens": None,
                        "inserted_tokens": None,
                        "actual_delta": None,
                        "initial_mask_tokens": None,
                    }

            evaluated = []
            for smiles in candidates[:target]:
                if self.oracle.finish:
                    break
                result = self._score_and_record(smiles, csv_path)
                if result is None:
                    continue
                child_smiles, child_score = result
                row = dict(metadata_by_smiles.get(smiles, {}))
                row.update(
                    {
                        "child_smiles": child_smiles,
                        "child_score": float(child_score),
                        "call": int(self.oracle.mol_buffer[child_smiles][1]),
                    }
                )
                evaluated.append(row)

            calls_after = len(self.oracle.mol_buffer)
            after_scores = [
                float(score) for score, _ in self.oracle.mol_buffer.values()
            ]
            top10_after = self._direct_top_mean(after_scores)
            frontier_gain = max(0.0, top10_after - top10_before)
            if frontier_gain > 1e-12:
                stagnant_rounds = 0
            else:
                stagnant_rounds += 1
            if calls_after == calls_before:
                empty_rounds += 1
            else:
                empty_rounds = 0

            threshold = (
                sorted(after_scores, reverse=True)[min(9, len(after_scores) - 1)]
                if after_scores
                else math.inf
            )
            transition_rows = []
            for row in evaluated:
                parent_score = row.get("parent_score")
                child = row["child_smiles"]
                transition_rows.append(
                    {
                        "oracle": oracle_name,
                        "call": row["call"],
                        "round": round_index,
                        "phase": phase,
                        "source": row.get("source", "unknown"),
                        "plan_phase": row.get("plan_phase", ""),
                        "parent_smiles": row.get("parent", ""),
                        "parent_score": "" if parent_score is None else parent_score,
                        "child_smiles": child,
                        "child_score": row["child_score"],
                        "delta": ""
                        if parent_score is None
                        else row["child_score"] - parent_score,
                        "entered_top10": int(row["child_score"] >= threshold - 1e-12),
                        "frontier_gain": frontier_gain,
                        "child_atoms": atom_count(child),
                        "child_tokens": len(tokenize_smiles(child)),
                        "child_qed": self._direct_molecular_qed(child),
                        "removed_tokens": row.get("removed_tokens"),
                        "inserted_tokens": row.get("inserted_tokens"),
                        "actual_length_delta": row.get("actual_delta"),
                        "initial_mask_tokens": row.get("initial_mask_tokens"),
                    }
                )
            self._append_direct_frontier_rows(transition_path, transition_rows)
            self._append_direct_frontier_rows(
                diagnostic_path,
                [
                    {
                        "round": round_index,
                        "calls_before": calls_before,
                        "calls_after": calls_after,
                        "phase": phase,
                        "local_requested": local_requested,
                        "local_generated": len(local),
                        "global_requested": global_requested,
                        "global_generated": len(global_candidates),
                        "top10_before": top10_before,
                        "top10_after": top10_after,
                        "frontier_gain": frontier_gain,
                        "positive_rate": positive_rate,
                        "stagnant_rounds": stagnant_rounds,
                        "empty_rounds": empty_rounds,
                    }
                ],
            )
            print(
                f"[Elastic-frontier {round_index:03d}] phase={phase} "
                f"calls={calls_after}/{self.args.max_oracle_calls} "
                f"local={len(local)}/{local_requested} "
                f"global={len(global_candidates)}/{global_requested} "
                f"top10={top10_after:.4f} gain={frontier_gain:.5f} "
                f"stagnant={stagnant_rounds}"
            )
            if empty_rounds >= self.args.direct_max_empty_rounds:
                raise RuntimeError(
                    "elastic_frontier PMO could not produce a novel in-range "
                    f"candidate for {empty_rounds} consecutive rounds"
                )

    def _run_motif_seeded(self, oracle_name):
        motifs = load_pmo_motifs(
            oracle_name,
            self.tk,
            max_len=self.args.max_len,
            limit=self.args.population_size,
            min_atoms=self.args.motif_min_atoms,
            max_atoms=self.args.motif_max_atoms,
        )
        csv_path = os.path.join(
            self.args.output_dir, f"{oracle_name}_{self.args.seed}.csv"
        )
        min_size, max_size = task_size_bounds(oracle_name)
        print(
            f"[motif_seeded:{oracle_name}] motifs={len(motifs)} size={min_size}-{max_size}"
        )

        while not self.oracle.finish:
            candidates, used = sample_csdnet_with_frozen_motifs(
                model=self.model,
                tk=self.tk,
                ref_lengths=self.ref_lengths,
                motifs=motifs,
                n_mol=self.args.candidate_batch_size,
                device=self.device,
                batch_size=self.args.batch_size,
                n_steps=self.args.n_steps,
                use_fsm_check=not self.args.disable_fsm_check,
                use_rdkit_kekulize_check=not self.args.disable_rdkit_kekulize_check,
                rdkit_check_interval=self.args.rdkit_check_interval,
                max_sample_retries=self.args.max_sample_retries,
                violation_neighborhood=self.args.violation_neighborhood,
                temperature_start=self.args.temperature_start,
                temperature_end=self.args.temperature_end,
                temperature_power=self.args.temperature_power,
            )
            for smi in candidates:
                if self.oracle.finish:
                    break
                atoms = atom_count(smi)
                if atoms < min_size or atoms > max_size:
                    continue
                result = self._score_and_record(smi, csv_path)
                if result is not None:
                    self._update_motif_pool(motifs, result[0], result[1])

    def _update_motif_pool(self, motifs, smiles, score):
        mol = Chem.MolFromSmiles(smiles)
        if mol is None:
            return
        known = {m["motif"] for m in motifs}
        try:
            new_motifs = extract_motifs(
                mol,
                min_atoms=self.args.motif_min_atoms,
                max_atoms=self.args.motif_max_atoms,
            )
        except Exception:
            return
        for motif in new_motifs.keys():
            if motif in known or not tokenizable(motif, self.tk, self.args.max_len - 4):
                continue
            known.add(motif)
            motifs.append(
                {
                    "motif": motif,
                    "score": float(score),
                    "support": 1.0,
                    "quality_rate": float(score),
                    "enrichment": float(score),
                    "mean_qed": 0.0,
                    "mean_sa": 0.0,
                    "motif_type": "pmo_dynamic",
                }
            )
        motifs.sort(key=lambda row: float(row["score"]), reverse=True)
        del motifs[self.args.population_size :]

    def _run_iterative_remask(self, oracle_name):
        population = load_pmo_fragments(oracle_name, self.args.population_size)
        csv_path = os.path.join(
            self.args.output_dir, f"{oracle_name}_{self.args.seed}.csv"
        )
        min_size, max_size = task_size_bounds(oracle_name)
        elites = []
        print(
            f"[iterative_remask:{oracle_name}] fragments={len(population)} size={min_size}-{max_size}"
        )

        while not self.oracle.finish:
            seeds = self._make_seed_batch(population, elites, min_size, max_size)
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
                **self._length_edit_kwargs(),
            )
            for smi in candidates:
                if self.oracle.finish:
                    break
                atoms = atom_count(smi)
                if atoms < min_size or atoms > max_size:
                    continue
                result = self._score_and_record(smi, csv_path)
                if result is None:
                    continue
                can, score = result
                elites.append((score, can))
                elites.sort(key=lambda item: item[0], reverse=True)
                del elites[self.args.elite_size :]
                self._update_fragment_population(population, can, score)

    def _make_seed_batch(self, population, elites, min_size, max_size):
        seeds = []
        attempts = 0
        while (
            len(seeds) < self.args.candidate_batch_size
            and attempts < self.args.candidate_batch_size * 80
        ):
            attempts += 1
            if elites and random.random() < self.args.elite_seed_prob:
                smi = random.choice(elites)[1]
            else:
                frag1, frag2 = random.sample([frag for _, frag in population], 2)
                smi = attach_fragments(frag1, frag2)
            can = canonical_smiles(smi) if smi else None
            if can is None:
                continue
            atoms = atom_count(can)
            if atoms < min_size or atoms > max_size:
                continue
            if not tokenizable(can, self.tk, self.args.max_len):
                continue
            seeds.append(can)
        return seeds

    def _update_fragment_population(self, population, smiles, score):
        known = {frag for _, frag in population}
        try:
            frags = local_genmol_cut(smiles)
        except Exception:
            return
        for frag in frags:
            if frag in known or Chem.MolFromSmiles(frag) is None:
                continue
            known.add(frag)
            population.append((float(score), frag))
        population.sort(key=lambda item: item[0], reverse=True)
        del population[self.args.population_size :]

    def _run_iterative_remask_v2(self, oracle_name):
        population = load_pmo_fragments(oracle_name, self.args.population_size)
        csv_path = os.path.join(
            self.args.output_dir, f"{oracle_name}_{self.args.seed}.csv"
        )
        min_size, max_size = task_size_bounds(oracle_name)
        elites = []
        recent_fps = []
        best_score = -float("inf")
        stagnant_calls = 0
        operator_stats = {
            "elite_small": 1.0,
            "elite_medium": 1.0,
            "fragment_small": 1.0,
            "fragment_medium": 1.0,
            "fragment_large": 1.0,
        }
        print(
            f"[iterative_remask_v2:{oracle_name}] fragments={len(population)} "
            f"size={min_size}-{max_size} remask={self.args.v2_remask_fractions}"
        )

        while not self.oracle.finish:
            groups = self._make_v2_seed_groups(
                population=population,
                elites=elites,
                min_size=min_size,
                max_size=max_size,
                stagnant_calls=stagnant_calls,
                operator_stats=operator_stats,
            )
            if not groups:
                continue

            calls_before = len(self.oracle.mol_buffer)
            improved = False
            for op_name, spec in groups.items():
                if self.oracle.finish or not spec["seeds"]:
                    continue
                group_best_before = best_score
                candidates = sample_csdnet_local_remask(
                    model=self.model,
                    tk=self.tk,
                    seed_smiles=spec["seeds"],
                    max_len=self.args.max_len,
                    device=self.device,
                    batch_size=self.args.batch_size,
                    n_steps=self.args.n_steps,
                    remask_fraction=spec["remask_fraction"],
                    min_remask_tokens=self.args.min_remask_tokens,
                    span_prob=self.args.span_prob,
                    use_fsm_check=not self.args.disable_fsm_check,
                    use_rdkit_kekulize_check=not self.args.disable_rdkit_kekulize_check,
                    rdkit_check_interval=self.args.rdkit_check_interval,
                    max_sample_retries=self.args.max_sample_retries,
                    violation_neighborhood=self.args.violation_neighborhood,
                    temperature_start=spec["temperature_start"],
                    temperature_end=self.args.temperature_end,
                    temperature_power=self.args.temperature_power,
                    **self._length_edit_kwargs(),
                )
                group_scores = []
                seen_batch = set()
                for smi in candidates:
                    if self.oracle.finish:
                        break
                    can = canonical_smiles(smi)
                    if can is None or can in seen_batch:
                        continue
                    seen_batch.add(can)
                    atoms = atom_count(can)
                    if atoms < min_size or atoms > max_size:
                        continue
                    fp = mol_fp_from_smiles(can)
                    if (
                        self.args.v2_near_duplicate_sim > 0
                        and max_tanimoto(fp, recent_fps)
                        >= self.args.v2_near_duplicate_sim
                    ):
                        continue
                    result = self._score_and_record(can, csv_path)
                    if result is None:
                        continue
                    can, score = result
                    group_scores.append(score)
                    if fp is not None:
                        recent_fps.append(fp)
                        if len(recent_fps) > 2000:
                            del recent_fps[: len(recent_fps) - 2000]
                    elites.append((score, can))
                    elites.sort(key=lambda item: item[0], reverse=True)
                    del elites[self.args.elite_size :]
                    self._update_fragment_population_v2(population, can, score)
                    if score > best_score + 1e-6:
                        best_score = score
                        improved = True

                self._update_v2_operator_stat(
                    operator_stats,
                    op_name,
                    group_scores,
                    group_best_before,
                )

            calls_after = len(self.oracle.mol_buffer)
            if improved:
                stagnant_calls = 0
            else:
                stagnant_calls += max(0, calls_after - calls_before)

    def _run_iterative_remask_v3(self, oracle_name):
        population = load_pmo_fragments(oracle_name, self.args.population_size)
        csv_path = os.path.join(
            self.args.output_dir, f"{oracle_name}_{self.args.seed}.csv"
        )
        diag_path = os.path.join(
            self.args.output_dir,
            f"diagnostics_{self.mode}_{oracle_name}_{self.args.seed}.csv",
        )
        min_size, max_size = task_size_bounds(oracle_name)
        elites = []
        recent_fps = []
        best_score = -float("inf")
        stagnant_calls = 0
        zero_rescue = False
        operator_stats = {
            "elite_small": 1.0,
            "elite_medium": 1.0,
            "fragment_small": 1.0,
            "fragment_medium": 1.0,
            "fragment_large": 1.0,
            "rescue_restart": 1.0,
        }
        print(
            f"[iterative_remask_v3:{oracle_name}] fragments={len(population)} "
            f"size={min_size}-{max_size} remask={self.args.v3_remask_fractions}"
        )

        while not self.oracle.finish:
            calls_before = len(self.oracle.mol_buffer)
            nonzero_before = self._nonzero_count()
            zero_rescue = (
                nonzero_before == 0
                and calls_before >= self.args.v3_zero_rescue_patience
            )
            groups = self._make_v3_seed_groups(
                population=population,
                elites=elites,
                min_size=min_size,
                max_size=max_size,
                stagnant_calls=stagnant_calls,
                zero_rescue=zero_rescue,
                operator_stats=operator_stats,
            )
            if not groups:
                continue

            improved = False
            for op_name, spec in groups.items():
                if self.oracle.finish or not spec["seeds"]:
                    continue
                group_best_before = best_score
                candidates = sample_csdnet_local_remask(
                    model=self.model,
                    tk=self.tk,
                    seed_smiles=spec["seeds"],
                    max_len=self.args.max_len,
                    device=self.device,
                    batch_size=self.args.batch_size,
                    n_steps=self.args.n_steps,
                    remask_fraction=spec["remask_fraction"],
                    min_remask_tokens=self.args.min_remask_tokens,
                    span_prob=spec["span_prob"],
                    use_fsm_check=not self.args.disable_fsm_check,
                    use_rdkit_kekulize_check=not self.args.disable_rdkit_kekulize_check,
                    rdkit_check_interval=self.args.rdkit_check_interval,
                    max_sample_retries=self.args.max_sample_retries,
                    violation_neighborhood=self.args.violation_neighborhood,
                    temperature_start=spec["temperature_start"],
                    temperature_end=self.args.temperature_end,
                    temperature_power=self.args.temperature_power,
                    **self._length_edit_kwargs(),
                )
                group_scores = []
                seen_batch = set()
                for smi in candidates:
                    if self.oracle.finish:
                        break
                    can = canonical_smiles(smi)
                    if can is None or can in seen_batch:
                        continue
                    seen_batch.add(can)
                    atoms = atom_count(can)
                    if atoms < min_size or atoms > max_size:
                        continue
                    fp = mol_fp_from_smiles(can)
                    near_dup = (
                        self.args.v3_near_duplicate_sim > 0
                        and max_tanimoto(fp, recent_fps)
                        >= self.args.v3_near_duplicate_sim
                    )
                    if near_dup and not zero_rescue:
                        continue
                    result = self._score_and_record(can, csv_path)
                    if result is None:
                        continue
                    can, score = result
                    group_scores.append(score)
                    if fp is not None:
                        recent_fps.append(fp)
                        if len(recent_fps) > self.args.v3_recent_memory:
                            del recent_fps[
                                : len(recent_fps) - self.args.v3_recent_memory
                            ]
                    if (
                        score > self.args.v3_nonzero_threshold
                        or len(elites) < self.args.elite_size // 3
                    ):
                        elites.append((score, can))
                        elites.sort(key=lambda item: item[0], reverse=True)
                        del elites[self.args.elite_size :]
                    self._update_fragment_population_v2(population, can, score)
                    if score > best_score + 1e-6:
                        best_score = score
                        improved = True

                self._update_v2_operator_stat(
                    operator_stats,
                    op_name,
                    group_scores,
                    group_best_before,
                )

            calls_after = len(self.oracle.mol_buffer)
            if improved:
                stagnant_calls = 0
            else:
                stagnant_calls += max(0, calls_after - calls_before)
            self._append_v3_diagnostics(
                diag_path,
                oracle_name=oracle_name,
                calls=calls_after,
                best_score=max(best_score, 0.0),
                nonzero=self._nonzero_count(),
                elites=len(elites),
                population=len(population),
                stagnant_calls=stagnant_calls,
                zero_rescue=zero_rescue,
                operator_stats=operator_stats,
            )

    def _run_iterative_remask_v4(self, oracle_name):
        population = load_pmo_fragments(oracle_name, self.args.population_size)
        motifs = load_pmo_motifs(
            oracle_name,
            self.tk,
            max_len=self.args.max_len,
            limit=self.args.v4_motif_pool_size,
            min_atoms=self.args.motif_min_atoms,
            max_atoms=self.args.motif_max_atoms,
        )
        csv_path = os.path.join(
            self.args.output_dir, f"{oracle_name}_{self.args.seed}.csv"
        )
        diag_path = os.path.join(
            self.args.output_dir,
            f"diagnostics_{self.mode}_{oracle_name}_{self.args.seed}.csv",
        )
        min_size, max_size = task_size_bounds(oracle_name)
        elites = []
        diverse = []
        recent_fps = []
        best_score = -float("inf")
        stagnant_calls = 0
        operator_stats = {
            "elite_small": 1.0,
            "elite_medium": 1.0,
            "diverse_medium": 1.0,
            "motif_seeded": 1.0,
            "fragment_medium": 1.0,
            "fragment_large": 1.0,
            "rescue_large": 1.0,
        }
        print(
            f"[iterative_remask_v4:{oracle_name}] fragments={len(population)} "
            f"motifs={len(motifs)} size={min_size}-{max_size} "
            f"remask={self.args.v4_remask_fractions}"
        )

        while not self.oracle.finish:
            calls_before = len(self.oracle.mol_buffer)
            nonzero_before = self._nonzero_count()
            zero_rescue = (
                nonzero_before == 0
                and calls_before >= self.args.v4_zero_rescue_patience
            )
            rescue = (
                zero_rescue or stagnant_calls >= self.args.v4_stagnation_rescue_patience
            )
            groups = self._make_v4_seed_groups(
                oracle_name=oracle_name,
                population=population,
                elites=elites,
                diverse=diverse,
                motifs=motifs,
                min_size=min_size,
                max_size=max_size,
                rescue=rescue,
                zero_rescue=zero_rescue,
                operator_stats=operator_stats,
            )
            if not groups:
                stagnant_calls += self.args.candidate_batch_size
                continue

            improved = False
            for op_name, spec in groups.items():
                if self.oracle.finish:
                    break
                group_best_before = best_score
                if spec.get("motif_seeded"):
                    if spec["n_mol"] <= 0:
                        continue
                    candidates, _ = sample_csdnet_with_frozen_motifs(
                        model=self.model,
                        tk=self.tk,
                        ref_lengths=self.ref_lengths,
                        motifs=motifs,
                        n_mol=spec["n_mol"],
                        device=self.device,
                        batch_size=self.args.batch_size,
                        n_steps=self.args.n_steps,
                        use_fsm_check=not self.args.disable_fsm_check,
                        use_rdkit_kekulize_check=not self.args.disable_rdkit_kekulize_check,
                        rdkit_check_interval=self.args.rdkit_check_interval,
                        max_sample_retries=self.args.max_sample_retries,
                        violation_neighborhood=self.args.violation_neighborhood,
                        temperature_start=spec["temperature_start"],
                        temperature_end=self.args.temperature_end,
                        temperature_power=self.args.temperature_power,
                    )
                else:
                    if not spec["seeds"]:
                        continue
                    candidates = sample_csdnet_local_remask(
                        model=self.model,
                        tk=self.tk,
                        seed_smiles=spec["seeds"],
                        max_len=self.args.max_len,
                        device=self.device,
                        batch_size=self.args.batch_size,
                        n_steps=self.args.n_steps,
                        remask_fraction=spec["remask_fraction"],
                        min_remask_tokens=self.args.min_remask_tokens,
                        span_prob=spec["span_prob"],
                        use_fsm_check=not self.args.disable_fsm_check,
                        use_rdkit_kekulize_check=not self.args.disable_rdkit_kekulize_check,
                        rdkit_check_interval=self.args.rdkit_check_interval,
                        max_sample_retries=self.args.max_sample_retries,
                        violation_neighborhood=self.args.violation_neighborhood,
                        temperature_start=spec["temperature_start"],
                        temperature_end=self.args.temperature_end,
                        temperature_power=self.args.temperature_power,
                        **self._length_edit_kwargs(),
                    )

                group_scores = []
                seen_batch = set()
                for smi in candidates:
                    if self.oracle.finish:
                        break
                    can = canonical_smiles(smi)
                    if (
                        can is None
                        or can in seen_batch
                        or can in self.oracle.mol_buffer
                    ):
                        continue
                    seen_batch.add(can)
                    atoms = atom_count(can)
                    if atoms < min_size or atoms > max_size:
                        continue
                    if not tokenizable(can, self.tk, self.args.max_len):
                        continue
                    fp = mol_fp_from_smiles(can)
                    near_dup = (
                        self.args.v4_near_duplicate_sim > 0
                        and max_tanimoto(fp, recent_fps)
                        >= self.args.v4_near_duplicate_sim
                    )
                    if near_dup and not rescue:
                        continue
                    result = self._score_and_record(can, csv_path)
                    if result is None:
                        continue
                    can, score = result
                    group_scores.append(score)
                    if fp is not None:
                        recent_fps.append(fp)
                        if len(recent_fps) > self.args.v4_recent_memory:
                            del recent_fps[
                                : len(recent_fps) - self.args.v4_recent_memory
                            ]
                    self._update_v4_archives(
                        population=population,
                        motifs=motifs,
                        elites=elites,
                        diverse=diverse,
                        smiles=can,
                        score=score,
                    )
                    if score > best_score + 1e-6:
                        best_score = score
                        improved = True

                self._update_v2_operator_stat(
                    operator_stats,
                    op_name,
                    group_scores,
                    group_best_before,
                )

            calls_after = len(self.oracle.mol_buffer)
            if improved:
                stagnant_calls = 0
            else:
                stagnant_calls += max(1, calls_after - calls_before)
            self._append_v4_diagnostics(
                diag_path,
                oracle_name=oracle_name,
                calls=calls_after,
                best_score=max(best_score, 0.0),
                nonzero=self._nonzero_count(),
                elites=len(elites),
                diverse=len(diverse),
                motifs=len(motifs),
                population=len(population),
                stagnant_calls=stagnant_calls,
                rescue=rescue,
                zero_rescue=zero_rescue,
                operator_stats=operator_stats,
            )

    def _run_iterative_remask_v5(self, oracle_name):
        population = load_pmo_fragments(oracle_name, self.args.population_size)
        motifs = load_pmo_motifs(
            oracle_name,
            self.tk,
            max_len=self.args.max_len,
            limit=self.args.v5_motif_pool_size,
            min_atoms=self.args.motif_min_atoms,
            max_atoms=self.args.motif_max_atoms,
        )
        csv_path = os.path.join(
            self.args.output_dir, f"{oracle_name}_{self.args.seed}.csv"
        )
        diag_path = os.path.join(
            self.args.output_dir,
            f"diagnostics_{self.mode}_{oracle_name}_{self.args.seed}.csv",
        )
        min_size, max_size = task_size_bounds(oracle_name)
        elites = []
        diverse = []
        recent_fps = []
        best_score = -float("inf")
        stagnant_calls = 0
        last_state = "warmup"
        operator_stats = {
            "elite_tiny": 1.0,
            "elite_small": 1.0,
            "elite_medium": 1.0,
            "diverse_medium": 1.0,
            "motif_seeded": 1.0,
            "fragment_medium": 1.0,
            "fragment_large": 1.0,
            "rescue_large": 1.0,
        }
        print(
            f"[iterative_remask_v5:{oracle_name}] fragments={len(population)} "
            f"motifs={len(motifs)} size={min_size}-{max_size} "
            f"remask={self.args.v5_remask_fractions}"
        )

        while not self.oracle.finish:
            calls_before = len(self.oracle.mol_buffer)
            state, state_metrics = self._v5_feedback_state(
                calls=calls_before,
                stagnant_calls=stagnant_calls,
                best_score=max(best_score, 0.0),
            )
            last_state = state
            groups = self._make_v5_seed_groups(
                population=population,
                elites=elites,
                diverse=diverse,
                motifs=motifs,
                min_size=min_size,
                max_size=max_size,
                state=state,
                operator_stats=operator_stats,
            )
            if not groups:
                stagnant_calls += self.args.candidate_batch_size
                continue

            improved = False
            for op_name, spec in groups.items():
                if self.oracle.finish:
                    break
                group_best_before = best_score
                if spec.get("motif_seeded"):
                    if spec["n_mol"] <= 0:
                        continue
                    candidates, _ = sample_csdnet_with_frozen_motifs(
                        model=self.model,
                        tk=self.tk,
                        ref_lengths=self.ref_lengths,
                        motifs=motifs,
                        n_mol=spec["n_mol"],
                        device=self.device,
                        batch_size=self.args.batch_size,
                        n_steps=self.args.n_steps,
                        use_fsm_check=not self.args.disable_fsm_check,
                        use_rdkit_kekulize_check=not self.args.disable_rdkit_kekulize_check,
                        rdkit_check_interval=self.args.rdkit_check_interval,
                        max_sample_retries=self.args.max_sample_retries,
                        violation_neighborhood=self.args.violation_neighborhood,
                        temperature_start=spec["temperature_start"],
                        temperature_end=self.args.temperature_end,
                        temperature_power=self.args.temperature_power,
                    )
                else:
                    if not spec["seeds"]:
                        continue
                    candidates = sample_csdnet_local_remask(
                        model=self.model,
                        tk=self.tk,
                        seed_smiles=spec["seeds"],
                        max_len=self.args.max_len,
                        device=self.device,
                        batch_size=self.args.batch_size,
                        n_steps=self.args.n_steps,
                        remask_fraction=spec["remask_fraction"],
                        min_remask_tokens=self.args.min_remask_tokens,
                        span_prob=spec["span_prob"],
                        use_fsm_check=not self.args.disable_fsm_check,
                        use_rdkit_kekulize_check=not self.args.disable_rdkit_kekulize_check,
                        rdkit_check_interval=self.args.rdkit_check_interval,
                        max_sample_retries=self.args.max_sample_retries,
                        violation_neighborhood=self.args.violation_neighborhood,
                        temperature_start=spec["temperature_start"],
                        temperature_end=self.args.temperature_end,
                        temperature_power=self.args.temperature_power,
                        **self._length_edit_kwargs(),
                    )

                group_scores = []
                seen_batch = set()
                for smi in candidates:
                    if self.oracle.finish:
                        break
                    can = canonical_smiles(smi)
                    if (
                        can is None
                        or can in seen_batch
                        or can in self.oracle.mol_buffer
                    ):
                        continue
                    seen_batch.add(can)
                    atoms = atom_count(can)
                    if atoms < min_size or atoms > max_size:
                        continue
                    if not tokenizable(can, self.tk, self.args.max_len):
                        continue
                    fp = mol_fp_from_smiles(can)
                    near_dup = (
                        self.args.v5_near_duplicate_sim > 0
                        and max_tanimoto(fp, recent_fps)
                        >= self.args.v5_near_duplicate_sim
                    )
                    # In exploit/refine/sparse modes, near neighbours of a good hit
                    # are useful because PMO rewards top-k accumulation.
                    if near_dup and state in {"balanced", "explore"}:
                        continue
                    result = self._score_and_record(can, csv_path)
                    if result is None:
                        continue
                    can, score = result
                    group_scores.append(score)
                    if fp is not None:
                        recent_fps.append(fp)
                        if len(recent_fps) > self.args.v5_recent_memory:
                            del recent_fps[
                                : len(recent_fps) - self.args.v5_recent_memory
                            ]
                    self._update_v5_archives(
                        population=population,
                        motifs=motifs,
                        elites=elites,
                        diverse=diverse,
                        smiles=can,
                        score=score,
                    )
                    if score > best_score + 1e-6:
                        best_score = score
                        improved = True

                self._update_v2_operator_stat(
                    operator_stats,
                    op_name,
                    group_scores,
                    group_best_before,
                )

            calls_after = len(self.oracle.mol_buffer)
            if improved:
                stagnant_calls = 0
            else:
                stagnant_calls += max(1, calls_after - calls_before)
            _, post_metrics = self._v5_feedback_state(
                calls=calls_after,
                stagnant_calls=stagnant_calls,
                best_score=max(best_score, 0.0),
            )
            self._append_v5_diagnostics(
                diag_path,
                oracle_name=oracle_name,
                calls=calls_after,
                state=last_state,
                best_score=max(best_score, 0.0),
                nonzero=self._nonzero_count(),
                avg_top10=post_metrics.get("avg_top10", 0.0),
                auc_top10=post_metrics.get("auc_top10", 0.0),
                nonzero_rate=post_metrics.get("nonzero_rate", 0.0),
                late_gap=post_metrics.get("late_gap", 0.0),
                elites=len(elites),
                diverse=len(diverse),
                motifs=len(motifs),
                population=len(population),
                stagnant_calls=stagnant_calls,
                operator_stats=operator_stats,
            )

    def _run_iterative_remask_v6(self, oracle_name):
        population = load_pmo_fragments(oracle_name, self.args.population_size)
        motifs = load_pmo_motifs(
            oracle_name,
            self.tk,
            max_len=self.args.max_len,
            limit=self.args.v5_motif_pool_size,
            min_atoms=self.args.motif_min_atoms,
            max_atoms=self.args.motif_max_atoms,
        )
        csv_path = os.path.join(
            self.args.output_dir, f"{oracle_name}_{self.args.seed}.csv"
        )
        diag_path = os.path.join(
            self.args.output_dir,
            f"diagnostics_{self.mode}_{oracle_name}_{self.args.seed}.csv",
        )
        min_size, max_size = task_size_bounds(oracle_name)
        elites = []
        diverse = []
        recent_fps = []
        best_score = -float("inf")
        stagnant_calls = 0
        operators = [
            "elite_tiny",
            "elite_small",
            "elite_medium",
            "diverse_medium",
            "motif_seeded",
            "fragment_medium",
            "fragment_large",
            "rescue_large",
        ]
        arm_stats = {op: {"ema": 0.50, "pulls": 0.0} for op in operators}
        print(
            f"[iterative_remask_v6:{oracle_name}] fragments={len(population)} "
            f"motifs={len(motifs)} size={min_size}-{max_size} "
            f"remask={self.args.v5_remask_fractions} bandit=ucb_ema"
        )

        while not self.oracle.finish:
            calls_before = len(self.oracle.mol_buffer)
            state, _ = self._v5_feedback_state(
                calls=calls_before,
                stagnant_calls=stagnant_calls,
                best_score=max(best_score, 0.0),
            )
            operator_multipliers = self._v6_operator_multipliers(
                state=state,
                arm_stats=arm_stats,
                has_elites=bool(elites),
                has_diverse=bool(diverse),
                has_motifs=bool(motifs),
            )
            groups = self._make_v5_seed_groups(
                population=population,
                elites=elites,
                diverse=diverse,
                motifs=motifs,
                min_size=min_size,
                max_size=max_size,
                state=state,
                operator_stats=operator_multipliers,
            )
            if not groups:
                stagnant_calls += self.args.candidate_batch_size
                continue

            improved = False
            for op_name, spec in groups.items():
                if self.oracle.finish:
                    break
                group_best_before = best_score
                if spec.get("motif_seeded"):
                    if spec["n_mol"] <= 0:
                        continue
                    candidates, _ = sample_csdnet_with_frozen_motifs(
                        model=self.model,
                        tk=self.tk,
                        ref_lengths=self.ref_lengths,
                        motifs=motifs,
                        n_mol=spec["n_mol"],
                        device=self.device,
                        batch_size=self.args.batch_size,
                        n_steps=self.args.n_steps,
                        use_fsm_check=not self.args.disable_fsm_check,
                        use_rdkit_kekulize_check=not self.args.disable_rdkit_kekulize_check,
                        rdkit_check_interval=self.args.rdkit_check_interval,
                        max_sample_retries=self.args.max_sample_retries,
                        violation_neighborhood=self.args.violation_neighborhood,
                        temperature_start=spec["temperature_start"],
                        temperature_end=self.args.temperature_end,
                        temperature_power=self.args.temperature_power,
                    )
                else:
                    if not spec["seeds"]:
                        continue
                    candidates = sample_csdnet_local_remask(
                        model=self.model,
                        tk=self.tk,
                        seed_smiles=spec["seeds"],
                        max_len=self.args.max_len,
                        device=self.device,
                        batch_size=self.args.batch_size,
                        n_steps=self.args.n_steps,
                        remask_fraction=spec["remask_fraction"],
                        min_remask_tokens=self.args.min_remask_tokens,
                        span_prob=spec["span_prob"],
                        use_fsm_check=not self.args.disable_fsm_check,
                        use_rdkit_kekulize_check=not self.args.disable_rdkit_kekulize_check,
                        rdkit_check_interval=self.args.rdkit_check_interval,
                        max_sample_retries=self.args.max_sample_retries,
                        violation_neighborhood=self.args.violation_neighborhood,
                        temperature_start=spec["temperature_start"],
                        temperature_end=self.args.temperature_end,
                        temperature_power=self.args.temperature_power,
                        **self._length_edit_kwargs(),
                    )

                group_scores = []
                seen_batch = set()
                for smi in candidates:
                    if self.oracle.finish:
                        break
                    can = canonical_smiles(smi)
                    if (
                        can is None
                        or can in seen_batch
                        or can in self.oracle.mol_buffer
                    ):
                        continue
                    seen_batch.add(can)
                    atoms = atom_count(can)
                    if atoms < min_size or atoms > max_size:
                        continue
                    if not tokenizable(can, self.tk, self.args.max_len):
                        continue
                    fp = mol_fp_from_smiles(can)
                    near_dup = (
                        self.args.v5_near_duplicate_sim > 0
                        and max_tanimoto(fp, recent_fps)
                        >= self.args.v5_near_duplicate_sim
                    )
                    if near_dup and state in {"warmup", "balanced", "explore"}:
                        continue
                    result = self._score_and_record(can, csv_path)
                    if result is None:
                        continue
                    can, score = result
                    group_scores.append(score)
                    if fp is not None:
                        recent_fps.append(fp)
                        if len(recent_fps) > self.args.v5_recent_memory:
                            del recent_fps[
                                : len(recent_fps) - self.args.v5_recent_memory
                            ]
                    self._update_v5_archives(
                        population=population,
                        motifs=motifs,
                        elites=elites,
                        diverse=diverse,
                        smiles=can,
                        score=score,
                    )
                    if score > best_score + 1e-6:
                        best_score = score
                        improved = True

                self._update_v6_arm_stat(
                    arm_stats=arm_stats,
                    op_name=op_name,
                    scores=group_scores,
                    best_before=group_best_before,
                )

            calls_after = len(self.oracle.mol_buffer)
            if improved:
                stagnant_calls = 0
            else:
                stagnant_calls += max(1, calls_after - calls_before)
            _, post_metrics = self._v5_feedback_state(
                calls=calls_after,
                stagnant_calls=stagnant_calls,
                best_score=max(best_score, 0.0),
            )
            self._append_v5_diagnostics(
                diag_path,
                oracle_name=oracle_name,
                calls=calls_after,
                state=state,
                best_score=max(best_score, 0.0),
                nonzero=self._nonzero_count(),
                avg_top10=post_metrics.get("avg_top10", 0.0),
                auc_top10=post_metrics.get("auc_top10", 0.0),
                nonzero_rate=post_metrics.get("nonzero_rate", 0.0),
                late_gap=post_metrics.get("late_gap", 0.0),
                elites=len(elites),
                diverse=len(diverse),
                motifs=len(motifs),
                population=len(population),
                stagnant_calls=stagnant_calls,
                operator_stats=self._v6_diag_operator_stats(arm_stats),
            )

    def _run_iterative_remask_v7(self, oracle_name):
        population = load_pmo_fragments(oracle_name, self.args.population_size)
        motifs = load_pmo_motifs(
            oracle_name,
            self.tk,
            max_len=self.args.max_len,
            limit=self.args.v5_motif_pool_size,
            min_atoms=self.args.motif_min_atoms,
            max_atoms=self.args.motif_max_atoms,
        )
        csv_path = os.path.join(
            self.args.output_dir, f"{oracle_name}_{self.args.seed}.csv"
        )
        diag_path = os.path.join(
            self.args.output_dir,
            f"diagnostics_{self.mode}_{oracle_name}_{self.args.seed}.csv",
        )
        min_size, max_size = task_size_bounds(oracle_name)
        elites = []
        diverse = []
        recent_fps = []
        best_score = -float("inf")
        stagnant_calls = 0
        operators = [
            "elite_tiny",
            "elite_small",
            "elite_medium",
            "diverse_medium",
            "motif_seeded",
            "fragment_medium",
            "fragment_large",
            "rescue_large",
            "length_shrink_rescue",
            "length_expand_rescue",
        ]
        arm_stats = {op: {"ema": 0.50, "pulls": 0.0} for op in operators}
        no_length = {
            "length_delta_choices": "0",
            "length_edit_prob": 0.0,
            "length_edit_min_span": 1,
            "length_edit_max_span": 1,
        }
        print(
            f"[iterative_remask_v7:{oracle_name}] fragments={len(population)} "
            f"motifs={len(motifs)} size={min_size}-{max_size} "
            f"remask={self.args.v5_remask_fractions} reward=top10_auc "
            f"length_rescue={self.args.v7_length_rescue_weight}"
        )

        while not self.oracle.finish:
            calls_before = len(self.oracle.mol_buffer)
            state, _ = self._v5_feedback_state(
                calls=calls_before,
                stagnant_calls=stagnant_calls,
                best_score=max(best_score, 0.0),
            )
            operator_multipliers = self._v7_operator_multipliers(
                state=state,
                arm_stats=arm_stats,
                has_elites=bool(elites),
                has_diverse=bool(diverse),
                has_motifs=bool(motifs),
                stagnant_calls=stagnant_calls,
            )
            groups = self._make_v7_seed_groups(
                population=population,
                elites=elites,
                diverse=diverse,
                motifs=motifs,
                min_size=min_size,
                max_size=max_size,
                state=state,
                stagnant_calls=stagnant_calls,
                operator_stats=operator_multipliers,
            )
            if not groups:
                stagnant_calls += self.args.candidate_batch_size
                continue

            improved = False
            for op_name, spec in groups.items():
                if self.oracle.finish:
                    break
                before_metrics = summarize_buffer(
                    self.oracle.mol_buffer,
                    max_oracle_calls=self.args.max_oracle_calls,
                    freq_log=self.args.freq_log,
                )
                if spec.get("motif_seeded"):
                    if spec["n_mol"] <= 0:
                        continue
                    candidates, _ = sample_csdnet_with_frozen_motifs(
                        model=self.model,
                        tk=self.tk,
                        ref_lengths=self.ref_lengths,
                        motifs=motifs,
                        n_mol=spec["n_mol"],
                        device=self.device,
                        batch_size=self.args.batch_size,
                        n_steps=self.args.n_steps,
                        use_fsm_check=not self.args.disable_fsm_check,
                        use_rdkit_kekulize_check=not self.args.disable_rdkit_kekulize_check,
                        rdkit_check_interval=self.args.rdkit_check_interval,
                        max_sample_retries=self.args.max_sample_retries,
                        violation_neighborhood=self.args.violation_neighborhood,
                        temperature_start=spec["temperature_start"],
                        temperature_end=self.args.temperature_end,
                        temperature_power=self.args.temperature_power,
                    )
                else:
                    if not spec["seeds"]:
                        continue
                    length_kwargs = spec.get("length_kwargs", no_length)
                    candidates = sample_csdnet_local_remask(
                        model=self.model,
                        tk=self.tk,
                        seed_smiles=spec["seeds"],
                        max_len=self.args.max_len,
                        device=self.device,
                        batch_size=self.args.batch_size,
                        n_steps=self.args.n_steps,
                        remask_fraction=spec["remask_fraction"],
                        min_remask_tokens=self.args.min_remask_tokens,
                        span_prob=spec["span_prob"],
                        use_fsm_check=not self.args.disable_fsm_check,
                        use_rdkit_kekulize_check=not self.args.disable_rdkit_kekulize_check,
                        rdkit_check_interval=self.args.rdkit_check_interval,
                        max_sample_retries=self.args.max_sample_retries,
                        violation_neighborhood=self.args.violation_neighborhood,
                        temperature_start=spec["temperature_start"],
                        temperature_end=self.args.temperature_end,
                        temperature_power=self.args.temperature_power,
                        **length_kwargs,
                    )

                group_scores = []
                seen_batch = set()
                for smi in candidates:
                    if self.oracle.finish:
                        break
                    can = canonical_smiles(smi)
                    if (
                        can is None
                        or can in seen_batch
                        or can in self.oracle.mol_buffer
                    ):
                        continue
                    seen_batch.add(can)
                    atoms = atom_count(can)
                    if atoms < min_size or atoms > max_size:
                        continue
                    if not tokenizable(can, self.tk, self.args.max_len):
                        continue
                    fp = mol_fp_from_smiles(can)
                    near_dup = (
                        self.args.v5_near_duplicate_sim > 0
                        and max_tanimoto(fp, recent_fps)
                        >= self.args.v5_near_duplicate_sim
                    )
                    if near_dup and state in {"warmup", "balanced", "explore"}:
                        continue
                    result = self._score_and_record(can, csv_path)
                    if result is None:
                        continue
                    can, score = result
                    group_scores.append(score)
                    if fp is not None:
                        recent_fps.append(fp)
                        if len(recent_fps) > self.args.v5_recent_memory:
                            del recent_fps[
                                : len(recent_fps) - self.args.v5_recent_memory
                            ]
                    self._update_v5_archives(
                        population=population,
                        motifs=motifs,
                        elites=elites,
                        diverse=diverse,
                        smiles=can,
                        score=score,
                    )
                    if score > best_score + 1e-6:
                        best_score = score
                        improved = True

                after_metrics = summarize_buffer(
                    self.oracle.mol_buffer,
                    max_oracle_calls=self.args.max_oracle_calls,
                    freq_log=self.args.freq_log,
                )
                self._update_v7_arm_stat(
                    arm_stats=arm_stats,
                    op_name=op_name,
                    scores=group_scores,
                    before_metrics=before_metrics,
                    after_metrics=after_metrics,
                )

            calls_after = len(self.oracle.mol_buffer)
            if improved:
                stagnant_calls = 0
            else:
                stagnant_calls += max(1, calls_after - calls_before)
            _, post_metrics = self._v5_feedback_state(
                calls=calls_after,
                stagnant_calls=stagnant_calls,
                best_score=max(best_score, 0.0),
            )
            self._append_v5_diagnostics(
                diag_path,
                oracle_name=oracle_name,
                calls=calls_after,
                state=state,
                best_score=max(best_score, 0.0),
                nonzero=self._nonzero_count(),
                avg_top10=post_metrics.get("avg_top10", 0.0),
                auc_top10=post_metrics.get("auc_top10", 0.0),
                nonzero_rate=post_metrics.get("nonzero_rate", 0.0),
                late_gap=post_metrics.get("late_gap", 0.0),
                elites=len(elites),
                diverse=len(diverse),
                motifs=len(motifs),
                population=len(population),
                stagnant_calls=stagnant_calls,
                operator_stats=self._v7_diag_operator_stats(arm_stats),
            )

    def _run_iterative_remask_v8(self, oracle_name):
        population = load_pmo_fragments(oracle_name, self.args.population_size)
        motifs = load_pmo_motifs(
            oracle_name,
            self.tk,
            max_len=self.args.max_len,
            limit=self.args.v5_motif_pool_size,
            min_atoms=self.args.motif_min_atoms,
            max_atoms=self.args.motif_max_atoms,
        )
        seed_min_size, seed_max_size = infer_fragment_pair_size_bounds(
            population=population,
            tk=self.tk,
            max_len=self.args.max_len,
            sample_n=self.args.v8_size_probe_samples,
            low_quantile=self.args.v8_size_low_quantile,
            high_quantile=self.args.v8_size_high_quantile,
            margin=self.args.v8_size_margin,
        )
        csv_path = os.path.join(
            self.args.output_dir, f"{oracle_name}_{self.args.seed}.csv"
        )
        diag_path = os.path.join(
            self.args.output_dir,
            f"diagnostics_{self.mode}_{oracle_name}_{self.args.seed}.csv",
        )
        transition_path = os.path.join(
            self.args.output_dir,
            f"transitions_{self.mode}_{oracle_name}_{self.args.seed}.csv",
        )
        elites = []
        diverse = []
        recent_fps = []
        best_score = -float("inf")
        stagnant_calls = 0
        operators = (
            "elite_tiny",
            "elite_small",
            "elite_medium",
            "diverse_medium",
            "motif_restart",
            "fragment_restart",
            "graph_swap",
            "graph_shrink",
            "graph_expand",
            "rescue_large",
        )
        context_stats = {}
        print(
            f"[iterative_remask_v8:{oracle_name}] fragments={len(population)} "
            f"motifs={len(motifs)} data_size={seed_min_size}-{seed_max_size} "
            f"remask={self.args.v5_remask_fractions} "
            "reward=parent_delta+top10_frontier graph_length=true"
        )

        while not self.oracle.finish:
            calls_before = len(self.oracle.mol_buffer)
            state, pre_metrics = self._v5_feedback_state(
                calls=calls_before,
                stagnant_calls=stagnant_calls,
                best_score=max(best_score, 0.0),
            )
            arm_stats = context_stats.setdefault(
                state,
                {op: {"ema": 0.50, "pulls": 0.0, "positive": 0.0} for op in operators},
            )
            operator_multipliers = self._v8_operator_multipliers(
                state=state,
                arm_stats=arm_stats,
                has_elites=bool(elites),
                has_diverse=bool(diverse),
                has_motifs=bool(motifs),
            )
            groups = self._make_v8_proposal_groups(
                population=population,
                elites=elites,
                diverse=diverse,
                motifs=motifs,
                min_size=seed_min_size,
                max_size=seed_max_size,
                state=state,
                operator_stats=operator_multipliers,
            )
            if not groups:
                stagnant_calls += self.args.candidate_batch_size
                continue

            frontier_improved = False
            group_items = list(groups.items())
            random.shuffle(group_items)
            for op_name, spec in group_items:
                if self.oracle.finish:
                    break

                lineage = []
                if spec.get("motif_seeded"):
                    n_mol = int(spec.get("n_mol", 0))
                    if n_mol <= 0:
                        continue
                    candidates, motif_used = sample_csdnet_with_frozen_motifs(
                        model=self.model,
                        tk=self.tk,
                        ref_lengths=self.ref_lengths,
                        motifs=motifs,
                        n_mol=n_mol,
                        device=self.device,
                        batch_size=self.args.batch_size,
                        n_steps=self.args.n_steps,
                        use_fsm_check=not self.args.disable_fsm_check,
                        use_rdkit_kekulize_check=not self.args.disable_rdkit_kekulize_check,
                        rdkit_check_interval=self.args.rdkit_check_interval,
                        max_sample_retries=self.args.max_sample_retries,
                        violation_neighborhood=self.args.violation_neighborhood,
                        temperature_start=spec["temperature_start"],
                        temperature_end=self.args.temperature_end,
                        temperature_power=self.args.temperature_power,
                    )
                    for smiles, motif in zip(candidates, motif_used):
                        lineage.append(
                            (
                                smiles,
                                {
                                    "seed": motif,
                                    "parent": None,
                                    "parent_score": None,
                                    "motif": motif,
                                },
                            )
                        )
                else:
                    proposals = spec.get("proposals", [])
                    if not proposals:
                        continue
                    indexed_candidates = sample_csdnet_local_remask(
                        model=self.model,
                        tk=self.tk,
                        seed_smiles=[proposal["seed"] for proposal in proposals],
                        max_len=self.args.max_len,
                        device=self.device,
                        batch_size=self.args.batch_size,
                        n_steps=self.args.n_steps,
                        remask_fraction=spec["remask_fraction"],
                        min_remask_tokens=self.args.min_remask_tokens,
                        span_prob=spec["span_prob"],
                        use_fsm_check=not self.args.disable_fsm_check,
                        use_rdkit_kekulize_check=not self.args.disable_rdkit_kekulize_check,
                        rdkit_check_interval=self.args.rdkit_check_interval,
                        max_sample_retries=self.args.max_sample_retries,
                        violation_neighborhood=self.args.violation_neighborhood,
                        temperature_start=spec["temperature_start"],
                        temperature_end=self.args.temperature_end,
                        temperature_power=self.args.temperature_power,
                        length_delta_choices="0",
                        length_edit_prob=0.0,
                        length_edit_min_span=1,
                        length_edit_max_span=1,
                        return_seed_indices=True,
                    )
                    for smiles, proposal_idx in indexed_candidates:
                        if 0 <= proposal_idx < len(proposals):
                            lineage.append((smiles, proposals[proposal_idx]))

                group_rewards = []
                seen_batch = set()
                for smiles, proposal in lineage:
                    if self.oracle.finish:
                        break
                    can = canonical_smiles(smiles)
                    if (
                        can is None
                        or can in seen_batch
                        or can in self.oracle.mol_buffer
                    ):
                        continue
                    seen_batch.add(can)
                    atoms = atom_count(can)
                    if atoms < self.args.v8_absolute_min_atoms:
                        continue
                    if not tokenizable(can, self.tk, self.args.max_len):
                        continue
                    fp = mol_fp_from_smiles(can)
                    near_dup = (
                        self.args.v8_near_duplicate_sim > 0
                        and max_tanimoto(fp, recent_fps)
                        >= self.args.v8_near_duplicate_sim
                    )
                    if near_dup and state in {"warmup", "balanced", "explore"}:
                        continue

                    before_scores = [
                        float(score) for score, _ in self.oracle.mol_buffer.values()
                    ]
                    before_top10 = self._v8_top_mean(before_scores, top_n=10)
                    before_threshold = self._buffer_score_threshold(top_n=10)
                    result = self._score_and_record(can, csv_path)
                    if result is None:
                        continue
                    can, score = result
                    after_scores = [
                        float(value) for value, _ in self.oracle.mol_buffer.values()
                    ]
                    after_top10 = self._v8_top_mean(after_scores, top_n=10)
                    reward, reward_parts = self._v8_transition_reward(
                        score=score,
                        parent_score=proposal.get("parent_score"),
                        before_scores=before_scores,
                        before_top10=before_top10,
                        after_top10=after_top10,
                        before_threshold=before_threshold,
                    )
                    group_rewards.append(reward)
                    if fp is not None:
                        recent_fps.append(fp)
                        if len(recent_fps) > self.args.v5_recent_memory:
                            del recent_fps[
                                : len(recent_fps) - self.args.v5_recent_memory
                            ]

                    self._update_v8_archives(
                        population=population,
                        motifs=motifs,
                        elites=elites,
                        diverse=diverse,
                        parent_smiles=proposal.get("parent"),
                        child_smiles=can,
                        child_score=score,
                        transition_reward=reward,
                        frozen_motif=proposal.get("motif"),
                    )
                    self._append_v8_transition(
                        transition_path,
                        oracle_name=oracle_name,
                        state=state,
                        operator=op_name,
                        parent_smiles=proposal.get("parent"),
                        seed_smiles=proposal.get("seed"),
                        child_smiles=can,
                        parent_score=proposal.get("parent_score"),
                        child_score=score,
                        reward=reward,
                        reward_parts=reward_parts,
                    )
                    if score > best_score + 1e-6:
                        best_score = score
                    if after_top10 > before_top10 + 1e-9:
                        frontier_improved = True

                self._update_v8_arm_stats(
                    arm_stats=arm_stats,
                    op_name=op_name,
                    rewards=group_rewards,
                )

            calls_after = len(self.oracle.mol_buffer)
            if frontier_improved:
                stagnant_calls = 0
            else:
                stagnant_calls += max(1, calls_after - calls_before)
            _, post_metrics = self._v5_feedback_state(
                calls=calls_after,
                stagnant_calls=stagnant_calls,
                best_score=max(best_score, 0.0),
            )
            self._append_v5_diagnostics(
                diag_path,
                oracle_name=oracle_name,
                calls=calls_after,
                state=state,
                best_score=max(best_score, 0.0),
                nonzero=self._nonzero_count(),
                avg_top10=post_metrics.get("avg_top10", 0.0),
                auc_top10=post_metrics.get("auc_top10", 0.0),
                nonzero_rate=post_metrics.get("nonzero_rate", 0.0),
                late_gap=post_metrics.get("late_gap", 0.0),
                elites=len(elites),
                diverse=len(diverse),
                motifs=len(motifs),
                population=len(population),
                stagnant_calls=stagnant_calls,
                operator_stats=self._v8_diag_operator_stats(context_stats, state),
            )

    def _run_iterative_remask_v9(self, oracle_name):
        min_size, max_size = task_size_bounds(oracle_name)
        prior_population, motifs, prior_source = initialize_v9_prior(
            mode=self.mode,
            oracle_name=oracle_name,
            tk=self.tk,
            max_len=self.args.max_len,
            population_size=self.args.v9_prior_population_size,
            motif_limit=self.args.v5_motif_pool_size,
            motif_min_atoms=self.args.motif_min_atoms,
            motif_max_atoms=self.args.motif_max_atoms,
        )
        adaptive_population = list(prior_population)
        online_prior = prior_source == "online_budgeted"
        csv_path = os.path.join(
            self.args.output_dir, f"{oracle_name}_{self.args.seed}.csv"
        )
        diag_path = os.path.join(
            self.args.output_dir,
            f"diagnostics_{self.mode}_{oracle_name}_{self.args.seed}.csv",
        )
        transition_path = os.path.join(
            self.args.output_dir,
            f"transitions_{self.mode}_{oracle_name}_{self.args.seed}.csv",
        )
        frontier_state_path = os.path.join(
            self.args.output_dir,
            f"frontier_state_{self.mode}_{oracle_name}_{self.args.seed}.json",
        )

        archive = V9LineageArchive(
            score_slots=self.args.v9_score_archive_size,
            lineage_slots=self.args.v9_lineage_archive_size,
            root_ucb_weight=self.args.v9_lineage_ucb_weight,
        )
        task_local_profile = resolve_local_sampler_profile()
        if task_local_profile in {
            "task_adaptive_local",
            "task_adaptive_refine",
        } and not bool(getattr(self.model, "corruption_level_conditioning", False)):
            raise RuntimeError(
                "task-adaptive PMO sampling requires the trajectory-refinement "
                "checkpoint; corruption-level conditioning is absent"
            )
        task_refinement_enabled = task_local_profile in {
            "task_adaptive_local",
            "task_adaptive_refine",
        } and bool(getattr(self.model, "corruption_level_conditioning", False))
        local_operators = (
            "elite_tiny",
            "elite_small",
            "elite_medium",
            "diverse_medium",
            "graph_shrink",
            "graph_swap",
            "graph_expand",
            "rescue_large",
        )
        if task_refinement_enabled:
            local_operators = (
                "elite_tiny",
                "elite_refine",
                *local_operators[1:],
            )

        def task_local_weights(state_name):
            weights = dict(v9_local_weights(state_name))
            if not task_refinement_enabled:
                return weights
            refine_share = 0.08 if state_name == "saturated" else 0.06
            weights = {
                name: float(weight) * (1.0 - refine_share)
                for name, weight in weights.items()
            }
            weights["elite_refine"] = refine_share
            return weights

        root_operators = ("attach_only", "motif_restart", "fragment_anchor")
        frontier_engine = None
        protected_frontier_engine = None
        if self.mode == "safe_frontier_final":
            protected_frontier_engine = BaselineProtectedFrontierEngine(
                SafePMOFrontierHead(
                    warmup_calls=self.args.v9_warmup_calls,
                )
            )
        elif self.mode == "iterative_remask_v9_gated":
            protected_frontier_engine = BaselineProtectedFrontierEngine(
                EvidenceGatedPMOHead(
                    warmup_calls=self.args.v9_warmup_calls,
                    probe_fraction=self.args.v9_gate_probe_fraction,
                    maximum_fraction=self.args.v9_gate_max_fraction,
                    window_calls=self.args.v9_gate_window_calls,
                    positive_margin=self.args.v9_gate_positive_margin,
                    negative_margin=self.args.v9_gate_negative_margin,
                    reprobe_calls=self.args.v9_gate_reprobe_calls,
                )
            )
        elif self.mode == "iterative_remask_v9_reversible":
            protected_frontier_engine = BaselineProtectedFrontierEngine(
                ReversibleEvidencePMOHead(
                    warmup_calls=self.args.v9_warmup_calls,
                    probe_fraction=self.args.v9_gate_probe_fraction,
                    maximum_fraction=self.args.v9_gate_max_fraction,
                    window_calls=self.args.v9_gate_window_calls,
                    promotion_windows=self.args.v9_gate_promotion_windows,
                    neutral_patience=self.args.v9_gate_neutral_patience,
                    confidence_z=self.args.v9_gate_confidence_z,
                    frontier_margin=self.args.v9_gate_frontier_margin,
                    entry_tolerance=self.args.v9_gate_entry_tolerance,
                    minimum_target_calls=self.args.v9_gate_min_target_calls,
                    minimum_reference_calls=(self.args.v9_gate_min_reference_calls),
                    reprobe_calls=self.args.v9_gate_reprobe_calls,
                    state_reprobe_calls=self.args.v9_gate_state_reprobe_calls,
                )
            )
        if self.mode in {
            "unified_frontier",
            "unified_frontier_v2",
            "unified_frontier_restored",
        }:
            adapter_class = {
                "unified_frontier": ScalarFrontierAdapter,
                "unified_frontier_v2": PMOFrontierAdapter,
                "unified_frontier_restored": RestoredPMOFrontierAdapter,
            }[self.mode]
            frontier_engine = UnifiedFrontierEngine(
                adapter=adapter_class(
                    warmup_calls=self.args.v9_warmup_calls,
                    saturation_threshold=self.args.v9_saturation_top10,
                    sparse_threshold=self.args.v9_sparse_nonzero_rate,
                    stagnation_patience=self.args.v9_stagnation_patience,
                    collapse_threshold=self.args.v9_lineage_collapse_threshold,
                ),
                operator_groups={
                    "root": root_operators,
                    "local": local_operators,
                },
                bandit_configs={
                    "root": {
                        "alpha": self.args.v9_root_bandit_alpha,
                        "temperature": self.args.v9_root_bandit_temperature,
                        "ucb_weight": self.args.v9_root_ucb_weight,
                        "min_multiplier": self.args.v9_min_operator_multiplier,
                        "base_floor": (
                            0.02 if self.mode == "unified_frontier_v2" else 0.30
                        ),
                    },
                    "local": {
                        "alpha": self.args.v9_bandit_alpha,
                        "temperature": self.args.v9_bandit_temperature,
                        "ucb_weight": self.args.v9_ucb_weight,
                        "min_multiplier": self.args.v9_min_operator_multiplier,
                        "base_floor": (
                            0.02 if self.mode == "unified_frontier_v2" else 0.30
                        ),
                    },
                },
                frontier_min_scale=self.args.v9_frontier_min_scale,
                delta_min_scale=self.args.v9_delta_min_scale,
            )
            root_bandit = frontier_engine.bandits["root"]
            local_bandit = frontier_engine.bandits["local"]
            if self.oracle.mol_buffer and os.path.exists(frontier_state_path):
                with open(frontier_state_path) as handle:
                    frontier_engine.load_state_dict(json.load(handle))
                print(
                    f"[{self.mode}:{oracle_name}] restored policy state "
                    f"at {len(self.oracle.mol_buffer)} oracle calls"
                )
        else:
            local_bandit = V9BatchBandit(
                local_operators,
                alpha=self.args.v9_bandit_alpha,
                temperature=self.args.v9_bandit_temperature,
                ucb_weight=self.args.v9_ucb_weight,
                min_multiplier=self.args.v9_min_operator_multiplier,
            )
            root_bandit = V9BatchBandit(
                root_operators,
                alpha=self.args.v9_root_bandit_alpha,
                temperature=self.args.v9_root_bandit_temperature,
                ucb_weight=self.args.v9_root_ucb_weight,
                min_multiplier=self.args.v9_min_operator_multiplier,
            )
        rng = random.Random(self.args.seed)
        frontier_history = []
        delta_history = []
        stagnant_calls = 0
        empty_rounds = 0
        global_rescue_attempts = 0
        self._v9_rejections = {
            "invalid": 0,
            "duplicate": 0,
            "out_of_bounds": 0,
            "untokenizable": 0,
        }
        restored_roots = self._v9_restore_archive_from_buffer(
            archive=archive,
            adaptive_population=adaptive_population,
            min_size=min_size,
            max_size=max_size,
        )
        if online_prior and self.oracle.mol_buffer:
            restored_rows = sorted(
                self.oracle.mol_buffer.items(),
                key=lambda item: float(item[1][0]),
                reverse=True,
            )[: self.args.v9_score_archive_size]
            for smiles, (score, _) in restored_rows:
                self._update_v5_motif_archive(motifs, smiles, float(score))
        if (
            protected_frontier_engine is not None
            and hasattr(protected_frontier_engine.task_head, "load_state_dict")
            and os.path.exists(frontier_state_path)
        ):
            try:
                with open(frontier_state_path) as handle:
                    saved_policy = json.load(handle)
                protected_frontier_engine.task_head.load_state_dict(
                    saved_policy.get("task_head") or {}
                )
                print(f"Restored protected PMO policy: {frontier_state_path}")
            except Exception as exc:
                print(
                    "Could not restore protected PMO policy; restarting its "
                    f"evidence gate. {exc}"
                )

        print(
            f"[iterative_remask_v9:{oracle_name}] "
            f"benchmark_size={min_size}-{max_size} "
            f"prior_fragments={len(prior_population)} motifs={len(motifs)} "
            f"prior_source={prior_source} "
            f"online_bootstrap_calls={self.args.v9_online_bootstrap_calls if online_prior else 0} "
            f"attach_warmup={self.args.v9_warmup_calls} "
            f"restored_roots={restored_roots} "
            f"policy={self.mode if (frontier_engine or protected_frontier_engine) else 'v9_split_policy'} "
            "reward=batch_frontier"
        )

        while not self.oracle.finish:
            calls_before_round = len(self.oracle.mol_buffer)
            metrics = summarize_buffer(
                self.oracle.mol_buffer,
                max_oracle_calls=self.args.max_oracle_calls,
                freq_log=self.args.freq_log,
            )
            lineage_metrics = archive.metrics(
                self.args.v9_score_archive_size + self.args.v9_lineage_archive_size
            )
            nonzero_rate = self._nonzero_count() / max(1, calls_before_round)
            online_bootstrap_active = online_prior and (
                calls_before_round < self.args.v9_online_bootstrap_calls
                or len(adaptive_population) < 2
                or len(motifs) < 2
            )
            if online_bootstrap_active:
                state = "warmup"
            elif frontier_engine is not None:
                state = frontier_engine.classify(
                    calls=calls_before_round,
                    avg_top10=float(metrics.get("avg_top10", 0.0)),
                    nonzero_rate=nonzero_rate,
                    stagnant_calls=stagnant_calls,
                    largest_root_fraction=lineage_metrics["largest_root_fraction"],
                )
            else:
                state = classify_v9_state(
                    calls=calls_before_round,
                    warmup_calls=self.args.v9_warmup_calls,
                    avg_top10=float(metrics.get("avg_top10", 0.0)),
                    nonzero_rate=nonzero_rate,
                    stagnant_calls=stagnant_calls,
                    largest_root_fraction=lineage_metrics["largest_root_fraction"],
                    saturation_threshold=self.args.v9_saturation_top10,
                    sparse_threshold=self.args.v9_sparse_nonzero_rate,
                    stagnation_patience=self.args.v9_stagnation_patience,
                    collapse_threshold=self.args.v9_lineage_collapse_threshold,
                )

            if online_bootstrap_active:
                if calls_before_round < self.args.v9_online_bootstrap_calls:
                    target = min(
                        self.args.candidate_batch_size,
                        self.args.v9_online_bootstrap_calls - calls_before_round,
                        self.args.max_oracle_calls - calls_before_round,
                    )
                else:
                    target = min(
                        self.args.candidate_batch_size,
                        self.args.max_oracle_calls - calls_before_round,
                    )
                root_counts = {"global_restart": max(1, target)}
                local_counts = {}
            elif frontier_engine is not None:
                if state == "warmup":
                    target = min(
                        self.args.candidate_batch_size,
                        self.args.v9_warmup_calls - calls_before_round,
                        self.args.max_oracle_calls - calls_before_round,
                    )
                else:
                    target = min(
                        self.args.candidate_batch_size,
                        self.args.max_oracle_calls - calls_before_round,
                    )
                allocation = frontier_engine.allocate(target, state=state)
                root_counts = allocation.get("root", {})
                local_counts = allocation.get("local", {})
            elif protected_frontier_engine is not None:
                if state == "warmup":
                    target = min(
                        self.args.candidate_batch_size,
                        self.args.v9_warmup_calls - calls_before_round,
                        self.args.max_oracle_calls - calls_before_round,
                    )
                else:
                    target = min(
                        self.args.candidate_batch_size,
                        self.args.max_oracle_calls - calls_before_round,
                    )

                policy_context = {
                    "state": state,
                    "calls": calls_before_round,
                    "avg_top10": float(metrics.get("avg_top10", 0.0)),
                    "nonzero_rate": nonzero_rate,
                    "stagnant_calls": stagnant_calls,
                    "largest_root_fraction": lineage_metrics["largest_root_fraction"],
                }
                protected_frontier_engine.classify(**policy_context)

                def allocate_v9_baseline(base_total):
                    if base_total <= 0:
                        return {}
                    if state == "warmup":
                        return {"root": {"attach_only": base_total}}
                    root_fraction, root_priors = elastic_prescreen_root_policy(
                        self.mode,
                        oracle_name,
                        state,
                    )
                    base_root_n = max(
                        1,
                        int(round(base_total * root_fraction)),
                    )
                    base_root_n = min(base_total, base_root_n)
                    return {
                        "root": allocate_weighted_counts(
                            base_root_n,
                            root_bandit.weighted(root_priors),
                        ),
                        "local": allocate_weighted_counts(
                            max(0, base_total - base_root_n),
                            local_bandit.weighted(task_local_weights(state)),
                        ),
                    }

                allocation = protected_frontier_engine.allocate(
                    target,
                    allocate_v9_baseline,
                    state=state,
                    context=policy_context,
                    available={
                        "root": root_operators,
                        "local": local_operators,
                    },
                )
                root_counts = allocation.get("root", {})
                local_counts = allocation.get("local", {})
            elif state == "warmup":
                remaining = min(
                    self.args.candidate_batch_size,
                    self.args.v9_warmup_calls - calls_before_round,
                    self.args.max_oracle_calls - calls_before_round,
                )
                root_counts = {"attach_only": max(1, remaining)}
                local_counts = {}
            else:
                target = min(
                    self.args.candidate_batch_size,
                    self.args.max_oracle_calls - calls_before_round,
                )
                root_fraction, root_priors = elastic_prescreen_root_policy(
                    self.mode,
                    oracle_name,
                    state,
                )
                root_n = max(1, int(round(target * root_fraction)))
                root_counts = allocate_weighted_counts(
                    root_n,
                    root_bandit.weighted(root_priors),
                )
                local_counts = allocate_weighted_counts(
                    max(0, target - root_n),
                    local_bandit.weighted(task_local_weights(state)),
                )

            combined_population = self._v9_combined_population(
                prior_population, adaptive_population
            )
            root_population = (
                combined_population
                if self.mode == "elastic_frontier_prescreen_v2"
                else adaptive_population
            )
            groups = self._v9_make_root_groups(
                counts=root_counts,
                population=root_population,
                fallback_population=combined_population,
                motifs=motifs,
                min_size=min_size,
                max_size=max_size,
                state=state,
            )
            if local_counts and archive.records:
                groups.update(
                    self._v9_make_local_groups(
                        counts=local_counts,
                        archive=archive,
                        population=combined_population,
                        min_size=min_size,
                        max_size=max_size,
                        state=state,
                        rng=rng,
                        frontier_engine=frontier_engine,
                    )
                )

            frontier_improved = False
            group_items = list(groups.items())
            if state != "warmup":
                rng.shuffle(group_items)
            for op_name, lineage in group_items:
                if self.oracle.finish:
                    break
                evaluated, gain = self._v9_evaluate_group(
                    oracle_name=oracle_name,
                    state=state,
                    operator=op_name,
                    lineage=lineage,
                    archive=archive,
                    adaptive_population=adaptive_population,
                    motifs=motifs,
                    min_size=min_size,
                    max_size=max_size,
                    csv_path=csv_path,
                    transition_path=transition_path,
                    root_bandit=root_bandit,
                    local_bandit=local_bandit,
                    frontier_history=frontier_history,
                    delta_history=delta_history,
                    frontier_engine=frontier_engine,
                    protected_frontier_engine=protected_frontier_engine,
                )
                frontier_improved = frontier_improved or gain > 1e-12

            calls_after_round = len(self.oracle.mol_buffer)
            calls_added = calls_after_round - calls_before_round
            if calls_added == 0:
                empty_rounds += 1
                stagnant_calls += 1
            else:
                empty_rounds = 0
                stagnant_calls = (
                    0 if frontier_improved else stagnant_calls + calls_added
                )

            if empty_rounds >= self.args.v9_empty_round_fallback:
                recovery_evaluated = 0
                if empty_rounds < self.args.v9_max_empty_rounds:
                    recovery_counts = {
                        "motif_restart": max(1, self.args.candidate_batch_size // 2),
                        "fragment_anchor": max(1, self.args.candidate_batch_size // 4),
                        "attach_only": max(1, self.args.candidate_batch_size // 4),
                    }
                    recovery_groups = self._v9_make_root_groups(
                        counts=recovery_counts,
                        population=root_population,
                        fallback_population=combined_population,
                        motifs=motifs,
                        min_size=min_size,
                        max_size=max_size,
                        state="fallback",
                    )
                    if archive.records:
                        recovery_groups.update(
                            self._v9_make_local_groups(
                                counts={
                                    "rescue_large": max(
                                        1, self.args.candidate_batch_size // 3
                                    ),
                                    "graph_swap": max(
                                        1, self.args.candidate_batch_size // 6
                                    ),
                                },
                                archive=archive,
                                population=combined_population,
                                min_size=min_size,
                                max_size=max_size,
                                state="fallback",
                                rng=rng,
                                frontier_engine=frontier_engine,
                            )
                        )
                else:
                    global_rescue_attempts += 1
                    recovery_groups = {
                        "global_restart": self._v9_make_global_restart_lineage(
                            count=self.args.candidate_batch_size,
                            min_size=min_size,
                            max_size=max_size,
                            attempt=global_rescue_attempts,
                        )
                    }

                for recovery_operator, recovery_lineage in recovery_groups.items():
                    if self.oracle.finish:
                        break
                    evaluated, _ = self._v9_evaluate_group(
                        oracle_name=oracle_name,
                        state="fallback",
                        operator=recovery_operator,
                        lineage=recovery_lineage,
                        archive=archive,
                        adaptive_population=adaptive_population,
                        motifs=motifs,
                        min_size=min_size,
                        max_size=max_size,
                        csv_path=csv_path,
                        transition_path=transition_path,
                        root_bandit=root_bandit,
                        local_bandit=local_bandit,
                        frontier_history=frontier_history,
                        delta_history=delta_history,
                        frontier_engine=frontier_engine,
                        protected_frontier_engine=protected_frontier_engine,
                    )
                    recovery_evaluated += evaluated

                if recovery_evaluated > 0:
                    print(
                        f"[V9 recovery:{oracle_name}] evaluated={recovery_evaluated} "
                        f"after_empty_rounds={empty_rounds} "
                        f"global_attempt={global_rescue_attempts}"
                    )
                    empty_rounds = 0
                    global_rescue_attempts = 0
                elif global_rescue_attempts >= self.args.v9_max_global_rescues:
                    self.oracle.save_result(self.oracle.task_label)
                    raise RuntimeError(
                        f"V9 recovery exhausted {global_rescue_attempts} global "
                        f"restarts without a new in-range candidate on {oracle_name}; "
                        f"saved {len(self.oracle.mol_buffer)} oracle calls for resume."
                    )

            post_metrics = summarize_buffer(
                self.oracle.mol_buffer,
                max_oracle_calls=self.args.max_oracle_calls,
                freq_log=self.args.freq_log,
            )
            self._append_v9_diagnostics(
                diag_path,
                oracle_name=oracle_name,
                calls=len(self.oracle.mol_buffer),
                state=state,
                avg_top10=float(post_metrics.get("avg_top10", 0.0)),
                auc_top10=float(post_metrics.get("auc_top10", 0.0)),
                nonzero_rate=self._nonzero_count()
                / max(1, len(self.oracle.mol_buffer)),
                stagnant_calls=stagnant_calls,
                lineage_metrics=archive.metrics(
                    self.args.v9_score_archive_size + self.args.v9_lineage_archive_size
                ),
                root_bandit=root_bandit,
                local_bandit=local_bandit,
                frontier_engine=frontier_engine,
                protected_frontier_engine=protected_frontier_engine,
            )
            if frontier_engine is not None:
                self._save_frontier_state(
                    frontier_state_path,
                    frontier_engine.state_dict(),
                )
            elif protected_frontier_engine is not None and hasattr(
                protected_frontier_engine.task_head,
                "state_dict",
            ):
                self._save_frontier_state(
                    frontier_state_path,
                    {"task_head": protected_frontier_engine.task_head.state_dict()},
                )

    def _run_iterative_remask_v10(self, oracle_name):
        """Hierarchical event-bandit search with structure-aware proposals."""
        min_size, max_size = task_size_bounds(oracle_name)
        prior_population = load_pmo_fragments(
            oracle_name, self.args.v9_prior_population_size
        )
        adaptive_population = list(prior_population)
        csv_path = os.path.join(
            self.args.output_dir, f"{oracle_name}_{self.args.seed}.csv"
        )
        diag_path = os.path.join(
            self.args.output_dir,
            f"diagnostics_{self.mode}_{oracle_name}_{self.args.seed}.csv",
        )
        transition_path = os.path.join(
            self.args.output_dir,
            f"transitions_{self.mode}_{oracle_name}_{self.args.seed}.csv",
        )
        state_path = os.path.join(
            self.args.output_dir,
            f"frontier_state_{self.mode}_{oracle_name}_{self.args.seed}.json",
        )

        archive = V9LineageArchive(
            score_slots=self.args.v9_score_archive_size,
            lineage_slots=self.args.v9_lineage_archive_size,
            root_ucb_weight=self.args.v9_lineage_ucb_weight,
        )
        group_bandit = BetaEventBandit(
            ("explore", "refine"),
            ucb_weight=self.args.v10_group_ucb_weight,
        )
        explore_bandit = BetaEventBandit(
            ("attach_only", "anchored_completion"),
            ucb_weight=self.args.v10_operator_ucb_weight,
        )
        refine_bandit = BetaEventBandit(
            ("peripheral_replace", "micro_remask", "graph_shrink"),
            ucb_weight=self.args.v10_operator_ucb_weight,
        )
        screen = KnnUCBScreen(
            k=self.args.v10_knn_k,
            history_limit=self.args.v10_surrogate_history,
            beta_start=self.args.v10_ucb_beta_start,
            beta_end=self.args.v10_ucb_beta_end,
            exploration_floor=self.args.v10_exploration_floor,
            min_history=self.args.v10_surrogate_start_calls,
        )
        rng = random.Random(self.args.seed)
        stagnant_calls = 0
        empty_rounds = 0
        self._v9_rejections = {
            "invalid": 0,
            "duplicate": 0,
            "out_of_bounds": 0,
            "untokenizable": 0,
            "core_lost": 0,
        }
        restored_roots = self._v9_restore_archive_from_buffer(
            archive=archive,
            adaptive_population=adaptive_population,
            min_size=min_size,
            max_size=max_size,
        )
        if self.oracle.mol_buffer and os.path.exists(state_path):
            try:
                with open(state_path) as handle:
                    saved_state = json.load(handle)
                group_bandit.load_state_dict(saved_state.get("group_bandit"))
                explore_bandit.load_state_dict(saved_state.get("explore_bandit"))
                refine_bandit.load_state_dict(saved_state.get("refine_bandit"))
                stagnant_calls = int(saved_state.get("stagnant_calls", 0))
                print(
                    f"[V10:{oracle_name}] restored event policy at "
                    f"{len(self.oracle.mol_buffer)} calls"
                )
            except Exception as exc:
                print(f"[V10:{oracle_name}] policy-state restore skipped: {exc}")

        print(
            f"[iterative_remask_v10:{oracle_name}] "
            f"benchmark_size={min_size}-{max_size} "
            f"prior_fragments={len(prior_population)} "
            f"warmup={self.args.v10_warmup_calls} "
            f"overgenerate={self.args.v10_overgenerate_factor:.2f} "
            f"restored_roots={restored_roots} "
            "policy=hierarchical_event_bandit screen=knn_ucb"
        )

        while not self.oracle.finish:
            calls_before = len(self.oracle.mol_buffer)
            remaining = self.args.max_oracle_calls - calls_before
            target = min(self.args.candidate_batch_size, remaining)
            metrics = summarize_buffer(
                self.oracle.mol_buffer,
                max_oracle_calls=self.args.max_oracle_calls,
                freq_log=self.args.freq_log,
            )
            lineage_metrics = archive.metrics(
                self.args.v9_score_archive_size + self.args.v9_lineage_archive_size
            )
            nonzero_rate = self._nonzero_count() / max(1, calls_before)
            state = classify_v9_state(
                calls=calls_before,
                warmup_calls=self.args.v10_warmup_calls,
                avg_top10=float(metrics.get("avg_top10", 0.0)),
                nonzero_rate=nonzero_rate,
                stagnant_calls=stagnant_calls,
                largest_root_fraction=lineage_metrics["largest_root_fraction"],
                saturation_threshold=self.args.v9_saturation_top10,
                sparse_threshold=self.args.v9_sparse_nonzero_rate,
                stagnation_patience=self.args.v9_stagnation_patience,
                collapse_threshold=self.args.v9_lineage_collapse_threshold,
            )

            if state == "warmup":
                proposal_budget = max(
                    target,
                    int(
                        math.ceil(target * min(1.5, self.args.v10_overgenerate_factor))
                    ),
                )
                group_counts = {"explore": proposal_budget}
                explore_counts = {"attach_only": proposal_budget}
                refine_counts = {}
            else:
                proposal_budget = max(
                    target,
                    int(math.ceil(target * self.args.v10_overgenerate_factor)),
                )
                group_priors, explore_priors, refine_priors = self._v10_priors(state)
                group_counts = allocate_weighted_counts(
                    proposal_budget, group_bandit.weighted(group_priors)
                )
                explore_counts = allocate_weighted_counts(
                    group_counts.get("explore", 0),
                    explore_bandit.weighted(explore_priors),
                )
                refine_counts = allocate_weighted_counts(
                    group_counts.get("refine", 0),
                    refine_bandit.weighted(refine_priors),
                )

            combined_population = self._v9_combined_population(
                prior_population, adaptive_population
            )
            groups = self._v10_make_explore_groups(
                explore_counts,
                adaptive_population or combined_population,
                min_size,
                max_size,
                rng,
            )
            if refine_counts and archive.records:
                groups.update(
                    self._v10_make_refine_groups(
                        refine_counts,
                        archive,
                        combined_population,
                        min_size,
                        max_size,
                        rng,
                    )
                )

            generated_counts = {name: len(rows) for name, rows in groups.items()}
            proposals = self._v10_prepare_candidates(
                groups, min_size=min_size, max_size=max_size
            )
            history = (
                self.oracle.mol_buffer
                if calls_before >= self.args.v10_surrogate_start_calls
                else {}
            )
            selected = screen.select(
                proposals,
                history=history,
                n_select=target,
                calls=calls_before,
                max_calls=self.args.max_oracle_calls,
                rng=rng,
            )
            selected_counts = {}
            for proposal in selected:
                name = proposal["operator"]
                selected_counts[name] = selected_counts.get(name, 0) + 1

            evaluated, frontier_gain = self._v10_evaluate_selected(
                oracle_name=oracle_name,
                state=state,
                selected=selected,
                archive=archive,
                adaptive_population=adaptive_population,
                min_size=min_size,
                max_size=max_size,
                csv_path=csv_path,
                transition_path=transition_path,
                group_bandit=group_bandit,
                explore_bandit=explore_bandit,
                refine_bandit=refine_bandit,
            )

            if evaluated == 0 and not self.oracle.finish:
                empty_rounds += 1
                fallback = self._v9_make_global_restart_lineage(
                    count=target,
                    min_size=min_size,
                    max_size=max_size,
                    attempt=empty_rounds,
                )
                fallback_rows = [
                    {
                        "smiles": smiles,
                        "group": "explore",
                        "operator": "global_restart",
                        **proposal,
                    }
                    for smiles, proposal in fallback
                ]
                evaluated, frontier_gain = self._v10_evaluate_selected(
                    oracle_name=oracle_name,
                    state="fallback",
                    selected=fallback_rows,
                    archive=archive,
                    adaptive_population=adaptive_population,
                    min_size=min_size,
                    max_size=max_size,
                    csv_path=csv_path,
                    transition_path=transition_path,
                    group_bandit=group_bandit,
                    explore_bandit=explore_bandit,
                    refine_bandit=refine_bandit,
                )
                if evaluated == 0 and empty_rounds >= self.args.v10_max_empty_rounds:
                    self.oracle.save_result(self.oracle.task_label)
                    raise RuntimeError(
                        f"V10 produced no new in-range candidates for "
                        f"{empty_rounds} consecutive rounds on {oracle_name}."
                    )
            else:
                empty_rounds = 0

            calls_added = len(self.oracle.mol_buffer) - calls_before
            if frontier_gain > 1e-12:
                stagnant_calls = 0
            else:
                stagnant_calls += max(1, calls_added)

            post_metrics = summarize_buffer(
                self.oracle.mol_buffer,
                max_oracle_calls=self.args.max_oracle_calls,
                freq_log=self.args.freq_log,
            )
            self._append_v10_diagnostics(
                diag_path,
                oracle_name=oracle_name,
                calls=len(self.oracle.mol_buffer),
                state=state,
                avg_top10=float(post_metrics.get("avg_top10", 0.0)),
                auc_top10=float(post_metrics.get("auc_top10", 0.0)),
                stagnant_calls=stagnant_calls,
                lineage_metrics=archive.metrics(
                    self.args.v9_score_archive_size + self.args.v9_lineage_archive_size
                ),
                generated_counts=generated_counts,
                selected_counts=selected_counts,
                group_bandit=group_bandit,
                explore_bandit=explore_bandit,
                refine_bandit=refine_bandit,
            )
            self._save_frontier_state(
                state_path,
                {
                    "calls": len(self.oracle.mol_buffer),
                    "stagnant_calls": stagnant_calls,
                    "group_bandit": group_bandit.state_dict(),
                    "explore_bandit": explore_bandit.state_dict(),
                    "refine_bandit": refine_bandit.state_dict(),
                },
            )

    @staticmethod
    def _v10_priors(state):
        if state == "saturated":
            group = {"explore": 0.12, "refine": 0.88}
            explore = {"attach_only": 0.35, "anchored_completion": 0.65}
            refine = {
                "peripheral_replace": 0.60,
                "micro_remask": 0.38,
                "graph_shrink": 0.02,
            }
        elif state in {"plateau", "collapsed", "sparse"}:
            group = {"explore": 0.62, "refine": 0.38}
            explore = {"attach_only": 0.45, "anchored_completion": 0.55}
            refine = {
                "peripheral_replace": 0.48,
                "micro_remask": 0.47,
                "graph_shrink": 0.05,
            }
        else:
            group = {"explore": 0.42, "refine": 0.58}
            explore = {"attach_only": 0.65, "anchored_completion": 0.35}
            refine = {
                "peripheral_replace": 0.55,
                "micro_remask": 0.42,
                "graph_shrink": 0.03,
            }
        return group, explore, refine

    def _v10_length_deltas(self):
        values = []
        for raw in str(self.args.v10_length_deltas).split(","):
            try:
                values.append(int(raw.strip()))
            except ValueError:
                continue
        return values or [0]

    def _v10_make_explore_groups(self, counts, population, min_size, max_size, rng):
        groups = {}
        attach_count = int(counts.get("attach_only", 0))
        if attach_count > 0:
            groups["attach_only"] = self._v9_make_attachment_lineage(
                attach_count,
                population,
                min_size,
                max_size,
                root_operator="attach_only",
            )

        anchor_count = int(counts.get("anchored_completion", 0))
        if anchor_count <= 0:
            return groups
        base = self._v9_make_attachment_lineage(
            max(anchor_count, int(math.ceil(anchor_count * 1.25))),
            population,
            min_size,
            max_size,
            root_operator="anchored_completion",
        )
        seeds = []
        plans = []
        metadata = []
        deltas = self._v10_length_deltas()
        for seed, _ in base:
            plan = adaptive_peripheral_edit_plan(
                seed,
                rng,
                delta=rng.choice(deltas),
                target_atom_fraction=self.args.v10_peripheral_fraction,
                max_atom_fraction=0.50,
                max_span_tokens=14,
            )
            if plan is None:
                plan = atom_span_edit_plan(
                    seed, rng, delta=rng.choice(deltas), span_tokens=5
                )
            if plan is None:
                continue
            seeds.append(seed)
            plans.append(plan)
            metadata.append(
                {
                    "seed": seed,
                    "parent_record": None,
                    "root_operator": "anchored_completion",
                    "preserve_from": seed,
                }
            )
        indexed = sample_csdnet_local_remask(
            model=self.model,
            tk=self.tk,
            seed_smiles=seeds,
            max_len=self.args.max_len,
            device=self.device,
            batch_size=self.args.batch_size,
            n_steps=self.args.n_steps,
            min_remask_tokens=self.args.min_remask_tokens,
            use_fsm_check=not self.args.disable_fsm_check,
            use_rdkit_kekulize_check=not self.args.disable_rdkit_kekulize_check,
            rdkit_check_interval=self.args.rdkit_check_interval,
            max_sample_retries=self.args.max_sample_retries,
            violation_neighborhood=self.args.violation_neighborhood,
            temperature_start=max(1.02, self.args.temperature_start),
            temperature_end=self.args.temperature_end,
            temperature_power=self.args.temperature_power,
            edit_plans=plans,
            return_seed_indices=True,
        )
        lineage = []
        for candidate, seed_idx in indexed:
            if 0 <= seed_idx < len(metadata):
                lineage.append((candidate, metadata[seed_idx]))
                if len(lineage) >= anchor_count:
                    break
        if lineage:
            groups["anchored_completion"] = lineage
        return groups

    def _v10_make_refine_groups(
        self, counts, archive, population, min_size, max_size, rng
    ):
        groups = {}
        deltas = self._v10_length_deltas()
        for operator, count in counts.items():
            count = int(count)
            if count <= 0:
                continue
            seeds = []
            plans = []
            metadata = []
            attempts = 0
            while len(seeds) < count and attempts < max(100, count * 40):
                attempts += 1
                parent = archive.choose_parent(
                    rng,
                    lineage_probability=0.45 if operator == "micro_remask" else 0.30,
                )
                if parent is None:
                    break
                seed = parent.smiles
                preserve_from = parent.smiles
                if operator == "graph_shrink":
                    seed = self._make_v8_graph_fragment_edit(
                        parent_smiles=parent.smiles,
                        population=population,
                        direction="shrink",
                        min_size=min_size,
                        max_size=max_size,
                    )
                    preserve_from = ""
                can = canonical_smiles(seed) if seed else None
                if can is None or not min_size <= atom_count(can) <= max_size:
                    continue

                if operator == "peripheral_replace":
                    delta = rng.choice(deltas)
                    plan = adaptive_peripheral_edit_plan(
                        can,
                        rng,
                        delta=delta,
                        target_atom_fraction=self.args.v10_peripheral_fraction,
                        max_atom_fraction=0.45,
                        max_span_tokens=14,
                    )
                elif operator == "micro_remask":
                    delta = rng.choice((-1, 0, 0, 0, 1))
                    plan = adaptive_peripheral_edit_plan(
                        can,
                        rng,
                        delta=delta,
                        target_atom_fraction=self.args.v10_micro_fraction,
                        max_atom_fraction=0.25,
                        max_span_tokens=7,
                    )
                    if plan is None:
                        plan = atom_span_edit_plan(can, rng, delta=delta, span_tokens=3)
                else:
                    plan = adaptive_peripheral_edit_plan(
                        can,
                        rng,
                        delta=0,
                        target_atom_fraction=self.args.v10_micro_fraction,
                        max_atom_fraction=0.30,
                        max_span_tokens=8,
                    )
                if plan is None:
                    continue
                seeds.append(can)
                plans.append(plan)
                metadata.append(
                    {
                        "seed": can,
                        "parent_record": parent,
                        "root_operator": parent.root_operator,
                        "preserve_from": preserve_from,
                    }
                )

            if not seeds:
                continue
            temperature = (
                max(0.90, self.args.temperature_start - 0.16)
                if operator == "micro_remask"
                else self.args.temperature_start
            )
            indexed = sample_csdnet_local_remask(
                model=self.model,
                tk=self.tk,
                seed_smiles=seeds,
                max_len=self.args.max_len,
                device=self.device,
                batch_size=self.args.batch_size,
                n_steps=self.args.n_steps,
                min_remask_tokens=self.args.min_remask_tokens,
                use_fsm_check=not self.args.disable_fsm_check,
                use_rdkit_kekulize_check=not self.args.disable_rdkit_kekulize_check,
                rdkit_check_interval=self.args.rdkit_check_interval,
                max_sample_retries=self.args.max_sample_retries,
                violation_neighborhood=self.args.violation_neighborhood,
                temperature_start=temperature,
                temperature_end=self.args.temperature_end,
                temperature_power=self.args.temperature_power,
                edit_plans=plans,
                return_seed_indices=True,
            )
            lineage = []
            for candidate, seed_idx in indexed:
                if 0 <= seed_idx < len(metadata):
                    lineage.append((candidate, metadata[seed_idx]))
            if lineage:
                groups[operator] = lineage
        return groups

    def _v10_prepare_candidates(self, groups, min_size, max_size):
        proposals = []
        seen = set()
        for operator, lineage in groups.items():
            group = (
                "explore"
                if operator in {"attach_only", "anchored_completion", "global_restart"}
                else "refine"
            )
            for smiles, metadata in lineage:
                can = canonical_smiles(smiles)
                if can is None:
                    self._v9_rejections["invalid"] += 1
                    continue
                if can in seen or can in self.oracle.mol_buffer:
                    self._v9_rejections["duplicate"] += 1
                    continue
                atoms = atom_count(can)
                if not min_size <= atoms <= max_size:
                    self._v9_rejections["out_of_bounds"] += 1
                    continue
                if not tokenizable(can, self.tk, self.args.max_len):
                    self._v9_rejections["untokenizable"] += 1
                    continue
                preserve_from = metadata.get("preserve_from") or ""
                if preserve_from and not preserves_murcko_scaffold(preserve_from, can):
                    self._v9_rejections["core_lost"] += 1
                    continue
                seen.add(can)
                proposals.append(
                    {
                        "smiles": can,
                        "group": group,
                        "operator": operator,
                        **metadata,
                    }
                )
        return proposals

    def _v10_evaluate_selected(
        self,
        oracle_name,
        state,
        selected,
        archive,
        adaptive_population,
        min_size,
        max_size,
        csv_path,
        transition_path,
        group_bandit,
        explore_bandit,
        refine_bandit,
    ):
        before_scores = [float(score) for score, _ in self.oracle.mol_buffer.values()]
        before_top10 = self._v8_top_mean(before_scores, top_n=10)
        rows = []
        seen = set()
        for proposal in selected:
            if self.oracle.finish:
                break
            can = canonical_smiles(proposal["smiles"])
            if can is None or can in seen or can in self.oracle.mol_buffer:
                continue
            seen.add(can)
            result = self._score_and_record(can, csv_path)
            if result is None:
                continue
            child_smiles, child_score = result
            parent = proposal.get("parent_record")
            rows.append(
                {
                    **proposal,
                    "child_smiles": child_smiles,
                    "child_score": float(child_score),
                    "parent_score": None if parent is None else float(parent.score),
                    "call_index": int(self.oracle.mol_buffer[child_smiles][1]),
                }
            )

        after_scores = [float(score) for score, _ in self.oracle.mol_buffer.values()]
        after_top10 = self._v8_top_mean(after_scores, top_n=10)
        threshold = (
            sorted(after_scores, reverse=True)[min(9, len(after_scores) - 1)]
            if after_scores
            else math.inf
        )
        for row in rows:
            row["entered_top10"] = row["child_score"] >= threshold - 1e-12
            row["parent_improved"] = (
                row["parent_score"] is not None
                and row["child_score"] > row["parent_score"] + 1e-12
            )

        for group_name in ("explore", "refine"):
            subset = [row for row in rows if row["group"] == group_name]
            group_bandit.update(
                group_name,
                sum(row["entered_top10"] for row in subset),
                len(subset),
            )
        for operator in ("attach_only", "anchored_completion", "global_restart"):
            subset = [row for row in rows if row["operator"] == operator]
            if operator in explore_bandit.stats:
                explore_bandit.update(
                    operator,
                    sum(row["entered_top10"] for row in subset),
                    len(subset),
                )
        for operator in ("peripheral_replace", "micro_remask", "graph_shrink"):
            subset = [row for row in rows if row["operator"] == operator]
            refine_bandit.update(
                operator,
                sum(row["parent_improved"] for row in subset),
                len(subset),
            )

        transition_rows = []
        for row in rows:
            parent = row.get("parent_record")
            if parent is None:
                root_id = hashlib.sha1(row["child_smiles"].encode("utf-8")).hexdigest()[
                    :16
                ]
                record = archive.add_root(
                    smiles=row["child_smiles"],
                    score=row["child_score"],
                    root_id=root_id,
                    root_operator=row["operator"],
                    created_call=row["call_index"],
                )
                if row["entered_top10"]:
                    archive.roots[root_id].top10_entries += 1
            else:
                record = archive.add_child(
                    smiles=row["child_smiles"],
                    score=row["child_score"],
                    parent=parent,
                    operator=row["operator"],
                    created_call=row["call_index"],
                    frontier_gain=(
                        max(0.0, after_top10 - before_top10)
                        if row["entered_top10"]
                        else 0.0
                    ),
                    entered_top10=row["entered_top10"],
                )
            if before_scores:
                percentile = sum(
                    score <= row["child_score"] for score in before_scores
                ) / len(before_scores)
            else:
                percentile = 1.0
            if row["entered_top10"] or percentile >= self.args.v9_archive_percentile:
                self._update_fragment_population_v2(
                    adaptive_population, row["child_smiles"], row["child_score"]
                )
            transition_rows.append(
                {
                    "oracle": oracle_name,
                    "call": row["call_index"],
                    "state": state,
                    "group": row["group"],
                    "operator": row["operator"],
                    "root_operator": record.root_operator,
                    "root_id": record.root_id,
                    "depth": record.depth,
                    "parent_smiles": "" if parent is None else parent.smiles,
                    "child_smiles": row["child_smiles"],
                    "parent_score": ""
                    if row["parent_score"] is None
                    else row["parent_score"],
                    "child_score": row["child_score"],
                    "delta": ""
                    if row["parent_score"] is None
                    else row["child_score"] - row["parent_score"],
                    "entered_top10": int(row["entered_top10"]),
                    "parent_improved": int(row["parent_improved"]),
                    "screen_mean": row.get("screen_mean", ""),
                    "screen_uncertainty": row.get("screen_uncertainty", ""),
                    "screen_acquisition": row.get("screen_acquisition", ""),
                    "child_atoms": atom_count(row["child_smiles"]),
                    "min_atoms": min_size,
                    "max_atoms": max_size,
                }
            )
        self._append_v9_transition_rows(transition_path, transition_rows)
        return len(rows), max(0.0, after_top10 - before_top10)

    def _append_v10_diagnostics(
        self,
        path,
        oracle_name,
        calls,
        state,
        avg_top10,
        auc_top10,
        stagnant_calls,
        lineage_metrics,
        generated_counts,
        selected_counts,
        group_bandit,
        explore_bandit,
        refine_bandit,
    ):
        row = {
            "oracle": oracle_name,
            "calls": calls,
            "state": state,
            "avg_top10": avg_top10,
            "auc_top10": auc_top10,
            "stagnant_calls": stagnant_calls,
            "root_count": lineage_metrics["root_count"],
            "largest_root_fraction": lineage_metrics["largest_root_fraction"],
            "lineage_entropy": lineage_metrics["lineage_entropy"],
            "generated_by_operator": json.dumps(generated_counts, sort_keys=True),
            "selected_by_operator": json.dumps(selected_counts, sort_keys=True),
            "group_bandit": json.dumps(group_bandit.snapshot(), sort_keys=True),
            "explore_bandit": json.dumps(explore_bandit.snapshot(), sort_keys=True),
            "refine_bandit": json.dumps(refine_bandit.snapshot(), sort_keys=True),
            "rejected_invalid": self._v9_rejections["invalid"],
            "rejected_duplicate": self._v9_rejections["duplicate"],
            "rejected_out_of_bounds": self._v9_rejections["out_of_bounds"],
            "rejected_untokenizable": self._v9_rejections["untokenizable"],
            "rejected_core_lost": self._v9_rejections["core_lost"],
        }
        os.makedirs(os.path.dirname(path), exist_ok=True)
        exists = os.path.exists(path)
        with open(path, "a", newline="") as handle:
            writer = csv.DictWriter(handle, fieldnames=list(row))
            if not exists:
                writer.writeheader()
            writer.writerow(row)

    def _v9_combined_population(self, prior_population, adaptive_population):
        by_fragment = {}
        for score, fragment in prior_population:
            by_fragment[fragment] = max(
                float(score), by_fragment.get(fragment, -math.inf)
            )
        for score, fragment in adaptive_population:
            by_fragment[fragment] = max(
                float(score), by_fragment.get(fragment, -math.inf)
            )
        rows = sorted(
            ((score, fragment) for fragment, score in by_fragment.items()),
            reverse=True,
        )
        return rows[: max(self.args.v9_prior_population_size * 2, 100)]

    def _v9_restore_archive_from_buffer(
        self,
        archive,
        adaptive_population,
        min_size,
        max_size,
    ):
        if not self.oracle.mol_buffer:
            return 0
        limit = self.args.v9_score_archive_size + self.args.v9_lineage_archive_size
        rows = sorted(
            self.oracle.mol_buffer.items(),
            key=lambda item: float(item[1][0]),
            reverse=True,
        )
        restored = 0
        for smiles, (score, call_index) in rows:
            if restored >= limit:
                break
            if not min_size <= atom_count(smiles) <= max_size:
                continue
            if not tokenizable(smiles, self.tk, self.args.max_len):
                continue
            root_id = hashlib.sha1(f"resume:{smiles}".encode("utf-8")).hexdigest()[:16]
            archive.add_root(
                smiles=smiles,
                score=float(score),
                root_id=root_id,
                root_operator="resume_buffer",
                created_call=int(call_index),
            )
            if restored < min(32, limit):
                self._update_fragment_population_v2(
                    adaptive_population,
                    smiles,
                    float(score),
                )
            restored += 1
        return restored

    def _v9_task_reference_lengths(self, min_size, max_size):
        """Restrict the atomic sequence prior to a task-compatible support."""
        token_min = max(3, min(self.args.max_len, int(min_size) + 2))
        token_max = max(
            token_min,
            min(self.args.max_len, int(max_size) * 2 + 2),
        )
        reference_lengths = getattr(self, "ref_lengths", ())
        lengths = [
            int(length)
            for length in reference_lengths
            if token_min <= int(length) <= token_max
        ]
        return lengths or list(range(token_min, token_max + 1))

    def _v9_make_global_restart_lineage(
        self,
        count,
        min_size,
        max_size,
        attempt,
    ):
        target = max(1, int(count))
        overgenerate = min(12, 3 + max(1, int(attempt)))
        restart_lengths = self._v9_task_reference_lengths(min_size, max_size)
        global_sampler_kwargs = progressive_global_sampler_kwargs()
        if not global_sampler_kwargs:
            global_sampler_kwargs = {
                "temperature_start": max(
                    self.args.v5_rescue_temperature,
                    self.args.temperature_start,
                ),
                "temperature_end": self.args.temperature_end,
                "temperature_power": self.args.temperature_power,
            }
        candidates = sample_csdnet(
            model=self.model,
            tk=self.tk,
            ref_lengths=restart_lengths,
            n_mol=target * overgenerate,
            device=self.device,
            batch_size=self.args.batch_size,
            n_steps=self.args.n_steps,
            use_fsm_check=not self.args.disable_fsm_check,
            use_rdkit_kekulize_check=not self.args.disable_rdkit_kekulize_check,
            rdkit_check_interval=self.args.rdkit_check_interval,
            max_sample_retries=self.args.max_sample_retries,
            violation_neighborhood=self.args.violation_neighborhood,
            **global_sampler_kwargs,
        )
        lineage = []
        seen = set()
        for candidate in candidates:
            can = canonical_smiles(candidate)
            if (
                can is None
                or can in seen
                or can in self.oracle.mol_buffer
                or not min_size <= atom_count(can) <= max_size
                or not tokenizable(can, self.tk, self.args.max_len)
            ):
                continue
            seen.add(can)
            lineage.append(
                (
                    can,
                    {
                        "seed": can,
                        "parent_record": None,
                        "root_operator": "global_restart",
                    },
                )
            )
            if len(lineage) >= target:
                break
        print(
            f"[V9 global restart] attempt={attempt} generated={len(candidates)} "
            f"accepted={len(lineage)} atom_range={min_size}-{max_size} "
            f"token_support={min(restart_lengths)}-{max(restart_lengths)} "
            f"length_prior_n={len(restart_lengths)}"
        )
        return lineage

    def _v9_make_attachment_lineage(
        self,
        count,
        population,
        min_size,
        max_size,
        root_operator,
        rank_state=None,
    ):
        lineage = []
        seen = set()
        attempts = 0
        target = max(0, int(count))
        while len(lineage) < target and attempts < max(200, target * 80):
            attempts += 1
            seed = self._make_fragment_seed_v2(
                population,
                min_size,
                max_size,
                prefer_top=False,
                rank_state=rank_state,
            )
            can = canonical_smiles(seed) if seed else None
            if can is None or can in seen or can in self.oracle.mol_buffer:
                continue
            seen.add(can)
            lineage.append(
                (
                    can,
                    {
                        "seed": can,
                        "parent_record": None,
                        "root_operator": root_operator,
                    },
                )
            )
        return lineage

    def _v9_make_root_groups(
        self,
        counts,
        population,
        fallback_population,
        motifs,
        min_size,
        max_size,
        state,
    ):
        groups = {}
        if self.mode == "elastic_frontier_prescreen_v2":
            motifs = select_rank_stratified_motifs(
                motifs,
                state,
                limit=self.args.v9_prescreen_active_motif_size,
            )
        root_local_profile = resolve_local_sampler_profile()
        if root_local_profile == "task_adaptive_refine":
            root_local_profile = "task_adaptive_local"
        for operator, count in counts.items():
            if count <= 0:
                continue
            if operator == "attach_only":
                active_population = population or fallback_population
                if active_population:
                    lineage = self._v9_make_attachment_lineage(
                        count,
                        active_population,
                        min_size,
                        max_size,
                        root_operator=operator,
                        rank_state=(
                            state
                            if self.mode == "elastic_frontier_prescreen_v2"
                            else None
                        ),
                    )
                else:
                    lineage = self._v9_make_global_restart_lineage(
                        count, min_size, max_size, attempt=0
                    )
            elif operator == "motif_restart":
                if not motifs:
                    groups[operator] = self._v9_make_global_restart_lineage(
                        count, min_size, max_size, attempt=0
                    )
                    continue
                request_n = max(
                    count,
                    int(math.ceil(count * self.args.v9_root_overgenerate_factor)),
                )
                candidates, used = sample_csdnet_with_frozen_motifs(
                    model=self.model,
                    tk=self.tk,
                    ref_lengths=self._v9_task_reference_lengths(
                        min_size,
                        max_size,
                    ),
                    motifs=motifs,
                    n_mol=request_n,
                    device=self.device,
                    batch_size=self.args.batch_size,
                    n_steps=self.args.n_steps,
                    use_fsm_check=not self.args.disable_fsm_check,
                    use_rdkit_kekulize_check=not self.args.disable_rdkit_kekulize_check,
                    rdkit_check_interval=self.args.rdkit_check_interval,
                    max_sample_retries=self.args.max_sample_retries,
                    violation_neighborhood=self.args.violation_neighborhood,
                    temperature_start=max(self.args.temperature_start, 1.25),
                    temperature_end=self.args.temperature_end,
                    temperature_power=self.args.temperature_power,
                    sampler_profile=root_local_profile,
                )
                lineage = []
                seen = set()
                for candidate, motif in zip(candidates, used):
                    can = canonical_smiles(candidate)
                    if (
                        can is None
                        or can in seen
                        or atom_count(can) < min_size
                        or atom_count(can) > max_size
                    ):
                        continue
                    seen.add(can)
                    lineage.append(
                        (
                            can,
                            {
                                "seed": motif,
                                "parent_record": None,
                                "root_operator": operator,
                            },
                        )
                    )
                    if len(lineage) >= count:
                        break
            elif operator == "fragment_anchor":
                active_population = population or fallback_population
                if not active_population:
                    groups[operator] = self._v9_make_global_restart_lineage(
                        count, min_size, max_size, attempt=0
                    )
                    continue
                seeds = [
                    row[0]
                    for row in self._v9_make_attachment_lineage(
                        max(count, int(math.ceil(count * 1.35))),
                        active_population,
                        min_size,
                        max_size,
                        root_operator=operator,
                        rank_state=(
                            state
                            if self.mode == "elastic_frontier_prescreen_v2"
                            else None
                        ),
                    )
                ]
                indexed = sample_csdnet_local_remask(
                    model=self.model,
                    tk=self.tk,
                    seed_smiles=seeds,
                    max_len=self.args.max_len,
                    device=self.device,
                    batch_size=self.args.batch_size,
                    n_steps=self.args.n_steps,
                    remask_fraction=self.args.v9_anchor_remask_fraction,
                    min_remask_tokens=self.args.min_remask_tokens,
                    span_prob=max(0.50, self.args.span_prob - 0.12),
                    use_fsm_check=not self.args.disable_fsm_check,
                    use_rdkit_kekulize_check=not self.args.disable_rdkit_kekulize_check,
                    rdkit_check_interval=self.args.rdkit_check_interval,
                    max_sample_retries=self.args.max_sample_retries,
                    violation_neighborhood=self.args.violation_neighborhood,
                    temperature_start=max(0.92, self.args.temperature_start - 0.10),
                    temperature_end=self.args.temperature_end,
                    temperature_power=self.args.temperature_power,
                    length_delta_choices="0",
                    length_edit_prob=0.0,
                    local_sampler_profile=root_local_profile,
                    return_seed_indices=True,
                )
                lineage = []
                for candidate, seed_idx in indexed:
                    if not 0 <= seed_idx < len(seeds):
                        continue
                    can = canonical_smiles(candidate)
                    if can is None or not min_size <= atom_count(can) <= max_size:
                        continue
                    lineage.append(
                        (
                            can,
                            {
                                "seed": seeds[seed_idx],
                                "parent_record": None,
                                "root_operator": operator,
                            },
                        )
                    )
                    if len(lineage) >= count:
                        break
            else:
                lineage = []
            if lineage:
                groups[operator] = lineage
        return groups

    def _v9_evaluate_group(
        self,
        oracle_name,
        state,
        operator,
        lineage,
        archive,
        adaptive_population,
        motifs,
        min_size,
        max_size,
        csv_path,
        transition_path,
        root_bandit,
        local_bandit,
        frontier_history,
        delta_history,
        frontier_engine=None,
        protected_frontier_engine=None,
    ):
        root_operators = {
            "attach_only",
            "motif_restart",
            "fragment_anchor",
            "global_restart",
        }
        before_scores = [float(score) for score, _ in self.oracle.mol_buffer.values()]
        before_top10 = self._v8_top_mean(before_scores, top_n=10)
        seen_batch = set()
        evaluated_rows = []

        for smiles, proposal in lineage:
            if self.oracle.finish:
                break
            can = canonical_smiles(smiles)
            if can is None:
                self._v9_rejections["invalid"] += 1
                continue
            if can in seen_batch or can in self.oracle.mol_buffer:
                self._v9_rejections["duplicate"] += 1
                continue
            seen_batch.add(can)
            atoms = atom_count(can)
            if not min_size <= atoms <= max_size:
                self._v9_rejections["out_of_bounds"] += 1
                continue
            if not tokenizable(can, self.tk, self.args.max_len):
                self._v9_rejections["untokenizable"] += 1
                continue

            result = self._score_and_record(can, csv_path)
            if result is None:
                continue
            child_smiles, child_score = result
            parent = proposal.get("parent_record")
            parent_score = None if parent is None else float(parent.score)
            call_index = int(self.oracle.mol_buffer[child_smiles][1])
            evaluated_rows.append(
                {
                    "child_smiles": child_smiles,
                    "child_score": float(child_score),
                    "child_atoms": atoms,
                    "seed_smiles": proposal.get("seed") or "",
                    "parent_record": parent,
                    "parent_score": parent_score,
                    "root_operator": proposal.get("root_operator") or operator,
                    "call_index": call_index,
                    "length_mode": proposal.get("length_mode", "fixed"),
                    "removed_tokens": proposal.get("removed_tokens"),
                    "inserted_tokens": proposal.get("inserted_tokens"),
                    "actual_length_delta": proposal.get("actual_delta"),
                }
            )

        after_scores = [float(score) for score, _ in self.oracle.mol_buffer.values()]
        after_top10 = self._v8_top_mean(after_scores, top_n=10)
        group_scores = [row["child_score"] for row in evaluated_rows]
        parent_scores = [row["parent_score"] for row in evaluated_rows]
        if frontier_engine is not None:
            group = "root" if operator in root_operators else "local"
            reward, reward_parts = frontier_engine.update_scalar_batch(
                group=group,
                operator=operator,
                scores=group_scores,
                before_scores=before_scores,
                before_top10=before_top10,
                after_top10=after_top10,
                parent_scores=parent_scores,
            )
        else:
            frontier_scale = max(
                self.args.v9_frontier_min_scale,
                float(np.median(frontier_history[-64:])) if frontier_history else 0.0,
            )
            delta_scale = max(
                self.args.v9_delta_min_scale,
                float(np.median(delta_history[-512:])) if delta_history else 0.0,
            )
            reward, reward_parts = batch_frontier_reward(
                scores=group_scores,
                before_scores=before_scores,
                before_top10=before_top10,
                after_top10=after_top10,
                parent_scores=parent_scores,
                frontier_scale=frontier_scale,
                delta_scale=delta_scale,
            )
            bandit = root_bandit if operator in root_operators else local_bandit
            bandit.update(operator, reward, len(evaluated_rows))

        if protected_frontier_engine is not None:
            top10_entries = int(
                round(reward_parts["entry_rate"] * min(10, len(evaluated_rows)))
            )
            protected_frontier_engine.observe_batch(
                operator=operator,
                reward=reward,
                evaluated=len(evaluated_rows),
                calls=len(self.oracle.mol_buffer),
                state=state,
                frontier_gain=reward_parts["frontier_gain"],
                entry_rate=reward_parts["entry_rate"],
                top10_entries=top10_entries,
            )

        if frontier_engine is None and reward_parts["frontier_gain"] > 0:
            frontier_history.append(float(reward_parts["frontier_gain"]))
            del frontier_history[:-256]
        for row in evaluated_rows:
            if row["parent_score"] is not None:
                if frontier_engine is None:
                    delta_history.append(
                        abs(float(row["child_score"]) - float(row["parent_score"]))
                    )
        if frontier_engine is None:
            del delta_history[:-2048]

        merged_scores = sorted(before_scores + group_scores, reverse=True)
        final_threshold = (
            merged_scores[min(9, len(merged_scores) - 1)] if merged_scores else math.inf
        )
        entry_budget = min(10, len(group_scores))
        ranked_rows = sorted(
            evaluated_rows,
            key=lambda row: row["child_score"],
            reverse=True,
        )
        entered_ids = {
            id(row)
            for row in ranked_rows
            if row["child_score"] >= final_threshold - 1e-12
        }
        if len(entered_ids) > entry_budget:
            entered_ids = {id(row) for row in ranked_rows[:entry_budget]}

        transition_rows = []
        frontier_share = reward_parts["frontier_gain"] / max(1, len(entered_ids))
        credited_root_operators = set()
        for row in evaluated_rows:
            entered_top10 = id(row) in entered_ids
            parent = row["parent_record"]
            if parent is None:
                root_id = hashlib.sha1(row["child_smiles"].encode("utf-8")).hexdigest()[
                    :16
                ]
                record = archive.add_root(
                    smiles=row["child_smiles"],
                    score=row["child_score"],
                    root_id=root_id,
                    root_operator=row["root_operator"],
                    created_call=row["call_index"],
                )
                if entered_top10:
                    archive.roots[root_id].top10_entries += 1
                    archive.roots[root_id].frontier_credit += frontier_share
            else:
                record = archive.add_child(
                    smiles=row["child_smiles"],
                    score=row["child_score"],
                    parent=parent,
                    operator=operator,
                    created_call=row["call_index"],
                    frontier_gain=frontier_share if entered_top10 else 0.0,
                    entered_top10=entered_top10,
                )
                credited_root_operators.add(parent.root_operator)

            if before_scores:
                percentile = sum(
                    old <= row["child_score"] for old in before_scores
                ) / len(before_scores)
            else:
                percentile = float(np.clip(row["child_score"], 0.0, 1.0))
            if entered_top10 or percentile >= self.args.v9_archive_percentile:
                self._update_fragment_population_v2(
                    adaptive_population,
                    row["child_smiles"],
                    row["child_score"],
                )
            if entered_top10:
                self._update_v5_motif_archive(
                    motifs,
                    row["child_smiles"],
                    row["child_score"],
                )

            transition_rows.append(
                {
                    "oracle": oracle_name,
                    "call": row["call_index"],
                    "state": state,
                    "operator": operator,
                    "root_operator": record.root_operator,
                    "root_id": record.root_id,
                    "depth": record.depth,
                    "parent_smiles": "" if parent is None else parent.smiles,
                    "seed_smiles": row["seed_smiles"],
                    "child_smiles": row["child_smiles"],
                    "parent_score": ""
                    if row["parent_score"] is None
                    else row["parent_score"],
                    "child_score": row["child_score"],
                    "delta": ""
                    if row["parent_score"] is None
                    else row["child_score"] - row["parent_score"],
                    "entered_batch_top10": int(entered_top10),
                    "batch_reward": reward,
                    "batch_frontier_gain": reward_parts["frontier_gain"],
                    "batch_frontier_signal": reward_parts["frontier_signal"],
                    "batch_entry_rate": reward_parts["entry_rate"],
                    "batch_tail_signal": reward_parts["tail_signal"],
                    "batch_positive_delta_signal": reward_parts[
                        "positive_delta_signal"
                    ],
                    "child_atoms": row["child_atoms"],
                    "min_atoms": min_size,
                    "max_atoms": max_size,
                    "length_mode": row["length_mode"],
                    "removed_tokens": row["removed_tokens"],
                    "inserted_tokens": row["inserted_tokens"],
                    "actual_length_delta": row["actual_length_delta"],
                }
            )

        delayed = 0.70 * reward_parts["frontier_signal"] + 0.30 * min(
            1.0, reward_parts["entry_rate"] / 0.10
        )
        if delayed > 0:
            for root_operator in credited_root_operators:
                if frontier_engine is not None:
                    frontier_engine.delayed_credit(
                        "root",
                        root_operator,
                        delayed,
                        alpha=self.args.v9_delayed_credit_alpha,
                    )
                else:
                    root_bandit.delayed_credit(
                        root_operator,
                        delayed,
                        alpha=self.args.v9_delayed_credit_alpha,
                    )

        self._append_v9_transition_rows(transition_path, transition_rows)
        return len(evaluated_rows), float(reward_parts["frontier_gain"])

    @staticmethod
    def _append_v9_transition_rows(path, rows):
        if not rows:
            return
        os.makedirs(os.path.dirname(path), exist_ok=True)
        fields = list(rows[0])
        exists = os.path.exists(path)
        with open(path, "a", newline="") as handle:
            writer = csv.DictWriter(handle, fieldnames=fields, extrasaction="ignore")
            if not exists:
                writer.writeheader()
            writer.writerows(rows)

    @staticmethod
    def _v9_bandit_snapshot(bandit):
        parts = []
        for operator, row in sorted(bandit.stats.items()):
            parts.append(
                f"{operator}:ema={float(row['ema']):.5f},"
                f"batches={int(row['batches'])},eval={int(row['evaluated'])}"
            )
        return ";".join(parts)

    def _append_v9_diagnostics(
        self,
        path,
        oracle_name,
        calls,
        state,
        avg_top10,
        auc_top10,
        nonzero_rate,
        stagnant_calls,
        lineage_metrics,
        root_bandit,
        local_bandit,
        frontier_engine=None,
        protected_frontier_engine=None,
    ):
        row = {
            "oracle": oracle_name,
            "calls": calls,
            "state": state,
            "avg_top10": avg_top10,
            "auc_top10": auc_top10,
            "nonzero_rate": nonzero_rate,
            "stagnant_calls": stagnant_calls,
            "root_count": lineage_metrics["root_count"],
            "largest_root_fraction": lineage_metrics["largest_root_fraction"],
            "lineage_entropy": lineage_metrics["lineage_entropy"],
            "rejected_invalid": self._v9_rejections["invalid"],
            "rejected_duplicate": self._v9_rejections["duplicate"],
            "rejected_out_of_bounds": self._v9_rejections["out_of_bounds"],
            "rejected_untokenizable": self._v9_rejections["untokenizable"],
            "root_bandit": self._v9_bandit_snapshot(root_bandit),
            "local_bandit": self._v9_bandit_snapshot(local_bandit),
        }
        if frontier_engine is not None:
            snapshot = frontier_engine.snapshot()
            row.update(
                {
                    "frontier_policy": snapshot["adapter"],
                    "frontier_scale": snapshot["frontier_scale"],
                    "delta_scale": snapshot["delta_scale"],
                }
            )
        if protected_frontier_engine is not None and hasattr(
            protected_frontier_engine.task_head,
            "snapshot",
        ):
            snapshot = protected_frontier_engine.task_head.snapshot()
            row.update(
                {
                    "protected_policy_phase": snapshot.get("phase"),
                    "protected_policy_reserve": snapshot.get("reserve"),
                    "protected_policy_effective_reserve": snapshot.get(
                        "effective_reserve", snapshot.get("reserve")
                    ),
                    "protected_policy_advantage": snapshot.get("last_advantage"),
                    "protected_policy_frontier_advantage": snapshot.get(
                        "last_frontier_advantage"
                    ),
                    "protected_policy_uncertainty": snapshot.get("last_uncertainty"),
                    "protected_policy_snapshot": json.dumps(
                        snapshot,
                        sort_keys=True,
                    ),
                }
            )
        os.makedirs(os.path.dirname(path), exist_ok=True)
        fields = list(row)
        exists = os.path.exists(path)
        with open(path, "a", newline="") as handle:
            writer = csv.DictWriter(handle, fieldnames=fields)
            if not exists:
                writer.writeheader()
            writer.writerow(row)

    @staticmethod
    def _save_frontier_state(path, state):
        os.makedirs(os.path.dirname(path), exist_ok=True)
        temporary = f"{path}.tmp.{os.getpid()}"
        with open(temporary, "w") as handle:
            json.dump(state, handle, sort_keys=True)
        os.replace(temporary, path)

    def _v9_make_local_groups(
        self,
        counts,
        archive,
        population,
        min_size,
        max_size,
        state,
        rng,
        frontier_engine=None,
    ):
        groups = {}
        score_pool, lineage_pool = archive.parent_pools()
        fractions = sorted(
            parse_float_list(self.args.v5_remask_fractions, [0.06, 0.14, 0.28, 0.50])
        )
        tiny = fractions[0]
        small = fractions[min(1, len(fractions) - 1)]
        medium = fractions[min(2, len(fractions) - 1)]
        large = fractions[-1]
        exploit_temp = max(0.90, self.args.temperature_start - 0.14)
        explore_temp = max(self.args.temperature_start, 1.25)
        specs = {
            "elite_tiny": (tiny, exploit_temp, max(0.40, self.args.span_prob - 0.16)),
            "elite_refine": (
                tiny,
                exploit_temp,
                max(0.40, self.args.span_prob - 0.16),
            ),
            "elite_small": (small, exploit_temp, max(0.45, self.args.span_prob - 0.10)),
            "elite_medium": (medium, self.args.temperature_start, self.args.span_prob),
            "diverse_medium": (
                medium,
                explore_temp,
                min(1.0, self.args.span_prob + 0.05),
            ),
            "graph_shrink": (
                small,
                self.args.temperature_start,
                min(1.0, self.args.span_prob + 0.04),
            ),
            "graph_swap": (
                small,
                self.args.temperature_start,
                min(1.0, self.args.span_prob + 0.06),
            ),
            "graph_expand": (
                medium,
                explore_temp,
                min(1.0, self.args.span_prob + 0.08),
            ),
            "rescue_large": (
                large,
                self.args.v5_rescue_temperature,
                min(1.0, self.args.span_prob + 0.18),
            ),
        }
        for operator, count in counts.items():
            if count <= 0 or operator not in specs:
                continue
            proposals = []
            attempts = 0
            while len(proposals) < count and attempts < max(100, count * 80):
                attempts += 1
                lineage_probability = 0.65 if operator == "diverse_medium" else 0.30
                if lineage_pool and rng.random() < lineage_probability:
                    parent = rng.choice(lineage_pool)
                elif score_pool:
                    parent = rng.choice(score_pool)
                elif lineage_pool:
                    parent = rng.choice(lineage_pool)
                else:
                    parent = None
                if parent is None:
                    break
                seed = parent.smiles
                if operator in {"graph_shrink", "graph_swap", "graph_expand"} and not (
                    bool(getattr(self.model, "is_elastic", False))
                    and operator in {"graph_shrink", "graph_expand"}
                ):
                    seed = self._make_v8_graph_fragment_edit(
                        parent_smiles=parent.smiles,
                        population=population,
                        direction=operator.removeprefix("graph_"),
                        min_size=min_size,
                        max_size=max_size,
                    )
                can = canonical_smiles(seed) if seed else None
                if can is None or not min_size <= atom_count(can) <= max_size:
                    continue
                proposals.append(
                    {
                        "seed": can,
                        "parent_record": parent,
                        "root_operator": parent.root_operator,
                    }
                )
            if not proposals:
                continue
            remask_fraction, temperature_start, span_prob = specs[operator]
            edit_plans = [None] * len(proposals)
            if bool(getattr(self.model, "is_elastic", False)) and not getattr(
                self.args, "disable_learned_insertion", False
            ):
                adapter = (
                    frontier_engine.adapter
                    if frontier_engine is not None
                    else ScalarFrontierAdapter()
                )
                insertion_fraction = max(
                    0.0,
                    min(
                        1.0,
                        adapter.insertion_fraction(state, None)
                        * getattr(
                            self.args,
                            "learned_insertion_fraction_scale",
                            1.0,
                        ),
                    ),
                )
                insertion_flags = allocate_insertion_flags(
                    len(proposals),
                    insertion_fraction,
                    rng=rng,
                )
                state_cap = {
                    "warmup": 2,
                    "saturated": 1,
                    "search": 3,
                    "sparse": 3,
                    "plateau": 4,
                    "collapsed": 4,
                    "fallback": 4,
                }.get(state, 3)
                max_growth = min(
                    max(
                        0,
                        getattr(self.args, "learned_insertion_max_growth", 4),
                    ),
                    state_cap,
                )
                max_shrink = min(
                    max(
                        0,
                        getattr(self.args, "learned_insertion_max_shrink", 4),
                    ),
                    state_cap,
                )
                for index, use_insertion in enumerate(insertion_flags):
                    if not use_insertion:
                        continue
                    tokens = tokenize_smiles(proposals[index]["seed"])
                    max_span_tokens = max(
                        1,
                        min(
                            len(tokens),
                            int(round(len(tokens) * remask_fraction)),
                        ),
                    )
                    plan = adaptive_peripheral_edit_plan(
                        proposals[index]["seed"],
                        rng,
                        delta=0,
                        target_atom_fraction=remask_fraction,
                        max_atom_fraction=min(
                            0.40,
                            max(0.12, remask_fraction * 1.5),
                        ),
                        max_span_tokens=max_span_tokens,
                    )
                    if plan is None:
                        plan = atom_span_edit_plan(
                            proposals[index]["seed"],
                            rng,
                            delta=0,
                            span_tokens=max_span_tokens,
                        )
                    if plan is None:
                        continue
                    removed = max(
                        1,
                        int(plan["stop"]) - int(plan["start"]),
                    )
                    plan["length_mode"] = "learned_insertion"
                    plan["min_replacement_len"] = max(0, removed - max_shrink)
                    plan["max_replacement_len"] = max(
                        plan["min_replacement_len"],
                        removed + max_growth,
                    )
                    if self.mode in ELASTIC_PRESCREEN_MODES:
                        # Materialize the hard lower bound before planning. The
                        # elastic head still decides every optional slot up to
                        # max_replacement_len, while the fill phase can refine
                        # the guaranteed slots progressively instead of adding
                        # them only during the terminal fallback.
                        plan["initial_replacement_len"] = plan["min_replacement_len"]
                    edit_plans[index] = plan
                    proposals[index]["length_mode"] = "learned_insertion"
            elastic_frontier = self.mode in ELASTIC_PRESCREEN_MODES
            indexed = sample_csdnet_local_remask(
                model=self.model,
                tk=self.tk,
                seed_smiles=[row["seed"] for row in proposals],
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
                temperature_start=temperature_start,
                temperature_end=self.args.temperature_end,
                temperature_power=self.args.temperature_power,
                length_delta_choices="0",
                length_edit_prob=0.0,
                edit_plans=edit_plans,
                return_seed_indices=True,
                return_diagnostics=True,
                learned_insertion_max_growth=getattr(
                    self.args,
                    "learned_insertion_max_growth",
                    4,
                ),
                learned_insertion_max_shrink=getattr(
                    self.args,
                    "learned_insertion_max_shrink",
                    4,
                ),
                learned_insertion_max_per_step=getattr(
                    self.args,
                    "learned_insertion_max_per_step",
                    4,
                ),
                learned_insertion_recursive_gap_insertions=elastic_frontier,
                learned_insertion_trajectory_mode=(
                    "plan_then_fill" if elastic_frontier else "coupled"
                ),
                learned_insertion_planning_fraction=0.30,
                learned_insertion_fill_mode=(
                    "progressive_remask" if elastic_frontier else "absorbing"
                ),
                learned_insertion_fill_remask_power=1.0,
                learned_insertion_fill_gumbel_scale=0.35,
                local_sampler_profile=(
                    "task_adaptive_refine"
                    if operator == "elite_refine"
                    else "task_adaptive_local"
                ),
            )
            lineage = []
            for sampled_item in indexed:
                candidate, proposal_idx = sampled_item[:2]
                diagnostics = sampled_item[2] if len(sampled_item) > 2 else {}
                if 0 <= proposal_idx < len(proposals):
                    proposals[proposal_idx].update(diagnostics)
                    lineage.append((candidate, proposals[proposal_idx]))
            if lineage:
                groups[operator] = lineage
        return groups

    def _nonzero_threshold(self):
        if self.mode in {
            "iterative_remask_v7",
            "iterative_remask_v8",
            "iterative_remask_v9",
            "iterative_remask_v9_no_prescreen",
            "elastic_frontier_prescreen",
            "elastic_frontier_prescreen_v2",
            "iterative_remask_v10",
            "safe_frontier_final",
            "iterative_remask_v9_gated",
            "iterative_remask_v9_reversible",
            "unified_frontier",
            "unified_frontier_v2",
            "unified_frontier_restored",
        }:
            return getattr(self.args, "v5_nonzero_threshold", 1e-8)
        if self.mode == "iterative_remask_v6":
            return getattr(self.args, "v5_nonzero_threshold", 1e-8)
        if self.mode == "iterative_remask_v5":
            return getattr(self.args, "v5_nonzero_threshold", 1e-8)
        if self.mode == "iterative_remask_v4":
            return getattr(self.args, "v4_nonzero_threshold", 1e-8)
        return getattr(self.args, "v3_nonzero_threshold", 1e-8)

    def _nonzero_count(self):
        return sum(
            1
            for score, _ in self.oracle.mol_buffer.values()
            if float(score) > self._nonzero_threshold()
        )

    @staticmethod
    def _append_v3_diagnostics(
        path,
        oracle_name,
        calls,
        best_score,
        nonzero,
        elites,
        population,
        stagnant_calls,
        zero_rescue,
        operator_stats,
    ):
        os.makedirs(os.path.dirname(path), exist_ok=True)
        exists = os.path.exists(path)
        fields = [
            "oracle",
            "calls",
            "best_score",
            "nonzero",
            "elites",
            "population",
            "stagnant_calls",
            "zero_rescue",
            "operator_stats",
        ]
        with open(path, "a", newline="") as f:
            writer = csv.DictWriter(f, fieldnames=fields)
            if not exists:
                writer.writeheader()
            writer.writerow(
                {
                    "oracle": oracle_name,
                    "calls": calls,
                    "best_score": best_score,
                    "nonzero": nonzero,
                    "elites": elites,
                    "population": population,
                    "stagnant_calls": stagnant_calls,
                    "zero_rescue": int(bool(zero_rescue)),
                    "operator_stats": ";".join(
                        f"{k}:{v:.3f}" for k, v in sorted(operator_stats.items())
                    ),
                }
            )

    @staticmethod
    def _append_v4_diagnostics(
        path,
        oracle_name,
        calls,
        best_score,
        nonzero,
        elites,
        diverse,
        motifs,
        population,
        stagnant_calls,
        rescue,
        zero_rescue,
        operator_stats,
    ):
        os.makedirs(os.path.dirname(path), exist_ok=True)
        exists = os.path.exists(path)
        fields = [
            "oracle",
            "calls",
            "best_score",
            "nonzero",
            "elites",
            "diverse",
            "motifs",
            "population",
            "stagnant_calls",
            "rescue",
            "zero_rescue",
            "operator_stats",
        ]
        with open(path, "a", newline="") as f:
            writer = csv.DictWriter(f, fieldnames=fields)
            if not exists:
                writer.writeheader()
            writer.writerow(
                {
                    "oracle": oracle_name,
                    "calls": calls,
                    "best_score": best_score,
                    "nonzero": nonzero,
                    "elites": elites,
                    "diverse": diverse,
                    "motifs": motifs,
                    "population": population,
                    "stagnant_calls": stagnant_calls,
                    "rescue": int(bool(rescue)),
                    "zero_rescue": int(bool(zero_rescue)),
                    "operator_stats": ";".join(
                        f"{k}:{v:.3f}" for k, v in sorted(operator_stats.items())
                    ),
                }
            )

    @staticmethod
    def _append_v5_diagnostics(
        path,
        oracle_name,
        calls,
        state,
        best_score,
        nonzero,
        avg_top10,
        auc_top10,
        nonzero_rate,
        late_gap,
        elites,
        diverse,
        motifs,
        population,
        stagnant_calls,
        operator_stats,
    ):
        os.makedirs(os.path.dirname(path), exist_ok=True)
        exists = os.path.exists(path)
        fields = [
            "oracle",
            "calls",
            "state",
            "best_score",
            "nonzero",
            "avg_top10",
            "auc_top10",
            "nonzero_rate",
            "late_gap",
            "elites",
            "diverse",
            "motifs",
            "population",
            "stagnant_calls",
            "operator_stats",
        ]
        with open(path, "a", newline="") as f:
            writer = csv.DictWriter(f, fieldnames=fields)
            if not exists:
                writer.writeheader()
            writer.writerow(
                {
                    "oracle": oracle_name,
                    "calls": calls,
                    "state": state,
                    "best_score": best_score,
                    "nonzero": nonzero,
                    "avg_top10": avg_top10,
                    "auc_top10": auc_top10,
                    "nonzero_rate": nonzero_rate,
                    "late_gap": late_gap,
                    "elites": elites,
                    "diverse": diverse,
                    "motifs": motifs,
                    "population": population,
                    "stagnant_calls": stagnant_calls,
                    "operator_stats": ";".join(
                        f"{k}:{v:.3f}" for k, v in sorted(operator_stats.items())
                    ),
                }
            )

    def _v5_feedback_state(self, calls, stagnant_calls, best_score):
        metrics = summarize_buffer(
            self.oracle.mol_buffer,
            max_oracle_calls=self.args.max_oracle_calls,
            freq_log=self.args.freq_log,
        )
        nonzero = self._nonzero_count()
        nonzero_rate = nonzero / max(1, calls)
        avg_top10 = float(metrics.get("avg_top10", 0.0))
        auc_top10 = float(metrics.get("auc_top10", 0.0))
        avg_top1 = float(metrics.get("avg_top1", best_score))
        late_gap = max(0.0, avg_top10 - auc_top10)
        metrics.update(
            {
                "nonzero": nonzero,
                "nonzero_rate": nonzero_rate,
                "late_gap": late_gap,
            }
        )

        if calls < self.args.v5_warmup_calls:
            return "warmup", metrics
        if stagnant_calls >= self.args.v5_stagnation_rescue_patience:
            return "rescue", metrics
        if nonzero_rate < self.args.v5_sparse_nonzero_rate:
            if avg_top1 >= self.args.v5_good_top1_threshold:
                return "sparse_exploit", metrics
            return "sparse", metrics
        if avg_top10 >= self.args.v5_high_top10_threshold:
            return "refine", metrics
        if (
            avg_top1 >= self.args.v5_good_top1_threshold
            and late_gap >= self.args.v5_late_gap_threshold
        ):
            return "exploit", metrics
        if (
            avg_top1 < self.args.v5_low_top1_threshold
            and nonzero_rate >= self.args.v5_sparse_nonzero_rate
        ):
            return "explore", metrics
        return "balanced", metrics

    def _make_v5_seed_groups(
        self,
        population,
        elites,
        diverse,
        motifs,
        min_size,
        max_size,
        state,
        operator_stats,
    ):
        fractions = sorted(
            parse_float_list(self.args.v5_remask_fractions, [0.06, 0.14, 0.28, 0.50])
        )
        tiny = fractions[0]
        small = fractions[min(1, len(fractions) - 1)]
        medium = fractions[min(2, len(fractions) - 1)]
        large = fractions[-1]
        target_n = max(
            self.args.candidate_batch_size,
            int(
                round(self.args.candidate_batch_size * self.args.v5_overgenerate_factor)
            ),
        )

        base_weights = self._v5_operator_weights(
            state=state,
            has_elites=bool(elites),
            has_diverse=bool(diverse),
            has_motifs=bool(motifs),
        )
        weighted_ops = []
        for op, base in base_weights.items():
            if base <= 0:
                continue
            weighted_ops.append((op, base * (0.45 + operator_stats.get(op, 1.0))))
        if not weighted_ops:
            weighted_ops = [("fragment_medium", 1.0)]

        exploit_temp = max(0.95, self.args.temperature_start - 0.12)
        explore_temp = max(self.args.temperature_start, 1.30)
        rescue_temp = self.args.v5_rescue_temperature
        specs = {
            "elite_tiny": {
                "remask_fraction": tiny,
                "temperature_start": exploit_temp,
                "span_prob": max(0.40, self.args.span_prob - 0.15),
                "seeds": [],
            },
            "elite_small": {
                "remask_fraction": small,
                "temperature_start": exploit_temp,
                "span_prob": max(0.45, self.args.span_prob - 0.10),
                "seeds": [],
            },
            "elite_medium": {
                "remask_fraction": medium,
                "temperature_start": self.args.temperature_start,
                "span_prob": self.args.span_prob,
                "seeds": [],
            },
            "diverse_medium": {
                "remask_fraction": medium,
                "temperature_start": explore_temp,
                "span_prob": min(1.0, self.args.span_prob + 0.05),
                "seeds": [],
            },
            "fragment_medium": {
                "remask_fraction": medium,
                "temperature_start": explore_temp,
                "span_prob": min(1.0, self.args.span_prob + 0.08),
                "seeds": [],
            },
            "fragment_large": {
                "remask_fraction": large,
                "temperature_start": max(explore_temp, 1.45),
                "span_prob": min(1.0, self.args.span_prob + 0.18),
                "seeds": [],
            },
            "rescue_large": {
                "remask_fraction": large,
                "temperature_start": rescue_temp,
                "span_prob": min(1.0, self.args.span_prob + 0.25),
                "seeds": [],
            },
            "motif_seeded": {
                "motif_seeded": True,
                "temperature_start": explore_temp
                if state in {"explore", "sparse"}
                else self.args.temperature_start,
                "n_mol": 0,
            },
        }

        attempts = 0
        total = 0
        while total < target_n and attempts < target_n * 180:
            attempts += 1
            op = self._weighted_choice(weighted_ops)
            if op == "motif_seeded":
                specs[op]["n_mol"] += 1
                total += 1
                continue
            if op.startswith("elite") and elites:
                top_n = max(1, min(len(elites), self.args.elite_size))
                smi = random.choice(elites[:top_n])[1]
            elif op == "diverse_medium" and diverse:
                smi = random.choice(diverse)[1]
            else:
                smi = self._make_fragment_seed_v2(
                    population,
                    min_size,
                    max_size,
                    prefer_top=state in {"exploit", "sparse_exploit", "refine"},
                )
            can = canonical_smiles(smi) if smi else None
            if can is None:
                continue
            atoms = atom_count(can)
            if atoms < min_size or atoms > max_size:
                continue
            if not tokenizable(can, self.tk, self.args.max_len):
                continue
            specs[op]["seeds"].append(can)
            total += 1

        out = {}
        for op, spec in specs.items():
            if spec.get("motif_seeded"):
                if spec["n_mol"] > 0:
                    out[op] = spec
            elif spec["seeds"]:
                out[op] = spec
        return out

    def _make_v7_seed_groups(
        self,
        population,
        elites,
        diverse,
        motifs,
        min_size,
        max_size,
        state,
        stagnant_calls,
        operator_stats,
    ):
        fractions = sorted(
            parse_float_list(self.args.v5_remask_fractions, [0.06, 0.14, 0.28, 0.50])
        )
        tiny = fractions[0]
        small = fractions[min(1, len(fractions) - 1)]
        medium = fractions[min(2, len(fractions) - 1)]
        large = fractions[-1]
        target_n = max(
            self.args.candidate_batch_size,
            int(
                round(self.args.candidate_batch_size * self.args.v5_overgenerate_factor)
            ),
        )

        base_weights = self._v7_operator_weights(
            state=state,
            has_elites=bool(elites),
            has_diverse=bool(diverse),
            has_motifs=bool(motifs),
            stagnant_calls=stagnant_calls,
        )
        weighted_ops = []
        for op, base in base_weights.items():
            if base <= 0:
                continue
            weighted_ops.append((op, base * (0.45 + operator_stats.get(op, 1.0))))
        if not weighted_ops:
            weighted_ops = [("fragment_medium", 1.0)]

        exploit_temp = max(0.95, self.args.temperature_start - 0.12)
        explore_temp = max(self.args.temperature_start, 1.30)
        rescue_temp = self.args.v5_rescue_temperature
        disabled_length = {
            "length_delta_choices": "0",
            "length_edit_prob": 0.0,
            "length_edit_min_span": 1,
            "length_edit_max_span": 1,
        }
        shrink_length = {
            "length_delta_choices": self.args.v7_length_shrink_deltas,
            "length_edit_prob": self.args.v7_length_edit_prob,
            "length_edit_min_span": self.args.v7_length_edit_min_span,
            "length_edit_max_span": self.args.v7_length_edit_max_span,
        }
        expand_length = {
            "length_delta_choices": self.args.v7_length_expand_deltas,
            "length_edit_prob": self.args.v7_length_edit_prob,
            "length_edit_min_span": self.args.v7_length_edit_min_span,
            "length_edit_max_span": self.args.v7_length_edit_max_span,
        }
        specs = {
            "elite_tiny": {
                "remask_fraction": tiny,
                "temperature_start": exploit_temp,
                "span_prob": max(0.40, self.args.span_prob - 0.15),
                "length_kwargs": disabled_length,
                "seeds": [],
            },
            "elite_small": {
                "remask_fraction": small,
                "temperature_start": exploit_temp,
                "span_prob": max(0.45, self.args.span_prob - 0.10),
                "length_kwargs": disabled_length,
                "seeds": [],
            },
            "elite_medium": {
                "remask_fraction": medium,
                "temperature_start": self.args.temperature_start,
                "span_prob": self.args.span_prob,
                "length_kwargs": disabled_length,
                "seeds": [],
            },
            "diverse_medium": {
                "remask_fraction": medium,
                "temperature_start": explore_temp,
                "span_prob": min(1.0, self.args.span_prob + 0.05),
                "length_kwargs": disabled_length,
                "seeds": [],
            },
            "fragment_medium": {
                "remask_fraction": medium,
                "temperature_start": explore_temp,
                "span_prob": min(1.0, self.args.span_prob + 0.08),
                "length_kwargs": disabled_length,
                "seeds": [],
            },
            "fragment_large": {
                "remask_fraction": large,
                "temperature_start": max(explore_temp, 1.45),
                "span_prob": min(1.0, self.args.span_prob + 0.18),
                "length_kwargs": disabled_length,
                "seeds": [],
            },
            "rescue_large": {
                "remask_fraction": large,
                "temperature_start": rescue_temp,
                "span_prob": min(1.0, self.args.span_prob + 0.25),
                "length_kwargs": disabled_length,
                "seeds": [],
            },
            "length_shrink_rescue": {
                "remask_fraction": max(small, medium * 0.80),
                "temperature_start": max(explore_temp, rescue_temp - 0.10),
                "span_prob": min(1.0, self.args.span_prob + 0.15),
                "length_kwargs": shrink_length,
                "seeds": [],
            },
            "length_expand_rescue": {
                "remask_fraction": max(small, medium * 0.80),
                "temperature_start": max(explore_temp, rescue_temp - 0.10),
                "span_prob": min(1.0, self.args.span_prob + 0.15),
                "length_kwargs": expand_length,
                "seeds": [],
            },
            "motif_seeded": {
                "motif_seeded": True,
                "temperature_start": explore_temp
                if state in {"explore", "sparse"}
                else self.args.temperature_start,
                "n_mol": 0,
            },
        }

        attempts = 0
        total = 0
        while total < target_n and attempts < target_n * 200:
            attempts += 1
            op = self._weighted_choice(weighted_ops)
            if op == "motif_seeded":
                specs[op]["n_mol"] += 1
                total += 1
                continue
            if op.startswith("elite") and elites:
                top_n = max(1, min(len(elites), self.args.elite_size))
                smi = random.choice(elites[:top_n])[1]
            elif op in {"length_shrink_rescue", "length_expand_rescue"} and elites:
                top_n = max(1, min(len(elites), max(10, self.args.elite_size // 2)))
                smi = random.choice(elites[:top_n])[1]
            elif op == "diverse_medium" and diverse:
                smi = random.choice(diverse)[1]
            else:
                smi = self._make_fragment_seed_v2(
                    population,
                    min_size,
                    max_size,
                    prefer_top=state in {"exploit", "sparse_exploit", "refine"},
                )
            can = canonical_smiles(smi) if smi else None
            if can is None:
                continue
            atoms = atom_count(can)
            if atoms < min_size or atoms > max_size:
                continue
            if not tokenizable(can, self.tk, self.args.max_len):
                continue
            specs[op]["seeds"].append(can)
            total += 1

        out = {}
        for op, spec in specs.items():
            if spec.get("motif_seeded"):
                if spec["n_mol"] > 0:
                    out[op] = spec
            elif spec["seeds"]:
                out[op] = spec
        return out

    def _v7_operator_weights(
        self, state, has_elites, has_diverse, has_motifs, stagnant_calls
    ):
        weights = self._v5_operator_weights(
            state=state,
            has_elites=has_elites,
            has_diverse=has_diverse,
            has_motifs=has_motifs,
        )
        weights.setdefault("length_shrink_rescue", 0.0)
        weights.setdefault("length_expand_rescue", 0.0)
        length_ready = (
            has_elites and stagnant_calls >= self.args.v7_length_rescue_after_stagnant
        )
        if state == "rescue" and has_elites:
            weights["length_shrink_rescue"] = self.args.v7_length_rescue_weight
            weights["length_expand_rescue"] = self.args.v7_length_rescue_weight
        elif length_ready:
            weights["length_shrink_rescue"] = self.args.v7_length_rescue_weight * 0.75
            weights["length_expand_rescue"] = self.args.v7_length_rescue_weight * 0.75
        elif state in {"explore", "sparse"} and has_elites:
            weights["length_shrink_rescue"] = self.args.v7_length_rescue_weight * 0.35
            weights["length_expand_rescue"] = self.args.v7_length_rescue_weight * 0.35
        return weights

    @staticmethod
    def _v5_operator_weights(state, has_elites, has_diverse, has_motifs):
        weights = {
            "elite_tiny": 0.16 if has_elites else 0.0,
            "elite_small": 0.18 if has_elites else 0.0,
            "elite_medium": 0.12 if has_elites else 0.0,
            "diverse_medium": 0.12 if has_diverse else 0.0,
            "motif_seeded": 0.22 if has_motifs else 0.0,
            "fragment_medium": 0.14,
            "fragment_large": 0.06,
            "rescue_large": 0.0,
        }
        if state == "warmup":
            weights.update(
                {
                    "elite_tiny": 0.10 if has_elites else 0.0,
                    "elite_small": 0.14 if has_elites else 0.0,
                    "elite_medium": 0.10 if has_elites else 0.0,
                    "diverse_medium": 0.16 if has_diverse else 0.0,
                    "motif_seeded": 0.26 if has_motifs else 0.0,
                    "fragment_medium": 0.22,
                    "fragment_large": 0.12,
                }
            )
        elif state == "sparse":
            weights.update(
                {
                    "elite_tiny": 0.08 if has_elites else 0.0,
                    "elite_small": 0.10 if has_elites else 0.0,
                    "elite_medium": 0.08 if has_elites else 0.0,
                    "diverse_medium": 0.18 if has_diverse else 0.0,
                    "motif_seeded": 0.30 if has_motifs else 0.0,
                    "fragment_medium": 0.18,
                    "fragment_large": 0.16,
                }
            )
        elif state == "sparse_exploit":
            weights.update(
                {
                    "elite_tiny": 0.32 if has_elites else 0.0,
                    "elite_small": 0.26 if has_elites else 0.0,
                    "elite_medium": 0.12 if has_elites else 0.0,
                    "diverse_medium": 0.08 if has_diverse else 0.0,
                    "motif_seeded": 0.14 if has_motifs else 0.0,
                    "fragment_medium": 0.06,
                    "fragment_large": 0.02,
                }
            )
        elif state == "exploit":
            weights.update(
                {
                    "elite_tiny": 0.34 if has_elites else 0.0,
                    "elite_small": 0.28 if has_elites else 0.0,
                    "elite_medium": 0.12 if has_elites else 0.0,
                    "diverse_medium": 0.08 if has_diverse else 0.0,
                    "motif_seeded": 0.12 if has_motifs else 0.0,
                    "fragment_medium": 0.04,
                    "fragment_large": 0.02,
                }
            )
        elif state == "explore":
            weights.update(
                {
                    "elite_tiny": 0.04 if has_elites else 0.0,
                    "elite_small": 0.08 if has_elites else 0.0,
                    "elite_medium": 0.08 if has_elites else 0.0,
                    "diverse_medium": 0.20 if has_diverse else 0.0,
                    "motif_seeded": 0.24 if has_motifs else 0.0,
                    "fragment_medium": 0.22,
                    "fragment_large": 0.14,
                }
            )
        elif state == "rescue":
            weights.update(
                {
                    "elite_tiny": 0.06 if has_elites else 0.0,
                    "elite_small": 0.08 if has_elites else 0.0,
                    "elite_medium": 0.06 if has_elites else 0.0,
                    "diverse_medium": 0.16 if has_diverse else 0.0,
                    "motif_seeded": 0.24 if has_motifs else 0.0,
                    "fragment_medium": 0.16,
                    "fragment_large": 0.12,
                    "rescue_large": 0.18,
                }
            )
        elif state == "refine":
            weights.update(
                {
                    "elite_tiny": 0.42 if has_elites else 0.0,
                    "elite_small": 0.24 if has_elites else 0.0,
                    "elite_medium": 0.10 if has_elites else 0.0,
                    "diverse_medium": 0.06 if has_diverse else 0.0,
                    "motif_seeded": 0.08 if has_motifs else 0.0,
                    "fragment_medium": 0.08,
                    "fragment_large": 0.02,
                }
            )
        return weights

    def _v6_operator_multipliers(
        self, state, arm_stats, has_elites, has_diverse, has_motifs
    ):
        base_weights = self._v5_operator_weights(
            state=state,
            has_elites=has_elites,
            has_diverse=has_diverse,
            has_motifs=has_motifs,
        )
        total_pulls = sum(
            float(stats.get("pulls", 0.0)) for stats in arm_stats.values()
        )
        total_pulls = max(total_pulls, 1.0)
        multipliers = {}
        for op_name, base in base_weights.items():
            if base <= 0:
                continue
            stats = arm_stats.setdefault(op_name, {"ema": 0.50, "pulls": 0.0})
            ema = float(stats.get("ema", 0.50))
            pulls = float(stats.get("pulls", 0.0))
            exploit = math.exp(self.args.v6_bandit_temperature * (ema - 0.50))
            explore = self.args.v6_ucb_weight * math.sqrt(
                math.log(total_pulls + 1.0) / (pulls + 1.0)
            )
            score = exploit + explore
            multipliers[op_name] = min(
                4.0,
                max(self.args.v6_min_operator_weight, score),
            )
        return multipliers

    def _update_v6_arm_stat(self, arm_stats, op_name, scores, best_before):
        stats = arm_stats.setdefault(op_name, {"ema": 0.50, "pulls": 0.0})
        old_ema = float(stats.get("ema", 0.50))
        alpha = min(1.0, max(0.0, self.args.v6_bandit_alpha))
        if scores:
            values = [float(score) for score in scores]
            top = sorted(values, reverse=True)
            topk_mean = float(np.mean(top[: min(10, len(top))]))
            best = float(top[0])
            nonzero_rate = sum(
                score > self._nonzero_threshold() for score in values
            ) / len(values)
            if np.isfinite(best_before):
                gain = max(0.0, best - float(best_before))
            else:
                gain = best
            gain = min(1.0, gain)
            reward = (
                self.args.v6_reward_topk_weight * topk_mean
                + self.args.v6_reward_best_weight * best
                + self.args.v6_reward_nonzero_weight * nonzero_rate
                + self.args.v6_reward_gain_weight * gain
            )
            reward = min(1.0, max(0.0, float(reward)))
            pulls = len(values)
        else:
            reward = 0.0
            pulls = 1
        stats["ema"] = (1.0 - alpha) * old_ema + alpha * reward
        stats["pulls"] = float(stats.get("pulls", 0.0)) + pulls

    @staticmethod
    def _v6_diag_operator_stats(arm_stats):
        out = {}
        for op_name, stats in arm_stats.items():
            out[f"{op_name}_ema"] = float(stats.get("ema", 0.0))
            out[f"{op_name}_pulls_k"] = float(stats.get("pulls", 0.0)) / 1000.0
        return out

    def _v7_operator_multipliers(
        self,
        state,
        arm_stats,
        has_elites,
        has_diverse,
        has_motifs,
        stagnant_calls,
    ):
        base_weights = self._v7_operator_weights(
            state=state,
            has_elites=has_elites,
            has_diverse=has_diverse,
            has_motifs=has_motifs,
            stagnant_calls=stagnant_calls,
        )
        total_pulls = sum(
            float(stats.get("pulls", 0.0)) for stats in arm_stats.values()
        )
        total_pulls = max(total_pulls, 1.0)
        multipliers = {}
        for op_name, base in base_weights.items():
            if base <= 0:
                continue
            stats = arm_stats.setdefault(op_name, {"ema": 0.50, "pulls": 0.0})
            ema = float(stats.get("ema", 0.50))
            pulls = float(stats.get("pulls", 0.0))
            exploit = math.exp(self.args.v7_bandit_temperature * (ema - 0.50))
            explore = self.args.v7_ucb_weight * math.sqrt(
                math.log(total_pulls + 1.0) / (pulls + 1.0)
            )
            score = exploit + explore
            multipliers[op_name] = min(
                4.0,
                max(self.args.v7_min_operator_weight, score),
            )
        return multipliers

    def _update_v7_arm_stat(
        self, arm_stats, op_name, scores, before_metrics, after_metrics
    ):
        stats = arm_stats.setdefault(op_name, {"ema": 0.50, "pulls": 0.0})
        old_ema = float(stats.get("ema", 0.50))
        alpha = min(1.0, max(0.0, self.args.v7_bandit_alpha))
        if scores:
            values = [float(score) for score in scores]
            nonzero_rate = sum(
                score > self._nonzero_threshold() for score in values
            ) / len(values)
            top10_threshold = self._buffer_score_threshold(top_n=10)
            if top10_threshold is None:
                top10_entry_rate = 0.0
            else:
                top10_entry_rate = sum(
                    score >= top10_threshold - 1e-12 for score in values
                ) / len(values)

            before_top10 = float(before_metrics.get("avg_top10", 0.0))
            after_top10 = float(after_metrics.get("avg_top10", 0.0))
            before_auc = float(before_metrics.get("auc_top10", 0.0))
            after_auc = float(after_metrics.get("auc_top10", 0.0))
            before_top1 = float(before_metrics.get("avg_top1", 0.0))
            after_top1 = float(after_metrics.get("avg_top1", 0.0))

            delta_top10 = min(1.0, max(0.0, after_top10 - before_top10) * 8.0)
            delta_auc = min(1.0, max(0.0, after_auc - before_auc) * 8.0)
            best_gain = min(1.0, max(0.0, after_top1 - before_top1) * 5.0)
            reward = (
                self.args.v7_reward_delta_top10_weight * delta_top10
                + self.args.v7_reward_delta_auc_weight * delta_auc
                + self.args.v7_reward_top10_entry_weight * top10_entry_rate
                + self.args.v7_reward_best_gain_weight * best_gain
                + self.args.v7_reward_nonzero_weight * nonzero_rate
            )
            reward = min(1.0, max(0.0, float(reward)))
            pulls = len(values)
        else:
            reward = 0.0
            pulls = 1
        stats["ema"] = (1.0 - alpha) * old_ema + alpha * reward
        stats["pulls"] = float(stats.get("pulls", 0.0)) + pulls

    def _buffer_score_threshold(self, top_n=10):
        if not self.oracle.mol_buffer:
            return None
        values = sorted(
            (float(score) for score, _ in self.oracle.mol_buffer.values()),
            reverse=True,
        )
        if not values:
            return None
        idx = min(max(1, top_n), len(values)) - 1
        return float(values[idx])

    @staticmethod
    def _v7_diag_operator_stats(arm_stats):
        out = {}
        for op_name, stats in arm_stats.items():
            out[f"{op_name}_ema"] = float(stats.get("ema", 0.0))
            out[f"{op_name}_pulls_k"] = float(stats.get("pulls", 0.0)) / 1000.0
        return out

    def _make_v8_proposal_groups(
        self,
        population,
        elites,
        diverse,
        motifs,
        min_size,
        max_size,
        state,
        operator_stats,
    ):
        fractions = sorted(
            parse_float_list(self.args.v5_remask_fractions, [0.06, 0.14, 0.28, 0.50])
        )
        tiny = fractions[0]
        small = fractions[min(1, len(fractions) - 1)]
        medium = fractions[min(2, len(fractions) - 1)]
        large = fractions[-1]
        target_n = max(
            self.args.candidate_batch_size,
            int(
                round(self.args.candidate_batch_size * self.args.v8_overgenerate_factor)
            ),
        )
        base_weights = self._v8_base_operator_weights(
            state=state,
            has_elites=bool(elites),
            has_diverse=bool(diverse),
            has_motifs=bool(motifs),
        )
        weighted_ops = [
            (op, base * (0.40 + operator_stats.get(op, 1.0)))
            for op, base in base_weights.items()
            if base > 0
        ]
        if not weighted_ops:
            weighted_ops = [("fragment_restart", 1.0)]

        exploit_temp = max(0.92, self.args.temperature_start - 0.14)
        explore_temp = max(self.args.temperature_start, 1.30)
        specs = {
            "elite_tiny": {
                "remask_fraction": tiny,
                "temperature_start": exploit_temp,
                "span_prob": max(0.40, self.args.span_prob - 0.16),
                "proposals": [],
            },
            "elite_small": {
                "remask_fraction": small,
                "temperature_start": exploit_temp,
                "span_prob": max(0.45, self.args.span_prob - 0.10),
                "proposals": [],
            },
            "elite_medium": {
                "remask_fraction": medium,
                "temperature_start": self.args.temperature_start,
                "span_prob": self.args.span_prob,
                "proposals": [],
            },
            "diverse_medium": {
                "remask_fraction": medium,
                "temperature_start": explore_temp,
                "span_prob": min(1.0, self.args.span_prob + 0.05),
                "proposals": [],
            },
            "fragment_restart": {
                "remask_fraction": medium,
                "temperature_start": explore_temp,
                "span_prob": min(1.0, self.args.span_prob + 0.08),
                "proposals": [],
            },
            "graph_swap": {
                "remask_fraction": small,
                "temperature_start": self.args.temperature_start,
                "span_prob": min(1.0, self.args.span_prob + 0.06),
                "proposals": [],
            },
            "graph_shrink": {
                "remask_fraction": small,
                "temperature_start": self.args.temperature_start,
                "span_prob": min(1.0, self.args.span_prob + 0.04),
                "proposals": [],
            },
            "graph_expand": {
                "remask_fraction": medium,
                "temperature_start": explore_temp,
                "span_prob": min(1.0, self.args.span_prob + 0.08),
                "proposals": [],
            },
            "rescue_large": {
                "remask_fraction": large,
                "temperature_start": self.args.v5_rescue_temperature,
                "span_prob": min(1.0, self.args.span_prob + 0.22),
                "proposals": [],
            },
            "motif_restart": {
                "motif_seeded": True,
                "temperature_start": explore_temp,
                "n_mol": 0,
            },
        }

        attempts = 0
        total = 0
        while total < target_n and attempts < target_n * 220:
            attempts += 1
            op = self._weighted_choice(weighted_ops)
            if op == "motif_restart":
                specs[op]["n_mol"] += 1
                total += 1
                continue

            parent_smiles = None
            parent_score = None
            seed = None
            if op.startswith("elite") or op == "rescue_large":
                source = elites if elites else diverse
                if source:
                    top_n = max(1, min(len(source), self.args.elite_size))
                    parent_score, parent_smiles = random.choice(source[:top_n])
                    seed = parent_smiles
            elif op == "diverse_medium" and diverse:
                parent_score, parent_smiles = random.choice(diverse)
                seed = parent_smiles
            elif op in {"graph_swap", "graph_shrink", "graph_expand"}:
                source = elites if elites else diverse
                if source:
                    top_n = max(1, min(len(source), self.args.elite_size))
                    parent_score, parent_smiles = random.choice(source[:top_n])
                    direction = op.removeprefix("graph_")
                    seed = self._make_v8_graph_fragment_edit(
                        parent_smiles=parent_smiles,
                        population=population,
                        direction=direction,
                        min_size=min_size,
                        max_size=max_size,
                    )
            else:
                seed = self._make_fragment_seed_v2(
                    population,
                    min_size,
                    max_size,
                    prefer_top=state in {"exploit", "sparse_exploit", "refine"},
                )

            can = canonical_smiles(seed) if seed else None
            if can is None or not tokenizable(can, self.tk, self.args.max_len):
                continue
            specs[op]["proposals"].append(
                {
                    "seed": can,
                    "parent": parent_smiles,
                    "parent_score": None
                    if parent_score is None
                    else float(parent_score),
                    "motif": None,
                }
            )
            total += 1

        out = {}
        for op, spec in specs.items():
            if spec.get("motif_seeded"):
                if spec["n_mol"] > 0:
                    out[op] = spec
            elif spec["proposals"]:
                out[op] = spec
        return out

    @staticmethod
    def _v8_base_operator_weights(state, has_elites, has_diverse, has_motifs):
        weights = {
            "elite_tiny": 0.14 if has_elites else 0.0,
            "elite_small": 0.16 if has_elites else 0.0,
            "elite_medium": 0.10 if has_elites else 0.0,
            "diverse_medium": 0.10 if has_diverse else 0.0,
            "motif_restart": 0.14 if has_motifs else 0.0,
            "fragment_restart": 0.12,
            "graph_swap": 0.10 if has_elites else 0.0,
            "graph_shrink": 0.05 if has_elites else 0.0,
            "graph_expand": 0.05 if has_elites else 0.0,
            "rescue_large": 0.04 if has_elites else 0.0,
        }
        if state == "warmup":
            weights.update(
                {
                    "elite_tiny": 0.05 if has_elites else 0.0,
                    "elite_small": 0.08 if has_elites else 0.0,
                    "elite_medium": 0.07 if has_elites else 0.0,
                    "diverse_medium": 0.12 if has_diverse else 0.0,
                    "motif_restart": 0.26 if has_motifs else 0.0,
                    "fragment_restart": 0.24,
                    "graph_swap": 0.08 if has_elites else 0.0,
                    "graph_shrink": 0.04 if has_elites else 0.0,
                    "graph_expand": 0.06 if has_elites else 0.0,
                    "rescue_large": 0.0,
                }
            )
        elif state in {"exploit", "sparse_exploit"}:
            weights.update(
                {
                    "elite_tiny": 0.30 if has_elites else 0.0,
                    "elite_small": 0.24 if has_elites else 0.0,
                    "elite_medium": 0.10 if has_elites else 0.0,
                    "diverse_medium": 0.06 if has_diverse else 0.0,
                    "motif_restart": 0.06 if has_motifs else 0.0,
                    "fragment_restart": 0.04,
                    "graph_swap": 0.10 if has_elites else 0.0,
                    "graph_shrink": 0.04 if has_elites else 0.0,
                    "graph_expand": 0.04 if has_elites else 0.0,
                    "rescue_large": 0.02 if has_elites else 0.0,
                }
            )
        elif state == "refine":
            weights.update(
                {
                    "elite_tiny": 0.38 if has_elites else 0.0,
                    "elite_small": 0.25 if has_elites else 0.0,
                    "elite_medium": 0.08 if has_elites else 0.0,
                    "diverse_medium": 0.05 if has_diverse else 0.0,
                    "motif_restart": 0.04 if has_motifs else 0.0,
                    "fragment_restart": 0.03,
                    "graph_swap": 0.08 if has_elites else 0.0,
                    "graph_shrink": 0.04 if has_elites else 0.0,
                    "graph_expand": 0.03 if has_elites else 0.0,
                    "rescue_large": 0.02 if has_elites else 0.0,
                }
            )
        elif state in {"explore", "sparse"}:
            weights.update(
                {
                    "elite_tiny": 0.04 if has_elites else 0.0,
                    "elite_small": 0.06 if has_elites else 0.0,
                    "elite_medium": 0.08 if has_elites else 0.0,
                    "diverse_medium": 0.14 if has_diverse else 0.0,
                    "motif_restart": 0.20 if has_motifs else 0.0,
                    "fragment_restart": 0.18,
                    "graph_swap": 0.12 if has_elites else 0.0,
                    "graph_shrink": 0.06 if has_elites else 0.0,
                    "graph_expand": 0.08 if has_elites else 0.0,
                    "rescue_large": 0.04 if has_elites else 0.0,
                }
            )
        elif state == "rescue":
            weights.update(
                {
                    "elite_tiny": 0.04 if has_elites else 0.0,
                    "elite_small": 0.06 if has_elites else 0.0,
                    "elite_medium": 0.05 if has_elites else 0.0,
                    "diverse_medium": 0.13 if has_diverse else 0.0,
                    "motif_restart": 0.16 if has_motifs else 0.0,
                    "fragment_restart": 0.14,
                    "graph_swap": 0.14 if has_elites else 0.0,
                    "graph_shrink": 0.09 if has_elites else 0.0,
                    "graph_expand": 0.09 if has_elites else 0.0,
                    "rescue_large": 0.10 if has_elites else 0.0,
                }
            )
        return weights

    def _v8_operator_multipliers(
        self,
        state,
        arm_stats,
        has_elites,
        has_diverse,
        has_motifs,
    ):
        base_weights = self._v8_base_operator_weights(
            state=state,
            has_elites=has_elites,
            has_diverse=has_diverse,
            has_motifs=has_motifs,
        )
        total_pulls = max(
            1.0,
            sum(float(stats.get("pulls", 0.0)) for stats in arm_stats.values()),
        )
        multipliers = {}
        for op_name, base in base_weights.items():
            if base <= 0:
                continue
            stats = arm_stats.setdefault(
                op_name,
                {"ema": 0.50, "pulls": 0.0, "positive": 0.0},
            )
            ema = float(stats.get("ema", 0.50))
            pulls = float(stats.get("pulls", 0.0))
            exploit = math.exp(self.args.v8_bandit_temperature * (ema - 0.50))
            explore = self.args.v8_ucb_weight * math.sqrt(
                math.log(total_pulls + 1.0) / (pulls + 1.0)
            )
            multipliers[op_name] = min(
                4.0,
                max(self.args.v8_min_operator_weight, exploit + explore),
            )
        return multipliers

    def _make_v8_graph_fragment_edit(
        self,
        parent_smiles,
        population,
        direction,
        min_size,
        max_size,
    ):
        parent = canonical_smiles(parent_smiles)
        if parent is None:
            return None
        parent_atoms = atom_count(parent)
        parent_fragment_set = set()
        try:
            for _ in range(max(1, self.args.v8_graph_cut_rounds)):
                parent_fragment_set.update(local_genmol_cut(parent))
            parent_fragments = list(parent_fragment_set)
        except Exception:
            return None
        parent_fragments = [
            frag
            for frag in parent_fragments
            if fragment_heavy_atom_count(frag) > 0
            and any(
                atom.GetAtomicNum() == 0 for atom in Chem.MolFromSmiles(frag).GetAtoms()
            )
        ]
        if not parent_fragments:
            return None
        parent_fragments.sort(key=fragment_heavy_atom_count, reverse=True)
        core_pool = parent_fragments[: max(1, (len(parent_fragments) + 1) // 2)]

        replacement_rows = []
        for score, frag in population:
            mol = Chem.MolFromSmiles(frag)
            size = fragment_heavy_atom_count(frag)
            if mol is None or size <= 0:
                continue
            if not any(atom.GetAtomicNum() == 0 for atom in mol.GetAtoms()):
                continue
            replacement_rows.append((float(score), size, frag))
        if direction == "shrink":
            replacement_rows.sort(key=lambda row: (row[1], -row[0]))
        elif direction == "expand":
            replacement_rows.sort(key=lambda row: (-row[1], -row[0]))
        else:
            replacement_rows.sort(key=lambda row: row[0], reverse=True)
        pool_size = max(20, min(len(replacement_rows), self.args.population_size // 2))
        replacements = [frag for _, _, frag in replacement_rows[:pool_size]]
        if not replacements:
            return None

        for _ in range(self.args.v8_graph_edit_attempts):
            core = random.choice(core_pool)
            replacement = random.choice(replacements)
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

    @staticmethod
    def _v8_top_mean(scores, top_n=10):
        if not scores:
            return 0.0
        values = sorted((float(score) for score in scores), reverse=True)
        return float(np.mean(values[: min(top_n, len(values))]))

    def _v8_transition_reward(
        self,
        score,
        parent_score,
        before_scores,
        before_top10,
        after_top10,
        before_threshold,
    ):
        score = float(score)
        if before_scores:
            percentile = sum(float(old) <= score for old in before_scores) / len(
                before_scores
            )
        else:
            percentile = float(np.clip(score, 0.0, 1.0))
        if parent_score is None:
            delta = None
            delta_signal = percentile
        else:
            delta = score - float(parent_score)
            scale = max(1e-6, self.args.v8_delta_scale)
            delta_signal = 0.5 + 0.5 * math.tanh(delta / scale)
        frontier_gain = max(0.0, float(after_top10) - float(before_top10))
        frontier_signal = min(
            1.0,
            frontier_gain / max(1e-6, self.args.v8_frontier_gain_scale),
        )
        if len(before_scores) < 10 or before_threshold is None:
            entered_top10 = float(percentile)
        else:
            entered_top10 = float(score >= float(before_threshold) - 1e-12)
        reward = (
            self.args.v8_reward_delta_weight * delta_signal
            + self.args.v8_reward_frontier_weight * frontier_signal
            + self.args.v8_reward_top10_entry_weight * entered_top10
            + self.args.v8_reward_percentile_weight * percentile
        )
        weight_sum = (
            self.args.v8_reward_delta_weight
            + self.args.v8_reward_frontier_weight
            + self.args.v8_reward_top10_entry_weight
            + self.args.v8_reward_percentile_weight
        )
        reward = reward / max(weight_sum, 1e-8)
        reward = float(np.clip(reward, 0.0, 1.0))
        return reward, {
            "delta": "" if delta is None else float(delta),
            "delta_signal": float(delta_signal),
            "frontier_gain": float(frontier_gain),
            "frontier_signal": float(frontier_signal),
            "entered_top10": float(entered_top10),
            "percentile": float(percentile),
        }

    def _update_v8_arm_stats(self, arm_stats, op_name, rewards):
        stats = arm_stats.setdefault(
            op_name,
            {"ema": 0.50, "pulls": 0.0, "positive": 0.0},
        )
        alpha = float(np.clip(self.args.v8_bandit_alpha, 0.0, 1.0))
        for reward in rewards:
            stats["ema"] = (1.0 - alpha) * float(stats["ema"]) + alpha * float(reward)
            stats["pulls"] = float(stats["pulls"]) + 1.0
            if reward >= self.args.v8_positive_reward_threshold:
                stats["positive"] = float(stats["positive"]) + 1.0
        if not rewards:
            stats["ema"] = (1.0 - alpha) * float(stats["ema"])
            stats["pulls"] = float(stats["pulls"]) + 1.0

    def _update_v8_archives(
        self,
        population,
        motifs,
        elites,
        diverse,
        parent_smiles,
        child_smiles,
        child_score,
        transition_reward,
        frozen_motif=None,
    ):
        elites.append((float(child_score), child_smiles))
        elites.sort(key=lambda item: item[0], reverse=True)
        del elites[self.args.elite_size :]
        self._update_v5_diverse_archive(diverse, child_smiles, child_score)
        self._update_v8_fragment_credit(
            population=population,
            parent_smiles=parent_smiles,
            child_smiles=child_smiles,
            transition_reward=transition_reward,
        )
        self._update_v8_motif_credit(
            motifs=motifs,
            parent_smiles=parent_smiles,
            child_smiles=child_smiles,
            transition_reward=transition_reward,
            frozen_motif=frozen_motif,
        )

    def _update_v8_fragment_credit(
        self,
        population,
        parent_smiles,
        child_smiles,
        transition_reward,
    ):
        try:
            child_fragments = local_genmol_cut(child_smiles)
        except Exception:
            return
        parent_mol = Chem.MolFromSmiles(parent_smiles) if parent_smiles else None
        old_scores = {frag: float(score) for score, frag in population}
        alpha = float(np.clip(self.args.v8_credit_alpha, 0.0, 1.0))
        updated = False
        for frag in child_fragments:
            clean = clean_dummy_fragment(frag)
            clean_mol = Chem.MolFromSmiles(clean) if clean else None
            if clean_mol is None:
                continue
            if parent_mol is not None and parent_mol.HasSubstructMatch(clean_mol):
                continue
            old = old_scores.get(frag, 0.50)
            old_scores[frag] = (1.0 - alpha) * old + alpha * float(transition_reward)
            updated = True
        if not updated:
            return
        items = sorted(old_scores.items(), key=lambda item: item[1], reverse=True)
        population[:] = [
            (score, frag) for frag, score in items[: self.args.population_size]
        ]

    def _update_v8_motif_credit(
        self,
        motifs,
        parent_smiles,
        child_smiles,
        transition_reward,
        frozen_motif=None,
    ):
        child_mol = Chem.MolFromSmiles(child_smiles)
        if child_mol is None:
            return
        try:
            child_map = extract_motifs(
                child_mol,
                min_atoms=self.args.motif_min_atoms,
                max_atoms=self.args.motif_max_atoms,
            )
        except Exception:
            child_map = {}
        parent_keys = set()
        if parent_smiles:
            parent_mol = Chem.MolFromSmiles(parent_smiles)
            if parent_mol is not None:
                try:
                    parent_keys = set(
                        extract_motifs(
                            parent_mol,
                            min_atoms=self.args.motif_min_atoms,
                            max_atoms=self.args.motif_max_atoms,
                        )
                    )
                except Exception:
                    parent_keys = set()
        credited = set(child_map) - parent_keys
        if frozen_motif:
            credited.add(frozen_motif)
        if not credited:
            return

        motif_map = {row["motif"]: row for row in motifs}
        alpha = float(np.clip(self.args.v8_credit_alpha, 0.0, 1.0))
        for motif in credited:
            if not tokenizable(motif, self.tk, self.args.max_len - 4):
                continue
            row = motif_map.get(motif)
            if row is None:
                row = {
                    "motif": motif,
                    "score": float(transition_reward),
                    "support": 0.0,
                    "quality_rate": float(transition_reward),
                    "enrichment": float(transition_reward),
                    "mean_qed": 0.0,
                    "mean_sa": 0.0,
                    "motif_type": "pmo_v8_delta_credit",
                }
                motifs.append(row)
                motif_map[motif] = row
            else:
                old = float(row.get("score", 0.50))
                row["score"] = (1.0 - alpha) * old + alpha * float(transition_reward)
            row["support"] = float(row.get("support", 0.0)) + 1.0
            row["quality_rate"] = float(row["score"])
            row["enrichment"] = float(row["score"])
        motifs.sort(
            key=lambda row: (
                float(row.get("score", 0.0)),
                float(row.get("support", 0.0)),
            ),
            reverse=True,
        )
        del motifs[self.args.v5_motif_pool_size :]

    @staticmethod
    def _append_v8_transition(
        path,
        oracle_name,
        state,
        operator,
        parent_smiles,
        seed_smiles,
        child_smiles,
        parent_score,
        child_score,
        reward,
        reward_parts,
    ):
        os.makedirs(os.path.dirname(path), exist_ok=True)
        fields = [
            "oracle",
            "state",
            "operator",
            "parent_smiles",
            "seed_smiles",
            "child_smiles",
            "parent_score",
            "child_score",
            "delta",
            "reward",
            "delta_signal",
            "frontier_gain",
            "frontier_signal",
            "entered_top10",
            "percentile",
            "parent_atoms",
            "seed_atoms",
            "child_atoms",
        ]
        exists = os.path.exists(path)
        with open(path, "a", newline="") as f:
            writer = csv.DictWriter(f, fieldnames=fields)
            if not exists:
                writer.writeheader()
            writer.writerow(
                {
                    "oracle": oracle_name,
                    "state": state,
                    "operator": operator,
                    "parent_smiles": parent_smiles or "",
                    "seed_smiles": seed_smiles or "",
                    "child_smiles": child_smiles,
                    "parent_score": "" if parent_score is None else float(parent_score),
                    "child_score": float(child_score),
                    "delta": reward_parts["delta"],
                    "reward": float(reward),
                    "delta_signal": reward_parts["delta_signal"],
                    "frontier_gain": reward_parts["frontier_gain"],
                    "frontier_signal": reward_parts["frontier_signal"],
                    "entered_top10": reward_parts["entered_top10"],
                    "percentile": reward_parts["percentile"],
                    "parent_atoms": atom_count(parent_smiles) if parent_smiles else "",
                    "seed_atoms": atom_count(seed_smiles) if seed_smiles else "",
                    "child_atoms": atom_count(child_smiles),
                }
            )

    @staticmethod
    def _v8_diag_operator_stats(context_stats, current_state):
        out = {}
        for op_name, stats in context_stats.get(current_state, {}).items():
            pulls = float(stats.get("pulls", 0.0))
            out[f"{current_state}_{op_name}_ema"] = float(stats.get("ema", 0.0))
            out[f"{current_state}_{op_name}_pulls"] = pulls
            out[f"{current_state}_{op_name}_positive_rate"] = (
                float(stats.get("positive", 0.0)) / pulls if pulls > 0 else 0.0
            )
        return out

    def _update_v5_archives(self, population, motifs, elites, diverse, smiles, score):
        elites.append((float(score), smiles))
        elites.sort(key=lambda item: item[0], reverse=True)
        del elites[self.args.elite_size :]

        self._update_v5_diverse_archive(diverse, smiles, score)
        self._update_fragment_population_v2(population, smiles, score)
        self._update_v5_motif_archive(motifs, smiles, score)

    def _update_v5_diverse_archive(self, diverse, smiles, score):
        old = (
            getattr(self.args, "v4_diverse_size", None),
            getattr(self.args, "v4_score_elite_fraction", None),
            getattr(self.args, "v4_diversity_weight", None),
        )
        self.args.v4_diverse_size = self.args.v5_diverse_size
        self.args.v4_score_elite_fraction = self.args.v5_score_elite_fraction
        self.args.v4_diversity_weight = self.args.v5_diversity_weight
        try:
            self._update_v4_diverse_archive(diverse, smiles, score)
        finally:
            if old[0] is not None:
                self.args.v4_diverse_size = old[0]
            if old[1] is not None:
                self.args.v4_score_elite_fraction = old[1]
            if old[2] is not None:
                self.args.v4_diversity_weight = old[2]

    def _update_v5_motif_archive(self, motifs, smiles, score):
        old_pool_size = (
            self.args.v4_motif_pool_size
            if hasattr(self.args, "v4_motif_pool_size")
            else None
        )
        self.args.v4_motif_pool_size = self.args.v5_motif_pool_size
        try:
            self._update_v4_motif_archive(motifs, smiles, score)
        finally:
            if old_pool_size is not None:
                self.args.v4_motif_pool_size = old_pool_size

    def _make_v4_seed_groups(
        self,
        oracle_name,
        population,
        elites,
        diverse,
        motifs,
        min_size,
        max_size,
        rescue,
        zero_rescue,
        operator_stats,
    ):
        fractions = sorted(
            parse_float_list(self.args.v4_remask_fractions, [0.08, 0.18, 0.32, 0.55])
        )
        small = fractions[0]
        medium = fractions[min(1, len(fractions) - 1)]
        large = fractions[-2] if len(fractions) > 2 else fractions[-1]
        rescue_large = fractions[-1]
        target_n = max(
            self.args.candidate_batch_size,
            int(
                round(self.args.candidate_batch_size * self.args.v4_overgenerate_factor)
            ),
        )

        base_weights = self._v4_base_operator_weights(
            oracle_name=oracle_name,
            has_elites=bool(elites),
            has_diverse=bool(diverse),
            has_motifs=bool(motifs),
            rescue=rescue,
            zero_rescue=zero_rescue,
        )
        weighted_ops = []
        for op, base in base_weights.items():
            if base <= 0:
                continue
            weighted_ops.append((op, base * (0.45 + operator_stats.get(op, 1.0))))
        if not weighted_ops:
            weighted_ops = [("fragment_medium", 1.0)]

        specs = {
            "elite_small": {
                "remask_fraction": small,
                "temperature_start": max(0.95, self.args.temperature_start - 0.15),
                "span_prob": max(0.45, self.args.span_prob - 0.10),
                "seeds": [],
            },
            "elite_medium": {
                "remask_fraction": medium,
                "temperature_start": self.args.temperature_start,
                "span_prob": self.args.span_prob,
                "seeds": [],
            },
            "diverse_medium": {
                "remask_fraction": medium,
                "temperature_start": max(self.args.temperature_start, 1.20),
                "span_prob": self.args.span_prob,
                "seeds": [],
            },
            "fragment_medium": {
                "remask_fraction": medium,
                "temperature_start": max(self.args.temperature_start, 1.25),
                "span_prob": min(1.0, self.args.span_prob + 0.05),
                "seeds": [],
            },
            "fragment_large": {
                "remask_fraction": large,
                "temperature_start": max(self.args.temperature_start, 1.45),
                "span_prob": min(1.0, self.args.span_prob + 0.15),
                "seeds": [],
            },
            "rescue_large": {
                "remask_fraction": rescue_large,
                "temperature_start": self.args.v4_rescue_temperature,
                "span_prob": min(1.0, self.args.span_prob + 0.25),
                "seeds": [],
            },
            "motif_seeded": {
                "motif_seeded": True,
                "temperature_start": max(self.args.temperature_start, 1.25),
                "n_mol": 0,
            },
        }

        attempts = 0
        total = 0
        while total < target_n and attempts < target_n * 180:
            attempts += 1
            op = self._weighted_choice(weighted_ops)
            if op == "motif_seeded":
                specs[op]["n_mol"] += 1
                total += 1
                continue
            if op.startswith("elite") and elites:
                top_n = max(1, min(len(elites), self.args.elite_size))
                smi = random.choice(elites[:top_n])[1]
            elif op == "diverse_medium" and diverse:
                smi = random.choice(diverse)[1]
            else:
                smi = self._make_fragment_seed_v2(
                    population,
                    min_size,
                    max_size,
                    prefer_top=(
                        op in {"fragment_medium", "fragment_large"} and not rescue
                    ),
                )
            can = canonical_smiles(smi) if smi else None
            if can is None:
                continue
            atoms = atom_count(can)
            if atoms < min_size or atoms > max_size:
                continue
            if not tokenizable(can, self.tk, self.args.max_len):
                continue
            specs[op]["seeds"].append(can)
            total += 1

        out = {}
        for op, spec in specs.items():
            if spec.get("motif_seeded"):
                if spec["n_mol"] > 0:
                    out[op] = spec
            elif spec["seeds"]:
                out[op] = spec
        return out

    @staticmethod
    def _v4_base_operator_weights(
        oracle_name,
        has_elites,
        has_diverse,
        has_motifs,
        rescue,
        zero_rescue,
    ):
        is_similarity_like = (
            "similarity" in oracle_name
            or "rediscovery" in oracle_name
            or oracle_name in {"scaffold_hop", "deco_hop"}
        )
        is_isomer = oracle_name.startswith("isomers_")
        is_classifier = oracle_name in {"drd2", "gsk3b", "jnk3"}

        weights = {
            "elite_small": 0.26 if has_elites else 0.0,
            "elite_medium": 0.18 if has_elites else 0.0,
            "diverse_medium": 0.12 if has_diverse else 0.0,
            "motif_seeded": 0.22 if has_motifs else 0.0,
            "fragment_medium": 0.16,
            "fragment_large": 0.06,
            "rescue_large": 0.0,
        }
        if is_similarity_like:
            weights.update(
                {
                    "elite_small": 0.34 if has_elites else 0.0,
                    "elite_medium": 0.20 if has_elites else 0.0,
                    "diverse_medium": 0.08 if has_diverse else 0.0,
                    "motif_seeded": 0.20 if has_motifs else 0.0,
                    "fragment_medium": 0.14,
                    "fragment_large": 0.04,
                }
            )
        elif is_isomer:
            weights.update(
                {
                    "elite_small": 0.18 if has_elites else 0.0,
                    "elite_medium": 0.12 if has_elites else 0.0,
                    "diverse_medium": 0.14 if has_diverse else 0.0,
                    "motif_seeded": 0.12 if has_motifs else 0.0,
                    "fragment_medium": 0.32,
                    "fragment_large": 0.12,
                }
            )
        elif is_classifier:
            weights.update(
                {
                    "elite_small": 0.20 if has_elites else 0.0,
                    "elite_medium": 0.18 if has_elites else 0.0,
                    "diverse_medium": 0.14 if has_diverse else 0.0,
                    "motif_seeded": 0.26 if has_motifs else 0.0,
                    "fragment_medium": 0.16,
                    "fragment_large": 0.06,
                }
            )

        if rescue:
            weights = {
                "elite_small": 0.12
                if has_elites and not zero_rescue
                else 0.04
                if has_elites
                else 0.0,
                "elite_medium": 0.10 if has_elites and not zero_rescue else 0.0,
                "diverse_medium": 0.14 if has_diverse else 0.0,
                "motif_seeded": 0.28 if has_motifs else 0.0,
                "fragment_medium": 0.20,
                "fragment_large": 0.12,
                "rescue_large": 0.18,
            }
        return weights

    def _update_v4_archives(self, population, motifs, elites, diverse, smiles, score):
        elites.append((float(score), smiles))
        elites.sort(key=lambda item: item[0], reverse=True)
        del elites[self.args.elite_size :]

        self._update_v4_diverse_archive(diverse, smiles, score)
        self._update_fragment_population_v2(population, smiles, score)
        self._update_v4_motif_archive(motifs, smiles, score)

    def _update_v4_diverse_archive(self, diverse, smiles, score):
        best = {}
        for old_score, old_smi in diverse:
            best[old_smi] = max(float(old_score), best.get(old_smi, -float("inf")))
        best[smiles] = max(float(score), best.get(smiles, -float("inf")))
        items = sorted(best.items(), key=lambda item: item[1], reverse=True)
        if len(items) <= self.args.v4_diverse_size:
            diverse[:] = [(score, smi) for smi, score in items]
            return

        elite_keep = max(
            2, int(self.args.v4_diverse_size * self.args.v4_score_elite_fraction)
        )
        selected = items[:elite_keep]
        remaining = items[elite_keep:]
        selected_fps = [
            fp
            for smi, _ in selected
            for fp in [mol_fp_from_smiles(smi)]
            if fp is not None
        ]
        min_score = min(score for _, score in items)
        max_score = max(score for _, score in items)
        denom = max(max_score - min_score, 1e-8)
        while len(selected) < self.args.v4_diverse_size and remaining:
            best_idx = 0
            best_value = -float("inf")
            for idx, (smi, item_score) in enumerate(remaining):
                fp = mol_fp_from_smiles(smi)
                novelty = 1.0 - max_tanimoto(fp, selected_fps)
                score_norm = (item_score - min_score) / denom
                value = score_norm + self.args.v4_diversity_weight * novelty
                if value > best_value:
                    best_value = value
                    best_idx = idx
            smi, item_score = remaining.pop(best_idx)
            selected.append((smi, item_score))
            fp = mol_fp_from_smiles(smi)
            if fp is not None:
                selected_fps.append(fp)
        selected.sort(key=lambda item: item[1], reverse=True)
        diverse[:] = [(score, smi) for smi, score in selected]

    def _update_v4_motif_archive(self, motifs, smiles, score):
        mol = Chem.MolFromSmiles(smiles)
        if mol is None:
            return
        motif_map = {row["motif"]: row for row in motifs}
        try:
            new_motifs = extract_motifs(
                mol,
                min_atoms=self.args.motif_min_atoms,
                max_atoms=self.args.motif_max_atoms,
            )
        except Exception:
            return
        for motif, kinds in new_motifs.items():
            if not tokenizable(motif, self.tk, self.args.max_len - 4):
                continue
            row = motif_map.get(motif)
            if row is None:
                row = {
                    "motif": motif,
                    "score": float(score),
                    "support": 0.0,
                    "quality_rate": float(score),
                    "enrichment": float(score),
                    "mean_qed": 0.0,
                    "mean_sa": 0.0,
                    "motif_type": "pmo_v4_dynamic:" + ";".join(sorted(kinds)),
                }
                motifs.append(row)
                motif_map[motif] = row
            row["score"] = max(float(row.get("score", 0.0)), float(score))
            row["support"] = float(row.get("support", 0.0)) + 1.0
            row["quality_rate"] = max(float(row.get("quality_rate", 0.0)), float(score))
            row["enrichment"] = max(float(row.get("enrichment", 0.0)), float(score))
        motifs.sort(
            key=lambda row: (float(row["score"]), float(row.get("support", 0.0))),
            reverse=True,
        )
        del motifs[self.args.v4_motif_pool_size :]

    def _make_v3_seed_groups(
        self,
        population,
        elites,
        min_size,
        max_size,
        stagnant_calls,
        zero_rescue,
        operator_stats,
    ):
        fractions = sorted(
            parse_float_list(self.args.v3_remask_fractions, [0.10, 0.25, 0.45, 0.65])
        )
        small = fractions[0]
        medium = fractions[min(1, len(fractions) - 1)]
        large = fractions[-2] if len(fractions) > 2 else fractions[-1]
        rescue_large = fractions[-1]
        rescue = (
            zero_rescue or stagnant_calls >= self.args.v3_stagnation_rescue_patience
        )

        base_weights = {
            "elite_small": 0.35 if elites else 0.0,
            "elite_medium": 0.20 if elites else 0.0,
            "fragment_small": 0.10,
            "fragment_medium": 0.25,
            "fragment_large": 0.10,
            "rescue_restart": 0.0,
        }
        if rescue:
            base_weights = {
                "elite_small": 0.20 if elites else 0.0,
                "elite_medium": 0.10 if elites else 0.0,
                "fragment_small": 0.10 if not zero_rescue else 0.05,
                "fragment_medium": 0.15,
                "fragment_large": 0.25,
                "rescue_restart": self.args.v3_restart_prob,
            }

        weighted_ops = []
        for op, base in base_weights.items():
            if base <= 0:
                continue
            weighted_ops.append((op, base * (0.5 + operator_stats.get(op, 1.0))))
        if not weighted_ops:
            weighted_ops = [("fragment_medium", 1.0)]

        specs = {
            "elite_small": {
                "remask_fraction": small,
                "temperature_start": self.args.temperature_start,
                "span_prob": self.args.span_prob,
                "seeds": [],
            },
            "elite_medium": {
                "remask_fraction": medium,
                "temperature_start": max(self.args.temperature_start, 1.2),
                "span_prob": self.args.span_prob,
                "seeds": [],
            },
            "fragment_small": {
                "remask_fraction": small,
                "temperature_start": self.args.temperature_start,
                "span_prob": self.args.span_prob,
                "seeds": [],
            },
            "fragment_medium": {
                "remask_fraction": medium,
                "temperature_start": max(self.args.temperature_start, 1.2),
                "span_prob": self.args.span_prob,
                "seeds": [],
            },
            "fragment_large": {
                "remask_fraction": large,
                "temperature_start": max(
                    self.args.temperature_start, self.args.v3_rescue_temperature
                ),
                "span_prob": min(1.0, self.args.span_prob + 0.15),
                "seeds": [],
            },
            "rescue_restart": {
                "remask_fraction": rescue_large,
                "temperature_start": self.args.v3_rescue_temperature,
                "span_prob": min(1.0, self.args.span_prob + 0.25),
                "seeds": [],
            },
        }

        attempts = 0
        while (
            sum(len(spec["seeds"]) for spec in specs.values())
            < self.args.candidate_batch_size
            and attempts < self.args.candidate_batch_size * 160
        ):
            attempts += 1
            op = self._weighted_choice(weighted_ops)
            if op.startswith("elite") and elites:
                top_n = max(1, min(len(elites), self.args.elite_size))
                smi = random.choice(elites[:top_n])[1]
            else:
                smi = self._make_fragment_seed_v2(
                    population,
                    min_size,
                    max_size,
                    prefer_top=(op == "fragment_small" and not rescue),
                )
            can = canonical_smiles(smi) if smi else None
            if can is None:
                continue
            atoms = atom_count(can)
            if atoms < min_size or atoms > max_size:
                continue
            if not tokenizable(can, self.tk, self.args.max_len):
                continue
            specs[op]["seeds"].append(can)

        return {op: spec for op, spec in specs.items() if spec["seeds"]}

    def _make_v2_seed_groups(
        self,
        population,
        elites,
        min_size,
        max_size,
        stagnant_calls,
        operator_stats,
    ):
        fractions = sorted(
            parse_float_list(self.args.v2_remask_fractions, [0.15, 0.30, 0.50])
        )
        small = fractions[0]
        medium = fractions[len(fractions) // 2]
        large = fractions[-1]
        rescue = stagnant_calls >= self.args.v2_rescue_patience

        base_weights = {
            "elite_small": 0.30 if elites else 0.0,
            "elite_medium": 0.20 if elites else 0.0,
            "fragment_small": 0.15 if rescue else 0.05,
            "fragment_medium": 0.30,
            "fragment_large": self.args.v2_restart_prob if rescue else 0.15,
        }
        if rescue:
            base_weights["elite_small"] = 0.40 if elites else 0.0
            base_weights["elite_medium"] = 0.10 if elites else 0.0
            base_weights["fragment_small"] = 0.35
            base_weights["fragment_medium"] = 0.10
            base_weights["fragment_large"] = max(0.15, self.args.v2_restart_prob)

        weighted_ops = []
        for op, base in base_weights.items():
            if base <= 0:
                continue
            weighted_ops.append((op, base * (0.5 + operator_stats.get(op, 1.0))))
        if not weighted_ops:
            weighted_ops = [("fragment_medium", 1.0)]

        specs = {
            "elite_small": {
                "remask_fraction": small,
                "temperature_start": 1.0,
                "seeds": [],
            },
            "elite_medium": {
                "remask_fraction": medium,
                "temperature_start": 1.2,
                "seeds": [],
            },
            "fragment_small": {
                "remask_fraction": small,
                "temperature_start": 1.0,
                "seeds": [],
            },
            "fragment_medium": {
                "remask_fraction": medium,
                "temperature_start": 1.2,
                "seeds": [],
            },
            "fragment_large": {
                "remask_fraction": large,
                "temperature_start": 1.5,
                "seeds": [],
            },
        }

        attempts = 0
        while (
            sum(len(spec["seeds"]) for spec in specs.values())
            < self.args.candidate_batch_size
            and attempts < self.args.candidate_batch_size * 120
        ):
            attempts += 1
            op = self._weighted_choice(weighted_ops)
            if op.startswith("elite") and elites:
                smi = random.choice(
                    elites[: max(1, min(len(elites), self.args.elite_size))]
                )[1]
            else:
                smi = self._make_fragment_seed_v2(
                    population,
                    min_size,
                    max_size,
                    prefer_top=rescue or op == "fragment_small",
                )
            can = canonical_smiles(smi) if smi else None
            if can is None:
                continue
            atoms = atom_count(can)
            if atoms < min_size or atoms > max_size:
                continue
            if not tokenizable(can, self.tk, self.args.max_len):
                continue
            specs[op]["seeds"].append(can)

        return {op: spec for op, spec in specs.items() if spec["seeds"]}

    @staticmethod
    def _weighted_choice(weighted_items):
        total = sum(max(0.0, weight) for _, weight in weighted_items)
        if total <= 0:
            return random.choice([item for item, _ in weighted_items])
        r = random.random() * total
        acc = 0.0
        for item, weight in weighted_items:
            acc += max(0.0, weight)
            if r <= acc:
                return item
        return weighted_items[-1][0]

    def _make_fragment_seed_v2(
        self,
        population,
        min_size,
        max_size,
        prefer_top=False,
        rank_state=None,
    ):
        fragments = [frag for _, frag in population]
        if prefer_top:
            top_n = max(2, min(len(fragments), self.args.population_size // 3))
            fragments = fragments[:top_n]
        for _ in range(200):
            if rank_state is None:
                frag1, frag2 = random.sample(fragments, 2)
            else:
                sampled = sample_rank_stratified_fragments(
                    population,
                    rank_state,
                    count=2,
                )
                if len(sampled) < 2:
                    return None
                frag1, frag2 = (row[1] for row in sampled)
            smiles = attach_fragments(frag1, frag2)
            can = canonical_smiles(smiles) if smiles else None
            if can is None:
                continue
            atoms = atom_count(can)
            if min_size <= atoms <= max_size:
                return can
        return None

    @staticmethod
    def _update_v2_operator_stat(operator_stats, op_name, scores, best_before):
        old = operator_stats.get(op_name, 1.0)
        if scores:
            gain = (
                max(0.0, max(scores) - best_before)
                if np.isfinite(best_before)
                else max(scores)
            )
            reward = 1.0 + min(2.0, gain)
            operator_stats[op_name] = 0.85 * old + 0.15 * reward
        else:
            operator_stats[op_name] = max(0.2, 0.95 * old)

    def _update_fragment_population_v2(self, population, smiles, score):
        frag_scores = {}
        for old_score, frag in population:
            frag_scores[frag] = max(
                float(old_score), frag_scores.get(frag, -float("inf"))
            )
        try:
            frags = local_genmol_cut(smiles)
        except Exception:
            return
        for frag in frags:
            if Chem.MolFromSmiles(frag) is None:
                continue
            frag_scores[frag] = max(float(score), frag_scores.get(frag, -float("inf")))

        items = sorted(frag_scores.items(), key=lambda item: item[1], reverse=True)
        if len(items) <= self.args.population_size:
            population[:] = [(score, frag) for frag, score in items]
            return

        elite_keep = max(
            2, int(self.args.population_size * self.args.v2_score_elite_fraction)
        )
        selected = items[:elite_keep]
        remaining = items[elite_keep:]
        selected_fps = [
            fp
            for frag, _ in selected
            for fp in [mol_fp_from_smiles(frag)]
            if fp is not None
        ]
        min_score = min(score for _, score in items)
        max_score = max(score for _, score in items)
        denom = max(max_score - min_score, 1e-8)

        while len(selected) < self.args.population_size and remaining:
            best_idx = 0
            best_value = -float("inf")
            for idx, (frag, frag_score) in enumerate(remaining):
                fp = mol_fp_from_smiles(frag)
                novelty = 1.0 - max_tanimoto(fp, selected_fps)
                score_norm = (frag_score - min_score) / denom
                value = score_norm + self.args.v2_diversity_weight * novelty
                if value > best_value:
                    best_value = value
                    best_idx = idx
            frag, frag_score = remaining.pop(best_idx)
            selected.append((frag, frag_score))
            fp = mol_fp_from_smiles(frag)
            if fp is not None:
                selected_fps.append(fp)

        selected.sort(key=lambda item: item[1], reverse=True)
        population[:] = [(score, frag) for frag, score in selected]
