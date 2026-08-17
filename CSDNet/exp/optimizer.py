#!/usr/bin/env python
import csv
import math
import os
import pickle
import random
from collections import defaultdict
from pathlib import Path
from time import time

import numpy as np
import pandas as pd
import torch
import torch.nn.functional as F
from rdkit import Chem, DataStructs, RDLogger
from rdkit.Chem import AllChem

from CSDNet.exp.pmo.base_optimizer import BaseOptimizer, top_auc
from CSDNet.util.checkpoint import load_backbone_from_checkpoint
from CSDNet.util.fsm import (
    ValenceFSMTracker,
    compute_rdkit_kekulize_penalties,
    expand_violation_mask,
    prepare_rdkit_kekulize_checker,
)
from CSDNet.util.motif import extract_motifs, sample_csdnet_with_frozen_motifs
from CSDNet.util.tokenizer import SMILESTokenizer, tokenize_smiles


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
    """Legacy benchmark-specific bounds retained for V1-V7 reproducibility."""
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


def load_ref_lengths(data_dir, tk, max_len, sample_n=50000):
    if not data_dir:
        return list(range(16, min(max_len, 96) + 1))
    try:
        from datasets import load_from_disk
        ds = load_from_disk(data_dir)
    except Exception as exc:
        print(f"Could not load data_dir={data_dir}; using uniform reference lengths. {exc}")
        return list(range(16, min(max_len, 96) + 1))

    col = "text" if "text" in ds.column_names else "smiles" if "smiles" in ds.column_names else None
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
    model = load_backbone_from_checkpoint(args.ckpt_path, tk, device=device)
    model = model.to(device)
    model.eval()
    return model, tk, device


def pmo_vocab_path(oracle_name):
    path = os.path.join(ROOT_DIR, "vocab", f"{oracle_name}.csv")
    if os.path.exists(path):
        return path
    if oracle_name.endswith("_current"):
        fallback = os.path.join(ROOT_DIR, "vocab", f"{oracle_name[:-len('_current')]}.csv")
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
        raise RuntimeError(f"Too few tokenizable PMO motifs for {oracle_name}: {len(motifs)}")
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
    ordered_by_time = dict(sorted(buffer.items(), key=lambda kv: kv[1][1], reverse=False))
    top = sorted(buffer.items(), key=lambda kv: kv[1][0], reverse=True)[:100]
    scores = [x[1][0] for x in top]
    return {
        "calls": len(buffer),
        "avg_top1": float(np.max(scores)),
        "avg_top10": float(np.mean(sorted(scores, reverse=True)[:10])),
        "avg_top100": float(np.mean(scores)),
        "auc_top1": float(top_auc(ordered_by_time, 1, True, freq_log, max_oracle_calls)),
        "auc_top10": float(top_auc(ordered_by_time, 10, True, freq_log, max_oracle_calls)),
        "auc_top100": float(top_auc(ordered_by_time, 100, True, freq_log, max_oracle_calls)),
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
        return tokens, choose_remask_positions(body_len, fraction, min_tokens, span_prob)

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
        span = tokens[start: start + span_len]
        if "." in span:
            continue
        if any(_is_length_edit_atom_token(tok) for tok in span):
            starts.append(start)
    if starts:
        start = random.choice(starts)
    else:
        start = random.randint(0, body_len - span_len)

    edited = tokens[:start] + ["<mask>"] * new_span_len + tokens[start + span_len:]
    max_body_len = max(1, body_len + max(deltas + [0]))
    edited = edited[:max_body_len]
    mask_positions = [
        idx + 1
        for idx, tok in enumerate(edited)
        if tok == "<mask>"
    ]
    return edited, mask_positions


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
    length_delta_choices=None,
    length_edit_prob=0.0,
    length_edit_min_span=1,
    length_edit_max_span=8,
    return_seed_indices=False,
):
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
    for offset in range(0, len(seed_smiles), batch_size):
        seeds = seed_smiles[offset: offset + batch_size]
        token_lists = []
        lengths = []
        mask_positions = []
        for smi in seeds:
            toks = tokenize_smiles(smi)
            toks = toks[: max_len - 2]
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
            toks = toks[: max_len - 2]
            mask_pos = [pos for pos in mask_pos if 0 < pos <= len(toks)]
            token_lists.append(toks)
            lengths.append(len(toks) + 2)
            mask_positions.append(mask_pos)

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
            mask_pos = [
                p for p in mask_positions[b]
                if 0 < p < L - 1 and x[b, p].item() == tk.mask_id
            ]
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

        step = 0
        active_rows = torch.ones(bsz, device=device, dtype=torch.bool)
        retries = torch.zeros(bsz, device=device, dtype=torch.long)
        while step < n_steps:
            if not active_rows.any():
                break
            active_positions = fillable & active_rows.unsqueeze(1)
            amask = (x != tk.pad_id).long()
            logits = model(x, amask)
            logits[:, :, tk.bos_id] = -1e9
            logits[:, :, tk.eos_id] = -1e9
            logits[:, :, tk.mask_id] = -1e9
            logits[:, :, tk.pad_id] = -1e9
            if unk_id != -1:
                logits[:, :, unk_id] = -1e9

            progress = step / max(n_steps - 1, 1)
            temperature = temperature_end + (temperature_start - temperature_end) * (
                1.0 - progress
            ) ** temperature_power
            cur_tokens = torch.distributions.Categorical(logits=logits / temperature).sample()
            lm_log_probs = F.log_softmax(logits, dim=-1)
            cur_scores = torch.gather(lm_log_probs, 2, cur_tokens.unsqueeze(-1)).squeeze(-1)
            if active_positions.any():
                x.masked_scatter_(active_positions, cur_tokens[active_positions])
                output_scores.masked_scatter_(active_positions, cur_scores[active_positions])

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
                if should_check_fsm:
                    penalties += fsm_tracker.compute_penalties(x)
                if should_check_rdkit:
                    chem, rdkit_focus_ids = rdkit_checker
                    penalties += compute_rdkit_kekulize_penalties(
                        x, tk, chem, rdkit_focus_ids
                    )
                check_positions = non_special & active_rows.unsqueeze(1)
                penalties = penalties.masked_fill(~check_positions, 0.0)
                output_scores += penalties.masked_fill(frozen, 0.0)

                violation_positions = (penalties < 0) & check_positions
                if violation_positions.any() and step != n_steps - 1:
                    repair_mask = expand_violation_mask(
                        violation_positions,
                        active_positions,
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
                        retry_violation_positions = violation_positions & retry_rows.unsqueeze(1)
                        retry_repair_mask = expand_violation_mask(
                            retry_violation_positions,
                            retry_positions,
                            radius=violation_neighborhood,
                        )
                        x.masked_fill_(retry_repair_mask, tk.mask_id)
                        output_scores.masked_fill_(retry_repair_mask, -math.inf)

                        retry_rate = math.cos((retry_step + 1) / n_steps * math.pi * 0.5)
                        retry_cutoff_len = (valid_lens * retry_rate).long()
                        retry_scores = output_scores.masked_fill(~retry_positions, 1000.0)
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
            remask_rate = math.cos(t / n_steps * math.pi * 0.5)
            cutoff_len = (valid_lens * remask_rate).long()
            scores = output_scores.masked_fill(~active_positions, 1000.0)
            gumbel = -torch.log(-torch.log(torch.rand_like(scores) + 1e-10) + 1e-10)
            scores = scores + gumbel * remask_rate
            sorted_idx = torch.argsort(scores, dim=1)
            ranks = torch.zeros_like(sorted_idx).scatter_(1, sorted_idx, ranks_template)
            bottom_mask = (ranks < cutoff_len) & active_positions
            x.masked_fill_(bottom_mask, tk.mask_id)
            output_scores.masked_fill_(bottom_mask, -math.inf)
            step += 1

        for i in range(bsz):
            seq = x[i].cpu().tolist()
            if tk.eos_id in seq:
                seq = seq[: seq.index(tk.eos_id) + 1]
            smi = tk.decode(seq).strip("'\"")
            can = canonical_smiles(smi)
            if can is not None:
                if return_seed_indices:
                    generated.append((can, offset + i))
                else:
                    generated.append(can)
    return generated


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
                    1 for score, _ in self.oracle.mol_buffer.values()
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

    def _run_motif_seeded(self, oracle_name):
        motifs = load_pmo_motifs(
            oracle_name,
            self.tk,
            max_len=self.args.max_len,
            limit=self.args.population_size,
            min_atoms=self.args.motif_min_atoms,
            max_atoms=self.args.motif_max_atoms,
        )
        csv_path = os.path.join(self.args.output_dir, f"{oracle_name}_{self.args.seed}.csv")
        min_size, max_size = task_size_bounds(oracle_name)
        print(f"[motif_seeded:{oracle_name}] motifs={len(motifs)} size={min_size}-{max_size}")

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
        del motifs[self.args.population_size:]

    def _run_iterative_remask(self, oracle_name):
        population = load_pmo_fragments(oracle_name, self.args.population_size)
        csv_path = os.path.join(self.args.output_dir, f"{oracle_name}_{self.args.seed}.csv")
        min_size, max_size = task_size_bounds(oracle_name)
        elites = []
        print(f"[iterative_remask:{oracle_name}] fragments={len(population)} size={min_size}-{max_size}")

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
                del elites[self.args.elite_size:]
                self._update_fragment_population(population, can, score)

    def _make_seed_batch(self, population, elites, min_size, max_size):
        seeds = []
        attempts = 0
        while len(seeds) < self.args.candidate_batch_size and attempts < self.args.candidate_batch_size * 80:
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
        del population[self.args.population_size:]

    def _run_iterative_remask_v2(self, oracle_name):
        population = load_pmo_fragments(oracle_name, self.args.population_size)
        csv_path = os.path.join(self.args.output_dir, f"{oracle_name}_{self.args.seed}.csv")
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
                        and max_tanimoto(fp, recent_fps) >= self.args.v2_near_duplicate_sim
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
                    del elites[self.args.elite_size:]
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
        csv_path = os.path.join(self.args.output_dir, f"{oracle_name}_{self.args.seed}.csv")
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
                        and max_tanimoto(fp, recent_fps) >= self.args.v3_near_duplicate_sim
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
                            del recent_fps[: len(recent_fps) - self.args.v3_recent_memory]
                    if score > self.args.v3_nonzero_threshold or len(elites) < self.args.elite_size // 3:
                        elites.append((score, can))
                        elites.sort(key=lambda item: item[0], reverse=True)
                        del elites[self.args.elite_size:]
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
        csv_path = os.path.join(self.args.output_dir, f"{oracle_name}_{self.args.seed}.csv")
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
            rescue = zero_rescue or stagnant_calls >= self.args.v4_stagnation_rescue_patience
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
                    if can is None or can in seen_batch or can in self.oracle.mol_buffer:
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
                        and max_tanimoto(fp, recent_fps) >= self.args.v4_near_duplicate_sim
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
                            del recent_fps[: len(recent_fps) - self.args.v4_recent_memory]
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
        csv_path = os.path.join(self.args.output_dir, f"{oracle_name}_{self.args.seed}.csv")
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
                    if can is None or can in seen_batch or can in self.oracle.mol_buffer:
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
                        and max_tanimoto(fp, recent_fps) >= self.args.v5_near_duplicate_sim
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
                            del recent_fps[: len(recent_fps) - self.args.v5_recent_memory]
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
        csv_path = os.path.join(self.args.output_dir, f"{oracle_name}_{self.args.seed}.csv")
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
                    if can is None or can in seen_batch or can in self.oracle.mol_buffer:
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
                        and max_tanimoto(fp, recent_fps) >= self.args.v5_near_duplicate_sim
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
                            del recent_fps[: len(recent_fps) - self.args.v5_recent_memory]
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
        csv_path = os.path.join(self.args.output_dir, f"{oracle_name}_{self.args.seed}.csv")
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
                    if can is None or can in seen_batch or can in self.oracle.mol_buffer:
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
                        and max_tanimoto(fp, recent_fps) >= self.args.v5_near_duplicate_sim
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
                            del recent_fps[: len(recent_fps) - self.args.v5_recent_memory]
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
        csv_path = os.path.join(self.args.output_dir, f"{oracle_name}_{self.args.seed}.csv")
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
                {
                    op: {"ema": 0.50, "pulls": 0.0, "positive": 0.0}
                    for op in operators
                },
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
                    if can is None or can in seen_batch or can in self.oracle.mol_buffer:
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
                        and max_tanimoto(fp, recent_fps) >= self.args.v8_near_duplicate_sim
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
                            del recent_fps[: len(recent_fps) - self.args.v5_recent_memory]

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

    def _nonzero_threshold(self):
        if self.mode in {"iterative_remask_v7", "iterative_remask_v8"}:
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
            1 for score, _ in self.oracle.mol_buffer.values()
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
        if avg_top1 < self.args.v5_low_top1_threshold and nonzero_rate >= self.args.v5_sparse_nonzero_rate:
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
            int(round(self.args.candidate_batch_size * self.args.v5_overgenerate_factor)),
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
                "temperature_start": explore_temp if state in {"explore", "sparse"} else self.args.temperature_start,
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
            int(round(self.args.candidate_batch_size * self.args.v5_overgenerate_factor)),
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
                "temperature_start": explore_temp if state in {"explore", "sparse"} else self.args.temperature_start,
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

    def _v7_operator_weights(self, state, has_elites, has_diverse, has_motifs, stagnant_calls):
        weights = self._v5_operator_weights(
            state=state,
            has_elites=has_elites,
            has_diverse=has_diverse,
            has_motifs=has_motifs,
        )
        weights.setdefault("length_shrink_rescue", 0.0)
        weights.setdefault("length_expand_rescue", 0.0)
        length_ready = (
            has_elites
            and stagnant_calls >= self.args.v7_length_rescue_after_stagnant
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

    def _v6_operator_multipliers(self, state, arm_stats, has_elites, has_diverse, has_motifs):
        base_weights = self._v5_operator_weights(
            state=state,
            has_elites=has_elites,
            has_diverse=has_diverse,
            has_motifs=has_motifs,
        )
        total_pulls = sum(float(stats.get("pulls", 0.0)) for stats in arm_stats.values())
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
            nonzero_rate = sum(score > self._nonzero_threshold() for score in values) / len(values)
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
        total_pulls = sum(float(stats.get("pulls", 0.0)) for stats in arm_stats.values())
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

    def _update_v7_arm_stat(self, arm_stats, op_name, scores, before_metrics, after_metrics):
        stats = arm_stats.setdefault(op_name, {"ema": 0.50, "pulls": 0.0})
        old_ema = float(stats.get("ema", 0.50))
        alpha = min(1.0, max(0.0, self.args.v7_bandit_alpha))
        if scores:
            values = [float(score) for score in scores]
            nonzero_rate = sum(score > self._nonzero_threshold() for score in values) / len(values)
            top10_threshold = self._buffer_score_threshold(top_n=10)
            if top10_threshold is None:
                top10_entry_rate = 0.0
            else:
                top10_entry_rate = sum(score >= top10_threshold - 1e-12 for score in values) / len(values)

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
            int(round(self.args.candidate_batch_size * self.args.v8_overgenerate_factor)),
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
                    "parent_score": None if parent_score is None else float(parent_score),
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
            and any(atom.GetAtomicNum() == 0 for atom in Chem.MolFromSmiles(frag).GetAtoms())
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
            percentile = sum(float(old) <= score for old in before_scores) / len(before_scores)
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
        del elites[self.args.elite_size:]
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
            key=lambda row: (float(row.get("score", 0.0)), float(row.get("support", 0.0))),
            reverse=True,
        )
        del motifs[self.args.v5_motif_pool_size:]

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
        del elites[self.args.elite_size:]

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
        old_pool_size = self.args.v4_motif_pool_size if hasattr(self.args, "v4_motif_pool_size") else None
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
            int(round(self.args.candidate_batch_size * self.args.v4_overgenerate_factor)),
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
                    prefer_top=(op in {"fragment_medium", "fragment_large"} and not rescue),
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
                "elite_small": 0.12 if has_elites and not zero_rescue else 0.04 if has_elites else 0.0,
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
        del elites[self.args.elite_size:]

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

        elite_keep = max(2, int(self.args.v4_diverse_size * self.args.v4_score_elite_fraction))
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
        motifs.sort(key=lambda row: (float(row["score"]), float(row.get("support", 0.0))), reverse=True)
        del motifs[self.args.v4_motif_pool_size:]

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
        rescue = zero_rescue or stagnant_calls >= self.args.v3_stagnation_rescue_patience

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
                "temperature_start": max(self.args.temperature_start, self.args.v3_rescue_temperature),
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
            sum(len(spec["seeds"]) for spec in specs.values()) < self.args.candidate_batch_size
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
        fractions = sorted(parse_float_list(self.args.v2_remask_fractions, [0.15, 0.30, 0.50]))
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
            "elite_small": {"remask_fraction": small, "temperature_start": 1.0, "seeds": []},
            "elite_medium": {"remask_fraction": medium, "temperature_start": 1.2, "seeds": []},
            "fragment_small": {"remask_fraction": small, "temperature_start": 1.0, "seeds": []},
            "fragment_medium": {"remask_fraction": medium, "temperature_start": 1.2, "seeds": []},
            "fragment_large": {"remask_fraction": large, "temperature_start": 1.5, "seeds": []},
        }

        attempts = 0
        while (
            sum(len(spec["seeds"]) for spec in specs.values()) < self.args.candidate_batch_size
            and attempts < self.args.candidate_batch_size * 120
        ):
            attempts += 1
            op = self._weighted_choice(weighted_ops)
            if op.startswith("elite") and elites:
                smi = random.choice(elites[: max(1, min(len(elites), self.args.elite_size))])[1]
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

    def _make_fragment_seed_v2(self, population, min_size, max_size, prefer_top=False):
        fragments = [frag for _, frag in population]
        if prefer_top:
            top_n = max(2, min(len(fragments), self.args.population_size // 3))
            fragments = fragments[:top_n]
        for _ in range(200):
            frag1, frag2 = random.sample(fragments, 2)
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
            gain = max(0.0, max(scores) - best_before) if np.isfinite(best_before) else max(scores)
            reward = 1.0 + min(2.0, gain)
            operator_stats[op_name] = 0.85 * old + 0.15 * reward
        else:
            operator_stats[op_name] = max(0.2, 0.95 * old)

    def _update_fragment_population_v2(self, population, smiles, score):
        frag_scores = {}
        for old_score, frag in population:
            frag_scores[frag] = max(float(old_score), frag_scores.get(frag, -float("inf")))
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

        elite_keep = max(2, int(self.args.population_size * self.args.v2_score_elite_fraction))
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
