#!/usr/bin/env python
import csv
import math
import os
import pickle
import random
import sys
from pathlib import Path
from time import time

REPO_ROOT = Path(__file__).resolve().parents[5]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

import numpy as np
import pandas as pd
import torch
import torch.nn.functional as F
from rdkit import Chem, RDLogger
from rdkit.Chem import AllChem

from csdnet_core import (
    ValenceFSMTracker,
    _compute_rdkit_kekulize_penalties,
    _expand_violation_mask,
    _prepare_rdkit_kekulize_checker,
)
from csdnet_eval import load_backbone_from_checkpoint
from csdnet_motif_guided_sampling import extract_motifs, sample_csdnet_with_frozen_motifs
from csdnet_tokenizer import SMILESTokenizer, tokenize_smiles
from scripts.exps.pmo.main.optimizer import BaseOptimizer, top_auc

try:
    from genmol.utils.utils_chem import cut as genmol_cut
except Exception:
    genmol_cut = None


RDLogger.DisableLog("rdApp.*")

ROOT_DIR = os.path.dirname(os.path.dirname(os.path.dirname(os.path.realpath(__file__))))
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


def load_pmo_motifs(oracle_name, tk, max_len, limit, min_atoms=4, max_atoms=36):
    path = os.path.join(ROOT_DIR, "vocab", f"{oracle_name}.csv")
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
    path = os.path.join(ROOT_DIR, "vocab", f"{oracle_name}.csv")
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
    exists = os.path.exists(path)
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
    ]
    with open(path, "a", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fields)
        if not exists:
            writer.writeheader()
        writer.writerow(row)


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
):
    model.eval()
    if not seed_smiles:
        return []
    fsm_start_step = int(n_steps * 0.8)
    retry_step = int(n_steps * 0.6)
    unk_id = getattr(tk, "unk_id", tk.vocab.get("<unk>", -1))

    fsm_tracker = None
    if use_fsm_check or use_rdkit_kekulize_check:
        fsm_tracker = ValenceFSMTracker(tk, dict_path="valence_dict.json")
    rdkit_checker = None
    if use_rdkit_kekulize_check:
        rdkit_checker = _prepare_rdkit_kekulize_checker(tk, fsm_tracker)

    generated = []
    for offset in range(0, len(seed_smiles), batch_size):
        seeds = seed_smiles[offset: offset + batch_size]
        token_lists = []
        lengths = []
        mask_positions = []
        for smi in seeds:
            toks = tokenize_smiles(smi)
            toks = toks[: max_len - 2]
            token_lists.append(toks)
            lengths.append(len(toks) + 2)
            mask_positions.append(
                choose_remask_positions(
                    len(toks),
                    fraction=remask_fraction,
                    min_tokens=min_remask_tokens,
                    span_prob=span_prob,
                )
            )

        maxL = max(lengths)
        bsz = len(seeds)
        x = torch.full((bsz, maxL), tk.pad_id, device=device, dtype=torch.long)
        frozen = torch.ones((bsz, maxL), device=device, dtype=torch.bool)
        for b, toks in enumerate(token_lists):
            ids = [tk.bos_id] + [tk.vocab.get(tok, unk_id) for tok in toks] + [tk.eos_id]
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
                    penalties += _compute_rdkit_kekulize_penalties(
                        x, tk, chem, rdkit_focus_ids
                    )
                check_positions = non_special & active_rows.unsqueeze(1)
                penalties = penalties.masked_fill(~check_positions, 0.0)
                output_scores += penalties.masked_fill(frozen, 0.0)

                violation_positions = (penalties < 0) & check_positions
                if violation_positions.any() and step != n_steps - 1:
                    repair_mask = _expand_violation_mask(
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
                        retry_repair_mask = _expand_violation_mask(
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

    def _optimize(self, oracle, config):
        self.oracle.assign_evaluator(oracle)
        t_start = time()
        if self.mode == "motif_seeded":
            self._run_motif_seeded(oracle.name)
        elif self.mode == "iterative_remask":
            self._run_iterative_remask(oracle.name)
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
                "oracle": oracle.name,
                "seed": self.args.seed,
                "elapsed_sec": elapsed,
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
        for motif_set in new_motifs.values():
            for motif in motif_set:
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
        if genmol_cut is None:
            return
        known = {frag for _, frag in population}
        try:
            frags = genmol_cut(smiles)
        except Exception:
            return
        for frag in frags:
            if frag in known or Chem.MolFromSmiles(frag) is None:
                continue
            known.add(frag)
            population.append((float(score), frag))
        population.sort(key=lambda item: item[0], reverse=True)
        del population[self.args.population_size:]
