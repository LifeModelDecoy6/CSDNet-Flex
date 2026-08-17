#!/usr/bin/env python
import argparse
import csv
import json
import math
import os
import pickle
import random
import sys
from collections import Counter, defaultdict

import numpy as np
import torch
import torch.nn.functional as F
from rdkit import Chem, RDConfig, RDLogger
from rdkit.Chem import AllChem, BRICS, QED
from rdkit.Chem.Scaffolds import MurckoScaffold

from CSDNet.exp.denovo.sampler_profiles import SAMPLER_PROFILES
from CSDNet.util.checkpoint import load_backbone_from_checkpoint
from CSDNet.util.fsm import (
    ValenceFSMTracker,
    compute_rdkit_kekulize_penalties,
    expand_violation_mask,
    prepare_rdkit_kekulize_checker,
)
from CSDNet.util.sampling import (
    _cosine_remask_rates,
    _length_conditioned_confidence_temperatures,
    _smooth_length_mix,
    sample_csdnet,
)
from CSDNet.util.tokenizer import SMILESTokenizer, tokenize_smiles


RDLogger.DisableLog("rdApp.*")


def load_sa_scorer():
    try:
        from datamol.descriptors import sas
        return sas, "datamol.descriptors.sas"
    except Exception:
        pass

    contrib = os.path.join(RDConfig.RDContribDir, "SA_Score")
    if contrib not in sys.path:
        sys.path.append(contrib)
    try:
        import sascorer
        return sascorer.calculateScore, "rdkit.Contrib.SA_Score.sascorer"
    except Exception as exc:
        raise SystemExit(
            "无法加载 SA scorer。请安装 datamol，或确保 RDKit Contrib "
            "SA_Score/sascorer.py 与 fpscores.pkl.gz 可用。"
        ) from exc


def get_smiles_column(ds):
    if "text" in ds.column_names:
        return "text"
    if "smiles" in ds.column_names:
        return "smiles"
    raise SystemExit(f"数据集中没有 text/smiles 列，实际列: {ds.column_names}")


def clean_brics_fragment(frag_smiles):
    mol = Chem.MolFromSmiles(frag_smiles)
    if mol is None:
        return None
    rw = Chem.RWMol(mol)
    for idx in sorted(
        [atom.GetIdx() for atom in rw.GetAtoms() if atom.GetAtomicNum() == 0],
        reverse=True,
    ):
        rw.RemoveAtom(idx)
    mol = rw.GetMol()
    try:
        Chem.SanitizeMol(mol)
    except Exception:
        return None
    if mol.GetNumHeavyAtoms() == 0:
        return None
    return Chem.MolToSmiles(mol, canonical=True)


def ring_system_atom_sets(mol):
    rings = [set(ring) for ring in mol.GetRingInfo().AtomRings()]
    systems = []
    while rings:
        current = rings.pop()
        changed = True
        while changed:
            changed = False
            rest = []
            for ring in rings:
                if current & ring:
                    current |= ring
                    changed = True
                else:
                    rest.append(ring)
            rings = rest
        systems.append(current)
    return systems


def canonical_motif(smiles, min_atoms, max_atoms):
    mol = Chem.MolFromSmiles(smiles)
    if mol is None:
        return None
    heavy = mol.GetNumHeavyAtoms()
    if heavy < min_atoms or heavy > max_atoms:
        return None
    if any(atom.GetAtomicNum() == 0 for atom in mol.GetAtoms()):
        return None
    return Chem.MolToSmiles(mol, canonical=True)


def extract_motifs(mol, min_atoms=4, max_atoms=32):
    motifs = defaultdict(set)

    try:
        scaffold = MurckoScaffold.GetScaffoldForMol(mol)
        if scaffold is not None and scaffold.GetNumHeavyAtoms() > 0:
            smi = Chem.MolToSmiles(scaffold, canonical=True)
            smi = canonical_motif(smi, min_atoms, max_atoms)
            if smi:
                motifs[smi].add("murcko")
    except Exception:
        pass

    try:
        for frag in BRICS.BRICSDecompose(mol, minFragmentSize=min_atoms):
            smi = clean_brics_fragment(frag)
            if smi:
                smi = canonical_motif(smi, min_atoms, max_atoms)
                if smi:
                    motifs[smi].add("brics")
    except Exception:
        pass

    for atom_set in ring_system_atom_sets(mol):
        try:
            smi = Chem.MolFragmentToSmiles(
                mol,
                atomsToUse=sorted(atom_set),
                canonical=True,
                kekuleSmiles=False,
            )
            smi = canonical_motif(smi, min_atoms, max_atoms)
            if smi:
                motifs[smi].add("ring_system")
        except Exception:
            continue

    return motifs


def quality_label(qed, sa, qed_threshold=0.6, sa_threshold=4.0):
    return float(qed) >= qed_threshold and float(sa) <= sa_threshold


def mine_motif_library(args):
    from datasets import load_from_disk

    ds = load_from_disk(args.data_dir)
    smiles_col = get_smiles_column(ds)
    if "qed" not in ds.column_names or "sa" not in ds.column_names:
        raise SystemExit("数据集缺少 qed/sa 列，请先运行属性计算脚本。")

    if not args.no_shuffle:
        ds = ds.shuffle(seed=args.seed)
    n = min(args.mine_n, len(ds))

    stats = {}
    total = 0
    quality_total = 0
    for row in ds.select(range(n)):
        smi = row.get(smiles_col, "")
        mol = Chem.MolFromSmiles(smi)
        if mol is None:
            continue
        try:
            qed = float(row["qed"])
            sa = float(row["sa"])
        except Exception:
            continue

        is_quality = quality_label(
            qed,
            sa,
            qed_threshold=args.qed_threshold,
            sa_threshold=args.sa_threshold,
        )
        total += 1
        quality_total += int(is_quality)

        motifs = extract_motifs(
            mol,
            min_atoms=args.motif_min_atoms,
            max_atoms=args.motif_max_atoms,
        )
        for motif, kinds in motifs.items():
            item = stats.setdefault(
                motif,
                {
                    "support": 0,
                    "quality_hits": 0,
                    "qed_sum": 0.0,
                    "sa_sum": 0.0,
                    "kinds": Counter(),
                },
            )
            item["support"] += 1
            item["quality_hits"] += int(is_quality)
            item["qed_sum"] += qed
            item["sa_sum"] += sa
            item["kinds"].update(kinds)

    if total == 0:
        raise SystemExit("没有从数据集中读到可用分子。")

    global_quality = quality_total / total
    rows = []
    max_support = max((v["support"] for v in stats.values()), default=1)
    for motif, item in stats.items():
        support = item["support"]
        if support < args.min_support:
            continue
        quality_rate = item["quality_hits"] / support
        mean_qed = item["qed_sum"] / support
        mean_sa = item["sa_sum"] / support
        enrichment = math.log((quality_rate + 1e-6) / (global_quality + 1e-6))
        support_term = math.log1p(support) / math.log1p(max_support)
        score = (
            quality_rate
            + args.enrichment_weight * enrichment
            + args.support_weight * support_term
        )
        mol = Chem.MolFromSmiles(motif)
        if mol is None:
            continue
        rows.append(
            {
                "motif": motif,
                "score": score,
                "support": support,
                "quality_hits": item["quality_hits"],
                "quality_rate": quality_rate,
                "global_quality_rate": global_quality,
                "enrichment": enrichment,
                "mean_qed": mean_qed,
                "mean_sa": mean_sa,
                "num_atoms": mol.GetNumHeavyAtoms(),
                "kinds": ";".join(sorted(item["kinds"])),
            }
        )

    rows.sort(key=lambda r: (r["score"], r["support"]), reverse=True)
    return rows[: args.top_motifs], {
        "n_scanned": n,
        "n_valid": total,
        "global_quality_rate": global_quality,
        "n_raw_motifs": len(stats),
        "n_selected_motifs": min(len(rows), args.top_motifs),
    }


def save_motif_library(rows, meta, output_dir):
    os.makedirs(output_dir, exist_ok=True)
    csv_path = os.path.join(output_dir, "motif_library.csv")
    json_path = os.path.join(output_dir, "motif_library_meta.json")
    fields = [
        "motif",
        "score",
        "support",
        "quality_hits",
        "quality_rate",
        "global_quality_rate",
        "enrichment",
        "mean_qed",
        "mean_sa",
        "num_atoms",
        "kinds",
    ]
    with open(csv_path, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fields)
        writer.writeheader()
        writer.writerows(rows)
    with open(json_path, "w") as f:
        json.dump(meta, f, indent=2, sort_keys=True)
    return csv_path


def load_motif_library(path, top_motifs):
    rows = []
    with open(path) as f:
        reader = csv.DictReader(f)
        for row in reader:
            row["score"] = float(row["score"])
            row["support"] = int(float(row["support"]))
            row["quality_rate"] = float(row["quality_rate"])
            row["mean_qed"] = float(row["mean_qed"])
            row["mean_sa"] = float(row["mean_sa"])
            row["num_atoms"] = int(float(row["num_atoms"]))
            rows.append(row)
    rows.sort(key=lambda r: (r["score"], r["support"]), reverse=True)
    return rows[:top_motifs]


def build_ref_lengths(args, tk):
    from datasets import load_from_disk

    ds = load_from_disk(args.data_dir)
    smiles_col = get_smiles_column(ds)
    sample_n = min(args.ref_sample_n, len(ds))
    texts = ds.select(range(sample_n))[smiles_col]
    return [
        max(3, min(len(tokenize_smiles(smi)) + 2, args.max_len))
        for smi in texts
        if smi
    ]


def motif_weights(rows):
    scores = np.array(
        [
            max(
                0.0,
                float(row.get("sampling_weight", row["score"])),
            )
            for row in rows
        ],
        dtype=np.float64,
    )
    if scores.sum() <= 0:
        scores = np.ones(len(rows), dtype=np.float64)
    scores = scores / scores.sum()
    return scores


@torch.no_grad()
def sample_csdnet_with_frozen_motifs(
    model,
    tk,
    ref_lengths,
    motifs,
    n_mol,
    device="cuda",
    batch_size=64,
    n_steps=500,
    use_fsm_check=True,
    use_rdkit_kekulize_check=True,
    rdkit_check_interval=25,
    max_sample_retries=3,
    violation_neighborhood=2,
    temperature_start=1.5,
    temperature_end=0.25,
    temperature_power=1.5,
    sampler_profile=None,
):
    sampler_profile = str(
        sampler_profile
        or os.environ.get("CSDNET_LOCAL_SAMPLER_PROFILE", "legacy")
    ).strip().lower()
    sampler_profile = {
        "promax_progressive_length_coupled": "progressive_length_coupled",
        "ztrajlc": "progressive_length_coupled",
        "promax_task_adaptive_local": "task_adaptive_local",
        "task_local": "task_adaptive_local",
    }.get(sampler_profile, sampler_profile)
    if sampler_profile not in {
        "legacy",
        "progressive_length_coupled",
        "task_adaptive_local",
    }:
        raise ValueError(f"Unsupported frozen-motif sampler profile: {sampler_profile}")
    progressive_length_coupled = sampler_profile in {
        "progressive_length_coupled",
        "task_adaptive_local",
    }
    if progressive_length_coupled:
        profile_name = (
            "promax_task_adaptive_local"
            if sampler_profile == "task_adaptive_local"
            else "promax_progressive_length_coupled"
        )
        profile = SAMPLER_PROFILES[profile_name]
        temperature_mode = profile.get(
            "local_temperature_mode", "profile_absolute"
        )
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
        else:
            temperature_start = profile["temperature_start"]
            temperature_end = profile["temperature_end"]
            temperature_power = profile["temperature_power"]
            adaptive_temperature_start_short = profile[
                "adaptive_temperature_start_short"
            ]
            adaptive_temperature_end_short = profile[
                "adaptive_temperature_end_short"
            ]
            adaptive_temperature_power_short = profile[
                "adaptive_temperature_power_short"
            ]
    model.eval()
    fsm_start_step = int(n_steps * 0.8)
    retry_step = int(n_steps * 0.6)
    unk_id = getattr(tk, "unk_id", tk.vocab.get("<unk>", -1))

    fsm_tracker = None
    if use_fsm_check or use_rdkit_kekulize_check:
        fsm_tracker = ValenceFSMTracker(tk)
    rdkit_checker = None
    if use_rdkit_kekulize_check:
        rdkit_checker = prepare_rdkit_kekulize_checker(tk, fsm_tracker)

    motif_probs = motif_weights(motifs)
    generated = []
    motif_used = []

    while len(generated) < n_mol:
        bsz = min(batch_size, n_mol - len(generated))
        chosen = np.random.choice(len(motifs), size=bsz, replace=True, p=motif_probs)
        motif_token_lists = [
            tokenize_smiles(motifs[idx]["motif"])
            for idx in chosen
        ]
        motif_id_lists = [
            [tk.vocab.get(tok, unk_id) for tok in toks]
            for toks in motif_token_lists
        ]
        motif_lens = [len(ids) for ids in motif_id_lists]

        lengths = []
        for mlen in motif_lens:
            L = max(mlen + 4, int(np.random.choice(ref_lengths)))
            lengths.append(L)
        if progressive_length_coupled:
            order = sorted(range(bsz), key=lengths.__getitem__)
            chosen = chosen[order]
            motif_id_lists = [motif_id_lists[index] for index in order]
            lengths = [lengths[index] for index in order]
        maxL = max(lengths)

        x = torch.full((bsz, maxL), tk.mask_id, device=device, dtype=torch.long)
        frozen = torch.zeros((bsz, maxL), device=device, dtype=torch.bool)
        x[:, 0] = tk.bos_id
        frozen[:, 0] = True
        for b, L in enumerate(lengths):
            x[b, L - 1] = tk.eos_id
            frozen[b, L - 1] = True
            if L < maxL:
                x[b, L:] = tk.pad_id
                frozen[b, L:] = True
            ids = motif_id_lists[b]
            mlen = len(ids)
            max_start = max(1, L - 1 - mlen)
            start = random.randint(1, max_start)
            x[b, start:start + mlen] = torch.tensor(ids, device=device)
            frozen[b, start:start + mlen] = True

        output_scores = torch.zeros_like(x, dtype=torch.float)
        non_special = (x != tk.pad_id) & (x != tk.bos_id) & (x != tk.eos_id)
        fillable = non_special & ~frozen
        valid_lens = fillable.sum(dim=1, keepdim=True).float().clamp(min=1)
        ranks_template = torch.arange(maxL, device=device).unsqueeze(0).expand(bsz, -1)

        row_lengths = torch.as_tensor(lengths, device=device, dtype=torch.float32)
        if progressive_length_coupled:
            length_mix = _smooth_length_mix(
                row_lengths,
                profile["adaptive_length_low"],
                profile["adaptive_length_high"],
            )
            confidence_length_mix = _smooth_length_mix(
                (
                    valid_lens.squeeze(1)
                    if profile.get("local_confidence_uses_editable_length", False)
                    else row_lengths
                ),
                profile["adaptive_confidence_length_low"],
                profile["adaptive_confidence_length_high"],
            )

            def interpolate(short_value, long_value):
                return float(short_value) + (
                    float(long_value) - float(short_value)
                ) * length_mix

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
                profile["gumbel_scale"],
            ).unsqueeze(1)
            row_remask_power = interpolate(
                profile["adaptive_remask_power_short"],
                profile["remask_power"],
            )
        else:
            confidence_length_mix = None
            row_temperature_start = torch.full_like(row_lengths, float(temperature_start))
            row_temperature_end = torch.full_like(row_lengths, float(temperature_end))
            row_temperature_power = torch.full_like(row_lengths, float(temperature_power))
            row_gumbel_scale = torch.ones((bsz, 1), device=device)
            row_remask_power = torch.ones_like(row_lengths)

        step = 0
        active_rows = torch.ones(bsz, device=device, dtype=torch.bool)
        retries = torch.zeros(bsz, device=device, dtype=torch.long)
        while step < n_steps:
            if not active_rows.any():
                break

            active_positions = fillable & active_rows.unsqueeze(1)
            sample_positions = active_positions
            if progressive_length_coupled:
                sample_positions = sample_positions & (x == tk.mask_id)
            amask = (x != tk.pad_id).long()
            logits = model(x, amask)
            logits[:, :, tk.bos_id] = -1e9
            logits[:, :, tk.eos_id] = -1e9
            logits[:, :, tk.mask_id] = -1e9
            logits[:, :, tk.pad_id] = -1e9
            if unk_id != -1:
                logits[:, :, unk_id] = -1e9

            progress = step / max(n_steps - 1, 1)
            temperature = row_temperature_end + (
                row_temperature_start - row_temperature_end
            ) * (1.0 - progress) ** row_temperature_power
            sample_logits = logits / temperature[:, None, None]
            cur_tokens = torch.distributions.Categorical(logits=sample_logits).sample()
            confidence_logits = logits
            if confidence_length_mix is not None:
                confidence_temperatures = _length_conditioned_confidence_temperatures(
                    sampling_temperatures=temperature,
                    length_mix=confidence_length_mix,
                    short_temperature=profile[
                        "adaptive_confidence_temperature_short"
                    ],
                )
                confidence_logits = logits / confidence_temperatures[:, None, None]
            lm_log_probs = F.log_softmax(confidence_logits, dim=-1)
            cur_scores = torch.gather(lm_log_probs, 2, cur_tokens.unsqueeze(-1)).squeeze(-1)

            if sample_positions.any():
                x.masked_scatter_(sample_positions, cur_tokens[sample_positions])
                output_scores.masked_scatter_(
                    sample_positions,
                    cur_scores[sample_positions],
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

                        retry_rate = _cosine_remask_rates(
                            retry_step + 1,
                            n_steps,
                            row_remask_power,
                        )
                        retry_cutoff_len = (
                            valid_lens * retry_rate.unsqueeze(1)
                        ).long()
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
            remask_rate = _cosine_remask_rates(t, n_steps, row_remask_power)
            cutoff_len = (valid_lens * remask_rate.unsqueeze(1)).long()
            scores = output_scores.masked_fill(~active_positions, 1000.0)
            gumbel = -torch.log(-torch.log(torch.rand_like(scores) + 1e-10) + 1e-10)
            scores = (
                scores
                + gumbel
                * row_gumbel_scale
                * remask_rate.unsqueeze(1)
            )
            sorted_idx = torch.argsort(scores, dim=1)
            ranks = torch.zeros_like(sorted_idx).scatter_(1, sorted_idx, ranks_template)
            bottom_mask = (ranks < cutoff_len) & active_positions
            x.masked_fill_(bottom_mask, tk.mask_id)
            output_scores.masked_fill_(bottom_mask, -math.inf)
            step += 1

        for i in range(bsz):
            seq = x[i].cpu().tolist()
            if tk.eos_id in seq:
                seq = seq[:seq.index(tk.eos_id) + 1]
            smi = tk.decode(seq).strip("'\"")
            if smi:
                generated.append(smi)
                motif_used.append(motifs[chosen[i]]["motif"])

    return generated, motif_used


def molecule_quality(mol, sa_score, qed_threshold, sa_threshold):
    try:
        qed = float(QED.qed(mol))
        sa = float(sa_score(mol))
    except Exception:
        return None
    sa_reward = max(0.0, min(1.0, (sa_threshold - sa) / max(sa_threshold, 1e-6)))
    return {
        "qed": qed,
        "sa": sa,
        "quality_pass": quality_label(qed, sa, qed_threshold, sa_threshold),
        "quality_reward": 0.5 * qed + 0.5 * sa_reward,
    }


def rerank_by_motif_and_quality(
    smiles,
    motif_used,
    motif_rows,
    select_n,
    output_dir,
    qed_threshold=0.6,
    sa_threshold=4.0,
    motif_weight=1.0,
    property_weight=0.5,
):
    sa_score, _ = load_sa_scorer()
    motif_score = {row["motif"]: float(row["score"]) for row in motif_rows}
    motif_queries = []
    for row in motif_rows:
        q = Chem.MolFromSmiles(row["motif"])
        if q is not None:
            motif_queries.append((row["motif"], float(row["score"]), q))

    rows = []
    seen = set()
    for idx, smi in enumerate(smiles):
        mol = Chem.MolFromSmiles(smi)
        if mol is None:
            continue
        can = Chem.MolToSmiles(mol, canonical=True)
        if can in seen:
            continue
        seen.add(can)

        props = molecule_quality(mol, sa_score, qed_threshold, sa_threshold)
        if props is None:
            continue

        matched = []
        for motif, score, query in motif_queries:
            try:
                if mol.HasSubstructMatch(query):
                    matched.append((motif, score))
            except Exception:
                continue
        best_motif = max(matched, key=lambda x: x[1]) if matched else ("", 0.0)
        seeded_motif = motif_used[idx] if idx < len(motif_used) else ""
        seeded_score = motif_score.get(seeded_motif, 0.0)
        best_score = max(best_motif[1], seeded_score)
        score = (
            motif_weight * best_score
            + property_weight * props["quality_reward"]
            + (0.25 if props["quality_pass"] else 0.0)
        )
        rows.append(
            {
                "smiles": smi,
                "canonical": can,
                "selection_score": score,
                "seeded_motif": seeded_motif,
                "best_matched_motif": best_motif[0],
                "best_motif_score": best_score,
                **props,
            }
        )

    rows.sort(key=lambda r: r["selection_score"], reverse=True)
    selected = rows[:select_n]
    os.makedirs(output_dir, exist_ok=True)
    csv_path = os.path.join(output_dir, "motif_selected_scores.csv")
    fields = [
        "smiles",
        "canonical",
        "selection_score",
        "seeded_motif",
        "best_matched_motif",
        "best_motif_score",
        "qed",
        "sa",
        "quality_pass",
        "quality_reward",
    ]
    with open(csv_path, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fields)
        writer.writeheader()
        writer.writerows(selected)
    return [r["smiles"] for r in selected], csv_path


def write_smiles(path, smiles):
    with open(path, "w") as f:
        for smi in smiles:
            f.write(smi + "\n")


def filter_motifs_by_tokenizer(motifs, tk):
    filtered = []
    for row in motifs:
        tokens = tokenize_smiles(row["motif"])
        if tokens and all(tok in tk.vocab for tok in tokens):
            filtered.append(row)
    return filtered


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--data_dir", default="csdnet_data/pubchem_10m_with_props_v2")
    parser.add_argument("--vocab", default="csdnet_vocab.pkl")
    parser.add_argument("--ckpt_path", required=True)
    parser.add_argument("--output_dir", default="results/csdnet_motif_guided_pubchem6m")
    parser.add_argument("--max_len", type=int, default=128)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--mine_n", type=int, default=200000)
    parser.add_argument("--ref_sample_n", type=int, default=50000)
    parser.add_argument("--motif_library", default=None)
    parser.add_argument("--top_motifs", type=int, default=256)
    parser.add_argument("--min_support", type=int, default=50)
    parser.add_argument("--motif_min_atoms", type=int, default=4)
    parser.add_argument("--motif_max_atoms", type=int, default=32)
    parser.add_argument("--qed_threshold", type=float, default=0.6)
    parser.add_argument("--sa_threshold", type=float, default=4.0)
    parser.add_argument("--enrichment_weight", type=float, default=0.25)
    parser.add_argument("--support_weight", type=float, default=0.10)
    parser.add_argument("--no_shuffle", action="store_true")
    parser.add_argument("--baseline_n", type=int, default=1000)
    parser.add_argument("--candidate_n", type=int, default=5000)
    parser.add_argument("--select_n", type=int, default=1000)
    parser.add_argument("--batch_size", type=int, default=64)
    parser.add_argument("--n_steps", type=int, default=500)
    parser.add_argument("--rdkit_check_interval", type=int, default=25)
    parser.add_argument("--max_sample_retries", type=int, default=3)
    parser.add_argument("--violation_neighborhood", type=int, default=2)
    parser.add_argument("--temperature_start", type=float, default=1.5)
    parser.add_argument("--temperature_end", type=float, default=0.25)
    parser.add_argument("--temperature_power", type=float, default=1.5)
    parser.add_argument("--disable_baseline", action="store_true")
    args = parser.parse_args()

    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    os.makedirs(args.output_dir, exist_ok=True)

    with open(args.vocab, "rb") as f:
        tk = SMILESTokenizer(pickle.load(f))

    if args.motif_library:
        motifs = load_motif_library(args.motif_library, args.top_motifs)
        motif_csv = args.motif_library
        meta = {"loaded_from": args.motif_library}
    else:
        print("🚀 Mining motif library from PubChem10M properties...")
        motifs, meta = mine_motif_library(args)
        motif_csv = save_motif_library(motifs, meta, args.output_dir)
    if not motifs:
        raise SystemExit("motif library 为空，请降低 --min_support 或扩大 --mine_n。")
    motifs = filter_motifs_by_tokenizer(motifs, tk)
    if not motifs:
        raise SystemExit("motif library 全部含有 tokenizer 词表外 token，请检查 vocab。")

    print(f"✔️ Motif library: {motif_csv}")
    print(f"✔️ Selected motifs: {len(motifs)}")
    print("Top motifs:")
    for row in motifs[:10]:
        print(
            f"  {row['motif']} | score={float(row['score']):.3f} "
            f"support={row['support']} quality={float(row['quality_rate']):.3f}"
        )

    ref_lengths = build_ref_lengths(args, tk)
    device = "cuda" if torch.cuda.is_available() else "cpu"
    model = load_backbone_from_checkpoint(args.ckpt_path, tk, device=device)
    model = model.to(device)
    model.eval()

    if not args.disable_baseline and args.baseline_n > 0:
        print(f"\n🔥 Generating baseline de novo set: n={args.baseline_n}")
        baseline = sample_csdnet(
            model=model,
            tk=tk,
            ref_lengths=ref_lengths,
            n_mol=args.baseline_n,
            cond=None,
            device=device,
            batch_size=args.batch_size,
            n_steps=args.n_steps,
            use_fsm_check=True,
            use_rdkit_kekulize_check=True,
            rdkit_check_interval=args.rdkit_check_interval,
            max_sample_retries=args.max_sample_retries,
            violation_neighborhood=args.violation_neighborhood,
            temperature_start=args.temperature_start,
            temperature_end=args.temperature_end,
            temperature_power=args.temperature_power,
        )
        write_smiles(os.path.join(args.output_dir, "baseline_generated_mols.txt"), baseline)

    print(f"\n🔥 Generating motif-seeded candidates: n={args.candidate_n}")
    candidates, motif_used = sample_csdnet_with_frozen_motifs(
        model=model,
        tk=tk,
        ref_lengths=ref_lengths,
        motifs=motifs,
        n_mol=args.candidate_n,
        device=device,
        batch_size=args.batch_size,
        n_steps=args.n_steps,
        use_fsm_check=True,
        use_rdkit_kekulize_check=True,
        rdkit_check_interval=args.rdkit_check_interval,
        max_sample_retries=args.max_sample_retries,
        violation_neighborhood=args.violation_neighborhood,
        temperature_start=args.temperature_start,
        temperature_end=args.temperature_end,
        temperature_power=args.temperature_power,
    )
    candidate_path = os.path.join(args.output_dir, "motif_seeded_candidates.txt")
    write_smiles(candidate_path, candidates)
    with open(os.path.join(args.output_dir, "motif_used_for_candidates.txt"), "w") as f:
        for motif in motif_used:
            f.write(motif + "\n")

    print(f"\n🔥 Selecting motif-guided set: n={args.select_n}")
    selected, selected_csv = rerank_by_motif_and_quality(
        candidates,
        motif_used,
        motifs,
        select_n=args.select_n,
        output_dir=args.output_dir,
        qed_threshold=args.qed_threshold,
        sa_threshold=args.sa_threshold,
    )
    selected_path = os.path.join(args.output_dir, "motif_guided_selected_mols.txt")
    write_smiles(selected_path, selected)
    summary = {
        "motif_meta": meta,
        "motif_library": motif_csv,
        "candidate_path": candidate_path,
        "selected_path": selected_path,
        "selected_score_csv": selected_csv,
        "n_candidates": len(candidates),
        "n_selected": len(selected),
    }
    with open(os.path.join(args.output_dir, "motif_guided_summary.json"), "w") as f:
        json.dump(summary, f, indent=2, sort_keys=True)
    print(f"✔️ Candidate molecules: {candidate_path}")
    print(f"✔️ Selected molecules: {selected_path}")
    print(f"✔️ Selected scores: {selected_csv}")


if __name__ == "__main__":
    main()
