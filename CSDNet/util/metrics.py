import csv
import json
import os
import sys
from collections import Counter

import numpy as np
from rdkit import Chem, RDConfig, RDLogger
from rdkit.Chem import AllChem, DataStructs, Descriptors, QED


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
            "Could not load SA scorer. Install datamol or make RDKit Contrib SA_Score available."
        ) from exc


def calculate_basic_metrics(smiles_list, train_set):
    valid_smiles = []
    mols = []
    for smi in smiles_list:
        try:
            mol = Chem.MolFromSmiles(smi)
            if mol is not None:
                valid_smiles.append(smi)
                mols.append(mol)
        except Exception:
            pass

    val = len(valid_smiles) / len(smiles_list) * 100 if smiles_list else 0
    if len(valid_smiles) == 0:
        return {"Validity": 0, "Uniqueness": 0, "Novelty": 0, "IntDiv": 0}

    unique_smiles = set(valid_smiles)
    uniq = len(unique_smiles) / len(valid_smiles) * 100
    novel = [s for s in unique_smiles if s not in train_set]
    nov = len(novel) / len(unique_smiles) * 100

    fps = [AllChem.GetMorganFingerprintAsBitVect(m, 2, nBits=1024) for m in mols]
    if len(fps) > 1000:
        indices = np.random.choice(len(fps), 1000, replace=False)
        fps = [fps[i] for i in indices]

    n_fps = len(fps)
    if n_fps < 2:
        intdiv = 0.0
    else:
        sim_sum = 0.0
        for i in range(n_fps):
            sims = DataStructs.BulkTanimotoSimilarity(fps[i], fps[i + 1:])
            sim_sum += sum(sims)
        total_pairs = n_fps * (n_fps - 1) / 2
        intdiv = 1.0 - sim_sum / total_pairs

    return {"Validity": val, "Uniqueness": uniq, "Novelty": nov, "IntDiv": intdiv}


def internal_diversity(mols, radius=2, n_bits=2048):
    if len(mols) < 2:
        return 0.0
    fps = [AllChem.GetMorganFingerprintAsBitVect(m, radius, nBits=n_bits) for m in mols]
    sim_sum = 0.0
    n_pairs = 0
    for i, fp in enumerate(fps):
        sims = DataStructs.BulkTanimotoSimilarity(fp, fps[i + 1:])
        sim_sum += float(sum(sims))
        n_pairs += len(sims)
    return 1.0 - sim_sum / n_pairs if n_pairs else 0.0


def evaluate_genmol_quality(
    smiles,
    denominator,
    qed_threshold=0.6,
    sa_threshold=4.0,
    fp_nbits=2048,
):
    sa_score, sa_source = load_sa_scorer()
    valid = []
    invalid = 0

    for smi in smiles:
        mol = Chem.MolFromSmiles(smi)
        if mol is None:
            invalid += 1
            continue
        can = Chem.MolToSmiles(mol, canonical=True)
        valid.append((smi, can, mol))

    unique = {}
    for smi, can, mol in valid:
        if can not in unique:
            unique[can] = (smi, mol)

    unique_rows = []
    quality_count = 0
    for can, (smi, mol) in unique.items():
        try:
            qed = float(QED.qed(mol))
            sa = float(sa_score(mol))
        except Exception:
            qed = float("nan")
            sa = float("nan")
        passed = bool(qed >= qed_threshold and sa <= sa_threshold)
        if passed:
            quality_count += 1
        unique_rows.append(
            {
                "smiles": smi,
                "canonical": can,
                "qed": qed,
                "sa": sa,
                "quality_pass": passed,
            }
        )

    unique_mols = [mol for _, mol in unique.values()]
    diversity = internal_diversity(unique_mols, n_bits=fp_nbits)
    n_total = max(int(denominator), 1)
    n_valid = len(valid)
    n_unique = len(unique)
    qeds = [row["qed"] for row in unique_rows if not np.isnan(row["qed"])]
    sas = [row["sa"] for row in unique_rows if not np.isnan(row["sa"])]

    metrics = {
        "TotalRequested": n_total,
        "TotalGenerated": len(smiles),
        "Valid": n_valid,
        "Invalid": invalid + max(n_total - len(smiles), 0),
        "UniqueValid": n_unique,
        "QualityCount": quality_count,
        "Validity": n_valid / n_total,
        "Uniqueness": n_unique / n_valid if n_valid else 0.0,
        "UniquenessTotal": n_unique / n_total,
        "Quality": quality_count / n_total,
        "Diversity": diversity,
        "MeanQEDUniqueValid": float(np.mean(qeds)) if qeds else 0.0,
        "MeanSAUniqueValid": float(np.mean(sas)) if sas else 0.0,
        "QEDThreshold": qed_threshold,
        "SAThreshold": sa_threshold,
        "SAReader": sa_source,
        "FingerprintBits": fp_nbits,
    }
    return metrics, unique_rows


def parse_quality_thresholds(text, base_qed, base_sa):
    thresholds = [(float(base_qed), float(base_sa))]
    if text:
        for item in text.split(","):
            item = item.strip()
            if not item:
                continue
            if ":" in item:
                left, right = item.split(":", 1)
            elif "/" in item:
                left, right = item.split("/", 1)
            else:
                raise SystemExit("Quality threshold format should look like '0.6:4,0.5:5'.")
            thresholds.append((float(left), float(right)))

    deduped = []
    seen = set()
    for qed_threshold, sa_threshold in thresholds:
        key = (round(qed_threshold, 8), round(sa_threshold, 8))
        if key not in seen:
            seen.add(key)
            deduped.append((qed_threshold, sa_threshold))
    return deduped


def quality_table(unique_rows, denominator, thresholds):
    n_total = max(int(denominator), 1)
    rows = []
    for qed_threshold, sa_threshold in thresholds:
        count = 0
        for row in unique_rows:
            qed = row["qed"]
            sa = row["sa"]
            if np.isnan(qed) or np.isnan(sa):
                continue
            if qed >= qed_threshold and sa <= sa_threshold:
                count += 1
        rows.append(
            {
                "qed_threshold": qed_threshold,
                "sa_threshold": sa_threshold,
                "quality_count": count,
                "quality": count / n_total,
            }
        )
    return rows


def write_genmol_quality_outputs(
    smiles,
    output_dir,
    denominator,
    qed_threshold=0.6,
    sa_threshold=4.0,
    quality_thresholds="0.6:4,0.5:5,0.7:3",
    fp_nbits=2048,
):
    metrics, unique_rows = evaluate_genmol_quality(
        smiles,
        denominator=denominator,
        qed_threshold=qed_threshold,
        sa_threshold=sa_threshold,
        fp_nbits=fp_nbits,
    )
    threshold_rows = quality_table(
        unique_rows,
        denominator,
        parse_quality_thresholds(quality_thresholds, qed_threshold, sa_threshold),
    )
    metrics["QualityThresholds"] = threshold_rows

    os.makedirs(output_dir, exist_ok=True)
    with open(os.path.join(output_dir, "genmol_denovo_metrics.json"), "w") as f:
        json.dump(metrics, f, indent=2, sort_keys=True)
    with open(os.path.join(output_dir, "genmol_unique_valid_properties.csv"), "w", newline="") as f:
        writer = csv.DictWriter(
            f,
            fieldnames=["smiles", "canonical", "qed", "sa", "quality_pass"],
        )
        writer.writeheader()
        writer.writerows(unique_rows)
    with open(os.path.join(output_dir, "genmol_quality_thresholds.csv"), "w", newline="") as f:
        writer = csv.DictWriter(
            f,
            fieldnames=["qed_threshold", "sa_threshold", "quality_count", "quality"],
        )
        writer.writeheader()
        writer.writerows(threshold_rows)

    lines = [
        f"Total requested: {metrics['TotalRequested']}",
        f"Total generated lines: {metrics['TotalGenerated']}",
        f"Validity: {metrics['Validity'] * 100:.2f}%",
        f"Uniqueness(valid): {metrics['Uniqueness'] * 100:.2f}%",
        f"Uniqueness(total): {metrics['UniquenessTotal'] * 100:.2f}%",
        f"Quality(QED >= {qed_threshold}, SA <= {sa_threshold}): {metrics['Quality'] * 100:.2f}%",
        f"Diversity: {metrics['Diversity']:.4f}",
        f"Mean QED(unique valid): {metrics['MeanQEDUniqueValid']:.4f}",
        f"Mean SA(unique valid): {metrics['MeanSAUniqueValid']:.4f}",
        f"SA scorer: {metrics['SAReader']}",
        "GenMol Quality thresholds:",
        *[
            f"  QED >= {row['qed_threshold']:g}, SA <= {row['sa_threshold']:g}: "
            f"{row['quality'] * 100:.2f}% ({row['quality_count']}/{metrics['TotalRequested']})"
            for row in threshold_rows
        ],
    ]
    with open(os.path.join(output_dir, "genmol_denovo_metrics.txt"), "w") as f:
        f.write("\n".join(lines) + "\n")
    with open(os.path.join(output_dir, "genmol_quality_thresholds.txt"), "w") as f:
        f.write("\n".join(lines[-(len(threshold_rows) + 1):]) + "\n")
    return metrics


def classify_invalid(smi):
    mol = Chem.MolFromSmiles(smi, sanitize=False)
    if mol is None:
        return "parse"
    try:
        Chem.SanitizeMol(mol)
        return None
    except Exception as exc:
        msg = str(exc).lower()
        if "kekul" in msg:
            return "kekulize"
        if "valence" in msg:
            return "valence"
        if "aromatic" in msg:
            return "aromatic"
        if "ring" in msg:
            return "ring"
        return "sanitize_other"


def mol_stats(mol):
    atoms = list(mol.GetAtoms())
    aromatic_atoms = [a for a in atoms if a.GetIsAromatic()]
    atom_rings = mol.GetRingInfo().AtomRings()
    aromatic_rings = 0
    for ring in atom_rings:
        if ring and all(mol.GetAtomWithIdx(idx).GetIsAromatic() for idx in ring):
            aromatic_rings += 1
    return {
        "heavy_atoms": mol.GetNumHeavyAtoms(),
        "mw": float(Descriptors.MolWt(mol)),
        "pure_carbon": bool(atoms) and all(a.GetAtomicNum() == 6 for a in atoms),
        "has_aromatic_atom": bool(aromatic_atoms),
        "aromatic_atom_count": len(aromatic_atoms),
        "ring_count": len(atom_rings),
        "aromatic_ring_count": aromatic_rings,
    }


def summarize_generation_structure(smiles):
    invalid_types = Counter()
    valid_rows = []
    canonical = []

    for smi in smiles:
        mol = Chem.MolFromSmiles(smi)
        if mol is None:
            invalid_types[classify_invalid(smi)] += 1
            continue
        try:
            stats = mol_stats(mol)
            can = Chem.MolToSmiles(mol, canonical=True)
        except Exception:
            invalid_types[classify_invalid(smi) or "postprocess"] += 1
            continue
        stats["smiles"] = smi
        stats["canonical"] = can
        valid_rows.append(stats)
        canonical.append(can)

    n_total = len(smiles)
    n_valid = len(valid_rows)
    n_unique = len(set(canonical))

    def frac(count, denom):
        return float(count) / denom if denom else 0.0

    pure_carbon = sum(r["pure_carbon"] for r in valid_rows)
    aromatic_mol = sum(r["has_aromatic_atom"] for r in valid_rows)
    has_aromatic_ring = sum(r["aromatic_ring_count"] > 0 for r in valid_rows)
    aliphatic_carbon_only = sum(r["pure_carbon"] and not r["has_aromatic_atom"] for r in valid_rows)
    non_pure_valid = [r for r in valid_rows if not r["pure_carbon"]]
    non_pure_unique = len({r["canonical"] for r in non_pure_valid})
    non_aliphatic_carbon_valid = [
        r for r in valid_rows if not (r["pure_carbon"] and not r["has_aromatic_atom"])
    ]
    non_aliphatic_carbon_unique = len({r["canonical"] for r in non_aliphatic_carbon_valid})

    means = {}
    for field in ["heavy_atoms", "mw", "aromatic_atom_count", "ring_count", "aromatic_ring_count"]:
        vals = [r[field] for r in valid_rows]
        means[f"mean_{field}"] = float(np.mean(vals)) if vals else float("nan")

    return {
        "n_total": n_total,
        "n_valid": n_valid,
        "validity": frac(n_valid, n_total),
        "n_unique_valid": n_unique,
        "uniqueness_valid": frac(n_unique, n_valid),
        "n_invalid": n_total - n_valid,
        "invalid_types": dict(invalid_types),
        "pure_carbon_valid": pure_carbon,
        "pure_carbon_valid_frac": frac(pure_carbon, n_valid),
        "aliphatic_carbon_only_valid": aliphatic_carbon_only,
        "aliphatic_carbon_only_valid_frac": frac(aliphatic_carbon_only, n_valid),
        "aromatic_mol_valid": aromatic_mol,
        "aromatic_mol_valid_frac": frac(aromatic_mol, n_valid),
        "aromatic_ring_mol_valid": has_aromatic_ring,
        "aromatic_ring_mol_valid_frac": frac(has_aromatic_ring, n_valid),
        "non_pure_valid": len(non_pure_valid),
        "non_pure_unique_valid": non_pure_unique,
        "non_pure_uniqueness": frac(non_pure_unique, len(non_pure_valid)),
        "non_aliphatic_carbon_valid": len(non_aliphatic_carbon_valid),
        "non_aliphatic_carbon_unique_valid": non_aliphatic_carbon_unique,
        "non_aliphatic_carbon_uniqueness": frac(
            non_aliphatic_carbon_unique,
            len(non_aliphatic_carbon_valid),
        ),
        **means,
    }


def write_generation_structure_summary(smiles, output_dir, input_label="generated_mols.txt"):
    os.makedirs(output_dir, exist_ok=True)
    summary = summarize_generation_structure(smiles)
    json_path = os.path.join(output_dir, "generation_structure_summary.json")
    txt_path = os.path.join(output_dir, "generation_structure_summary.txt")
    with open(json_path, "w") as f:
        json.dump(summary, f, indent=2, sort_keys=True)
    lines = [
        f"Input: {input_label}",
        f"Total: {summary['n_total']}",
        f"Validity: {summary['validity'] * 100:.2f}%",
        f"Uniqueness(valid): {summary['uniqueness_valid'] * 100:.2f}%",
        f"Pure-carbon(valid): {summary['pure_carbon_valid']} "
        f"({summary['pure_carbon_valid_frac'] * 100:.2f}%)",
        f"Aliphatic carbon-only(valid): {summary['aliphatic_carbon_only_valid']} "
        f"({summary['aliphatic_carbon_only_valid_frac'] * 100:.2f}%)",
        f"Aromatic molecule(valid): {summary['aromatic_mol_valid']} "
        f"({summary['aromatic_mol_valid_frac'] * 100:.2f}%)",
        f"Aromatic-ring molecule(valid): {summary['aromatic_ring_mol_valid']} "
        f"({summary['aromatic_ring_mol_valid_frac'] * 100:.2f}%)",
        f"Non-pure-carbon uniqueness: {summary['non_pure_uniqueness'] * 100:.2f}%",
        "Non-aliphatic-carbon uniqueness: "
        f"{summary['non_aliphatic_carbon_uniqueness'] * 100:.2f}%",
        f"Mean aromatic atoms(valid): {summary['mean_aromatic_atom_count']:.3f}",
        f"Mean aromatic rings(valid): {summary['mean_aromatic_ring_count']:.3f}",
        f"Invalid types: {summary['invalid_types']}",
    ]
    with open(txt_path, "w") as f:
        f.write("\n".join(lines) + "\n")
    return summary
