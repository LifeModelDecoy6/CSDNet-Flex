#!/usr/bin/env python
import argparse
import csv
import json
import os
import sys

import numpy as np
from rdkit import Chem, RDConfig, RDLogger
from rdkit.Chem import AllChem, DataStructs, QED


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


def evaluate(smiles, denominator, qed_threshold=0.6, sa_threshold=4.0, fp_nbits=2048):
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
                raise SystemExit(
                    "Quality threshold 格式应类似 '0.6:4,0.5:5,0.7:3'。"
                )
            thresholds.append((float(left), float(right)))

    deduped = []
    seen = set()
    for qed_threshold, sa_threshold in thresholds:
        key = (round(qed_threshold, 8), round(sa_threshold, 8))
        if key in seen:
            continue
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


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--input", required=True, help="generated_mols.txt")
    parser.add_argument("--output_dir", default=None)
    parser.add_argument("--num_samples", type=int, default=0,
                        help="GenMol style denominator; default uses number of lines in input")
    parser.add_argument("--qed_threshold", type=float, default=0.6)
    parser.add_argument("--sa_threshold", type=float, default=4.0)
    parser.add_argument(
        "--quality_thresholds",
        type=str,
        default="0.6:4,0.5:5,0.7:3",
        help="Comma-separated GenMol Quality thresholds, e.g. 0.6:4,0.5:5,0.7:3",
    )
    parser.add_argument("--fp_nbits", type=int, default=2048)
    args = parser.parse_args()

    with open(args.input) as f:
        smiles = [line.strip() for line in f if line.strip()]

    denominator = args.num_samples if args.num_samples > 0 else len(smiles)
    metrics, unique_rows = evaluate(
        smiles,
        denominator=denominator,
        qed_threshold=args.qed_threshold,
        sa_threshold=args.sa_threshold,
        fp_nbits=args.fp_nbits,
    )
    threshold_rows = quality_table(
        unique_rows,
        denominator,
        parse_quality_thresholds(
            args.quality_thresholds,
            args.qed_threshold,
            args.sa_threshold,
        ),
    )
    metrics["QualityThresholds"] = threshold_rows

    out_dir = args.output_dir or os.path.dirname(args.input) or "."
    os.makedirs(out_dir, exist_ok=True)

    json_path = os.path.join(out_dir, "genmol_denovo_metrics.json")
    txt_path = os.path.join(out_dir, "genmol_denovo_metrics.txt")
    csv_path = os.path.join(out_dir, "genmol_unique_valid_properties.csv")
    threshold_csv_path = os.path.join(out_dir, "genmol_quality_thresholds.csv")
    threshold_txt_path = os.path.join(out_dir, "genmol_quality_thresholds.txt")

    with open(json_path, "w") as f:
        json.dump(metrics, f, indent=2, sort_keys=True)

    with open(csv_path, "w", newline="") as f:
        writer = csv.DictWriter(
            f,
            fieldnames=["smiles", "canonical", "qed", "sa", "quality_pass"],
        )
        writer.writeheader()
        writer.writerows(unique_rows)

    with open(threshold_csv_path, "w", newline="") as f:
        writer = csv.DictWriter(
            f,
            fieldnames=["qed_threshold", "sa_threshold", "quality_count", "quality"],
        )
        writer.writeheader()
        writer.writerows(threshold_rows)

    lines = [
        f"Input: {args.input}",
        f"Total requested: {metrics['TotalRequested']}",
        f"Total generated lines: {metrics['TotalGenerated']}",
        f"Validity: {metrics['Validity'] * 100:.2f}%",
        f"Uniqueness(valid): {metrics['Uniqueness'] * 100:.2f}%",
        f"Uniqueness(total): {metrics['UniquenessTotal'] * 100:.2f}%",
        f"Quality(QED >= {args.qed_threshold}, SA <= {args.sa_threshold}): "
        f"{metrics['Quality'] * 100:.2f}%",
        f"Diversity: {metrics['Diversity']:.4f}",
        f"Mean QED(unique valid): {metrics['MeanQEDUniqueValid']:.4f}",
        f"Mean SA(unique valid): {metrics['MeanSAUniqueValid']:.4f}",
        f"SA scorer: {metrics['SAReader']}",
    ]
    threshold_lines = [
        "GenMol Quality thresholds:",
        *[
            "  "
            f"QED >= {row['qed_threshold']:g}, SA <= {row['sa_threshold']:g}: "
            f"{row['quality'] * 100:.2f}% ({row['quality_count']}/{metrics['TotalRequested']})"
            for row in threshold_rows
        ],
    ]
    lines.extend(threshold_lines)
    with open(txt_path, "w") as f:
        f.write("\n".join(lines) + "\n")
    with open(threshold_txt_path, "w") as f:
        f.write("\n".join(threshold_lines) + "\n")

    print("\n".join(lines))
    print(f"Saved: {json_path}")
    print(f"Saved: {txt_path}")
    print(f"Saved: {csv_path}")
    print(f"Saved: {threshold_csv_path}")
    print(f"Saved: {threshold_txt_path}")


if __name__ == "__main__":
    main()
