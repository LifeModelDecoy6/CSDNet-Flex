#!/usr/bin/env python
import argparse
import os

from CSDNet.util.metrics import write_genmol_quality_outputs


def main():
    parser = argparse.ArgumentParser(description="Compute GenMol-style de novo Quality metrics.")
    parser.add_argument("--input", required=True, help="generated_mols.txt")
    parser.add_argument("--output_dir", default=None)
    parser.add_argument("--num_samples", type=int, default=0)
    parser.add_argument("--qed_threshold", type=float, default=0.6)
    parser.add_argument("--sa_threshold", type=float, default=4.0)
    parser.add_argument("--quality_thresholds", default="0.6:4,0.5:5,0.7:3")
    parser.add_argument("--fp_nbits", type=int, default=2048)
    args = parser.parse_args()

    with open(args.input) as f:
        smiles = [line.strip() for line in f if line.strip()]

    output_dir = args.output_dir or os.path.dirname(args.input) or "."
    denominator = args.num_samples if args.num_samples > 0 else len(smiles)
    metrics = write_genmol_quality_outputs(
        smiles,
        output_dir=output_dir,
        denominator=denominator,
        qed_threshold=args.qed_threshold,
        sa_threshold=args.sa_threshold,
        quality_thresholds=args.quality_thresholds,
        fp_nbits=args.fp_nbits,
    )
    print(f"Quality: {metrics['Quality'] * 100:.2f}%")
    print(f"Validity: {metrics['Validity'] * 100:.2f}%")
    print(f"Uniqueness(valid): {metrics['Uniqueness'] * 100:.2f}%")
    print(f"Diversity: {metrics['Diversity']:.4f}")


if __name__ == "__main__":
    main()
