#!/usr/bin/env python
import argparse
import os

from CSDNet.util.metrics import write_generation_structure_summary


def main():
    parser = argparse.ArgumentParser(description="Analyze generated SMILES structure classes and invalid reasons.")
    parser.add_argument("--input", required=True)
    parser.add_argument("--output_dir", default=None)
    args = parser.parse_args()

    with open(args.input) as f:
        smiles = [line.strip() for line in f if line.strip()]

    output_dir = args.output_dir or os.path.dirname(args.input) or "."
    summary = write_generation_structure_summary(smiles, output_dir, input_label=args.input)
    print(f"Validity: {summary['validity'] * 100:.2f}%")
    print(f"Pure-carbon(valid): {summary['pure_carbon_valid_frac'] * 100:.2f}%")
    print(f"Aromatic molecule(valid): {summary['aromatic_mol_valid_frac'] * 100:.2f}%")
    print(f"Invalid types: {summary['invalid_types']}")


if __name__ == "__main__":
    main()
