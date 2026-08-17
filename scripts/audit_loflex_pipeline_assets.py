#!/usr/bin/env python3
"""Fail fast when the LoFlex 6M pipeline deployment is incomplete."""

from __future__ import annotations

import csv
import importlib
import importlib.util
import pickle
import sys
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))


REQUIRED_FILES = (
    "run_csdnet_6m_loflex_geometric_zinc10_pipeline_4xrtx6000.pbs",
    "train_csdnet_elastic_hf_streaming.py",
    "train_csdnet_hf_streaming.py",
    "evaluate_genmol_denovo_metrics.py",
    "csdnet_hf_smiles.py",
    "csdnet_tokenizer.py",
    "csdnet_vocab.pkl",
    "scripts/audit_loflex_aligned_6m.py",
    "scripts/prepare_zinc250k.sh",
    "scripts/run_loflex_geometric_posttrain_eval.sh",
    "data/fragments.csv",
    "data/zinc250k.csv",
)

REQUIRED_LOCAL_MODULES = (
    "train_csdnet_elastic_hf_streaming",
    "evaluate_genmol_denovo_metrics",
    "CSDNet.model.elastic_backbone",
    "CSDNet.model.elastic_lightning_module",
    "CSDNet.model.elastic_schedule",
    "CSDNet.model.unified_backbone",
    "CSDNet.util.checkpoint",
    "CSDNet.util.elastic_sampling",
    "CSDNet.util.unified_sampling",
    "CSDNet.util.sampling",
    "CSDNet.util.fsm",
    "CSDNet.exp.denovo.evaluate",
    "CSDNet.exp.denovo.aggregate_promax",
    "CSDNet.exp.frag.run_unified_frontier",
    "CSDNet.exp.frag.aggregate_frontier",
)

REQUIRED_EXTERNAL_MODULES = (
    "torch",
    "lightning",
    "transformers",
    "datasets",
    "safe",
    "rdkit",
    "datamol",
    "numpy",
    "pandas",
)


def check_files(failures):
    for relative in REQUIRED_FILES:
        path = ROOT / relative
        valid = path.is_file() and path.stat().st_size > 0
        print(f"{'OK' if valid else 'FAIL':4s} file {relative}")
        if not valid:
            failures.append(f"file:{relative}")


def check_module_specs(modules, label, failures):
    for name in modules:
        try:
            found = importlib.util.find_spec(name) is not None
        except (ImportError, ModuleNotFoundError):
            found = False
        print(f"{'OK' if found else 'FAIL':4s} {label} module {name}")
        if not found:
            failures.append(f"module:{name}")


def import_entrypoints(failures):
    for name in REQUIRED_LOCAL_MODULES:
        try:
            importlib.import_module(name)
        except Exception as exc:
            print(f"FAIL import {name}: {type(exc).__name__}: {exc}")
            failures.append(f"import:{name}")
        else:
            print(f"OK   import {name}")


def check_assets(failures):
    vocab_path = ROOT / "csdnet_vocab.pkl"
    if vocab_path.is_file():
        try:
            with vocab_path.open("rb") as handle:
                vocab = pickle.load(handle)
            required = {"<pad>", "<mask>", "<bos>", "<eos>", "C", "N", "Cl"}
            missing = sorted(required.difference(vocab))
            if missing:
                raise ValueError(f"missing tokens {missing}")
            print(f"OK   vocabulary entries={len(vocab)}")
        except Exception as exc:
            print(f"FAIL vocabulary: {type(exc).__name__}: {exc}")
            failures.append("asset:vocabulary")

    fragments_path = ROOT / "data/fragments.csv"
    if fragments_path.is_file():
        try:
            with fragments_path.open(newline="", encoding="utf-8") as handle:
                reader = csv.DictReader(handle)
                fields = set(reader.fieldnames or ())
                rows = sum(1 for _ in reader)
            expected = {
                "name",
                "smiles",
                "linker_design",
                "motif_extension",
                "scaffold_decoration",
                "superstructure_generation",
            }
            if not expected.issubset(fields) or rows < 1:
                raise ValueError(f"fields={sorted(fields)}, rows={rows}")
            print(f"OK   fragment benchmark rows={rows}")
        except Exception as exc:
            print(f"FAIL fragment benchmark: {type(exc).__name__}: {exc}")
            failures.append("asset:fragments")

    zinc_path = ROOT / "data/zinc250k.csv"
    if zinc_path.is_file():
        try:
            with zinc_path.open(newline="", encoding="utf-8") as handle:
                reader = csv.DictReader(handle)
                fields = {field.strip().lower() for field in (reader.fieldnames or ())}
                rows = sum(1 for _ in reader)
            if "smiles" not in fields or rows != 249455:
                raise ValueError(f"fields={sorted(fields)}, rows={rows}")
            print(f"OK   ZINC250K rows={rows}")
        except Exception as exc:
            print(f"FAIL ZINC250K: {type(exc).__name__}: {exc}")
            failures.append("asset:zinc250k")


def main():
    failures = []
    check_files(failures)
    check_module_specs(REQUIRED_LOCAL_MODULES, "local", failures)
    check_module_specs(REQUIRED_EXTERNAL_MODULES, "external", failures)
    import_entrypoints(failures)
    check_assets(failures)

    if failures:
        raise SystemExit("PIPELINE ASSET AUDIT FAILED:\n  " + "\n  ".join(failures))
    print("PIPELINE ASSET AUDIT PASSED: training and evaluation deployment is complete.")


if __name__ == "__main__":
    main()
