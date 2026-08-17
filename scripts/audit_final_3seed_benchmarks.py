#!/usr/bin/env python3
"""Fast, side-effect-free release audit for the three final benchmarks."""

import argparse
import csv
import json
from pathlib import Path


PMO_TASKS = (
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
)


def require_file(path):
    path = Path(path)
    if not path.is_file() or path.stat().st_size == 0:
        raise SystemExit(f"Missing or empty required file: {path}")
    return path


def require_source_marker(path, marker):
    text = require_file(path).read_text(encoding="utf-8")
    if marker not in text:
        raise SystemExit(f"Missing release marker {marker!r} in {path}")


def audit_checkpoint(path):
    import torch

    checkpoint = torch.load(require_file(path), map_location="cpu", weights_only=False)
    state = checkpoint.get("state_dict", checkpoint)
    keys = tuple(state)
    required = ("theta_insertion_head", "phi_insertion_head")
    missing = [name for name in required if not any(name in key for key in keys)]
    if missing:
        raise SystemExit(
            "Checkpoint is not the required ElasticCSDNet model; missing: "
            + ", ".join(missing)
        )
    print(
        "Checkpoint OK:",
        path,
        f"global_step={checkpoint.get('global_step')}",
        f"epoch={checkpoint.get('epoch')}",
    )


def audit_prior(path, schema):
    payload = json.loads(require_file(path).read_text(encoding="utf-8"))
    if payload.get("schema") != schema:
        raise SystemExit(f"Unexpected schema in {path}: {payload.get('schema')!r}")
    if payload.get("tokenizer") != "csdnet_atomic_smiles":
        raise SystemExit(f"Prior is not atom-tokenized: {path}")
    if int(payload.get("max_len", -1)) != 256:
        raise SystemExit(f"Prior max_len is not 256: {path}")
    print(f"Atomic-token prior OK: {path} ({schema})")


def audit_pmo_priors(root, minimum_rows):
    for task in PMO_TASKS:
        path = require_file(root / "CSDNet" / "exp" / "pmo" / "vocab" / f"{task}.csv")
        with path.open(newline="", encoding="utf-8") as handle:
            reader = csv.DictReader(handle)
            if not {"frag", "score"}.issubset(reader.fieldnames or ()):
                raise SystemExit(f"Invalid PMO prior columns: {path}")
            rows = sum(1 for _ in reader)
        if rows < minimum_rows:
            raise SystemExit(
                f"PMO prior has {rows} rows, requires {minimum_rows}: {path}"
            )
    print(f"PMO priors OK: {len(PMO_TASKS)} tasks, >= {minimum_rows} rows each")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--root", type=Path, default=Path.cwd())
    parser.add_argument("--checkpoint", required=True)
    parser.add_argument("--atomic_length_prior", required=True)
    parser.add_argument("--fragment_gap_prior", required=True)
    parser.add_argument("--minimum_pmo_rows", type=int, default=2000)
    args = parser.parse_args()
    root = args.root.resolve()

    audit_checkpoint(root / args.checkpoint)
    audit_prior(
        root / args.atomic_length_prior,
        "csdnet_atomic_smiles_length_prior_v1",
    )
    audit_prior(
        root / args.fragment_gap_prior,
        "csdnet_fragment_gap_length_prior_v1",
    )
    require_file(root / "csdnet_vocab.pkl")
    require_file(root / "data" / "fragments.csv")
    audit_pmo_priors(root, args.minimum_pmo_rows)

    require_source_marker(
        root / "CSDNet" / "exp" / "frag" / "direct_infill.py",
        '"structural_feasible"',
    )
    require_source_marker(
        root / "CSDNet" / "util" / "elastic_sampling.py",
        "online_scan_progressive_completion_neural_then_projection",
    )
    require_source_marker(
        root / "CSDNet" / "exp" / "lead" / "run.py",
        "_elastic_joint_v4_legacy_fraction",
    )
    require_source_marker(
        root / "CSDNet" / "exp" / "pmo" / "optimizer.py",
        "elastic_prescreen_rank_tier_weights",
    )
    print("FINAL AUDIT PASSED: Frag + Lead + PMO release assets")


if __name__ == "__main__":
    main()
