#!/usr/bin/env python
"""Build a benchmark-independent fragment gap-length prior from ZINC250K."""

from __future__ import annotations

import argparse
import hashlib
import itertools
import json
import pickle
import random
from collections import Counter
from pathlib import Path

import pandas as pd
from rdkit import Chem
from rdkit.Chem import BRICS
from rdkit.Chem.Scaffolds import MurckoScaffold

from CSDNet.exp.frag.fragment_length_prior import (
    FRAGMENT_GAP_PRIOR_SCHEMA,
    SUPPORTED_GEOMETRIES,
)
from CSDNet.util.tokenizer import SMILESTokenizer, tokenize_smiles


class PriorAccumulator:
    def __init__(self, bin_width: int):
        self.bin_width = int(bin_width)
        self.geometries = {
            geometry: {"groups": {}, "gap_groups": {}, "global": self._empty()}
            for geometry in SUPPORTED_GEOMETRIES
        }

    @staticmethod
    def _empty():
        return {
            "count": 0,
            "token_histogram": Counter(),
            "atom_histogram": Counter(),
        }

    @staticmethod
    def _update(group, token_count: int, atom_count: int):
        group["count"] += 1
        group["token_histogram"][int(token_count)] += 1
        group["atom_histogram"][int(atom_count)] += 1

    def add(
        self,
        geometry: str,
        *,
        fixed_tokens: int,
        gap_count: int,
        missing_tokens: int,
        missing_atoms: int,
    ):
        fixed_tokens = max(0, int(fixed_tokens))
        gap_count = max(1, int(gap_count))
        missing_tokens = max(gap_count, int(missing_tokens))
        missing_atoms = max(gap_count, int(missing_atoms))
        data = self.geometries[geometry]
        fixed_bin = (fixed_tokens // self.bin_width) * self.bin_width
        key = f"{fixed_bin}:{gap_count}"
        group = data["groups"].setdefault(key, self._empty())
        gap_group = data["gap_groups"].setdefault(
            str(gap_count), self._empty()
        )
        for target in (group, gap_group, data["global"]):
            self._update(target, missing_tokens, missing_atoms)

    @staticmethod
    def _serialise_group(group):
        return {
            "count": int(group["count"]),
            "token_histogram": {
                str(value): int(count)
                for value, count in sorted(group["token_histogram"].items())
            },
            "atom_histogram": {
                str(value): int(count)
                for value, count in sorted(group["atom_histogram"].items())
            },
        }

    def serialise(self):
        output = {}
        for geometry, data in self.geometries.items():
            output[geometry] = {
                "groups": {
                    key: self._serialise_group(group)
                    for key, group in sorted(data["groups"].items())
                },
                "gap_groups": {
                    key: self._serialise_group(group)
                    for key, group in sorted(data["gap_groups"].items())
                },
                "global": self._serialise_group(data["global"]),
            }
        return output


def _stable_rng(smiles: str, seed: int) -> random.Random:
    digest = hashlib.sha256(f"{seed}:{smiles}".encode()).digest()
    return random.Random(int.from_bytes(digest[:8], "big"))


def _candidate_bonds(mol: Chem.Mol) -> list[int]:
    indices = []
    for (left, right), _ in BRICS.FindBRICSBonds(mol):
        bond = mol.GetBondBetweenAtoms(int(left), int(right))
        if bond is not None:
            indices.append(int(bond.GetIdx()))
    if indices:
        return sorted(set(indices))
    return [
        int(bond.GetIdx())
        for bond in mol.GetBonds()
        if not bond.IsInRing()
        and bond.GetBondType() == Chem.BondType.SINGLE
        and bond.GetBeginAtom().GetAtomicNum() > 0
        and bond.GetEndAtom().GetAtomicNum() > 0
    ]


def _groups_after_cuts(mol: Chem.Mol, bond_indices) -> list[tuple[int, ...]]:
    editable = Chem.RWMol(mol)
    for bond_index in bond_indices:
        bond = mol.GetBondWithIdx(int(bond_index))
        editable.RemoveBond(bond.GetBeginAtomIdx(), bond.GetEndAtomIdx())
    return [
        tuple(int(index) for index in group)
        for group in Chem.GetMolFrags(
            editable.GetMol(),
            asMols=False,
            sanitizeFrags=False,
        )
    ]


def _component_stats(mol, atoms, vocab) -> tuple[int, int] | None:
    atoms = tuple(sorted({int(index) for index in atoms}))
    if not atoms:
        return None
    try:
        smiles = Chem.MolFragmentToSmiles(
            mol,
            atomsToUse=list(atoms),
            canonical=True,
            isomericSmiles=False,
        )
    except Exception:
        return None
    tokens = tokenize_smiles(smiles)
    if not tokens or any(token not in vocab for token in tokens):
        return None
    heavy_atoms = sum(
        mol.GetAtomWithIdx(index).GetAtomicNum() > 0 for index in atoms
    )
    if heavy_atoms <= 0:
        return None
    return len(tokens), int(heavy_atoms)


def _record_terminal(mol, bonds, vocab, accumulator, rng):
    if not bonds:
        return
    bond_index = int(rng.choice(bonds))
    groups = _groups_after_cuts(mol, [bond_index])
    if len(groups) != 2:
        return
    stats = [_component_stats(mol, group, vocab) for group in groups]
    if any(item is None for item in stats):
        return
    for fixed, missing in ((stats[0], stats[1]), (stats[1], stats[0])):
        accumulator.add(
            "single_attachment",
            fixed_tokens=fixed[0],
            gap_count=1,
            missing_tokens=missing[0],
            missing_atoms=missing[1],
        )


def _record_linker(mol, bonds, vocab, accumulator, rng):
    if len(bonds) < 2:
        return
    pairs = list(itertools.combinations(bonds, 2))
    rng.shuffle(pairs)
    for pair in pairs[:16]:
        groups = _groups_after_cuts(mol, pair)
        if len(groups) != 3:
            continue
        atom_to_group = {
            atom: group_index
            for group_index, group in enumerate(groups)
            for atom in group
        }
        incidences = [0, 0, 0]
        for bond_index in pair:
            bond = mol.GetBondWithIdx(int(bond_index))
            touched = {
                atom_to_group.get(int(bond.GetBeginAtomIdx())),
                atom_to_group.get(int(bond.GetEndAtomIdx())),
            }
            for group_index in touched:
                if group_index is not None:
                    incidences[group_index] += 1
        middle = [index for index, count in enumerate(incidences) if count == 2]
        if len(middle) != 1:
            continue
        middle_index = middle[0]
        missing = _component_stats(mol, groups[middle_index], vocab)
        sides = [
            _component_stats(mol, group, vocab)
            for index, group in enumerate(groups)
            if index != middle_index
        ]
        if missing is None or any(side is None for side in sides):
            continue
        accumulator.add(
            "multi_anchor",
            fixed_tokens=sum(side[0] for side in sides),
            gap_count=1,
            missing_tokens=missing[0],
            missing_atoms=missing[1],
        )
        return


def _outside_components(mol, core_atoms):
    core_atoms = set(int(index) for index in core_atoms)
    outside = {
        atom.GetIdx() for atom in mol.GetAtoms() if atom.GetIdx() not in core_atoms
    }
    components = []
    while outside:
        start = outside.pop()
        component = {start}
        stack = [start]
        while stack:
            index = stack.pop()
            atom = mol.GetAtomWithIdx(index)
            for neighbor in atom.GetNeighbors():
                neighbor_index = neighbor.GetIdx()
                if neighbor_index in outside:
                    outside.remove(neighbor_index)
                    component.add(neighbor_index)
                    stack.append(neighbor_index)
        if any(
            neighbor.GetIdx() in core_atoms
            for index in component
            for neighbor in mol.GetAtomWithIdx(index).GetNeighbors()
        ):
            components.append(tuple(sorted(component)))
    return components


def _record_scaffold_geometries(mol, vocab, accumulator, max_gap_count):
    scaffold = MurckoScaffold.GetScaffoldForMol(mol)
    if scaffold is None or scaffold.GetNumHeavyAtoms() == 0:
        return
    match = mol.GetSubstructMatch(scaffold)
    if not match or len(match) >= mol.GetNumAtoms():
        return
    fixed = _component_stats(mol, match, vocab)
    components = _outside_components(mol, match)
    component_stats = [
        _component_stats(mol, component, vocab) for component in components
    ]
    component_stats = [item for item in component_stats if item is not None]
    if fixed is None or not component_stats:
        return
    missing_tokens = sum(item[0] for item in component_stats)
    missing_atoms = sum(item[1] for item in component_stats)
    accumulator.add(
        "substructure_expand",
        fixed_tokens=fixed[0],
        gap_count=1,
        missing_tokens=missing_tokens,
        missing_atoms=missing_atoms,
    )
    if len(component_stats) >= 2:
        component_stats = sorted(component_stats, reverse=True)[:max_gap_count]
        accumulator.add(
            "multi_attachment",
            fixed_tokens=fixed[0],
            gap_count=len(component_stats),
            missing_tokens=sum(item[0] for item in component_stats),
            missing_atoms=sum(item[1] for item in component_stats),
        )


def build(args):
    input_path = Path(args.input)
    output_path = Path(args.output)
    with Path(args.vocab).open("rb") as handle:
        tokenizer = SMILESTokenizer(pickle.load(handle))
    accumulator = PriorAccumulator(args.fixed_bin_width)
    frame = pd.read_csv(input_path, usecols=[args.smiles_col])
    if args.max_molecules > 0:
        frame = frame.iloc[: args.max_molecules]

    valid_rows = 0
    tokenizable_rows = 0
    for row_index, value in enumerate(frame[args.smiles_col]):
        mol = Chem.MolFromSmiles(str(value))
        if mol is None:
            continue
        valid_rows += 1
        Chem.RemoveStereochemistry(mol)
        smiles = Chem.MolToSmiles(mol, canonical=True, isomericSmiles=False)
        tokens = tokenize_smiles(smiles)
        if (
            not tokens
            or len(tokens) + 2 > args.max_len
            or any(token not in tokenizer.vocab for token in tokens)
        ):
            continue
        tokenizable_rows += 1
        rng = _stable_rng(smiles, args.seed)
        bonds = _candidate_bonds(mol)
        _record_terminal(mol, bonds, tokenizer.vocab, accumulator, rng)
        _record_linker(mol, bonds, tokenizer.vocab, accumulator, rng)
        _record_scaffold_geometries(
            mol,
            tokenizer.vocab,
            accumulator,
            args.max_gap_count,
        )
        if args.progress_every and (row_index + 1) % args.progress_every == 0:
            print(
                f"Processed {row_index + 1}/{len(frame)} rows; "
                f"tokenizable={tokenizable_rows}",
                flush=True,
            )

    source_md5 = hashlib.md5(input_path.read_bytes()).hexdigest()
    payload = {
        "schema": FRAGMENT_GAP_PRIOR_SCHEMA,
        "tokenizer": "csdnet_atomic_smiles",
        "source": str(input_path),
        "source_md5": source_md5,
        "source_rows": int(len(frame)),
        "valid_rows": int(valid_rows),
        "tokenizable_rows": int(tokenizable_rows),
        "max_len": int(args.max_len),
        "fixed_bin_width": int(args.fixed_bin_width),
        "minimum_group_count": int(args.minimum_group_count),
        "seed": int(args.seed),
        "geometries": accumulator.serialise(),
    }
    for geometry in SUPPORTED_GEOMETRIES:
        count = payload["geometries"][geometry]["global"]["count"]
        if count < args.minimum_geometry_count:
            raise RuntimeError(
                f"Only {count} usable observations for {geometry}; "
                f"minimum is {args.minimum_geometry_count}"
            )
    output_path.parent.mkdir(parents=True, exist_ok=True)
    temporary = output_path.with_suffix(output_path.suffix + ".tmp")
    temporary.write_text(json.dumps(payload, indent=2, sort_keys=True))
    temporary.replace(output_path)
    print(json.dumps({
        "output": str(output_path),
        "source_rows": len(frame),
        "valid_rows": valid_rows,
        "tokenizable_rows": tokenizable_rows,
        "geometry_counts": {
            geometry: payload["geometries"][geometry]["global"]["count"]
            for geometry in SUPPORTED_GEOMETRIES
        },
    }, indent=2))


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--input", default="data/zinc250k.csv")
    parser.add_argument("--smiles_col", default="smiles")
    parser.add_argument("--vocab", default="csdnet_vocab.pkl")
    parser.add_argument(
        "--output",
        default="data/zinc250k_fragment_gap_prior_atom256.json",
    )
    parser.add_argument("--max_len", type=int, default=256)
    parser.add_argument("--fixed_bin_width", type=int, default=8)
    parser.add_argument("--minimum_group_count", type=int, default=32)
    parser.add_argument("--minimum_geometry_count", type=int, default=100)
    parser.add_argument("--max_gap_count", type=int, default=8)
    parser.add_argument("--max_molecules", type=int, default=0)
    parser.add_argument("--progress_every", type=int, default=10000)
    parser.add_argument("--seed", type=int, default=42)
    return parser.parse_args()


if __name__ == "__main__":
    build(parse_args())
