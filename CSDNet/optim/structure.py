"""Task-agnostic structure-preserving edit plans for token diffusion."""

from __future__ import annotations

import math

from rdkit import Chem
from rdkit.Chem.Scaffolds import MurckoScaffold

from CSDNet.util.tokenizer import tokenize_smiles


_NON_ATOM_TOKENS = {
    "(",
    ")",
    ".",
    "=",
    "#",
    "-",
    "+",
    "\\",
    "/",
    ":",
    "~",
    "@",
    "?",
    ">",
    ">>",
    "$",
}


def is_atom_token(token):
    if not token or token in _NON_ATOM_TOKENS:
        return False
    if token.isdigit() or (token.startswith("%") and token[1:].isdigit()):
        return False
    return token == "*" or token.startswith("[") or token[0].isalpha()


def _component_without_bond(mol, start_atom, blocked_bond_idx):
    visited = {int(start_atom)}
    stack = [int(start_atom)]
    while stack:
        atom_idx = stack.pop()
        atom = mol.GetAtomWithIdx(atom_idx)
        for bond in atom.GetBonds():
            if bond.GetIdx() == blocked_bond_idx:
                continue
            other = bond.GetOtherAtomIdx(atom_idx)
            if other not in visited:
                visited.add(other)
                stack.append(other)
    return visited


def _atom_token_positions(smiles, mol):
    tokens = tokenize_smiles(smiles)
    positions = [idx for idx, token in enumerate(tokens) if is_atom_token(token)]
    if len(positions) != mol.GetNumAtoms():
        return tokens, None
    return tokens, {
        atom_idx: token_idx for atom_idx, token_idx in enumerate(positions)
    }


def murcko_scaffold(smiles):
    mol = Chem.MolFromSmiles(smiles)
    if mol is None:
        return None
    scaffold = MurckoScaffold.GetScaffoldForMol(mol)
    if scaffold is None or scaffold.GetNumAtoms() == 0:
        return None
    return scaffold


def preserves_murcko_scaffold(parent_smiles, child_smiles):
    """Check the core frozen by a peripheral plan survived decoding."""
    scaffold = murcko_scaffold(parent_smiles)
    if scaffold is None:
        return True
    child = Chem.MolFromSmiles(child_smiles)
    return child is not None and child.HasSubstructMatch(scaffold)


def adaptive_peripheral_edit_plan(
    smiles,
    rng,
    delta=0,
    target_atom_fraction=0.15,
    max_atom_fraction=0.55,
    max_span_tokens=None,
):
    """Mask a terminal component while freezing the Murcko/ring core."""
    mol = Chem.MolFromSmiles(smiles)
    if mol is None or mol.GetNumAtoms() < 2:
        return None
    tokens, atom_positions = _atom_token_positions(smiles, mol)
    if atom_positions is None:
        return None

    scaffold = MurckoScaffold.GetScaffoldForMol(mol)
    core_atoms = set()
    if scaffold is not None and scaffold.GetNumAtoms() > 0:
        core_atoms = set(mol.GetSubstructMatch(scaffold))
    if not core_atoms:
        core_atoms = {
            atom.GetIdx() for atom in mol.GetAtoms() if atom.IsInRing()
        }

    all_atoms = set(range(mol.GetNumAtoms()))
    max_component_atoms = max(
        1, int(math.ceil(mol.GetNumAtoms() * float(max_atom_fraction)))
    )
    target_atoms = max(
        1, int(round(mol.GetNumAtoms() * float(target_atom_fraction)))
    )
    plans = []
    for bond in mol.GetBonds():
        if bond.IsInRing() or bond.GetBondType() != Chem.BondType.SINGLE:
            continue
        left = _component_without_bond(
            mol, bond.GetBeginAtomIdx(), bond.GetIdx()
        )
        right = all_atoms - left
        components = []
        if core_atoms:
            if left.isdisjoint(core_atoms) and core_atoms.issubset(right):
                components.append(left)
            if right.isdisjoint(core_atoms) and core_atoms.issubset(left):
                components.append(right)
        else:
            components.append(left if len(left) <= len(right) else right)

        for component in components:
            if not component or len(component) > max_component_atoms:
                continue
            positions = sorted(atom_positions[idx] for idx in component)
            start, stop = positions[0], positions[-1] + 1
            represented = {
                atom_idx
                for atom_idx, token_idx in atom_positions.items()
                if start <= token_idx < stop
            }
            if represented != component:
                continue
            if max_span_tokens is not None and stop - start > max_span_tokens:
                continue
            if "." in tokens[start:stop]:
                continue
            plans.append(
                {
                    "start": start,
                    "stop": stop,
                    "delta": int(delta),
                    "peripheral": True,
                    "component_atoms": len(component),
                }
            )

    if not plans:
        return None
    scale = max(1.0, target_atoms * 0.5)
    weights = [
        math.exp(-abs(plan["component_atoms"] - target_atoms) / scale)
        for plan in plans
    ]
    return rng.choices(plans, weights=weights, k=1)[0]


def atom_span_edit_plan(smiles, rng, delta=0, span_tokens=4):
    """Fallback contiguous atom-centered plan for molecules without side chains."""
    tokens = tokenize_smiles(smiles)
    if not tokens:
        return None
    atom_positions = [
        idx for idx, token in enumerate(tokens) if is_atom_token(token)
    ]
    if not atom_positions:
        return None
    span_tokens = max(1, min(int(span_tokens), len(tokens)))
    centers = list(atom_positions)
    rng.shuffle(centers)
    for center in centers:
        low = max(0, center - span_tokens + 1)
        high = min(center, len(tokens) - span_tokens)
        start = rng.randint(low, high) if high >= low else low
        stop = start + span_tokens
        if "." not in tokens[start:stop]:
            return {"start": start, "stop": stop, "delta": int(delta)}
    return None
