"""Task-agnostic frontier utilities for constrained lead optimization."""

import math

from rdkit import Chem
from rdkit.Chem import rdFMCS
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


def constraint_state(
    dock,
    qed,
    sa,
    similarity,
    start_dock,
    similarity_threshold,
    docking_margin=0.0,
    residual_l1_weight=0.0,
):
    """Return strict checks and an augmented-Chebyshev search residual.

    The benchmark accepts any docking score strictly above ``start_dock``. A
    positive search margin keeps equality from appearing fully feasible to the
    optimizer while leaving that official success rule unchanged.
    """
    start_dock = max(float(start_dock), 1e-8)
    docking_target = start_dock + max(0.0, float(docking_margin))
    similarity_threshold = max(float(similarity_threshold), 1e-8)
    normalized = {
        "dock": float(dock) / docking_target,
        "qed": float(qed) / 0.6,
        "sa": float(sa) / (6.0 / 9.0),
        "sim": float(similarity) / similarity_threshold,
    }
    deficits = {key: max(0.0, 1.0 - value) for key, value in normalized.items()}
    max_deficit = max(deficits.values())
    mean_deficit = sum(deficits.values()) / len(deficits)
    residual_l1_weight = max(0.0, float(residual_l1_weight))
    checks = {
        "dock": float(dock) > float(start_dock),
        "qed": float(qed) >= 0.6,
        "sa": float(sa) >= 6.0 / 9.0,
        "sim": float(similarity) >= float(similarity_threshold),
    }
    return {
        "normalized": normalized,
        "deficits": deficits,
        "residual": max_deficit + residual_l1_weight * mean_deficit,
        "max_deficit": max_deficit,
        "mean_deficit": mean_deficit,
        "bottleneck": max(deficits, key=deficits.get),
        "docking_target": docking_target,
        "stage": sum(checks.values()),
        "checks": checks,
        "strict": all(checks.values()),
    }


def upper_tail_reward(
    rewards,
    tail_fraction=0.20,
    min_tail=2,
    mean_weight=0.20,
):
    """Aggregate one operator batch for a best-of-budget objective.

    The upper-tail mean rewards operators that occasionally produce useful lead
    candidates. A small whole-batch mean term prevents one isolated outlier from
    completely determining future allocation.
    """
    values = [float(value) for value in rewards]
    if not values:
        return 0.0
    tail_fraction = min(1.0, max(0.0, float(tail_fraction)))
    mean_weight = min(1.0, max(0.0, float(mean_weight)))
    min_tail = max(1, int(min_tail))
    tail_size = max(min_tail, int(math.ceil(len(values) * tail_fraction)))
    tail_size = min(len(values), tail_size)
    ordered = sorted(values, reverse=True)
    tail_mean = sum(ordered[:tail_size]) / tail_size
    batch_mean = sum(values) / len(values)
    return (1.0 - mean_weight) * tail_mean + mean_weight * batch_mean


def pair_frontier_names(item):
    """Return the pairwise feasible regions occupied by one candidate."""
    checks = item["checks"]
    quality_ok = checks["qed"] and checks["sa"]
    labels = set()
    if checks["sim"] and quality_ok:
        labels.add("sq")
    if checks["dock"] and quality_ok:
        labels.add("qd")
    if checks["sim"] and checks["dock"]:
        labels.add("sd")
    return labels


def transition_reward(
    parent,
    child,
    crossing_bonus=0.12,
    regression_penalty=0.14,
    pair_bonus=0.12,
    strict_bonus=0.45,
    mean_deficit_weight=0.25,
):
    """Score a proposal by task-agnostic progress toward joint feasibility.

    Improving an already-good scalar score is useful, but crossing a benchmark
    constraint or entering a new pairwise feasible region is more valuable.
    Conversely, losing a constraint that the parent already satisfied is
    explicitly penalized. This keeps all decisions tied to the four benchmark
    constraints rather than to target identities.
    """
    parent_checks = parent["checks"]
    child_checks = child["checks"]
    crossed = sum(
        not parent_checks[name] and child_checks[name]
        for name in parent_checks
    )
    regressed = sum(
        parent_checks[name] and not child_checks[name]
        for name in parent_checks
    )
    pair_gain = len(pair_frontier_names(child) - pair_frontier_names(parent))
    residual_gain = float(parent["residual"]) - float(child["residual"])
    mean_deficit_gain = float(parent["mean_deficit"]) - float(child["mean_deficit"])
    reward = (
        residual_gain
        + max(0.0, float(mean_deficit_weight)) * mean_deficit_gain
        + max(0.0, float(crossing_bonus)) * crossed
        - max(0.0, float(regression_penalty)) * regressed
        + max(0.0, float(pair_bonus)) * pair_gain
    )
    if child["strict"]:
        reward += max(0.0, float(strict_bonus))
    return {
        "reward": reward,
        "residual_gain": residual_gain,
        "mean_deficit_gain": mean_deficit_gain,
        "crossed": crossed,
        "regressed": regressed,
        "pair_gain": pair_gain,
    }


def archive_constraint_need(
    items,
    deficit_keys,
    *,
    joint=False,
    top_k=20,
    readiness_weight=0.35,
    residual_scale=4.0,
):
    """Estimate how useful an operator is from its current source archive.

    The estimate combines the unresolved target deficit with archive readiness.
    A joint operator is rewarded only when at least two constraints remain
    unresolved, so it does not crowd out a precise one-constraint repair.
    """
    top = list(items)[: max(1, int(top_k))]
    if not top:
        return 0.0
    keys = tuple(deficit_keys)
    weights = [1.0 / math.sqrt(rank + 1.0) for rank in range(len(top))]
    need_values = []
    for item in top:
        values = sorted(
            (max(0.0, float(item["deficits"].get(key, 0.0))) for key in keys),
            reverse=True,
        )
        if joint:
            need = sum(values[:2]) / 2.0 if len(values) >= 2 else 0.0
        else:
            need = max(values, default=0.0)
        need_values.append(need)
    weighted_need = sum(
        weight * value for weight, value in zip(weights, need_values)
    ) / sum(weights)
    best_residual = min(max(0.0, float(item["residual"])) for item in top)
    readiness = math.exp(-max(0.0, float(residual_scale)) * best_residual)
    return weighted_need + max(0.0, float(readiness_weight)) * readiness


def frontier_labels(item, similarity_threshold, similarity_slack):
    checks = item["checks"]
    labels = []
    if item["sim"] >= max(0.0, similarity_threshold - similarity_slack):
        labels.append("s")
    quality_ok = checks["qed"] and checks["sa"]
    if checks["sim"] and quality_ok:
        labels.append("sq")
    if quality_ok and checks["dock"]:
        labels.append("qd")
    if checks["sim"] and checks["dock"]:
        labels.append("sd")
    if item["strict"]:
        labels.append("strict")
    return labels


def _archive_key(item, label):
    if label == "sq":
        return (item["dock"], item["sim"], min(item["qed"], item["sa"]))
    if label == "qd":
        return (item["sim"], item["dock"], min(item["qed"], item["sa"]))
    if label == "sd":
        quality_margin = min(item["normalized"]["qed"], item["normalized"]["sa"])
        return (quality_margin, item["sim"], item["dock"])
    if label == "strict":
        return (item["dock"], item["sim"], item["qed"], item["sa"])
    return (-item["residual"], item["sim"], item["dock"])


def merge_archive(existing, candidates, label, max_size):
    deduplicated = {}
    for item in list(existing) + list(candidates):
        smiles = item["smiles"]
        previous = deduplicated.get(smiles)
        if previous is None or _archive_key(item, label) > _archive_key(previous, label):
            deduplicated[smiles] = item
    return sorted(
        deduplicated.values(),
        key=lambda item: _archive_key(item, label),
        reverse=True,
    )[:max_size]


def allocate_counts(total, scores, minimum_each=1):
    """Allocate an integer proposal budget while keeping every arm alive."""
    if total <= 0 or not scores:
        return {name: 0 for name in scores}
    names = list(scores)
    counts = {name: 0 for name in names}
    if total >= len(names) * minimum_each:
        for name in names:
            counts[name] = minimum_each
        total -= len(names) * minimum_each
    if total <= 0:
        return counts

    positive = {name: max(1e-8, float(scores[name])) for name in names}
    denominator = sum(positive.values())
    exact = {name: total * positive[name] / denominator for name in names}
    floors = {name: int(math.floor(value)) for name, value in exact.items()}
    for name, value in floors.items():
        counts[name] += value
    remainder = total - sum(floors.values())
    order = sorted(names, key=lambda name: exact[name] - floors[name], reverse=True)
    for name in order[:remainder]:
        counts[name] += 1
    return counts


def completion_recovery_multipliers(
    archive_presence,
    *,
    completion_boost=4.0,
    start_boost=3.0,
    joint_boost=2.0,
):
    """Prioritize the operator that can complete an existing pair frontier.

    The mapping is task agnostic: ``sq`` needs docking, ``qd`` needs
    similarity, and ``sd`` needs molecular quality. When no pairwise-feasible
    archive exists, recovery falls back to repairing the start lineage and to
    joint moves instead of inventing target-specific rules.
    """
    present = {name: bool(archive_presence.get(name)) for name in ("sq", "qd", "sd")}
    completion_boost = max(1.0, float(completion_boost))
    start_boost = max(1.0, float(start_boost))
    joint_boost = max(1.0, float(joint_boost))
    multipliers = {
        "start_repair": 1.0,
        "dock_refine": completion_boost if present["sq"] else 1.0,
        "similarity_repair": completion_boost if present["qd"] else 1.0,
        "quality_repair": completion_boost if present["sd"] else 1.0,
        "joint_repair": 1.0,
    }
    pair_count = sum(present.values())
    if pair_count == 0:
        multipliers["start_repair"] = start_boost
        multipliers["joint_repair"] = joint_boost
    elif pair_count >= 2:
        multipliers["joint_repair"] = joint_boost
    return multipliers


def recovery_v2_state(
    *,
    iteration,
    warmup_iterations,
    has_strict,
    has_stage_three,
    has_generated_similarity,
    pair_frontier_count,
):
    """Classify a failed lead search from observed, target-agnostic evidence."""
    if int(iteration) <= max(0, int(warmup_iterations)):
        return "warmup"
    if has_strict:
        return "refine"
    if has_stage_three:
        return "complete"
    if not has_generated_similarity:
        return "seed_anchor"
    if int(pair_frontier_count) > 0:
        return "bridge"
    return "explore"


def recovery_v2_operator_multipliers(state, archive_presence):
    """Return generic repair priorities for one observed recovery state."""
    present = {
        name: bool(archive_presence.get(name)) for name in ("sq", "qd", "sd")
    }
    multipliers = {
        "start_repair": 1.0,
        "dock_refine": 1.0,
        "similarity_repair": 1.0,
        "quality_repair": 1.0,
        "joint_repair": 1.0,
    }
    if state == "seed_anchor":
        multipliers.update(start_repair=7.0, joint_repair=3.0)
    elif state == "complete":
        multipliers.update(
            dock_refine=7.0 if present["sq"] else 1.0,
            similarity_repair=7.0 if present["qd"] else 1.0,
            quality_repair=7.0 if present["sd"] else 1.0,
            joint_repair=1.5,
        )
    elif state == "bridge":
        multipliers.update(start_repair=2.0, joint_repair=4.0)
        for operator, archive_name in (
            ("dock_refine", "sq"),
            ("similarity_repair", "qd"),
            ("quality_repair", "sd"),
        ):
            if present[archive_name]:
                multipliers[operator] = 2.0
    elif state == "explore":
        multipliers.update(start_repair=2.5, joint_repair=2.5)
    elif state == "refine":
        multipliers.update(
            dock_refine=3.0 if present["sq"] else 1.0,
            similarity_repair=3.0 if present["qd"] else 1.0,
            quality_repair=3.0 if present["sd"] else 1.0,
        )
    return multipliers


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
    return tokens, {atom_idx: token_idx for atom_idx, token_idx in enumerate(positions)}


def peripheral_edit_plan(smiles, rng, delta=0, max_atom_fraction=0.35, max_span_tokens=None):
    """Choose a contiguous peripheral component while keeping the graph core frozen."""
    mol = Chem.MolFromSmiles(smiles)
    if mol is None or mol.GetNumAtoms() < 2:
        return None
    tokens, atom_positions = _atom_token_positions(smiles, mol)
    if atom_positions is None:
        return None

    scaffold = MurckoScaffold.GetScaffoldForMol(mol)
    core_atoms = set()
    if scaffold is not None and scaffold.GetNumAtoms() > 0:
        match = mol.GetSubstructMatch(scaffold)
        core_atoms = set(match)
    if not core_atoms:
        core_atoms = {atom.GetIdx() for atom in mol.GetAtoms() if atom.IsInRing()}

    all_atoms = set(range(mol.GetNumAtoms()))
    max_component_atoms = max(1, int(math.ceil(mol.GetNumAtoms() * max_atom_fraction)))
    plans = []
    for bond in mol.GetBonds():
        if bond.IsInRing() or bond.GetBondType() != Chem.BondType.SINGLE:
            continue
        left = _component_without_bond(mol, bond.GetBeginAtomIdx(), bond.GetIdx())
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
            component_positions = sorted(atom_positions[idx] for idx in component)
            start = component_positions[0]
            stop = component_positions[-1] + 1
            span_atom_indices = {
                atom_idx
                for atom_idx, token_idx in atom_positions.items()
                if start <= token_idx < stop
            }
            if span_atom_indices != component:
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
    weights = [1.0 / math.sqrt(max(1, plan["component_atoms"])) for plan in plans]
    return rng.choices(plans, weights=weights, k=1)[0]


def adaptive_peripheral_edit_plan(
    smiles,
    rng,
    delta=0,
    target_atom_fraction=0.15,
    max_atom_fraction=0.55,
    max_span_tokens=None,
):
    """Select a peripheral component near a state-dependent trust-region size."""
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
        core_atoms = {atom.GetIdx() for atom in mol.GetAtoms() if atom.IsInRing()}

    all_atoms = set(range(mol.GetNumAtoms()))
    max_component_atoms = max(1, int(math.ceil(mol.GetNumAtoms() * max_atom_fraction)))
    target_atoms = max(1, int(round(mol.GetNumAtoms() * target_atom_fraction)))
    plans = []
    for bond in mol.GetBonds():
        if bond.IsInRing() or bond.GetBondType() != Chem.BondType.SINGLE:
            continue
        left = _component_without_bond(mol, bond.GetBeginAtomIdx(), bond.GetIdx())
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


def seed_directed_atom_edit_plan(parent_smiles, seed_smiles, rng):
    """Mask one non-common parent atom while preserving the seed-like core.

    This is a generic similarity projection: it uses only the task's supplied
    seed molecule and never inspects the protein or target identity.
    """
    parent = Chem.MolFromSmiles(parent_smiles)
    seed = Chem.MolFromSmiles(seed_smiles)
    if parent is None or seed is None or parent.GetNumAtoms() < 2:
        return None
    tokens, atom_positions = _atom_token_positions(parent_smiles, parent)
    if atom_positions is None:
        return None

    result = rdFMCS.FindMCS(
        [parent, seed],
        atomCompare=rdFMCS.AtomCompare.CompareElements,
        bondCompare=rdFMCS.BondCompare.CompareOrder,
        ringMatchesRingOnly=True,
        completeRingsOnly=True,
        timeout=1,
    )
    if not result.smartsString:
        return None
    pattern = Chem.MolFromSmarts(result.smartsString)
    if pattern is None:
        return None
    matches = parent.GetSubstructMatches(pattern, uniquify=True, maxMatches=32)
    if not matches:
        return None

    # Prefer the alignment that leaves the smallest, most peripheral repair.
    match = max(
        matches,
        key=lambda atoms: (
            len(atoms),
            sum(parent.GetAtomWithIdx(idx).IsInRing() for idx in atoms),
        ),
    )
    outside = set(range(parent.GetNumAtoms())) - set(match)
    if not outside:
        return None
    candidates = sorted(
        outside,
        key=lambda idx: (
            parent.GetAtomWithIdx(idx).IsInRing(),
            parent.GetAtomWithIdx(idx).GetDegree(),
            idx,
        ),
    )
    best_key = (
        parent.GetAtomWithIdx(candidates[0]).IsInRing(),
        parent.GetAtomWithIdx(candidates[0]).GetDegree(),
    )
    best = [
        idx
        for idx in candidates
        if (
            parent.GetAtomWithIdx(idx).IsInRing(),
            parent.GetAtomWithIdx(idx).GetDegree(),
        )
        == best_key
    ]
    atom_idx = rng.choice(best)
    token_idx = atom_positions[atom_idx]
    return {
        "start": token_idx,
        "stop": token_idx + 1,
        "delta": 0,
        "peripheral": not parent.GetAtomWithIdx(atom_idx).IsInRing(),
        "component_atoms": 1,
        "seed_directed": True,
    }


def atom_span_edit_plan(smiles, rng, delta=0, span_tokens=4):
    """Fallback atom-centered span plan when no peripheral component is available."""
    tokens = tokenize_smiles(smiles)
    if not tokens:
        return None
    atom_positions = [idx for idx, token in enumerate(tokens) if is_atom_token(token)]
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
            return {
                "start": start,
                "stop": stop,
                "delta": int(delta),
                "peripheral": False,
                "component_atoms": None,
            }
    return None
