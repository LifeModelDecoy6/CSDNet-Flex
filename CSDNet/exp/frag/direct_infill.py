"""Fair direct constrained generation for the fragment benchmark.

Every proposal is sampled independently from the supplied molecular fragment.
The target molecule and generated candidates never enter proposal construction,
and failed structural checks are not replaced with additional model calls.
"""

from __future__ import annotations

import random
import re
from dataclasses import dataclass
from pathlib import Path

from rdkit import Chem

from CSDNet.exp.frag.task_head import (
    FragmentConstraintSpec,
    normalize_dummy_atoms,
)
from CSDNet.util.length_prior import load_atomic_length_prior
from CSDNet.util.tokenizer import tokenize_smiles


DUMMY_TOKEN = re.compile(r"^(?:\[(?:\d+)?\*\]|\*)$")
PLACEHOLDER_MAP_START = 900


@dataclass(frozen=True)
class MaskedTemplate:
    seed_smiles: str
    edit_plans: tuple[dict[str, int | str], ...]
    target_length: int
    added_tokens: int
    attachment_count: int


def load_length_prior(path: str | Path, max_len: int | None = None) -> list[int]:
    """Load body-token lengths for fragment template construction.

    Atomic JSON priors store complete lengths including BOS/EOS.  Fragment
    templates work in body-token coordinates, so those two special tokens are
    removed exactly once here.  Pickle support is retained for historical runs.
    """
    path = Path(path)
    if path.suffix.lower() == ".json":
        complete_lengths, _ = load_atomic_length_prior(path, max_len=max_len)
        lengths = [length - 2 for length in complete_lengths]
    else:
        import pickle

        with path.open("rb") as handle:
            values = pickle.load(handle)
        lengths = [int(value) for value in values if int(value) >= 3]
    if not lengths:
        raise ValueError(f"No usable lengths in {path}")
    return lengths


def _strip_stereo(smiles: str, canonical: bool = False) -> str:
    mol = Chem.MolFromSmiles(str(smiles))
    if mol is None:
        raise ValueError(f"Could not parse fragment: {smiles}")
    Chem.RemoveStereochemistry(mol)
    return Chem.MolToSmiles(
        mol,
        canonical=canonical,
        isomericSmiles=False,
    )


def _dummy_token_positions(tokens: list[str]) -> list[int]:
    return [index for index, token in enumerate(tokens) if DUMMY_TOKEN.match(token)]


def _positive_partition(total: int, parts: int, rng: random.Random) -> list[int]:
    if parts <= 0:
        return []
    total = max(parts, int(total))
    if parts == 1:
        return [total]
    cuts = sorted(rng.sample(range(1, total), parts - 1))
    bounds = [0, *cuts, total]
    return [bounds[index + 1] - bounds[index] for index in range(parts)]


def _sample_added_length(
    *,
    fixed_tokens: int,
    attachment_count: int,
    max_body_tokens: int,
    length_prior: list[int],
    min_added_tokens: int,
    rng: random.Random,
    added_token_range: tuple[int, int] | None = None,
    target_length_range: tuple[int, int] | None = None,
    length_quantile: float | None = None,
) -> tuple[int, int]:
    if added_token_range is not None and target_length_range is not None:
        raise ValueError("choose either added-token or target-length support")

    def draw(lower: int, upper: int) -> int:
        if length_quantile is None:
            return rng.randint(lower, upper)
        quantile = min(max(float(length_quantile), 0.0), 1.0)
        width = upper - lower + 1
        return lower + min(int(quantile * width), width - 1)

    available = max(attachment_count, max_body_tokens - fixed_tokens)
    if added_token_range is not None:
        lower, upper = (int(value) for value in added_token_range)
        lower = max(attachment_count, min_added_tokens, lower)
        upper = max(lower, upper)
        lower = min(lower, available)
        upper = min(upper, available)
        total_added = draw(lower, upper)
        return fixed_tokens + total_added, total_added

    if target_length_range is not None:
        lower, upper = (int(value) for value in target_length_range)
        if lower > upper:
            raise ValueError("target_length_range must be ordered")
        support_width = upper - lower
        feasible_min = fixed_tokens + max(attachment_count, min_added_tokens)
        if feasible_min > upper:
            lower = feasible_min
            upper = feasible_min + support_width
        else:
            lower = max(lower, feasible_min)
        upper = min(upper, max_body_tokens)
        lower = min(lower, upper)
        target = draw(lower, upper)
        total_added = target - fixed_tokens
        return target, total_added

    feasible_min = fixed_tokens + max(attachment_count, min_added_tokens)
    support = sorted(
        int(target)
        for target in length_prior
        if feasible_min <= int(target) <= max_body_tokens
    )
    if support:
        if length_quantile is None:
            target = int(rng.choice(support))
        else:
            quantile = min(max(float(length_quantile), 0.0), 1.0)
            index = min(int(quantile * len(support)), len(support) - 1)
            target = support[index]
    else:
        target = min(max_body_tokens, feasible_min)
    total_added = min(available, max(attachment_count, target - fixed_tokens))
    return fixed_tokens + total_added, total_added


def _plans_for_dummy_fragment(
    fragment: str,
    *,
    max_len: int,
    length_prior: list[int],
    min_added_tokens: int,
    rng: random.Random,
    added_token_range: tuple[int, int] | None = None,
    target_length_range: tuple[int, int] | None = None,
    length_quantile: float | None = None,
) -> MaskedTemplate:
    seed = _strip_stereo(fragment)
    tokens = tokenize_smiles(seed)
    positions = _dummy_token_positions(tokens)
    if not positions:
        raise ValueError(f"Fragment has no attachment dummy: {fragment}")

    fixed_tokens = len(tokens) - len(positions)
    target, total_added = _sample_added_length(
        fixed_tokens=fixed_tokens,
        attachment_count=len(positions),
        max_body_tokens=max_len - 2,
        length_prior=length_prior,
        min_added_tokens=min_added_tokens,
        rng=rng,
        added_token_range=added_token_range,
        target_length_range=target_length_range,
        length_quantile=length_quantile,
    )
    span_lengths = _positive_partition(total_added, len(positions), rng)
    plans = tuple(
        {
            "start": position,
            "stop": position + 1,
            "replacement_len": span_length,
        }
        for position, span_length in zip(positions, span_lengths)
    )
    return MaskedTemplate(
        seed_smiles=seed,
        edit_plans=plans,
        target_length=target,
        added_tokens=total_added,
        attachment_count=len(positions),
    )


def _add_attachment_dummy(smiles: str, rng: random.Random) -> str:
    mol = Chem.MolFromSmiles(str(smiles))
    if mol is None:
        raise ValueError(f"Could not parse superstructure fragment: {smiles}")
    Chem.RemoveStereochemistry(mol)
    candidates = [
        atom.GetIdx()
        for atom in mol.GetAtoms()
        if atom.GetAtomicNum() > 0 and atom.GetNumImplicitHs() > 0
    ]
    if not candidates:
        raise ValueError(f"No hydrogen-bearing attachment atom in: {smiles}")
    rw = Chem.RWMol(mol)
    dummy = Chem.Atom(0)
    dummy.SetIsotope(1)
    dummy_index = rw.AddAtom(dummy)
    rw.AddBond(rng.choice(candidates), dummy_index, Chem.BondType.SINGLE)
    attached = rw.GetMol()
    Chem.SanitizeMol(attached)
    return Chem.MolToSmiles(attached, canonical=False, isomericSmiles=False)


def _remove_single_dummy(fragment: str):
    mol = Chem.MolFromSmiles(str(fragment))
    if mol is None:
        raise ValueError(f"Could not parse linker terminal: {fragment}")
    Chem.RemoveStereochemistry(mol)
    dummy_atoms = [atom for atom in mol.GetAtoms() if atom.GetAtomicNum() == 0]
    if len(dummy_atoms) != 1 or dummy_atoms[0].GetDegree() != 1:
        raise ValueError(f"Linker terminal needs one degree-one dummy: {fragment}")

    dummy = dummy_atoms[0]
    dummy_index = dummy.GetIdx()
    neighbor_index = dummy.GetNeighbors()[0].GetIdx()
    bond_type = mol.GetBondBetweenAtoms(dummy_index, neighbor_index).GetBondType()
    rw = Chem.RWMol(mol)
    rw.RemoveAtom(dummy_index)
    if dummy_index < neighbor_index:
        neighbor_index -= 1
    cleaned = rw.GetMol()
    # An aromatic attachment atom can be impossible to kekulize while it is
    # temporarily uncapped.  The full linked molecule is sanitized after the
    # placeholder chain restores that missing bond.
    return cleaned, neighbor_index, bond_type


def _build_linker_placeholder(
    left_fragment: str,
    right_fragment: str,
    linker_tokens: int,
) -> tuple[str, list[int]]:
    left, left_anchor, left_bond = _remove_single_dummy(left_fragment)
    right, right_anchor, right_bond = _remove_single_dummy(right_fragment)
    combined = Chem.CombineMols(left, right)
    rw = Chem.RWMol(combined)
    right_anchor += left.GetNumAtoms()

    placeholders = []
    for offset in range(max(1, int(linker_tokens))):
        atom = Chem.Atom("C")
        atom.SetAtomMapNum(PLACEHOLDER_MAP_START + offset)
        placeholders.append(rw.AddAtom(atom))

    rw.AddBond(left_anchor, placeholders[0], left_bond)
    for first, second in zip(placeholders, placeholders[1:]):
        rw.AddBond(first, second, Chem.BondType.SINGLE)
    rw.AddBond(placeholders[-1], right_anchor, right_bond)

    mol = rw.GetMol()
    Chem.SanitizeMol(mol)
    seed = Chem.MolToSmiles(mol, canonical=True, isomericSmiles=False)
    tokens = tokenize_smiles(seed)
    positions = [
        index
        for index, token in enumerate(tokens)
        if any(f":{PLACEHOLDER_MAP_START + offset}]" in token for offset in range(len(placeholders)))
    ]
    if len(positions) != len(placeholders):
        raise ValueError(
            f"Lost linker placeholders: expected {len(placeholders)}, found {len(positions)}"
        )
    return seed, positions


def _linker_template(
    fragment_pair: str,
    *,
    max_len: int,
    length_prior: list[int],
    min_added_tokens: int,
    rng: random.Random,
    added_token_range: tuple[int, int] | None = None,
    target_length_range: tuple[int, int] | None = None,
    length_quantile: float | None = None,
) -> MaskedTemplate:
    parts = str(fragment_pair).split(".")
    if len(parts) != 2:
        raise ValueError(f"Expected two linker terminals: {fragment_pair}")

    fixed_tokens = sum(
        len(tokenize_smiles(_strip_stereo(part))) - 1
        for part in parts
    )
    target, linker_tokens = _sample_added_length(
        fixed_tokens=fixed_tokens,
        attachment_count=1,
        max_body_tokens=max_len - 2,
        length_prior=length_prior,
        min_added_tokens=min_added_tokens,
        rng=rng,
        added_token_range=added_token_range,
        target_length_range=target_length_range,
        length_quantile=length_quantile,
    )
    seed, positions = _build_linker_placeholder(
        normalize_dummy_atoms(parts[0]),
        normalize_dummy_atoms(parts[1]),
        linker_tokens,
    )
    plans = tuple(
        {
            "start": position,
            "stop": position + 1,
            "replacement_len": 1,
            # These masks stand for vertices in an RDKit-built linker chain,
            # not arbitrary positions in a free-form SMILES substring.
            "token_constraint": "chain_atom",
        }
        for position in positions
    )
    return MaskedTemplate(
        seed_smiles=seed,
        edit_plans=plans,
        target_length=target,
        added_tokens=len(positions),
        attachment_count=2,
    )


def build_masked_template(
    spec: FragmentConstraintSpec,
    *,
    max_len: int,
    length_prior: list[int],
    min_added_tokens: int,
    rng: random.Random,
    added_token_range: tuple[int, int] | None = None,
    target_length_range: tuple[int, int] | None = None,
    length_quantile: float | None = None,
) -> MaskedTemplate:
    if spec.requires_bridge:
        return _linker_template(
            spec.fragment,
            max_len=max_len,
            length_prior=length_prior,
            min_added_tokens=min_added_tokens,
            rng=rng,
            added_token_range=added_token_range,
            target_length_range=target_length_range,
            length_quantile=length_quantile,
        )
    fragment = spec.fragment
    if spec.attachment_count == 0:
        fragment = _add_attachment_dummy(fragment, rng)
    return _plans_for_dummy_fragment(
        fragment,
        max_len=max_len,
        length_prior=length_prior,
        min_added_tokens=min_added_tokens,
        rng=rng,
        added_token_range=added_token_range,
        target_length_range=target_length_range,
        length_quantile=length_quantile,
    )


def build_native_projected_template(
    spec: FragmentConstraintSpec,
    *,
    max_len: int,
    rng: random.Random,
    initial_gap_tokens: int = 1,
) -> MaskedTemplate:
    """Project a fragment condition into the elastic model's native state.

    Each required attachment is initialized by one masked token. The fragment
    runner keeps every materialized position inside that region open to learned
    recursive insertion, while all condition tokens remain fixed. The insertion
    head therefore controls additional length without an empirical length prior
    or a task-specific length band.
    """
    gap_count = (
        1
        if spec.requires_bridge or spec.attachment_count == 0
        else int(spec.attachment_count)
    )
    projected = build_masked_template(
        spec,
        max_len=max_len,
        length_prior=[gap_count],
        min_added_tokens=gap_count,
        rng=rng,
        added_token_range=(gap_count, gap_count),
    )
    initial_gap_tokens = int(initial_gap_tokens)
    if initial_gap_tokens not in (0, 1):
        raise ValueError("initial_gap_tokens must be either 0 or 1")
    plans = tuple(
        {
            "start": int(plan["start"]),
            "stop": int(plan["stop"]),
            "length_mode": "learned_insertion",
            "initial_replacement_len": initial_gap_tokens,
            "min_replacement_len": 1,
            "max_replacement_len": max(1, int(max_len) - 2),
            **(
                {"token_constraint": str(plan["token_constraint"])}
                if plan.get("token_constraint") is not None
                else {}
            ),
        }
        for plan in projected.edit_plans
    )
    return MaskedTemplate(
        seed_smiles=projected.seed_smiles,
        edit_plans=plans,
        target_length=projected.target_length,
        added_tokens=gap_count,
        attachment_count=projected.attachment_count,
    )


def apply_native_gap_constraint_policy(
    template: MaskedTemplate,
    *,
    geometry: str,
    attempt_index: int,
    case_seed: int,
    policy: str,
) -> tuple[tuple[dict[str, int | str], ...], bool]:
    """Apply task-agnostic structural roles to native editable gaps.

    Both adaptive policies are keyed only by attachment geometry. The
    calibrated policy removes the chain restriction from bridges, where it
    suppresses useful chemistry, and concentrates it on multi-attachment
    decorations, where it provides a strong structural floor.
    """
    if policy not in {
        "none",
        "geometry_adaptive",
        "geometry_calibrated",
        "structural_feasible",
        "all_chain",
    }:
        raise ValueError(f"Unsupported native gap constraint policy: {policy!r}")
    constrained = policy == "all_chain"
    token_constraint = "chain_atom" if constrained else None
    if policy == "geometry_adaptive":
        if geometry == "substructure_expand":
            constrained = True
        elif geometry in {"multi_anchor", "multi_attachment"}:
            constrained = (int(attempt_index) + int(case_seed)) % 4 == 0
    elif policy == "geometry_calibrated":
        if geometry == "substructure_expand":
            constrained = True
        elif geometry == "multi_attachment":
            # A second deterministic permutation keeps the 75% structural
            # allocation approximately balanced within every length stratum.
            constraint_slot = (
                int(attempt_index) * 53 + int(case_seed) * 17
            ) % 100
            constrained = constraint_slot < 75
    elif policy == "structural_feasible":
        # A bridge or multi-attachment completion must enter and leave every
        # editable gap through an atom. Only those two dynamic boundary roles
        # are constrained; the interior remains free to express rings,
        # branches, bond orders and bracket atoms. Terminal expansion keeps the
        # proven all-chain floor, while a single attachment remains unrestricted.
        if geometry in {"multi_anchor", "multi_attachment"}:
            constrained = True
            token_constraint = "atom_bounded"
        elif geometry == "substructure_expand":
            constrained = True
            token_constraint = "chain_atom"

    plans = []
    for plan in template.edit_plans:
        row = dict(plan)
        row.pop("token_constraint", None)
        if constrained:
            row["token_constraint"] = token_constraint or "chain_atom"
        plans.append(row)
    return tuple(plans), constrained


def native_gap_insertion_rate_scale(
    *,
    geometry: str,
    base_scale: float,
    policy: str,
) -> float:
    """Calibrate local insertion intensity from attachment geometry alone.

    A terminal completion with one visible attachment systematically
    suppresses the learned local insertion rate. Bridge, multi-attachment and
    unconstrained-expansion geometries retain the checkpoint's native rate;
    increasing bridge intensity costs more structural validity than it gains
    in quality.
    """
    if policy not in {"uniform", "geometry_adaptive"}:
        raise ValueError(f"Unsupported insertion-rate policy: {policy!r}")
    scale = float(base_scale)
    if scale <= 0.0:
        raise ValueError("base_scale must be positive")
    if policy == "geometry_adaptive" and geometry == "single_attachment":
        scale *= 1.4
    return scale


def native_nucleus_support(
    *,
    geometry: str,
    start: int,
    end: int,
    policy: str,
) -> tuple[int, int]:
    """Return a geometry-only nucleus-support schedule.

    The calibrated variant opens early token support only for multi-anchor
    bridges, whose downloaded trajectories show within-case mode collapse.
    Other geometries preserve the checkpoint's top-p behavior.
    """
    if policy not in {"uniform", "multi_anchor_annealed"}:
        raise ValueError(f"Unsupported nucleus-support policy: {policy!r}")
    start = int(start)
    end = int(end)
    if start < 1 or end < 1:
        raise ValueError("Nucleus-support counts must be positive")
    if policy == "multi_anchor_annealed" and geometry != "multi_anchor":
        return 1, 1
    return start, end


def native_sampler_arm(
    *,
    attempt_index: int,
    case_seed: int,
    exploration_fraction: float,
    policy: str,
    prefill_source: str | None = None,
) -> str:
    """Assign a proposal to a fixed core/exploration mixture.

    The assignment is a deterministic permutation of proposal indices.  It
    therefore defines one sampling distribution before generation begins and
    cannot react to molecular-property scores or discard weak proposals.
    Every block of 100 attempts receives the requested integer percentage of
    exploration trajectories, independently of batch ordering.
    """
    if policy not in {"uniform", "fixed_diversity", "prefill_guarded"}:
        raise ValueError(f"Unsupported sampler portfolio policy: {policy!r}")
    fraction = float(exploration_fraction)
    if not 0.0 <= fraction <= 1.0:
        raise ValueError("exploration_fraction must be in [0, 1]")
    if policy == "uniform":
        return "core"
    if policy == "prefill_guarded":
        if prefill_source is None:
            raise ValueError(
                "prefill_guarded requires the proposal prefill source"
            )
        # Only proposals drawn from the empirical upper-length stratum use
        # wider token support. This couples extra entropy to restored length
        # capacity without perturbing the high-quality anchor trajectories.
        return (
            "exploration"
            if str(prefill_source).startswith("zinc_upper_")
            else "core"
        )
    if fraction == 0.0:
        return "core"

    threshold = int(round(100 * fraction))
    slot = (int(attempt_index) * 37 + int(case_seed)) % 100
    return "exploration" if slot < threshold else "core"
