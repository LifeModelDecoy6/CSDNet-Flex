"""Fragment-constrained task head for the shared V1 frontier engine.

The benchmark target molecule is deliberately absent from every proposal
path.  It is retained only as evaluation metadata for the final distance
metric.  Online adaptation uses structural completion signals only; QED, SA,
diversity, and distance never enter the policy.
"""

from __future__ import annotations

import ast
import itertools
import math
import random
import re
from collections import Counter
from dataclasses import dataclass
from typing import Iterable, Sequence

from rdkit import Chem

from CSDNet.exp.pmo.optimizer import attach_fragments, canonical_smiles
from CSDNet.util.tokenizer import tokenize_smiles


CANONICAL_TASKS = (
    "linker_design",
    "scaffold_morphing",
    "motif_extension",
    "scaffold_decoration",
    "superstructure_generation",
)

TASK_ALIASES = {
    "superstructure_design": "superstructure_generation",
}

SINGLE_CAPS = (
    "[1*]C",
    "[1*]CC",
    "[1*]CCC",
    "[1*]CO",
    "[1*]CN",
    "[1*]O",
    "[1*]N",
    "[1*]C(F)F",
    "[1*]C(=O)N",
    "[1*]c1ccccc1",
)

LINKER_TEMPLATES = (
    "[1*]C[1*]",
    "[1*]CC[1*]",
    "[1*]CCC[1*]",
    "[1*]CCCC[1*]",
    "[1*]CO[1*]",
    "[1*]OC[1*]",
    "[1*]CCO[1*]",
    "[1*]OCC[1*]",
    "[1*]CNC[1*]",
    "[1*]NCC[1*]",
    "[1*]CCN[1*]",
    "[1*]OCCO[1*]",
    "[1*]NCCN[1*]",
    "[1*]C=C[1*]",
    "[1*]C#C[1*]",
)


def _linear_atom_fragments(*, max_atoms: int, linker: bool) -> tuple[str, ...]:
    """Build a small target-agnostic C/N/O seed grammar."""
    fragments = []
    for length in range(1, max(1, int(max_atoms)) + 1):
        for atoms in itertools.product(("C", "N", "O"), repeat=length):
            body = "".join(atoms)
            suffix = "[1*]" if linker else ""
            fragments.append(f"[1*]{body}{suffix}")
    return tuple(fragments)


V2_SINGLE_CAPS = tuple(
    dict.fromkeys((*SINGLE_CAPS, *_linear_atom_fragments(max_atoms=3, linker=False)))
)

V2_LINKER_TEMPLATES = tuple(
    dict.fromkeys(
        (*LINKER_TEMPLATES, *_linear_atom_fragments(max_atoms=4, linker=True))
    )
)

OPERATORS = (
    "legacy_completion",
    "anchor_growth",
    "local_repair",
    "structural_restart",
    "bridge_closure",
    "decoration_fill",
    "superstructure_expand",
)

_ATOM_TOKEN = re.compile(r"^(?:\[[^\]]+\]|Br|Cl|B|C|N|O|P|S|F|I|b|c|n|o|p|s|\*)$")


def normalize_task(task: str) -> str:
    task = TASK_ALIASES.get(str(task), str(task))
    if task not in CANONICAL_TASKS:
        raise ValueError(f"Unsupported fragment task: {task}")
    return task


def task_column(task: str) -> str:
    task = normalize_task(task)
    if task in {"linker_design", "scaffold_morphing"}:
        return "linker_design"
    return task


def _dummy_count(mol: Chem.Mol | None) -> int:
    if mol is None:
        return 0
    return sum(atom.GetAtomicNum() == 0 for atom in mol.GetAtoms())


def normalize_dummy_atoms(smiles: str) -> str | None:
    mol = Chem.MolFromSmiles(str(smiles))
    if mol is None:
        return None
    for atom in mol.GetAtoms():
        if atom.GetAtomicNum() == 0:
            atom.SetIsotope(1)
    return Chem.MolToSmiles(mol, canonical=True)


def prepare_model_seed(smiles: str, tokenizer, max_len: int) -> str | None:
    """Return the non-isomeric representation supported by the model vocab.

    GenMol's fragment-completion path also ignores stereochemistry.  Keeping
    this conversion at the model boundary avoids mapping frozen chiral atom
    tokens to ``<unk>`` while the original benchmark query remains unchanged.
    """
    mol = Chem.MolFromSmiles(str(smiles))
    if mol is None:
        return None
    Chem.RemoveStereochemistry(mol)
    can = Chem.MolToSmiles(mol, canonical=True, isomericSmiles=False)
    tokens = tokenize_smiles(can)
    if len(tokens) + 2 > int(max_len):
        return None
    if not all(token in tokenizer.vocab for token in tokens):
        return None
    return can


def query_without_dummies(fragment: str) -> Chem.Mol | None:
    mol = Chem.MolFromSmiles(str(fragment))
    if mol is None:
        return None
    try:
        query = Chem.DeleteSubstructs(mol, Chem.MolFromSmiles("*"))
        if query is not None and query.GetNumHeavyAtoms() > 0:
            return query
    except Exception:
        pass
    rw = Chem.RWMol(mol)
    for index in sorted(
        (atom.GetIdx() for atom in rw.GetAtoms() if atom.GetAtomicNum() == 0),
        reverse=True,
    ):
        rw.RemoveAtom(index)
    query = rw.GetMol()
    try:
        Chem.SanitizeMol(query)
    except Exception:
        return None
    if query.GetNumHeavyAtoms() == 0:
        return None
    return query


@dataclass(frozen=True)
class FragmentConstraintSpec:
    task: str
    name: str
    original: str
    fragment: str
    queries: tuple[Chem.Mol, ...]
    attachment_count: int
    requires_bridge: bool

    @property
    def geometry(self) -> str:
        if self.requires_bridge:
            return "multi_anchor"
        if self.attachment_count > 1:
            return "multi_attachment"
        if self.attachment_count == 0:
            return "substructure_expand"
        return "single_attachment"


@dataclass(frozen=True)
class CandidateAssessment:
    smiles: str | None
    valid: bool
    connected: bool
    no_dummies: bool
    preserved: int
    required: int
    structural_success: bool

    @property
    def preserved_fraction(self) -> float:
        return self.preserved / max(1, self.required)

    @property
    def structural_score(self) -> float:
        if not self.valid:
            return 0.0
        return min(
            1.0,
            0.55 * float(self.structural_success)
            + 0.25 * self.preserved_fraction
            + 0.10 * float(self.connected)
            + 0.10 * float(self.no_dummies),
        )

    def transition(self) -> dict[str, float | bool]:
        return {
            "valid": self.valid,
            "connected": self.connected,
            "no_dummies": self.no_dummies,
            "preserved_fraction": self.preserved_fraction,
            "strict": self.structural_success,
            "structural_score": self.structural_score,
        }


def build_constraint_spec(task: str, row) -> FragmentConstraintSpec:
    task = normalize_task(task)
    fragment = str(row[task_column(task)])
    if task in {"linker_design", "scaffold_morphing"}:
        parts = fragment.split(".")
        if len(parts) != 2:
            raise ValueError(
                f"{task} requires exactly two terminal fragments: {fragment}"
            )
        queries = tuple(query_without_dummies(part) for part in parts)
        if any(query is None for query in queries):
            raise ValueError(f"Could not parse linker queries: {fragment}")
        attachment_count = sum(_dummy_count(Chem.MolFromSmiles(part)) for part in parts)
        requires_bridge = True
    else:
        query = query_without_dummies(fragment)
        if query is None:
            raise ValueError(f"Could not parse fragment query: {fragment}")
        queries = (query,)
        attachment_count = _dummy_count(Chem.MolFromSmiles(fragment))
        requires_bridge = False
    return FragmentConstraintSpec(
        task=task,
        name=str(row["name"]),
        original=str(row["smiles"]),
        fragment=fragment,
        queries=queries,
        attachment_count=attachment_count,
        requires_bridge=requires_bridge,
    )


def assess_candidate(smiles: str | None, spec: FragmentConstraintSpec):
    can = canonical_smiles(smiles) if smiles else None
    mol = Chem.MolFromSmiles(can) if can else None
    if mol is None:
        return CandidateAssessment(
            None, False, False, False, 0, len(spec.queries), False
        )
    connected = len(Chem.GetMolFrags(mol)) == 1
    no_dummies = _dummy_count(mol) == 0
    preserved = sum(mol.HasSubstructMatch(query) for query in spec.queries)
    success = connected and no_dummies and preserved == len(spec.queries)
    return CandidateAssessment(
        can,
        True,
        connected,
        no_dummies,
        preserved,
        len(spec.queries),
        success,
    )


def _contains_all_queries(smiles: str, queries: Sequence[Chem.Mol]) -> bool:
    mol = Chem.MolFromSmiles(smiles)
    return bool(
        mol is not None
        and len(Chem.GetMolFrags(mol)) == 1
        and _dummy_count(mol) == 0
        and all(mol.HasSubstructMatch(query) for query in queries)
    )


def _attach_all_dummies(
    fragment: str,
    caps: Sequence[str],
    rng: random.Random,
) -> str | None:
    current = normalize_dummy_atoms(fragment)
    if current is None:
        return None
    for _ in range(12):
        mol = Chem.MolFromSmiles(current)
        if mol is None:
            return None
        if _dummy_count(mol) == 0:
            return canonical_smiles(current)
        cap = caps[rng.randrange(len(caps))]
        current = attach_fragments(current, cap)
        if current is None:
            return None
    return None


def _attach_cap_to_atom(
    scaffold: str,
    cap: str,
    atom_index: int,
) -> str | None:
    base = Chem.MolFromSmiles(scaffold)
    cap_mol = Chem.MolFromSmiles(cap)
    if base is None or cap_mol is None:
        return None
    dummy_atoms = [atom for atom in cap_mol.GetAtoms() if atom.GetAtomicNum() == 0]
    if len(dummy_atoms) != 1 or dummy_atoms[0].GetDegree() != 1:
        return None
    dummy_index = dummy_atoms[0].GetIdx()
    neighbor_index = dummy_atoms[0].GetNeighbors()[0].GetIdx()
    combo = Chem.CombineMols(base, cap_mol)
    rw = Chem.RWMol(combo)
    offset = base.GetNumAtoms()
    rw.AddBond(int(atom_index), offset + neighbor_index, Chem.BondType.SINGLE)
    rw.RemoveAtom(offset + dummy_index)
    result = rw.GetMol()
    try:
        Chem.SanitizeMol(result)
    except Exception:
        return None
    return Chem.MolToSmiles(result, canonical=True)


def _linker_seed_pool(
    spec: FragmentConstraintSpec,
    limit: int,
    *,
    templates: Sequence[str] = LINKER_TEMPLATES,
    rng: random.Random | None = None,
) -> list[str]:
    left, right = spec.fragment.split(".")
    left = normalize_dummy_atoms(left)
    right = normalize_dummy_atoms(right)
    if left is None or right is None:
        return []
    seeds = []
    seen = set()
    templates = list(templates)
    if rng is not None:
        rng.shuffle(templates)
    for linker in templates:
        for first, second in ((left, right), (right, left)):
            linked = attach_fragments(first, linker)
            linked = attach_fragments(linked, second) if linked else None
            can = canonical_smiles(linked) if linked else None
            if can and can not in seen and _contains_all_queries(can, spec.queries):
                seen.add(can)
                seeds.append(can)
                if len(seeds) >= limit:
                    return seeds
    return seeds


def _completion_seed_pool(
    spec: FragmentConstraintSpec,
    limit: int,
    rng: random.Random,
    *,
    caps: Sequence[str] = SINGLE_CAPS,
) -> list[str]:
    seeds = []
    seen = set()
    if spec.attachment_count:
        attempts = max(limit * 5, len(caps) * 2)
        for offset in range(attempts):
            shuffled_caps = list(caps)
            rng.shuffle(shuffled_caps)
            shuffled_caps = (
                shuffled_caps[offset % len(shuffled_caps) :]
                + shuffled_caps[: offset % len(shuffled_caps)]
            )
            seed = _attach_all_dummies(spec.fragment, shuffled_caps, rng)
            if seed and seed not in seen and _contains_all_queries(seed, spec.queries):
                seen.add(seed)
                seeds.append(seed)
                if len(seeds) >= limit:
                    break
        return seeds

    scaffold = canonical_smiles(spec.fragment)
    mol = Chem.MolFromSmiles(scaffold) if scaffold else None
    if mol is None:
        return []
    atom_indices = [
        atom.GetIdx()
        for atom in mol.GetAtoms()
        if atom.GetAtomicNum() > 0 and atom.GetTotalNumHs() > 0
    ]
    rng.shuffle(atom_indices)
    caps = list(caps)
    rng.shuffle(caps)
    for atom_index in atom_indices:
        for cap in caps:
            seed = _attach_cap_to_atom(scaffold, cap, atom_index)
            if seed and seed not in seen and _contains_all_queries(seed, spec.queries):
                seen.add(seed)
                seeds.append(seed)
                if len(seeds) >= limit:
                    return seeds
    return seeds


def build_seed_pool(
    spec: FragmentConstraintSpec,
    limit: int = 48,
    rng: random.Random | None = None,
) -> list[str]:
    rng = rng or random.Random(0)
    if spec.requires_bridge:
        return _linker_seed_pool(spec, limit)
    return _completion_seed_pool(spec, limit, rng)


def build_seed_pool_v2(
    spec: FragmentConstraintSpec,
    limit: int = 48,
    rng: random.Random | None = None,
) -> list[str]:
    """Build a broader seed pool without consulting the benchmark target."""
    rng = rng or random.Random(0)
    if spec.requires_bridge:
        return _linker_seed_pool(
            spec,
            limit,
            templates=V2_LINKER_TEMPLATES,
            rng=rng,
        )
    return _completion_seed_pool(
        spec,
        limit,
        rng,
        caps=V2_SINGLE_CAPS,
    )


def available_operators(spec: FragmentConstraintSpec) -> tuple[str, ...]:
    operators = [
        "legacy_completion",
        "anchor_growth",
        "local_repair",
        "structural_restart",
    ]
    if spec.requires_bridge:
        operators.append("bridge_closure")
    elif spec.attachment_count > 1:
        operators.append("decoration_fill")
    elif spec.attachment_count == 0:
        operators.append("superstructure_expand")
    return tuple(operators)


@dataclass(frozen=True)
class OperatorProfile:
    max_span_tokens: int
    max_growth_tokens: int
    max_shrink_tokens: int
    learned_insertion: bool
    temperature_start: float
    span_prob: float


OPERATOR_PROFILES = {
    "legacy_completion": OperatorProfile(6, 0, 0, False, 1.08, 0.82),
    "anchor_growth": OperatorProfile(9, 2, 1, True, 1.18, 0.86),
    "local_repair": OperatorProfile(3, 0, 0, False, 0.96, 0.92),
    "structural_restart": OperatorProfile(16, 3, 2, True, 1.34, 0.78),
    "bridge_closure": OperatorProfile(14, 2, 2, True, 1.16, 0.92),
    "decoration_fill": OperatorProfile(8, 2, 1, True, 1.10, 0.90),
    "superstructure_expand": OperatorProfile(10, 3, 0, True, 1.16, 0.88),
}

OPERATOR_PROFILES_V2 = {
    # Fixed-length completion remains available as a conservative floor.
    "legacy_completion": OperatorProfile(6, 0, 0, False, 1.08, 0.82),
    "local_repair": OperatorProfile(3, 0, 0, False, 0.96, 0.92),
    # These bounds define a trust region only. The insertion head, rather than
    # a hand-written delta list, chooses the replacement length inside it.
    "anchor_growth": OperatorProfile(11, 3, 2, True, 1.24, 0.84),
    "structural_restart": OperatorProfile(18, 4, 3, True, 1.38, 0.78),
    "bridge_closure": OperatorProfile(16, 3, 2, True, 1.22, 0.90),
    "decoration_fill": OperatorProfile(10, 3, 2, True, 1.16, 0.88),
    "superstructure_expand": OperatorProfile(12, 4, 2, True, 1.22, 0.86),
}


def _atom_token_positions(tokens: Sequence[str]) -> list[int]:
    return [index for index, token in enumerate(tokens) if _ATOM_TOKEN.match(token)]


def _protected_atom_indices(mol: Chem.Mol, queries: Sequence[Chem.Mol]) -> set[int]:
    protected = set()
    for query in queries:
        matches = mol.GetSubstructMatches(query, uniquify=True)
        if not matches:
            continue
        match = min(matches, key=lambda item: (len(protected.intersection(item)), item))
        protected.update(match)
    return protected


def editable_token_spans(
    smiles: str,
    queries: Sequence[Chem.Mol],
) -> tuple[str | None, list[tuple[int, int]]]:
    mol = Chem.MolFromSmiles(smiles)
    if mol is None:
        return None, []
    can = Chem.MolToSmiles(mol, canonical=True)
    try:
        atom_order = list(ast.literal_eval(mol.GetProp("_smilesAtomOutputOrder")))
    except Exception:
        return can, []
    tokens = tokenize_smiles(can)
    atom_positions = _atom_token_positions(tokens)
    if len(atom_positions) != len(atom_order):
        return can, []
    atom_to_token = dict(zip(atom_order, atom_positions))
    protected_atoms = _protected_atom_indices(mol, queries)
    protected_tokens = sorted(
        atom_to_token[index] for index in protected_atoms if index in atom_to_token
    )
    unprotected_tokens = sorted(
        atom_to_token[index]
        for index in atom_order
        if index not in protected_atoms and index in atom_to_token
    )
    if not unprotected_tokens:
        return can, []

    boundaries = [-1, *protected_tokens, len(tokens)]
    spans = []
    for left, right in zip(boundaries, boundaries[1:]):
        inside = [pos for pos in unprotected_tokens if left < pos < right]
        if inside:
            spans.append((min(inside), max(inside) + 1))
    return can, spans


def make_edit_plan(
    seed: str,
    queries: Sequence[Chem.Mol],
    operator: str,
    rng: random.Random,
    profiles: dict[str, OperatorProfile] | None = None,
) -> tuple[str | None, dict[str, int] | None]:
    profile = (profiles or OPERATOR_PROFILES)[operator]
    can, spans = editable_token_spans(seed, queries)
    if can is None or not spans:
        return can, None
    if operator in {"structural_restart", "bridge_closure", "decoration_fill"}:
        start, stop = max(spans, key=lambda span: span[1] - span[0])
    else:
        start, stop = rng.choice(spans)
    if stop - start > profile.max_span_tokens:
        window_start = rng.randint(start, stop - profile.max_span_tokens)
        start, stop = window_start, window_start + profile.max_span_tokens
    return can, {"start": start, "stop": stop, "delta": 0}


@dataclass(frozen=True)
class FragmentConstraintAdapter:
    """Structural adapter for the unchanged V1 frontier engine."""

    warmup_attempts: int = 25
    feasible_rate: float = 0.35
    plateau_patience: int = 2
    collapse_threshold: float = 0.75

    def classify(
        self,
        *,
        attempts,
        structural_success_rate,
        valid_rate,
        stagnant_rounds,
        largest_root_fraction,
        available_operators=None,
        geometry=None,
    ):
        if int(attempts) < int(self.warmup_attempts):
            return "warmup"
        if float(structural_success_rate) >= float(self.feasible_rate):
            return "feasible"
        if float(largest_root_fraction) >= float(self.collapse_threshold):
            return "collapsed"
        if int(stagnant_rounds) >= int(self.plateau_patience):
            return "plateau"
        if float(valid_rate) < 0.20:
            return "sparse"
        return "search"

    @staticmethod
    def group_fractions(state, context=None):
        return {"proposal": 1.0}

    @staticmethod
    def insertion_fraction(state, context=None):
        return {
            "warmup": 0.55,
            "feasible": 0.40,
            "plateau": 0.70,
            "collapsed": 0.75,
            "sparse": 0.50,
            "search": 0.60,
        }.get(state, 0.55)

    @staticmethod
    def operator_priors(group, state, context=None):
        if group != "proposal":
            return {}
        context = dict(context or {})
        available = set(context.get("available_operators", OPERATORS))
        geometry = context.get("geometry")
        priors = {
            "warmup": {
                "legacy_completion": 0.35,
                "anchor_growth": 0.30,
                "local_repair": 0.15,
                "structural_restart": 0.20,
            },
            "feasible": {
                "legacy_completion": 0.15,
                "anchor_growth": 0.25,
                "local_repair": 0.40,
                "structural_restart": 0.10,
            },
            "plateau": {
                "legacy_completion": 0.15,
                "anchor_growth": 0.20,
                "local_repair": 0.10,
                "structural_restart": 0.30,
            },
            "collapsed": {
                "legacy_completion": 0.20,
                "anchor_growth": 0.20,
                "local_repair": 0.05,
                "structural_restart": 0.35,
            },
            "sparse": {
                "legacy_completion": 0.30,
                "anchor_growth": 0.25,
                "local_repair": 0.10,
                "structural_restart": 0.25,
            },
            "search": {
                "legacy_completion": 0.20,
                "anchor_growth": 0.30,
                "local_repair": 0.20,
                "structural_restart": 0.15,
            },
        }.get(state, {})
        semantic_operator = {
            "multi_anchor": "bridge_closure",
            "multi_attachment": "decoration_fill",
            "substructure_expand": "superstructure_expand",
        }.get(geometry)
        if semantic_operator in available:
            priors[semantic_operator] = {
                "warmup": 0.25,
                "feasible": 0.20,
                "plateau": 0.35,
                "collapsed": 0.35,
                "sparse": 0.30,
                "search": 0.30,
            }.get(state, 0.25)
        return {
            operator: weight
            for operator, weight in priors.items()
            if operator in available and weight > 0.0
        }

    @staticmethod
    def operator_floors(group, state, context=None):
        if group != "proposal":
            return {}
        available = set(dict(context or {}).get("available_operators", OPERATORS))
        if "legacy_completion" not in available:
            return {}
        return {"legacy_completion": 0.15}

    @staticmethod
    def batch_reward(transitions: Iterable[dict]):
        rows = list(transitions)
        if not rows:
            return 0.0, {
                "strict_rate": 0.0,
                "valid_rate": 0.0,
                "preserved_fraction": 0.0,
                "structural_tail": 0.0,
            }
        scores = sorted(float(row.get("structural_score", 0.0)) for row in rows)
        tail_n = max(1, int(math.ceil(0.25 * len(scores))))
        tail = sum(scores[-tail_n:]) / tail_n
        mean = sum(scores) / len(scores)
        reward = min(1.0, 0.70 * tail + 0.30 * mean)
        return reward, {
            "strict_rate": sum(bool(row.get("strict", False)) for row in rows)
            / len(rows),
            "valid_rate": sum(bool(row.get("valid", False)) for row in rows)
            / len(rows),
            "preserved_fraction": sum(
                float(row.get("preserved_fraction", 0.0)) for row in rows
            )
            / len(rows),
            "structural_tail": tail,
        }


@dataclass(frozen=True)
class FragmentConstraintAdapterV2:
    """Novelty-aware structural adapter for fragment-constrained generation.

    Structural preservation remains a hard requirement. Once that requirement
    is routinely met, the adapter rewards incremental unique yield and balanced
    use of independent seed lineages. It never observes the target molecule,
    QED, SA, diversity, or the benchmark distance metric.
    """

    warmup_attempts: int = 25
    feasible_rate: float = 0.35
    unique_target: float = 0.70
    plateau_patience: int = 2
    collapse_threshold: float = 0.75

    def classify(
        self,
        *,
        attempts,
        structural_success_rate,
        unique_success_rate,
        valid_rate,
        stagnant_rounds,
        largest_root_fraction,
        available_operators=None,
        geometry=None,
        dynamic_seed_pool_size=None,
    ):
        if int(attempts) < int(self.warmup_attempts):
            return "warmup"
        if float(valid_rate) < 0.20:
            return "sparse"
        if float(structural_success_rate) >= float(self.feasible_rate):
            if float(unique_success_rate) < float(self.unique_target):
                return "diversify"
            return "feasible"
        if float(largest_root_fraction) >= float(self.collapse_threshold):
            return "collapsed"
        if int(stagnant_rounds) >= int(self.plateau_patience):
            return "plateau"
        return "search"

    @staticmethod
    def group_fractions(state, context=None):
        return {"proposal": 1.0}

    @staticmethod
    def insertion_fraction(state, context=None):
        return {
            "warmup": 0.60,
            "feasible": 0.48,
            "diversify": 0.70,
            "plateau": 0.76,
            "collapsed": 0.80,
            "sparse": 0.52,
            "search": 0.64,
        }.get(state, 0.60)

    @staticmethod
    def operator_priors(group, state, context=None):
        if group != "proposal":
            return {}
        context = dict(context or {})
        available = set(context.get("available_operators", OPERATORS))
        geometry = context.get("geometry")
        priors = {
            "warmup": {
                "legacy_completion": 0.15,
                "anchor_growth": 0.25,
                "local_repair": 0.10,
                "structural_restart": 0.30,
            },
            "feasible": {
                "legacy_completion": 0.15,
                "anchor_growth": 0.30,
                "local_repair": 0.15,
                "structural_restart": 0.20,
            },
            "diversify": {
                "legacy_completion": 0.05,
                "anchor_growth": 0.25,
                "local_repair": 0.05,
                "structural_restart": 0.40,
            },
            "plateau": {
                "legacy_completion": 0.08,
                "anchor_growth": 0.24,
                "local_repair": 0.04,
                "structural_restart": 0.39,
            },
            "collapsed": {
                "legacy_completion": 0.08,
                "anchor_growth": 0.22,
                "local_repair": 0.03,
                "structural_restart": 0.42,
            },
            "sparse": {
                "legacy_completion": 0.20,
                "anchor_growth": 0.25,
                "local_repair": 0.10,
                "structural_restart": 0.25,
            },
            "search": {
                "legacy_completion": 0.12,
                "anchor_growth": 0.30,
                "local_repair": 0.08,
                "structural_restart": 0.30,
            },
        }.get(state, {})
        semantic_operator = {
            "multi_anchor": "bridge_closure",
            "multi_attachment": "decoration_fill",
            "substructure_expand": "superstructure_expand",
        }.get(geometry)
        if semantic_operator in available:
            priors[semantic_operator] = {
                "warmup": 0.20,
                "feasible": 0.20,
                "diversify": 0.25,
                "plateau": 0.25,
                "collapsed": 0.25,
                "sparse": 0.20,
                "search": 0.20,
            }.get(state, 0.20)
        return {
            operator: weight
            for operator, weight in priors.items()
            if operator in available and weight > 0.0
        }

    @staticmethod
    def operator_floors(group, state, context=None):
        if group != "proposal":
            return {}
        context = dict(context or {})
        available = set(context.get("available_operators", OPERATORS))
        floors = {}
        if "legacy_completion" in available:
            floors["legacy_completion"] = 0.05
        if state in {"diversify", "plateau", "collapsed"}:
            if "structural_restart" in available:
                floors["structural_restart"] = 0.15
            semantic_operator = {
                "multi_anchor": "bridge_closure",
                "multi_attachment": "decoration_fill",
                "substructure_expand": "superstructure_expand",
            }.get(context.get("geometry"))
            if semantic_operator in available:
                floors[semantic_operator] = 0.10
        return floors

    @staticmethod
    def batch_reward(transitions: Iterable[dict]):
        rows = list(transitions)
        if not rows:
            return 0.0, {
                "strict_rate": 0.0,
                "novel_rate": 0.0,
                "lineage_credit": 0.0,
                "structural_mean": 0.0,
            }
        strict_rate = sum(bool(row.get("strict", False)) for row in rows) / len(rows)
        novel_rate = sum(
            bool(row.get("strict", False)) and bool(row.get("novel", False))
            for row in rows
        ) / len(rows)
        lineage_credit = sum(
            float(row.get("lineage_credit", 0.0))
            if bool(row.get("strict", False))
            else 0.0
            for row in rows
        ) / len(rows)
        structural_mean = sum(
            float(row.get("structural_score", 0.0)) for row in rows
        ) / len(rows)
        reward = (
            0.20 * strict_rate
            + 0.55 * novel_rate
            + 0.10 * lineage_credit
            + 0.15 * structural_mean
        )
        return max(0.0, min(1.0, reward)), {
            "strict_rate": strict_rate,
            "novel_rate": novel_rate,
            "lineage_credit": lineage_credit,
            "structural_mean": structural_mean,
        }


def largest_root_fraction(root_ids: Iterable[int]) -> float:
    counts = Counter(int(root_id) for root_id in root_ids)
    total = sum(counts.values())
    return max(counts.values()) / total if total else 0.0
