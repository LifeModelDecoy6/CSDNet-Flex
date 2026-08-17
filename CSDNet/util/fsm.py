import json
import os
from collections import Counter

import torch

from CSDNet.util.paths import DEFAULT_VALENCE_DICT


class SmilesSyntaxTracker:
    """Localize definite grammar failures in fully decoded SMILES rows.

    Square-bracket atoms are indivisible tokenizer tokens, so balanced square
    brackets are guaranteed by construction. The tracker still rejects any
    bracket token that is not in the atom table. Rows containing masks are
    left untouched because an unknown token can still repair their grammar.
    """

    BOND_TOKENS = {"-", "=", "#", "$", ":", "/", "\\", "~"}
    AROMATIC_TO_ALIPHATIC = {
        "b": "B",
        "c": "C",
        "n": "N",
        "o": "O",
        "p": "P",
        "s": "S",
        "se": "Se",
        "as": "As",
    }

    def __init__(self, tokenizer, is_atom, ring_numbers):
        self.tk = tokenizer
        self.is_atom = list(is_atom)
        self.ring_numbers = list(ring_numbers)
        vocab_size = len(tokenizer.vocab)
        self.is_bond = [False] * vocab_size
        self.is_open_branch = [False] * vocab_size
        self.is_close_branch = [False] * vocab_size
        self.is_dot = [False] * vocab_size
        self.is_invalid_bracket = [False] * vocab_size
        self.aromatic_to_aliphatic = [-1] * vocab_size

        for token, token_id in tokenizer.vocab.items():
            self.is_bond[token_id] = token in self.BOND_TOKENS
            self.is_open_branch[token_id] = token == "("
            self.is_close_branch[token_id] = token == ")"
            self.is_dot[token_id] = token == "."
            self.is_invalid_bracket[token_id] = (
                token.startswith("[")
                and token.endswith("]")
                and not self.is_atom[token_id]
            )
            aliphatic = self.AROMATIC_TO_ALIPHATIC.get(token)
            if aliphatic in tokenizer.vocab:
                self.aromatic_to_aliphatic[token_id] = tokenizer.vocab[
                    aliphatic
                ]

    def project_completed_sequence(
        self,
        token_ids,
        editable=None,
        constraints=None,
    ):
        """Project a decoded row onto a conservative SMILES grammar subset.

        The operation is deterministic and proposal preserving: it never
        samples, discards, or replaces a molecule with another proposal. Only
        editable syntax tokens are removed, and simple aromatic atom tokens
        are converted to their aliphatic counterparts when their connected
        component contains no paired ring closure and therefore cannot be an
        aromatic cycle.

        ``editable`` and ``constraints`` are transformed alongside the token
        sequence so protected local-infill context remains aligned.
        """
        sequence = list(token_ids)
        if editable is None:
            editable = [
                token_id
                not in {
                    self.tk.pad_id,
                    self.tk.bos_id,
                    self.tk.eos_id,
                }
                for token_id in sequence
            ]
        else:
            editable = list(editable)
        if constraints is None:
            constraints = [None] * len(sequence)
        else:
            constraints = list(constraints)
        if len(editable) != len(sequence) or len(constraints) != len(sequence):
            raise ValueError(
                "FSM projection metadata must align with the token sequence"
            )

        records = []
        for token_id, can_edit, constraint in zip(
            sequence,
            editable,
            constraints,
        ):
            if token_id == self.tk.pad_id:
                continue
            records.append([int(token_id), bool(can_edit), constraint])
            if token_id == self.tk.eos_id:
                break
        if not records or records[0][0] != self.tk.bos_id:
            records.insert(0, [self.tk.bos_id, False, None])
        if records[-1][0] != self.tk.eos_id:
            records.append([self.tk.eos_id, False, None])

        diagnostics = Counter()

        def remove_indices(indices, reason):
            removable = {
                index
                for index in indices
                if 0 <= index < len(records) and records[index][1]
            }
            if not removable:
                return False
            diagnostics[reason] += len(removable)
            records[:] = [
                record
                for index, record in enumerate(records)
                if index not in removable
            ]
            return True

        # Removing one malformed delimiter can expose another. A short fixed
        # point iteration is sufficient because every successful pass strictly
        # shortens the sequence.
        for _ in range(8):
            changed = False
            branch_stack = []
            branch_has_atom = []
            extra_closes = []
            empty_pairs = []
            for index, (token_id, _, _) in enumerate(records):
                if self.is_open_branch[token_id]:
                    branch_stack.append(index)
                    branch_has_atom.append(False)
                elif self.is_close_branch[token_id]:
                    if not branch_stack:
                        extra_closes.append(index)
                    else:
                        open_index = branch_stack.pop()
                        has_atom = branch_has_atom.pop()
                        if not has_atom:
                            empty_pairs.append((open_index, index))
                elif self.is_atom[token_id] and branch_has_atom:
                    branch_has_atom[-1] = True

            if extra_closes:
                changed |= remove_indices(
                    extra_closes,
                    "removed_extra_close_branch",
                )
            if not changed and branch_stack:
                changed |= remove_indices(
                    reversed(branch_stack),
                    "removed_unclosed_branch",
                )
            if not changed and empty_pairs:
                for open_index, close_index in reversed(empty_pairs):
                    if records[open_index][1] and records[close_index][1]:
                        changed |= remove_indices(
                            (open_index, close_index),
                            "removed_empty_branch_delimiter",
                        )
                        break
            if changed:
                continue

            ring_positions = {}
            for index, (token_id, _, _) in enumerate(records):
                ring_number = self.ring_numbers[token_id]
                if ring_number >= 0:
                    ring_positions.setdefault(ring_number, []).append(index)
            for positions in ring_positions.values():
                if len(positions) % 2 == 0:
                    continue
                removable = next(
                    (
                        index
                        for index in reversed(positions)
                        if records[index][1]
                    ),
                    None,
                )
                if removable is None:
                    continue
                to_remove = [removable]
                previous = removable - 1
                if (
                    previous > 0
                    and records[previous][1]
                    and self.is_bond[records[previous][0]]
                ):
                    to_remove.append(previous)
                changed |= remove_indices(
                    to_remove,
                    "removed_unmatched_ring",
                )
                break
            if changed:
                continue

            malformed = []
            body_indices = [
                index
                for index, (token_id, _, _) in enumerate(records)
                if token_id not in {
                    self.tk.bos_id,
                    self.tk.eos_id,
                    self.tk.pad_id,
                }
            ]
            for body_offset, index in enumerate(body_indices):
                token_id = records[index][0]
                if not self.is_bond[token_id] and not self.is_dot[token_id]:
                    continue
                previous_id = (
                    records[body_indices[body_offset - 1]][0]
                    if body_offset > 0
                    else None
                )
                next_id = (
                    records[body_indices[body_offset + 1]][0]
                    if body_offset + 1 < len(body_indices)
                    else None
                )
                previous_invalid = (
                    previous_id is None
                    or self.is_bond[previous_id]
                    or self.is_dot[previous_id]
                )
                next_invalid = (
                    next_id is None
                    or self.is_bond[next_id]
                    or self.is_close_branch[next_id]
                    or self.is_dot[next_id]
                )
                if previous_invalid or next_invalid:
                    malformed.append(index)
            if malformed:
                changed |= remove_indices(
                    malformed,
                    "removed_malformed_bond_or_dot",
                )
            if not changed:
                break

        # Aromatic atoms are cyclic by definition. If a disconnected SMILES
        # component has no paired ring label, simple lowercase aromatic atom
        # tokens cannot be valid and are projected to the corresponding
        # aliphatic elements. Bracket atoms are deliberately left to RDKit and
        # neural repair because their charge/hydrogen semantics are richer.
        component_start = 1
        for component_stop in range(1, len(records)):
            token_id = records[component_stop][0]
            is_boundary = (
                token_id == self.tk.eos_id or self.is_dot[token_id]
            )
            if not is_boundary:
                continue
            component = range(component_start, component_stop)
            ring_counts = Counter(
                self.ring_numbers[records[index][0]]
                for index in component
                if self.ring_numbers[records[index][0]] >= 0
            )
            has_ring_pair = any(count >= 2 for count in ring_counts.values())
            if not has_ring_pair:
                for index in component:
                    token_id = records[index][0]
                    replacement = self.aromatic_to_aliphatic[token_id]
                    if replacement >= 0 and records[index][1]:
                        records[index][0] = replacement
                        diagnostics["normalized_ringless_aromatic"] += 1
            component_start = component_stop + 1

        projected = [record[0] for record in records]
        projected_editable = [record[1] for record in records]
        projected_constraints = [record[2] for record in records]
        diagnostics["sequence_changed"] = int(projected != sequence)
        return (
            projected,
            projected_editable,
            projected_constraints,
            dict(diagnostics),
        )

    def compute_penalties(self, token_ids_batch):
        bsz, seq_len = token_ids_batch.shape
        penalties = torch.zeros(
            (bsz, seq_len),
            device=token_ids_batch.device,
            dtype=torch.float,
        )
        sequences = token_ids_batch.cpu().tolist()
        special_ids = {
            self.tk.pad_id,
            self.tk.bos_id,
            self.tk.eos_id,
        }
        unk_id = getattr(self.tk, "unk_id", -1)

        for row, sequence in enumerate(sequences):
            body = []
            has_mask = False
            for position, token_id in enumerate(sequence):
                if token_id == self.tk.eos_id:
                    break
                if token_id == self.tk.mask_id:
                    has_mask = True
                    continue
                if token_id in special_ids:
                    continue
                body.append((position, token_id))

            # Partial sequences do not admit a definite global grammar
            # diagnosis; a remaining mask may supply any missing delimiter.
            if has_mask:
                continue

            branch_stack = []
            rings = {}
            active_atom = -1
            pending_bond = -1
            trailing_dot = -1

            def mark(*positions):
                for position in positions:
                    if position is not None and position >= 0:
                        penalties[row, position] = -1000.0

            for position, token_id in body:
                if token_id == unk_id or self.is_invalid_bracket[token_id]:
                    mark(position)
                    continue

                if self.is_bond[token_id]:
                    if active_atom < 0 or pending_bond >= 0:
                        mark(position, pending_bond)
                    pending_bond = position
                    trailing_dot = -1
                    continue

                if self.is_open_branch[token_id]:
                    if active_atom < 0 or pending_bond >= 0:
                        mark(position, pending_bond)
                    if branch_stack and not branch_stack[-1][2]:
                        mark(position, branch_stack[-1][1])
                    branch_stack.append([active_atom, position, False])
                    pending_bond = -1
                    trailing_dot = -1
                    continue

                if self.is_close_branch[token_id]:
                    if pending_bond >= 0:
                        mark(pending_bond, position)
                    if not branch_stack:
                        mark(position)
                    else:
                        parent_atom, open_position, has_branch_atom = (
                            branch_stack.pop()
                        )
                        if not has_branch_atom or active_atom < 0:
                            mark(open_position, position, trailing_dot)
                        active_atom = parent_atom
                    pending_bond = -1
                    trailing_dot = -1
                    continue

                ring_number = self.ring_numbers[token_id]
                if ring_number >= 0:
                    if active_atom < 0:
                        mark(position, pending_bond)
                    elif branch_stack and not branch_stack[-1][2]:
                        mark(position, branch_stack[-1][1])
                    elif ring_number in rings:
                        start_atom, start_position = rings.pop(ring_number)
                        if start_atom == active_atom:
                            mark(start_atom, start_position, position)
                    else:
                        rings[ring_number] = (active_atom, position)
                    pending_bond = -1
                    trailing_dot = -1
                    continue

                if self.is_dot[token_id]:
                    if active_atom < 0 or pending_bond >= 0:
                        mark(position, pending_bond, trailing_dot)
                    active_atom = -1
                    pending_bond = -1
                    trailing_dot = position
                    continue

                if self.is_atom[token_id]:
                    active_atom = position
                    pending_bond = -1
                    trailing_dot = -1
                    if branch_stack:
                        branch_stack[-1][2] = True
                    continue

                # Every non-special vocabulary token must have a recognized
                # role in the atom-level SMILES grammar.
                mark(position)

            mark(pending_bond, trailing_dot)
            for _, open_position, _ in branch_stack:
                mark(open_position)
            for start_atom, ring_position in rings.values():
                mark(start_atom, ring_position)

            # Lowercase aromatic atoms require a cyclic aromatic component.
            # This deliberately handles only the definite case needed for
            # sampling repair: a component with aromatic atom tokens but no
            # paired ring label cannot become a valid aromatic cycle once the
            # row is fully decoded. Components containing a ring pair are left
            # to RDKit, which understands fused and charged aromatic systems.
            component = []
            for position, token_id in body + [(-1, self.tk.eos_id)]:
                if position >= 0 and not self.is_dot[token_id]:
                    component.append((position, token_id))
                    continue
                ring_counts = Counter(
                    self.ring_numbers[value]
                    for _, value in component
                    if self.ring_numbers[value] >= 0
                )
                has_ring_pair = any(
                    count >= 2 for count in ring_counts.values()
                )
                if not has_ring_pair:
                    for aromatic_position, aromatic_id in component:
                        if self.aromatic_to_aliphatic[aromatic_id] >= 0:
                            mark(aromatic_position)
                component = []

        return penalties


class ValenceFSMTracker:
    """Valence and syntax FSM used for late-stage sampling repair."""

    def __init__(self, tokenizer, dict_path=None):
        self.tk = tokenizer
        dict_path = dict_path or DEFAULT_VALENCE_DICT
        dict_path = os.fspath(dict_path)

        with open(dict_path, "r") as f:
            valency_dict = json.load(f)

        vocab_size = len(tokenizer.vocab)
        self.quota_map = [0] * vocab_size
        self.is_atom = [False] * vocab_size
        self.bond_cost = [0] * vocab_size
        # -1 is the non-ring sentinel; zero is itself a valid ring label.
        self.is_ring = [-1] * vocab_size
        self.is_open_bracket = [False] * vocab_size
        self.is_close_bracket = [False] * vocab_size
        self.is_dot = [False] * vocab_size

        for tok, tid in tokenizer.vocab.items():
            if tok in valency_dict:
                self.quota_map[tid] = valency_dict[tok]
                self.is_atom[tid] = True
            elif tok == "=":
                self.bond_cost[tid] = 2
            elif tok == "#":
                self.bond_cost[tid] = 3
            elif tok in ("-", "/", "\\", ":"):
                self.bond_cost[tid] = 1
            elif tok == "$":
                self.bond_cost[tid] = 4
            elif tok == "(":
                self.is_open_bracket[tid] = True
            elif tok == ")":
                self.is_close_bracket[tid] = True
            elif tok == ".":
                self.is_dot[tid] = True
            elif tok.isdigit():
                self.is_ring[tid] = int(tok)
            elif tok.startswith("%") and tok[1:].isdigit():
                self.is_ring[tid] = int(tok[1:])

        self.syntax_tracker = SmilesSyntaxTracker(
            tokenizer,
            is_atom=self.is_atom,
            ring_numbers=self.is_ring,
        )

    def compute_penalties(self, token_ids_batch):
        bsz, seq_len = token_ids_batch.shape
        penalties = torch.zeros(
            (bsz, seq_len),
            device=token_ids_batch.device,
            dtype=torch.float,
        )
        seqs = token_ids_batch.cpu().tolist()

        for b in range(bsz):
            seq = seqs[b]
            stack = []
            rings = {}
            active_idx = -1
            b_cost = 1
            bond_idx = -1
            quotas = {}
            parents = {}

            for i, tid in enumerate(seq):
                if tid in (self.tk.pad_id, self.tk.bos_id, self.tk.eos_id, self.tk.mask_id):
                    continue

                if self.bond_cost[tid] > 0:
                    if bond_idx != -1:
                        # A bond may qualify an atom or ring closure, but two
                        # consecutive bond symbols are never valid SMILES.
                        penalties[b, bond_idx] = -1000.0
                        penalties[b, i] = -1000.0
                    if active_idx == -1:
                        # This also catches an explicit single bond at the
                        # beginning of a component, which b_cost alone cannot.
                        penalties[b, i] = -1000.0
                    b_cost = self.bond_cost[tid]
                    bond_idx = i
                    continue

                if self.is_dot[tid]:
                    active_idx = -1
                    b_cost = 1
                    bond_idx = -1
                    continue

                if self.is_open_bracket[tid]:
                    if active_idx != -1:
                        stack.append((active_idx, b_cost, i, False))
                    else:
                        penalties[b, i] = -1000.0
                    if b_cost != 1 and bond_idx != -1:
                        penalties[b, bond_idx] = -1000.0
                        penalties[b, i] = -1000.0
                    b_cost = 1
                    bond_idx = -1
                    continue

                if self.is_close_bracket[tid]:
                    if stack:
                        active_idx, b_cost, open_pos, branch_has_atom = stack.pop()
                        if not branch_has_atom:
                            penalties[b, open_pos] = -1000.0
                            penalties[b, i] = -1000.0
                    else:
                        penalties[b, i] = -1000.0
                    bond_idx = -1
                    continue

                r_num = self.is_ring[tid]
                if r_num >= 0:
                    if r_num in rings:
                        start_idx, cost, start_pos = rings.pop(r_num)
                        ring_cost = max(cost, b_cost) if bond_idx != -1 else cost
                        if active_idx != -1 and start_idx != -1:
                            if active_idx == start_idx:
                                penalties[b, i] = -1000.0
                                penalties[b, start_pos] = -1000.0
                                penalties[b, active_idx] = -1000.0
                            elif parents.get(active_idx) == start_idx or parents.get(start_idx) == active_idx:
                                penalties[b, i] = -1000.0
                                penalties[b, start_pos] = -1000.0
                                penalties[b, active_idx] = -1000.0
                                penalties[b, start_idx] = -1000.0
                            else:
                                quotas[active_idx] -= ring_cost
                                quotas[start_idx] -= ring_cost
                                if quotas[active_idx] < 0:
                                    penalties[b, active_idx] = -1000.0
                                if quotas[start_idx] < 0:
                                    penalties[b, start_idx] = -1000.0
                        else:
                            penalties[b, i] = -1000.0
                            penalties[b, start_pos] = -1000.0
                    else:
                        rings[r_num] = (active_idx, b_cost, i)
                        if active_idx == -1:
                            penalties[b, i] = -1000.0
                    b_cost = 1
                    bond_idx = -1
                    continue

                if self.is_atom[tid]:
                    quota = self.quota_map[tid]
                    quotas[i] = quota
                    if stack:
                        parent_idx, parent_cost, open_pos, _ = stack[-1]
                        stack[-1] = (parent_idx, parent_cost, open_pos, True)

                    if active_idx != -1:
                        parents[i] = active_idx
                        quotas[active_idx] -= b_cost
                        quotas[i] -= b_cost
                        if quotas[active_idx] < 0:
                            penalties[b, active_idx] = -1000.0
                        if quotas[i] < 0:
                            penalties[b, i] = -1000.0
                    elif b_cost != 1 and bond_idx != -1:
                        penalties[b, bond_idx] = -1000.0

                    active_idx = i
                    b_cost = 1
                    bond_idx = -1

            for _, _, open_pos, _ in stack:
                penalties[b, open_pos] = -1000.0
            for start_idx, _, ring_pos in rings.values():
                penalties[b, ring_pos] = -1000.0
                if start_idx != -1:
                    penalties[b, start_idx] = -1000.0
            if b_cost != 1 and bond_idx != -1:
                penalties[b, bond_idx] = -1000.0

        penalties += self.syntax_tracker.compute_penalties(token_ids_batch)
        return penalties


def prepare_rdkit_kekulize_checker(tk, fsm_tracker=None):
    try:
        from rdkit import Chem, RDLogger

        RDLogger.DisableLog("rdApp.*")
    except Exception:
        return None

    focus_ids = []
    aromatic_atoms = {"b", "c", "n", "o", "p", "s"}
    bond_tokens = {"=", "#", "-", "/", "\\", ":", "$"}
    for tok, tid in tk.vocab.items():
        is_aromatic = tok in aromatic_atoms
        if tok.startswith("[") and len(tok) > 2:
            inner = tok[1:-1]
            is_aromatic = is_aromatic or bool(inner) and inner[0].islower()
        is_ring = tok.isdigit() or (
            tok.startswith("%") and tok[1:].isdigit()
        )
        is_bond = tok in bond_tokens
        if fsm_tracker is not None:
            is_ring = fsm_tracker.is_ring[tid] >= 0
            is_bond = fsm_tracker.bond_cost[tid] > 0
        if is_aromatic or is_ring or is_bond:
            focus_ids.append(tid)

    return Chem, set(focus_ids)


def rdkit_smiles_is_valid(smiles, chem=None):
    """Return whether a decoded SMILES passes full RDKit sanitization."""
    if not smiles:
        return False
    if chem is None:
        try:
            from rdkit import Chem as chem
        except Exception:
            return False
    try:
        return chem.MolFromSmiles(smiles) is not None
    except Exception:
        return False


def _mark_invalid_smiles_tokens(penalties, row, seq, special_ids, focus_ids=None):
    marked = False
    if focus_ids:
        for i, tid in enumerate(seq):
            if tid in special_ids:
                continue
            if tid in focus_ids:
                penalties[row, i] = -1000.0
                marked = True

    if not marked:
        for i, tid in enumerate(seq):
            if tid not in special_ids:
                penalties[row, i] = -1000.0


def compute_rdkit_sanitization_penalties(token_ids_batch, tk, chem, focus_ids):
    """Mark decoded rows that RDKit cannot parse or fully sanitize.

    Aromatic, ring, and bond tokens are preferred repair locations for
    kekulization failures. Other parser or sanitizer failures remask the whole
    non-special sequence because RDKit cannot reliably localize their source.
    """
    bsz, seq_len = token_ids_batch.shape
    penalties = torch.zeros(
        (bsz, seq_len),
        device=token_ids_batch.device,
        dtype=torch.float,
    )
    seqs = token_ids_batch.cpu().tolist()
    special_ids = {tk.pad_id, tk.bos_id, tk.eos_id, tk.mask_id}

    for b, seq in enumerate(seqs):
        smi = tk.decode(seq).strip("'\"")
        if not smi:
            continue

        mol = chem.MolFromSmiles(smi, sanitize=False)
        if mol is None:
            _mark_invalid_smiles_tokens(
                penalties,
                b,
                seq,
                special_ids,
            )
            continue

        try:
            chem.SanitizeMol(mol)
        except Exception as exc:
            focus = focus_ids if "kekulize" in str(exc).lower() else None
            _mark_invalid_smiles_tokens(
                penalties,
                b,
                seq,
                special_ids,
                focus,
            )

    return penalties


def compute_rdkit_kekulize_penalties(token_ids_batch, tk, chem, focus_ids):
    """Backward-compatible alias for the complete sanitization checker."""
    return compute_rdkit_sanitization_penalties(
        token_ids_batch,
        tk,
        chem,
        focus_ids,
    )


def expand_violation_mask(mask, valid_mask, radius=2):
    expanded = mask & valid_mask
    for shift in range(1, max(0, radius) + 1):
        left = torch.zeros_like(mask)
        right = torch.zeros_like(mask)
        left[:, shift:] = mask[:, :-shift]
        right[:, :-shift] = mask[:, shift:]
        expanded = expanded | left | right
    return expanded & valid_mask
