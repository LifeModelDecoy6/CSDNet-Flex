import re


SMILES_REGEX = re.compile(
    r"(\%\d{2}"
    r"|Br|Cl"
    r"|\[[^\[\]]+\]"
    r"|\("
    r"|\)"
    r"|\.|=|#|-|\+|\\|/|:"
    r"|~|@|\?"
    r"|>>?"
    r"|\*|\$"
    r"|[0-9]"
    r"|[A-Za-z])"
)


def tokenize_smiles(smiles):
    return SMILES_REGEX.findall(smiles or "")


def is_aromatic_token(tok):
    if tok in {"b", "c", "n", "o", "p", "s", "se", "as"}:
        return True
    if tok.startswith("[") and tok.endswith("]") and len(tok) > 2:
        inner = tok[1:-1]
        return bool(inner) and inner[0].islower()
    return False


def is_ring_token(tok):
    return tok.isdigit() or (tok.startswith("%") and tok[1:].isdigit())


def ring_number(tok):
    if tok.isdigit():
        return int(tok)
    if tok.startswith("%") and tok[1:].isdigit():
        return int(tok[1:])
    return None


def is_bond_token(tok):
    return tok in {"-", "=", "#", "$", ":", "/", "\\", "~"}


class SMILESTokenizer:
    def __init__(self, vocab):
        self.vocab = {t: i for i, t in enumerate(vocab)}
        self.inv = {i: t for t, i in self.vocab.items()}
        self.vocab_size = len(vocab)
        self.pad_id = self.vocab["<pad>"]
        self.mask_id = self.vocab["<mask>"]
        self.bos_id = self.vocab["<bos>"]
        self.eos_id = self.vocab["<eos>"]
        self.unk_id = self.vocab.get("<unk>", -1)

        scaffold_chars = set("bcnsopBCNSOP1234567890()=#-:/\\")
        self.scaffold_ids = set()
        self.aromatic_ids = set()
        for tok, idx in self.vocab.items():
            if any(c in scaffold_chars for c in tok) or tok.startswith("%"):
                self.scaffold_ids.add(idx)
            if is_aromatic_token(tok):
                self.aromatic_ids.add(idx)

    @staticmethod
    def _is_aromatic_token(tok):
        return is_aromatic_token(tok)

    def aromatic_context_mask(self, smi, max_len):
        tokens = tokenize_smiles(smi)
        max_body = max(0, max_len - 2)
        tokens = tokens[:max_body]
        mask = [0.0] * max_len

        active_pos = None
        active_aromatic = False
        branch_stack = []
        rings = {}
        pending_bond_pos = None
        pending_bond_tok = None

        def mark(pos):
            if pos is not None and 0 <= pos < max_len:
                mask[pos] = 1.0

        for body_idx, tok in enumerate(tokens, start=1):
            if is_bond_token(tok):
                pending_bond_pos = body_idx
                pending_bond_tok = tok
                if tok == ":":
                    mark(body_idx)
                continue

            if tok == "(":
                if active_pos is not None:
                    branch_stack.append((active_pos, active_aromatic))
                pending_bond_pos = None
                pending_bond_tok = None
                continue

            if tok == ")":
                if branch_stack:
                    active_pos, active_aromatic = branch_stack.pop()
                pending_bond_pos = None
                pending_bond_tok = None
                continue

            if tok == ".":
                active_pos = None
                active_aromatic = False
                pending_bond_pos = None
                pending_bond_tok = None
                continue

            rnum = ring_number(tok)
            if rnum is not None:
                if active_pos is not None:
                    if active_aromatic:
                        mark(body_idx)
                        mark(active_pos)
                    if rnum in rings:
                        open_pos, open_aromatic, open_ring_pos, open_bond_pos = rings.pop(rnum)
                        aromatic_edge = (
                            (open_aromatic and active_aromatic)
                            or pending_bond_tok == ":"
                            or open_bond_pos is not None
                        )
                        if aromatic_edge:
                            mark(open_pos)
                            mark(active_pos)
                            mark(open_ring_pos)
                            mark(body_idx)
                            mark(open_bond_pos)
                            mark(pending_bond_pos)
                    else:
                        rings[rnum] = (
                            active_pos,
                            active_aromatic,
                            body_idx,
                            pending_bond_pos if pending_bond_tok == ":" else None,
                        )
                pending_bond_pos = None
                pending_bond_tok = None
                continue

            atom_aromatic = is_aromatic_token(tok)
            if atom_aromatic:
                mark(body_idx)

            if active_pos is not None:
                aromatic_edge = (
                    (active_aromatic and atom_aromatic)
                    or pending_bond_tok == ":"
                )
                if aromatic_edge:
                    mark(active_pos)
                    mark(body_idx)
                    mark(pending_bond_pos)

            active_pos = body_idx
            active_aromatic = atom_aromatic
            pending_bond_pos = None
            pending_bond_tok = None

        return mask

    def encode(self, smi, max_len):
        if max_len < 2:
            raise ValueError("max_len must be at least 2 to hold BOS/EOS tokens")

        tokens = ["<bos>"] + tokenize_smiles(smi) + ["<eos>"]
        ids = [self.vocab.get(t, self.unk_id) for t in tokens]

        if len(ids) > max_len:
            ids = ids[:max_len]
            ids[-1] = self.eos_id
        else:
            ids += [self.pad_id] * (max_len - len(ids))
        return ids

    def token_length(self, smi, include_special=True):
        length = len(tokenize_smiles(smi))
        return length + 2 if include_special else length

    def decode(self, ids):
        toks = []
        for i in ids:
            if i == self.eos_id:
                break
            if i in (self.pad_id, self.bos_id):
                continue
            toks.append(self.inv.get(i, ""))
        return "".join(toks)
