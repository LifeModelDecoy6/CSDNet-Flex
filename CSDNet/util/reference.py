import itertools

from CSDNet.util.hf_smiles import (
    extract_smiles_from_row,
    load_dataset_with_optional_token,
    resolve_hf_token,
)
from CSDNet.util.tokenizer import tokenize_smiles


def build_reference_from_hf_stream(args, tk):
    token = resolve_hf_token(args.hf_token, args.hf_token_env, required=False)
    ds = load_dataset_with_optional_token(
        args.hf_dataset,
        split=args.hf_split,
        streaming=args.hf_streaming,
        token=token,
    )

    ref_lengths = []
    train_set = set()
    iterator = ds if args.hf_streaming else iter(ds)
    if args.hf_scan_limit > 0:
        iterator = itertools.islice(iterator, args.hf_scan_limit)

    for row in iterator:
        smi = extract_smiles_from_row(
            row,
            smiles_col=args.hf_smiles_col,
            safe_col=args.hf_safe_col,
            allow_safe_decode=not args.disable_hf_safe_decode,
        )
        if not smi:
            continue
        tokens = tokenize_smiles(smi)
        if args.hf_skip_long and len(tokens) + 2 > args.max_len:
            continue
        if args.hf_skip_unknown and tk.unk_id != -1:
            if any(tok not in tk.vocab for tok in tokens):
                continue
        if len(ref_lengths) < args.hf_ref_sample_n:
            ref_lengths.append(max(3, min(len(tokens) + 2, args.max_len)))
        if len(train_set) < args.hf_novelty_sample_n:
            train_set.add(smi)
        if len(ref_lengths) >= args.hf_ref_sample_n and len(train_set) >= args.hf_novelty_sample_n:
            break

    if not ref_lengths:
        raise SystemExit("No usable SMILES read from HF reference dataset.")
    return ref_lengths, train_set


def build_reference_from_disk(args, tk):
    try:
        from datasets import load_from_disk
    except ImportError as exc:
        raise SystemExit("Missing datasets dependency.") from exc

    ds = load_from_disk(args.data_dir)
    smiles_col = "text" if "text" in ds.column_names else "smiles"
    sample_n = min(50000, len(ds))
    sample_texts = ds.select(range(sample_n))[smiles_col]
    ref_lengths = [
        max(3, min(len(tokenize_smiles(smi)) + 2, args.max_len))
        for smi in sample_texts
    ]
    train_set = set(ds[smiles_col])
    return ref_lengths, train_set
