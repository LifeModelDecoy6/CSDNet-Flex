import os


SMILES_COLUMN_CANDIDATES = (
    "smiles",
    "SMILES",
    "canonical_smiles",
    "canonicalized_smiles",
    "text",
)


def load_dataset_with_optional_token(dataset_name, split, streaming=False, token=None):
    try:
        from datasets import load_dataset
    except ImportError as exc:
        raise SystemExit("缺少 datasets 依赖，请先安装项目依赖。") from exc

    kwargs = dict(split=split, streaming=streaming)
    if token:
        kwargs["token"] = token
    try:
        return load_dataset(dataset_name, **kwargs)
    except TypeError:
        if token:
            kwargs.pop("token", None)
            kwargs["use_auth_token"] = token
        return load_dataset(dataset_name, **kwargs)


def resolve_hf_token(token=None, token_env="HF_TOKEN", required=False):
    resolved = token or (os.environ.get(token_env) if token_env else None)
    if required and not resolved:
        raise SystemExit(
            f"缺少 HuggingFace token。请设置 {token_env}，或传入 --hf_token。"
        )
    return resolved


def decode_safe_to_smiles(safe_string):
    try:
        import safe as sf
    except Exception:
        return ""

    try:
        smi = sf.decode(safe_string, canonical=True, ignore_errors=True)
    except Exception:
        return ""
    return smi or ""


def extract_smiles_from_row(
    row,
    smiles_col="auto",
    safe_col="safe",
    allow_safe_decode=True,
):
    if smiles_col and smiles_col != "auto":
        value = row.get(smiles_col, "")
        return str(value).strip() if value is not None else ""

    for col in SMILES_COLUMN_CANDIDATES:
        value = row.get(col)
        if value:
            return str(value).strip()

    if allow_safe_decode and safe_col:
        safe_value = row.get(safe_col)
        if safe_value:
            return decode_safe_to_smiles(str(safe_value))

    return ""
