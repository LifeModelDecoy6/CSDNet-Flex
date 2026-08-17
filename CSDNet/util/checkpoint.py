import torch

from CSDNet.model.backbone import CSDNetBackbone
from CSDNet.model.elastic_backbone import ElasticCSDNetBackbone


def infer_backbone_config(state_dict, checkpoint):
    hparams = checkpoint.get("hyper_parameters", {}) if isinstance(checkpoint, dict) else {}
    architecture_type = str(hparams.get("architecture_type", ""))
    if not architecture_type:
        if any(key.startswith("backbone.gap_head.") for key in state_dict):
            architecture_type = "unified_csdnet"
        elif any(
            key.startswith("backbone.theta_insertion_head.")
            for key in state_dict
        ):
            architecture_type = "elastic_csdnet"
        else:
            architecture_type = "csdnet"
    hidden_size = int(hparams.get("hidden_size", 768))
    num_layers = int(hparams.get("num_layers", 12))
    num_heads = int(hparams.get("num_heads", 12))
    intermediate = int(hparams.get("intermediate", 3072))
    cond_dim = int(hparams.get("cond_dim", 5))

    emb_key = "backbone.esm.esm.embeddings.word_embeddings.weight"
    if emb_key in state_dict:
        hidden_size = int(state_dict[emb_key].shape[1])

    for cond_key in ("backbone.cond_proj.0.weight", "cond_proj.0.weight"):
        if cond_key in state_dict:
            cond_dim = int(state_dict[cond_key].shape[1])
            break

    layer_prefix = "backbone.esm.esm.encoder.layer."
    layer_ids = set()
    for key in state_dict:
        if key.startswith(layer_prefix):
            rest = key[len(layer_prefix):]
            layer_ids.add(int(rest.split(".", 1)[0]))
    if layer_ids:
        num_layers = max(layer_ids) + 1

    inter_key = "backbone.esm.esm.encoder.layer.0.intermediate.dense.weight"
    if inter_key in state_dict:
        intermediate = int(state_dict[inter_key].shape[0])

    position_key = "backbone.esm.esm.embeddings.position_embeddings.weight"
    max_position_embeddings = int(
        hparams.get("max_position_embeddings", 256)
    )
    if position_key in state_dict:
        max_position_embeddings = int(state_dict[position_key].shape[0])

    if hidden_size % num_heads != 0:
        for candidate in (16, 12, 8, 4, 2, 1):
            if hidden_size % candidate == 0:
                num_heads = candidate
                break

    fixed_unmask_rate = hparams.get("fixed_unmask_rate")
    if "fixed_unmask_rate" not in hparams:
        has_learned_unmask = any(
            key.startswith("backbone.theta_unmask_head.")
            for key in state_dict
        )
        fixed_unmask_rate = None if has_learned_unmask else 1.0
    elif fixed_unmask_rate is not None:
        fixed_unmask_rate = float(fixed_unmask_rate)

    corruption_level_conditioning = bool(
        hparams.get("corruption_level_conditioning", False)
    ) or any(
        key.startswith("backbone.corruption_level_embedding.")
        for key in state_dict
    )
    max_gap_count = int(hparams.get("max_gap_count", 8))
    gap_bias_key = "backbone.gap_head.4.bias"
    if gap_bias_key in state_dict:
        max_gap_count = int(state_dict[gap_bias_key].shape[0] - 1)

    return {
        "architecture_type": architecture_type,
        "cond_dim": cond_dim,
        "hidden_size": hidden_size,
        "num_layers": num_layers,
        "num_heads": num_heads,
        "intermediate": intermediate,
        "max_position_embeddings": max_position_embeddings,
        "position_embedding_type": str(
            hparams.get(
                "position_embedding_type",
                (
                    "rotary"
                    if architecture_type in {"elastic_csdnet", "unified_csdnet"}
                    else "absolute"
                ),
            )
        ),
        "hidden_dropout_prob": float(
            hparams.get("hidden_dropout_prob", 0.1)
        ),
        "attention_probs_dropout_prob": float(
            hparams.get("attention_probs_dropout_prob", 0.1)
        ),
        "layer_norm_eps": float(hparams.get("layer_norm_eps", 1e-12)),
        "initializer_range": float(hparams.get("initializer_range", 0.02)),
        "rate_min": float(hparams.get("rate_min", 0.001)),
        "rate_max": float(hparams.get("rate_max", 20.0)),
        "rate_initial": float(hparams.get("rate_initial", 1.0)),
        "rate_parameterization": str(
            hparams.get("rate_parameterization", "sigmoid")
        ),
        "theta_rate_min": hparams.get("theta_rate_min"),
        "phi_rate_min": hparams.get("phi_rate_min"),
        "rate_output_bias": hparams.get("rate_output_bias"),
        "fixed_unmask_rate": fixed_unmask_rate,
        "kuma_shape_a": float(hparams.get("kuma_shape_a", 2.0)),
        "gradient_checkpointing": bool(
            hparams.get(
                "gradient_checkpointing",
                architecture_type in {"elastic_csdnet", "unified_csdnet"},
            )
        ),
        "corruption_level_conditioning": corruption_level_conditioning,
        "max_gap_count": max_gap_count,
    }


def load_backbone_from_checkpoint(ckpt_path, tk, device, use_ema=True):
    # Deserialize on CPU first. Loading storages directly onto CUDA can fail on
    # busy cluster nodes before PyTorch has established a healthy CUDA context.
    try:
        checkpoint = torch.load(ckpt_path, map_location="cpu", weights_only=False)
    except TypeError:
        checkpoint = torch.load(ckpt_path, map_location="cpu")

    state_dict = checkpoint.get("state_dict", checkpoint)
    cfg = infer_backbone_config(state_dict, checkpoint)
    print(
        "Checkpoint architecture: "
        f"type={cfg['architecture_type']}, "
        f"H={cfg['hidden_size']}, L={cfg['num_layers']}, "
        f"Heads={cfg['num_heads']}, FFN={cfg['intermediate']}, cond_dim={cfg['cond_dim']}"
    )

    if cfg["architecture_type"] == "elastic_csdnet":
        model = ElasticCSDNetBackbone(
            vocab_size=tk.vocab_size,
            cond_dim=cfg["cond_dim"],
            hidden_size=cfg["hidden_size"],
            num_layers=cfg["num_layers"],
            num_heads=cfg["num_heads"],
            intermediate=cfg["intermediate"],
            pad_token_id=tk.pad_id,
            mask_token_id=tk.mask_id,
            max_position_embeddings=cfg["max_position_embeddings"],
            position_embedding_type=cfg["position_embedding_type"],
            hidden_dropout_prob=cfg["hidden_dropout_prob"],
            attention_probs_dropout_prob=(
                cfg["attention_probs_dropout_prob"]
            ),
            layer_norm_eps=cfg["layer_norm_eps"],
            initializer_range=cfg["initializer_range"],
            rate_min=cfg["rate_min"],
            rate_max=cfg["rate_max"],
            rate_initial=cfg["rate_initial"],
            rate_parameterization=cfg["rate_parameterization"],
            theta_rate_min=cfg["theta_rate_min"],
            phi_rate_min=cfg["phi_rate_min"],
            rate_output_bias=cfg["rate_output_bias"],
            fixed_unmask_rate=cfg["fixed_unmask_rate"],
            kuma_shape_a=cfg["kuma_shape_a"],
            gradient_checkpointing=cfg["gradient_checkpointing"],
        )
    elif cfg["architecture_type"] == "unified_csdnet":
        # Keep optional architecture branches isolated. Elastic and ordinary
        # checkpoints must remain loadable when the unified experiment module
        # is not deployed on a worker node.
        from CSDNet.model.unified_backbone import UnifiedCSDNetBackbone

        model = UnifiedCSDNetBackbone(
            vocab_size=tk.vocab_size,
            hidden_size=cfg["hidden_size"],
            num_layers=cfg["num_layers"],
            num_heads=cfg["num_heads"],
            intermediate=cfg["intermediate"],
            pad_token_id=tk.pad_id,
            mask_token_id=tk.mask_id,
            max_position_embeddings=cfg["max_position_embeddings"],
            max_gap_count=cfg["max_gap_count"],
            position_embedding_type=cfg["position_embedding_type"],
            gradient_checkpointing=cfg["gradient_checkpointing"],
            hidden_dropout_prob=cfg["hidden_dropout_prob"],
            attention_probs_dropout_prob=cfg["attention_probs_dropout_prob"],
            layer_norm_eps=cfg["layer_norm_eps"],
            initializer_range=cfg["initializer_range"],
        )
    else:
        model = CSDNetBackbone(
            vocab_size=tk.vocab_size,
            cond_dim=cfg["cond_dim"],
            hidden_size=cfg["hidden_size"],
            num_layers=cfg["num_layers"],
            num_heads=cfg["num_heads"],
            intermediate=cfg["intermediate"],
            pad_token_id=tk.pad_id,
            mask_token_id=tk.mask_id,
            max_position_embeddings=cfg["max_position_embeddings"],
            hidden_dropout_prob=cfg["hidden_dropout_prob"],
            attention_probs_dropout_prob=(
                cfg["attention_probs_dropout_prob"]
            ),
            layer_norm_eps=cfg["layer_norm_eps"],
            initializer_range=cfg["initializer_range"],
            position_embedding_type=cfg["position_embedding_type"],
            gradient_checkpointing=cfg["gradient_checkpointing"],
            corruption_level_conditioning=cfg["corruption_level_conditioning"],
        )

    backbone_state = {
        k[len("backbone."):]: v
        for k, v in state_dict.items()
        if k.startswith("backbone.")
    }
    if not backbone_state:
        backbone_state = {k: v for k, v in state_dict.items() if not k.startswith("ema.")}

    missing, unexpected = model.load_state_dict(backbone_state, strict=False)
    if missing:
        print(f"Warning: missing {len(missing)} backbone keys.")
    if unexpected:
        print(f"Warning: ignored {len(unexpected)} unexpected backbone keys.")

    ema_state = {
        k[len("ema."):]: v
        for k, v in state_dict.items()
        if k.startswith("ema.")
    }
    if use_ema and ema_state:
        param_dict = dict(model.named_parameters())
        loaded = 0
        for safe_name, tensor in ema_state.items():
            name = safe_name.replace("___", ".")
            param = param_dict.get(name)
            if param is not None and tuple(param.shape) == tuple(tensor.shape):
                param.data.copy_(tensor.to(param.device))
                loaded += 1
        print(f"Applied EMA weights: {loaded}/{len(param_dict)} parameters.")
    elif use_ema:
        print("No EMA weights found; using raw backbone weights.")

    return model
