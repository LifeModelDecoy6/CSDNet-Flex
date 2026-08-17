MODEL_CONFIGS = {
    "6m": {
        "hidden_size": 256,
        "num_layers": 8,
        "num_heads": 8,
        "intermediate": 1024,
        "lr": 5e-4,
        "warmup_steps": 2000,
        "batch_size": 256,
    },
    "30m": {
        "hidden_size": 512,
        "num_layers": 8,
        "num_heads": 8,
        "intermediate": 2048,
        "lr": 5e-4,
        "warmup_steps": 4000,
        "batch_size": 256,
    },
    "90m": {
        "hidden_size": 768,
        "num_layers": 12,
        "num_heads": 12,
        "intermediate": 3072,
        "lr": 2e-4,
        "warmup_steps": 8000,
        "batch_size": 256,
    },
}


def get_model_config(name):
    key = str(name).lower()
    if key not in MODEL_CONFIGS:
        raise KeyError(f"Unknown CSDNet size '{name}'. Choices: {sorted(MODEL_CONFIGS)}")
    return dict(MODEL_CONFIGS[key])
