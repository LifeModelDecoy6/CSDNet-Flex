from __future__ import annotations

import argparse
from pathlib import Path
from typing import Sequence

import yaml


def parse_args_with_yaml_config(
    parser: argparse.ArgumentParser,
    argv: Sequence[str] | None = None,
) -> argparse.Namespace:
    """Apply flat YAML values as argparse defaults, then parse CLI overrides."""
    config_parser = argparse.ArgumentParser(add_help=False)
    config_parser.add_argument("--config")
    config_args, _ = config_parser.parse_known_args(argv)

    if config_args.config:
        config_path = Path(config_args.config)
        with config_path.open("r", encoding="utf-8") as handle:
            values = yaml.safe_load(handle) or {}
        if not isinstance(values, dict):
            raise ValueError(
                f"Training config must be a YAML mapping: {config_path}"
            )

        allowed = {
            action.dest
            for action in parser._actions
            if action.dest is not argparse.SUPPRESS
        }
        unknown = sorted(set(values) - allowed)
        if unknown:
            raise ValueError(
                f"Unknown training config keys in {config_path}: {unknown}"
            )
        parser.set_defaults(**values)

    return parser.parse_args(argv)
