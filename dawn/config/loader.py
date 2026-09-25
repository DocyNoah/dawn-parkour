from __future__ import annotations

from collections.abc import Mapping
from pathlib import Path
from typing import Any

import yaml

from dawn.config.experiment_cfg import ExperimentCfg
from dawn.rl.configs.factory import train_cfg_from_algorithm_name


def _load_yaml_dict(path: Path) -> dict[str, Any]:
    if not path.exists():
        raise FileNotFoundError(f"Experiment config file does not exist: {path}")
    raw = yaml.safe_load(path.read_text(encoding="utf-8"))
    if raw is None:
        return {}
    if not isinstance(raw, dict):
        raise ValueError(f"Experiment config YAML must be a mapping at top-level: {path}")
    return raw


def _merge_cfg(obj: Any, data: Mapping[str, Any], namespace: str = "") -> None:
    for key, value in data.items():
        key_namespace = f"{namespace}/{key}" if namespace else key
        if isinstance(obj, dict):
            obj[key] = value
            continue
        if not hasattr(obj, key):
            raise KeyError(f"[Config]: Key not found under namespace: {key_namespace}.")

        current = getattr(obj, key)
        if isinstance(value, Mapping) and not isinstance(current, dict) and current is not None:
            _merge_cfg(current, value, namespace=key_namespace)
            continue

        if isinstance(value, list) and (isinstance(current, tuple) or key.endswith("_shape")):
            value = tuple(value)
        elif current is not None and value is not None and not isinstance(value, type(current)):
            raise ValueError(
                f"[Config]: Incorrect type under namespace: {key_namespace}."
                f" Expected: {type(current)}, Received: {type(value)}."
            )

        setattr(obj, key, value)


def load_logged_experiment_cfg(log_dir: str | Path) -> ExperimentCfg:
    path = Path(log_dir) / "params" / "experiment.yaml"
    raw_cfg = _load_yaml_dict(path)

    cfg = ExperimentCfg(train=train_cfg_from_algorithm_name(raw_cfg["train"]["algorithm_name"]))
    try:
        _merge_cfg(cfg, raw_cfg)
    except (KeyError, ValueError, TypeError) as exc:
        raise ValueError(f"Malformed experiment config YAML at {path}: {exc}") from exc
    return cfg


__all__ = ["load_logged_experiment_cfg"]
