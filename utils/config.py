"""Configuration management: YAML loading, deep-merge, and dotted overrides.

We keep configuration as plain (nested) dicts wrapped in a small ``Config``
helper so that the rest of the code can use attribute-style access
(``cfg.model.hidden_dim``) while still being trivially serialisable to JSON for
checkpointing and result logging.
"""
from __future__ import annotations

import copy
import os
from typing import Any, Dict, List

import yaml


class Config(dict):
    """A dict that also supports attribute access, recursively."""

    def __getattr__(self, key: str) -> Any:
        try:
            value = self[key]
        except KeyError as exc:
            raise AttributeError(key) from exc
        if isinstance(value, dict) and not isinstance(value, Config):
            value = Config(value)
            self[key] = value
        return value

    def __setattr__(self, key: str, value: Any) -> None:
        self[key] = value

    def to_dict(self) -> Dict[str, Any]:
        out: Dict[str, Any] = {}
        for key, value in self.items():
            out[key] = value.to_dict() if isinstance(value, Config) else value
        return out


def _deep_merge(base: Dict[str, Any], override: Dict[str, Any]) -> Dict[str, Any]:
    """Recursively merge ``override`` into a copy of ``base``."""
    result = copy.deepcopy(base)
    for key, value in override.items():
        if (
            key in result
            and isinstance(result[key], dict)
            and isinstance(value, dict)
        ):
            result[key] = _deep_merge(result[key], value)
        else:
            result[key] = copy.deepcopy(value)
    return result


def _load_yaml(path: str) -> Dict[str, Any]:
    with open(path, "r", encoding="utf-8") as handle:
        data = yaml.safe_load(handle)
    return data or {}


def _coerce(value: str) -> Any:
    """Best-effort cast of a CLI string into bool/int/float/None."""
    lowered = value.lower()
    if lowered in {"true", "false"}:
        return lowered == "true"
    if lowered in {"none", "null"}:
        return None
    if value.startswith(("[", "{")):
        parsed = yaml.safe_load(value)
        if isinstance(parsed, (list, dict)):
            return parsed
    for cast in (int, float):
        try:
            return cast(value)
        except ValueError:
            continue
    return value


def apply_overrides(cfg: Dict[str, Any], overrides: List[str]) -> Dict[str, Any]:
    """Apply ``key.sub=value`` style overrides (used by the CLI)."""
    cfg = copy.deepcopy(cfg)
    for item in overrides:
        if "=" not in item:
            raise ValueError(f"Override '{item}' must be of form key.sub=value")
        dotted, raw = item.split("=", 1)
        keys = dotted.split(".")
        node = cfg
        for key in keys[:-1]:
            node = node.setdefault(key, {})
        node[keys[-1]] = _coerce(raw)
    return cfg


def load_config(
    dataset: str,
    config_dir: str = "configs",
    overrides: List[str] | None = None,
) -> Config:
    """Load ``base.yaml`` merged with the per-dataset YAML and CLI overrides."""
    base_path = os.path.join(config_dir, "base.yaml")
    base = _load_yaml(base_path) if os.path.exists(base_path) else {}

    ds_path = os.path.join(config_dir, f"{dataset.lower()}.yaml")
    if not os.path.exists(ds_path):
        raise FileNotFoundError(f"No config found for dataset '{dataset}' at {ds_path}")
    ds_cfg = _load_yaml(ds_path)

    merged = _deep_merge(base, ds_cfg)
    if overrides:
        merged = apply_overrides(merged, overrides)

    merged.setdefault("dataset", {})["name"] = dataset
    return Config(merged)


def resolve_data_paths(cfg: Config) -> Dict[str, Any]:
    """Build absolute train/test/label paths from the dataset config block."""
    ds = cfg.dataset
    root = ds.get("data_root", "data")
    name = ds["name"]

    def _join(filename):
        if filename is None:
            return None
        if isinstance(filename, (list, tuple)):
            return [_join(item) for item in filename]
        return os.path.join(root, filename)

    return {
        "dataset_name": name,
        "train_path": _join(ds.get("train_file")),
        "test_path": _join(ds.get("test_file")),
        "test_label_path": _join(ds.get("label_file")),
    }
