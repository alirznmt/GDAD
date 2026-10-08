"""Dataset provenance helpers used by benchmark training and evaluation."""
from __future__ import annotations

import hashlib
import os
from typing import Any, Dict

from utils.config import resolve_data_paths


def _file_record(path):
    if path is None:
        return None
    if isinstance(path, (list, tuple)):
        return [_file_record(item) for item in path]
    absolute = os.path.abspath(path)
    if not os.path.isfile(absolute):
        raise FileNotFoundError(f"Required dataset file not found: {absolute}")

    digest = hashlib.sha256()
    with open(absolute, "rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)

    return {
        "path": absolute,
        "size_bytes": os.path.getsize(absolute),
        "sha256": digest.hexdigest(),
    }


def collect_data_provenance(cfg) -> Dict[str, Any]:
    """Return stable file hashes for the resolved train/test/label inputs."""
    paths = resolve_data_paths(cfg)
    return {
        "dataset": paths["dataset_name"],
        "files": {
            "train": _file_record(paths["train_path"]),
            "test": _file_record(paths["test_path"]),
            "label": _file_record(paths["test_label_path"]),
        },
    }


def short_provenance(provenance: Dict[str, Any]) -> str:
    parts = []
    for role, record in provenance["files"].items():
        if record is not None:
            if isinstance(record, list):
                hashes = ",".join(item["sha256"][:8] for item in record)
                parts.append(f"{role}=[{hashes}]")
            else:
                parts.append(f"{role}={record['sha256'][:12]}")
    return ", ".join(parts)


def assert_matching_provenance(saved: Dict[str, Any], current: Dict[str, Any]) -> None:
    """Reject evaluation if its data differ from the checkpoint's data."""
    for role, saved_record in saved.get("files", {}).items():
        current_record = current.get("files", {}).get(role)
        if saved_record is None and current_record is None:
            continue
        if saved_record is None or current_record is None:
            raise ValueError(f"Dataset provenance mismatch for {role} file")
        if isinstance(saved_record, list) or isinstance(current_record, list):
            saved_items = saved_record if isinstance(saved_record, list) else [saved_record]
            current_items = current_record if isinstance(current_record, list) else [current_record]
            saved_hashes = [item.get("sha256") for item in saved_items]
            current_hashes = [item.get("sha256") for item in current_items]
            matches = saved_hashes == current_hashes
        else:
            matches = saved_record.get("sha256") == current_record.get("sha256")
        if not matches:
            raise ValueError(
                f"Dataset provenance mismatch for {role}: checkpoint and "
                "evaluation files have different SHA-256 hashes."
            )
