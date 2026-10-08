"""Fast, read-only validation for the HAI/WADI reviewer-extension datasets."""
from __future__ import annotations

import argparse
import json
import os
import sys
from pathlib import Path

import numpy as np

PROJECT_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
if PROJECT_ROOT not in sys.path:
    sys.path.insert(0, PROJECT_ROOT)

from datasets import load_npy_labels
from utils.config import load_config, resolve_data_paths


def _paths(value):
    return list(value) if isinstance(value, (list, tuple)) else [value]


def _array_records(paths, role):
    records = []
    for path in _paths(paths):
        if not os.path.isfile(path):
            raise FileNotFoundError(f"Missing {role} file: {os.path.abspath(path)}")
        if Path(path).suffix.lower() != ".npy":
            raise ValueError(
                f"Reviewer-extension {role} inputs must be preprocessed NPY files: {path}"
            )
        array = np.load(path, mmap_mode="r", allow_pickle=False)
        records.append(
            {
                "path": os.path.abspath(path),
                "shape": [int(value) for value in array.shape],
                "dtype": str(array.dtype),
                "size_bytes": int(os.path.getsize(path)),
            }
        )
    return records


def inspect_dataset(dataset, config_dir, data_root):
    overrides = []
    if data_root:
        overrides.append(
            f"dataset.data_root={os.path.join(os.path.abspath(data_root), dataset)}"
        )
    cfg = load_config(dataset, config_dir=config_dir, overrides=overrides)
    resolved = resolve_data_paths(cfg)
    train = _array_records(resolved["train_path"], "training")
    test = _array_records(resolved["test_path"], "test")
    labels_meta = _array_records(resolved["test_label_path"], "label")

    for record in [*train, *test]:
        if len(record["shape"]) != 2:
            raise ValueError(f"Expected 2-D feature data: {record}")
    feature_counts = {record["shape"][1] for record in [*train, *test]}
    if len(feature_counts) != 1:
        raise ValueError(f"{dataset} feature counts disagree: {sorted(feature_counts)}")
    nodes = feature_counts.pop()
    expected = cfg.dataset.get("expected_num_nodes")
    if expected is not None and nodes != int(expected):
        raise ValueError(f"{dataset}: found {nodes} sensors, expected {expected}")

    labels = load_npy_labels(resolved["test_label_path"])
    train_rows = sum(record["shape"][0] for record in train)
    test_rows = sum(record["shape"][0] for record in test)
    if len(test) != len(labels_meta):
        raise ValueError(
            f"{dataset}: {len(test)} test files but {len(labels_meta)} label files"
        )
    if len(test) > 1:
        per_file_mismatches = [
            (data["path"], label["path"], data["shape"][0], label["shape"][0])
            for data, label in zip(test, labels_meta)
            if data["shape"][0] != label["shape"][0]
        ]
        if per_file_mismatches:
            raise ValueError(
                f"{dataset}: per-file test/label lengths disagree: "
                f"{per_file_mismatches}"
            )
    mismatch = abs(test_rows - len(labels))
    allowed = int(cfg.dataset.get("max_length_mismatch", 0))
    if mismatch > allowed:
        raise ValueError(
            f"{dataset}: {test_rows} test rows vs {len(labels)} labels; "
            f"allowed mismatch is {allowed}"
        )
    anomaly_fraction = float(labels[: min(test_rows, len(labels))].mean())
    if not 0.0 < anomaly_fraction < 0.5:
        raise ValueError(
            f"{dataset}: anomaly fraction is {anomaly_fraction:.5f}. Expected a "
            "non-zero minority anomaly class; verify label polarity and files."
        )

    for optional_key in ("sensor_names_file", "sensor_groups_file", "topology_file"):
        value = cfg.dataset.get(optional_key)
        if value:
            path = value if os.path.isabs(value) else os.path.join(cfg.dataset.data_root, value)
            if not os.path.isfile(path):
                raise FileNotFoundError(f"Configured {optional_key} does not exist: {path}")

    batch_size = int(cfg.train.get("batch_size", 32))
    adjacency_mb = batch_size * nodes * nodes * 4 / (1024.0 ** 2)
    return {
        "dataset": dataset,
        "status": "ok",
        "train_files": train,
        "test_files": test,
        "label_files": labels_meta,
        "train_rows": int(train_rows),
        "test_rows": int(test_rows),
        "label_rows": int(len(labels)),
        "trimmed_rows": int(mismatch),
        "sensors": int(nodes),
        "anomaly_fraction": anomaly_fraction,
        "window_size": int(cfg.data.get("window_size", 100)),
        "train_stride": int(cfg.data.get("stride", 50)),
        "batch_size": batch_size,
        "one_dense_adjacency_batch_mb": float(adjacency_mb),
        "metadata": {
            key: cfg.dataset.get(key)
            for key in ("sensor_names_file", "sensor_groups_file", "topology_file")
        },
    }


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--datasets", nargs="+", default=["HAI", "WADI"])
    parser.add_argument("--config_dir", default="configs")
    parser.add_argument("--data_root", default="data")
    parser.add_argument("--output", default=None)
    args = parser.parse_args()

    reports = [
        inspect_dataset(dataset, args.config_dir, args.data_root)
        for dataset in args.datasets
    ]
    for report in reports:
        print(
            f"OK {report['dataset']}: train={report['train_rows']:,}, "
            f"test={report['test_rows']:,}, sensors={report['sensors']}, "
            f"anomaly_fraction={report['anomaly_fraction']:.5f}, "
            f"batch={report['batch_size']}, "
            f"adjacency_tensor={report['one_dense_adjacency_batch_mb']:.1f} MiB"
        )
        if not any(report["metadata"].values()):
            print(
                "  note: no optional sensor names/groups/topology metadata configured; "
                "structural analysis will use positional sensor ids"
            )
    if args.output:
        os.makedirs(os.path.dirname(os.path.abspath(args.output)), exist_ok=True)
        with open(args.output, "w", encoding="utf-8") as handle:
            json.dump(reports, handle, indent=2)


if __name__ == "__main__":
    main()
