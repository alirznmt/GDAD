"""Small synthetic regression test for reviewer-extension plumbing."""
from __future__ import annotations

import json
import os
import sys
import tempfile

import numpy as np

PROJECT_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
if PROJECT_ROOT not in sys.path:
    sys.path.insert(0, PROJECT_ROOT)

from datasets import build_datasets, normalize_binary_labels
from structural_analysis import (
    analyse_dynamic,
    analyse_seeds,
    load_dir,
    summarise_dynamic_seeds,
)
from utils.config import Config, resolve_data_paths
from utils.provenance import assert_matching_provenance, collect_data_provenance
from utils.scoring import ema_smooth, windows_to_timeline


def main():
    with tempfile.TemporaryDirectory(prefix="gdad-reviewer-selftest-") as root:
        train_paths = []
        test_paths = []
        label_paths = []
        for index in range(2):
            train = np.full((6, 3), index + 1, dtype=np.float32)
            test = np.full((5, 3), 10 * (index + 1), dtype=np.float32)
            labels = np.array([0, 0, 1, 0, 1], dtype=np.float32)
            train_path = os.path.join(root, f"train{index}.npy")
            test_path = os.path.join(root, f"test{index}.npy")
            label_path = os.path.join(root, f"label{index}.npy")
            np.save(train_path, train)
            np.save(test_path, test)
            np.save(label_path, labels)
            train_paths.append(train_path)
            test_paths.append(test_path)
            label_paths.append(label_path)

        train_ds, val_ds, test_ds, _ = build_datasets(
            "HAI",
            train_paths,
            test_paths,
            label_paths,
            window_size=4,
            stride=2,
            train_ratio=0.5,
        )
        assert train_ds.segment_lengths == [6]
        assert val_ds.segment_lengths == [6]
        assert test_ds.segment_lengths == [5, 5]
        boundary_window = test_ds[test_ds.start_indices.index(4)][0]
        assert np.allclose(boundary_window[1:].numpy(), 0.0)
        assert normalize_binary_labels([1, -1, 1, -1]).tolist() == [0, 1, 0, 1]
        timeline = windows_to_timeline(
            np.array(
                [np.full(4, value, dtype=np.float32) for value in range(1, 7)]
            ),
            test_ds.start_indices,
            4,
            10,
            segment_end_indices=test_ds.segment_end_indices,
        )
        assert timeline[5] == 4.0, "first segment tail leaked into second segment"
        smoothed = ema_smooth(
            np.array([0.0] * 5 + [10.0] * 5),
            alpha=0.9,
            segment_lengths=[5, 5],
        )
        assert smoothed[5] == 10.0, "EMA did not reset at the file boundary"

        cfg = Config(
            {
                "dataset": {
                    "name": "HAI",
                    "data_root": root,
                    "train_file": [os.path.basename(path) for path in train_paths],
                    "test_file": [os.path.basename(path) for path in test_paths],
                    "label_file": [os.path.basename(path) for path in label_paths],
                }
            }
        )
        resolved = resolve_data_paths(cfg)
        assert isinstance(resolved["train_path"], list)
        provenance = collect_data_provenance(cfg)
        assert len(provenance["files"]["train"]) == 2
        assert_matching_provenance(provenance, provenance)

        artifact = os.path.join(root, "artifacts", "HAI")
        os.makedirs(artifact)
        names = ["a", "b", "c"]
        with open(os.path.join(artifact, "sensor_names.json"), "w", encoding="utf-8") as handle:
            json.dump(names, handle)
        topology = np.array([[0, 1, 0], [0, 0, 1], [1, 0, 0]], dtype=np.uint8)
        np.save(os.path.join(artifact, "known_topology.npy"), topology)
        dynamic_reports = {}
        for seed in (42, 43):
            stat = np.array(
                [[0.6, 0.4, 0.0], [0.0, 0.7, 0.3], [0.2, 0.0, 0.8]],
                dtype=np.float32,
            )
            np.save(os.path.join(artifact, f"A_stat_seed{seed}.npy"), stat)
            manifest = {
                "dataset": "HAI",
                "seed": seed,
                "graph_mode": "hybrid",
                "static_gate_sigmoid_alpha": 0.5,
                "top_k": 2,
                "num_nodes": 3,
            }
            with open(
                os.path.join(artifact, f"export_manifest_seed{seed}.json"),
                "w",
                encoding="utf-8",
            ) as handle:
                json.dump(manifest, handle)
            with open(
                os.path.join(artifact, "export_manifest.json"),
                "w",
                encoding="utf-8",
            ) as handle:
                json.dump(manifest, handle)
            for label, shift in (("normal", 0.0), ("anomalous", 0.08)):
                for window in (0, 1):
                    for draw in (0, 1):
                        matrix = stat + shift
                        matrix = matrix / matrix.sum(axis=1, keepdims=True)
                        np.save(
                            os.path.join(
                                artifact,
                                f"A_dyn_seed{seed}_{label}_w{window}_t10_m{draw}.npy",
                            ),
                            matrix,
                        )

        loaded = load_dir(artifact)
        assert sorted(loaded["dyn_by_seed"]) == [42, 43]
        assert analyse_seeds(loaded["stat"])["seeds"] == [42, 43]
        for seed in (42, 43):
            view = dict(loaded)
            view["dyn"] = loaded["dyn_by_seed"][seed]
            view["manifest"] = loaded["manifests"][seed]
            dynamic_reports[seed] = analyse_dynamic(
                view, loaded["stat"][seed], np.random.default_rng(0), 10
            )
        summary = summarise_dynamic_seeds(dynamic_reports)
        assert summary and summary["seeds"] == [42, 43]

    print("Reviewer-extension synthetic self-test: OK")


if __name__ == "__main__":
    main()
