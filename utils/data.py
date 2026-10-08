"""Data-loader orchestration around the shared ``datasets.py`` interface.

The wrapper ensures that:

* training/validation use a (possibly overlapping) training stride, while
* the test set uses ``stride == window_size`` so that every timestep is scored
  exactly once (clean pointwise coverage).

The ``StandardScaler`` is fit on the train split only inside ``build_datasets``
and is deterministic, so calling it twice yields identical normalisation.
"""
from __future__ import annotations

from typing import Tuple

from torch.utils.data import DataLoader, Dataset

import datasets as ds_module
from utils.seed import worker_init_fn


def build_dataloaders(cfg) -> Tuple[DataLoader, DataLoader, DataLoader, Dataset, int]:
    """Return train/val/test loaders, the test dataset, and the sensor count."""
    from utils.config import resolve_data_paths

    paths = resolve_data_paths(cfg)
    window = cfg.data.get("window_size", 96)
    train_stride = cfg.data.get("stride", window)
    train_ratio = cfg.data.get("train_ratio", 0.85)
    batch_size = cfg.train.get("batch_size", 64)
    num_workers = cfg.train.get("num_workers", 0)

    # Load and scale once. Large HAI/WADI arrays must not be duplicated just
    # to use a different scoring stride.
    train_ds, val_ds, test_ds_train_stride, _ = ds_module.build_datasets(
        dataset_name=paths["dataset_name"],
        train_path=paths["train_path"],
        test_path=paths["test_path"],
        test_label_path=paths["test_label_path"],
        window_size=window,
        stride=train_stride,
        train_ratio=train_ratio,
        max_length_mismatch=cfg.dataset.get("max_length_mismatch", 0),
    )

    # Test with non-overlapping stride for exact per-timestep coverage. This
    # view shares the already-loaded tensors and respects file boundaries.
    test_ds = test_ds_train_stride.with_stride(window, include_last=True)

    dimensions = {
        "train": int(train_ds.dimensions),
        "validation": int(val_ds.dimensions),
        "test": int(test_ds.dimensions),
    }
    if len(set(dimensions.values())) != 1:
        raise ValueError(
            f"Feature-count mismatch for {paths['dataset_name']}: {dimensions}. "
            "Train, validation, and test files must use the same sensors."
        )

    num_nodes = dimensions["train"]
    expected_num_nodes = cfg.dataset.get("expected_num_nodes")
    if expected_num_nodes is not None and num_nodes != int(expected_num_nodes):
        raise ValueError(
            f"Refusing to run {paths['dataset_name']} with {num_nodes} sensors; "
            f"the benchmark config requires {int(expected_num_nodes)}. "
            "The data may be synthetic, stale, or incorrectly preprocessed."
        )

    train_loader = DataLoader(
        train_ds,
        batch_size=batch_size,
        shuffle=True,
        drop_last=True,
        num_workers=num_workers,
        worker_init_fn=worker_init_fn,
        pin_memory=True,
    )
    val_loader = DataLoader(
        val_ds,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
        worker_init_fn=worker_init_fn,
        pin_memory=True,
    )
    test_loader = DataLoader(
        test_ds,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
        worker_init_fn=worker_init_fn,
        pin_memory=True,
    )

    if len(train_loader) == 0:
        raise ValueError(
            f"No complete training batches for {paths['dataset_name']}. "
            "Check the dataset size, window/stride, and train.batch_size."
        )

    return train_loader, val_loader, test_loader, test_ds, num_nodes


def build_val_loader_for_scoring(cfg, val_dataset=None) -> DataLoader:
    """Validation loader with non-overlapping stride (for residual statistics)."""
    window = cfg.data.get("window_size", 96)
    batch_size = cfg.train.get("batch_size", 64)
    num_workers = cfg.train.get("num_workers", 0)

    if val_dataset is None:
        from utils.config import resolve_data_paths

        paths = resolve_data_paths(cfg)
        train_ratio = cfg.data.get("train_ratio", 0.85)
        _, val_dataset, _, _ = ds_module.build_datasets(
            dataset_name=paths["dataset_name"],
            train_path=paths["train_path"],
            test_path=paths["test_path"],
            test_label_path=paths["test_label_path"],
            window_size=window,
            stride=window,
            train_ratio=train_ratio,
            max_length_mismatch=cfg.dataset.get("max_length_mismatch", 0),
        )
    val_ds = val_dataset.with_stride(window, include_last=True)
    return DataLoader(
        val_ds,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
        worker_init_fn=worker_init_fn,
    )
