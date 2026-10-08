import os
from collections.abc import Sequence

import numpy as np
import pandas as pd
import torch
from sklearn.preprocessing import StandardScaler
from torch.utils.data import Dataset


def _as_paths(path_or_paths):
    if isinstance(path_or_paths, (str, os.PathLike)):
        return [os.fspath(path_or_paths)]
    if isinstance(path_or_paths, Sequence):
        paths = [os.fspath(path) for path in path_or_paths]
        if paths:
            return paths
    raise ValueError(f"Expected one or more dataset paths, got {path_or_paths!r}")


def load_npy_data(data_path, *, return_segment_lengths=False):
    """Load one or more chronological NPY files.

    HAI is distributed as several continuous files.  They are concatenated for
    scaling, while their lengths are retained so window creation never crosses
    a file boundary.
    """
    arrays = []
    lengths = []
    for path in _as_paths(data_path):
        data = np.load(path, allow_pickle=False)
        if data.ndim != 2:
            raise ValueError(f"Expected a 2-D array at {path}, got shape {data.shape}")
        data = np.nan_to_num(data).astype(np.float32, copy=False)
        arrays.append(data)
        lengths.append(int(data.shape[0]))
    feature_counts = {array.shape[1] for array in arrays}
    if len(feature_counts) != 1:
        raise ValueError(
            f"Feature-count mismatch across {len(arrays)} files: "
            f"{[array.shape for array in arrays]}"
        )
    combined = arrays[0] if len(arrays) == 1 else np.concatenate(arrays, axis=0)
    return (combined, lengths) if return_segment_lengths else combined


def load_npy_labels(labels_path, *, return_segment_lengths=False):
    arrays = []
    lengths = []
    for path in _as_paths(labels_path):
        labels = np.load(path, allow_pickle=False)
        labels = np.asarray(labels).squeeze()
        if labels.ndim != 1:
            raise ValueError(
                f"Expected one binary label per timestamp at {path}, got {labels.shape}"
            )
        arrays.append(labels)
        lengths.append(int(labels.shape[0]))
    labels = arrays[0] if len(arrays) == 1 else np.concatenate(arrays, axis=0)
    labels = normalize_binary_labels(labels)
    return (labels, lengths) if return_segment_lengths else labels


def normalize_binary_labels(labels):
    """Return canonical float32 labels (0 normal, 1 anomalous).

    The common WADI convention uses +1 for normal and -1 for attack.  Native
    0/1 labels (including HAI) are preserved.
    """
    labels = np.asarray(labels).reshape(-1)
    if not np.isfinite(labels).all():
        raise ValueError("Labels contain NaN or infinite values")
    unique = set(np.unique(labels).tolist())
    if unique.issubset({0, 1, 0.0, 1.0}):
        return labels.astype(np.float32)
    if unique.issubset({-1, 1, -1.0, 1.0}):
        return (labels < 0).astype(np.float32)
    raise ValueError(
        "Labels must use {0,1} (1=anomaly) or WADI's {+1,-1} "
        f"(+1=normal, -1=anomaly); found {sorted(unique)}"
    )


def load_csv_data(data_path, labels_path=None, is_test=False):
    df = pd.read_csv(os.path.join(data_path))

    if 'PSM' in data_path:
        data = df.values[:, 1:]
    else:
        data = df.values[:, :-1]

    data = np.nan_to_num(data).astype(np.float32)

    labels = None

    if is_test:
        if labels_path is not None:
            df_labels = pd.read_csv(os.path.join(labels_path))
            labels = df_labels.values[:, 1:]
        else:
            labels = df.values[:, -1:]

        labels = normalize_binary_labels(labels)

    return data, labels


class WindowDataset(Dataset):
    def __init__(
        self,
        data,
        labels=None,
        transform=None,
        window_size=96,
        stride=96,
        include_last=True,
        segment_lengths=None,
    ):
        self.data = (
            data.to(dtype=torch.float32)
            if isinstance(data, torch.Tensor)
            else torch.from_numpy(np.asarray(data)).to(dtype=torch.float32)
        )
        self.labels = (
            None
            if labels is None
            else (
                labels.to(dtype=torch.float32)
                if isinstance(labels, torch.Tensor)
                else torch.from_numpy(np.asarray(labels)).to(dtype=torch.float32)
            )
        )

        self.transform = transform
        self.window_size = window_size
        self.stride = stride
        self.dimensions = self.data.shape[1]
        self.include_last = bool(include_last)
        self.segment_lengths = (
            [int(len(self.data))]
            if segment_lengths is None
            else [int(length) for length in segment_lengths if int(length) > 0]
        )
        if sum(self.segment_lengths) != len(self.data):
            raise ValueError(
                f"Segment lengths sum to {sum(self.segment_lengths)}, "
                f"but data contain {len(self.data)} timestamps"
            )
        self._rebuild_indices()

    def _rebuild_indices(self):
        self.start_indices = []
        self.segment_end_indices = []
        offset = 0
        for length in self.segment_lengths:
            if self.include_last:
                max_start = self.stride * (
                    length // self.stride + int(length % self.stride > 0)
                )
                local_starts = range(0, max_start, self.stride)
            else:
                local_starts = range(
                    0, max(0, length - self.window_size + 1), self.stride
                )
            segment_end = offset + length
            for local_start in local_starts:
                self.start_indices.append(offset + local_start)
                self.segment_end_indices.append(segment_end)
            offset = segment_end

    def with_stride(self, stride, *, include_last=True):
        """Return a lightweight view sharing tensors but using new windows."""
        return WindowDataset(
            self.data,
            labels=self.labels,
            transform=self.transform,
            window_size=self.window_size,
            stride=stride,
            include_last=include_last,
            segment_lengths=self.segment_lengths,
        )

    def __len__(self):
        return len(self.start_indices)

    def __getitem__(self, idx):
        start = self.start_indices[idx]
        end = start + self.window_size
        segment_end = self.segment_end_indices[idx]

        if end <= segment_end:
            sample = self.data[start:end]

            if self.labels is not None:
                labels = self.labels[start:end]
        else:
            sample = self.data[start:segment_end]

            pad_len = end - segment_end
            fill_tensor = torch.zeros((pad_len, self.dimensions))
            sample = torch.cat([sample, fill_tensor], dim=0)

            if self.labels is not None:
                labels = self.labels[start:segment_end]
                fill_labels = torch.zeros(pad_len)
                labels = torch.cat([labels, fill_labels], dim=0)

        if self.transform:
            sample = self.transform(sample)

        if self.labels is not None:
            return sample, labels

        return sample


def build_datasets(
    dataset_name,
    train_path,
    test_path,
    test_label_path,
    window_size=96,
    stride=96,
    train_ratio=0.85,
    transform=None,
    max_length_mismatch=0,
):
    """
    Leakage-safe dataset builder.

    Important:
    - scaler is fit only on train split.
    - validation starts after train split.
    - test is transformed by train scaler, not fit on test.
    """

    train_paths = _as_paths(train_path)
    test_paths = _as_paths(test_path)
    label_paths = _as_paths(test_label_path)
    all_npy = all(
        os.path.splitext(path)[1].lower() == ".npy"
        for path in [*train_paths, *test_paths, *label_paths]
    )

    if not all_npy and dataset_name in ['SWaT', 'PSM']:
        if len(train_paths) != 1 or len(test_paths) != 1 or len(label_paths) != 1:
            raise ValueError("Multi-file CSV loading is not supported; preprocess to NPY")
        full_train_data, _ = load_csv_data(train_path, is_test=False)
        test_data, test_labels = load_csv_data(
            test_path,
            labels_path=test_label_path,
            is_test=True
        )
        train_segment_lengths = [len(full_train_data)]
        test_segment_lengths = [len(test_data)]
    else:
        full_train_data, train_segment_lengths = load_npy_data(
            train_paths, return_segment_lengths=True
        )
        test_data, test_segment_lengths = load_npy_data(
            test_paths, return_segment_lengths=True
        )
        test_labels, label_segment_lengths = load_npy_labels(
            label_paths, return_segment_lengths=True
        )
        if len(test_segment_lengths) != len(label_segment_lengths):
            raise ValueError(
                f"{dataset_name} has {len(test_segment_lengths)} test files but "
                f"{len(label_segment_lengths)} label files"
            )
        if len(test_segment_lengths) > 1 and test_segment_lengths != label_segment_lengths:
            raise ValueError(
                f"{dataset_name} per-file test/label lengths disagree: "
                f"data={test_segment_lengths}, labels={label_segment_lengths}. "
                "Align every segment before concatenation."
            )

    mismatch = abs(len(test_data) - len(test_labels))
    if mismatch:
        if mismatch > int(max_length_mismatch):
            raise ValueError(
                f"{dataset_name} test/label length mismatch: {len(test_data)} data "
                f"vs {len(test_labels)} labels (allowed {max_length_mismatch})"
            )
        aligned_length = min(len(test_data), len(test_labels))
        test_data = test_data[:aligned_length]
        test_labels = test_labels[:aligned_length]
        remaining = aligned_length
        trimmed_segments = []
        for length in test_segment_lengths:
            take = min(length, remaining)
            if take:
                trimmed_segments.append(take)
            remaining -= take
            if remaining <= 0:
                break
        test_segment_lengths = trimmed_segments

    split_idx = int(full_train_data.shape[0] * train_ratio)

    train_data_raw = full_train_data[:split_idx]
    val_data_raw = full_train_data[split_idx:]

    scaler = StandardScaler()
    train_data = scaler.fit_transform(train_data_raw).astype(np.float32)

    val_data = scaler.transform(val_data_raw).astype(np.float32)
    test_data = scaler.transform(test_data).astype(np.float32)

    def sliced_segments(lengths, start, stop):
        result = []
        offset = 0
        for length in lengths:
            left = max(start, offset)
            right = min(stop, offset + length)
            if right > left:
                result.append(right - left)
            offset += length
        return result

    train_segments = sliced_segments(train_segment_lengths, 0, split_idx)
    val_segments = sliced_segments(
        train_segment_lengths, split_idx, len(full_train_data)
    )

    train_dataset = WindowDataset(
        data=train_data,
        labels=None,
        transform=transform,
        window_size=window_size,
        stride=stride,
        segment_lengths=train_segments,
    )

    val_dataset = WindowDataset(
        data=val_data,
        labels=None,
        transform=transform,
        window_size=window_size,
        stride=stride,
        segment_lengths=val_segments,
    )

    test_dataset = WindowDataset(
        data=test_data,
        labels=test_labels,
        transform=transform,
        window_size=window_size,
        stride=stride,
        segment_lengths=test_segment_lengths,
    )

    return train_dataset, val_dataset, test_dataset, scaler
