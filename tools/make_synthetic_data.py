"""Generate tiny synthetic datasets matching the datasets.py file conventions.

This is ONLY for smoke-testing the training / evaluation / HPO pipeline without
the real benchmark data. It writes small .npy / .csv files with injected
anomalies so the whole stack can be exercised end to end.

Usage:
    python tools/make_synthetic_data.py --root data
"""
from __future__ import annotations

import argparse
import os

import numpy as np
import pandas as pd


def _make_series(n: int, dim: int, rng: np.random.Generator) -> np.ndarray:
    t = np.linspace(0, 8 * np.pi, n)[:, None]
    freqs = rng.uniform(0.5, 2.0, size=(1, dim))
    phase = rng.uniform(0, 2 * np.pi, size=(1, dim))
    base = np.sin(t * freqs + phase) + 0.1 * rng.standard_normal((n, dim))
    return base.astype(np.float32)


def _inject(series: np.ndarray, rng: np.random.Generator, ratio: float):
    n = series.shape[0]
    labels = np.zeros(n, dtype=np.float32)
    n_anom = max(1, int(n * ratio))
    starts = rng.integers(0, n - 5, size=max(1, n_anom // 5))
    for s in starts:
        length = rng.integers(2, 6)
        series[s:s + length] += rng.uniform(3, 6) * rng.choice([-1, 1])
        labels[s:s + length] = 1.0
    return series, labels


def make_npy(root: str, name: str, dim: int, rng):
    d = os.path.join(root, name)
    os.makedirs(d, exist_ok=True)
    train = _make_series(2000, dim, rng)
    test = _make_series(1000, dim, rng)
    test, labels = _inject(test, rng, ratio=0.1)
    np.save(os.path.join(d, f"{name}_train.npy"), train)
    np.save(os.path.join(d, f"{name}_test.npy"), test)
    np.save(os.path.join(d, f"{name}_test_label.npy"), labels)


def make_swat(root: str, rng):
    d = os.path.join(root, "SWaT")
    os.makedirs(d, exist_ok=True)
    dim = 51
    train = _make_series(2000, dim, rng)
    test = _make_series(1000, dim, rng)
    test, labels = _inject(test, rng, ratio=0.12)
    # train csv: features + dummy last column (datasets.py drops last col)
    tr = np.concatenate([train, np.zeros((len(train), 1), np.float32)], axis=1)
    pd.DataFrame(tr).to_csv(os.path.join(d, "swat_train.csv"), index=False)
    te = np.concatenate([test, labels[:, None]], axis=1)
    pd.DataFrame(te).to_csv(os.path.join(d, "swat_test.csv"), index=False)


def make_psm(root: str, rng):
    d = os.path.join(root, "PSM")
    os.makedirs(d, exist_ok=True)
    dim = 25
    train = _make_series(2000, dim, rng)
    test = _make_series(1000, dim, rng)
    test, labels = _inject(test, rng, ratio=0.28)
    # PSM: first column is a timestamp that datasets.py drops.
    tr = pd.DataFrame(train)
    tr.insert(0, "timestamp", np.arange(len(train)))
    tr.to_csv(os.path.join(d, "train.csv"), index=False)
    te = pd.DataFrame(test)
    te.insert(0, "timestamp", np.arange(len(test)))
    te.to_csv(os.path.join(d, "test.csv"), index=False)
    lab = pd.DataFrame({"timestamp": np.arange(len(test)), "label": labels})
    lab.to_csv(os.path.join(d, "test_label.csv"), index=False)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--root", default="data")
    parser.add_argument("--seed", type=int, default=0)
    args = parser.parse_args()
    rng = np.random.default_rng(args.seed)

    make_npy(args.root, "SMAP", 25, rng)
    make_npy(args.root, "MSL", 55, rng)
    make_npy(args.root, "SMD", 38, rng)
    make_swat(args.root, rng)
    make_psm(args.root, rng)
    print(f"Synthetic data written under: {args.root}")


if __name__ == "__main__":
    main()
