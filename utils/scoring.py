"""Anomaly-score post-processing.

The model produces a per-(window, time, channel) residual tensor. This module
turns that into a single per-timestep anomaly score aligned with the original
test sequence. The steps are deliberately chosen to help *pointwise* detection:

1. **Per-channel z-normalisation** using statistics from normal (validation)
   residuals, so heterogeneous sensors contribute comparably.
2. **Channel aggregation** (mean or max over channels).
3. **Overlap-add reconstruction** back to the original timeline, averaging the
   contributions of overlapping windows.
4. **Optional EMA smoothing** to suppress single-sample spikes.
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import List, Optional

import numpy as np


@dataclass
class ChannelNormalizer:
    """Per-channel residual standardiser fitted on normal data."""

    mean: np.ndarray
    std: np.ndarray

    @classmethod
    def fit(
        cls, residuals: np.ndarray, eps: float = 1e-6, method: str = "zscore"
    ) -> "ChannelNormalizer":
        """Fit on residuals of shape ``[num_windows, W, N]`` (normal only).

        ``method``:
            * ``"zscore"`` -> center = mean, scale = std.
            * ``"robust"`` -> center = median, scale = 1.4826 * MAD (outlier-robust,
              often better when the normal residuals are heavy-tailed).
        """
        flat = residuals.reshape(-1, residuals.shape[-1])
        if method == "robust":
            center = np.median(flat, axis=0)
            mad = np.median(np.abs(flat - center), axis=0)
            scale = 1.4826 * mad + eps
        else:
            center = flat.mean(axis=0)
            scale = flat.std(axis=0) + eps
        return cls(mean=center, std=scale)

    def transform(self, residuals: np.ndarray) -> np.ndarray:
        return (residuals - self.mean) / self.std


@dataclass
class LevelChannelNormalizer:
    """Normal-data calibration for each ``(noise level, sensor)`` pair."""

    center: np.ndarray  # [L, N]
    scale: np.ndarray   # [L, N]
    method: str

    @classmethod
    def fit(
        cls,
        residuals: np.ndarray,
        eps: float = 1e-6,
        method: str = "robust",
    ) -> "LevelChannelNormalizer":
        """Fit on normal residuals shaped ``[windows, levels, W, sensors]``."""
        residuals = np.asarray(residuals, dtype=np.float64)
        if residuals.ndim != 4:
            raise ValueError(
                "Per-level residuals must have shape [K, L, W, N], got "
                f"{residuals.shape}"
            )

        # [K, L, W, N] -> [L, K*W, N]
        flat = residuals.transpose(1, 0, 2, 3).reshape(
            residuals.shape[1], -1, residuals.shape[-1]
        )
        normalized_method = str(method).lower()
        if normalized_method == "robust":
            center = np.median(flat, axis=1)
            mad = np.median(np.abs(flat - center[:, None, :]), axis=1)
            scale = 1.4826 * mad + eps
        elif normalized_method == "zscore":
            center = flat.mean(axis=1)
            scale = flat.std(axis=1) + eps
        else:
            raise ValueError(
                f"Unknown level calibration method '{method}'. "
                "Expected 'robust' or 'zscore'."
            )
        return cls(center=center, scale=scale, method=normalized_method)

    def transform(
        self,
        residuals: np.ndarray,
        clip_min: float | None = 0.0,
    ) -> np.ndarray:
        residuals = np.asarray(residuals)
        if residuals.ndim != 4:
            raise ValueError(
                "Per-level residuals must have shape [K, L, W, N], got "
                f"{residuals.shape}"
            )
        dtype = np.float32 if residuals.dtype.itemsize <= 4 else np.float64
        residuals = residuals.astype(dtype, copy=False)
        center = self.center.astype(dtype, copy=False)
        scale = self.scale.astype(dtype, copy=False)
        calibrated = (
            residuals - center[None, :, None, :]
        ) / scale[None, :, None, :]
        if clip_min is not None:
            calibrated = np.maximum(calibrated, float(clip_min))
        return calibrated


def aggregate_levels(residuals: np.ndarray, mode: str = "mean") -> np.ndarray:
    """Collapse ``[K, L, W, N]`` to ``[K, W, N]`` after calibration."""
    residuals = np.asarray(residuals)
    if residuals.ndim != 4:
        raise ValueError(
            "Per-level residuals must have shape [K, L, W, N], got "
            f"{residuals.shape}"
        )
    normalized_mode = str(mode).lower()
    if normalized_mode == "mean":
        return residuals.mean(axis=1)
    if normalized_mode == "median":
        return np.median(residuals, axis=1)
    if normalized_mode == "max":
        return residuals.max(axis=1)
    raise ValueError(
        f"Unknown level aggregation mode '{mode}'. "
        "Expected 'mean', 'median', or 'max'."
    )


def aggregate_channels(residuals: np.ndarray, mode: str = "mean") -> np.ndarray:
    """Collapse the channel axis: ``[..., N] -> [...]``."""
    if mode == "mean":
        return residuals.mean(axis=-1)
    if mode == "max":
        return residuals.max(axis=-1)
    if mode == "sum":
        return residuals.sum(axis=-1)
    raise ValueError(f"Unknown channel aggregation mode: {mode}")


def windows_to_timeline(
    window_scores: np.ndarray,
    start_indices: List[int],
    window_size: int,
    length: int,
    segment_end_indices: Optional[List[int]] = None,
) -> np.ndarray:
    """Overlap-add per-window scores back onto a length-``length`` timeline.

    ``window_scores`` has shape ``[num_windows, window_size]``. Overlapping
    contributions are averaged. Padding beyond ``length`` is discarded. When
    the dataset was assembled from multiple files, ``segment_end_indices``
    also prevents a padded tail window from contributing to the next file.
    """
    if segment_end_indices is not None and len(segment_end_indices) != len(start_indices):
        raise ValueError(
            "segment_end_indices must have one entry per window start "
            f"({len(segment_end_indices)} != {len(start_indices)})"
        )
    acc = np.zeros(length, dtype=np.float64)
    cnt = np.zeros(length, dtype=np.float64)

    for w, start in enumerate(start_indices):
        segment_end = (
            length
            if segment_end_indices is None
            else int(segment_end_indices[w])
        )
        end = min(start + window_size, segment_end, length)
        valid = end - start
        if valid <= 0:
            continue
        acc[start:end] += window_scores[w, :valid]
        cnt[start:end] += 1.0

    cnt = np.maximum(cnt, 1.0)
    return acc / cnt


def ema_smooth(
    scores: np.ndarray,
    alpha: float = 0.0,
    segment_lengths: Optional[List[int]] = None,
) -> np.ndarray:
    """Causal EMA, resetting at independent file/sequence boundaries."""
    if alpha <= 0.0 or len(scores) == 0:
        return scores
    if segment_lengths is None:
        segment_lengths = [len(scores)]
    segment_lengths = [int(value) for value in segment_lengths]
    if any(value <= 0 for value in segment_lengths) or sum(segment_lengths) != len(scores):
        raise ValueError(
            "segment_lengths must be positive and sum to the score length "
            f"({sum(segment_lengths)} != {len(scores)})"
        )
    out = np.empty_like(scores, dtype=np.float64)
    offset = 0
    for segment_length in segment_lengths:
        segment_end = offset + segment_length
        out[offset] = scores[offset]
        for i in range(offset + 1, segment_end):
            out[i] = alpha * out[i - 1] + (1.0 - alpha) * scores[i]
        offset = segment_end
    return out


def build_timeline_score(
    residuals: np.ndarray,
    start_indices: List[int],
    window_size: int,
    length: int,
    normalizer: Optional[ChannelNormalizer] = None,
    channel_agg: str = "mean",
    ema_alpha: float = 0.0,
    segment_end_indices: Optional[List[int]] = None,
    segment_lengths: Optional[List[int]] = None,
) -> np.ndarray:
    """Full pipeline: residuals ``[num_windows, W, N]`` -> timeline ``[length]``."""
    if normalizer is not None:
        residuals = normalizer.transform(residuals)
    window_scores = aggregate_channels(residuals, mode=channel_agg)
    timeline = windows_to_timeline(
        window_scores,
        start_indices,
        window_size,
        length,
        segment_end_indices=segment_end_indices,
    )
    timeline = ema_smooth(
        timeline, alpha=ema_alpha, segment_lengths=segment_lengths
    )
    return timeline
