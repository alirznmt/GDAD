"""Spectral utilities for SNR-based implicit conditioning (SIC).

Adapted from the ICDiffAD idea of estimating an *instance-level* signal-to-noise
ratio (ISNR) by low-pass filtering the signal along **time** and comparing the
low-frequency (signal) energy to the residual (high-frequency) energy.

Everything here is fully vectorized and device-agnostic (CPU/CUDA). Our tensors
use the project convention ``x : [B, W, N]`` (batch, time, sensors), unlike
ICDiffAD's ``[B, 1, H, W]`` image layout.
"""
from __future__ import annotations

from typing import Tuple

import torch
import torch.nn.functional as F


def gaussian_kernel1d(kernel_size: int, sigma: float, device, dtype) -> torch.Tensor:
    """Return a normalized 1-D Gaussian kernel of shape ``[kernel_size]``."""
    if kernel_size % 2 == 0:  # force odd so padding stays symmetric
        kernel_size += 1
    half = (kernel_size - 1) / 2.0
    coords = torch.arange(kernel_size, device=device, dtype=dtype) - half
    sigma = max(float(sigma), 1e-6)
    k = torch.exp(-(coords ** 2) / (2.0 * sigma ** 2))
    k = k / k.sum().clamp_min(1e-12)
    return k


def gaussian_lowpass_time(x: torch.Tensor, kernel_size: int, sigma: float) -> torch.Tensor:
    """Depthwise Gaussian low-pass along the time axis.

    Args:
        x: ``[B, W, N]``.
    Returns:
        low-frequency component, same shape ``[B, W, N]``.
    """
    b, w, n = x.shape
    k = gaussian_kernel1d(kernel_size, sigma, x.device, x.dtype)  # [ks]
    ks = k.numel()
    weight = k.view(1, 1, ks).repeat(n, 1, 1)  # [N, 1, ks] depthwise

    xt = x.transpose(1, 2)  # [B, N, W]
    pad = ks // 2
    # Reflect padding avoids spurious edge energy at window boundaries.
    pad = min(pad, max(w - 1, 0))
    if pad > 0:
        xt = F.pad(xt, (pad, pad), mode="reflect")
        # If kernel is larger than the (padded) signal, fall back to 'replicate'.
        if xt.shape[-1] < ks:
            xt = F.pad(x.transpose(1, 2), (ks // 2, ks // 2), mode="replicate")
    low = F.conv1d(xt, weight, groups=n)  # [B, N, W]
    if low.shape[-1] != w:  # guard against off-by-one from odd kernels
        low = low[..., :w]
    return low.transpose(1, 2).contiguous()  # [B, W, N]


def estimate_isnr(
    x: torch.Tensor,
    kernel_size: int = 31,
    sigma: float = 5.0,
    granularity: str = "window",
    eps: float = 1e-9,
) -> Tuple[torch.Tensor, torch.Tensor]:
    """Estimate instance SNR and signal power.

    Args:
        x: ``[B, W, N]``.
        granularity: ``"window"`` -> reduce over (time, sensors) => shape ``[B]``;
                     ``"node"``   -> reduce over time only          => shape ``[B, N]``.
    Returns:
        (isnr, power) where ``isnr = E[low^2] / E[(x-low)^2]`` and
        ``power = E[x^2]`` at the requested granularity.
    """
    low = gaussian_lowpass_time(x, kernel_size, sigma)
    resid = x - low

    if granularity == "window":
        dims = (1, 2)
    elif granularity == "node":
        dims = (1,)
    else:
        raise ValueError(f"Unknown ISNR granularity: {granularity}")

    signal = (low ** 2).mean(dim=dims)
    noise = (resid ** 2).mean(dim=dims) + eps
    power = (x ** 2).mean(dim=dims)
    isnr = signal / noise
    return isnr, power
