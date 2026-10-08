"""SNR-based Implicit Conditioning (SIC) sampler for GDAD.

This is an *inference-only* add-on inspired by ICDiffAD. Instead of scoring a
window by one-step denoising at a fixed noise level, SIC:

1. estimates the window's instance SNR (ISNR) via Gaussian low-pass filtering,
2. maps that ISNR to an adaptive corruption level (a diffusion step ``t*``),
3. partially corrupts the *real input* to ``t*`` (not pure Gaussian noise),
4. runs a short deterministic (DDIM, eta=0) reverse process back to ``x0``,
5. returns the per-(time, channel) reconstruction residual ``(x0 - x0_hat)^2``.

Intuition for pointwise precision: clean/normal windows have high ISNR -> tiny
corruption -> near-lossless reconstruction -> very small residual. Anomalous
windows have low ISNR -> stronger corruption that erases the (high-frequency)
anomaly and reconstructs a "normal-looking" window -> large, localized residual.
This lowers and tightens the normal-score floor, reducing false positives.

The graph-temporal denoiser is used unchanged as the reverse operator, so the
graph identity of the model is fully preserved. No retraining is required and
existing checkpoints work as-is.
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Optional

import torch

from utils.spectral import estimate_isnr


@dataclass
class SICConfig:
    gaussian_kernel_size: int = 31
    gaussian_sigma: float = 5.0
    granularity: str = "window"           # "window" | "node"
    min_reverse_step_ratio: float = 0.05  # clamp t* >= ratio * T
    max_reverse_step_ratio: float = 0.4   # clamp t* <= ratio * T (bounds cost)
    ddim_steps: int = 8                   # reverse steps (independent of t*)
    deterministic_noise: bool = True      # seed the corruption noise
    num_samples: int = 1                  # average residual over corruptions
    seed: int = 42
    clip_denoised: bool = False           # clamp x0_hat to observed range
    clip_range: float = 5.0               # used only if clip_denoised


class SICSampler:
    """Wraps a ``GaussianDiffusion`` to produce SIC reconstruction residuals."""

    def __init__(self, diffusion, cfg: SICConfig) -> None:
        self.diffusion = diffusion
        self.cfg = cfg
        self.T = int(diffusion.timesteps)

    # ------------------------------------------------------------------ helpers
    def _gather(self, buf: torch.Tensor, t: torch.Tensor) -> torch.Tensor:
        """Gather schedule value at per-sample step ``t`` -> ``[B, 1, 1]``."""
        return buf.gather(0, t).view(-1, 1, 1)

    @torch.no_grad()
    def estimate_start_step(self, x0: torch.Tensor) -> torch.Tensor:
        """Return an adaptive per-window start step ``t* : [B]`` (long)."""
        cfg = self.cfg
        isnr, power = estimate_isnr(
            x0, cfg.gaussian_kernel_size, cfg.gaussian_sigma, granularity="window"
        )  # both [B]

        # Target retained-signal fraction (alpha_bar*) from ISNR and signal power.
        alpha_star = isnr / (power + isnr + 1e-9)             # [B] in (0, 1)
        alpha_star = alpha_star.clamp(1e-4, 1.0 - 1e-4)
        sqrt_alpha_star = torch.sqrt(alpha_star)             # [B]

        # Nearest step on the (integer) sqrt(alpha_bar) grid.
        grid = self.diffusion.sqrt_alphas_cumprod            # [T]
        dist = torch.abs(grid.unsqueeze(0) - sqrt_alpha_star.unsqueeze(1))  # [B, T]
        t_star = torch.argmin(dist, dim=1)                   # [B]

        t_min = max(1, int(round(cfg.min_reverse_step_ratio * self.T)))
        t_max = max(t_min, int(round(cfg.max_reverse_step_ratio * self.T)))
        t_max = min(t_max, self.T - 1)
        t_star = t_star.clamp(t_min, t_max)
        return t_star.long()

    @torch.no_grad()
    def reconstruct(self, x0: torch.Tensor) -> torch.Tensor:
        """Return SIC reconstruction of ``x0`` (same shape ``[B, W, N]``)."""
        cfg = self.cfg
        diff = self.diffusion
        device = x0.device
        t_start = self.estimate_start_step(x0)               # [B]
        ac = diff.alphas_cumprod                             # [T]

        recon_last: Optional[torch.Tensor] = None
        for s in range(max(1, cfg.num_samples)):
            if cfg.deterministic_noise:
                gen = torch.Generator(device=device)
                gen.manual_seed(cfg.seed + s)
                noise = torch.randn(x0.shape, generator=gen, device=device, dtype=x0.dtype)
            else:
                noise = torch.randn_like(x0)

            # Partial corruption of the real input to level t_start.
            ac_start = self._gather(ac, t_start)             # [B,1,1]
            x = torch.sqrt(ac_start) * x0 + torch.sqrt(1.0 - ac_start) * noise

            # Per-sample strided DDIM (eta=0): fixed #steps regardless of t_start.
            x0_hat = x
            steps = max(1, cfg.ddim_steps)
            for k in range(steps):
                frac_cur = (steps - k) / steps
                frac_next = (steps - k - 1) / steps
                t_cur = (frac_cur * t_start.float()).round().long().clamp(0, self.T - 1)
                t_next = (frac_next * t_start.float()).round().long().clamp(0, self.T - 1)

                eps_hat = diff.model(x, t_cur)               # graph-temporal denoiser
                ac_cur = self._gather(ac, t_cur)
                ac_next = self._gather(ac, t_next)
                x0_hat = (x - torch.sqrt(1.0 - ac_cur) * eps_hat) / torch.sqrt(ac_cur)
                if cfg.clip_denoised:
                    x0_hat = x0_hat.clamp(-cfg.clip_range, cfg.clip_range)
                x = torch.sqrt(ac_next) * x0_hat + torch.sqrt(1.0 - ac_next) * eps_hat

            recon = x0_hat
            recon_last = recon if recon_last is None else recon_last + recon
        return recon_last / max(1, cfg.num_samples)

    @torch.no_grad()
    def residual(self, x0: torch.Tensor) -> torch.Tensor:
        """Per-(time, channel) SIC reconstruction residual ``[B, W, N]``.

        Averaging over ``num_samples`` corruption draws is handled inside
        ``reconstruct`` (via the per-sample seed offset).
        """
        recon = self.reconstruct(x0)
        return (x0 - recon) ** 2


def build_sic_config(cfg) -> SICConfig:
    """Create a SICConfig from the project config's ``sic`` block."""
    s = cfg.get("sic", {}) if hasattr(cfg, "get") else {}
    seed = int(cfg.get("seed", 42)) if hasattr(cfg, "get") else 42
    return SICConfig(
        gaussian_kernel_size=int(s.get("gaussian_kernel_size", 31)),
        gaussian_sigma=float(s.get("gaussian_sigma", 5.0)),
        granularity=str(s.get("granularity", "window")),
        min_reverse_step_ratio=float(s.get("min_reverse_step_ratio", 0.05)),
        max_reverse_step_ratio=float(s.get("max_reverse_step_ratio", 0.4)),
        ddim_steps=int(s.get("ddim_steps", 8)),
        deterministic_noise=bool(s.get("deterministic_noise", True)),
        num_samples=int(s.get("num_samples", 1)),
        seed=int(s.get("seed", seed)),
        clip_denoised=bool(s.get("clip_denoised", False)),
        clip_range=float(s.get("clip_range", 5.0)),
    )
