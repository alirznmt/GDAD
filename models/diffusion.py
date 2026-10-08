"""Gaussian diffusion (DDPM) utilities used by GDAD.

We implement epsilon-prediction DDPM with a cosine noise schedule, the training
loss, and the inference-time *denoising residual* used for anomaly scoring. Full
ancestral / DDIM sampling is included for completeness and reconstruction, but
anomaly detection only needs the cheap one-step denoising residuals at a small
set of noise levels.
"""
from __future__ import annotations

from typing import Dict, List

import torch
import torch.nn as nn
import torch.nn.functional as F


def cosine_beta_schedule(timesteps: int, s: float = 0.008) -> torch.Tensor:
    """Cosine schedule from Nichol & Dhariwal (2021)."""
    steps = timesteps + 1
    x = torch.linspace(0, timesteps, steps)
    alphas_cumprod = torch.cos(((x / timesteps) + s) / (1 + s) * torch.pi * 0.5) ** 2
    alphas_cumprod = alphas_cumprod / alphas_cumprod[0]
    betas = 1 - (alphas_cumprod[1:] / alphas_cumprod[:-1])
    return torch.clip(betas, 1e-5, 0.999)


def linear_beta_schedule(timesteps: int) -> torch.Tensor:
    return torch.linspace(1e-4, 0.02, timesteps)


def snr_beta_schedule(
    timesteps: int,
    target_snr_db: float = -10.0,
    mt: float = 10.0,
    curve: str = "linear",
) -> torch.Tensor:
    """SNR-pinned schedule (ICDiffAD-style).

    Builds ``alpha_t = M_t / M_{t-1}`` such that the terminal cumulative product
    equals ``tsnr / (1 + tsnr)`` where ``tsnr = 10**(target_snr_db/10)``. This
    guarantees the terminal signal-to-noise ratio equals the *target* SNR, i.e.
    the forward process never collapses to pure noise. ``curve`` shapes the
    per-step decay g(t): ``linear`` (g=t), ``quadratic`` (g=t^2), or
    ``cosine`` (g=1-cos(pi t / T)).
    """
    import math

    tsnr = 10.0 ** (target_snr_db / 10.0)
    m0 = mt * (1.0 + tsnr) / tsnr  # so that MT/M0 = tsnr/(1+tsnr)

    if curve == "linear":
        def g(t: int) -> float:
            return float(t)
    elif curve == "quadratic":
        def g(t: int) -> float:
            return float(t) ** 2
    elif curve == "cosine":
        def g(t: int) -> float:
            return 1.0 - math.cos(math.pi * t / timesteps)
    else:
        raise ValueError(f"Unknown snr curve: {curve}")

    g_vals = [g(t) for t in range(1, timesteps + 1)]
    sum_g = sum(g_vals) or 1.0
    scale = math.log(mt / m0) / sum_g
    ratios = [math.exp(gv * scale) for gv in g_vals]  # alpha_t values
    betas = torch.tensor([1.0 - r for r in ratios], dtype=torch.float32)
    return torch.clip(betas, 1e-6, 0.999)


def _extract(a: torch.Tensor, t: torch.Tensor, shape) -> torch.Tensor:
    """Gather schedule values at indices ``t`` and broadcast to ``shape``."""
    out = a.gather(0, t)
    return out.reshape(t.shape[0], *([1] * (len(shape) - 1)))


class GaussianDiffusion(nn.Module):
    """DDPM wrapper around an epsilon-prediction denoiser."""

    def __init__(
        self,
        model: nn.Module,
        timesteps: int = 100,
        schedule: str = "cosine",
        loss_type: str = "huber",
        snr_target_db: float = -10.0,
        snr_mt: float = 10.0,
        snr_curve: str = "linear",
    ) -> None:
        super().__init__()
        self.model = model
        self.timesteps = timesteps
        self.loss_type = loss_type

        if schedule == "cosine":
            betas = cosine_beta_schedule(timesteps)
        elif schedule == "linear":
            betas = linear_beta_schedule(timesteps)
        elif schedule == "snr":
            betas = snr_beta_schedule(timesteps, snr_target_db, snr_mt, snr_curve)
        else:
            raise ValueError(f"Unknown schedule: {schedule}")

        alphas = 1.0 - betas
        alphas_cumprod = torch.cumprod(alphas, dim=0)
        alphas_cumprod_prev = F.pad(alphas_cumprod[:-1], (1, 0), value=1.0)

        # Register as buffers so they move with .to(device) and are checkpointed.
        self.register_buffer("betas", betas)
        self.register_buffer("alphas_cumprod", alphas_cumprod)
        self.register_buffer("alphas_cumprod_prev", alphas_cumprod_prev)
        self.register_buffer("sqrt_alphas_cumprod", torch.sqrt(alphas_cumprod))
        self.register_buffer(
            "sqrt_one_minus_alphas_cumprod", torch.sqrt(1.0 - alphas_cumprod)
        )
        self.register_buffer("sqrt_recip_alphas_cumprod", torch.sqrt(1.0 / alphas_cumprod))
        self.register_buffer(
            "sqrt_recipm1_alphas_cumprod", torch.sqrt(1.0 / alphas_cumprod - 1.0)
        )
        posterior_variance = (
            betas * (1.0 - alphas_cumprod_prev) / (1.0 - alphas_cumprod)
        )
        self.register_buffer("posterior_variance", posterior_variance)

    # ----------------------------------------------------------------- forward
    def q_sample(
        self, x_start: torch.Tensor, t: torch.Tensor, noise: torch.Tensor
    ) -> torch.Tensor:
        """Sample ``x_t ~ q(x_t | x_0)``."""
        sqrt_ac = _extract(self.sqrt_alphas_cumprod, t, x_start.shape)
        sqrt_omac = _extract(self.sqrt_one_minus_alphas_cumprod, t, x_start.shape)
        return sqrt_ac * x_start + sqrt_omac * noise

    def predict_x0_from_eps(
        self, x_t: torch.Tensor, t: torch.Tensor, eps: torch.Tensor
    ) -> torch.Tensor:
        sqrt_recip = _extract(self.sqrt_recip_alphas_cumprod, t, x_t.shape)
        sqrt_recipm1 = _extract(self.sqrt_recipm1_alphas_cumprod, t, x_t.shape)
        return sqrt_recip * x_t - sqrt_recipm1 * eps

    def _loss(self, pred: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
        if self.loss_type == "l2":
            return F.mse_loss(pred, target)
        if self.loss_type == "l1":
            return F.l1_loss(pred, target)
        if self.loss_type == "huber":
            return F.smooth_l1_loss(pred, target)
        raise ValueError(f"Unknown loss type: {self.loss_type}")

    def training_loss(self, x_start: torch.Tensor) -> torch.Tensor:
        """Sample a random step per item and return the denoising loss."""
        b = x_start.shape[0]
        t = torch.randint(0, self.timesteps, (b,), device=x_start.device).long()
        noise = torch.randn_like(x_start)
        x_t = self.q_sample(x_start, t, noise)
        eps_pred = self.model(x_t, t)
        return self._loss(eps_pred, noise)

    # --------------------------------------------------------------- scoring
    @torch.no_grad()
    def denoising_residual(
        self,
        x_start: torch.Tensor,
        eval_steps: List[int],
        num_samples: int = 1,
    ) -> Dict[str, torch.Tensor]:
        """Per-(time, channel) denoising residuals averaged over noise levels.

        For each step in ``eval_steps`` we draw ``num_samples`` noise vectors,
        noise the input, predict the noise, and accumulate two residuals:

        * ``recon``: ``(x0 - x0_hat)^2`` in observation space (interpretable,
          per-channel, the primary signal for pointwise detection), and
        * ``eps``:  ``(eps - eps_hat)^2`` in noise space (complementary).

        Returns a dict of tensors shaped like ``x_start`` (``[B, W, N]``).
        """
        recon_acc = torch.zeros_like(x_start)
        eps_acc = torch.zeros_like(x_start)
        count = 0

        for step in eval_steps:
            t = torch.full(
                (x_start.shape[0],), int(step), device=x_start.device, dtype=torch.long
            )
            for _ in range(num_samples):
                noise = torch.randn_like(x_start)
                x_t = self.q_sample(x_start, t, noise)
                eps_pred = self.model(x_t, t)
                x0_hat = self.predict_x0_from_eps(x_t, t, eps_pred)
                recon_acc += (x_start - x0_hat) ** 2
                eps_acc += (noise - eps_pred) ** 2
                count += 1

        recon_acc /= max(count, 1)
        eps_acc /= max(count, 1)
        return {"recon": recon_acc, "eps": eps_acc}

    # --------------------------------------------------------------- sampling
    @torch.no_grad()
    def ddim_reconstruct(
        self,
        x_start: torch.Tensor,
        start_step: int,
        num_steps: int = 10,
    ) -> torch.Tensor:
        """Noise ``x_start`` to ``start_step`` then DDIM-denoise back to x0.

        Useful as a reconstruction-based residual that conditions on the global
        window structure. Deterministic (eta=0).
        """
        device = x_start.device
        t0 = torch.full((x_start.shape[0],), start_step, device=device, dtype=torch.long)
        noise = torch.randn_like(x_start)
        x_t = self.q_sample(x_start, t0, noise)

        step_seq = torch.linspace(start_step, 0, num_steps + 1).long().tolist()
        for i in range(len(step_seq) - 1):
            cur, nxt = step_seq[i], step_seq[i + 1]
            t = torch.full((x_start.shape[0],), cur, device=device, dtype=torch.long)
            eps = self.model(x_t, t)
            x0 = self.predict_x0_from_eps(x_t, t, eps)
            ac_next = self.alphas_cumprod[nxt] if nxt >= 0 else torch.tensor(1.0, device=device)
            x_t = torch.sqrt(ac_next) * x0 + torch.sqrt(1.0 - ac_next) * eps
        return x_t
