"""Top-level GDAD model.

GDAD = Graph-conditioned Denoising Diffusion for pointwise Anomaly Detection.

It wires the graph-temporal denoiser into a Gaussian diffusion process and
exposes:

* ``training_loss(x)``     -- DDPM denoising loss for unsupervised training, and
* ``score_window(x, ...)`` -- per-(time, channel) anomaly residuals for scoring.
"""
from __future__ import annotations

from typing import Dict, List

import torch
import torch.nn as nn

from .denoiser import GraphTemporalDenoiser
from .diffusion import GaussianDiffusion


def assert_effective_graph_mode(model: nn.Module, expected_mode: str | None) -> str:
    """Fail fast when config wiring does not reach the graph learner.

    Graph-ablation checkpoints are scientifically invalid if a requested mode
    is merely recorded in the config but ignored by the instantiated model.
    """
    learner = getattr(
        getattr(getattr(model, "diffusion", None), "model", None),
        "graph_learner",
        None,
    )
    effective_mode = getattr(learner, "mode", None)
    if effective_mode is None:
        raise RuntimeError(
            "The instantiated model does not expose graph_learner.mode. "
            "The graph-mode wiring is missing or an outdated model source "
            "tree is being used."
        )

    normalized_expected = (
        None
        if expected_mode is None
        else str(expected_mode).lower().replace("-", "_")
    )
    aliases = {
        "none": "identity",
        "no_graph": "identity",
        "no_cross_sensor": "identity",
    }
    normalized_expected = aliases.get(normalized_expected, normalized_expected)
    if normalized_expected is not None and effective_mode != normalized_expected:
        raise RuntimeError(
            "Graph-mode wiring mismatch: "
            f"config requested '{normalized_expected}', but the instantiated "
            f"GraphLearner uses '{effective_mode}'. Synchronize models/gdad.py, "
            "models/denoiser.py, and models/graph.py before training."
        )
    return str(effective_mode)


class GDAD(nn.Module):
    def __init__(
        self,
        num_nodes: int,
        hidden_dim: int = 64,
        num_layers: int = 4,
        kernel_size: int = 3,
        time_dim: int = 128,
        node_dim: int = 16,
        top_k: int = 0,
        dynamic_graph: bool = True,
        graph_mode: str | None = None,
        groups: int = 8,
        timesteps: int = 100,
        schedule: str = "cosine",
        loss_type: str = "huber",
        snr_target_db: float = -10.0,
        snr_mt: float = 10.0,
        snr_curve: str = "linear",
    ) -> None:
        super().__init__()
        self.num_nodes = num_nodes

        denoiser = GraphTemporalDenoiser(
            num_nodes=num_nodes,
            hidden_dim=hidden_dim,
            num_layers=num_layers,
            kernel_size=kernel_size,
            time_dim=time_dim,
            node_dim=node_dim,
            top_k=top_k,
            dynamic_graph=dynamic_graph,
            graph_mode=graph_mode,
            groups=groups,
        )
        self.diffusion = GaussianDiffusion(
            model=denoiser,
            timesteps=timesteps,
            schedule=schedule,
            loss_type=loss_type,
            snr_target_db=snr_target_db,
            snr_mt=snr_mt,
            snr_curve=snr_curve,
        )

    def training_loss(self, x: torch.Tensor) -> torch.Tensor:
        return self.diffusion.training_loss(x)

    @torch.no_grad()
    def score_window(
        self,
        x: torch.Tensor,
        eval_steps: List[int],
        num_samples: int = 1,
        recon_weight: float = 1.0,
        eps_weight: float = 0.0,
    ) -> Dict[str, torch.Tensor]:
        """Return aggregate residuals plus ``eps_by_level`` as ``[B,L,W,N]``."""
        res = self.diffusion.denoising_residual(x, eval_steps, num_samples=num_samples)
        combined = recon_weight * res["recon"] + eps_weight * res["eps"]
        res["combined"] = combined
        return res


def build_model(cfg, num_nodes: int) -> GDAD:
    """Instantiate GDAD from a config block plus the runtime sensor count."""
    m = cfg.model
    d = cfg.diffusion
    # scheduler_type is the new flag: "original" keeps the existing schedule
    # (cosine/linear); "snr" activates the SNR-pinned schedule. Falls back to the
    # legacy "schedule" key so old configs/checkpoints behave identically.
    scheduler_type = d.get("scheduler_type", "original")
    if scheduler_type in (None, "original"):
        schedule = d.get("schedule", "cosine")
    else:
        schedule = scheduler_type
    return GDAD(
        num_nodes=num_nodes,
        hidden_dim=m.get("hidden_dim", 64),
        num_layers=m.get("num_layers", 4),
        kernel_size=m.get("kernel_size", 3),
        time_dim=m.get("time_dim", 128),
        node_dim=m.get("node_dim", 16),
        top_k=m.get("top_k", 0),
        dynamic_graph=m.get("dynamic_graph", True),
        graph_mode=m.get("graph_mode"),
        groups=m.get("groups", 8),
        timesteps=d.get("timesteps", 100),
        schedule=schedule,
        loss_type=d.get("loss_type", "huber"),
        snr_target_db=d.get("target_snr_db", -10.0),
        snr_mt=d.get("snr_mt", 10.0),
        snr_curve=d.get("snr_curve", "linear"),
    )
