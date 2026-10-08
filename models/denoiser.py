r"""Graph-temporal epsilon-prediction network (the diffusion denoiser).

Architecture (per spatio-temporal block):

    h --> [Gated dilated TCN over time] --+
      \                                   |
       \--> [Graph conv over sensors] ----+--> residual + GroupNorm + GELU

Time-step conditioning is injected with FiLM (per-channel scale/shift derived
from a sinusoidal embedding of the diffusion step ``t``). The network maps a
noised window ``x_t: [B, W, N]`` and step ``t: [B]`` to the predicted noise
``eps: [B, W, N]`` (same shape), staying in observation space so that residuals
remain per-channel and per-timestep -- exactly what pointwise scoring needs.
"""
from __future__ import annotations

import math

import torch
import torch.nn as nn
import torch.nn.functional as F

from .graph import GraphConv, GraphLearner


def sinusoidal_embedding(t: torch.Tensor, dim: int) -> torch.Tensor:
    """Standard transformer/DDPM sinusoidal timestep embedding."""
    half = dim // 2
    device = t.device
    freqs = torch.exp(
        -math.log(10000.0) * torch.arange(half, device=device).float() / max(half - 1, 1)
    )
    args = t.float().unsqueeze(1) * freqs.unsqueeze(0)
    emb = torch.cat([torch.sin(args), torch.cos(args)], dim=1)
    if dim % 2 == 1:  # zero-pad if odd
        emb = F.pad(emb, (0, 1))
    return emb


class TemporalGatedConv(nn.Module):
    """Gated dilated 1D conv applied per node over the time axis."""

    def __init__(self, channels: int, kernel_size: int, dilation: int) -> None:
        super().__init__()
        pad = (kernel_size - 1) * dilation // 2
        self.filter = nn.Conv1d(channels, channels, kernel_size, padding=pad, dilation=dilation)
        self.gate = nn.Conv1d(channels, channels, kernel_size, padding=pad, dilation=dilation)
        self.out = nn.Conv1d(channels, channels, 1)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # x: [B, N, C, W] -> [B*N, C, W]
        b, n, c, w = x.shape
        h = x.reshape(b * n, c, w)
        h = torch.tanh(self.filter(h)) * torch.sigmoid(self.gate(h))
        h = self.out(h)
        # Guard against dilation length drift.
        if h.shape[-1] != w:
            h = h[..., :w]
        return h.reshape(b, n, c, w)


class STBlock(nn.Module):
    """One spatio-temporal residual block with FiLM time conditioning."""

    def __init__(
        self,
        channels: int,
        time_dim: int,
        kernel_size: int = 3,
        dilation: int = 1,
        groups: int = 8,
    ) -> None:
        super().__init__()
        self.temporal = TemporalGatedConv(channels, kernel_size, dilation)
        self.spatial = GraphConv(channels)
        self.film = nn.Linear(time_dim, 2 * channels)
        g = math.gcd(groups, channels)
        self.norm = nn.GroupNorm(g, channels)

    def forward(self, x: torch.Tensor, adj: torch.Tensor, temb: torch.Tensor) -> torch.Tensor:
        # FiLM parameters: [B, 2C] -> scale/shift broadcast over N, W
        scale, shift = self.film(temb).chunk(2, dim=-1)
        scale = scale[:, None, :, None]
        shift = shift[:, None, :, None]

        temporal = self.temporal(x)
        spatial = self.spatial(x, adj)
        h = x + temporal + spatial
        h = h * (1.0 + scale) + shift

        b, n, c, w = h.shape
        h = self.norm(h.reshape(b * n, c, w)).reshape(b, n, c, w)
        h = F.gelu(h)
        return h


class GraphTemporalDenoiser(nn.Module):
    """Epsilon network: ``(x_t, t) -> eps_hat`` with shape ``[B, W, N]``."""

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
    ) -> None:
        super().__init__()
        self.num_nodes = num_nodes
        self.hidden_dim = hidden_dim

        self.input_proj = nn.Conv1d(1, hidden_dim, kernel_size=kernel_size,
                                    padding=kernel_size // 2)

        self.time_mlp = nn.Sequential(
            nn.Linear(time_dim, time_dim),
            nn.GELU(),
            nn.Linear(time_dim, time_dim),
        )
        self.time_dim = time_dim

        self.graph_learner = GraphLearner(
            num_nodes=num_nodes,
            node_dim=node_dim,
            summary_dim=hidden_dim,
            top_k=top_k,
            dynamic=dynamic_graph,
            mode=graph_mode,
        )

        # Exponentially increasing dilations give a wide temporal receptive field.
        self.blocks = nn.ModuleList(
            [
                STBlock(
                    channels=hidden_dim,
                    time_dim=time_dim,
                    kernel_size=kernel_size,
                    dilation=2 ** (i % 4),
                    groups=groups,
                )
                for i in range(num_layers)
            ]
        )

        self.output_proj = nn.Sequential(
            nn.Conv1d(hidden_dim, hidden_dim, 1),
            nn.GELU(),
            nn.Conv1d(hidden_dim, 1, 1),
        )

    def forward(self, x_t: torch.Tensor, t: torch.Tensor) -> torch.Tensor:
        # x_t: [B, W, N] -> node-major [B, N, W]
        b, w, n = x_t.shape
        x = x_t.permute(0, 2, 1).contiguous()  # [B, N, W]

        # Value embedding: shared Conv1d over time per node.
        h = self.input_proj(x.reshape(b * n, 1, w))  # [B*N, C, W]
        h = h.reshape(b, n, self.hidden_dim, w)  # [B, N, C, W]

        # Timestep embedding.
        temb = sinusoidal_embedding(t, self.time_dim)
        temb = self.time_mlp(temb)  # [B, time_dim]

        # Data-dependent graph from per-node temporal summary.
        node_summary = h.mean(dim=-1)  # [B, N, C]
        adj = self.graph_learner(node_summary)  # [B, N, N] or [1, N, N]
        if adj.shape[0] == 1 and b > 1:
            adj = adj.expand(b, -1, -1)

        for block in self.blocks:
            h = block(h, adj, temb)

        out = self.output_proj(h.reshape(b * n, self.hidden_dim, w))  # [B*N, 1, W]
        out = out.reshape(b, n, w).permute(0, 2, 1).contiguous()  # [B, W, N]
        return out
