"""Graph construction and graph convolution for GDAD.

The original model is the ``hybrid`` mode: a sparse directed static affinity
and a dense, corruption-instance-dependent query-key affinity are mixed by one
learned global gate.  Explicit graph modes are also available for controlled
ablations.  When ``mode`` is omitted, the legacy ``dynamic`` boolean retains
its original behaviour so existing configs and checkpoints remain compatible.
"""
from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F


class GraphLearner(nn.Module):
    """Produce a batched row-normalized hybrid adjacency ``[B, N, N]``."""

    def __init__(
        self,
        num_nodes: int,
        node_dim: int = 16,
        summary_dim: int = 32,
        top_k: int = 0,
        dynamic: bool = True,
        mode: str | None = None,
    ) -> None:
        super().__init__()
        self.num_nodes = int(num_nodes)
        self.top_k = int(top_k)
        self.node_dim = int(node_dim)

        if mode is None:
            # Backward-compatible path used by the original configs.
            self.mode = "hybrid" if dynamic else "static"
            build_dynamic_parameters = bool(dynamic)
        else:
            normalized_mode = str(mode).lower().replace("-", "_")
            aliases = {
                "none": "identity",
                "no_graph": "identity",
                "no_cross_sensor": "identity",
            }
            self.mode = aliases.get(normalized_mode, normalized_mode)
            valid_modes = {"identity", "static", "dynamic", "hybrid"}
            if self.mode not in valid_modes:
                raise ValueError(
                    f"Unknown graph mode '{mode}'. Expected one of "
                    f"{sorted(valid_modes)}."
                )
            # Explicit ablation modes keep the same parameter inventory as the
            # hybrid model. Unused parameters are deliberate: they make the
            # identity/static/dynamic controls parameter-count matched.
            build_dynamic_parameters = True

        self.dynamic = self.mode in {"dynamic", "hybrid"}

        self.emb_src = nn.Parameter(torch.randn(num_nodes, node_dim) * 0.1)
        self.emb_dst = nn.Parameter(torch.randn(num_nodes, node_dim) * 0.1)

        if build_dynamic_parameters:
            self.q_proj = nn.Linear(summary_dim, node_dim)
            self.k_proj = nn.Linear(summary_dim, node_dim)

        self.alpha = nn.Parameter(torch.tensor(0.0))

    def _static_topk_softmax(self, logits: torch.Tensor) -> torch.Tensor:
        """Row-softmax with exact top-k support for the static graph."""
        if self.top_k <= 0 or self.top_k >= self.num_nodes:
            return torch.softmax(logits, dim=-1)

        indices = torch.topk(logits, self.top_k, dim=-1).indices
        mask = torch.zeros_like(logits, dtype=torch.bool)
        mask.scatter_(-1, indices, True)
        masked_logits = logits.masked_fill(~mask, -torch.inf)
        return torch.softmax(masked_logits, dim=-1)

    def forward(self, node_summary: torch.Tensor | None = None) -> torch.Tensor:
        if self.mode == "identity":
            batch_size = 1 if node_summary is None else node_summary.shape[0]
            identity = torch.eye(
                self.num_nodes,
                device=self.emb_src.device,
                dtype=self.emb_src.dtype,
            )
            return identity.unsqueeze(0).expand(batch_size, -1, -1)

        # Learnable, directed static graph. This is the branch controlled by
        # model.top_k in the original implementation.
        static_logits = F.relu(self.emb_src @ self.emb_dst.t())
        static = self._static_topk_softmax(static_logits)

        if self.mode == "static":
            return static.unsqueeze(0)

        if node_summary is None:
            raise ValueError(f"Graph mode '{self.mode}' requires node summaries")

        # The dynamic affinity is dense and is computed from the noised input
        # representation, so it is specific to the corruption instance.
        q = self.q_proj(node_summary)
        k = self.k_proj(node_summary)
        dynamic_logits = q @ k.transpose(1, 2) / (self.node_dim ** 0.5)
        dynamic = torch.softmax(dynamic_logits, dim=-1)

        if self.mode == "dynamic":
            return dynamic

        gate = torch.sigmoid(self.alpha)
        return gate * static.unsqueeze(0) + (1.0 - gate) * dynamic


class GraphConv(nn.Module):
    """Node mixing at every timestep. Input/output: ``[B, N, C, W]``."""

    def __init__(self, channels: int) -> None:
        super().__init__()
        self.lin_neigh = nn.Linear(channels, channels)
        self.lin_self = nn.Linear(channels, channels)

    def forward(self, x: torch.Tensor, adj: torch.Tensor) -> torch.Tensor:
        h = x.permute(0, 3, 1, 2)  # [B, W, N, C]
        neigh = torch.einsum("bij,bwjc->bwic", adj, h)
        out = self.lin_neigh(neigh) + self.lin_self(h)
        return out.permute(0, 2, 3, 1)
