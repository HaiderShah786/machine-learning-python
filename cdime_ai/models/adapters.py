"""Per-domain bottleneck adapters for parameter-efficient continual learning.

Each domain gets its own lightweight residual adapter (Houlsby-style). Only the
active domain's adapter is trained while the shared backbone is regularised by
EWC, giving plasticity for the new domain without overwriting shared features.
"""
from __future__ import annotations

import torch
import torch.nn as nn


class Adapter(nn.Module):
    def __init__(self, dim: int, bottleneck: int = 64, dropout: float = 0.1):
        super().__init__()
        self.down = nn.Linear(dim, bottleneck)
        self.act = nn.GELU()
        self.up = nn.Linear(bottleneck, dim)
        self.drop = nn.Dropout(dropout)
        self.norm = nn.LayerNorm(dim)
        nn.init.zeros_(self.up.weight)   # start as identity (residual)
        nn.init.zeros_(self.up.bias)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        h = self.up(self.drop(self.act(self.down(x))))
        return self.norm(x + h)


class DomainAdapterBank(nn.Module):
    """A bank of adapters, one per domain, selected by ``active_domain``."""

    def __init__(self, dim: int, n_domains: int, bottleneck: int = 64, dropout: float = 0.1):
        super().__init__()
        self.adapters = nn.ModuleList(
            [Adapter(dim, bottleneck, dropout) for _ in range(n_domains)]
        )
        self.active_domain = 0

    def set_domain(self, idx: int) -> None:
        self.active_domain = idx

    def forward(self, x: torch.Tensor, domain: int | None = None) -> torch.Tensor:
        idx = self.active_domain if domain is None else domain
        return self.adapters[idx](x)
