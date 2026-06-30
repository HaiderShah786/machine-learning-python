"""Replay memory (experience rehearsal) for domain-incremental learning.

Stores a bounded set of exemplars (full multimodal samples) from each past
domain using reservoir sampling. During training on a new domain, a fraction of
each mini-batch is drawn from replay so that (a) past domains are rehearsed and
(b) batches contain *multiple environments*, which is what makes the IRM/V-REx
causal penalties effective.
"""
from __future__ import annotations

from typing import Dict, List, Optional
import random
import torch


class ReplayMemory:
    def __init__(self, capacity_per_domain: int = 200):
        self.capacity = capacity_per_domain
        self.buffer: Dict[int, List[dict]] = {}
        self._count: Dict[int, int] = {}

    def add_loader(self, loader, domain_idx: int) -> None:
        """Reservoir-sample exemplars from a finished domain's train loader."""
        self.buffer.setdefault(domain_idx, [])
        self._count.setdefault(domain_idx, 0)
        buf = self.buffer[domain_idx]
        for batch in loader:
            b = batch["image"].size(0)
            for i in range(b):
                sample = {k: (v[i].detach().cpu() if torch.is_tensor(v) else v)
                          for k, v in batch.items()}
                self._count[domain_idx] += 1
                if len(buf) < self.capacity:
                    buf.append(sample)
                else:
                    j = random.randint(0, self._count[domain_idx] - 1)
                    if j < self.capacity:
                        buf[j] = sample

    def is_empty(self) -> bool:
        return sum(len(v) for v in self.buffer.values()) == 0

    def sample(self, n: int) -> Optional[dict]:
        """Return a collated batch of ``n`` exemplars drawn across past domains."""
        pool = [s for samples in self.buffer.values() for s in samples]
        if not pool or n <= 0:
            return None
        chosen = random.choices(pool, k=min(n, len(pool)))
        keys = chosen[0].keys()
        out = {}
        for k in keys:
            if torch.is_tensor(chosen[0][k]):
                out[k] = torch.stack([c[k] for c in chosen])
            else:
                out[k] = [c[k] for c in chosen]
        return out
