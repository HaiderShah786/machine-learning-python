"""Elastic Weight Consolidation (EWC) for catastrophic-forgetting mitigation.

After finishing a domain we estimate the diagonal Fisher information of the
shared parameters and anchor them. While training the next domain a quadratic
penalty keeps important weights near their consolidated values.

Supports multiple anchored tasks (online sum of per-task penalties).
"""
from __future__ import annotations

from typing import Dict, List
import torch
import torch.nn as nn
import torch.nn.functional as F

from ..utils import move_batch


class EWC:
    def __init__(self, ewc_lambda: float = 50.0):
        self.ewc_lambda = ewc_lambda
        self.tasks: List[Dict[str, torch.Tensor]] = []   # stored {param_name: theta*}
        self.fishers: List[Dict[str, torch.Tensor]] = []  # stored {param_name: F}

    @staticmethod
    def _trainable_named_params(model: nn.Module):
        return {n: p for n, p in model.named_parameters() if p.requires_grad}

    @torch.enable_grad()
    def consolidate(self, model: nn.Module, loader, device: str,
                    domain_idx: int, max_batches: int = 30) -> None:
        """Estimate diagonal Fisher on ``loader`` and snapshot current weights."""
        model.eval()
        params = self._trainable_named_params(model)
        fisher = {n: torch.zeros_like(p) for n, p in params.items()}

        n_seen = 0
        for i, batch in enumerate(loader):
            if i >= max_batches:
                break
            batch = move_batch(batch, device)
            model.zero_grad()
            logits = model(batch, domain=domain_idx)
            # Sample-style Fisher: gradient of the log-likelihood of the
            # model's own predictions (Bernoulli per label).
            probs = torch.sigmoid(logits).detach()
            loss = F.binary_cross_entropy_with_logits(logits, probs)
            loss.backward()
            for n, p in params.items():
                if p.grad is not None:
                    fisher[n] += p.grad.detach() ** 2 * batch["image"].size(0)
            n_seen += batch["image"].size(0)

        for n in fisher:
            fisher[n] /= max(n_seen, 1)

        self.fishers.append(fisher)
        self.tasks.append({n: p.detach().clone() for n, p in params.items()})
        model.zero_grad()

    def penalty(self, model: nn.Module) -> torch.Tensor:
        """Quadratic EWC penalty summed over all consolidated tasks."""
        if not self.tasks:
            return torch.zeros((), device=next(model.parameters()).device)
        params = self._trainable_named_params(model)
        total = torch.zeros((), device=next(model.parameters()).device)
        for theta_star, fisher in zip(self.tasks, self.fishers):
            for n, p in params.items():
                if n in theta_star and n in fisher:
                    total = total + (fisher[n] * (p - theta_star[n]) ** 2).sum()
        return 0.5 * self.ewc_lambda * total
