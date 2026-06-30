"""Temperature scaling for post-hoc calibration (Guo et al., 2017).

Learns a single scalar T on a held-out validation set that divides the logits
before the sigmoid, minimising calibration error without changing accuracy.
"""
from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F

from ..utils import move_batch


class TemperatureScaler(nn.Module):
    def __init__(self):
        super().__init__()
        self.log_temp = nn.Parameter(torch.zeros(1))   # T = exp(log_temp), init 1

    @property
    def temperature(self) -> float:
        return float(self.log_temp.exp().item())

    def forward(self, logits: torch.Tensor) -> torch.Tensor:
        return logits / self.log_temp.exp()

    @torch.enable_grad()
    def fit(self, model, loader, domain: int, device: str, max_iter: int = 100):
        """Optimise T on ``loader`` via LBFGS over BCE of scaled logits."""
        model.eval()
        all_logits, all_targets = [], []
        with torch.no_grad():
            for batch in loader:
                batch = move_batch(batch, device)
                all_logits.append(model(batch, domain=domain).detach())
                all_targets.append(batch["label"])
        logits = torch.cat(all_logits)
        targets = torch.cat(all_targets)

        self.to(device)
        opt = torch.optim.LBFGS([self.log_temp], lr=0.05, max_iter=max_iter)

        def closure():
            opt.zero_grad()
            loss = F.binary_cross_entropy_with_logits(self(logits), targets)
            loss.backward()
            return loss

        opt.step(closure)
        return self.temperature
