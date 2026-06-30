"""Causal learning objectives: IRM + V-REx (causal regularisation).

Conceptual framing (proposal Section 5, "Causal Module: SCM, IRM, causal
regularization"):

    We treat each hospital/domain as an *environment* drawn from a Structural
    Causal Model in which the disease -> imaging-finding mechanism is invariant,
    while nuisance/shortcut variables (scanner, markers, demographics) vary.
    Invariant Risk Minimisation (IRM) seeks a representation whose optimal
    classifier is simultaneously optimal in every environment, i.e. it relies
    only on the stable causal mechanism. V-REx adds an explicit penalty on the
    variance of risk across environments (causal regularisation), discouraging
    the model from exploiting environment-specific shortcuts.
"""
from __future__ import annotations

from typing import Dict
import torch
import torch.nn.functional as F


def _bce(logits: torch.Tensor, targets: torch.Tensor) -> torch.Tensor:
    return F.binary_cross_entropy_with_logits(logits, targets)


def irm_penalty(logits: torch.Tensor, targets: torch.Tensor) -> torch.Tensor:
    """IRMv1 gradient penalty for one environment.

    Multiplies the logits by a constant dummy classifier ``w = 1`` and returns
    the squared gradient of the risk w.r.t. ``w``. A small penalty means the
    representation's classifier is already locally optimal in this environment.
    """
    scale = torch.ones(1, device=logits.device, requires_grad=True)
    loss = _bce(logits * scale, targets)
    grad = torch.autograd.grad(loss, [scale], create_graph=True)[0]
    return (grad ** 2).sum()


def environment_losses(logits: torch.Tensor, targets: torch.Tensor,
                       domains: torch.Tensor) -> Dict[int, torch.Tensor]:
    """Split a mixed-domain batch into per-environment BCE losses."""
    losses: Dict[int, torch.Tensor] = {}
    for d in domains.unique():
        m = domains == d
        if m.sum() > 0:
            losses[int(d.item())] = _bce(logits[m], targets[m])
    return losses


def causal_objective(logits: torch.Tensor, targets: torch.Tensor,
                     domains: torch.Tensor, irm_lambda: float = 1.0,
                     vrex_lambda: float = 0.1):
    """Return (irm_penalty, vrex_penalty) summed/averaged over environments.

    When a batch contains a single environment (common in the continual setting
    where each step trains on one domain), the replay buffer supplies exemplars
    from past domains so multiple environments co-occur and these penalties
    become meaningful. With one environment, V-REx is 0 and IRM still pushes
    toward a locally-invariant classifier.
    """
    env_logits, env_targets = {}, {}
    for d in domains.unique():
        m = domains == d
        env_logits[int(d)] = logits[m]
        env_targets[int(d)] = targets[m]

    irm_terms, risks = [], []
    for d in env_logits:
        if env_logits[d].shape[0] == 0:
            continue
        irm_terms.append(irm_penalty(env_logits[d], env_targets[d]))
        risks.append(_bce(env_logits[d], env_targets[d]))

    irm_val = torch.stack(irm_terms).mean() if irm_terms else logits.sum() * 0.0
    if len(risks) > 1:
        vrex_val = torch.stack(risks).var(unbiased=False)
    else:
        vrex_val = logits.sum() * 0.0
    return irm_lambda * irm_val, vrex_lambda * vrex_val
