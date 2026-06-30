"""Counterfactual explanations.

Answers "what minimal change to the image would flip this diagnosis?" by
gradient ascent/descent on the input toward the decision boundary for a target
class, under an L2 proximity constraint. The resulting difference map highlights
the causally-relevant region (it should overlap the lesion, not the shortcut).
"""
from __future__ import annotations

import torch


def generate_counterfactual(model, batch, class_idx: int, domain: int = 0,
                            target_prob: float = 0.5, steps: int = 30,
                            lr: float = 0.02, proximity: float = 1.0):
    """Return (cf_image, delta, orig_prob, cf_prob) for the first sample.

    Flips the prediction for ``class_idx`` toward ``target_prob`` with a small,
    proximity-regularised perturbation.
    """
    model.eval()
    x = batch["image"][:1].clone().detach()
    single = {k: (v[:1].clone() if torch.is_tensor(v) else v) for k, v in batch.items()}

    with torch.no_grad():
        orig_prob = torch.sigmoid(model(single, domain=domain))[0, class_idx].item()
    # Push toward the opposite class.
    target = 0.0 if orig_prob >= 0.5 else 1.0

    delta = torch.zeros_like(x, requires_grad=True)
    opt = torch.optim.Adam([delta], lr=lr)
    target_t = torch.tensor([[target]], device=x.device)

    for _ in range(steps):
        opt.zero_grad()
        single["image"] = x + delta
        logit = model(single, domain=domain)[:, class_idx:class_idx + 1]
        bce = torch.nn.functional.binary_cross_entropy_with_logits(logit, target_t)
        loss = bce + proximity * (delta ** 2).mean()
        loss.backward()
        opt.step()

    with torch.no_grad():
        single["image"] = x + delta
        cf_prob = torch.sigmoid(model(single, domain=domain))[0, class_idx].item()

    return (x + delta).detach(), delta.detach(), orig_prob, cf_prob
