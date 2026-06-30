"""Monte-Carlo Dropout for epistemic uncertainty estimation."""
from __future__ import annotations

import torch


@torch.no_grad()
def mc_dropout_predict(model, batch, domain: int, n_samples: int = 20):
    """Run ``n_samples`` stochastic forward passes with dropout enabled.

    Returns
    -------
    mean_prob : (B, C) predictive mean probability
    std_prob  : (B, C) predictive std (epistemic uncertainty per label)
    entropy   : (B,)   mean predictive entropy across labels
    """
    model.enable_mc_dropout()
    probs = []
    for _ in range(n_samples):
        logits = model(batch, domain=domain)
        probs.append(torch.sigmoid(logits))
    probs = torch.stack(probs, dim=0)          # (S, B, C)
    mean_prob = probs.mean(0)
    std_prob = probs.std(0)
    eps = 1e-8
    ent = -(mean_prob * (mean_prob + eps).log()
            + (1 - mean_prob) * (1 - mean_prob + eps).log())
    return mean_prob, std_prob, ent.mean(dim=1)
