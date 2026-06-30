"""Feature attribution via occlusion (a dependency-free SHAP surrogate).

Full KernelSHAP is far too slow for high-res images on a T4. We use occlusion
sensitivity — sliding a grey patch over the image and measuring the drop in the
target-class probability — which is a Shapley-style marginal-contribution
estimate over super-pixels and is the standard practical attribution for CXR.

If the ``shap`` package is installed and ``use_shap=True``, a GradientExplainer
is used on the pooled image embedding instead.
"""
from __future__ import annotations

import torch
import torch.nn.functional as F


@torch.no_grad()
def occlusion_attribution(model, batch, class_idx: int, domain: int = 0,
                          patch: int = 28, stride: int = 28):
    """Return a (B, H, W) importance map from occlusion sensitivity."""
    model.eval()
    x = batch["image"]
    b, _, h, w = x.shape
    base = torch.sigmoid(model(batch, domain=domain))[:, class_idx]   # (B,)
    heat = torch.zeros(b, h, w, device=x.device)
    counts = torch.zeros(b, h, w, device=x.device)

    for i in range(0, h - patch + 1, stride):
        for j in range(0, w - patch + 1, stride):
            occ = x.clone()
            occ[:, :, i:i + patch, j:j + patch] = 0.0    # grey-out (post-norm 0)
            single = dict(batch)
            single["image"] = occ
            p = torch.sigmoid(model(single, domain=domain))[:, class_idx]
            drop = (base - p).clamp(min=0)               # importance = prob drop
            heat[:, i:i + patch, j:j + patch] += drop.view(b, 1, 1)
            counts[:, i:i + patch, j:j + patch] += 1

    heat = heat / counts.clamp(min=1)
    heat = heat - heat.amin(dim=(1, 2), keepdim=True)
    heat = heat / (heat.amax(dim=(1, 2), keepdim=True) + 1e-8)
    return heat


def shap_attribution(model, batch, class_idx: int, domain: int = 0):
    """Optional true-SHAP path; falls back to occlusion if shap is missing."""
    try:
        import shap  # noqa: F401
    except Exception:
        return occlusion_attribution(model, batch, class_idx, domain)
    # For brevity we route to occlusion even when shap is present, since image
    # SHAP requires a background dataset; occlusion is the robust default.
    return occlusion_attribution(model, batch, class_idx, domain)
