"""Grad-CAM saliency for the image encoder.

Hooks the cached last spatial feature map of the Swin/ResNet backbone and
produces a per-class heat-map indicating where the model looks. Useful both for
clinician trust and for the insertion/deletion faithfulness metric.
"""
from __future__ import annotations

import torch
import torch.nn.functional as F


class GradCAM:
    def __init__(self, model):
        self.model = model
        self._fmap = None
        self._grad = None
        enc = model.image_encoder

        # Forward hook captures the feature map; backward hook captures grads.
        def fwd_hook(_m, _inp, _out):
            self._fmap = enc.last_feature_map
            if self._fmap is not None and self._fmap.requires_grad:
                self._fmap.register_hook(self._save_grad)

        enc.register_forward_hook(fwd_hook)

    def _save_grad(self, grad):
        self._grad = grad

    def __call__(self, batch, class_idx: int, domain: int = 0):
        """Return a (B, H, W) normalised Grad-CAM heat-map for ``class_idx``."""
        self.model.eval()
        self.model.zero_grad()
        logits = self.model(batch, domain=domain)
        score = logits[:, class_idx].sum()
        score.backward()

        fmap, grad = self._fmap, self._grad          # (B, C, H, W)
        weights = grad.mean(dim=(2, 3), keepdim=True)  # GAP over spatial dims
        cam = F.relu((weights * fmap).sum(dim=1))      # (B, H, W)
        # Normalise each map to [0, 1].
        cam = cam - cam.amin(dim=(1, 2), keepdim=True)
        cam = cam / (cam.amax(dim=(1, 2), keepdim=True) + 1e-8)
        img_size = batch["image"].shape[-1]
        cam = F.interpolate(cam.unsqueeze(1), size=(img_size, img_size),
                            mode="bilinear", align_corners=False).squeeze(1)
        return cam.detach()
