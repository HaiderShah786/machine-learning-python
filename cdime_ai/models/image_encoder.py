"""Swin Transformer image encoder (timm) exposing spatial tokens for Grad-CAM.

Returns both a sequence of patch tokens (for cross-attention fusion and
attention-rollout explainability) and the final spatial feature map (for
Grad-CAM). Falls back to a small ResNet if timm/Swin is unavailable so the
pipeline still runs.
"""
from __future__ import annotations

import torch
import torch.nn as nn


class SwinImageEncoder(nn.Module):
    def __init__(self, backbone: str = "swin_tiny_patch4_window7_224",
                 pretrained: bool = True, out_dim: int = 256):
        super().__init__()
        self.out_dim = out_dim
        self._kind = None
        try:
            import timm
            # features_only gives us the last spatial map (H', W', C) for Grad-CAM.
            self.backbone = timm.create_model(
                backbone, pretrained=pretrained, features_only=True, out_indices=(3,)
            )
            self.feat_dim = self.backbone.feature_info.channels()[-1]
            self._kind = "swin"
        except Exception as e:  # pragma: no cover - offline / no timm
            print(f"[image_encoder] timm Swin unavailable ({e}); using ResNet fallback.")
            from torchvision.models import resnet18, ResNet18_Weights
            w = ResNet18_Weights.DEFAULT if pretrained else None
            net = resnet18(weights=w)
            self.backbone = nn.Sequential(*list(net.children())[:-2])  # -> (B,512,h,w)
            self.feat_dim = 512
            self._kind = "resnet"

        self.proj = nn.Linear(self.feat_dim, out_dim)
        # Cache the last spatial feature map for Grad-CAM hooks.
        self.last_feature_map: torch.Tensor | None = None

    def forward(self, x: torch.Tensor):
        """Return (tokens, pooled).

        tokens : (B, N, out_dim) patch-token sequence for fusion / attention maps
        pooled : (B, out_dim)    mean-pooled global image embedding
        """
        if self._kind == "swin":
            feat = self.backbone(x)[-1]            # timm Swin: (B, H, W, C)
            if feat.dim() == 4 and feat.shape[-1] == self.feat_dim:
                # channels-last (H,W,C) -> (B,C,H,W)
                feat = feat.permute(0, 3, 1, 2).contiguous()
        else:
            feat = self.backbone(x)                # (B, C, H, W)

        feat.retain_grad() if feat.requires_grad else None
        self.last_feature_map = feat               # (B, C, H, W) for Grad-CAM

        b, c, h, w = feat.shape
        tokens = feat.flatten(2).transpose(1, 2)   # (B, N=h*w, C)
        tokens = self.proj(tokens)                 # (B, N, out_dim)
        pooled = tokens.mean(dim=1)                # (B, out_dim)
        return tokens, pooled
