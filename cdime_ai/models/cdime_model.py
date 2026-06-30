"""CDIME-AI: the full multimodal causal continual model.

Combines the Swin image encoder, ClinicalBERT text encoder, cross-attention
fusion, per-domain adapters and a multi-label classifier head. Ablation
switches let a single class realise every configuration in proposal Section 7.
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Optional

import torch
import torch.nn as nn

from ..config import Config
from .image_encoder import SwinImageEncoder
from .text_encoder import ClinicalBERTEncoder
from .fusion import CrossAttentionFusion
from .adapters import DomainAdapterBank


@dataclass
class AblationFlags:
    use_text: bool = True          # A5: False -> image-only
    use_cross_attention: bool = True  # A6: False -> simple concatenation
    use_adapters: bool = True      # A2/A7
    enable_mc_dropout: bool = True # A4 (uncertainty handled in trainer)


class ConcatFusion(nn.Module):
    """Simple concatenation baseline for ablation A6."""

    def __init__(self, dim: int):
        super().__init__()
        self.out_dim = dim * 2
        self.last_cross_attention = None

    def forward(self, img_tokens, txt_tokens, txt_mask=None):
        return torch.cat([img_tokens.mean(1), txt_tokens.mean(1)], dim=-1)


class CDIMEModel(nn.Module):
    def __init__(self, cfg: Config, ablation: Optional[AblationFlags] = None):
        super().__init__()
        self.cfg = cfg
        self.ab = ablation or AblationFlags()
        d = cfg.model.fusion_dim
        n_domains = len(cfg.domains)

        self.image_encoder = SwinImageEncoder(
            cfg.model.image_backbone, cfg.model.pretrained_image, out_dim=d)

        if self.ab.use_text:
            self.text_encoder = ClinicalBERTEncoder(
                cfg.model.text_backbone, cfg.model.text_backbone_fallback,
                out_dim=d, freeze_layers=cfg.model.freeze_text_layers)
            if self.ab.use_cross_attention:
                self.fusion = CrossAttentionFusion(
                    d, cfg.model.fusion_heads, cfg.model.fusion_layers, cfg.model.dropout)
            else:
                self.fusion = ConcatFusion(d)
            fused_dim = self.fusion.out_dim
        else:
            self.text_encoder = None
            self.fusion = None
            fused_dim = d

        self.use_adapters = self.ab.use_adapters
        if self.use_adapters:
            self.adapters = DomainAdapterBank(
                fused_dim, n_domains, cfg.model.adapter_dim, cfg.model.dropout)

        self.dropout = nn.Dropout(cfg.model.dropout)   # MC-Dropout layer
        self.classifier = nn.Sequential(
            nn.Linear(fused_dim, fused_dim // 2),
            nn.GELU(),
            nn.Dropout(cfg.model.dropout),
            nn.Linear(fused_dim // 2, cfg.num_classes),
        )

    def set_domain(self, idx: int) -> None:
        if self.use_adapters:
            self.adapters.set_domain(idx)

    def enable_mc_dropout(self) -> None:
        """Put ONLY dropout layers in train mode for MC-Dropout sampling."""
        self.eval()
        for m in self.modules():
            if isinstance(m, nn.Dropout):
                m.train()

    def encode(self, batch, domain: Optional[int] = None) -> torch.Tensor:
        img_tokens, img_pooled = self.image_encoder(batch["image"])
        if self.ab.use_text and self.text_encoder is not None:
            txt_tokens, _ = self.text_encoder(batch["input_ids"], batch["attention_mask"])
            fused = self.fusion(img_tokens, txt_tokens, batch["attention_mask"])
        else:
            fused = img_pooled
        if self.use_adapters:
            fused = self.adapters(fused, domain)
        return fused

    def forward(self, batch, domain: Optional[int] = None,
                return_features: bool = False):
        features = self.encode(batch, domain)
        logits = self.classifier(self.dropout(features))
        if return_features:
            return logits, features
        return logits
