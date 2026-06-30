"""Cross-attention transformer fusion of image patch tokens and report tokens.

Image tokens attend to text tokens and vice-versa over several layers; the
fused [CLS]-style summary tokens are concatenated into the joint representation.
Attention weights from the last layer are cached for the attention-map
explainability module.
"""
from __future__ import annotations

import torch
import torch.nn as nn


class CrossAttentionBlock(nn.Module):
    def __init__(self, dim: int, heads: int, dropout: float):
        super().__init__()
        self.attn = nn.MultiheadAttention(dim, heads, dropout=dropout, batch_first=True)
        self.norm_q = nn.LayerNorm(dim)
        self.norm_kv = nn.LayerNorm(dim)
        self.ffn = nn.Sequential(
            nn.Linear(dim, dim * 2), nn.GELU(), nn.Dropout(dropout), nn.Linear(dim * 2, dim)
        )
        self.norm_ffn = nn.LayerNorm(dim)
        self.last_attn: torch.Tensor | None = None

    def forward(self, q, kv, kv_mask=None):
        qn, kvn = self.norm_q(q), self.norm_kv(kv)
        key_padding = (kv_mask == 0) if kv_mask is not None else None
        out, attn = self.attn(qn, kvn, kvn, key_padding_mask=key_padding,
                              need_weights=True, average_attn_weights=True)
        self.last_attn = attn.detach()                 # (B, Lq, Lkv)
        q = q + out
        q = q + self.ffn(self.norm_ffn(q))
        return q


class CrossAttentionFusion(nn.Module):
    def __init__(self, dim: int = 256, heads: int = 4, layers: int = 2, dropout: float = 0.3):
        super().__init__()
        self.img2txt = nn.ModuleList(
            [CrossAttentionBlock(dim, heads, dropout) for _ in range(layers)])
        self.txt2img = nn.ModuleList(
            [CrossAttentionBlock(dim, heads, dropout) for _ in range(layers)])
        self.out_dim = dim * 2

    def forward(self, img_tokens, txt_tokens, txt_mask=None):
        i, t = img_tokens, txt_tokens
        for blk_it, blk_ti in zip(self.img2txt, self.txt2img):
            new_i = blk_it(i, t, txt_mask)             # image attends to text
            new_t = blk_ti(t, i, None)                 # text attends to image
            i, t = new_i, new_t
        img_summary = i.mean(dim=1)                     # (B, dim)
        txt_summary = t.mean(dim=1)                     # (B, dim)
        fused = torch.cat([img_summary, txt_summary], dim=-1)   # (B, 2*dim)
        return fused

    @property
    def last_cross_attention(self) -> torch.Tensor | None:
        """Image->text attention from the final fusion layer (B, N_img, L_txt)."""
        return self.img2txt[-1].last_attn
