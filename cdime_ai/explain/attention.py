"""Cross-attention map extraction from the fusion module.

Surfaces which report tokens the image attends to (and the spatial attention
over image patches), giving a multimodal rationale for each prediction.
"""
from __future__ import annotations

import torch


@torch.no_grad()
def image_to_text_attention(model, batch, domain: int = 0):
    """Return mean image->text attention (B, L_txt) over report tokens.

    Higher weight => that report token contributed more to the fused image
    representation. Returns None for image-only / concat-fusion ablations.
    """
    model.eval()
    _ = model(batch, domain=domain)
    fusion = getattr(model, "fusion", None)
    if fusion is None or getattr(fusion, "last_cross_attention", None) is None:
        return None
    attn = fusion.last_cross_attention          # (B, N_img, L_txt)
    return attn.mean(dim=1)                      # average over image queries


def top_text_evidence(attn_row: torch.Tensor, input_ids: torch.Tensor,
                      tokenizer, k: int = 5):
    """Return the top-k report tokens by attention weight (decoded strings)."""
    if attn_row is None:
        return []
    vals, idx = attn_row.topk(min(k, attn_row.numel()))
    toks = tokenizer.convert_ids_to_tokens(input_ids[idx].tolist())
    return [(t, float(v)) for t, v in zip(toks, vals)]
