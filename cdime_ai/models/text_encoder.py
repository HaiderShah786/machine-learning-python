"""ClinicalBERT text encoder for radiology reports (with DistilBERT fallback)."""
from __future__ import annotations

import torch
import torch.nn as nn


class ClinicalBERTEncoder(nn.Module):
    def __init__(self, backbone: str = "emilyalsentzer/Bio_ClinicalBERT",
                 fallback: str = "distilbert-base-uncased",
                 out_dim: int = 256, freeze_layers: int = 8):
        super().__init__()
        from transformers import AutoModel
        try:
            self.bert = AutoModel.from_pretrained(backbone)
            self.name = backbone
        except Exception as e:  # pragma: no cover - offline
            print(f"[text_encoder] {backbone} unavailable ({e}); falling back to {fallback}.")
            self.bert = AutoModel.from_pretrained(fallback)
            self.name = fallback

        hidden = self.bert.config.hidden_size
        self.proj = nn.Linear(hidden, out_dim)
        self.out_dim = out_dim
        self._freeze_lower(freeze_layers)

    def _freeze_lower(self, n: int) -> None:
        """Freeze embeddings + the lowest ``n`` transformer layers to save memory."""
        if n <= 0:
            return
        for p in self.bert.embeddings.parameters():
            p.requires_grad = False
        layers = None
        if hasattr(self.bert, "encoder") and hasattr(self.bert.encoder, "layer"):
            layers = self.bert.encoder.layer            # BERT-style
        elif hasattr(self.bert, "transformer") and hasattr(self.bert.transformer, "layer"):
            layers = self.bert.transformer.layer        # DistilBERT-style
        if layers is not None:
            for layer in layers[:n]:
                for p in layer.parameters():
                    p.requires_grad = False

    def forward(self, input_ids: torch.Tensor, attention_mask: torch.Tensor):
        """Return (tokens, pooled).

        tokens : (B, L, out_dim) projected token embeddings for cross-attention
        pooled : (B, out_dim)    CLS-token (or mean) embedding
        """
        out = self.bert(input_ids=input_ids, attention_mask=attention_mask)
        seq = out.last_hidden_state                    # (B, L, hidden)
        tokens = self.proj(seq)                         # (B, L, out_dim)
        # Masked mean pooling as the global text embedding.
        mask = attention_mask.unsqueeze(-1).float()
        pooled = (tokens * mask).sum(1) / mask.sum(1).clamp(min=1e-6)
        return tokens, pooled
