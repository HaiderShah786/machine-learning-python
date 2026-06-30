"""Tokenizer loader (ClinicalBERT with offline-safe fallback)."""
from __future__ import annotations

from .config import Config


def get_tokenizer(cfg: Config):
    from transformers import AutoTokenizer
    try:
        return AutoTokenizer.from_pretrained(cfg.model.text_backbone)
    except Exception as e:  # pragma: no cover - offline
        print(f"[tokenizer] {cfg.model.text_backbone} unavailable ({e}); "
              f"using {cfg.model.text_backbone_fallback}.")
        return AutoTokenizer.from_pretrained(cfg.model.text_backbone_fallback)
