"""CDIME-AI: Causal Domain-Incremental Multimodal Explainable AI for CXR.

A reference implementation of the CDIME-AI framework: a multimodal
(image + radiology report) classifier that learns causal, domain-invariant
representations, adapts continually across hospitals without catastrophic
forgetting, explains its predictions, quantifies uncertainty and produces
evidence-based therapeutic decision support.

Quick start
-----------
>>> from cdime_ai.config import Config
>>> from cdime_ai.tokenizer import get_tokenizer
>>> from cdime_ai.data.loaders import build_domain_loaders
>>> from cdime_ai.engine.continual_runner import run_continual
>>> cfg = Config.fast_debug()
>>> tok = get_tokenizer(cfg)
>>> loaders = build_domain_loaders(cfg, tok)
>>> results = run_continual(cfg, loaders, tok)
"""
from .config import Config, LABELS

__version__ = "1.0.0"
__all__ = ["Config", "LABELS"]
