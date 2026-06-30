"""Central configuration for the CDIME-AI framework.

All defaults are tuned for a free Google Colab T4 GPU (~16 GB VRAM) with low
host RAM. Increase the values marked ``# scale-up`` when running on bigger
hardware to reproduce full-scale Q1 results.
"""
from __future__ import annotations

from dataclasses import dataclass, field, asdict
from typing import List, Optional
import json
import torch


# The five canonical CheXpert competition pathologies. These overlap across
# CheXpert, MIMIC-CXR and VinDr-CXR which makes them ideal for the
# domain-incremental protocol (same label space, shifting input distribution).
LABELS: List[str] = [
    "Atelectasis",
    "Cardiomegaly",
    "Consolidation",
    "Edema",
    "Pleural Effusion",
]


@dataclass
class DataConfig:
    """Data-loading configuration."""

    # "synthetic" runs the full pipeline end-to-end with no downloads (default,
    # T4-friendly). "real" uses the on-disk CheXpert / MIMIC-CXR / VinDr-CXR
    # CSVs configured in data/datasets.py.
    mode: str = "synthetic"

    image_size: int = 224
    max_text_len: int = 96            # short radiology impressions; keeps BERT cheap

    # Per-domain sample counts. Tiny by default so a full 3-domain continual run
    # finishes in minutes on a T4. # scale-up
    train_per_domain: int = 600
    val_per_domain: int = 150
    test_per_domain: int = 200

    num_workers: int = 2              # low for limited Colab RAM
    pin_memory: bool = True

    # Strength of the domain-dependent spurious shortcut injected by the
    # synthetic generator (0 = none, 1 = perfectly predictive shortcut).
    # This is what lets us *demonstrate* the causal module's benefit.
    shortcut_strength: float = 0.9


@dataclass
class ModelConfig:
    image_backbone: str = "swin_tiny_patch4_window7_224"
    text_backbone: str = "emilyalsentzer/Bio_ClinicalBERT"
    # Falls back automatically to this if the ClinicalBERT download fails
    # (e.g. offline) so the pipeline never hard-crashes.
    text_backbone_fallback: str = "distilbert-base-uncased"

    fusion_dim: int = 256
    fusion_heads: int = 4
    fusion_layers: int = 2
    dropout: float = 0.3              # also used as MC-Dropout rate at inference
    adapter_dim: int = 64            # bottleneck width of per-domain adapters

    pretrained_image: bool = True
    freeze_text_layers: int = 8       # freeze lower BERT layers to save memory


@dataclass
class TrainConfig:
    epochs_per_domain: int = 4        # # scale-up (8-15 for full results)
    batch_size: int = 8              # T4-safe with Swin-Tiny + ClinicalBERT
    grad_accum_steps: int = 2         # effective batch = 16
    lr: float = 2e-4
    text_lr: float = 1e-5             # smaller LR for the pretrained text encoder
    weight_decay: float = 1e-4
    amp: bool = True                  # mixed precision -> ~2x less VRAM
    max_grad_norm: float = 1.0

    # Continual-learning hyper-parameters
    use_ewc: bool = True
    ewc_lambda: float = 50.0
    use_replay: bool = True
    replay_size: int = 200            # exemplars retained per past domain
    replay_ratio: float = 0.5         # fraction of a batch drawn from replay
    use_adapters: bool = True

    # Causal hyper-parameters
    use_irm: bool = True
    irm_lambda: float = 1.0
    irm_anneal_epochs: int = 1        # warm-up before applying full IRM penalty
    use_causal_reg: bool = True
    causal_reg_lambda: float = 0.1


@dataclass
class Config:
    seed: int = 42
    domains: List[str] = field(default_factory=lambda: ["CheXpert", "MIMIC-CXR", "VinDr-CXR"])
    labels: List[str] = field(default_factory=lambda: list(LABELS))
    output_dir: str = "results"
    device: str = "cuda" if torch.cuda.is_available() else "cpu"

    data: DataConfig = field(default_factory=DataConfig)
    model: ModelConfig = field(default_factory=ModelConfig)
    train: TrainConfig = field(default_factory=TrainConfig)

    @property
    def num_classes(self) -> int:
        return len(self.labels)

    def to_dict(self) -> dict:
        return asdict(self)

    def save(self, path: str) -> None:
        with open(path, "w") as f:
            json.dump(self.to_dict(), f, indent=2)

    @classmethod
    def fast_debug(cls) -> "Config":
        """Tiny config for smoke-testing the whole pipeline in <1 min."""
        c = cls()
        c.data.train_per_domain = 40
        c.data.val_per_domain = 16
        c.data.test_per_domain = 24
        c.train.epochs_per_domain = 1
        c.train.batch_size = 4
        c.train.grad_accum_steps = 1
        c.train.replay_size = 16
        return c
