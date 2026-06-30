"""Shared utilities: seeding, logging, device helpers, checkpointing."""
from __future__ import annotations

import os
import random
import logging
from typing import Optional

import numpy as np
import torch


def set_seed(seed: int = 42) -> None:
    """Seed every RNG for reproducible (seed-averaged) experiments."""
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    # Deterministic cuDNN is slower; keep benchmark on for T4 speed but seed all.
    torch.backends.cudnn.benchmark = True


def get_logger(name: str = "cdime") -> logging.Logger:
    logger = logging.getLogger(name)
    if not logger.handlers:
        handler = logging.StreamHandler()
        handler.setFormatter(logging.Formatter("[%(asctime)s] %(message)s", "%H:%M:%S"))
        logger.addHandler(handler)
        logger.setLevel(logging.INFO)
    return logger


def count_parameters(model: torch.nn.Module) -> int:
    return sum(p.numel() for p in model.parameters() if p.requires_grad)


def move_batch(batch: dict, device: str) -> dict:
    """Move a dict batch of tensors to ``device`` (non-tensor values untouched)."""
    out = {}
    for k, v in batch.items():
        out[k] = v.to(device, non_blocking=True) if torch.is_tensor(v) else v
    return out


def save_checkpoint(model: torch.nn.Module, path: str) -> None:
    os.makedirs(os.path.dirname(path) or ".", exist_ok=True)
    torch.save(model.state_dict(), path)


def load_checkpoint(model: torch.nn.Module, path: str, device: str = "cpu") -> torch.nn.Module:
    state = torch.load(path, map_location=device)
    model.load_state_dict(state, strict=False)
    return model
