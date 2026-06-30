"""Build per-domain train/val/test DataLoaders for the continual sequence."""
from __future__ import annotations

from typing import Dict
from torch.utils.data import DataLoader

from ..config import Config
from .synthetic import SyntheticCXRDataset
from .datasets import RealCXRDataset, REAL_PATHS


def _make_loader(ds, cfg: Config, shuffle: bool) -> DataLoader:
    return DataLoader(
        ds,
        batch_size=cfg.train.batch_size,
        shuffle=shuffle,
        num_workers=cfg.data.num_workers,
        pin_memory=cfg.data.pin_memory,
        drop_last=shuffle,
    )


def build_domain_loaders(cfg: Config, tokenizer) -> Dict[str, Dict[str, DataLoader]]:
    """Return ``{domain_name: {"train"/"val"/"test": DataLoader}}``."""
    out: Dict[str, Dict[str, DataLoader]] = {}
    for di, domain in enumerate(cfg.domains):
        loaders = {}
        if cfg.data.mode == "synthetic":
            sizes = {
                "train": cfg.data.train_per_domain,
                "val": cfg.data.val_per_domain,
                "test": cfg.data.test_per_domain,
            }
            for split, n in sizes.items():
                ds = SyntheticCXRDataset(cfg, tokenizer, n=n, domain_idx=di,
                                         split=split, seed=cfg.seed)
                loaders[split] = _make_loader(ds, cfg, shuffle=(split == "train"))
        else:
            csv_path, img_root = REAL_PATHS[domain]
            limits = {
                "train": cfg.data.train_per_domain,
                "val": cfg.data.val_per_domain,
                "test": cfg.data.test_per_domain,
            }
            for split, lim in limits.items():
                ds = RealCXRDataset(cfg, tokenizer, csv_path, img_root,
                                    domain_idx=di, split=split, limit=lim)
                loaders[split] = _make_loader(ds, cfg, shuffle=(split == "train"))
        out[domain] = loaders
    return out
