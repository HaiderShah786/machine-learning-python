"""Real-dataset adapters for CheXpert, MIMIC-CXR and VinDr-CXR.

These are used when ``cfg.data.mode == "real"``. They expect a small CSV per
domain (path + label columns) so you can point them at *subsets* that fit on
Colab disk. If a CSV/image is missing the loader raises a clear error telling
you what to download.

Expected CSV columns
--------------------
CheXpert / MIMIC-CXR: standard CheXpert label columns (-1 uncertain handled via
the U-Ones policy -> 1). MIMIC additionally needs a ``report`` text column.
VinDr-CXR: a ``findings`` multi-label set; we map to the 5 shared pathologies.

For full-scale runs replace these with the official loaders; the rest of the
framework is dataset-agnostic (it only consumes the batch dict produced here).
"""
from __future__ import annotations

import os
from typing import Optional

import numpy as np
import pandas as pd
import torch
from PIL import Image
from torch.utils.data import Dataset

from ..config import Config


def _u_ones(v: float) -> float:
    """CheXpert uncertainty policy: map uncertain (-1) and NaN to positive/neg."""
    if v == 1:
        return 1.0
    if v == -1:       # U-Ones: treat uncertain as positive
        return 1.0
    return 0.0


class RealCXRDataset(Dataset):
    def __init__(self, cfg: Config, tokenizer, csv_path: str, img_root: str,
                 domain_idx: int, split: str = "train", limit: Optional[int] = None):
        if not os.path.exists(csv_path):
            raise FileNotFoundError(
                f"CSV not found: {csv_path}. Download the dataset subset and "
                f"point cfg at it. See cdime_ai/data/datasets.py docstring."
            )
        self.cfg = cfg
        self.tokenizer = tokenizer
        self.img_root = img_root
        self.labels = cfg.labels
        self.size = cfg.data.image_size
        self.max_len = cfg.data.max_text_len
        self.domain_idx = domain_idx

        df = pd.read_csv(csv_path)
        if limit:
            df = df.sample(n=min(limit, len(df)), random_state=cfg.seed).reset_index(drop=True)
        self.df = df
        self.path_col = "Path" if "Path" in df.columns else "path"
        self.text_col = "report" if "report" in df.columns else None

    def __len__(self) -> int:
        return len(self.df)

    def _load_image(self, rel_path: str) -> torch.Tensor:
        path = os.path.join(self.img_root, rel_path)
        if not os.path.exists(path):
            raise FileNotFoundError(f"Image missing: {path}")
        img = Image.open(path).convert("RGB").resize((self.size, self.size))
        arr = np.asarray(img, dtype=np.float32) / 255.0
        t = torch.from_numpy(arr).permute(2, 0, 1)
        mean = torch.tensor([0.485, 0.456, 0.406]).view(3, 1, 1)
        std = torch.tensor([0.229, 0.224, 0.225]).view(3, 1, 1)
        return (t - mean) / std

    def __getitem__(self, idx: int):
        row = self.df.iloc[idx]
        img = self._load_image(str(row[self.path_col]))
        y = np.array([_u_ones(row.get(l, 0)) for l in self.labels], dtype=np.float32)

        text = str(row[self.text_col]) if self.text_col else "chest radiograph"
        enc = self.tokenizer(
            text, padding="max_length", truncation=True,
            max_length=self.max_len, return_tensors="pt",
        )
        return {
            "image": img,
            "input_ids": enc["input_ids"].squeeze(0),
            "attention_mask": enc["attention_mask"].squeeze(0),
            "label": torch.from_numpy(y),
            "domain": torch.tensor(self.domain_idx, dtype=torch.long),
        }


# Map domain name -> (csv path, image root). Edit these for your Colab layout.
REAL_PATHS = {
    "CheXpert":  ("data/chexpert/subset.csv",  "data/chexpert"),
    "MIMIC-CXR": ("data/mimic/subset.csv",     "data/mimic"),
    "VinDr-CXR": ("data/vindr/subset.csv",     "data/vindr"),
}
