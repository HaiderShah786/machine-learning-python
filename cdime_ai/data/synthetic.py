"""Synthetic multimodal chest-X-ray generator with a *causal* structure.

Why synthetic data?  The real datasets (CheXpert ~440 GB raw, MIMIC-CXR ~377 GB)
cannot be downloaded on free Colab. To make the *entire* CDIME-AI pipeline —
continual learning, causal regularisation, ablations and statistical testing —
runnable and reproducible on a T4, we generate data that mimics the multimodal,
multi-label, domain-shifted structure of the real task.

Causal design (this is the important part for the paper):
    * Each image embeds a *causal* lesion signal whose appearance is stable
      across domains -> P(Y | causal features) is invariant.
    * Each image also embeds a *spurious shortcut* (a corner marker) whose
      correlation with the label **flips across domains**. An ERM model latches
      onto the shortcut and fails on the next domain; an IRM / causally
      regularised model learns the invariant signal and generalises.

This lets us empirically demonstrate the value of the causal module exactly as
claimed in the proposal's Research Gap (shortcut learning) and Ablation A1.
"""
from __future__ import annotations

from typing import List, Tuple
import numpy as np
import torch
from torch.utils.data import Dataset

from ..config import Config


# Per-pathology canonical lesion location (row, col) and phrasing for reports.
_LESION_SITES = {
    "Atelectasis":      (0.30, 0.30),
    "Cardiomegaly":     (0.60, 0.50),
    "Consolidation":    (0.40, 0.70),
    "Edema":            (0.50, 0.40),
    "Pleural Effusion": (0.80, 0.65),
}
_PHRASES = {
    "Atelectasis":      "patchy volume loss in the left upper zone",
    "Cardiomegaly":     "enlarged cardiac silhouette",
    "Consolidation":    "dense airspace opacity in the right lung",
    "Edema":            "bilateral perihilar haziness",
    "Pleural Effusion": "blunting of the right costophrenic angle",
}
_NEGATION = "no acute cardiopulmonary abnormality"


def _draw_blob(img: np.ndarray, cy: float, cx: float, intensity: float, sigma: float) -> None:
    """Add a soft Gaussian blob (a lesion) to a single-channel image in place."""
    h, w = img.shape
    yy, xx = np.mgrid[0:h, 0:w]
    y0, x0 = cy * h, cx * w
    blob = intensity * np.exp(-(((yy - y0) ** 2 + (xx - x0) ** 2) / (2 * (sigma * h) ** 2)))
    img += blob


class SyntheticCXRDataset(Dataset):
    """Generates (image, tokenized report, multi-label) triples for one domain.

    Parameters
    ----------
    domain_idx : index of the domain in the continual sequence; controls the
        direction of the spurious shortcut so it flips between domains.
    """

    def __init__(
        self,
        cfg: Config,
        tokenizer,
        n: int,
        domain_idx: int,
        split: str = "train",
        seed: int = 0,
    ):
        self.cfg = cfg
        self.labels = cfg.labels
        self.n_classes = len(self.labels)
        self.size = cfg.data.image_size
        self.tokenizer = tokenizer
        self.max_len = cfg.data.max_text_len
        self.domain_idx = domain_idx
        self.shortcut_strength = cfg.data.shortcut_strength

        rng = np.random.default_rng(seed + 1000 * domain_idx + (0 if split == "train" else 7))
        # Multi-label targets (each pathology present independently ~30%).
        self.y = (rng.random((n, self.n_classes)) < 0.3).astype(np.float32)
        # Ensure no all-zero degenerate rows dominate.
        empties = self.y.sum(1) == 0
        self.y[empties, rng.integers(0, self.n_classes, empties.sum())] = 1.0
        self.seeds = rng.integers(0, 2 ** 31 - 1, size=n)
        # Domain-specific global intensity / contrast shift (covariate shift).
        self.domain_bias = 0.05 * domain_idx
        self.domain_contrast = 1.0 + 0.1 * domain_idx

    def __len__(self) -> int:
        return len(self.y)

    def _render(self, idx: int) -> np.ndarray:
        rng = np.random.default_rng(self.seeds[idx])
        s = self.size
        # Base "lung field" texture.
        img = 0.25 + 0.05 * rng.standard_normal((s, s)).astype(np.float32)
        img = np.clip(img, 0, 1)
        y = self.y[idx]

        # --- Causal lesion signals (domain-invariant) ---
        for c, present in enumerate(y):
            if present:
                cy, cx = _LESION_SITES[self.labels[c]]
                _draw_blob(img, cy, cx, intensity=0.6, sigma=0.06)

        # --- Spurious shortcut (domain-dependent direction) ---
        # A small bright corner marker. In even domains it correlates with the
        # FIRST positive label; in odd domains the correlation is inverted.
        first_pos = int(y.argmax()) if y.sum() > 0 else 0
        corr = self.shortcut_strength
        flip = -1 if (self.domain_idx % 2 == 1) else 1
        # Marker present iff (label parity) agrees with domain-flipped rule.
        rule = (first_pos % 2 == 0)
        marker = (rng.random() < corr) == (rule if flip == 1 else not rule)
        if marker:
            img[:14, :14] += 0.7  # top-left corner shortcut patch

        # --- Domain covariate shift ---
        img = np.clip((img + self.domain_bias) * self.domain_contrast, 0, 1)
        return img.astype(np.float32)

    def _report(self, idx: int) -> str:
        y = self.y[idx]
        findings = [_PHRASES[self.labels[c]] for c, p in enumerate(y) if p]
        if not findings:
            return _NEGATION + "."
        return "Findings: " + "; ".join(findings) + "."

    def __getitem__(self, idx: int):
        img = self._render(idx)                       # (H, W) in [0,1]
        img = torch.from_numpy(img).unsqueeze(0).repeat(3, 1, 1)  # 3-channel
        # ImageNet-style normalisation (matches timm Swin pretraining).
        mean = torch.tensor([0.485, 0.456, 0.406]).view(3, 1, 1)
        std = torch.tensor([0.229, 0.224, 0.225]).view(3, 1, 1)
        img = (img - mean) / std

        report = self._report(idx)
        enc = self.tokenizer(
            report,
            padding="max_length",
            truncation=True,
            max_length=self.max_len,
            return_tensors="pt",
        )
        return {
            "image": img,
            "input_ids": enc["input_ids"].squeeze(0),
            "attention_mask": enc["attention_mask"].squeeze(0),
            "label": torch.from_numpy(self.y[idx]),
            "domain": torch.tensor(self.domain_idx, dtype=torch.long),
        }
