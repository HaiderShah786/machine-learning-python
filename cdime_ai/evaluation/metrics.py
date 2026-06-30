"""Evaluation metrics: diagnostic, continual-learning, calibration, faithfulness.

All diagnostic metrics are computed in a multi-label setting (macro-averaged
over the 5 pathologies). Continual-learning metrics follow Lopez-Paz & Ranzato
(GEM, 2017) using the accuracy matrix R[i, j] = accuracy on domain j after
training on domain i.
"""
from __future__ import annotations

from typing import Dict, List
import numpy as np
import torch

from sklearn.metrics import (
    roc_auc_score, f1_score, precision_score, recall_score, accuracy_score,
)


# -----------------------------------------------------------------------------
# Diagnostic metrics
# -----------------------------------------------------------------------------
@torch.no_grad()
def collect_predictions(model, loader, domain: int, device: str):
    model.eval()
    from ..utils import move_batch
    probs, targets = [], []
    for batch in loader:
        batch = move_batch(batch, device)
        logits = model(batch, domain=domain)
        probs.append(torch.sigmoid(logits).cpu().numpy())
        targets.append(batch["label"].cpu().numpy())
    return np.concatenate(probs), np.concatenate(targets)


def diagnostic_metrics(probs: np.ndarray, targets: np.ndarray,
                       threshold: float = 0.5) -> Dict[str, float]:
    preds = (probs >= threshold).astype(int)
    out: Dict[str, float] = {}
    out["accuracy"] = float(accuracy_score(targets.flatten(), preds.flatten()))
    out["precision"] = float(precision_score(targets, preds, average="macro", zero_division=0))
    out["recall"] = float(recall_score(targets, preds, average="macro", zero_division=0))
    out["f1"] = float(f1_score(targets, preds, average="macro", zero_division=0))
    # Specificity (macro): TN / (TN + FP)
    specs = []
    for c in range(targets.shape[1]):
        tn = ((preds[:, c] == 0) & (targets[:, c] == 0)).sum()
        fp = ((preds[:, c] == 1) & (targets[:, c] == 0)).sum()
        specs.append(tn / (tn + fp + 1e-8))
    out["specificity"] = float(np.mean(specs))
    # AUROC (macro, skip degenerate single-class columns)
    aucs = []
    for c in range(targets.shape[1]):
        if len(np.unique(targets[:, c])) > 1:
            aucs.append(roc_auc_score(targets[:, c], probs[:, c]))
    out["auroc"] = float(np.mean(aucs)) if aucs else float("nan")
    return out


# -----------------------------------------------------------------------------
# Continual-learning metrics (from an accuracy matrix R)
# -----------------------------------------------------------------------------
def continual_metrics(R: np.ndarray) -> Dict[str, float]:
    """R[i, j] = test accuracy on domain j after training through domain i.

    Returns Average Accuracy, Forgetting Measure, Backward & Forward Transfer.
    """
    T = R.shape[0]
    avg_acc = float(np.mean(R[T - 1, :]))                      # final-row mean

    # Forgetting: for each earlier task, best-ever minus final accuracy.
    forgetting = []
    for j in range(T - 1):
        best = np.max(R[:T - 1, j]) if T > 1 else R[T - 1, j]
        forgetting.append(best - R[T - 1, j])
    forget = float(np.mean(forgetting)) if forgetting else 0.0

    # Backward transfer: effect of later learning on earlier tasks.
    bwt = float(np.mean([R[T - 1, j] - R[j, j] for j in range(T - 1)])) if T > 1 else 0.0

    # Forward transfer: accuracy on a task before training on it (vs. R diagonal-1).
    fwt = float(np.mean([R[j - 1, j] for j in range(1, T)])) if T > 1 else 0.0

    return {
        "average_accuracy": avg_acc,
        "forgetting": forget,
        "backward_transfer": bwt,
        "forward_transfer": fwt,
    }


# -----------------------------------------------------------------------------
# Calibration metrics
# -----------------------------------------------------------------------------
def expected_calibration_error(probs: np.ndarray, targets: np.ndarray,
                               n_bins: int = 15) -> float:
    """Multi-label ECE: flatten all (sample, label) confidences into bins."""
    p = probs.flatten()
    y = targets.flatten()
    conf = np.where(p >= 0.5, p, 1 - p)        # confidence of the predicted class
    correct = (p >= 0.5).astype(int) == y
    bins = np.linspace(0, 1, n_bins + 1)
    ece = 0.0
    for b in range(n_bins):
        m = (conf > bins[b]) & (conf <= bins[b + 1])
        if m.sum() > 0:
            ece += m.mean() * abs(correct[m].mean() - conf[m].mean())
    return float(ece)


def brier_score(probs: np.ndarray, targets: np.ndarray) -> float:
    return float(np.mean((probs - targets) ** 2))


# -----------------------------------------------------------------------------
# Explainability faithfulness: insertion / deletion (Petsiuk et al., 2018)
# -----------------------------------------------------------------------------
@torch.no_grad()
def deletion_auc(model, batch, heatmap: torch.Tensor, class_idx: int,
                 domain: int, steps: int = 20) -> float:
    """Lower is better: progressively delete most-important pixels; AUC of prob."""
    model.eval()
    x = batch["image"][:1].clone()
    hm = heatmap[:1].flatten()
    order = torch.argsort(hm, descending=True)      # most important first
    n = order.numel()
    single = {k: (v[:1].clone() if torch.is_tensor(v) else v) for k, v in batch.items()}
    probs = []
    for s in range(steps + 1):
        k = int(n * s / steps)
        masked = x.clone().view(1, 3, -1)
        masked[:, :, order[:k]] = 0.0
        single["image"] = masked.view_as(x)
        probs.append(torch.sigmoid(model(single, domain=domain))[0, class_idx].item())
    return float(np.trapz(probs, dx=1.0 / steps))


def localization_overlap(heatmap: torch.Tensor, mask: torch.Tensor) -> float:
    """IoU-style overlap between thresholded saliency and a ground-truth ROI."""
    hm = (heatmap >= heatmap.mean()).float()
    inter = (hm * mask).sum()
    union = ((hm + mask) >= 1).float().sum()
    return float((inter / (union + 1e-8)).item())
