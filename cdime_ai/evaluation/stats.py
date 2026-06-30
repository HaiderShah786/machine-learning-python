"""Statistical significance testing (proposal Section 8).

Provides seed-averaged reporting (mean ± std), paired significance tests
(paired t-test / Wilcoxon with a normality check), McNemar's test for paired
classifier comparison, and bootstrap confidence intervals for AUROC / F1.
"""
from __future__ import annotations

from typing import Dict, List, Sequence, Tuple
import numpy as np
from scipy import stats
from sklearn.metrics import roc_auc_score, f1_score


def mean_std(values: Sequence[float]) -> Tuple[float, float]:
    a = np.asarray(values, dtype=float)
    return float(a.mean()), float(a.std(ddof=1)) if len(a) > 1 else 0.0


def paired_comparison(a: Sequence[float], b: Sequence[float],
                      alpha: float = 0.05) -> Dict:
    """Compare two paired sets of per-seed scores.

    Picks a paired t-test if both samples look normal (Shapiro), else the
    Wilcoxon signed-rank test. Returns the test used, statistic, p-value and
    whether the difference is significant at ``alpha``.
    """
    a = np.asarray(a, float)
    b = np.asarray(b, float)
    diff = a - b
    if len(diff) < 3 or np.allclose(diff, 0):
        return {"test": "none", "p_value": 1.0, "significant": False,
                "mean_diff": float(diff.mean())}
    # Normality of the differences.
    normal = stats.shapiro(diff).pvalue > 0.05 if len(diff) >= 3 else True
    if normal:
        stat, p = stats.ttest_rel(a, b)
        test = "paired_t"
    else:
        stat, p = stats.wilcoxon(a, b)
        test = "wilcoxon"
    return {"test": test, "statistic": float(stat), "p_value": float(p),
            "significant": bool(p < alpha), "mean_diff": float(diff.mean())}


def mcnemar_test(y_true: np.ndarray, pred_a: np.ndarray, pred_b: np.ndarray,
                 alpha: float = 0.05) -> Dict:
    """McNemar's test on paired correct/incorrect outcomes of two models."""
    y_true, pred_a, pred_b = map(lambda x: np.asarray(x).flatten(), (y_true, pred_a, pred_b))
    correct_a = pred_a == y_true
    correct_b = pred_b == y_true
    b01 = int(np.sum(correct_a & ~correct_b))   # a right, b wrong
    b10 = int(np.sum(~correct_a & correct_b))   # a wrong, b right
    if b01 + b10 == 0:
        return {"test": "mcnemar", "p_value": 1.0, "significant": False, "b01": b01, "b10": b10}
    # Exact binomial (robust for small/large discordant counts).
    p = stats.binomtest(min(b01, b10), b01 + b10, 0.5).pvalue
    return {"test": "mcnemar", "p_value": float(p), "significant": bool(p < alpha),
            "b01": b01, "b10": b10}


def bootstrap_ci(probs: np.ndarray, targets: np.ndarray, metric: str = "auroc",
                 n_boot: int = 1000, alpha: float = 0.05, seed: int = 0) -> Dict:
    """Bootstrap percentile CI for macro AUROC or macro F1."""
    rng = np.random.default_rng(seed)
    n = len(targets)
    scores = []
    for _ in range(n_boot):
        idx = rng.integers(0, n, n)
        p, t = probs[idx], targets[idx]
        try:
            if metric == "auroc":
                vals = [roc_auc_score(t[:, c], p[:, c])
                        for c in range(t.shape[1]) if len(np.unique(t[:, c])) > 1]
                s = np.mean(vals) if vals else np.nan
            else:
                s = f1_score(t, (p >= 0.5).astype(int), average="macro", zero_division=0)
            if not np.isnan(s):
                scores.append(s)
        except ValueError:
            continue
    scores = np.array(scores)
    lo, hi = np.percentile(scores, [100 * alpha / 2, 100 * (1 - alpha / 2)])
    return {"metric": metric, "mean": float(scores.mean()),
            "ci_low": float(lo), "ci_high": float(hi), "n_boot": len(scores)}
