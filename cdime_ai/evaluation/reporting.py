"""Publication-ready reporting: ablation bar charts and LaTeX tables.

Turns the dicts produced by ``run_continual`` and ``run_ablation_study`` into
(a) matplotlib figures and (b) LaTeX ``tabular`` strings you can paste straight
into a paper. Significance vs. the full model is annotated with stars
(``*`` p<0.05, ``**`` p<0.01, ``***`` p<0.001).
"""
from __future__ import annotations

from typing import Dict, List, Optional


def _stars(p: Optional[float]) -> str:
    if p is None:
        return ""
    if p < 0.001:
        return "***"
    if p < 0.01:
        return "**"
    if p < 0.05:
        return "*"
    return ""


# -----------------------------------------------------------------------------
# Plots
# -----------------------------------------------------------------------------
def ablation_bar_chart(summary: Dict, metric: str = "average_accuracy",
                       title: Optional[str] = None, ax=None,
                       annotate_sig: bool = True):
    """Horizontal bar chart of one metric across variants (mean ± std).

    Returns the matplotlib Axes. The full model is highlighted; bars are sorted
    by mean so the contribution of each ablated component is visually obvious.
    """
    import numpy as np
    import matplotlib.pyplot as plt

    names = list(summary.keys())
    means = [summary[n].get(metric, {}).get("mean", float("nan")) for n in names]
    stds = [summary[n].get(metric, {}).get("std", 0.0) for n in names]
    sig_key = "sig_avg_acc" if metric == "average_accuracy" else "sig_auroc"
    pvals = [summary[n].get(sig_key, {}).get("p_value") for n in names]

    order = sorted(range(len(names)), key=lambda i: (means[i] if means[i] == means[i] else -1))
    names = [names[i] for i in order]
    means = [means[i] for i in order]
    stds = [stds[i] for i in order]
    pvals = [pvals[i] for i in order]

    if ax is None:
        _, ax = plt.subplots(figsize=(9, max(3, 0.5 * len(names))))
    colors = ["#2c7fb8" if "full" in n else "#a6bddb" for n in names]
    bars = ax.barh(names, means, xerr=stds, color=colors, capsize=3)
    ax.set_xlabel(metric.replace("_", " ").title())
    ax.set_title(title or f"Ablation: {metric.replace('_', ' ').title()}")
    if annotate_sig:
        for b, m, p in zip(bars, means, pvals):
            s = _stars(p)
            if s:
                ax.text(b.get_width() + (max(means) * 0.01 if means else 0.01),
                        b.get_y() + b.get_height() / 2, s, va="center", fontsize=11)
    ax.grid(axis="x", alpha=0.3)
    return ax


def plot_r_matrix(R, domains: List[str], ax=None):
    """Heatmap of the continual-learning accuracy matrix R[i, j]."""
    import numpy as np
    import matplotlib.pyplot as plt
    R = np.array(R)
    if ax is None:
        _, ax = plt.subplots(figsize=(5, 4))
    im = ax.imshow(R, cmap="viridis", vmin=0, vmax=1)
    ax.set_xticks(range(len(domains))); ax.set_xticklabels(domains, rotation=30, ha="right")
    ax.set_yticks(range(len(domains))); ax.set_yticklabels([f"after {d}" for d in domains])
    for i in range(R.shape[0]):
        for j in range(R.shape[1]):
            ax.text(j, i, f"{R[i, j]:.2f}", ha="center", va="center",
                    color="white" if R[i, j] < 0.6 else "black", fontsize=9)
    ax.set_title("Accuracy matrix R[i, j]")
    import matplotlib.pyplot as plt
    plt.colorbar(im, ax=ax, fraction=0.046, pad=0.04)
    return ax


# -----------------------------------------------------------------------------
# LaTeX tables
# -----------------------------------------------------------------------------
def ablation_to_latex(summary: Dict,
                      metrics: List[str] = ("average_accuracy", "forgetting", "auroc", "f1", "ece"),
                      caption: str = "Ablation study (mean$\\pm$std over seeds). "
                                     "Significance vs.\\ CDIME-AI (full): "
                                     "$^{*}p{<}0.05$, $^{**}p{<}0.01$, $^{***}p{<}0.001$.",
                      label: str = "tab:ablation") -> str:
    """Return a full LaTeX ``table`` (booktabs) for the ablation summary."""
    headers = {
        "average_accuracy": "Avg.\\ Acc.", "forgetting": "Forget.$\\downarrow$",
        "auroc": "AUROC", "f1": "F1", "ece": "ECE$\\downarrow$", "brier": "Brier$\\downarrow$",
        "backward_transfer": "BWT", "forward_transfer": "FWT", "accuracy": "Acc.",
    }
    cols = "l" + "c" * len(metrics)
    lines = [
        "\\begin{table}[t]", "\\centering",
        f"\\caption{{{caption}}}", f"\\label{{{label}}}",
        f"\\begin{{tabular}}{{{cols}}}", "\\toprule",
        "Variant & " + " & ".join(headers.get(m, m) for m in metrics) + " \\\\",
        "\\midrule",
    ]
    for name in summary:
        row = summary[name]
        cells = []
        for m in metrics:
            d = row.get(m, {})
            mean, std = d.get("mean", float("nan")), d.get("std", 0.0)
            star = ""
            if m == "average_accuracy":
                star = _stars(row.get("sig_avg_acc", {}).get("p_value"))
            elif m == "auroc":
                star = _stars(row.get("sig_auroc", {}).get("p_value"))
            cells.append(f"{mean:.3f}$\\pm${std:.3f}{('$^{'+star+'}$') if star else ''}")
        safe = name.replace("_", "\\_")
        bold = "\\textbf{" + safe + "}" if "full" in name else safe
        lines.append(bold + " & " + " & ".join(cells) + " \\\\")
    lines += ["\\bottomrule", "\\end{tabular}", "\\end{table}"]
    return "\n".join(lines)


def continual_to_latex(continual: Dict, bootstrap_auroc: Optional[Dict] = None,
                       caption: str = "Continual-learning performance of CDIME-AI "
                                      "across the CheXpert $\\to$ MIMIC-CXR $\\to$ VinDr-CXR sequence.",
                       label: str = "tab:continual") -> str:
    """LaTeX table for the headline continual-learning metrics."""
    rows = [
        ("Average Accuracy", continual.get("average_accuracy")),
        ("Forgetting $\\downarrow$", continual.get("forgetting")),
        ("Backward Transfer", continual.get("backward_transfer")),
        ("Forward Transfer", continual.get("forward_transfer")),
    ]
    lines = [
        "\\begin{table}[t]", "\\centering",
        f"\\caption{{{caption}}}", f"\\label{{{label}}}",
        "\\begin{tabular}{lc}", "\\toprule", "Metric & Value \\\\", "\\midrule",
    ]
    for name, v in rows:
        lines.append(f"{name} & {v:.3f} \\\\")
    if bootstrap_auroc:
        lines.append(f"AUROC (95\\% CI) & {bootstrap_auroc['mean']:.3f} "
                     f"[{bootstrap_auroc['ci_low']:.3f}, {bootstrap_auroc['ci_high']:.3f}] \\\\")
    lines += ["\\bottomrule", "\\end{tabular}", "\\end{table}"]
    return "\n".join(lines)
