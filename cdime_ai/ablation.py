"""Ablation study configurations (proposal Section 7) and the experiment driver.

Each ablation toggles one component and is run over multiple seeds. We report
mean ± std for every metric and run the paired significance tests from
``evaluation.stats`` against the full CDIME-AI model.
"""
from __future__ import annotations

from copy import deepcopy
from typing import Callable, Dict, List, Tuple
import numpy as np

from .config import Config
from .models.cdime_model import AblationFlags
from .engine.continual_runner import run_continual
from .evaluation.stats import mean_std, paired_comparison
from .utils import get_logger

logger = get_logger()


def _cfg(base: Config, **train_overrides) -> Config:
    c = deepcopy(base)
    for k, v in train_overrides.items():
        setattr(c.train, k, v)
    return c


def build_variants(base: Config) -> Dict[str, Tuple[Config, AblationFlags]]:
    """Return {name: (config, ablation_flags)} for the full study."""
    full_flags = AblationFlags(use_text=True, use_cross_attention=True,
                               use_adapters=True, enable_mc_dropout=True)
    return {
        # Reference model
        "CDIME-AI (full)": (deepcopy(base), full_flags),

        # Baselines named in Section 6
        "Baseline: multimodal, no continual": (
            _cfg(base, use_ewc=False, use_replay=False, use_adapters=False,
                 use_irm=False, use_causal_reg=False), full_flags),
        "Baseline: continual, no causal": (
            _cfg(base, use_irm=False, use_causal_reg=False), full_flags),

        # A1 remove causal learning
        "A1: no causal": (_cfg(base, use_irm=False, use_causal_reg=False), full_flags),
        # A2 remove domain-incremental learning
        "A2: no domain-incremental": (
            _cfg(base, use_ewc=False, use_replay=False, use_adapters=False),
            AblationFlags(use_text=True, use_cross_attention=True,
                          use_adapters=False, enable_mc_dropout=True)),
        # A3 remove explainability (no effect on predictive metrics; tracked separately)
        "A3: no explainability": (deepcopy(base), full_flags),
        # A4 remove uncertainty estimation (no MC-dropout / temp scaling)
        "A4: no uncertainty": (deepcopy(base),
                               AblationFlags(use_text=True, use_cross_attention=True,
                                             use_adapters=True, enable_mc_dropout=False)),
        # A5 image-only
        "A5: image-only": (deepcopy(base),
                           AblationFlags(use_text=False, use_cross_attention=False,
                                         use_adapters=True, enable_mc_dropout=True)),
        # A6 concat instead of cross-attention
        "A6: concat fusion": (deepcopy(base),
                              AblationFlags(use_text=True, use_cross_attention=False,
                                            use_adapters=True, enable_mc_dropout=True)),
        # A7 each CL strategy individually
        "A7a: EWC only": (_cfg(base, use_replay=False, use_adapters=False), full_flags),
        "A7b: Replay only": (
            _cfg(base, use_ewc=False, use_adapters=False),
            AblationFlags(use_text=True, use_cross_attention=True,
                          use_adapters=False, enable_mc_dropout=True)),
        "A7c: Adapters only": (_cfg(base, use_ewc=False, use_replay=False), full_flags),
    }


def run_seeded(cfg: Config, flags: AblationFlags, tokenizer, seeds: List[int],
               loader_builder: Callable) -> Dict[str, List[float]]:
    """Run one variant across seeds; collect per-seed scalar metrics."""
    keys = ["average_accuracy", "forgetting", "backward_transfer", "forward_transfer"]
    diag_keys = ["accuracy", "f1", "auroc", "ece", "brier"]
    acc: Dict[str, List[float]] = {k: [] for k in keys + diag_keys}

    for s in seeds:
        c = deepcopy(cfg)
        c.seed = s
        calibrate = flags.enable_mc_dropout  # A4 disables calibration too
        loaders = loader_builder(c, tokenizer)
        res = run_continual(c, loaders, tokenizer, ablation=flags, calibrate=calibrate)
        for k in keys:
            acc[k].append(res["continual"][k])
        # Average the final-stage diagnostic metrics over seen domains.
        final = res["final_diagnostic"]
        for dk in diag_keys:
            vals = [m[dk] for m in final.values() if dk in m and not np.isnan(m.get(dk, np.nan))]
            acc[dk].append(float(np.mean(vals)) if vals else float("nan"))
    return acc


def run_ablation_study(base: Config, tokenizer, loader_builder: Callable,
                       seeds: List[int] = (42, 43, 44)) -> Dict:
    """Run every variant over seeds and compare each against the full model."""
    variants = build_variants(base)
    raw: Dict[str, Dict[str, List[float]]] = {}
    for name, (cfg, flags) in variants.items():
        logger.info(f"\n########## Variant: {name} ##########")
        raw[name] = run_seeded(cfg, flags, tokenizer, list(seeds), loader_builder)

    # Summarise mean ± std and significance vs. full model.
    ref = raw["CDIME-AI (full)"]
    table: Dict[str, Dict] = {}
    for name, metrics in raw.items():
        row = {}
        for k, vals in metrics.items():
            m, sd = mean_std([v for v in vals if not np.isnan(v)])
            row[k] = {"mean": m, "std": sd}
        # Significance on average_accuracy and auroc vs the full model.
        if name != "CDIME-AI (full)":
            row["sig_avg_acc"] = paired_comparison(
                ref["average_accuracy"], metrics["average_accuracy"])
            row["sig_auroc"] = paired_comparison(ref["auroc"], metrics["auroc"])
        table[name] = row
    return {"raw": raw, "summary": table, "seeds": list(seeds)}
