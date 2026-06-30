"""End-to-end CDIME-AI experiment driver.

Runs the full continual protocol, an explainability + uncertainty + decision
demo, and (optionally) the ablation study with statistical significance tests.
Results are written as JSON to ``cfg.output_dir``.

Usage
-----
    python -m cdime_ai.main --fast            # smoke test (<1 min, CPU ok)
    python -m cdime_ai.main                    # default T4 run
    python -m cdime_ai.main --ablation --seeds 42 43 44
    python -m cdime_ai.main --mode real        # use on-disk datasets
"""
from __future__ import annotations

import argparse
import json
import os

import numpy as np
import torch

from .config import Config
from .tokenizer import get_tokenizer
from .data.loaders import build_domain_loaders
from .engine.continual_runner import run_continual
from .ablation import run_ablation_study
from .evaluation.stats import bootstrap_ci, mcnemar_test
from .evaluation.metrics import collect_predictions
from .uncertainty.mc_dropout import mc_dropout_predict
from .explain.gradcam import GradCAM
from .explain.attention import image_to_text_attention, top_text_evidence
from .explain.counterfactual import generate_counterfactual
from .decision.therapeutic import make_decision
from .utils import get_logger, set_seed, move_batch, count_parameters

logger = get_logger()


def parse_args():
    p = argparse.ArgumentParser(description="CDIME-AI experiment driver")
    p.add_argument("--fast", action="store_true", help="tiny smoke-test config")
    p.add_argument("--mode", default="synthetic", choices=["synthetic", "real"])
    p.add_argument("--ablation", action="store_true", help="run full ablation study")
    p.add_argument("--seeds", type=int, nargs="+", default=[42, 43, 44])
    p.add_argument("--output", default="results")
    return p.parse_args()


def explain_and_decide(model, loaders, cfg: Config, tokenizer) -> dict:
    """Run the explainability + uncertainty + decision-support demo on one batch."""
    logger.info("=== Explainability / uncertainty / decision-support demo ===")
    last_domain = len(cfg.domains) - 1
    batch = next(iter(loaders[cfg.domains[last_domain]]["test"]))
    batch = move_batch(batch, cfg.device)

    # Uncertainty via MC-Dropout
    mean_prob, std_prob, entropy = mc_dropout_predict(model, batch, last_domain, n_samples=15)

    # Grad-CAM + attention + counterfactual for the first sample / top class
    top_class = int(mean_prob[0].argmax().item())
    cam = GradCAM(model)(batch, class_idx=top_class, domain=last_domain)
    attn = image_to_text_attention(model, batch, domain=last_domain)
    evidence = top_text_evidence(attn[0] if attn is not None else None,
                                 batch["input_ids"][0], tokenizer, k=4)
    cf_img, delta, p0, p1 = generate_counterfactual(
        model, batch, class_idx=top_class, domain=last_domain)

    # Decision support for the first study
    report = make_decision(
        cfg.labels,
        probs=mean_prob[0].cpu().numpy(),
        uncertainties=std_prob[0].cpu().numpy(),
        evidence=[f"{t} ({w:.2f})" for t, w in evidence] or None,
    )
    logger.info("\n" + report.to_text())

    return {
        "top_class": cfg.labels[top_class],
        "mc_mean_prob": mean_prob[0].cpu().tolist(),
        "mc_std_prob": std_prob[0].cpu().tolist(),
        "predictive_entropy": float(entropy[0].item()),
        "gradcam_shape": list(cam.shape),
        "gradcam_peak": float(cam[0].max().item()),
        "text_evidence": evidence,
        "counterfactual": {"class": cfg.labels[top_class],
                           "orig_prob": p0, "cf_prob": p1,
                           "perturbation_l2": float(delta.norm().item())},
        "decision_report": report.to_text(),
    }


def main():
    args = parse_args()
    cfg = Config.fast_debug() if args.fast else Config()
    cfg.data.mode = args.mode
    cfg.output_dir = args.output
    os.makedirs(cfg.output_dir, exist_ok=True)
    set_seed(cfg.seed)

    logger.info(f"Device: {cfg.device} | mode: {cfg.data.mode} | "
                f"domains: {cfg.domains}")
    tokenizer = get_tokenizer(cfg)

    # ---- Main continual run ----
    loaders = build_domain_loaders(cfg, tokenizer)
    results = run_continual(cfg, loaders, tokenizer)
    model = results.pop("model")
    logger.info(f"Trainable params: {count_parameters(model):,}")

    # ---- Bootstrap CIs + McNemar on the final stage (last domain) ----
    last = len(cfg.domains) - 1
    probs, targets = collect_predictions(model, loaders[cfg.domains[last]]["test"],
                                         domain=last, device=cfg.device)
    results["bootstrap_auroc"] = bootstrap_ci(probs, targets, "auroc", n_boot=500, seed=cfg.seed)
    results["bootstrap_f1"] = bootstrap_ci(probs, targets, "f1", n_boot=500, seed=cfg.seed)

    # ---- Explainability + uncertainty + decision support ----
    results["demo"] = explain_and_decide(model, loaders, cfg, tokenizer)

    with open(os.path.join(cfg.output_dir, "main_results.json"), "w") as f:
        json.dump(results, f, indent=2)
    cfg.save(os.path.join(cfg.output_dir, "config.json"))
    logger.info(f"Saved main results to {cfg.output_dir}/main_results.json")

    # ---- Ablation study (optional, multi-seed + significance) ----
    if args.ablation:
        logger.info("\n===== Running ablation study (this is the long part) =====")
        study = run_ablation_study(cfg, tokenizer, build_domain_loaders, seeds=args.seeds)
        with open(os.path.join(cfg.output_dir, "ablation_results.json"), "w") as f:
            json.dump({"summary": study["summary"], "seeds": study["seeds"]}, f, indent=2)
        logger.info(f"Saved ablation results to {cfg.output_dir}/ablation_results.json")
        _print_ablation_table(study["summary"])


def _print_ablation_table(summary: dict) -> None:
    logger.info("\n%-38s %12s %12s %10s %8s" %
                ("Variant", "AvgAcc", "Forget", "AUROC", "p(acc)"))
    for name, row in summary.items():
        aa = row.get("average_accuracy", {})
        fg = row.get("forgetting", {})
        au = row.get("auroc", {})
        sig = row.get("sig_avg_acc", {})
        logger.info("%-38s %5.3f±%.3f %5.3f±%.3f %5.3f   %7s" % (
            name[:38], aa.get("mean", 0), aa.get("std", 0),
            fg.get("mean", 0), fg.get("std", 0), au.get("mean", 0),
            f"{sig.get('p_value', float('nan')):.3f}" if sig else "  ref"))


if __name__ == "__main__":
    main()
