# CDIME-AI

**Causal Domain-Incremental Multimodal Explainable AI for Trustworthy Chest
X-ray Diagnosis and Therapeutic Decision Support**

A complete, runnable reference implementation of the CDIME-AI research proposal.
It is engineered to run **end-to-end on a free Google Colab T4 GPU** while
implementing every component of the methodology, so you can produce a full,
reproducible results table (with ablations and statistical tests) suitable for a
Q1 submission, then scale up on bigger hardware.

---

## What it implements (proposal → code map)

| Proposal component | Implementation |
|---|---|
| Image encoder: **Swin Transformer** | `models/image_encoder.py` (timm `swin_tiny`, ResNet fallback, exposes feature maps for Grad-CAM) |
| Text encoder: **ClinicalBERT** | `models/text_encoder.py` (`Bio_ClinicalBERT`, DistilBERT fallback, lower-layer freezing) |
| Fusion: **Cross-Attention Transformer** | `models/fusion.py` (bidirectional image↔text cross-attention) |
| Causal: **SCM / IRM / causal regularization** | `causal/irm.py` (IRMv1 gradient penalty + V-REx variance penalty across domains/environments) |
| Continual: **EWC / Replay / Domain Adapters** | `continual/ewc.py`, `continual/replay.py`, `models/adapters.py` |
| Explainability: **Grad-CAM / SHAP / Attention / Counterfactual** | `explain/` |
| Uncertainty: **MC-Dropout / Temperature Scaling** | `uncertainty/` |
| Decision support | `decision/therapeutic.py` (diagnosis + confidence + evidence + follow-up, with human-review flagging) |
| Experimental protocol (Phase 1→2→3) | `engine/continual_runner.py` |
| Ablations A1–A7 + baselines | `ablation.py` |
| Statistical testing | `evaluation/stats.py` (paired t-test / Wilcoxon, McNemar, bootstrap CIs) |
| Metrics | `evaluation/metrics.py` (diagnostic, CL: Avg-Acc/Forgetting/BWT/FWT, ECE/Brier, faithfulness) |

## Quick start

```bash
pip install -r cdime_ai/requirements.txt

# Fast smoke test (<1 min, CPU ok)
python -m cdime_ai.main --fast

# Full default T4 run (continual + explain + decision + bootstrap CIs)
python -m cdime_ai.main

# Add the ablation study with significance tests over seeds
python -m cdime_ai.main --ablation --seeds 42 43 44
```

Or open **`notebooks/CDIME_AI_Colab.ipynb`** in Colab (set runtime to T4) and run
top to bottom.

## Why synthetic data by default (and why it's not a shortcut)

The real datasets cannot be downloaded on free Colab (MIMIC-CXR ≈ 377 GB). The
synthetic generator (`data/synthetic.py`) produces multimodal, multi-label,
domain-shifted CXR-like data with a **causal structure**: a domain-invariant
lesion signal **plus** a spurious corner-marker shortcut whose correlation with
the label **flips between domains**. This:

1. lets the whole pipeline (training, continual learning, ablations, stats) run
   in minutes on a T4, and
2. **empirically demonstrates the causal module's value** — an ERM model latches
   onto the flipping shortcut and forgets/fails across domains, while the
   IRM + V-REx model learns the invariant signal (this is exactly proposal
   ablation **A1** and the "shortcut learning" gap in Section 2).

## Switching to real data

Edit `REAL_PATHS` in `data/datasets.py` to point at your CheXpert / MIMIC-CXR /
VinDr-CXR subset CSVs + image roots, then set `cfg.data.mode = "real"`. Every
other component is dataset-agnostic. Use subsets (a few thousand studies per
domain) on Colab.

## Reproducing Q1-quality numbers

* Run ≥5 seeds (`--seeds 42 43 44 45 46`); the code reports mean ± std and
  paired significance vs. the full model automatically.
* Scale `train_per_domain`, `epochs_per_domain`, and `replay_size` up.
* Report the full ablation table + bootstrap CIs + McNemar comparisons that the
  driver writes to `results/`.

## Outputs

* `results/main_results.json` — continual metrics, R-matrix, per-stage
  diagnostic/calibration metrics, bootstrap CIs, explainability + decision demo.
* `results/ablation_results.json` — per-variant mean ± std + significance.

## Disclaimer

Research/decision-support software only — **not** a medical device and not for
autonomous diagnosis. Therapeutic rules are conservative placeholders requiring
clinician review.
