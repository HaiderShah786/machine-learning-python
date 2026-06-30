"""Domain-incremental orchestration (proposal Section 6, Experimental Protocol).

Phase 1: train on CheXpert.
Phase 2: incrementally update on MIMIC-CXR (no retrain from scratch).
Phase 3: incrementally update on VinDr-CXR.

After every phase we evaluate on *all* domains seen so far to populate the
accuracy matrix R used for continual-learning metrics, and we consolidate EWC +
fill the replay buffer from the just-finished domain.
"""
from __future__ import annotations

from typing import Dict, List, Optional
import numpy as np

from ..config import Config
from ..models.cdime_model import CDIMEModel, AblationFlags
from ..continual.ewc import EWC
from ..continual.replay import ReplayMemory
from ..uncertainty.temperature_scaling import TemperatureScaler
from ..evaluation.metrics import (
    collect_predictions, diagnostic_metrics, continual_metrics,
    expected_calibration_error, brier_score,
)
from ..utils import get_logger, set_seed
from .trainer import train_one_domain

logger = get_logger()


def run_continual(cfg: Config, loaders: Dict[str, Dict], tokenizer,
                  ablation: Optional[AblationFlags] = None,
                  calibrate: bool = True) -> Dict:
    """Run the full continual sequence; return metrics + the trained model."""
    set_seed(cfg.seed)
    ablation = ablation or AblationFlags(
        use_adapters=cfg.train.use_adapters)
    model = CDIMEModel(cfg, ablation).to(cfg.device)

    ewc = EWC(cfg.train.ewc_lambda) if cfg.train.use_ewc else None
    replay = ReplayMemory(cfg.train.replay_size) if cfg.train.use_replay else None

    domains = cfg.domains
    T = len(domains)
    R = np.zeros((T, T))                   # R[i, j]: acc on j after training i
    per_stage: List[Dict] = []
    temperatures: List[float] = []

    for i, domain in enumerate(domains):
        logger.info(f"=== Phase {i + 1}: training on {domain} ===")
        train_one_domain(model, loaders[domain]["train"], cfg, i, ewc, replay)

        # Optional post-hoc calibration on the current domain's val split.
        temp = 1.0
        if calibrate:
            scaler = TemperatureScaler()
            temp = scaler.fit(model, loaders[domain]["val"], i, cfg.device)
        temperatures.append(temp)

        # Evaluate on every domain seen so far (and future ones for FWT).
        stage_metrics = {}
        for j, eval_domain in enumerate(domains):
            probs, targets = collect_predictions(
                model, loaders[eval_domain]["test"], domain=i, device=cfg.device)
            dm = diagnostic_metrics(probs, targets)
            R[i, j] = dm["accuracy"]
            if j <= i:
                dm["ece"] = expected_calibration_error(probs, targets)
                dm["brier"] = brier_score(probs, targets)
                stage_metrics[eval_domain] = dm
        per_stage.append(stage_metrics)
        logger.info(f"  After {domain}: "
                    + ", ".join(f"{d}={R[i, j]:.3f}" for j, d in enumerate(domains)))

        # Consolidate knowledge of the finished domain.
        if ewc is not None:
            ewc.consolidate(model, loaders[domain]["train"], cfg.device, i)
        if replay is not None:
            replay.add_loader(loaders[domain]["train"], i)

    cl = continual_metrics(R)
    logger.info(f"Continual metrics: {cl}")

    return {
        "R_matrix": R.tolist(),
        "continual": cl,
        "per_stage": per_stage,
        "temperatures": temperatures,
        "final_diagnostic": per_stage[-1],
        "model": model,
    }
