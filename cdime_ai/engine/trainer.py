"""Single-domain trainer with EWC, replay rehearsal and causal (IRM+V-REx) loss.

This is the per-phase workhorse of the continual runner. It mixes replay
exemplars into each batch so that (a) past domains are rehearsed against
forgetting and (b) batches span multiple environments, which is required for the
IRM / V-REx causal penalties to do anything.
"""
from __future__ import annotations

from typing import Optional
import torch
import torch.nn as nn
import torch.nn.functional as F

from ..config import Config
from ..utils import move_batch, get_logger
from ..causal.irm import causal_objective
from ..continual.ewc import EWC
from ..continual.replay import ReplayMemory

logger = get_logger()


def build_optimizer(model: nn.Module, cfg: Config):
    """Separate (smaller) LR group for the pretrained text encoder."""
    text_params, other_params = [], []
    for n, p in model.named_parameters():
        if not p.requires_grad:
            continue
        (text_params if "text_encoder" in n else other_params).append(p)
    groups = [{"params": other_params, "lr": cfg.train.lr}]
    if text_params:
        groups.append({"params": text_params, "lr": cfg.train.text_lr})
    return torch.optim.AdamW(groups, weight_decay=cfg.train.weight_decay)


def _merge_batches(cur: dict, replay: Optional[dict], device: str) -> dict:
    cur = move_batch(cur, device)
    if replay is None:
        return cur
    replay = move_batch(replay, device)
    out = {}
    for k in cur:
        if torch.is_tensor(cur[k]):
            out[k] = torch.cat([cur[k], replay[k]], dim=0)
        else:
            out[k] = cur[k]
    return out


def train_one_domain(model, loader, cfg: Config, domain_idx: int,
                     ewc: Optional[EWC] = None,
                     replay: Optional[ReplayMemory] = None) -> None:
    device = cfg.device
    model.to(device)
    model.set_domain(domain_idx)
    model.train()
    opt = build_optimizer(model, cfg)
    amp_on = cfg.train.amp and device == "cuda"
    scaler = torch.amp.GradScaler("cuda", enabled=amp_on)

    use_irm = cfg.train.use_irm
    use_vrex = cfg.train.use_causal_reg
    replay_n = int(cfg.train.batch_size * cfg.train.replay_ratio)

    for epoch in range(cfg.train.epochs_per_domain):
        # IRM warm-up: ramp the penalty in after a few epochs for stability.
        irm_w = cfg.train.irm_lambda if epoch >= cfg.train.irm_anneal_epochs else 0.0
        running = {"bce": 0.0, "irm": 0.0, "vrex": 0.0, "ewc": 0.0}
        nb = 0
        opt.zero_grad()
        for step, batch in enumerate(loader):
            rep = replay.sample(replay_n) if (replay is not None and not replay.is_empty()) else None
            merged = _merge_batches(batch, rep, device)

            with torch.amp.autocast("cuda", enabled=amp_on):
                logits = model(merged, domain=domain_idx)
                bce = F.binary_cross_entropy_with_logits(logits, merged["label"])
                loss = bce

                # IRM / V-REx need float32 grads of the dummy scalar; compute
                # outside autocast for numerical stability.
            if use_irm or use_vrex:
                logits_f = logits.float()
                irm_t, vrex_t = causal_objective(
                    logits_f, merged["label"], merged["domain"],
                    irm_lambda=(irm_w if use_irm else 0.0),
                    vrex_lambda=(cfg.train.causal_reg_lambda if use_vrex else 0.0))
                loss = loss + irm_t + vrex_t
                running["irm"] += float(irm_t.detach())
                running["vrex"] += float(vrex_t.detach())

            if ewc is not None and cfg.train.use_ewc:
                ewc_pen = ewc.penalty(model)
                loss = loss + ewc_pen
                running["ewc"] += float(ewc_pen.detach())

            loss = loss / cfg.train.grad_accum_steps
            scaler.scale(loss).backward()

            if (step + 1) % cfg.train.grad_accum_steps == 0:
                scaler.unscale_(opt)
                torch.nn.utils.clip_grad_norm_(model.parameters(), cfg.train.max_grad_norm)
                scaler.step(opt)
                scaler.update()
                opt.zero_grad()

            running["bce"] += float(bce.detach())
            nb += 1

        logger.info(
            f"  [domain {domain_idx} epoch {epoch + 1}/{cfg.train.epochs_per_domain}] "
            f"bce={running['bce']/max(nb,1):.4f} irm={running['irm']/max(nb,1):.4f} "
            f"vrex={running['vrex']/max(nb,1):.4f} ewc={running['ewc']/max(nb,1):.2f}")
