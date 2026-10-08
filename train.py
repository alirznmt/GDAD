"""Training entry point for GDAD (single dataset).

Implements the full unsupervised training loop: AdamW, warmup + cosine LR
schedule, AMP (on CUDA), gradient clipping, EMA of weights (optional), early
stopping on a stable validation denoising loss, and best-checkpoint saving.

Usage:
    python train.py --dataset SMAP
    python train.py --dataset PSM --override train.epochs=30 model.hidden_dim=96
"""
from __future__ import annotations

import argparse
import math
import os
from typing import List

import torch
from torch.optim.lr_scheduler import LambdaLR

from models import assert_effective_graph_mode, build_model
from utils.config import load_config
from utils.data import build_dataloaders
from utils.logger import get_logger
from utils.provenance import collect_data_provenance, short_provenance
from utils.seed import set_seed


def build_scheduler(optimizer, warmup_steps: int, total_steps: int) -> LambdaLR:
    """Linear warmup followed by cosine decay to ~0."""

    def lr_lambda(step: int) -> float:
        if step < warmup_steps:
            return (step + 1) / max(1, warmup_steps)
        progress = (step - warmup_steps) / max(1, total_steps - warmup_steps)
        return 0.5 * (1.0 + math.cos(math.pi * min(progress, 1.0)))

    return LambdaLR(optimizer, lr_lambda)


@torch.no_grad()
def validate(model, loader, device, val_steps: List[int]) -> float:
    """Deterministic-ish validation loss averaged over a fixed set of steps."""
    model.eval()
    total, count = 0.0, 0
    diffusion = model.diffusion
    for batch in loader:
        x = (batch[0] if isinstance(batch, (list, tuple)) else batch).to(device)
        for step in val_steps:
            t = torch.full((x.shape[0],), int(step), device=device, dtype=torch.long)
            noise = torch.randn_like(x)
            x_t = diffusion.q_sample(x, t, noise)
            eps = diffusion.model(x_t, t)
            loss = diffusion._loss(eps, noise)
            total += loss.item() * x.shape[0]
            count += x.shape[0]
    return total / max(count, 1)


def train_one_dataset(cfg, logger=None, epoch_callback=None) -> str:
    """Train GDAD on one dataset; return the path to the best checkpoint.

    Args:
        cfg: resolved config.
        logger: optional logger.
        epoch_callback: optional ``fn(epoch, train_loss, val_loss) -> bool``
            invoked after each validation. If it returns True, training stops
            early (used by HPO pruners). It may also raise to abort the run
            (e.g. ``optuna.TrialPruned``); such exceptions are not caught here.
    """
    name = cfg.dataset["name"]
    ckpt_dir = os.path.join(cfg.train.get("ckpt_dir", "checkpoints"), name)
    os.makedirs(ckpt_dir, exist_ok=True)
    if logger is None:
        logger = get_logger("gdad", log_dir=os.path.join("logs", name))

    set_seed(cfg.get("seed", 42), deterministic=cfg.train.get("deterministic", True))
    device = torch.device(
        "cuda" if (torch.cuda.is_available() and cfg.train.get("use_cuda", True)) else "cpu"
    )
    logger.info(f"[{name}] device={device}")

    data_provenance = collect_data_provenance(cfg)
    logger.info(f"[{name}] data provenance: {short_provenance(data_provenance)}")
    train_loader, val_loader, _, _, num_nodes = build_dataloaders(cfg)
    logger.info(f"[{name}] sensors={num_nodes} | train_batches={len(train_loader)}")

    model = build_model(cfg, num_nodes=num_nodes).to(device)
    effective_graph_mode = assert_effective_graph_mode(
        model,
        cfg.model.get("graph_mode"),
    )
    logger.info(
        f"[{name}] effective_graph_mode={effective_graph_mode} | "
        f"requested_graph_mode={cfg.model.get('graph_mode')}"
    )
    n_params = sum(p.numel() for p in model.parameters())
    logger.info(f"[{name}] model params={n_params/1e6:.2f}M")

    epochs = cfg.train.get("epochs", 30)
    lr = cfg.train.get("lr", 1e-3)
    weight_decay = cfg.train.get("weight_decay", 1e-5)
    grad_clip = cfg.train.get("grad_clip", 1.0)
    patience = cfg.train.get("patience", 7)

    optimizer = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=weight_decay)
    total_steps = max(1, epochs * len(train_loader))
    warmup_steps = int(cfg.train.get("warmup_frac", 0.05) * total_steps)
    scheduler = build_scheduler(optimizer, warmup_steps, total_steps)

    use_amp = device.type == "cuda" and cfg.train.get("amp", True)
    scaler = torch.cuda.amp.GradScaler(enabled=use_amp)

    val_steps = cfg.eval.get("eval_steps", [int(0.1 * cfg.diffusion.get("timesteps", 100))])
    best_val = float("inf")
    bad_epochs = 0
    best_path = os.path.join(ckpt_dir, "best.pt")

    for epoch in range(1, epochs + 1):
        model.train()
        running = 0.0
        for batch in train_loader:
            x = (batch[0] if isinstance(batch, (list, tuple)) else batch).to(device)
            optimizer.zero_grad(set_to_none=True)
            with torch.cuda.amp.autocast(enabled=use_amp):
                loss = model.training_loss(x)
            scaler.scale(loss).backward()
            if grad_clip:
                scaler.unscale_(optimizer)
                torch.nn.utils.clip_grad_norm_(model.parameters(), grad_clip)
            scaler.step(optimizer)
            scaler.update()
            scheduler.step()
            running += loss.item()

        train_loss = running / max(len(train_loader), 1)
        val_loss = validate(model, val_loader, device, val_steps)
        logger.info(
            f"[{name}] epoch {epoch:03d}/{epochs} | "
            f"train={train_loss:.5f} | val={val_loss:.5f} | "
            f"lr={scheduler.get_last_lr()[0]:.2e}"
        )

        if val_loss < best_val - 1e-6:
            best_val = val_loss
            bad_epochs = 0
            torch.save(
                {
                    "model": model.state_dict(),
                    "config": cfg.to_dict(),
                    "num_nodes": num_nodes,
                    "data_provenance": data_provenance,
                    "effective_graph_mode": effective_graph_mode,
                    "epoch": epoch,
                    "val_loss": val_loss,
                },
                best_path,
            )
            logger.info(f"[{name}] saved best checkpoint (val={val_loss:.5f})")
        else:
            bad_epochs += 1
            if bad_epochs >= patience:
                logger.info(f"[{name}] early stopping at epoch {epoch}")
                break

        if epoch_callback is not None:
            if epoch_callback(epoch, train_loss, val_loss):
                logger.info(f"[{name}] training aborted by callback at epoch {epoch}")
                break

    if not os.path.exists(best_path):  # ensure at least one checkpoint exists
        torch.save(
            {"model": model.state_dict(), "config": cfg.to_dict(),
             "num_nodes": num_nodes, "data_provenance": data_provenance,
             "effective_graph_mode": effective_graph_mode,
             "epoch": 0, "val_loss": best_val},
            best_path,
        )
    logger.info(f"[{name}] training done. best_val={best_val:.5f}")
    return best_path


def parse_args():
    parser = argparse.ArgumentParser(description="Train GDAD on a dataset.")
    parser.add_argument("--dataset", required=True, type=str)
    parser.add_argument("--config_dir", default="configs", type=str)
    parser.add_argument("--override", nargs="*", default=[], help="key.sub=value")
    return parser.parse_args()


def main():
    args = parse_args()
    cfg = load_config(args.dataset, config_dir=args.config_dir, overrides=args.override)
    logger = get_logger("gdad", log_dir=os.path.join("logs", args.dataset))
    train_one_dataset(cfg, logger=logger)


if __name__ == "__main__":
    main()
