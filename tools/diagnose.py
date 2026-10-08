"""Score-distribution diagnostics for GDAD (Phase 2).

Given a dataset + checkpoint (+ optional scoring overrides), this reports the
empirical evidence needed to understand *why* precision is low, WITHOUT using
test labels to select anything the model sees:

* normal vs. anomaly score statistics and distribution overlap,
* AUROC / AUPRC,
* best-F1 pointwise threshold with predicted anomaly ratio + TP/FP/FN,
* label-free (ratio / POT) predicted anomaly ratios,
* window-boundary effect on scores,
* per-channel mean residual (which sensors drive the score),
* reconstruction variance across repeated stochastic inference runs.

Usage:
    python tools/diagnose.py --dataset SMAP --ckpt checkpoints/SMAP/best.pt
    python tools/diagnose.py --dataset SWaT --ckpt checkpoints/SWaT/best.pt \
        --override sic.enabled=true score.sic_weight=1.0 score.denoise_weight=0.0 \
        --repeats 3
"""
from __future__ import annotations

import argparse
import json
import os
from typing import Dict

import numpy as np

from evaluate import score_dataset
from utils.config import load_config
from utils.logger import get_logger
from utils.metrics import (
    auroc_auprc,
    best_f1_search,
    pot_threshold,
    precision_recall_f1,
    threshold_by_anomaly_ratio,
)


def _dist_stats(x: np.ndarray) -> Dict[str, float]:
    return {
        "mean": float(np.mean(x)),
        "std": float(np.std(x)),
        "median": float(np.median(x)),
        "q90": float(np.quantile(x, 0.90)),
        "q99": float(np.quantile(x, 0.99)),
        "max": float(np.max(x)),
    }


def _overlap(normal: np.ndarray, anom: np.ndarray, bins: int = 100) -> Dict[str, float]:
    """Histogram-intersection and Bhattacharyya coefficient (1 = identical)."""
    lo = float(min(normal.min(), anom.min()))
    hi = float(max(normal.max(), anom.max()))
    if hi <= lo:
        return {"hist_intersection": 1.0, "bhattacharyya": 1.0}
    edges = np.linspace(lo, hi, bins + 1)
    pn, _ = np.histogram(normal, bins=edges, density=False)
    pa, _ = np.histogram(anom, bins=edges, density=False)
    pn = pn / max(pn.sum(), 1)
    pa = pa / max(pa.sum(), 1)
    return {
        "hist_intersection": float(np.minimum(pn, pa).sum()),
        "bhattacharyya": float(np.sqrt(pn * pa).sum()),
    }


def _boundary_effect(scores: np.ndarray, window: int) -> Dict[str, float]:
    idx = np.arange(len(scores))
    pos = idx % window
    is_boundary = (pos <= 1) | (pos >= window - 2)
    interior = ~is_boundary
    return {
        "mean_boundary": float(scores[is_boundary].mean()) if is_boundary.any() else float("nan"),
        "mean_interior": float(scores[interior].mean()) if interior.any() else float("nan"),
    }


def _per_channel(res: np.ndarray, top_k: int = 10) -> Dict[str, list]:
    """Mean residual per channel over all test windows/timesteps: res [nw, W, N]."""
    per_ch = res.reshape(-1, res.shape[-1]).mean(axis=0)
    order = np.argsort(-per_ch)[:top_k]
    return {
        "top_channels": [int(i) for i in order],
        "top_values": [float(per_ch[i]) for i in order],
        "channel_mean_overall": float(per_ch.mean()),
        "channel_mean_max": float(per_ch.max()),
    }


def diagnose(cfg, ckpt_path: str, repeats: int, logger) -> Dict:
    name = cfg.dataset["name"]
    base_seed = int(cfg.eval.get("seed", cfg.get("seed", 42)))

    scored = score_dataset(cfg, ckpt_path, logger=logger)
    scores = scored["test_scores"]
    labels = scored["test_labels"]
    val_scores = scored["val_scores"]

    normal = scores[labels == 0]
    anom = scores[labels == 1]

    auroc, auprc = auroc_auprc(scores, labels)
    pw = best_f1_search(scores, labels, adjust=False)
    ratio_thr = threshold_by_anomaly_ratio(scores, cfg.eval.get("anomaly_ratio", 0.05))
    pot_thr = pot_threshold(val_scores, scores,
                            q=cfg.eval.get("pot_q", 1e-3),
                            level=cfg.eval.get("pot_level", 0.98))

    pred = scores > pw.threshold
    tp = int(((pred == 1) & (labels == 1)).sum())
    fp = int(((pred == 1) & (labels == 0)).sum())
    fn = int(((pred == 0) & (labels == 1)).sum())

    report: Dict = {
        "dataset": name,
        "checkpoint": ckpt_path,
        "sic_enabled": bool(cfg.get("sic", {}).get("enabled", False)),
        "score_weights": {
            "denoise": float(cfg.get("score", {}).get("denoise_weight", 1.0)),
            "sic": float(cfg.get("score", {}).get("sic_weight", 0.0)),
        },
        "true_anomaly_ratio": float(labels.mean()),
        "auroc": auroc,
        "auprc": auprc,
        "normal_score": _dist_stats(normal),
        "anomaly_score": _dist_stats(anom),
        "separation": {
            "anom_mean_minus_normal_mean": float(anom.mean() - normal.mean()),
            "normal_std": float(normal.std()),
            **_overlap(normal, anom),
        },
        "best_f1_pointwise": {
            "threshold": float(pw.threshold),
            "precision": pw.precision,
            "recall": pw.recall,
            "f1": pw.f1,
            "predicted_anomaly_ratio": float(pred.mean()),
            "tp": tp, "fp": fp, "fn": fn,
        },
        "label_free_predicted_ratio": {
            "ratio_threshold": float((scores > ratio_thr).mean()),
            "pot_threshold": float((scores > pot_thr).mean()),
        },
        "boundary_effect": _boundary_effect(scores, cfg.data.get("window_size", 100)),
    }

    if "denoise_res" in scored["aux"]:
        report["per_channel_denoise"] = _per_channel(scored["aux"]["denoise_res"])
    if "sic_res" in scored["aux"]:
        report["per_channel_sic"] = _per_channel(scored["aux"]["sic_res"])

    # Reconstruction variance across repeated stochastic inference runs.
    if repeats > 1:
        runs = [scores]
        for r in range(1, repeats):
            cfg2 = load_config(name, config_dir=cfg.get("_config_dir", "configs"))
            # carry over the same overrides via the already-resolved cfg values
            for sec in ("data", "model", "diffusion", "train", "eval", "sic", "score", "dataset"):
                if sec in cfg:
                    cfg2[sec] = json.loads(json.dumps(cfg[sec]))
            cfg2.setdefault("eval", {})["seed"] = base_seed + r
            runs.append(score_dataset(cfg2, ckpt_path, logger=logger)["test_scores"])
        stack = np.stack(runs, axis=0)  # [repeats, length]
        run_std = stack.std(axis=0)     # [length]
        report["reconstruction_variance"] = {
            "repeats": repeats,
            "mean_std_normal": float(run_std[labels == 0].mean()),
            "mean_std_anomaly": float(run_std[labels == 1].mean()),
            "mean_std_overall": float(run_std.mean()),
        }

    return report


def parse_args():
    p = argparse.ArgumentParser(description="GDAD score diagnostics.")
    p.add_argument("--dataset", required=True)
    p.add_argument("--ckpt", default=None)
    p.add_argument("--config_dir", default="configs")
    p.add_argument("--override", nargs="*", default=[])
    p.add_argument("--repeats", type=int, default=1,
                   help=">1 measures score variance across stochastic runs.")
    p.add_argument("--out", default=None)
    return p.parse_args()


def main():
    args = parse_args()
    cfg = load_config(args.dataset, config_dir=args.config_dir, overrides=args.override)
    cfg["_config_dir"] = args.config_dir
    ckpt = args.ckpt or os.path.join(
        cfg.train.get("ckpt_dir", "checkpoints"), args.dataset, "best.pt"
    )
    logger = get_logger("gdad-diagnose", log_dir=os.path.join("logs", args.dataset))
    report = diagnose(cfg, ckpt, args.repeats, logger)

    out = args.out or os.path.join("results", args.dataset, "diagnosis.json")
    os.makedirs(os.path.dirname(out), exist_ok=True)
    with open(out, "w", encoding="utf-8") as fh:
        json.dump(report, fh, indent=2)
    logger.info(f"Diagnosis written to {out}")
    print(json.dumps(report, indent=2))


if __name__ == "__main__":
    main()
