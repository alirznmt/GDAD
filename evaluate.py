"""Evaluation entry point for GDAD.

Pipeline:
1. Load a trained checkpoint.
2. Compute per-(window, time, channel) denoising residuals on the validation
   (normal) split -> fit a per-channel residual normaliser and a label-free
   threshold source.
3. Compute residuals on the test split -> build a per-timestep anomaly score.
4. Report pointwise (primary) and point-adjusted P/R/F1, plus AUROC/AUPRC, under
   three thresholding strategies (oracle best-F1, anomaly-ratio, POT).

Usage:
    python evaluate.py --dataset SMAP --ckpt checkpoints/SMAP/best.pt
"""
from __future__ import annotations

import argparse
import json
import os
from typing import Dict, List

import numpy as np
import torch

from models import assert_effective_graph_mode, build_model
from utils.config import load_config
from utils.data import build_dataloaders, build_val_loader_for_scoring
from utils.logger import get_logger
from utils.metrics import evaluate_all,debug_adjusted_evaluation
from utils.provenance import (
    assert_matching_provenance,
    collect_data_provenance,
    short_provenance,
)
from utils.scoring import (
    ChannelNormalizer,
    LevelChannelNormalizer,
    aggregate_levels,
    build_timeline_score,
)
from utils.seed import set_seed


@torch.no_grad()
def compute_residuals(model, loader, device, cfg) -> np.ndarray:
    """Return combined residuals for every window: ``[num_windows, W, N]``."""
    model.eval()
    eval_steps: List[int] = list(cfg.eval.get("eval_steps", [10, 25, 50]))
    num_samples = cfg.eval.get("num_samples", 1)
    recon_w = cfg.eval.get("recon_weight", 1.0)
    eps_w = cfg.eval.get("eps_weight", 0.0)

    chunks = []
    for batch in loader:
        x = (batch[0] if isinstance(batch, (list, tuple)) else batch).to(device)
        res = model.score_window(
            x, eval_steps, num_samples=num_samples,
            recon_weight=recon_w, eps_weight=eps_w,
        )
        chunks.append(res["combined"].cpu().numpy())
    return np.concatenate(chunks, axis=0)  # [num_windows, W, N]


@torch.no_grad()
def compute_eps_residuals_by_level(model, loader, device, cfg) -> np.ndarray:
    """Return epsilon residuals shaped ``[num_windows, levels, W, N]``."""
    model.eval()
    eval_steps: List[int] = list(cfg.eval.get("eval_steps", [10, 25, 50]))
    num_samples = int(cfg.eval.get("num_samples", 1))

    chunks = []
    for batch in loader:
        x = (batch[0] if isinstance(batch, (list, tuple)) else batch).to(device)
        # Query one level at a time.  This is algebraically identical to the
        # multi-level call, preserves the same number/order of noise draws, and
        # remains compatible with older GDAD checkpoints/source trees whose
        # ``denoising_residual`` returned only aggregate ``eps`` residuals.
        level_chunks = []
        for step in eval_steps:
            res = model.score_window(
                x,
                [int(step)],
                num_samples=num_samples,
                recon_weight=0.0,
                eps_weight=1.0,
            )
            if "eps_by_level" in res:
                eps_level = res["eps_by_level"][:, 0]
            elif "eps" in res:
                eps_level = res["eps"]
            else:
                raise KeyError(
                    "model.score_window() returned neither 'eps_by_level' nor "
                    "the legacy 'eps' residual"
                )
            level_chunks.append(eps_level)
        chunks.append(torch.stack(level_chunks, dim=1).cpu().numpy())
    return np.concatenate(chunks, axis=0)


@torch.no_grad()
def compute_sic_residuals(model, loader, device, cfg) -> np.ndarray:
    """SIC reconstruction residuals for every window: ``[num_windows, W, N]``.

    Inference-only add-on (Candidate B). Requires no retraining and is only
    invoked when ``sic.enabled`` is set in the config.
    """
    from models.sic import SICSampler, build_sic_config

    model.eval()
    sampler = SICSampler(model.diffusion, build_sic_config(cfg))
    chunks = []
    for batch in loader:
        x = (batch[0] if isinstance(batch, (list, tuple)) else batch).to(device)
        res = sampler.residual(x)
        chunks.append(res.cpu().numpy())
    return np.concatenate(chunks, axis=0)


def load_model_from_ckpt(cfg, ckpt_path: str, device, data_num_nodes: int):
    ckpt = torch.load(ckpt_path, map_location=device)
    num_nodes = ckpt["num_nodes"]
    if int(num_nodes) != int(data_num_nodes):
        raise ValueError(
            f"Checkpoint expects {num_nodes} sensors but the current dataset "
            f"has {data_num_nodes}. Refusing an incompatible evaluation."
        )

    current_provenance = collect_data_provenance(cfg)
    saved_provenance = ckpt.get("data_provenance")
    if saved_provenance is not None:
        assert_matching_provenance(saved_provenance, current_provenance)
    model = build_model(cfg, num_nodes=num_nodes).to(device)
    assert_effective_graph_mode(model, cfg.model.get("graph_mode"))
    model.load_state_dict(ckpt["model"])
    return model, num_nodes, current_provenance


def score_dataset(cfg, ckpt_path: str, logger=None) -> Dict:
    """Compute timeline anomaly scores for one dataset (shared by eval + diagnose).

    Returns a dict with ``test_scores``, ``val_scores`` (both ``[length]``),
    ``test_labels`` (``[test_len]``), and the raw per-window residual components
    (``denoise_res``, and ``sic_res`` when SIC is enabled) for per-channel
    analysis. Honors all scoring flags (``sic.*``, ``score.*``).
    """
    name = cfg.dataset["name"]
    eval_seed = int(cfg.eval.get("seed", cfg.get("seed", 42)))
    set_seed(eval_seed, deterministic=cfg.train.get("deterministic", True))
    if logger is None:
        logger = get_logger("gdad", log_dir=os.path.join("logs", name))

    device = torch.device(
        "cuda" if (torch.cuda.is_available() and cfg.train.get("use_cuda", True)) else "cpu"
    )

    _, training_val_loader, test_loader, test_ds, data_num_nodes = build_dataloaders(cfg)
    val_loader = build_val_loader_for_scoring(
        cfg, val_dataset=training_val_loader.dataset
    )

    model, num_nodes, data_provenance = load_model_from_ckpt(
        cfg, ckpt_path, device, data_num_nodes
    )
    logger.info(f"[{name}] loaded checkpoint {ckpt_path} (sensors={num_nodes})")
    logger.info(f"[{name}] data provenance: {short_provenance(data_provenance)}")

    window = cfg.data.get("window_size", 96)
    channel_agg = cfg.eval.get("channel_agg", "mean")
    ema_alpha = cfg.eval.get("ema_alpha", 0.0)
    use_channel_norm = cfg.eval.get("channel_norm", True)
    scoring_mode = str(cfg.eval.get("scoring_mode", "original")).lower()
    logger.info(f"[{name}] scoring_mode={scoring_mode}")

    norm_method = cfg.get("score", {}).get("normalization", "zscore")
    sic_enabled = bool(cfg.get("sic", {}).get("enabled", False))
    w_denoise = float(cfg.get("score", {}).get("denoise_weight", 1.0))
    w_sic = float(cfg.get("score", {}).get("sic_weight", 0.0))

    test_len = int(test_ds.data.shape[0])
    val_ds = val_loader.dataset
    val_len = int(val_ds.data.shape[0])

    def _timeline(res, ds, length, norm):
        return build_timeline_score(
            res, ds.start_indices, window, length,
            normalizer=norm, channel_agg=channel_agg, ema_alpha=ema_alpha,
            segment_end_indices=getattr(ds, "segment_end_indices", None),
            segment_lengths=getattr(ds, "segment_lengths", None),
        )

    # --- denoising component -------------------------------------------------
    if scoring_mode == "original":
        # Model-faithful path: aggregate levels first, then calibrate sensors.
        val_res = compute_residuals(model, val_loader, device, cfg)
        denoise_norm = (
            ChannelNormalizer.fit(val_res, method=norm_method)
            if use_channel_norm
            else None
        )
        test_res = compute_residuals(model, test_loader, device, cfg)
        denoise_test_tl = _timeline(test_res, test_ds, test_len, denoise_norm)
        denoise_val_tl = _timeline(val_res, val_ds, val_len, denoise_norm)
        scoring_metadata = {"mode": "original"}
    elif scoring_mode == "calibrated_eps":
        # Scoring-only research path: calibrate each (level, sensor) on normal
        # validation residuals before aggregating levels.
        level_norm_method = str(cfg.eval.get("level_norm", "robust"))
        level_agg = str(cfg.eval.get("level_agg", "mean"))
        level_clip_min = cfg.eval.get("level_clip_min", 0.0)

        val_levels = compute_eps_residuals_by_level(model, val_loader, device, cfg)
        level_normalizer = LevelChannelNormalizer.fit(
            val_levels,
            method=level_norm_method,
        )
        val_calibrated = level_normalizer.transform(
            val_levels,
            clip_min=level_clip_min,
        )
        val_res = aggregate_levels(val_calibrated, mode=level_agg)
        del val_calibrated, val_levels

        test_levels = compute_eps_residuals_by_level(model, test_loader, device, cfg)
        test_calibrated = level_normalizer.transform(
            test_levels,
            clip_min=level_clip_min,
        )
        test_res = aggregate_levels(test_calibrated, mode=level_agg)
        del test_calibrated, test_levels

        denoise_test_tl = _timeline(test_res, test_ds, test_len, None)
        denoise_val_tl = _timeline(val_res, val_ds, val_len, None)
        scoring_metadata = {
            "mode": "calibrated_eps",
            "level_norm": level_norm_method,
            "level_agg": level_agg,
            "level_clip_min": level_clip_min,
            "eval_steps": list(cfg.eval.get("eval_steps", [])),
        }
    else:
        raise ValueError(
            f"Unknown eval.scoring_mode '{scoring_mode}'. Expected "
            "'original' or 'calibrated_eps'."
        )

    aux: Dict = {"denoise_res": test_res}

    # --- optional SIC component (Candidate B), fused on normalized timelines ---
    if sic_enabled and w_sic != 0.0:
        logger.info(f"[{name}] SIC enabled (w_denoise={w_denoise}, w_sic={w_sic})")
        val_sic = compute_sic_residuals(model, val_loader, device, cfg)
        test_sic = compute_sic_residuals(model, test_loader, device, cfg)
        sic_norm = (
            ChannelNormalizer.fit(val_sic, method=norm_method) if use_channel_norm else None
        )
        sic_test_tl = _timeline(test_sic, test_ds, test_len, sic_norm)
        sic_val_tl = _timeline(val_sic, val_ds, val_len, sic_norm)
        test_scores = w_denoise * denoise_test_tl + w_sic * sic_test_tl
        val_scores = w_denoise * denoise_val_tl + w_sic * sic_val_tl
        aux["sic_res"] = test_sic
    else:
        test_scores = denoise_test_tl
        val_scores = denoise_val_tl

    test_labels = test_ds.labels.numpy()[:test_len].astype(int)
    return {
        "test_scores": test_scores,
        "val_scores": val_scores,
        "test_labels": test_labels,
        "test_len": test_len,
        "data_provenance": data_provenance,
        "scoring_metadata": scoring_metadata,
        "aux": aux,
    }


def evaluate_one_dataset(cfg, ckpt_path: str, logger=None) -> Dict:
    name = cfg.dataset["name"]
    if logger is None:
        logger = get_logger("gdad", log_dir=os.path.join("logs", name))

    scored = score_dataset(cfg, ckpt_path, logger=logger)
    test_scores = scored["test_scores"]
    val_scores = scored["val_scores"]
    test_labels = scored["test_labels"]
    test_len = scored["test_len"]

    anomaly_ratio = cfg.eval.get("anomaly_ratio", 0.05)
    pot_q = cfg.eval.get("pot_q", 1e-3)
    pot_level = cfg.eval.get("pot_level", 0.98)

    results = evaluate_all(
        test_scores=test_scores,
        test_labels=test_labels,
        val_scores=val_scores,
        anomaly_ratio=anomaly_ratio,
        pot_q=pot_q,
        pot_level=pot_level,
        debug=False,

    )

    results["dataset"] = name
    results["num_anomalies"] = int(test_labels.sum())
    results["test_len"] = test_len
    results["data_provenance"] = scored["data_provenance"]
    results["scoring"] = scored["scoring_metadata"]

    pw = results["pointwise"]["best_f1"]
    adj = results["adjusted"]["best_f1"]

    logger.info(
        f"[{name}] POINTWISE best-F1: "
        f"P={pw['precision']:.4f} "
        f"R={pw['recall']:.4f} "
        f"F1={pw['f1']:.4f} "
        f"PW-ADD={pw['raw_add']:.4f}"
    )

    logger.info(
        f"[{name}] ADJUSTED best-F1: "
        f"P={adj['precision']:.4f} "
        f"R={adj['recall']:.4f} "
        f"F1={adj['f1']:.4f} "
        f"PA-ADD={adj['adjusted_add']:.4f} "
        f"RAW-ADD@PA-THR={adj['raw_add']:.4f}"
    )

    logger.info(
        f"[{name}] RANGE/VUS: "
        f"R-AUC-ROC={results['rauc_roc']:.4f} "
        f"R-AUC-PR={results['rauc_pr']:.4f} "
        f"VUS-ROC={results['vus_roc']:.4f} "
        f"VUS-PR={results['vus_pr']:.4f}"
    )

    out_dir = os.path.join(cfg.eval.get("result_dir", "results"), name)
    os.makedirs(out_dir, exist_ok=True)
    with open(os.path.join(out_dir, "metrics.json"), "w", encoding="utf-8") as fh:
        json.dump(results, fh, indent=2)
    np.save(os.path.join(out_dir, "test_scores.npy"), test_scores)

    return results


def parse_args():
    parser = argparse.ArgumentParser(description="Evaluate GDAD on a dataset.")
    parser.add_argument("--dataset", required=True, type=str)
    parser.add_argument("--ckpt", default=None, type=str)
    parser.add_argument("--config_dir", default="configs", type=str)
    parser.add_argument("--override", nargs="*", default=[])
    return parser.parse_args()


def main():
    args = parse_args()
    cfg = load_config(args.dataset, config_dir=args.config_dir, overrides=args.override)
    ckpt = args.ckpt or os.path.join(
        cfg.train.get("ckpt_dir", "checkpoints"), args.dataset, "best.pt"
    )
    logger = get_logger("gdad", log_dir=os.path.join("logs", args.dataset))
    evaluate_one_dataset(cfg, ckpt, logger=logger)



if __name__ == "__main__":

    main()
