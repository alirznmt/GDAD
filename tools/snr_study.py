"""Train and analyze the GDAD graph-by-SNR study.

This script deliberately keeps the original architecture and training loss.
It adds controlled graph modes and scoring-only analyses:

* identity: no cross-sensor mixing, parameter-count matched;
* static: learned sparse directed affinity only;
* dynamic: corruption-instance-dependent dense affinity only;
* hybrid: the original learned global mixture.

The analysis evaluates five controlled scoring paths from one shared residual
field: configured original, configured raw epsilon, configured calibrated
epsilon, study-grid raw epsilon, and study-grid calibrated epsilon.  It also
records calibrated single-SNR responses and graph-by-SNR interactions.
"""
from __future__ import annotations

import argparse
import gc
import json
import math
import os
import sys
import time
from datetime import datetime, timezone
from typing import Any, Dict, Iterable, List, Sequence, Tuple

import numpy as np
import pandas as pd
import torch

PROJECT_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
if PROJECT_ROOT not in sys.path:
    sys.path.insert(0, PROJECT_ROOT)

from evaluate import compute_eps_residuals_by_level, load_model_from_ckpt
from train import train_one_dataset
from utils.config import Config, load_config
from utils.data import build_dataloaders, build_val_loader_for_scoring
from utils.logger import get_logger
from utils.metrics import (
    add_values_at_threshold,
    auroc_auprc,
    best_f1_search,
    compute_range_metrics,
    pot_threshold,
    precision_recall_f1,
)
from utils.scoring import (
    ChannelNormalizer,
    LevelChannelNormalizer,
    aggregate_levels,
    build_timeline_score,
)
from utils.seed import set_seed


GRAPH_MODES = ("identity", "static", "dynamic", "hybrid")
DEFAULT_DATASETS = ("SMAP", "MSL", "PSM", "SWaT")
SUPPORTED_DATASETS = (*DEFAULT_DATASETS, "HAI", "WADI")
AGGREGATE_STEP = -1


def _checkpoint_path(study_dir: str, dataset: str, mode: str, seed: int) -> str:
    return os.path.join(
        study_dir,
        "checkpoints",
        mode,
        f"seed_{seed}",
        dataset,
        "best.pt",
    )


def _json_dump(path: str, value: Any) -> None:
    os.makedirs(os.path.dirname(path), exist_ok=True)
    with open(path, "w", encoding="utf-8") as handle:
        json.dump(value, handle, indent=2, allow_nan=True)


def _device(allow_cpu: bool) -> torch.device:
    if torch.cuda.is_available():
        return torch.device("cuda")
    if allow_cpu:
        return torch.device("cpu")
    raise RuntimeError(
        "CUDA is unavailable. These runs are expensive on CPU; rerun with "
        "--allow_cpu only if CPU execution is intentional."
    )


def _study_overrides(
    base: Sequence[str],
    study_dir: str,
    dataset: str,
    mode: str,
    seed: int,
    data_root: str | None,
) -> List[str]:
    ckpt_root = os.path.join(study_dir, "checkpoints", mode, f"seed_{seed}")
    overrides = [
        *base,
        f"seed={seed}",
        f"model.graph_mode={mode}",
        f"train.ckpt_dir={ckpt_root}",
        "diffusion.scheduler_type=original",
        "sic.enabled=false",
        "score.sic_weight=0.0",
    ]
    if data_root:
        overrides.append(
            f"dataset.data_root={os.path.join(os.path.abspath(data_root), dataset)}"
        )
    return overrides


def _checkpoint_manifest_record(
    checkpoint_path: str,
    dataset: str,
    mode: str,
    seed: int,
    status: str,
    training_seconds: float | None = None,
) -> Dict[str, Any]:
    checkpoint = torch.load(checkpoint_path, map_location="cpu")
    resolved = checkpoint.get("config", {})
    record: Dict[str, Any] = {
        "dataset": dataset,
        "graph_mode": mode,
        "seed": seed,
        "status": status,
        "checkpoint": os.path.abspath(checkpoint_path),
        "checkpoint_epoch": checkpoint.get("epoch"),
        "checkpoint_val_loss": checkpoint.get("val_loss"),
        "num_nodes": checkpoint.get("num_nodes"),
        "resolved_config": {
            key: resolved.get(key)
            for key in (
                "seed",
                "data",
                "model",
                "diffusion",
                "train",
                "eval",
                "sic",
                "score",
            )
            if key in resolved
        },
        "data_provenance": checkpoint.get("data_provenance"),
        "recorded_utc": datetime.now(timezone.utc).isoformat(),
    }
    if training_seconds is not None:
        record["training_seconds"] = float(training_seconds)
    return record


def _upsert_manifest(
    manifest: List[Dict[str, Any]],
    record: Dict[str, Any],
) -> List[Dict[str, Any]]:
    result = [
        item
        for item in manifest
        if not (
            item.get("dataset") == record["dataset"]
            and item.get("graph_mode") == record["graph_mode"]
            and int(item.get("seed", -1)) == int(record["seed"])
        )
    ]
    result.append(record)
    return result


def train_variants(args: argparse.Namespace) -> None:
    _device(args.allow_cpu)
    os.makedirs(args.study_dir, exist_ok=True)
    manifest_path = os.path.join(args.study_dir, "training_manifest.json")
    manifest: List[Dict[str, Any]] = []
    if os.path.isfile(manifest_path):
        with open(manifest_path, "r", encoding="utf-8") as handle:
            manifest = json.load(handle)

    for dataset in args.datasets:
        for mode in args.graph_modes:
            for seed in args.seeds:
                expected_path = _checkpoint_path(
                    args.study_dir, dataset, mode, seed
                )
                if os.path.isfile(expected_path) and not args.force:
                    print(f"SKIP existing checkpoint: {expected_path}")
                    record = _checkpoint_manifest_record(
                        expected_path,
                        dataset,
                        mode,
                        seed,
                        status="existing_checkpoint",
                    )
                    manifest = _upsert_manifest(manifest, record)
                    _json_dump(manifest_path, manifest)
                    continue

                overrides = _study_overrides(
                    args.override,
                    args.study_dir,
                    dataset,
                    mode,
                    seed,
                    args.data_root,
                )
                cfg = load_config(
                    dataset,
                    config_dir=args.config_dir,
                    overrides=overrides,
                )
                cfg.train["use_cuda"] = not args.allow_cpu or torch.cuda.is_available()
                log_dir = os.path.join(
                    args.study_dir,
                    "logs",
                    dataset,
                    mode,
                    f"seed_{seed}",
                )
                logger = get_logger(
                    f"snr-study-{dataset}-{mode}-{seed}",
                    log_dir=log_dir,
                )
                logger.info(
                    f"SNR study training | dataset={dataset} | "
                    f"graph_mode={mode} | seed={seed}"
                )

                started = time.perf_counter()
                checkpoint = train_one_dataset(cfg, logger=logger)
                elapsed = time.perf_counter() - started
                record = _checkpoint_manifest_record(
                    checkpoint,
                    dataset,
                    mode,
                    seed,
                    status="trained",
                    training_seconds=elapsed,
                )
                manifest = _upsert_manifest(manifest, record)
                _json_dump(manifest_path, manifest)
                logger.info(f"Completed study checkpoint: {checkpoint}")


def _config_from_checkpoint(path: str) -> Config:
    checkpoint = torch.load(path, map_location="cpu")
    raw = checkpoint.get("config")
    if raw is None:
        raise ValueError(f"Checkpoint has no resolved config: {path}")
    return Config(raw)


def _steps_near_target_snr(
    model,
    target_snr_db: Sequence[float],
) -> Tuple[List[int], List[float], List[float]]:
    alpha_bar = model.diffusion.alphas_cumprod.detach().cpu().numpy()
    snr = alpha_bar / np.maximum(1.0 - alpha_bar, 1e-20)
    snr_db = 10.0 * np.log10(np.maximum(snr, 1e-20))

    selected: List[Tuple[int, float, float]] = []
    seen = set()
    for target in target_snr_db:
        step = int(np.argmin(np.abs(snr_db - float(target))))
        if step in seen:
            continue
        seen.add(step)
        selected.append((step, float(target), float(snr_db[step])))

    # Present the response curve from high to low actual SNR.
    selected.sort(key=lambda item: item[2], reverse=True)
    return (
        [item[0] for item in selected],
        [item[1] for item in selected],
        [item[2] for item in selected],
    )


def _snr_db_for_steps(model, steps: Sequence[int]) -> List[float]:
    """Return the actual schedule SNR (dB) for explicit integer steps."""
    alpha_bar = model.diffusion.alphas_cumprod.detach().cpu().numpy()
    values: List[float] = []
    for raw_step in steps:
        step = int(raw_step)
        if step < 0 or step >= len(alpha_bar):
            raise ValueError(
                f"Configured evaluation step {step} is outside "
                f"[0, {len(alpha_bar) - 1}]"
            )
        snr = float(alpha_bar[step]) / max(1.0 - float(alpha_bar[step]), 1e-20)
        values.append(float(10.0 * np.log10(max(snr, 1e-20))))
    return values


def _unique_steps(steps: Sequence[int]) -> List[int]:
    """Preserve order while rejecting an empty evaluation-step collection."""
    result: List[int] = []
    seen = set()
    for raw_step in steps:
        step = int(raw_step)
        if step not in seen:
            seen.add(step)
            result.append(step)
    if not result:
        raise ValueError("At least one configured evaluation step is required")
    return result


def _timeline(
    residuals: np.ndarray,
    dataset,
    length: int,
    window: int,
    channel_agg: str,
    ema_alpha: float,
    normalizer: ChannelNormalizer | None = None,
) -> np.ndarray:
    return build_timeline_score(
        residuals,
        dataset.start_indices,
        window,
        length,
        normalizer=normalizer,
        channel_agg=channel_agg,
        ema_alpha=ema_alpha,
        segment_end_indices=getattr(dataset, "segment_end_indices", None),
        segment_lengths=getattr(dataset, "segment_lengths", None),
    )


def _calibrated_timelines(
    val_levels: np.ndarray,
    test_levels: np.ndarray,
    val_dataset,
    test_dataset,
    window: int,
    channel_agg: str,
    ema_alpha: float,
    normalization: str,
    level_agg: str,
    clip_min: float | None,
) -> Tuple[List[np.ndarray], List[np.ndarray], np.ndarray, np.ndarray]:
    normalizer = LevelChannelNormalizer.fit(
        val_levels,
        method=normalization,
    )
    val_calibrated = normalizer.transform(val_levels, clip_min=clip_min)
    test_calibrated = normalizer.transform(test_levels, clip_min=clip_min)

    val_len = int(val_dataset.data.shape[0])
    test_len = int(test_dataset.data.shape[0])
    val_single: List[np.ndarray] = []
    test_single: List[np.ndarray] = []
    for level in range(val_calibrated.shape[1]):
        val_single.append(
            _timeline(
                val_calibrated[:, level],
                val_dataset,
                val_len,
                window,
                channel_agg,
                ema_alpha,
            )
        )
        test_single.append(
            _timeline(
                test_calibrated[:, level],
                test_dataset,
                test_len,
                window,
                channel_agg,
                ema_alpha,
            )
        )

    val_multi_res = aggregate_levels(val_calibrated, mode=level_agg)
    test_multi_res = aggregate_levels(test_calibrated, mode=level_agg)
    val_multi = _timeline(
        val_multi_res,
        val_dataset,
        val_len,
        window,
        channel_agg,
        ema_alpha,
    )
    test_multi = _timeline(
        test_multi_res,
        test_dataset,
        test_len,
        window,
        channel_agg,
        ema_alpha,
    )
    return val_single, test_single, val_multi, test_multi


def _aggregate_then_sensor_calibrate(
    val_levels: np.ndarray,
    test_levels: np.ndarray,
    coefficients: np.ndarray,
    val_dataset,
    test_dataset,
    window: int,
    channel_agg: str,
    ema_alpha: float,
    normalization: str | None,
) -> Tuple[np.ndarray, np.ndarray]:
    coefficients = np.asarray(coefficients, dtype=np.float32)
    val_res = (
        val_levels * coefficients[None, :, None, None]
    ).mean(axis=1)
    test_res = (
        test_levels * coefficients[None, :, None, None]
    ).mean(axis=1)
    normalizer = (
        ChannelNormalizer.fit(val_res, method=normalization)
        if normalization is not None
        else None
    )
    val_scores = _timeline(
        val_res,
        val_dataset,
        int(val_dataset.data.shape[0]),
        window,
        channel_agg,
        ema_alpha,
        normalizer=normalizer,
    )
    test_scores = _timeline(
        test_res,
        test_dataset,
        int(test_dataset.data.shape[0]),
        window,
        channel_agg,
        ema_alpha,
        normalizer=normalizer,
    )
    return val_scores, test_scores


def _metrics(
    val_scores: np.ndarray,
    test_scores: np.ndarray,
    labels: np.ndarray,
    cfg,
    include_range: bool,
) -> Dict[str, float]:
    auroc, auprc = auroc_auprc(test_scores, labels)
    threshold = pot_threshold(
        val_scores,
        test_scores,
        q=float(cfg.eval.get("pot_q", 1e-3)),
        level=float(cfg.eval.get("pot_level", 0.98)),
    )
    pot = precision_recall_f1(test_scores, labels, threshold, adjust=False)
    pot_add = add_values_at_threshold(test_scores, labels, threshold)
    oracle = best_f1_search(test_scores, labels, adjust=False)
    oracle_add = add_values_at_threshold(test_scores, labels, oracle.threshold)
    range_metrics = (
        compute_range_metrics(labels, test_scores)
        if include_range
        else {
            "rauc_roc": float("nan"),
            "rauc_pr": float("nan"),
            "vus_roc": float("nan"),
            "vus_pr": float("nan"),
        }
    )
    return {
        "auroc": auroc,
        "auprc": auprc,
        "pot_threshold": threshold,
        "pot_precision": pot.precision,
        "pot_recall": pot.recall,
        "pot_f1": pot.f1,
        "pot_raw_add": pot_add["raw_add"],
        "oracle_threshold": oracle.threshold,
        "oracle_precision": oracle.precision,
        "oracle_recall": oracle.recall,
        "oracle_f1": oracle.f1,
        "oracle_raw_add": oracle_add["raw_add"],
        **range_metrics,
    }


def _base_row(
    dataset: str,
    mode: str,
    seed: int,
    method: str,
    gate: float,
    parameters: int,
    forward_seconds: float,
    forwards_per_window: int,
    analysis_forwards_per_window: int,
    peak_cuda_memory_mb: float,
) -> Dict[str, Any]:
    return {
        "dataset": dataset,
        "graph_mode": mode,
        "seed": seed,
        "method": method,
        "step": AGGREGATE_STEP,
        "target_snr_db": float("nan"),
        "actual_snr_db": float("nan"),
        "graph_gate": gate,
        "parameters": parameters,
        "forward_seconds": forward_seconds,
        "forwards_per_window": forwards_per_window,
        "analysis_forwards_per_window": analysis_forwards_per_window,
        "peak_cuda_memory_mb": peak_cuda_memory_mb,
        "estimated_method_forward_seconds": (
            forward_seconds
            * float(forwards_per_window)
            / max(float(analysis_forwards_per_window), 1.0)
        ),
    }


def _analyze_one(
    args: argparse.Namespace,
    dataset: str,
    mode: str,
    seed: int,
    device: torch.device,
) -> Tuple[List[Dict[str, Any]], Dict[str, Any]]:
    checkpoint_path = _checkpoint_path(args.study_dir, dataset, mode, seed)
    if not os.path.isfile(checkpoint_path):
        raise FileNotFoundError(
            f"Missing checkpoint for dataset={dataset}, mode={mode}, seed={seed}: "
            f"{checkpoint_path}"
        )

    cfg = _config_from_checkpoint(checkpoint_path)
    configured_steps = _unique_steps(
        args.configured_steps
        if args.configured_steps is not None
        else cfg.eval.get("eval_steps", [10, 25, 50])
    )
    num_samples = int(
        cfg.eval.get("num_samples", 1)
        if args.num_samples is None
        else args.num_samples
    )
    if num_samples < 1:
        raise ValueError("num_samples must be at least one")
    ema_alpha = float(
        cfg.eval.get("ema_alpha", 0.0)
        if args.ema_alpha is None
        else args.ema_alpha
    )
    channel_agg = str(
        cfg.eval.get("channel_agg", "mean")
        if args.channel_agg is None
        else args.channel_agg
    )
    checkpoint_normalization = str(
        cfg.get("score", {}).get("normalization", "zscore")
    ).lower()
    level_normalization = (
        checkpoint_normalization
        if args.normalization == "checkpoint"
        else str(args.normalization)
    )
    original_sensor_normalization = (
        checkpoint_normalization
        if bool(cfg.eval.get("channel_norm", True))
        else None
    )
    if args.data_root:
        cfg.dataset["data_root"] = os.path.join(
            os.path.abspath(args.data_root),
            dataset,
        )
    cfg.train["use_cuda"] = device.type == "cuda"
    cfg.eval["num_samples"] = num_samples
    cfg.eval["ema_alpha"] = ema_alpha
    cfg.eval["channel_agg"] = channel_agg

    _, training_val_loader, test_loader, test_dataset, data_num_nodes = build_dataloaders(cfg)
    val_loader = build_val_loader_for_scoring(
        cfg, val_dataset=training_val_loader.dataset
    )
    model, num_nodes, provenance = load_model_from_ckpt(
        cfg,
        checkpoint_path,
        device,
        data_num_nodes,
    )
    study_steps, targets, study_actual_snr_db = _steps_near_target_snr(
        model,
        args.target_snr_db,
    )
    configured_actual_snr_db = _snr_db_for_steps(model, configured_steps)
    union_steps = _unique_steps([*configured_steps, *study_steps])
    _snr_db_for_steps(model, union_steps)  # validates all selected steps
    cfg.eval["eval_steps"] = union_steps

    # Reset evaluation randomness for common noise draws across graph modes.
    set_seed(seed + int(args.eval_seed_offset), deterministic=True)
    if device.type == "cuda":
        torch.cuda.reset_peak_memory_stats(device)
        torch.cuda.synchronize()
    started = time.perf_counter()
    val_union = compute_eps_residuals_by_level(model, val_loader, device, cfg)
    test_union = compute_eps_residuals_by_level(model, test_loader, device, cfg)
    if device.type == "cuda":
        torch.cuda.synchronize()
    forward_seconds = time.perf_counter() - started
    peak_cuda_memory_mb = (
        float(torch.cuda.max_memory_allocated(device)) / (1024.0 ** 2)
        if device.type == "cuda"
        else float("nan")
    )

    window = int(cfg.data.get("window_size", 96))
    labels = test_dataset.labels.numpy()[: int(test_dataset.data.shape[0])].astype(int)
    union_index = {step: index for index, step in enumerate(union_steps)}
    configured_indices = [union_index[step] for step in configured_steps]
    study_indices = [union_index[step] for step in study_steps]

    # ------------------------------------------------------------------
    # Three configured-step paths isolate residual/calibration changes while
    # preserving the checkpoint's original evaluation levels and score knobs.
    val_configured = val_union[:, configured_indices]
    test_configured = test_union[:, configured_indices]
    _, _, val_configured_cal, test_configured_cal = _calibrated_timelines(
        val_configured,
        test_configured,
        val_loader.dataset,
        test_dataset,
        window,
        channel_agg,
        ema_alpha,
        level_normalization,
        args.level_agg,
        args.clip_min,
    )
    configured_ones = np.ones(len(configured_steps), dtype=np.float32)
    val_configured_raw, test_configured_raw = _aggregate_then_sensor_calibrate(
        val_configured,
        test_configured,
        configured_ones,
        val_loader.dataset,
        test_dataset,
        window,
        channel_agg,
        ema_alpha,
        original_sensor_normalization,
    )
    configured_alpha_bar = (
        model.diffusion.alphas_cumprod.detach().cpu().numpy()[configured_steps]
    )
    configured_inverse_snr = (1.0 - configured_alpha_bar) / np.maximum(
        configured_alpha_bar,
        1e-20,
    )
    configured_coefficients = (
        float(cfg.eval.get("recon_weight", 1.0)) * configured_inverse_snr
        + float(cfg.eval.get("eps_weight", 0.0))
    )
    val_configured_original, test_configured_original = (
        _aggregate_then_sensor_calibrate(
            val_configured,
            test_configured,
            configured_coefficients,
            val_loader.dataset,
            test_dataset,
            window,
            channel_agg,
            ema_alpha,
            original_sensor_normalization,
        )
    )
    del val_configured, test_configured

    # ------------------------------------------------------------------
    # Six-level study paths use exactly the same stored residual field.  The
    # only difference between them is aggregate-first sensor calibration versus
    # level/sensor calibration before aggregation.
    val_study = val_union[:, study_indices]
    test_study = test_union[:, study_indices]
    val_single, test_single, val_study_cal, test_study_cal = _calibrated_timelines(
        val_study,
        test_study,
        val_loader.dataset,
        test_dataset,
        window,
        channel_agg,
        ema_alpha,
        level_normalization,
        args.level_agg,
        args.clip_min,
    )
    study_ones = np.ones(len(study_steps), dtype=np.float32)
    val_study_raw, test_study_raw = _aggregate_then_sensor_calibrate(
        val_study,
        test_study,
        study_ones,
        val_loader.dataset,
        test_dataset,
        window,
        channel_agg,
        ema_alpha,
        original_sensor_normalization,
    )
    del val_study, test_study

    learner = model.diffusion.model.graph_learner
    gate = float(torch.sigmoid(learner.alpha).detach().cpu())
    parameters = sum(parameter.numel() for parameter in model.parameters())
    analysis_forwards_per_window = len(union_steps) * num_samples
    rows: List[Dict[str, Any]] = []

    for index, (step, target, actual) in enumerate(
        zip(study_steps, targets, study_actual_snr_db)
    ):
        row = _base_row(
            dataset,
            mode,
            seed,
            "study_single_calibrated",
            gate,
            parameters,
            forward_seconds,
            num_samples,
            analysis_forwards_per_window,
            peak_cuda_memory_mb,
        )
        row.update(
            {
                "step": step,
                "target_snr_db": target,
                "actual_snr_db": actual,
                **_metrics(
                    val_single[index],
                    test_single[index],
                    labels,
                    cfg,
                    args.include_range,
                ),
            }
        )
        rows.append(row)

    aggregate_methods = (
        (
            "configured_original",
            val_configured_original,
            test_configured_original,
            len(configured_steps),
        ),
        (
            "configured_raw_eps",
            val_configured_raw,
            test_configured_raw,
            len(configured_steps),
        ),
        (
            "configured_calibrated_eps",
            val_configured_cal,
            test_configured_cal,
            len(configured_steps),
        ),
        (
            "study_raw_eps",
            val_study_raw,
            test_study_raw,
            len(study_steps),
        ),
        (
            "study_calibrated_eps",
            val_study_cal,
            test_study_cal,
            len(study_steps),
        ),
    )
    for method, val_scores, test_scores, method_levels in aggregate_methods:
        row = _base_row(
            dataset,
            mode,
            seed,
            method,
            gate,
            parameters,
            forward_seconds,
            method_levels * num_samples,
            analysis_forwards_per_window,
            peak_cuda_memory_mb,
        )
        row.update(
            _metrics(
                val_scores,
                test_scores,
                labels,
                cfg,
                args.include_range,
            )
        )
        rows.append(row)

    metadata = {
        "dataset": dataset,
        "graph_mode": mode,
        "seed": seed,
        "checkpoint": os.path.abspath(checkpoint_path),
        "sensors": num_nodes,
        "configured_steps": configured_steps,
        "configured_actual_snr_db": configured_actual_snr_db,
        "study_steps": study_steps,
        "target_snr_db": targets,
        "study_actual_snr_db": study_actual_snr_db,
        "union_steps": union_steps,
        "num_samples": num_samples,
        "level_normalization": level_normalization,
        "original_sensor_normalization": original_sensor_normalization,
        "level_agg": args.level_agg,
        "clip_min": args.clip_min,
        "channel_agg": channel_agg,
        "ema_alpha": ema_alpha,
        "recon_weight": float(cfg.eval.get("recon_weight", 1.0)),
        "eps_weight": float(cfg.eval.get("eps_weight", 0.0)),
        "channel_norm": bool(cfg.eval.get("channel_norm", True)),
        "analysis_forwards_per_window": analysis_forwards_per_window,
        "data_provenance": provenance,
        "forward_seconds": forward_seconds,
        "peak_cuda_memory_mb": peak_cuda_memory_mb,
    }

    del model, val_union, test_union
    gc.collect()
    if device.type == "cuda":
        torch.cuda.empty_cache()
    return rows, metadata


def _write_summaries(frame: pd.DataFrame, output_dir: str) -> None:
    group_columns = [
        "dataset",
        "graph_mode",
        "method",
        "step",
        "target_snr_db",
        "actual_snr_db",
    ]
    summary = (
        frame.groupby(group_columns, dropna=False)
        .agg(
            runs=("seed", "nunique"),
            auprc_mean=("auprc", "mean"),
            auprc_std=("auprc", "std"),
            auroc_mean=("auroc", "mean"),
            auroc_std=("auroc", "std"),
            pot_f1_mean=("pot_f1", "mean"),
            pot_f1_std=("pot_f1", "std"),
            pot_raw_add_mean=("pot_raw_add", "mean"),
            oracle_f1_mean=("oracle_f1", "mean"),
            oracle_f1_std=("oracle_f1", "std"),
            forward_seconds_mean=("forward_seconds", "mean"),
            peak_cuda_memory_mb_mean=("peak_cuda_memory_mb", "mean"),
            forwards_per_window=("forwards_per_window", "first"),
            estimated_method_forward_seconds_mean=(
                "estimated_method_forward_seconds",
                "mean",
            ),
        )
        .reset_index()
    )
    summary.to_csv(os.path.join(output_dir, "summary.csv"), index=False)

    controlled_methods = [
        "configured_original",
        "configured_raw_eps",
        "configured_calibrated_eps",
        "study_raw_eps",
        "study_calibrated_eps",
    ]
    controlled = frame[frame["method"].isin(controlled_methods)]
    causal_pivot = controlled.pivot_table(
        index=["dataset", "graph_mode", "seed"],
        columns="method",
        values="auprc",
        aggfunc="first",
    ).reset_index()
    if set(controlled_methods).issubset(causal_pivot.columns):
        contrast_formulas = {
            "epsilon_residual": (
                "configured_raw_eps",
                "configured_original",
            ),
            "per_level_calibration_configured": (
                "configured_calibrated_eps",
                "configured_raw_eps",
            ),
            "additional_noise_levels_raw": (
                "study_raw_eps",
                "configured_raw_eps",
            ),
            "additional_noise_levels_calibrated": (
                "study_calibrated_eps",
                "configured_calibrated_eps",
            ),
            "per_level_calibration_study": (
                "study_calibrated_eps",
                "study_raw_eps",
            ),
        }
        for contrast, (positive, negative) in contrast_formulas.items():
            causal_pivot[f"delta_{contrast}"] = (
                causal_pivot[positive] - causal_pivot[negative]
            )
        causal_pivot.to_csv(
            os.path.join(output_dir, "causal_contrasts.csv"),
            index=False,
        )
        contrast_columns = [
            column for column in causal_pivot.columns if column.startswith("delta_")
        ]
        contrast_summary = (
            causal_pivot.groupby(["dataset", "graph_mode"])[contrast_columns]
            .agg(["mean", "std", "count"])
        )
        contrast_summary.columns = [
            f"{metric}_{statistic}"
            for metric, statistic in contrast_summary.columns
        ]
        contrast_summary.reset_index().to_csv(
            os.path.join(output_dir, "causal_contrasts_summary.csv"),
            index=False,
        )

        try:
            from scipy.stats import wilcoxon

            causal_test_rows: List[Dict[str, Any]] = []
            for graph_mode in sorted(causal_pivot["graph_mode"].unique()):
                mode_frame = causal_pivot[causal_pivot["graph_mode"] == graph_mode]
                scopes: List[Tuple[str, pd.DataFrame]] = [
                    (
                        str(dataset),
                        mode_frame[mode_frame["dataset"] == dataset],
                    )
                    for dataset in sorted(mode_frame["dataset"].unique())
                ]
                scopes.append(("GLOBAL", mode_frame))
                for scope, scope_frame in scopes:
                    for contrast_column in contrast_columns:
                        differences = scope_frame[contrast_column].dropna().to_numpy()
                        if len(differences) >= 3 and np.any(differences != 0.0):
                            statistic, p_value = wilcoxon(differences)
                        else:
                            statistic, p_value = float("nan"), float("nan")
                        causal_test_rows.append(
                            {
                                "scope": scope,
                                "graph_mode": graph_mode,
                                "contrast": contrast_column.removeprefix("delta_"),
                                "test": "paired Wilcoxon contrast vs zero",
                                "blocks": int(len(differences)),
                                "delta_auprc_mean": (
                                    float(np.mean(differences))
                                    if len(differences)
                                    else float("nan")
                                ),
                                "statistic": float(statistic),
                                "p_value": float(p_value),
                            }
                        )
            pd.DataFrame(causal_test_rows).to_csv(
                os.path.join(output_dir, "causal_contrast_tests.csv"),
                index=False,
            )
        except ImportError:
            print(
                "SciPy unavailable; causal_contrast_tests.csv was not generated."
            )

    single = frame[frame["method"] == "study_single_calibrated"]
    pivot = single.pivot_table(
        index=["dataset", "seed", "step", "actual_snr_db"],
        columns="graph_mode",
        values="auprc",
        aggfunc="first",
    ).reset_index()
    if {"identity", "hybrid"}.issubset(pivot.columns):
        pivot["delta_auprc_hybrid_minus_identity"] = (
            pivot["hybrid"] - pivot["identity"]
        )
        interaction_path = os.path.join(output_dir, "interaction.csv")
        pivot.to_csv(interaction_path, index=False)
        interaction_summary = (
            pivot.groupby(["dataset", "step", "actual_snr_db"])
            .agg(
                runs=("seed", "nunique"),
                delta_auprc_mean=(
                    "delta_auprc_hybrid_minus_identity",
                    "mean",
                ),
                delta_auprc_std=(
                    "delta_auprc_hybrid_minus_identity",
                    "std",
                ),
            )
            .reset_index()
        )
        interaction_summary.to_csv(
            os.path.join(output_dir, "interaction_summary.csv"),
            index=False,
        )

        # Non-parametric repeated-measures test of the null hypothesis that
        # hybrid-minus-identity gain is constant across SNR levels. Seeds are
        # the repeated blocks; the global test uses dataset-seed blocks.
        try:
            from scipy.stats import friedmanchisquare

            test_rows: List[Dict[str, Any]] = []
            scopes: List[Tuple[str, pd.DataFrame]] = [
                (str(dataset), pivot[pivot["dataset"] == dataset])
                for dataset in sorted(pivot["dataset"].unique())
            ]
            scopes.append(("GLOBAL", pivot))
            for scope, scope_frame in scopes:
                index_columns = ["seed"] if scope != "GLOBAL" else ["dataset", "seed"]
                wide = scope_frame.pivot_table(
                    index=index_columns,
                    columns="step",
                    values="delta_auprc_hybrid_minus_identity",
                    aggfunc="first",
                ).dropna(axis=0, how="any")
                if wide.shape[0] >= 3 and wide.shape[1] >= 3:
                    statistic, p_value = friedmanchisquare(
                        *[wide[column].to_numpy() for column in wide.columns]
                    )
                else:
                    statistic, p_value = float("nan"), float("nan")
                test_rows.append(
                    {
                        "scope": scope,
                        "test": "Friedman across SNR levels",
                        "blocks": int(wide.shape[0]),
                        "levels": int(wide.shape[1]),
                        "statistic": float(statistic),
                        "p_value": float(p_value),
                    }
                )
            pd.DataFrame(test_rows).to_csv(
                os.path.join(output_dir, "interaction_tests.csv"),
                index=False,
            )
        except ImportError:
            print("SciPy unavailable; interaction_tests.csv was not generated.")

    # Aggregate graph comparison for the five controlled scoring paths.  This
    # complements the per-SNR interaction table above and supports paired
    # hybrid-versus-identity tests across dataset/seed blocks.
    aggregate = frame[frame["step"] == AGGREGATE_STEP]
    graph_pivot = aggregate.pivot_table(
        index=["dataset", "seed", "method"],
        columns="graph_mode",
        values="auprc",
        aggfunc="first",
    ).reset_index()
    if {"identity", "hybrid"}.issubset(graph_pivot.columns):
        graph_pivot["delta_auprc_hybrid_minus_identity"] = (
            graph_pivot["hybrid"] - graph_pivot["identity"]
        )
        graph_pivot.to_csv(
            os.path.join(output_dir, "graph_comparison.csv"),
            index=False,
        )
        graph_summary = (
            graph_pivot.groupby(["dataset", "method"])
            .agg(
                runs=("seed", "nunique"),
                delta_auprc_mean=(
                    "delta_auprc_hybrid_minus_identity",
                    "mean",
                ),
                delta_auprc_std=(
                    "delta_auprc_hybrid_minus_identity",
                    "std",
                ),
            )
            .reset_index()
        )
        graph_summary.to_csv(
            os.path.join(output_dir, "graph_comparison_summary.csv"),
            index=False,
        )

        try:
            from scipy.stats import wilcoxon

            graph_test_rows: List[Dict[str, Any]] = []
            for method in sorted(graph_pivot["method"].unique()):
                method_frame = graph_pivot[graph_pivot["method"] == method]
                scopes: List[Tuple[str, pd.DataFrame]] = [
                    (
                        str(dataset),
                        method_frame[method_frame["dataset"] == dataset],
                    )
                    for dataset in sorted(method_frame["dataset"].unique())
                ]
                scopes.append(("GLOBAL", method_frame))
                for scope, scope_frame in scopes:
                    differences = scope_frame[
                        "delta_auprc_hybrid_minus_identity"
                    ].dropna().to_numpy()
                    if len(differences) >= 3 and np.any(differences != 0.0):
                        statistic, p_value = wilcoxon(differences)
                    else:
                        statistic, p_value = float("nan"), float("nan")
                    graph_test_rows.append(
                        {
                            "scope": scope,
                            "method": method,
                            "test": "paired Wilcoxon hybrid vs identity",
                            "blocks": int(len(differences)),
                            "delta_auprc_mean": (
                                float(np.mean(differences))
                                if len(differences)
                                else float("nan")
                            ),
                            "statistic": float(statistic),
                            "p_value": float(p_value),
                        }
                    )
            pd.DataFrame(graph_test_rows).to_csv(
                os.path.join(output_dir, "graph_tests.csv"),
                index=False,
            )
        except ImportError:
            print("SciPy unavailable; graph_tests.csv was not generated.")


def _write_plots(frame: pd.DataFrame, output_dir: str) -> None:
    try:
        import matplotlib.pyplot as plt
    except ImportError:
        print("matplotlib is unavailable; CSV/JSON outputs were still written.")
        return

    plot_dir = os.path.join(output_dir, "plots")
    os.makedirs(plot_dir, exist_ok=True)
    single = frame[frame["method"] == "study_single_calibrated"]
    for dataset in sorted(single["dataset"].unique()):
        subset = single[single["dataset"] == dataset]
        fig, axis = plt.subplots(figsize=(7.2, 4.6))
        for mode in GRAPH_MODES:
            mode_rows = subset[subset["graph_mode"] == mode]
            if mode_rows.empty:
                continue
            grouped = (
                mode_rows.groupby("actual_snr_db")
                .agg(
                    mean=("auprc", "mean"),
                    std=("auprc", "std"),
                    count=("seed", "nunique"),
                )
                .reset_index()
                .sort_values("actual_snr_db")
            )
            x = grouped["actual_snr_db"].to_numpy()
            y = grouped["mean"].to_numpy()
            ci = (
                1.96
                * grouped["std"].fillna(0.0).to_numpy()
                / np.sqrt(np.maximum(grouped["count"].to_numpy(), 1))
            )
            axis.plot(x, y, marker="o", label=mode)
            axis.fill_between(x, y - ci, y + ci, alpha=0.15)
        axis.set_title(f"{dataset}: calibrated single-level response")
        axis.set_xlabel("Actual SNR (dB)")
        axis.set_ylabel("Pointwise AUPRC")
        axis.grid(alpha=0.25)
        axis.legend()
        fig.tight_layout()
        fig.savefig(
            os.path.join(plot_dir, f"{dataset.lower()}_snr_response.png"),
            dpi=180,
        )
        plt.close(fig)

    pivot = single.pivot_table(
        index=["dataset", "seed", "step", "actual_snr_db"],
        columns="graph_mode",
        values="auprc",
        aggfunc="first",
    ).reset_index()
    if {"identity", "hybrid"}.issubset(pivot.columns):
        pivot["delta"] = pivot["hybrid"] - pivot["identity"]
        for dataset in sorted(pivot["dataset"].unique()):
            subset = pivot[pivot["dataset"] == dataset]
            grouped = (
                subset.groupby("actual_snr_db")
                .agg(
                    mean=("delta", "mean"),
                    std=("delta", "std"),
                    count=("seed", "nunique"),
                )
                .reset_index()
                .sort_values("actual_snr_db")
            )
            x = grouped["actual_snr_db"].to_numpy()
            y = grouped["mean"].to_numpy()
            ci = (
                1.96
                * grouped["std"].fillna(0.0).to_numpy()
                / np.sqrt(np.maximum(grouped["count"].to_numpy(), 1))
            )
            fig, axis = plt.subplots(figsize=(7.2, 4.6))
            axis.axhline(0.0, color="black", linewidth=1.0)
            axis.plot(x, y, marker="o", color="tab:purple")
            axis.fill_between(x, y - ci, y + ci, alpha=0.18, color="tab:purple")
            axis.set_title(f"{dataset}: graph-by-SNR interaction")
            axis.set_xlabel("Actual SNR (dB)")
            axis.set_ylabel("AUPRC(hybrid) - AUPRC(identity)")
            axis.grid(alpha=0.25)
            fig.tight_layout()
            fig.savefig(
                os.path.join(plot_dir, f"{dataset.lower()}_graph_snr_delta.png"),
                dpi=180,
            )
            plt.close(fig)


def _analysis_output_dir(args: argparse.Namespace) -> str:
    name = str(args.analysis_name)
    if not name or not all(character.isalnum() or character in "-_" for character in name):
        raise ValueError(
            "analysis_name must contain only letters, numbers, '-' or '_'"
        )
    output_dir = os.path.join(args.study_dir, name)
    if os.path.isdir(output_dir) and any(os.scandir(output_dir)) and not args.resume:
        raise FileExistsError(
            f"Analysis output already exists and is not empty: {output_dir}. "
            "Choose a new --analysis_name or pass --resume."
        )
    return output_dir


def analyze_variants(args: argparse.Namespace) -> None:
    device = _device(args.allow_cpu)
    output_dir = _analysis_output_dir(args)
    os.makedirs(output_dir, exist_ok=True)
    all_rows: List[Dict[str, Any]] = []
    metadata: List[Dict[str, Any]] = []
    metrics_path = os.path.join(output_dir, "metrics.csv")
    metadata_path = os.path.join(output_dir, "run_metadata.json")
    if args.resume and os.path.isfile(metrics_path):
        all_rows = pd.read_csv(metrics_path).to_dict(orient="records")
    if args.resume and os.path.isfile(metadata_path):
        with open(metadata_path, "r", encoding="utf-8") as handle:
            metadata = json.load(handle)
    completed = {
        (item.get("dataset"), item.get("graph_mode"), int(item.get("seed", -1)))
        for item in metadata
    }

    for dataset in args.datasets:
        for mode in args.graph_modes:
            for seed in args.seeds:
                key = (dataset, mode, int(seed))
                if key in completed:
                    print(
                        f"SKIP analyzed dataset={dataset} graph_mode={mode} "
                        f"seed={seed}"
                    )
                    continue
                print(
                    f"ANALYZE dataset={dataset} graph_mode={mode} seed={seed}"
                )
                rows, run_metadata = _analyze_one(
                    args,
                    dataset,
                    mode,
                    seed,
                    device,
                )
                all_rows.extend(rows)
                metadata.append(run_metadata)
                completed.add(key)
                pd.DataFrame(all_rows).to_csv(metrics_path, index=False)
                _json_dump(metadata_path, metadata)

    frame = pd.DataFrame(all_rows)
    if frame.empty:
        raise RuntimeError("No study rows were generated")
    _write_summaries(frame, output_dir)
    _write_plots(frame, output_dir)
    _json_dump(
        os.path.join(output_dir, "analysis_config.json"),
        {
            "created_utc": datetime.now(timezone.utc).isoformat(),
            "datasets": args.datasets,
            "graph_modes": args.graph_modes,
            "seeds": args.seeds,
            "target_snr_db": args.target_snr_db,
            "configured_steps_override": args.configured_steps,
            "num_samples": args.num_samples,
            "normalization": args.normalization,
            "level_agg": args.level_agg,
            "clip_min": args.clip_min,
            "channel_agg": args.channel_agg,
            "ema_alpha": args.ema_alpha,
            "include_range": args.include_range,
            "primary_metric": "pointwise AUPRC",
            "threshold_protocol": "POT fitted on normal validation scores",
            "oracle_metrics": "diagnostic only",
            "controlled_methods": {
                "configured_original": (
                    "checkpoint eval steps; original recon+eps weighting; "
                    "aggregate levels before checkpoint sensor normalization"
                ),
                "configured_raw_eps": (
                    "checkpoint eval steps; equal epsilon aggregation before "
                    "checkpoint sensor normalization"
                ),
                "configured_calibrated_eps": (
                    "checkpoint eval steps; level/sensor calibration before "
                    "aggregation"
                ),
                "study_raw_eps": (
                    "target-SNR steps; equal epsilon aggregation before "
                    "checkpoint sensor normalization"
                ),
                "study_calibrated_eps": (
                    "target-SNR steps; level/sensor calibration before "
                    "aggregation"
                ),
            },
            "causal_contrasts": {
                "epsilon_residual": "configured_raw_eps - configured_original",
                "per_level_calibration": (
                    "configured_calibrated_eps - configured_raw_eps"
                ),
                "additional_noise_levels_raw": (
                    "study_raw_eps - configured_raw_eps"
                ),
                "additional_noise_levels_calibrated": (
                    "study_calibrated_eps - configured_calibrated_eps"
                ),
                "calibration_on_study_grid": (
                    "study_calibrated_eps - study_raw_eps"
                ),
            },
        },
    )
    print(f"Study outputs written to: {output_dir}")


def _common_parser(parser: argparse.ArgumentParser) -> None:
    parser.add_argument(
        "--datasets",
        nargs="+",
        default=list(DEFAULT_DATASETS),
        choices=list(SUPPORTED_DATASETS),
    )
    parser.add_argument(
        "--graph_modes",
        nargs="+",
        default=list(GRAPH_MODES),
        choices=list(GRAPH_MODES),
    )
    parser.add_argument("--seeds", nargs="+", type=int, default=[42, 43, 44, 45, 46])
    parser.add_argument("--study_dir", default="results/snr_study")
    parser.add_argument(
        "--data_root",
        default=None,
        help="Optional parent containing one subdirectory per dataset",
    )
    parser.add_argument("--allow_cpu", action="store_true")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    subparsers = parser.add_subparsers(dest="command", required=True)

    train_parser = subparsers.add_parser("train", help="Train graph variants")
    _common_parser(train_parser)
    train_parser.add_argument("--config_dir", default="configs")
    train_parser.add_argument("--override", nargs="*", default=[])
    train_parser.add_argument(
        "--force",
        action="store_true",
        help="Retrain even when the expected checkpoint already exists",
    )

    analyze_parser = subparsers.add_parser(
        "analyze",
        help="Compute calibrated per-SNR metrics and interaction tables",
    )
    _common_parser(analyze_parser)
    analyze_parser.add_argument(
        "--target_snr_db",
        nargs="+",
        type=float,
        default=[20.0, 10.0, 3.0, 0.0, -3.0, -10.0],
    )
    analyze_parser.add_argument(
        "--configured_steps",
        nargs="+",
        type=int,
        default=None,
        help="Override checkpoint eval_steps for the configured three-path control",
    )
    analyze_parser.add_argument(
        "--num_samples",
        type=int,
        default=None,
        help="Defaults to the value stored in each checkpoint",
    )
    analyze_parser.add_argument(
        "--normalization",
        choices=["checkpoint", "robust", "zscore"],
        default="checkpoint",
        help="Per-level calibration; 'checkpoint' uses score.normalization",
    )
    analyze_parser.add_argument(
        "--level_agg",
        choices=["mean", "median", "max"],
        default="mean",
    )
    analyze_parser.add_argument("--clip_min", type=float, default=0.0)
    analyze_parser.add_argument(
        "--channel_agg",
        choices=["mean", "max", "sum"],
        default=None,
        help="Defaults to the value stored in each checkpoint",
    )
    analyze_parser.add_argument(
        "--ema_alpha",
        type=float,
        default=None,
        help="Defaults to the value stored in each checkpoint",
    )
    analyze_parser.add_argument(
        "--analysis_name",
        default="analysis",
        help="Versioned output directory name under study_dir",
    )
    analyze_parser.add_argument(
        "--resume",
        action="store_true",
        help="Continue an interrupted analysis and skip completed run triples",
    )
    analyze_parser.add_argument("--eval_seed_offset", type=int, default=10000)
    analyze_parser.add_argument(
        "--include_range",
        action="store_true",
        help="Also compute Range-AUC/VUS for every method (considerably slower)",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    args.study_dir = os.path.abspath(args.study_dir)
    if args.command == "train":
        train_variants(args)
    else:
        analyze_variants(args)


if __name__ == "__main__":
    main()
