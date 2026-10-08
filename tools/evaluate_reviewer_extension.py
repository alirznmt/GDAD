"""Resumable configured-score evaluation of every reviewer-extension checkpoint."""
from __future__ import annotations

import argparse
import json
import os
import sys
import time
from datetime import datetime, timezone

import numpy as np
import pandas as pd
import torch

PROJECT_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
if PROJECT_ROOT not in sys.path:
    sys.path.insert(0, PROJECT_ROOT)

from evaluate import evaluate_one_dataset
from models import build_model
from utils.config import Config
from utils.logger import get_logger


def checkpoint_path(study_dir, dataset, mode, seed):
    return os.path.join(
        study_dir, "checkpoints", mode, f"seed_{seed}", dataset, "best.pt"
    )


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--datasets", nargs="+", default=["HAI", "WADI"])
    parser.add_argument(
        "--graph_modes",
        nargs="+",
        default=["identity", "static", "dynamic", "hybrid"],
    )
    parser.add_argument("--seeds", nargs="+", type=int, default=[42, 43, 44, 45, 46])
    parser.add_argument("--study_dir", required=True)
    parser.add_argument("--data_root", default=None)
    parser.add_argument("--output_dir", default=None)
    parser.add_argument("--resume", action="store_true")
    parser.add_argument("--allow_cpu", action="store_true")
    args = parser.parse_args()

    if not torch.cuda.is_available() and not args.allow_cpu:
        raise RuntimeError("CUDA is unavailable; pass --allow_cpu only intentionally")
    output_dir = args.output_dir or os.path.join(args.study_dir, "reviewer_eval")
    os.makedirs(output_dir, exist_ok=True)
    metrics_path = os.path.join(output_dir, "metrics.csv")
    metadata_path = os.path.join(output_dir, "run_metadata.json")
    rows = []
    metadata = []
    if args.resume and os.path.isfile(metrics_path):
        rows = pd.read_csv(metrics_path).to_dict(orient="records")
    if args.resume and os.path.isfile(metadata_path):
        with open(metadata_path, encoding="utf-8") as handle:
            metadata = json.load(handle)
    completed = {
        (item["dataset"], item["graph_mode"], int(item["seed"]))
        for item in metadata
    }

    for dataset in args.datasets:
        for mode in args.graph_modes:
            for seed in args.seeds:
                key = (dataset, mode, int(seed))
                if key in completed:
                    print(f"SKIP evaluated {dataset} {mode} seed={seed}")
                    continue
                ckpt = checkpoint_path(args.study_dir, dataset, mode, seed)
                if not os.path.isfile(ckpt):
                    raise FileNotFoundError(f"Missing checkpoint: {ckpt}")
                checkpoint = torch.load(ckpt, map_location="cpu")
                cfg = Config(checkpoint["config"])
                if args.data_root:
                    cfg.dataset["data_root"] = os.path.join(
                        os.path.abspath(args.data_root), dataset
                    )
                cfg.train["use_cuda"] = torch.cuda.is_available()
                parameter_model = build_model(cfg, int(checkpoint["num_nodes"]))
                parameters = int(
                    sum(value.numel() for value in parameter_model.parameters())
                )
                del parameter_model
                run_dir = os.path.join(output_dir, "runs", dataset, mode, f"seed_{seed}")
                cfg.eval["result_dir"] = run_dir
                logger = get_logger(
                    f"reviewer-eval-{dataset}-{mode}-{seed}",
                    log_dir=os.path.join(run_dir, "logs"),
                )
                if torch.cuda.is_available():
                    torch.cuda.empty_cache()
                    torch.cuda.reset_peak_memory_stats()
                    torch.cuda.synchronize()
                started = time.perf_counter()
                result = evaluate_one_dataset(cfg, ckpt, logger=logger)
                if torch.cuda.is_available():
                    torch.cuda.synchronize()
                elapsed = time.perf_counter() - started
                peak_mb = (
                    torch.cuda.max_memory_allocated() / (1024.0 ** 2)
                    if torch.cuda.is_available()
                    else float("nan")
                )
                state = checkpoint["model"]
                alpha = next(
                    (
                        value
                        for name, value in state.items()
                        if name.endswith("graph_learner.alpha")
                    ),
                    None,
                )
                gate = (
                    float(torch.sigmoid(alpha).item())
                    if alpha is not None
                    else float("nan")
                )
                pot = result["pointwise"]["pot"]
                oracle = result["pointwise"]["best_f1"]
                row = {
                    "dataset": dataset,
                    "graph_mode": mode,
                    "seed": int(seed),
                    "method": "configured_original",
                    "step": -1,
                    "graph_gate": gate,
                    "parameters": parameters,
                    "sensors": int(checkpoint["num_nodes"]),
                    "evaluation_seconds": float(elapsed),
                    "forward_seconds": float(elapsed),
                    "estimated_method_forward_seconds": float(elapsed),
                    "peak_cuda_memory_mb": float(peak_mb),
                    "auroc": result["auroc"],
                    "auprc": result["auprc"],
                    "pot_threshold": pot["threshold"],
                    "pot_precision": pot["precision"],
                    "pot_recall": pot["recall"],
                    "pot_f1": pot["f1"],
                    "oracle_threshold": oracle["threshold"],
                    "oracle_precision": oracle["precision"],
                    "oracle_recall": oracle["recall"],
                    "oracle_f1": oracle["f1"],
                    "rauc_roc": result["rauc_roc"],
                    "rauc_pr": result["rauc_pr"],
                    "vus_roc": result["vus_roc"],
                    "vus_pr": result["vus_pr"],
                }
                rows.append(row)
                metadata.append(
                    {
                        "dataset": dataset,
                        "graph_mode": mode,
                        "seed": int(seed),
                        "checkpoint": os.path.abspath(ckpt),
                        "checkpoint_epoch": checkpoint.get("epoch"),
                        "checkpoint_val_loss": checkpoint.get("val_loss"),
                        "data_provenance": result.get("data_provenance"),
                        "evaluation_seconds": float(elapsed),
                        "peak_cuda_memory_mb": float(peak_mb),
                        "completed_utc": datetime.now(timezone.utc).isoformat(),
                    }
                )
                completed.add(key)
                pd.DataFrame(rows).to_csv(metrics_path, index=False)
                with open(metadata_path, "w", encoding="utf-8") as handle:
                    json.dump(metadata, handle, indent=2, allow_nan=True)
                print(
                    f"DONE {dataset} {mode} seed={seed}: "
                    f"AUPRC={result['auprc']:.4f}, elapsed={elapsed:.1f}s"
                )

    print(f"Evaluation outputs written to {output_dir}")


if __name__ == "__main__":
    main()
