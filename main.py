"""Orchestrator: train and/or evaluate GDAD across multiple datasets and
produce per-dataset and averaged results tables.

Two result tables are produced per run:

* PRIMARY -- label-free **POT** threshold, fitted on the (normal) validation
  scores and applied to test. This never touches test labels to pick the
  threshold and is the publishable number.
* DIAGNOSTIC -- **oracle best-F1** threshold, which peeks at the test labels.
  Reported only as an optimistic upper bound; not used for model selection.

Examples:
    python main.py --datasets SMAP MSL SWaT PSM
    python main.py --datasets SMAP --mode eval
    python main.py --datasets all --mode train_eval
"""
from __future__ import annotations

import argparse
import json
import os
from typing import Dict, List

from evaluate import evaluate_one_dataset
from train import train_one_dataset
from utils.config import load_config
from utils.logger import get_logger

# Four benchmark datasets: the default 'all' set and the only set averaged over.
ALL_DATASETS = ["SMAP", "MSL", "SWaT", "PSM"]
# Additional datasets stay explicit so historical ``--datasets all`` remains
OPTIONAL_DATASETS = ["HAI"]
SUPPORTED_DATASETS = ALL_DATASETS + OPTIONAL_DATASETS


def _row(name: str, res: Dict, key: str) -> Dict[str, float]:
    """Build a table row using threshold strategy ``key`` (``pot`` | ``best_f1``)."""
    pw = res["pointwise"][key]
    adj = res["adjusted"][key]
    return {
        "dataset": name,
        "PW_P": pw["precision"],
        "PW_R": pw["recall"],
        "PW_F1": pw["f1"],
        "ADJ_P": adj["precision"],
        "ADJ_R": adj["recall"],
        "ADJ_F1": adj["f1"],
        "AUROC": res["auroc"],
        "AUPRC": res["auprc"],
    }


def _format_table(rows: List[Dict[str, float]]) -> str:
    cols = ["dataset", "PW_P", "PW_R", "PW_F1", "ADJ_P", "ADJ_R", "ADJ_F1", "AUROC", "AUPRC"]
    header = "| " + " | ".join(cols) + " |"
    sep = "| " + " | ".join(["---"] * len(cols)) + " |"
    lines = [header, sep]
    for r in rows:
        cells = [str(r["dataset"])] + [
            f"{r[c]:.4f}" if isinstance(r[c], float) else str(r[c]) for c in cols[1:]
        ]
        lines.append("| " + " | ".join(cells) + " |")
    return "\n".join(lines)


def _average_row(rows: List[Dict[str, float]]) -> Dict[str, float]:
    keys = ["PW_P", "PW_R", "PW_F1", "ADJ_P", "ADJ_R", "ADJ_F1", "AUROC", "AUPRC"]
    avg = {"dataset": "AVERAGE"}
    for k in keys:
        vals = [r[k] for r in rows if r[k] == r[k]]  # skip NaN
        avg[k] = sum(vals) / len(vals) if vals else float("nan")
    return avg


def parse_args():
    parser = argparse.ArgumentParser(description="Run GDAD across datasets.")
    parser.add_argument("--datasets", nargs="+", default=["all"])
    parser.add_argument(
        "--mode", choices=["train", "eval", "train_eval"], default="train_eval"
    )
    parser.add_argument("--config_dir", default="configs", type=str)
    parser.add_argument("--override", nargs="*", default=[])
    parser.add_argument("--result_dir", default="results", type=str)
    return parser.parse_args()


def _resolve_datasets(raw: List[str]) -> List[str]:
    if raw == ["all"] or "all" in raw:
        return list(ALL_DATASETS)
    bad = [d for d in raw if d not in set(SUPPORTED_DATASETS)]
    if bad:
        raise ValueError(
            f"Unknown dataset(s): {bad}. Valid: {SUPPORTED_DATASETS} or 'all' "
            f"(= {ALL_DATASETS})."
        )
    return raw


def main():
    args = parse_args()
    datasets = _resolve_datasets(args.datasets)
    logger = get_logger("gdad", log_dir="logs")

    pot_rows: List[Dict[str, float]] = []
    oracle_rows: List[Dict[str, float]] = []
    for name in datasets:
        cfg = load_config(name, config_dir=args.config_dir, overrides=args.override)
        ckpt_path = os.path.join(cfg.train.get("ckpt_dir", "checkpoints"), name, "best.pt")

        if args.mode in ("train", "train_eval"):
            ckpt_path = train_one_dataset(cfg, logger=logger)

        if args.mode in ("eval", "train_eval"):
            res = evaluate_one_dataset(cfg, ckpt_path, logger=logger)
            pot_rows.append(_row(name, res, key="pot"))
            oracle_rows.append(_row(name, res, key="best_f1"))

    if pot_rows:
        # Average only over the datasets actually evaluated (four by default).
        pot_rows.append(_average_row(pot_rows))
        oracle_rows.append(_average_row(oracle_rows))
        pot_table = _format_table(pot_rows)
        oracle_table = _format_table(oracle_rows)

        logger.info(
            "\n=== GDAD PRIMARY results (label-free POT threshold) ===\n" + pot_table
        )
        logger.info(
            "\n=== GDAD DIAGNOSTIC results (oracle best-F1, peeks at test labels) "
            "===\n" + oracle_table
        )

        os.makedirs(args.result_dir, exist_ok=True)
        with open(os.path.join(args.result_dir, "summary.md"), "w", encoding="utf-8") as fh:
            fh.write("# GDAD results\n\n")
            fh.write(
                f"Datasets: {', '.join(datasets)} "
                f"(AVERAGE row computed over these {len(datasets)} datasets).\n\n"
            )
            fh.write("## Primary — label-free POT threshold (validation-fitted)\n\n")
            fh.write(pot_table + "\n\n")
            fh.write(
                "## Diagnostic — oracle test best-F1 threshold "
                "(not used for model selection)\n\n"
            )
            fh.write(oracle_table + "\n")
        with open(os.path.join(args.result_dir, "summary.json"), "w", encoding="utf-8") as fh:
            json.dump(
                {"datasets": datasets, "primary_pot": pot_rows, "oracle_best_f1": oracle_rows},
                fh,
                indent=2,
            )
        logger.info(f"Saved summary to {args.result_dir}/summary.md")


if __name__ == "__main__":
    main()
