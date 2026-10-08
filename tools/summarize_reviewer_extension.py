"""Build compact paper-ready tables from the HAI/WADI extension campaign."""
from __future__ import annotations

import argparse
import json
import os

import numpy as np
import pandas as pd


PRIMARY_METHOD = "configured_original"


def _format_mean_sd(mean, sd):
    return f"{mean:.4f} +/- {sd:.4f}" if np.isfinite(sd) else f"{mean:.4f}"


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--analysis_dir", required=True)
    parser.add_argument("--structural_root", default=None)
    parser.add_argument("--output_dir", default=None)
    args = parser.parse_args()

    output_dir = args.output_dir or args.analysis_dir
    os.makedirs(output_dir, exist_ok=True)
    metrics = pd.read_csv(os.path.join(args.analysis_dir, "metrics.csv"))
    primary = metrics[
        (metrics["method"] == PRIMARY_METHOD) & (metrics["step"] == -1)
    ].copy()
    if primary.empty:
        raise RuntimeError(f"No {PRIMARY_METHOD} aggregate rows found")

    aggregate = (
        primary.groupby(["dataset", "graph_mode"])
        .agg(
            seeds=("seed", "nunique"),
            auprc_mean=("auprc", "mean"),
            auprc_sd=("auprc", "std"),
            auroc_mean=("auroc", "mean"),
            auroc_sd=("auroc", "std"),
            pot_f1_mean=("pot_f1", "mean"),
            pot_f1_sd=("pot_f1", "std"),
            oracle_f1_mean=("oracle_f1", "mean"),
            oracle_f1_sd=("oracle_f1", "std"),
            inference_seconds_mean=("estimated_method_forward_seconds", "mean"),
            peak_cuda_memory_mb_mean=("peak_cuda_memory_mb", "mean"),
            parameters=("parameters", "first"),
        )
        .reset_index()
    )
    aggregate.to_csv(os.path.join(output_dir, "performance_summary.csv"), index=False)

    index = ["dataset", "seed"]
    contrast_rows = []
    for metric in ("auprc", "pot_f1", "oracle_f1"):
        wide = primary.pivot_table(
            index=index, columns="graph_mode", values=metric, aggfunc="first"
        ).reset_index()
        if "hybrid" not in wide:
            continue
        for comparator in ("identity", "static", "dynamic"):
            if comparator not in wide:
                continue
            delta_name = f"hybrid_minus_{comparator}"
            wide[delta_name] = wide["hybrid"] - wide[comparator]
            for dataset, group in wide.groupby("dataset"):
                values = group[delta_name].dropna().to_numpy()
                statistic = p_value = float("nan")
                if len(values) >= 3 and np.any(values != 0):
                    try:
                        from scipy.stats import wilcoxon

                        statistic, p_value = wilcoxon(values)
                    except (ImportError, ValueError):
                        pass
                contrast_rows.append(
                    {
                        "dataset": dataset,
                        "metric": metric,
                        "contrast": delta_name,
                        "seeds": int(len(values)),
                        "delta_mean": float(values.mean()),
                        "delta_sd": float(values.std(ddof=1)) if len(values) > 1 else 0.0,
                        "positive_seeds": int((values > 0).sum()),
                        "wilcoxon_statistic": float(statistic),
                        "wilcoxon_p": float(p_value),
                    }
                )
    contrasts = pd.DataFrame(contrast_rows)
    contrasts.to_csv(os.path.join(output_dir, "graph_contrasts.csv"), index=False)

    lines = [
        "# HAI/WADI reviewer-extension results",
        "",
        "Primary method: the checkpoint-configured original score. AUPRC is the "
        "primary threshold-free metric; POT F1 is validation-thresholded; oracle "
        "F1 is diagnostic only.",
        "",
        "## Graph ablation",
        "",
        "| Dataset | Graph | Seeds | AUPRC | POT F1 | Oracle F1 | Evaluation s | Peak GPU MiB |",
        "| --- | --- | ---: | ---: | ---: | ---: | ---: | ---: |",
    ]
    for row in aggregate.itertuples(index=False):
        lines.append(
            f"| {row.dataset} | {row.graph_mode} | {row.seeds} | "
            f"{_format_mean_sd(row.auprc_mean, row.auprc_sd)} | "
            f"{_format_mean_sd(row.pot_f1_mean, row.pot_f1_sd)} | "
            f"{_format_mean_sd(row.oracle_f1_mean, row.oracle_f1_sd)} | "
            f"{row.inference_seconds_mean:.2f} | {row.peak_cuda_memory_mb_mean:.0f} |"
        )

    lines += [
        "",
        "## Hybrid graph contrasts",
        "",
        "| Dataset | Metric | Contrast | Mean delta | Positive seeds | Wilcoxon p |",
        "| --- | --- | --- | ---: | ---: | ---: |",
    ]
    for row in contrasts.itertuples(index=False):
        p_value = "-" if not np.isfinite(row.wilcoxon_p) else f"{row.wilcoxon_p:.4g}"
        lines.append(
            f"| {row.dataset} | {row.metric} | {row.contrast} | "
            f"{row.delta_mean:.4f} +/- {row.delta_sd:.4f} | "
            f"{row.positive_seeds}/{row.seeds} | {p_value} |"
        )

    if args.structural_root:
        lines += ["", "## Structural reports", ""]
        for dataset in sorted(primary["dataset"].unique()):
            report_path = os.path.join(
                args.structural_root, dataset, "structural_report.json"
            )
            if not os.path.isfile(report_path):
                lines.append(f"- {dataset}: structural report missing")
                continue
            with open(report_path, encoding="utf-8") as handle:
                report = json.load(handle)
            stability = report.get("seed_stability", {})
            dynamic = report.get("dynamic_seed_summary", {})
            detail = []
            if stability:
                detail.append(
                    "persistent support Jaccard "
                    f"{stability['support_jaccard']['mean']:.3f}"
                )
            if dynamic:
                detail.append(f"instance analysis across {len(dynamic['seeds'])} seeds")
            lines.append(f"- {dataset}: " + ("; ".join(detail) or "report available"))

    output_path = os.path.join(output_dir, "reviewer_extension_summary.md")
    with open(output_path, "w", encoding="utf-8") as handle:
        handle.write("\n".join(lines) + "\n")
    print(f"Wrote {output_path}")


if __name__ == "__main__":
    main()
