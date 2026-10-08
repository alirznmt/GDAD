"""Synthetic forward-pass scaling benchmark for GDAD's dense sensor graph."""
from __future__ import annotations

import argparse
import os
import sys
import time

import pandas as pd
import torch

PROJECT_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
if PROJECT_ROOT not in sys.path:
    sys.path.insert(0, PROJECT_ROOT)

from models import build_model
from utils.config import load_config


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--node_counts", nargs="+", type=int, default=[25, 51, 55, 86, 123, 256])
    parser.add_argument(
        "--graph_modes",
        nargs="+",
        default=["identity", "static", "dynamic", "hybrid"],
    )
    parser.add_argument("--batch_size", type=int, default=8)
    parser.add_argument("--window_size", type=int, default=100)
    parser.add_argument("--warmup", type=int, default=3)
    parser.add_argument("--repeats", type=int, default=10)
    parser.add_argument("--config_dir", default="configs")
    parser.add_argument("--output_dir", required=True)
    parser.add_argument("--allow_cpu", action="store_true")
    args = parser.parse_args()

    if not torch.cuda.is_available() and not args.allow_cpu:
        raise RuntimeError("CUDA unavailable; pass --allow_cpu only intentionally")
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    os.makedirs(args.output_dir, exist_ok=True)
    rows = []
    for nodes in args.node_counts:
        for mode in args.graph_modes:
            cfg = load_config("WADI", config_dir=args.config_dir)
            cfg.model["graph_mode"] = mode
            model = None
            try:
                torch.manual_seed(0)
                if device.type == "cuda":
                    torch.cuda.empty_cache()
                    torch.cuda.reset_peak_memory_stats(device)
                model = build_model(cfg, num_nodes=nodes).to(device).eval()
                x = torch.randn(
                    args.batch_size,
                    args.window_size,
                    nodes,
                    device=device,
                )
                t = torch.full(
                    (args.batch_size,), 25, dtype=torch.long, device=device
                )
                with torch.inference_mode():
                    for _ in range(args.warmup):
                        model.diffusion.model(x, t)
                    if device.type == "cuda":
                        torch.cuda.synchronize()
                    started = time.perf_counter()
                    for _ in range(args.repeats):
                        model.diffusion.model(x, t)
                    if device.type == "cuda":
                        torch.cuda.synchronize()
                seconds = (time.perf_counter() - started) / args.repeats
                peak_mb = (
                    torch.cuda.max_memory_allocated(device) / (1024.0 ** 2)
                    if device.type == "cuda"
                    else float("nan")
                )
                rows.append(
                    {
                        "nodes": nodes,
                        "graph_mode": mode,
                        "batch_size": args.batch_size,
                        "window_size": args.window_size,
                        "seconds_per_denoiser_call": seconds,
                        "peak_cuda_memory_mb": peak_mb,
                        "dense_adjacency_batch_mb": (
                            args.batch_size * nodes * nodes * 4 / (1024.0 ** 2)
                        ),
                        "parameters": sum(p.numel() for p in model.parameters()),
                        "status": "ok",
                    }
                )
                print(
                    f"OK nodes={nodes} mode={mode}: {seconds:.5f}s, "
                    f"peak={peak_mb:.1f} MiB"
                )
            except torch.cuda.OutOfMemoryError as error:
                rows.append(
                    {
                        "nodes": nodes,
                        "graph_mode": mode,
                        "batch_size": args.batch_size,
                        "window_size": args.window_size,
                        "status": f"OOM: {error}",
                    }
                )
                print(f"OOM nodes={nodes} mode={mode}")
            finally:
                del model
                if device.type == "cuda":
                    torch.cuda.empty_cache()

    frame = pd.DataFrame(rows)
    csv_path = os.path.join(args.output_dir, "graph_scaling.csv")
    frame.to_csv(csv_path, index=False)
    lines = [
        "# GDAD graph-scaling benchmark",
        "",
        f"Batch size {args.batch_size}, window {args.window_size}, "
        f"{args.repeats} timed denoiser calls after {args.warmup} warmups.",
        "",
        "| Nodes | Graph mode | Seconds/call | Peak GPU MiB | Dense adjacency MiB | Status |",
        "| ---: | --- | ---: | ---: | ---: | --- |",
    ]
    for row in rows:
        lines.append(
            f"| {row['nodes']} | {row['graph_mode']} | "
            f"{row.get('seconds_per_denoiser_call', float('nan')):.5f} | "
            f"{row.get('peak_cuda_memory_mb', float('nan')):.1f} | "
            f"{row.get('dense_adjacency_batch_mb', float('nan')):.2f} | "
            f"{row['status']} |"
        )
    markdown_path = os.path.join(args.output_dir, "graph_scaling.md")
    with open(markdown_path, "w", encoding="utf-8") as handle:
        handle.write("\n".join(lines) + "\n")
    print(f"Wrote {csv_path} and {markdown_path}")


if __name__ == "__main__":
    main()
