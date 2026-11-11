"""Measure CPU inference latency.

    python tools/benchmark_cpu.py --random
    python tools/benchmark_cpu.py --checkpoint model/bam_v1.pth --threads 4 8 12
    python tools/benchmark_cpu.py --random --widths 16 20 24 32 --csv reports/bench.csv

Reports milliseconds per crop, which is what the stroke budget is built from.
"""

from __future__ import annotations

import argparse
import csv
import json
import os
import sys
import time
from pathlib import Path

import torch

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from training.unet import BAMUNet, estimate_gflops, load_checkpoint_model  # noqa: E402


def build(width: int, checkpoint: str | None) -> BAMUNet:
    if checkpoint:
        model = load_checkpoint_model(checkpoint)
    else:
        model = BAMUNet(width=width)
    model.eval()
    return model


def time_model(model, batch: int, size: int, threads: int, repeats: int = 5) -> float:
    torch.set_num_threads(max(1, threads))
    model = model.to(memory_format=torch.channels_last)
    x = torch.randn(batch, 4, size, size).to(memory_format=torch.channels_last)

    with torch.inference_mode():
        model(x)
        model(x)
        started = time.perf_counter()
        for _ in range(repeats):
            model(x)
        elapsed = (time.perf_counter() - started) / repeats

    return elapsed / batch * 1000.0


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Benchmark BAM on CPU")
    parser.add_argument("--checkpoint", default=None)
    parser.add_argument("--random", action="store_true", help="use untrained weights")
    parser.add_argument("--widths", nargs="+", type=int, default=[20])
    parser.add_argument("--threads", nargs="+", type=int, default=[4, 8])
    parser.add_argument("--batch", nargs="+", type=int, default=[1, 8, 16])
    parser.add_argument("--size", type=int, default=256)
    parser.add_argument("--csv", default=None)
    args = parser.parse_args(argv)

    if not args.checkpoint and not args.random:
        parser.error("pass --checkpoint or --random")

    rows = []
    for width in args.widths:
        model = build(width, args.checkpoint)
        params = sum(p.numel() for p in model.parameters())
        gflops = estimate_gflops(model, size=args.size)
        for threads in args.threads:
            for batch in args.batch:
                ms = time_model(model, batch, args.size, threads)
                rows.append(
                    {
                        "width": width,
                        "params": params,
                        "gflops": round(gflops, 3),
                        "threads": threads,
                        "batch": batch,
                        "ms_per_crop": round(ms, 2),
                    }
                )
                print(
                    f"width={width:>3} params={params/1e6:.2f}M gflops={gflops:5.2f} "
                    f"threads={threads:>2} batch={batch:>2} -> {ms:6.2f} ms/crop"
                )

    print(f"\ncpu_count={os.cpu_count()} torch={torch.__version__} size={args.size}")

    if args.csv:
        path = Path(args.csv)
        path.parent.mkdir(parents=True, exist_ok=True)
        with path.open("w", newline="", encoding="utf-8") as handle:
            writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
            writer.writeheader()
            writer.writerows(rows)
        print(f"wrote {path}")

    best = min(rows, key=lambda r: r["ms_per_crop"])
    print(json.dumps({"fastest": best}, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
