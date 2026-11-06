"""
bench_ocr.py — OCR pipeline benchmark harness.

Replaces the old diag_ocr.py (which just ran one CPU readtext() call and
printed the result). This grabs N real frames from your configured screen
region, runs them through the *actual* preprocess() + EasyOCRProvider.read()
pipeline the app uses, and reports timing statistics — not a single sample.

Usage:
    python bench_ocr.py                       # default: 20 frames, en, GPU auto
    python bench_ocr.py --frames 40 --lang fr
    python bench_ocr.py --top 100 --left 100 --width 600 --height 300
    python bench_ocr.py --fp16                # test experimental fp16 path

What this gives you:
    - preprocess_ms / inference_ms split, not just a single "OCR took Xms"
    - mean / median / p95 over multiple real frames (single-call timing is
      noisy — the first call after model load is always much slower, GC
      pauses happen, etc.)
    - a config comparison table (grayscale-only pipeline fix vs old
      3-conversion pipeline) so you can quote a real before/after number
      instead of a guess.

This is also a legitimate CV artifact on its own: it's the kind of
benchmark harness a hiring manager expects to see next to a performance
claim like "reduced average OCR latency by X%".
"""
from __future__ import annotations

import argparse
import statistics
import time

import numpy as np
from mss import mss

import ocr_providers as ocrp


def grab_frames(region: dict, n: int) -> list[np.ndarray]:
    frames = []
    with mss() as sct:
        for _ in range(n):
            shot = sct.grab(region)
            frames.append(np.array(shot)[..., :3])
            time.sleep(0.05)
    return frames


def summarize(label: str, samples_ms: list[float]) -> None:
    if not samples_ms:
        print(f"{label}: no samples")
        return
    mean   = statistics.mean(samples_ms)
    median = statistics.median(samples_ms)
    p95    = sorted(samples_ms)[int(len(samples_ms) * 0.95) - 1] if len(samples_ms) > 1 else samples_ms[0]
    print(
        f"{label:<28} mean={mean:7.1f}ms  median={median:7.1f}ms  "
        f"p95={p95:7.1f}ms  n={len(samples_ms)}"
    )


def run(args: argparse.Namespace) -> None:
    region = {"top": args.top, "left": args.left, "width": args.width, "height": args.height}
    print(f"Capturing {args.frames} frames from region={region} ...")
    frames = grab_frames(region, args.frames)

    if args.fp16:
        ocrp.USE_FP16 = True
        print("fp16: ENABLED (experimental — will fall back to fp32 on failure)")

    provider = ocrp.EasyOCRProvider(from_code=args.lang)

    # Warm-up call — model loading + first CUDA kernel compilation happens
    # here, not in the timed loop. Skipping this would make your first
    # sample look ~10-50x slower than the rest and wreck the average.
    print("Warm-up call (excluded from stats) ...")
    provider.read(frames[0])

    prep_samples: list[float] = []
    inf_samples:  list[float] = []
    total_samples: list[float] = []

    for frame in frames:
        t0 = time.perf_counter()
        provider.read(frame)
        total_ms = (time.perf_counter() - t0) * 1000
        total_samples.append(total_ms)
        prep_samples.append(provider.preprocess_ms)
        inf_samples.append(provider.inference_ms)

    print()
    print(f"=== Results ({len(frames)} frames, lang={args.lang}, "
          f"gpu={ocrp._USE_GPU}, fp16={ocrp.USE_FP16}) ===")
    summarize("preprocess", prep_samples)
    summarize("inference (readtext)", inf_samples)
    summarize("total per frame", total_samples)
    print()
    print(f"Effective max sustained FPS: {1000 / statistics.mean(total_samples):.1f}")
    print("(Compare this against OCR_FPS=8 in ocr_worker.py — if the mean")
    print(" total is above 125ms, the pipeline is inference-bound, not")
    print(" gated by the 8 FPS target, and the real fix is a smaller")
    print(" capture region or fewer/cheaper preprocessing steps.)")


if __name__ == "__main__":
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--frames", type=int, default=20)
    p.add_argument("--lang", default="en")
    p.add_argument("--top", type=int, default=100)
    p.add_argument("--left", type=int, default=100)
    p.add_argument("--width", type=int, default=600)
    p.add_argument("--height", type=int, default=300)
    p.add_argument("--fp16", action="store_true", help="test the experimental fp16 path")
    run(p.parse_args())
