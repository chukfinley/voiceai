"""Benchmark: can the streaming engine hold the 80 ms/frame real-time budget?

Feeds random user codes through StreamingEngine (no Mimi vocoding by default;
add --with-mimi to include decode cost) and reports per-frame latency
percentiles. Also exercises the sliding-window re-prefill so its spike shows
up in p99/max.

Smoke (CPU, tiny backbone):
    uv run python scripts/bench_streaming.py --smoke

Real (GPU):
    uv run python scripts/bench_streaming.py --model runs/stage2/final --frames 500
"""
from __future__ import annotations

import argparse
import time
from pathlib import Path

import numpy as np
import torch


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("--model", type=Path, default=None, help="VoiceAILM ckpt dir; omit for fresh init")
    p.add_argument("--backbone", default="Qwen/Qwen3-1.7B")
    p.add_argument("--frames", type=int, default=200)
    p.add_argument("--max-frames", type=int, default=4096)
    p.add_argument("--window-frames", type=int, default=2048)
    p.add_argument("--with-mimi", action="store_true")
    p.add_argument("--device", default="cuda")
    p.add_argument("--smoke", action="store_true", help="tiny random backbone on CPU")
    args = p.parse_args()

    from voiceai.inference.streaming import StreamingEngine
    from voiceai.model.voiceai_lm import VoiceAIConfig, VoiceAILM

    device = args.device if torch.cuda.is_available() else "cpu"
    if args.smoke:
        device = "cpu"
        args.backbone = "hf-internal-testing/tiny-random-LlamaForCausalLM"
        args.frames = min(args.frames, 60)
        args.max_frames, args.window_frames = 40, 20  # force re-prefill in-run

    if args.model is not None:
        model = VoiceAILM.from_pretrained(args.model)
    else:
        cfg = VoiceAIConfig(
            backbone=args.backbone,
            freeze_backbone=True,
            dtype="float32" if device == "cpu" else "bfloat16",
            depth_dim=64 if args.smoke else 512,
            depth_layers=1 if args.smoke else 4,
        )
        model = VoiceAILM(cfg)
    model = model.to(device).eval()

    mimi = None
    if args.with_mimi:
        from voiceai.model.mimi_utils import load_mimi
        mimi = load_mimi(device=device)

    engine = StreamingEngine(
        model, mimi, device=device,
        max_frames=args.max_frames, window_frames=args.window_frames,
    )

    K = model.cfg.num_codebooks
    V = model.cfg.codebook_size
    times = []
    for i in range(args.frames):
        user = torch.randint(0, V, (K,), device=device)
        t0 = time.perf_counter()
        engine.step(user)
        if device == "cuda":
            torch.cuda.synchronize()
        times.append((time.perf_counter() - t0) * 1000)

    t = np.array(times[5:])  # skip warmup
    budget = 80.0
    print(f"frames: {len(t)}  device: {device}  backbone: {args.backbone}")
    print(f"per-frame ms — mean {t.mean():.1f}  p50 {np.percentile(t, 50):.1f}  "
          f"p95 {np.percentile(t, 95):.1f}  p99 {np.percentile(t, 99):.1f}  max {t.max():.1f}")
    rt = "REAL-TIME OK" if np.percentile(t, 95) < budget else "TOO SLOW for 80 ms budget"
    print(f"{rt} (p95 {'<' if np.percentile(t, 95) < budget else '>='} {budget:.0f} ms)")
    engine.close()


if __name__ == "__main__":
    main()
