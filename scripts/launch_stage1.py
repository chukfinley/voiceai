"""One-command launcher for Stage 1 training.

Does:
  1. Validates env (HF token, wandb, GPU)
  2. Downloads ~100h LibriSpeech-clean + CommonVoice English if not present
  3. Builds manifest
  4. Smoke-tests pipeline (5 steps on tiny stand-in) to catch bugs FAST
  5. Launches real Stage 1 training

Usage:
    uv run python scripts/launch_stage1.py \\
        --output runs/stage1 \\
        --steps 30000 \\
        --hours 100
"""
from __future__ import annotations

import argparse
import json
import os
import subprocess
import sys
from pathlib import Path


def check_env() -> None:
    print("[env] checking GPU…")
    try:
        import torch

        if not torch.cuda.is_available():
            print("[env] WARNING: no CUDA — training will be CPU-only and slow")
        else:
            print(f"[env] GPU: {torch.cuda.get_device_name(0)} ({torch.cuda.get_device_properties(0).total_memory / 1e9:.0f} GB)")
    except Exception as e:
        print(f"[env] torch import failed: {e}")
        sys.exit(1)

    if not os.getenv("HF_TOKEN") and not Path.home().joinpath(".huggingface", "token").exists():
        print("[env] WARNING: HF_TOKEN not set — Qwen3.5-0.8B download may fail")

    if not os.getenv("WANDB_API_KEY"):
        print("[env] note: WANDB_API_KEY not set — training will run without wandb logging")


def maybe_download_data(data_dir: Path, hours: float) -> Path:
    manifest = data_dir / "manifest.jsonl"
    if manifest.exists() and manifest.stat().st_size > 0:
        n = sum(1 for _ in manifest.open())
        print(f"[data] manifest already exists with {n} entries → skip download")
        return manifest
    print(f"[data] downloading up to {hours}h of speech…")
    subprocess.run(
        [
            sys.executable,
            "-m",
            "scripts.download_data",
            "--out",
            str(data_dir),
            "--librispeech",
            "--commonvoice",
            "--max-hours",
            str(hours),
        ],
        check=True,
    )
    return manifest


def smoke_test(backbone_tiny: str) -> None:
    print("[smoke] running 5-step tiny-model smoke…")
    subprocess.run(
        [sys.executable, "scripts/smoke_test.py"],
        check=True,
        env={**os.environ, "PYTHONNOUSERSITE": "1"},
    )
    print("[smoke] OK")


def launch_training(manifest: Path, output: Path, steps: int, backbone: str) -> None:
    output.mkdir(parents=True, exist_ok=True)
    print(f"[train] starting Stage 1: backbone={backbone} steps={steps} → {output}")
    cmd = [
        sys.executable,
        "-m",
        "voiceai.training.stage1_adapter",
        "--manifest",
        str(manifest),
        "--output",
        str(output),
        "--backbone",
        backbone,
        "--steps",
        str(steps),
    ]
    print(f"[train] cmd: {' '.join(cmd)}")
    subprocess.run(cmd, check=True, env={**os.environ, "PYTHONNOUSERSITE": "1"})


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("--output", type=Path, default=Path("runs/stage1"))
    p.add_argument("--data", type=Path, default=Path("data/stage1"))
    p.add_argument("--steps", type=int, default=30000)
    p.add_argument("--hours", type=float, default=100.0)
    p.add_argument("--backbone", default="Qwen/Qwen3.5-0.8B")
    p.add_argument("--skip-smoke", action="store_true")
    p.add_argument("--skip-download", action="store_true")
    args = p.parse_args()

    check_env()
    if not args.skip_smoke:
        smoke_test("hf-internal-testing/tiny-random-LlamaForCausalLM")
    if args.skip_download:
        manifest = args.data / "manifest.jsonl"
        if not manifest.exists():
            print(f"[fatal] no manifest at {manifest} and --skip-download given")
            sys.exit(1)
    else:
        manifest = maybe_download_data(args.data, args.hours)

    launch_training(manifest, args.output, args.steps, args.backbone)
    print(f"\n✓ Stage 1 done. Output: {args.output}/final")
    print(f"  Next: launch Stage 2 with --stage1 {args.output}/final")


if __name__ == "__main__":
    main()
