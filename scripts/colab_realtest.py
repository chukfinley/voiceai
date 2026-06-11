#!/usr/bin/env python
"""One-shot REAL mini training run — Colab T4, single command.

Unlike the smoke test (random-weight stand-in model, learns nothing), this is
a *real* training run:
  - downloads ~64 real LibriSpeech clips (real speech + real transcripts)
  - builds a manifest, lazy Mimi-encodes them
  - trains the Stage-1 audio adapter on the REAL Qwen3.5-0.8B backbone
  - real audio -> real text cross-entropy loss that actually goes down

Run once:
    python scripts/colab_realtest.py

Default backbone is Qwen3-0.6B (recognised by current transformers). The repo's
production target Qwen3.5-0.8B (model_type `qwen3_5`) needs transformers-from-
source; pass it explicitly once that's installed:
    python scripts/colab_realtest.py --backbone Qwen/Qwen3.5-0.8B
"""
from __future__ import annotations

import json
import os
import subprocess
import sys
from pathlib import Path

REPO = Path(__file__).resolve().parent.parent
SRC = REPO / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))
os.chdir(REPO)

DATA = REPO / "data" / "realtest"
WAVS = DATA / "wav"
MANIFEST = DATA / "manifest.jsonl"
OUT = REPO / "runs" / "realtest"


def build_data(max_clips: int) -> None:
    if MANIFEST.exists() and sum(1 for _ in MANIFEST.open()) >= 8:
        print(f"[data] manifest already exists: {MANIFEST}")
        return
    import io

    import soundfile as sf
    from datasets import Audio, load_dataset

    print("[data] downloading real LibriSpeech sample…")
    ds = load_dataset(
        "hf-internal-testing/librispeech_asr_dummy", "clean", split="validation"
    )
    # Don't let `datasets` auto-decode audio (newer versions need torchcodec);
    # read the raw bytes ourselves with soundfile instead.
    ds = ds.cast_column("audio", Audio(decode=False))
    WAVS.mkdir(parents=True, exist_ok=True)
    n = 0
    with MANIFEST.open("w") as f:
        for i, ex in enumerate(ds):
            if n >= max_clips:
                break
            audio = ex["audio"]
            if audio.get("bytes"):
                arr, sr = sf.read(io.BytesIO(audio["bytes"]))
            else:
                arr, sr = sf.read(audio["path"])
            text = (ex.get("text") or "").strip()
            if not text:
                continue
            wav_path = WAVS / f"clip_{i:04d}.wav"
            sf.write(str(wav_path), arr, sr)
            f.write(
                json.dumps(
                    {"audio": str(wav_path), "text": text, "duration": round(len(arr) / sr, 2)}
                )
                + "\n"
            )
            n += 1
    print(f"[data] wrote {n} REAL clips -> {MANIFEST}")


def train(backbone: str, steps: int) -> None:
    cmd = [
        sys.executable, "-m", "voiceai.training.stage1_adapter",
        "--manifest", str(MANIFEST),
        "--output", str(OUT),
        "--backbone", backbone,
        "--steps", str(steps),
        "--batch-size", "2",
        "--grad-accum", "2",
        "--lr", "3e-4",
        "--warmup", "20",
        "--log-every", "10",
        "--ckpt-every", "100000",  # no mid-run ckpt for a short test
        "--max-audio-s", "12",
        "--device", "cuda",
        "--dtype", "bfloat16",
        "--wandb-disable",
    ]
    print("[train] REAL stage-1 adapter pretrain:\n  " + " ".join(cmd))
    if subprocess.run(cmd).returncode != 0:
        sys.exit("[train] REAL test FAILED")
    print(f"\n[done] adapters -> {OUT}/   loss should have dropped over {steps} steps.")


if __name__ == "__main__":
    import argparse

    p = argparse.ArgumentParser()
    p.add_argument("--backbone", default="Qwen/Qwen3-0.6B")
    p.add_argument("--steps", type=int, default=200)
    p.add_argument("--max-clips", type=int, default=64)
    a = p.parse_args()
    build_data(a.max_clips)
    train(a.backbone, a.steps)
