#!/usr/bin/env python
"""Whisper-encoder ASR (SLAM-ASR recipe) — the grounding fix.

The Mimi-code prefix-LM did not ground (teacher forcing + raw acoustic tokens →
the LLM ignores the audio, WER >170%). This uses a frozen Whisper encoder
(pretrained on 680k h, semantically rich) → small trained bridge MLP → frozen
Qwen. Whisper features are near-transcript, so the bridge maps them into the LLM
space and ASR actually works.

Streams real LibriSpeech, trains the bridge, transcribes held-out (unseen) clips.

    uv run --extra train python scripts/colab_asr_whisper.py --push-to-hub
"""
from __future__ import annotations

import io
import json
import os
import subprocess
import sys
from pathlib import Path

REPO = Path(__file__).resolve().parent.parent
if str(REPO / "src") not in sys.path:
    sys.path.insert(0, str(REPO / "src"))
os.chdir(REPO)

DATA = REPO / "data" / "asr_whisper"
WAVS = DATA / "wav"
TRAIN_MANIFEST = DATA / "train.jsonl"
TEST_MANIFEST = DATA / "test.jsonl"
OUT = REPO / "runs" / "asr_whisper"


def build_data(n_train: int, n_test: int) -> None:
    if TRAIN_MANIFEST.exists() and TEST_MANIFEST.exists():
        if sum(1 for _ in TRAIN_MANIFEST.open()) >= n_train * 0.9:
            print(f"[data] reuse existing manifests")
            return
    import soundfile as sf
    from datasets import Audio, load_dataset

    print(f"[data] streaming {n_train + n_test} real LibriSpeech clips…")
    ds = load_dataset("openslr/librispeech_asr", "clean", split="train.100", streaming=True)
    ds = ds.cast_column("audio", Audio(decode=False))
    WAVS.mkdir(parents=True, exist_ok=True)
    rows = []
    for i, ex in enumerate(ds):
        if len(rows) >= n_train + n_test:
            break
        au = ex["audio"]
        arr, sr = sf.read(io.BytesIO(au["bytes"])) if au.get("bytes") else sf.read(au["path"])
        txt = (ex.get("text") or "").strip()
        if not txt:
            continue
        wav = WAVS / f"clip_{i:05d}.wav"
        sf.write(str(wav), arr, sr)
        rows.append({"audio": str(wav), "text": txt, "duration": round(len(arr) / sr, 2)})
    train, test = rows[:-n_test], rows[-n_test:]
    TRAIN_MANIFEST.write_text("".join(json.dumps(r) + "\n" for r in train))
    TEST_MANIFEST.write_text("".join(json.dumps(r) + "\n" for r in test))
    print(f"[data] {len(train)} train + {len(test)} held-out test clips")


def build_heldout(n_test: int, split: str = "test") -> None:
    """Stream a small held-out set (default LibriSpeech test.clean) for eval."""
    if TEST_MANIFEST.exists() and sum(1 for _ in TEST_MANIFEST.open()) >= n_test:
        print("[data] reuse held-out test manifest")
        return
    import soundfile as sf
    from datasets import Audio, load_dataset

    print(f"[data] streaming {n_test} held-out clips from {split}…")
    ds = load_dataset("openslr/librispeech_asr", "clean", split=split, streaming=True)
    ds = ds.cast_column("audio", Audio(decode=False))
    WAVS.mkdir(parents=True, exist_ok=True)
    rows = []
    for i, ex in enumerate(ds):
        if len(rows) >= n_test:
            break
        au = ex["audio"]
        arr, sr = sf.read(io.BytesIO(au["bytes"])) if au.get("bytes") else sf.read(au["path"])
        txt = (ex.get("text") or "").strip()
        if not txt:
            continue
        wav = WAVS / f"test_{i:05d}.wav"
        sf.write(str(wav), arr, sr)
        rows.append({"audio": str(wav), "text": txt})
    TEST_MANIFEST.write_text("".join(json.dumps(r) + "\n" for r in rows))
    print(f"[data] {len(rows)} held-out test clips")


def train(whisper_id: str, llm_id: str, steps: int, stream: bool = False,
          hf_split: str = "train.360", max_clips: int = 0, ckpt_every: int = 100000) -> None:
    cmd = [
        sys.executable, "-m", "voiceai.training.stage1_whisper",
        "--output", str(OUT),
        "--whisper-id", whisper_id,
        "--llm-id", llm_id,
        "--steps", str(steps),
        # frame stacking (downsample_k=5) cuts the prefix 1500->300, so batch 8 fits.
        "--batch-size", "8",
        "--grad-accum", "2",
        "--lr", "1e-4",
        "--warmup", "200",
        "--log-every", "50",
        "--ckpt-every", str(ckpt_every),
        "--device", "cuda",
        "--dtype", "bfloat16",
        "--wandb-disable",
    ]
    if stream:
        cmd += ["--hf-dataset", "openslr/librispeech_asr", "--hf-config", "clean",
                "--hf-split", hf_split, "--max-clips", str(max_clips), "--num-workers", "6"]
    else:
        cmd += ["--manifest", str(TRAIN_MANIFEST)]
    print("[train] Whisper-bridge ASR:\n  " + " ".join(cmd))
    env = {**os.environ, "PYTORCH_CUDA_ALLOC_CONF": "expandable_segments:True"}
    if subprocess.run(cmd, env=env).returncode != 0:
        sys.exit("[train] FAILED")


def evaluate() -> None:
    import torch

    from voiceai.model.whisper_lm import WhisperLM

    device = "cuda" if torch.cuda.is_available() else "cpu"
    dtype = torch.bfloat16 if device == "cuda" else torch.float32
    import soundfile as sf

    model = WhisperLM.from_pretrained(OUT / "final").to(device).eval()
    fe = model.feature_extractor
    tok = model.tokenizer

    def transcribe(rows, label):
        print(f"\n[eval] {label} — {len(rows)} clips:\n")
        refs, hyps = [], []
        for r in rows:
            arr, sr = sf.read(r["audio"], dtype="float32")
            if arr.ndim > 1:
                arr = arr.mean(axis=1)
            if sr != 16000:
                import librosa
                arr = librosa.resample(arr, orig_sr=sr, target_sr=16000)
            feats = fe(arr, sampling_rate=16000, return_tensors="pt").input_features.to(device).to(dtype)
            ids = model.generate(feats, max_new_tokens=120)
            hyp = tok.decode(ids, skip_special_tokens=True).strip()
            refs.append(r["text"].lower())
            hyps.append(hyp.lower())
            print(f"  REF: {r['text'][:90]}")
            print(f"  HYP: {hyp[:90]}\n")
        try:
            from jiwer import wer
            print(f"[eval] WER {label}: {wer(refs, hyps):.1%}")
        except Exception:
            pass

    if TRAIN_MANIFEST.exists():
        train_rows = [json.loads(l) for l in TRAIN_MANIFEST.open() if l.strip()][:6]
        transcribe(train_rows, "TRAIN clips (should fit)")
    test_rows = [json.loads(l) for l in TEST_MANIFEST.open() if l.strip()]
    transcribe(test_rows, "HELD-OUT clips (real ASR)")


def push_to_hub(repo: str | None) -> None:
    from huggingface_hub import HfApi

    final = OUT / "final"
    if not (final / "bridge.pt").exists():
        sys.exit(f"[hub] no checkpoint at {final}")
    api = HfApi(token=os.environ.get("HF_TOKEN"))
    if not repo:
        repo = f"{api.whoami()['name']}/voiceai-asr-whisper"
    api.create_repo(repo, exist_ok=True, private=True, repo_type="model")
    print(f"[hub] uploading {final} -> {repo} …")
    api.upload_folder(folder_path=str(final), repo_id=repo, repo_type="model")
    print(f"[hub] saved: https://huggingface.co/{repo}")


if __name__ == "__main__":
    import argparse

    p = argparse.ArgumentParser()
    p.add_argument("--whisper-id", default="openai/whisper-small")
    p.add_argument("--llm-id", default="Qwen/Qwen3-0.6B")
    p.add_argument("--clips", type=int, default=3000)
    p.add_argument("--test-clips", type=int, default=16)
    p.add_argument("--steps", type=int, default=6000)
    p.add_argument("--push-to-hub", action="store_true")
    p.add_argument("--hub-repo", default=None)
    # scaling: stream train data straight from HF (no pre-download) for big runs
    p.add_argument("--stream", action="store_true", help="stream train from LibriSpeech HF (for 100h+ runs)")
    p.add_argument("--hf-split", default="train.360", help="LibriSpeech train split to stream (train.100/train.360)")
    p.add_argument("--max-clips", type=int, default=0, help="cap streamed clips/epoch (0=all)")
    p.add_argument("--ckpt-every", type=int, default=100000)
    a = p.parse_args()
    if a.stream:
        build_heldout(a.test_clips)
        train(a.whisper_id, a.llm_id, a.steps, stream=True, hf_split=a.hf_split,
              max_clips=a.max_clips, ckpt_every=a.ckpt_every)
    else:
        build_data(a.clips, a.test_clips)
        train(a.whisper_id, a.llm_id, a.steps, ckpt_every=a.ckpt_every)
    evaluate()
    if a.push_to_hub:
        push_to_hub(a.hub_repo)
