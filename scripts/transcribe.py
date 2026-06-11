#!/usr/bin/env python
"""Transcribe a wav file with the trained Whisper-bridge ASR model.

    uv run --extra train python scripts/transcribe.py path/to/audio.wav

Loads the trained bridge from HuggingFace (chukfinley/voiceai-asr-whisper) by
default — only the ~10MB bridge is downloaded; the frozen Whisper encoder and
Qwen backbone are pulled from their own repos. Pass --model <dir> to use a local
checkpoint (e.g. runs/asr_whisper_final).
"""
from __future__ import annotations

import sys
from pathlib import Path

REPO = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO / "src"))


def main() -> None:
    import argparse

    p = argparse.ArgumentParser()
    p.add_argument("wav", type=str, help="path to a 16kHz (or any) wav file")
    p.add_argument("--model", default="runs/asr_whisper_final",
                   help="local checkpoint dir with bridge.pt, or 'hub' to pull from HF")
    p.add_argument("--hub-repo", default="chukfinley/voiceai-asr-whisper")
    p.add_argument("--max-new-tokens", type=int, default=200)
    a = p.parse_args()

    import soundfile as sf
    import torch

    from voiceai.model.whisper_lm import WhisperLM

    if a.model == "hub" or not (Path(a.model) / "bridge.pt").exists():
        from huggingface_hub import snapshot_download
        local = snapshot_download(repo_id=a.hub_repo, repo_type="model")
    else:
        local = a.model

    device = "cuda" if torch.cuda.is_available() else "cpu"
    dtype = torch.bfloat16 if device == "cuda" else torch.float32
    model = WhisperLM.from_pretrained(local).to(device).eval()

    arr, sr = sf.read(a.wav, dtype="float32")
    if arr.ndim > 1:
        arr = arr.mean(axis=1)
    if sr != 16000:
        import librosa
        arr = librosa.resample(arr, orig_sr=sr, target_sr=16000)
    feats = model.feature_extractor(arr, sampling_rate=16000, return_tensors="pt").input_features
    feats = feats.to(device).to(dtype)
    ids = model.generate(feats, max_new_tokens=a.max_new_tokens)
    text = model.tokenizer.decode(ids, skip_special_tokens=True).strip()
    print(text)


if __name__ == "__main__":
    main()
