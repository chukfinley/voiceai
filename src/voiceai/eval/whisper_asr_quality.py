"""Eval the Whisper-bridge ASR model."""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import soundfile as sf
import torch


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("--model", required=True, type=Path)
    p.add_argument("--manifest", required=True, type=Path)
    p.add_argument("--out", type=Path, default=None)
    p.add_argument("--device", default="cuda")
    p.add_argument("--n", type=int, default=50)
    args = p.parse_args()

    from ..model.whisper_lm import WhisperLM
    from jiwer import wer as wer_fn

    model = WhisperLM.from_pretrained(args.model).to(args.device).eval()

    with args.manifest.open() as f:
        rows = [json.loads(line) for line in f if line.strip()]
    rows = rows[: args.n]

    refs, hyps = [], []
    for i, ex in enumerate(rows):
        try:
            audio, sr = sf.read(ex["audio"], dtype="float32")
        except Exception:
            continue
        if audio.ndim > 1:
            audio = audio.mean(axis=1)
        if sr != 16000:
            import librosa
            audio = librosa.resample(audio, orig_sr=sr, target_sr=16000)
        feats = model.feature_extractor(audio, sampling_rate=16000, return_tensors="pt").input_features
        feats = feats.to(args.device).to(torch.bfloat16)
        ids = model.generate(feats, max_new_tokens=120)
        pred = model.tokenizer.decode(ids, skip_special_tokens=True)
        refs.append(ex["text"].strip().lower())
        hyps.append(pred.strip().lower())
        if i < 5:
            print(f"REF: {refs[-1][:100]}")
            print(f"HYP: {hyps[-1][:100]}")
            print()

    score = float(wer_fn(refs, hyps)) if refs else 1.0
    result = {"wer": score, "n": len(refs), "sample_hyps": hyps[:10], "sample_refs": refs[:10]}
    print(f"WER: {score:.3f} ({len(refs)} samples)")
    if args.out:
        args.out.parent.mkdir(parents=True, exist_ok=True)
        args.out.write_text(json.dumps(result, indent=2))


if __name__ == "__main__":
    main()
