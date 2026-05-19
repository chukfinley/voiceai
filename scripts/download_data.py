"""Download base corpora for Stage 1 audio adapter pretraining.

Pulls LibriSpeech-clean-100 (English) + Common Voice 17 English subset.
Builds a unified manifest JSONL.
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("--out", type=Path, required=True)
    p.add_argument("--librispeech", action="store_true")
    p.add_argument("--commonvoice", action="store_true")
    p.add_argument("--cv-split", default="train")
    p.add_argument("--max-hours", type=float, default=200.0)
    args = p.parse_args()

    args.out.mkdir(parents=True, exist_ok=True)
    manifest = args.out / "manifest.jsonl"
    entries = []

    if args.librispeech:
        entries.extend(_load_librispeech(args.out / "librispeech", args.max_hours))
    if args.commonvoice:
        entries.extend(_load_commonvoice(args.out / "commonvoice", args.cv_split, args.max_hours))

    with manifest.open("w") as f:
        for e in entries:
            f.write(json.dumps(e) + "\n")
    print(f"wrote {len(entries)} entries → {manifest}")


def _load_librispeech(out_dir: Path, max_hours: float) -> list[dict]:
    from datasets import load_dataset
    import soundfile as sf

    out_dir.mkdir(parents=True, exist_ok=True)
    ds = load_dataset("openslr/librispeech_asr", "clean", split="train.100", streaming=True)
    total_s = 0
    entries = []
    for i, ex in enumerate(ds):
        if total_s / 3600 >= max_hours:
            break
        wav_path = out_dir / f"ls_{i:06d}.wav"
        audio = ex["audio"]["array"]
        sr = ex["audio"]["sampling_rate"]
        sf.write(wav_path, audio, sr)
        entries.append({"audio": str(wav_path), "text": ex["text"], "duration": len(audio) / sr})
        total_s += len(audio) / sr
    return entries


def _load_commonvoice(out_dir: Path, split: str, max_hours: float) -> list[dict]:
    from datasets import load_dataset
    import soundfile as sf

    out_dir.mkdir(parents=True, exist_ok=True)
    ds = load_dataset(
        "mozilla-foundation/common_voice_17_0",
        "en",
        split=split,
        streaming=True,
        trust_remote_code=True,
    )
    total_s = 0
    entries = []
    for i, ex in enumerate(ds):
        if total_s / 3600 >= max_hours:
            break
        wav_path = out_dir / f"cv_{i:06d}.wav"
        audio = ex["audio"]["array"]
        sr = ex["audio"]["sampling_rate"]
        sf.write(wav_path, audio, sr)
        entries.append({"audio": str(wav_path), "text": ex["sentence"], "duration": len(audio) / sr})
        total_s += len(audio) / sr
    return entries


if __name__ == "__main__":
    main()
