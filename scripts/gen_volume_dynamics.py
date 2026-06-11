"""Volume-dynamics augmentation: shouting / whispering variants of ASR data.

Takes an existing manifest.jsonl and writes loud/quiet variants with
<emo:shouting> / <emo:whispering> tags baked into the text target, so Stage 1
learns to NOTICE how loud the user is, not just what they said.

DSP is a proxy (gain + soft-clip for shouting, attenuation + noise floor for
whispering — real shouting also shifts spectral tilt). Real acted emotion
comes from crema_d/meld; this script covers the pure loudness axis cheaply
and at scale.

Usage:
    uv run python scripts/gen_volume_dynamics.py \
        --manifest data/hf/librispeech/manifest.jsonl \
        --out data/volume_dynamics --max-samples 5000
"""
from __future__ import annotations

import argparse
import json
import random
from pathlib import Path

import numpy as np
import soundfile as sf
from tqdm.auto import tqdm


def make_shout(audio: np.ndarray, rng: random.Random) -> np.ndarray:
    gain = rng.uniform(4.0, 8.0)
    # tanh soft clip ≈ the compression/distortion of a loud voice into a mic
    return np.tanh(audio * gain).astype(np.float32)


def make_whisper(audio: np.ndarray, rng: random.Random) -> np.ndarray:
    gain = rng.uniform(0.05, 0.15)
    out = audio * gain
    # mild noise floor so it doesn't just look like "same clip, lower volume"
    out = out + rng.uniform(0.001, 0.003) * np.random.default_rng(rng.randrange(2**31)).standard_normal(len(audio)).astype(np.float32)
    return out.astype(np.float32)


VARIANTS = {
    "shouting": make_shout,
    "whispering": make_whisper,
}


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("--manifest", required=True, type=Path)
    p.add_argument("--out", required=True, type=Path)
    p.add_argument("--max-samples", type=int, default=5000,
                   help="source samples to augment (each yields one random variant)")
    p.add_argument("--seed", type=int, default=0)
    args = p.parse_args()

    rng = random.Random(args.seed)
    audio_dir = args.out / "audio"
    audio_dir.mkdir(parents=True, exist_ok=True)

    lines = [l for l in args.manifest.read_text().splitlines() if l.strip()]
    rng.shuffle(lines)
    lines = lines[: args.max_samples]

    n = 0
    with (args.out / "manifest.jsonl").open("w") as f:
        for line in tqdm(lines, desc="volume variants"):
            try:
                meta = json.loads(line)
                audio, sr = sf.read(meta["audio"], dtype="float32")
            except Exception:
                continue
            if audio.ndim > 1:
                audio = audio.mean(axis=1)
            # skip clips that already carry an emotion tag
            if meta.get("text", "").startswith("<emo:"):
                continue
            kind = rng.choice(list(VARIANTS))
            out_audio = VARIANTS[kind](audio, rng)
            wav_path = audio_dir / f"{n:08d}_{kind}.wav"
            sf.write(wav_path, out_audio, sr)
            f.write(json.dumps({
                "audio": str(wav_path),
                "text": f"<emo:{kind}> {meta.get('text', '')}".strip(),
                "duration": len(out_audio) / sr,
                "emotion": kind,
                "source": f"volume_dynamics/{meta.get('source', '')}",
            }) + "\n")
            n += 1

    print(f"wrote {n} variants → {args.out / 'manifest.jsonl'}")


if __name__ == "__main__":
    main()
