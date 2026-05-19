"""Pre-encode all Stage 1 LibriSpeech audio with Mimi.

Writes one .npy per clip with int16 codes [K, T_frames]. Updates manifest
with a `codes` field pointing at the .npy. Speeds up training ~3x by
removing Mimi.encode from the data loader hot path (lets num_workers>0).
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import soundfile as sf
import torch
from tqdm import tqdm


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("--manifest", type=Path, default=Path("data/stage1/manifest.jsonl"))
    p.add_argument("--out-dir", type=Path, default=Path("data/stage1/codes"))
    p.add_argument("--out-manifest", type=Path, default=Path("data/stage1/manifest_encoded.jsonl"))
    p.add_argument("--device", default="cuda")
    p.add_argument("--max-audio-s", type=float, default=20.0)
    args = p.parse_args()

    from voiceai.model.mimi_utils import load_mimi, resample_to_mimi

    args.out_dir.mkdir(parents=True, exist_ok=True)

    mimi = load_mimi(device=args.device)
    mimi_dtype = next(mimi.parameters()).dtype

    with open(args.manifest) as f:
        rows = [json.loads(l) for l in f if l.strip()]

    out_f = open(args.out_manifest, "w")
    done = skipped = failed = 0
    for meta in tqdm(rows):
        audio_path = Path(meta["audio"])
        out_npy = args.out_dir / (audio_path.stem + ".npy")
        if out_npy.exists():
            meta["codes"] = str(out_npy)
            out_f.write(json.dumps(meta) + "\n")
            skipped += 1
            continue
        try:
            audio, sr = sf.read(audio_path, dtype="float32")
            if audio.ndim > 1:
                audio = audio.mean(axis=1)
            if len(audio) / sr > args.max_audio_s:
                audio = audio[: int(args.max_audio_s * sr)]
            t = torch.from_numpy(audio).unsqueeze(0).unsqueeze(0)
            t = resample_to_mimi(t, sr)
            t = t.to(device=args.device, dtype=mimi_dtype)
            with torch.no_grad():
                codes = mimi.encode(t)[0].cpu().numpy().astype(np.int16)
            np.save(out_npy, codes)
            meta["codes"] = str(out_npy)
            out_f.write(json.dumps(meta) + "\n")
            out_f.flush()
            done += 1
        except Exception as e:
            print(f"fail {audio_path.name}: {e}")
            failed += 1
            continue

    out_f.close()
    print(f"done: {done} encoded, {skipped} already had codes, {failed} failed")
    print(f"wrote manifest: {args.out_manifest}")


if __name__ == "__main__":
    main()
