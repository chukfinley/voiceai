"""Standalone Mimi encoder — picks up any raw audio dirs without .npz yet.

Useful if you ran data generators with --backend kokoro (no encode), then
later want to encode-only on a different machine (e.g. faster CPU server
or rented cheap GPU).

Scans data/*/raw/ for paired _user.wav/_asst.wav files, encodes both with
Mimi, writes .npz/.json under data/*/encoded/.

CPU mode is supported but slow. For a quick burst rent RTX 4000 Ada $0.26/h.
"""
from __future__ import annotations

import argparse
import json
import re
from pathlib import Path

import numpy as np
import soundfile as sf
import torch
from tqdm.auto import tqdm


def find_pairs(root: Path) -> list[tuple[Path, Path, str]]:
    pairs = []
    for raw in root.rglob("raw"):
        if not raw.is_dir():
            continue
        for user_wav in raw.glob("*_user.wav"):
            sid = re.sub(r"_user\.wav$", "", user_wav.name)
            asst_wav = raw / f"{sid}_asst.wav"
            if asst_wav.exists():
                pairs.append((user_wav, asst_wav, sid))
    return pairs


def encode_pair(mimi, user_wav: Path, asst_wav: Path, device: str) -> tuple[np.ndarray, np.ndarray, float]:
    from voiceai.model.mimi_utils import mimi_encode, resample_to_mimi

    u, sr_u = sf.read(user_wav, dtype="float32")
    a, sr_a = sf.read(asst_wav, dtype="float32")
    if u.ndim > 1:
        u = u.mean(axis=1)
    if a.ndim > 1:
        a = a.mean(axis=1)
    n = max(len(u), len(a))
    if len(u) < n:
        u = np.pad(u, (0, n - len(u)))
    if len(a) < n:
        a = np.pad(a, (0, n - len(a)))
    dur = n / sr_u
    ut = torch.from_numpy(u).unsqueeze(0).unsqueeze(0).to(device)
    at = torch.from_numpy(a).unsqueeze(0).unsqueeze(0).to(device)
    ut = resample_to_mimi(ut, sr_u)
    at = resample_to_mimi(at, sr_a)
    dtype = next(mimi.parameters()).dtype
    ut = ut.to(dtype)
    at = at.to(dtype)
    with torch.no_grad():
        u_codes = mimi_encode(mimi, ut)[0].cpu().numpy()
        a_codes = mimi_encode(mimi, at)[0].cpu().numpy()
    return u_codes, a_codes, dur


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("--root", type=Path, default=Path("data"))
    p.add_argument("--device", default="cpu")
    p.add_argument("--limit", type=int, default=None)
    p.add_argument("--skip-existing", action="store_true", default=True)
    args = p.parse_args()

    from voiceai.model.mimi_utils import load_mimi
    from voiceai.training.data.dual_stream import DualStreamSample, save_sample as save_ds

    mimi = load_mimi(device=args.device)
    pairs = find_pairs(args.root)
    if args.limit:
        pairs = pairs[: args.limit]
    print(f"found {len(pairs)} pairs to encode")

    n_done = 0
    n_skipped = 0
    n_failed = 0
    for user_wav, asst_wav, sid in tqdm(pairs):
        enc_dir = user_wav.parent.parent / "encoded"
        enc_dir.mkdir(exist_ok=True)
        out_npz = enc_dir / f"{sid}.npz"
        if args.skip_existing and out_npz.exists():
            n_skipped += 1
            continue
        try:
            u_codes, a_codes, dur = encode_pair(mimi, user_wav, asst_wav, args.device)
        except Exception as e:
            print(f"fail {sid}: {e}")
            n_failed += 1
            continue
        category = user_wav.parent.parent.name
        ds = DualStreamSample(
            user_codes=u_codes,
            asst_codes=a_codes,
            text_ids=np.array([], dtype=np.int32),
            text_align=np.array([], dtype=np.int32),
            aux={"category": category},
            sample_id=sid,
            duration_s=dur,
        )
        save_ds(ds, enc_dir)
        n_done += 1

    print(f"\ndone: {n_done} encoded, {n_skipped} skipped, {n_failed} failed")


if __name__ == "__main__":
    main()
