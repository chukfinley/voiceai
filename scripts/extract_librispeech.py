"""Robust LibriSpeech extractor: reads parquet shards, writes manifest line
by line, resumable.

Already cached parquet shards in HF cache → just extracts audio+text.
"""
from __future__ import annotations

import argparse
import io
import json
import os
import sys
from pathlib import Path


def main() -> None:
    os.environ.setdefault("HF_HUB_ENABLE_HF_TRANSFER", "1")
    p = argparse.ArgumentParser()
    p.add_argument("--out", type=Path, required=True)
    p.add_argument("--max-hours", type=float, default=100)
    p.add_argument("--resume", action="store_true", default=True)
    args = p.parse_args()

    import pyarrow.parquet as pq
    import soundfile as sf
    from huggingface_hub import snapshot_download

    # Make sure shards are cached
    cache = snapshot_download(
        repo_id="openslr/librispeech_asr",
        repo_type="dataset",
        allow_patterns=["clean/train.100/*.parquet"],
        max_workers=8,
    )
    pq_files = sorted(Path(cache).glob("clean/train.100/*.parquet"))
    print(f"[{len(pq_files)} shards cached at {cache}]", flush=True)

    audio_dir = args.out / "librispeech"
    audio_dir.mkdir(parents=True, exist_ok=True)
    manifest_path = args.out / "manifest.jsonl"

    # Determine starting index
    existing = sorted(audio_dir.glob("ls_*.flac"))
    start_idx = 0
    if args.resume and existing:
        last = existing[-1].stem  # ls_NNNNNNN
        try:
            start_idx = int(last.split("_")[1]) + 1
        except Exception:
            start_idx = len(existing)
    print(f"[resume from idx {start_idx}, {len(existing)} files already on disk]", flush=True)

    mf = manifest_path.open("a", buffering=1)
    total_s = 0.0
    written = 0
    idx = start_idx
    every = 100

    for shard_i, pq_path in enumerate(pq_files):
        if total_s / 3600 >= args.max_hours:
            break
        print(f"[shard {shard_i+1}/{len(pq_files)}] {pq_path.name}", flush=True)
        try:
            tbl = pq.read_table(pq_path)
        except Exception as e:
            print(f"  skip {pq_path}: {e}", flush=True)
            continue
        for row in range(tbl.num_rows):
            if total_s / 3600 >= args.max_hours:
                break
            out_path = audio_dir / f"ls_{idx:07d}.flac"
            idx += 1
            if out_path.exists():
                # Re-read duration from existing file
                try:
                    info = sf.info(out_path)
                    total_s += info.frames / info.samplerate
                    written += 1
                except Exception:
                    pass
                continue
            try:
                audio_struct = tbl.column("audio")[row].as_py()
                text = str(tbl.column("text")[row].as_py())
                raw = audio_struct.get("bytes")
                if raw is None and audio_struct.get("path"):
                    raw = Path(audio_struct["path"]).read_bytes()
                if not raw:
                    continue
                audio, sr = sf.read(io.BytesIO(raw), dtype="float32")
                if audio.ndim > 1:
                    audio = audio.mean(axis=1)
                sf.write(out_path, audio, sr, format="FLAC")
                dur = len(audio) / sr
                mf.write(
                    json.dumps(
                        {"audio": str(out_path), "text": text, "duration": dur, "source": "librispeech"}
                    )
                    + "\n"
                )
                total_s += dur
                written += 1
                if written % every == 0:
                    print(f"  {written} written, {total_s/3600:.2f}h", flush=True)
            except Exception as e:
                print(f"  row {row} fail: {e}", flush=True)
                continue
    mf.close()
    print(f"[DONE] {written} files, {total_s/3600:.2f}h total → {manifest_path}", flush=True)


if __name__ == "__main__":
    main()
