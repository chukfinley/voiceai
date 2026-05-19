"""Fast HF dataset downloader using snapshot_download (parallel raw fetch).

Bypasses the slow `datasets.load_dataset` route. Downloads parquet shards
directly via huggingface_hub which is multi-threaded and saturates the
network.

Usage:
    HF_HUB_ENABLE_HF_TRANSFER=1 uv run python scripts/fast_download.py \\
        --out data/stage1 \\
        --hours 100
"""
from __future__ import annotations

import argparse
import io
import json
import os
from pathlib import Path

from huggingface_hub import snapshot_download
from tqdm.auto import tqdm


def parquet_files_for_librispeech() -> list[str]:
    """LibriSpeech train.100 is split across ~20 parquet shards."""
    return [f"clean/train.100/{i:04d}.parquet" for i in range(8)]


def fetch_librispeech(out: Path, max_hours: float) -> list[dict]:
    import pyarrow.parquet as pq
    import soundfile as sf

    cache = snapshot_download(
        repo_id="openslr/librispeech_asr",
        repo_type="dataset",
        allow_patterns=["clean/train.100/*.parquet"],
        max_workers=8,
    )
    print(f"librispeech cache: {cache}")
    pq_files = sorted(Path(cache).glob("clean/train.100/*.parquet"))
    print(f"found {len(pq_files)} parquet shards")

    audio_dir = out / "librispeech"
    audio_dir.mkdir(parents=True, exist_ok=True)
    entries: list[dict] = []
    total_s = 0.0
    pbar = tqdm(total=int(max_hours * 3600), unit="audio_s", desc="librispeech")
    idx = 0
    for pq_path in pq_files:
        if total_s / 3600 >= max_hours:
            break
        try:
            table = pq.read_table(pq_path)
        except Exception as e:
            print(f"skip {pq_path}: {e}")
            continue
        for row in range(table.num_rows):
            if total_s / 3600 >= max_hours:
                break
            try:
                audio_field = table.column("audio")[row].as_py()
                text = str(table.column("text")[row].as_py())
                raw = audio_field.get("bytes")
                if raw is None and audio_field.get("path"):
                    raw = Path(audio_field["path"]).read_bytes()
                if raw is None:
                    continue
                audio, sr = sf.read(io.BytesIO(raw), dtype="float32")
            except Exception:
                continue
            out_path = audio_dir / f"ls_{idx:07d}.flac"
            sf.write(out_path, audio, sr, format="FLAC")
            dur = len(audio) / sr
            entries.append({"audio": str(out_path), "text": text, "duration": dur, "source": "librispeech"})
            total_s += dur
            idx += 1
            pbar.update(int(dur))
    pbar.close()
    return entries


def fetch_commonvoice(out: Path, max_hours: float) -> list[dict]:
    import pyarrow.parquet as pq
    import soundfile as sf

    cache = snapshot_download(
        repo_id="mozilla-foundation/common_voice_17_0",
        repo_type="dataset",
        allow_patterns=["transcript/en/*.tsv", "audio/en/train/*.tar"],
        max_workers=8,
    )
    print(f"commonvoice cache: {cache}")
    # CommonVoice ships as tar files of mp3s — different shape.
    # For now, fallback to using the parquet variant via datasets library
    # but with non-streaming + parallel.
    from datasets import Audio, load_dataset

    ds = load_dataset(
        "mozilla-foundation/common_voice_17_0",
        "en",
        split="train",
        trust_remote_code=True,
        num_proc=8,
    )
    ds = ds.cast_column("audio", Audio(decode=False))
    audio_dir = out / "commonvoice"
    audio_dir.mkdir(parents=True, exist_ok=True)
    entries: list[dict] = []
    total_s = 0.0
    pbar = tqdm(total=int(max_hours * 3600), unit="audio_s", desc="commonvoice")
    for i, ex in enumerate(ds):
        if total_s / 3600 >= max_hours:
            break
        ab = ex["audio"]
        raw = ab.get("bytes")
        if raw is None and ab.get("path"):
            with open(ab["path"], "rb") as f:
                raw = f.read()
        if raw is None:
            continue
        try:
            audio, sr = sf.read(io.BytesIO(raw), dtype="float32")
        except Exception:
            continue
        out_path = audio_dir / f"cv_{i:07d}.wav"
        sf.write(out_path, audio, sr)
        dur = len(audio) / sr
        entries.append({"audio": str(out_path), "text": ex["sentence"], "duration": dur, "source": "commonvoice"})
        total_s += dur
        pbar.update(int(dur))
    pbar.close()
    return entries


def main() -> None:
    os.environ.setdefault("HF_HUB_ENABLE_HF_TRANSFER", "1")
    p = argparse.ArgumentParser()
    p.add_argument("--out", type=Path, required=True)
    p.add_argument("--hours-ls", type=float, default=100)
    p.add_argument("--hours-cv", type=float, default=0, help="0 = skip commonvoice")
    args = p.parse_args()

    args.out.mkdir(parents=True, exist_ok=True)
    all_entries = []

    print("=== LibriSpeech ===")
    all_entries += fetch_librispeech(args.out, args.hours_ls)

    if args.hours_cv > 0:
        print("=== CommonVoice ===")
        all_entries += fetch_commonvoice(args.out, args.hours_cv)

    manifest = args.out / "manifest.jsonl"
    with manifest.open("w") as f:
        for e in all_entries:
            f.write(json.dumps(e) + "\n")
    print(f"wrote {len(all_entries)} entries → {manifest}")


if __name__ == "__main__":
    main()
