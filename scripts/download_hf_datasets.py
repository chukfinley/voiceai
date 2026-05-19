"""Download HuggingFace datasets for all training stages.

Each dataset target:
  - audio_adapter (Stage 1):  librispeech, common_voice, peoples_speech, gigaspeech
  - dual_stream (Stage 2):    spokenwoz, meld, intrinsicvoice
  - text_dialog (synth seed): daily_dialog, dailydialog++

Pick what you want via --datasets librispeech common_voice spokenwoz ...

Outputs:
  - data/hf/<dataset>/raw/  audio files extracted
  - data/hf/<dataset>/manifest.jsonl  unified manifest
  - if --encode-mimi: data/hf/<dataset>/encoded/<id>.npz dual-stream samples
"""
from __future__ import annotations

import argparse
import json
import os
from collections.abc import Iterator
from pathlib import Path
from typing import Callable

import numpy as np
import torch
from tqdm.auto import tqdm


# ---------------------------------------------------------------------------
# Dataset adapters: each yields dicts with {audio: np.ndarray, sr: int, text: str, ...}
# ---------------------------------------------------------------------------


def _librispeech(split: str = "train.clean.100", max_hours: float = 100) -> Iterator[dict]:
    from datasets import load_dataset

    ds = load_dataset("openslr/librispeech_asr", "clean", split=split, streaming=True)
    total = 0
    for ex in ds:
        if total / 3600 >= max_hours:
            break
        a = ex["audio"]
        yield {
            "audio": np.asarray(a["array"], dtype=np.float32),
            "sr": a["sampling_rate"],
            "text": ex["text"],
            "source": "librispeech",
        }
        total += len(a["array"]) / a["sampling_rate"]


def _common_voice(split: str = "train", max_hours: float = 100) -> Iterator[dict]:
    from datasets import load_dataset

    ds = load_dataset(
        "mozilla-foundation/common_voice_17_0",
        "en",
        split=split,
        streaming=True,
        trust_remote_code=True,
    )
    total = 0
    for ex in ds:
        if total / 3600 >= max_hours:
            break
        a = ex["audio"]
        if not a or "array" not in a:
            continue
        yield {
            "audio": np.asarray(a["array"], dtype=np.float32),
            "sr": a["sampling_rate"],
            "text": ex["sentence"],
            "source": "common_voice",
        }
        total += len(a["array"]) / a["sampling_rate"]


def _peoples_speech(split: str = "train", max_hours: float = 100) -> Iterator[dict]:
    from datasets import load_dataset

    ds = load_dataset(
        "MLCommons/peoples_speech",
        "clean",
        split=split,
        streaming=True,
        trust_remote_code=True,
    )
    total = 0
    for ex in ds:
        if total / 3600 >= max_hours:
            break
        a = ex["audio"]
        yield {
            "audio": np.asarray(a["array"], dtype=np.float32),
            "sr": a["sampling_rate"],
            "text": ex.get("text", ""),
            "source": "peoples_speech",
        }
        total += len(a["array"]) / a["sampling_rate"]


def _spokenwoz(split: str = "train", max_dialogs: int = 1000) -> Iterator[dict]:
    """SpokenWOZ — paired task-oriented dialog audio.

    Each example contains channel-separated user and system audio.
    """
    from datasets import load_dataset

    ds = load_dataset("Spoken-WOZ/spokenwoz", split=split, streaming=True, trust_remote_code=True)
    for i, ex in enumerate(ds):
        if i >= max_dialogs:
            break
        try:
            user_a = ex.get("user_audio") or ex.get("audio_user")
            sys_a = ex.get("system_audio") or ex.get("audio_system")
            if user_a is None or sys_a is None:
                continue
            yield {
                "user_audio": np.asarray(user_a["array"], dtype=np.float32),
                "asst_audio": np.asarray(sys_a["array"], dtype=np.float32),
                "sr": user_a["sampling_rate"],
                "dialog_id": ex.get("dialog_id", str(i)),
                "source": "spokenwoz",
            }
        except Exception:
            continue


def _intrinsicvoice(max_samples: int = 5000) -> Iterator[dict]:
    """IntrinsicVoice-500k synth speech-to-speech pairs."""
    from datasets import load_dataset

    try:
        ds = load_dataset("OpenS2S/IntrinsicVoice-500k", split="train", streaming=True, trust_remote_code=True)
    except Exception:
        return
    for i, ex in enumerate(ds):
        if i >= max_samples:
            break
        try:
            user_a = ex.get("user_audio") or ex.get("input_audio")
            asst_a = ex.get("assistant_audio") or ex.get("output_audio")
            if user_a is None or asst_a is None:
                continue
            yield {
                "user_audio": np.asarray(user_a["array"], dtype=np.float32),
                "asst_audio": np.asarray(asst_a["array"], dtype=np.float32),
                "sr": user_a["sampling_rate"],
                "source": "intrinsicvoice",
            }
        except Exception:
            continue


def _daily_dialog(max_samples: int = 5000) -> Iterator[dict]:
    """DailyDialog — text-only, but we use scripts as seed for TTS rendering."""
    from datasets import load_dataset

    ds = load_dataset("daily_dialog", split="train", trust_remote_code=True)
    for i, ex in enumerate(ds):
        if i >= max_samples:
            break
        yield {
            "turns": ex["dialog"],
            "source": "daily_dialog",
        }


DATASETS: dict[str, Callable] = {
    "librispeech": _librispeech,
    "common_voice": _common_voice,
    "peoples_speech": _peoples_speech,
    "spokenwoz": _spokenwoz,
    "intrinsicvoice": _intrinsicvoice,
    "daily_dialog": _daily_dialog,
}

ADAPTER_DATASETS = {"librispeech", "common_voice", "peoples_speech"}
PAIRED_DATASETS = {"spokenwoz", "intrinsicvoice"}
TEXT_DATASETS = {"daily_dialog"}


def save_adapter(out: Path, items: Iterator[dict]) -> None:
    """Save single-stream (audio, transcript) for Stage 1."""
    import soundfile as sf

    out.mkdir(parents=True, exist_ok=True)
    audio_dir = out / "audio"
    audio_dir.mkdir(exist_ok=True)
    manifest = out / "manifest.jsonl"
    with manifest.open("w") as f:
        for i, item in enumerate(tqdm(items, desc=f"writing {out.name}")):
            wav_path = audio_dir / f"{i:08d}.wav"
            sf.write(wav_path, item["audio"], item["sr"])
            f.write(
                json.dumps(
                    {
                        "audio": str(wav_path),
                        "text": item.get("text", ""),
                        "duration": len(item["audio"]) / item["sr"],
                        "source": item.get("source", ""),
                    }
                )
                + "\n"
            )


def save_paired(out: Path, items: Iterator[dict], mimi, device: str) -> None:
    """Save dual-stream samples (encoded with Mimi if mimi given)."""
    import soundfile as sf

    from voiceai.training.data.mixing import encode_dual_stream, save_dual_stream_sample

    raw_dir = out / "raw"
    raw_dir.mkdir(parents=True, exist_ok=True)
    enc_dir = out / "encoded"
    enc_dir.mkdir(parents=True, exist_ok=True)
    metas = []
    for i, item in enumerate(tqdm(items, desc=f"paired {out.name}")):
        sid = f"hf_{out.name}_{i:08d}"
        u = item["user_audio"]
        a = item["asst_audio"]
        if len(u) == 0 or len(a) == 0:
            continue
        sf.write(raw_dir / f"{sid}_user.wav", u, item["sr"])
        sf.write(raw_dir / f"{sid}_asst.wav", a, item["sr"])
        meta = {
            "sample_id": sid,
            "duration_s": float(max(len(u), len(a)) / item["sr"]),
            "source": item.get("source", out.name),
            "category": f"hf_{out.name}",
        }
        metas.append(meta)
        if mimi is not None:
            try:
                u_codes, a_codes = encode_dual_stream(u, a, mimi, sr=item["sr"], device=device)
                save_dual_stream_sample(
                    user_codes=u_codes,
                    asst_codes=a_codes,
                    text_ids=np.array([], dtype=np.int32),
                    text_align=np.array([], dtype=np.int32),
                    aux={"source": item.get("source"), "category": meta["category"]},
                    sample_id=sid,
                    out_root=enc_dir,
                    duration_s=meta["duration_s"],
                )
            except Exception as e:
                print(f"encode fail {sid}: {e}")
    (out / "samples.jsonl").write_text("\n".join(json.dumps(m) for m in metas))


def save_text_seed(out: Path, items: Iterator[dict]) -> None:
    """Save text dialog seeds for later TTS rendering by gen_diverse_dialogs."""
    out.mkdir(parents=True, exist_ok=True)
    with (out / "scripts.jsonl").open("w") as f:
        for item in tqdm(items, desc=f"text {out.name}"):
            turns = item.get("turns", [])
            if not turns:
                continue
            structured = [
                {"role": "user" if i % 2 == 0 else "assistant", "text": t.strip()}
                for i, t in enumerate(turns)
            ]
            f.write(json.dumps({"title": "daily_dialog", "turns": structured, "source": "daily_dialog"}) + "\n")


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("--out", type=Path, default=Path("data/hf"))
    p.add_argument(
        "--datasets",
        nargs="+",
        default=["librispeech", "common_voice"],
        choices=list(DATASETS.keys()),
    )
    p.add_argument("--max-hours", type=float, default=100.0, help="for adapter datasets")
    p.add_argument("--max-samples", type=int, default=5000, help="for paired/text datasets")
    p.add_argument("--encode-mimi", action="store_true")
    p.add_argument("--device", default="cuda")
    p.add_argument("--combine-adapter-manifest", action="store_true")
    args = p.parse_args()

    args.out.mkdir(parents=True, exist_ok=True)

    mimi = None
    if args.encode_mimi and any(d in PAIRED_DATASETS for d in args.datasets):
        from voiceai.model.mimi_utils import load_mimi

        mimi = load_mimi(device=args.device, dtype=torch.bfloat16)

    for name in args.datasets:
        out = args.out / name
        if (out / "manifest.jsonl").exists() or (out / "samples.jsonl").exists() or (out / "scripts.jsonl").exists():
            print(f"[{name}] already exists, skipping")
            continue
        fn = DATASETS[name]
        try:
            if name in ADAPTER_DATASETS:
                save_adapter(out, fn(max_hours=args.max_hours))
            elif name in PAIRED_DATASETS:
                save_paired(out, fn(max_samples=args.max_samples), mimi, args.device)
            elif name in TEXT_DATASETS:
                save_text_seed(out, fn(max_samples=args.max_samples))
        except Exception as e:
            print(f"[{name}] failed: {e}")

    if args.combine_adapter_manifest:
        combined = args.out / "adapter_manifest.jsonl"
        n = 0
        with combined.open("w") as out_f:
            for name in args.datasets:
                if name not in ADAPTER_DATASETS:
                    continue
                mpath = args.out / name / "manifest.jsonl"
                if not mpath.exists():
                    continue
                with mpath.open() as in_f:
                    for line in in_f:
                        out_f.write(line)
                        n += 1
        print(f"combined adapter manifest: {n} entries → {combined}")


if __name__ == "__main__":
    main()
