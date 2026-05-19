"""Generate sound-recognition training samples.

The model should NOT only understand speech — it should also recognize
non-speech sounds (dog bark, doorbell, glass break, applause, etc.). Mimi
codec encodes any audio, so we just need labeled examples.

Approach:
  1. Use ESC-50 (50 environmental sound classes) from HuggingFace.
  2. For each clip: assistant says "I heard X" or describes the sound.
  3. Optionally precede with user asking "what do you hear?"

This is enough for the model to learn the audio→description mapping.
"""
from __future__ import annotations

import argparse
import json
import random
import uuid
from pathlib import Path

import numpy as np
import torch
from tqdm.auto import tqdm

from voiceai.training.data.mixing import encode_dual_stream, save_dual_stream_sample
from voiceai.training.data.tts_util import KOKORO_VOICES_FEMALE, KOKORO_VOICES_MALE, synth


ESC50_LABELS = [
    "dog", "rooster", "pig", "cow", "frog", "cat", "hen", "insects", "sheep", "crow",
    "rain", "sea waves", "crackling fire", "crickets", "chirping birds", "water drops",
    "wind", "pouring water", "toilet flush", "thunderstorm",
    "crying baby", "sneezing", "clapping", "breathing", "coughing",
    "footsteps", "laughing", "brushing teeth", "snoring", "drinking sipping",
    "door knock", "mouse click", "keyboard typing", "door wood creaks", "can opening",
    "washing machine", "vacuum cleaner", "clock alarm", "clock tick", "glass breaking",
    "helicopter", "chainsaw", "siren", "car horn", "engine",
    "train", "church bells", "airplane", "fireworks", "hand saw",
]


USER_PROMPTS = [
    "What did you just hear?",
    "Can you identify that sound?",
    "What was that noise?",
    "Did you catch that sound?",
    "What's that I just heard?",
    "Tell me what made that sound.",
]


def speak_response(label: str, rng: random.Random) -> str:
    templates = [
        f"That was a {label}.",
        f"Sounds like a {label}.",
        f"I heard a {label}.",
        f"That's a {label}.",
        f"I'd say that was a {label}.",
    ]
    return rng.choice(templates)


def load_esc50_sample(rng: random.Random):
    """Load one ESC-50 clip from HF."""
    from datasets import load_dataset

    ds = load_dataset("ashraq/esc50", split="train", streaming=True)
    n = rng.randint(0, 1999)
    for i, ex in enumerate(ds):
        if i == n:
            return ex
    return None


def build_one(rng: random.Random, backend: str) -> dict | None:
    ex = load_esc50_sample(rng)
    if ex is None:
        return None
    sound_audio = np.asarray(ex["audio"]["array"], dtype=np.float32)
    sound_sr = ex["audio"]["sampling_rate"]
    label = ex["category"]

    user_voice = rng.choice(KOKORO_VOICES_FEMALE)
    asst_voice = rng.choice(KOKORO_VOICES_MALE)
    sr = 24000

    if sound_sr != sr:
        import librosa

        sound_audio = librosa.resample(sound_audio, orig_sr=sound_sr, target_sr=sr)

    user_q_text = rng.choice(USER_PROMPTS)
    user_q_audio, _ = synth(user_q_text, voice=user_voice, backend=backend)
    asst_text = speak_response(label, rng)
    asst_audio_clip, _ = synth(asst_text, voice=asst_voice, backend=backend)

    gap_n = int(0.4 * sr)
    total = len(user_q_audio) + gap_n + len(sound_audio) + gap_n + len(asst_audio_clip) + sr // 2

    user_track = np.zeros(total, dtype=np.float32)
    asst_track = np.zeros(total, dtype=np.float32)

    cursor = 0
    user_track[cursor : cursor + len(user_q_audio)] = user_q_audio
    cursor += len(user_q_audio) + gap_n
    sound_start = cursor
    user_track[cursor : cursor + len(sound_audio)] = sound_audio
    cursor += len(sound_audio) + gap_n
    asst_start = cursor
    asst_track[cursor : cursor + len(asst_audio_clip)] = asst_audio_clip

    return {
        "sample_id": f"sr_{uuid.uuid4().hex[:10]}",
        "label": label,
        "user_question": user_q_text,
        "asst_response": asst_text,
        "sound_start_s": sound_start / sr,
        "asst_start_s": asst_start / sr,
        "user_audio": user_track,
        "asst_audio": asst_track,
        "sr": sr,
        "category": "sound_recognition",
    }


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("--out", type=Path, required=True)
    p.add_argument("--n", type=int, default=200)
    p.add_argument("--backend", choices=["kokoro", "melotts", "gtts"], default="kokoro")
    p.add_argument("--encode-mimi", action="store_true")
    p.add_argument("--device", default="cuda")
    p.add_argument("--seed", type=int, default=0)
    args = p.parse_args()

    args.out.mkdir(parents=True, exist_ok=True)
    raw_dir = args.out / "raw"
    raw_dir.mkdir(exist_ok=True)
    enc_dir = args.out / "encoded"
    enc_dir.mkdir(exist_ok=True)

    rng = random.Random(args.seed)
    metas = []

    mimi = None
    if args.encode_mimi:
        from voiceai.model.mimi_utils import load_mimi

        mimi = load_mimi(device=args.device, dtype=torch.bfloat16)

    import soundfile as sf

    for i in tqdm(range(args.n)):
        try:
            s = build_one(rng, args.backend)
        except Exception as e:
            print(f"skip {i}: {e}")
            continue
        if s is None:
            continue
        sid = s["sample_id"]
        sf.write(raw_dir / f"{sid}_user.wav", s["user_audio"], s["sr"])
        sf.write(raw_dir / f"{sid}_asst.wav", s["asst_audio"], s["sr"])

        meta = {
            "sample_id": sid,
            "category": s["category"],
            "label": s["label"],
            "user_question": s["user_question"],
            "asst_response": s["asst_response"],
            "duration_s": float(len(s["user_audio"]) / s["sr"]),
        }
        metas.append(meta)

        if mimi is not None:
            u_codes, a_codes = encode_dual_stream(
                s["user_audio"], s["asst_audio"], mimi, sr=s["sr"], device=args.device
            )
            save_dual_stream_sample(
                user_codes=u_codes,
                asst_codes=a_codes,
                text_ids=np.array([], dtype=np.int32),
                text_align=np.array([], dtype=np.int32),
                aux={"category": s["category"], "label": s["label"]},
                sample_id=sid,
                out_root=enc_dir,
                duration_s=meta["duration_s"],
            )

    (args.out / "samples.jsonl").write_text("\n".join(json.dumps(m) for m in metas))
    print(f"wrote {len(metas)} sound-recognition samples → {args.out}")


if __name__ == "__main__":
    main()
