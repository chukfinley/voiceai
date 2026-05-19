"""Generate time-limited session samples (30s cap, etc.).

User sets a duration limit at session start, then assistant must stop
producing audio after that duration elapses, regardless of pending content.

Teaches the model to respect <duration_limit:Ns> and <remaining:Ns>
control tokens.
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


SESSIONS = [
    {
        "instruction": "You have thirty seconds total to answer my questions. Then stop.",
        "limit_s": 30.0,
        "qa": [
            ("Capital of France?", "Paris."),
            ("Boiling point of water?", "One hundred Celsius."),
            ("Largest mammal?", "Blue whale."),
            ("Year humans landed on the moon?", "Nineteen sixty nine."),
            ("Fastest land animal?", "Cheetah."),
            ("Author of nineteen eighty-four?", "George Orwell."),
            ("Speed of light?", "About three hundred thousand kilometers per second."),
            ("Smallest country?", "Vatican City."),
        ],
    },
    {
        "instruction": "We have fifteen seconds. Quick answers only. Stop when time's up.",
        "limit_s": 15.0,
        "qa": [
            ("Two plus two?", "Four."),
            ("Color of grass?", "Green."),
            ("Hottest planet?", "Venus."),
            ("Largest ocean?", "Pacific."),
        ],
    },
    {
        "instruction": "I'll be back in twenty seconds. Until then, just say silent.",
        "limit_s": 20.0,
        "qa": [],
    },
    {
        "instruction": "Talk for ten seconds about the weather, then stop.",
        "limit_s": 10.0,
        "qa": [
            ("Talk about the weather.", "It's been quite changeable lately. Some days warm, some cool. Lots of clouds passing through. Spring is unpredictable."),
        ],
    },
    {
        "instruction": "Forty-five seconds: tell me a short story about a cat.",
        "limit_s": 45.0,
        "qa": [
            ("Go.", "Once there was a cat named Whiskers. She loved sitting in sunny windows. One day she spotted a butterfly and chased it through the garden. After a while she got tired and curled up under a rose bush. She slept all afternoon, dreaming of fish."),
        ],
    },
]


def build_one(rng: random.Random, backend: str) -> dict:
    session = rng.choice(SESSIONS)
    user_voice = rng.choice(KOKORO_VOICES_FEMALE)
    asst_voice = rng.choice(KOKORO_VOICES_MALE)
    sr = 24000
    limit_s = session["limit_s"]

    instr_audio, _ = synth(session["instruction"], voice=user_voice, backend=backend)
    ack_audio, _ = synth("Okay, ready.", voice=asst_voice, backend=backend)

    pieces: list[tuple[str, np.ndarray]] = []
    pieces.append(("user", instr_audio))
    pieces.append(("asst", ack_audio))

    countdown_start_cursor = len(instr_audio) + len(ack_audio) + int(0.5 * sr)
    cursor_n = countdown_start_cursor

    for q, a in session["qa"]:
        q_audio, _ = synth(q, voice=user_voice, backend=backend)
        a_audio, _ = synth(a, voice=asst_voice, backend=backend)
        elapsed_after_q = (cursor_n + len(q_audio) + int(0.2 * sr)) - countdown_start_cursor
        elapsed_after_a = (cursor_n + len(q_audio) + int(0.2 * sr) + len(a_audio)) - countdown_start_cursor
        if elapsed_after_a / sr > limit_s:
            cap_n = int(limit_s * sr) - elapsed_after_q
            if cap_n > 0:
                a_audio = a_audio[:cap_n]
            else:
                pieces.append(("user", q_audio))
                cursor_n += len(q_audio) + int(0.2 * sr)
                continue
        pieces.append(("user", q_audio))
        pieces.append(("asst", a_audio))
        cursor_n += len(q_audio) + int(0.2 * sr) + len(a_audio) + int(0.2 * sr)
        if cursor_n - countdown_start_cursor >= int(limit_s * sr):
            break

    tail_n = int(2.0 * sr)
    total = cursor_n + tail_n
    user_track = np.zeros(total, dtype=np.float32)
    asst_track = np.zeros(total, dtype=np.float32)

    transcript = []
    pos = 0
    for role, audio in pieces:
        n = len(audio)
        if role == "user":
            user_track[pos : pos + n] = audio
        else:
            asst_track[pos : pos + n] = audio
        transcript.append({"role": role, "start_s": pos / sr, "end_s": (pos + n) / sr})
        pos += n + int(0.2 * sr)

    cap_pos = countdown_start_cursor + int(limit_s * sr)
    if cap_pos < len(asst_track):
        asst_track[cap_pos:] = 0.0

    return {
        "sample_id": f"tl_{uuid.uuid4().hex[:10]}",
        "instruction": session["instruction"],
        "limit_s": limit_s,
        "transcript": transcript,
        "countdown_start_s": float(countdown_start_cursor / sr),
        "user_audio": user_track,
        "asst_audio": asst_track,
        "sr": sr,
        "category": "time_limited_session",
    }


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("--out", type=Path, required=True)
    p.add_argument("--n", type=int, default=100)
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
        sid = s["sample_id"]
        sf.write(raw_dir / f"{sid}_user.wav", s["user_audio"], s["sr"])
        sf.write(raw_dir / f"{sid}_asst.wav", s["asst_audio"], s["sr"])

        meta = {
            "sample_id": sid,
            "category": s["category"],
            "instruction": s["instruction"],
            "limit_s": s["limit_s"],
            "countdown_start_s": s["countdown_start_s"],
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
                aux={
                    "category": s["category"],
                    "limit_s": s["limit_s"],
                    "countdown_start_s": s["countdown_start_s"],
                },
                sample_id=sid,
                out_root=enc_dir,
                duration_s=meta["duration_s"],
            )

    (args.out / "samples.jsonl").write_text("\n".join(json.dumps(m) for m in metas))
    print(f"wrote {len(metas)} time-limited samples → {args.out}")


if __name__ == "__main__":
    main()
