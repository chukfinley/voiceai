"""Generate time-aware training samples with REAL audio (not just specs).

Each sample contains a scenario where the assistant MUST wait a specific
duration before responding. We teach the model to emit `<silent>` tokens
during the wait, then speak at the right time.
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

from voiceai.training.data.mixing import (
    encode_dual_stream,
    overlay_at,
    pad_or_trim,
    save_dual_stream_sample,
    silent_track,
)
from voiceai.training.data.tts_util import (
    KOKORO_VOICES_FEMALE,
    KOKORO_VOICES_MALE,
    synth,
)


SCENARIOS = [
    # (user_text, wait_seconds, asst_text)
    ("Wait three seconds then say hello.", 3.0, "Hello."),
    ("In five seconds tell me a joke.", 5.0, "Why don't scientists trust atoms? Because they make up everything."),
    ("Count to three after a four second pause.", 4.0, "One. Two. Three."),
    ("Pause for two seconds and then describe a cat.", 2.0, "A cat is a small furry animal that purrs."),
    ("Wait six seconds and then tell me the time.", 6.0, "It's been six seconds."),
    ("Be silent for ten seconds.", 10.0, "Okay, ten seconds passed."),
    ("After three seconds say good morning.", 3.0, "Good morning."),
    ("Take a five second pause and then say done.", 5.0, "Done."),
    ("Stay quiet for seven seconds then continue.", 7.0, "Continuing now."),
    ("Hold on four seconds then answer yes.", 4.0, "Yes."),
    ("Pause exactly eight seconds before responding.", 8.0, "Now I respond."),
    ("Two second wait, then count by twos to ten.", 2.0, "Two. Four. Six. Eight. Ten."),
]

# Also include self-initiate scenarios — user says nothing, after N seconds asst speaks
SELF_INIT_SCENARIOS = [
    ("Are you still there?", 8.0),
    ("Hello? Is anyone listening?", 10.0),
    ("I notice you've been quiet. Should I continue?", 12.0),
    ("Just checking in.", 15.0),
]


def build_wait_scenario(rng: random.Random, backend: str) -> dict:
    user_text, wait_s, asst_text = rng.choice(SCENARIOS)
    user_voice = rng.choice(KOKORO_VOICES_FEMALE)
    asst_voice = rng.choice(KOKORO_VOICES_MALE)

    user_audio_raw, sr = synth(user_text, voice=user_voice, backend=backend)
    asst_clip, _ = synth(asst_text, voice=asst_voice, backend=backend)
    asst_dur_s = len(asst_clip) / sr
    user_dur_s = len(user_audio_raw) / sr
    total_s = user_dur_s + wait_s + asst_dur_s + 1.0

    user_full = pad_or_trim(user_audio_raw, total_s, sr=sr)
    asst_full = silent_track(total_s, sr=sr)
    asst_full = overlay_at(asst_full, asst_clip, at_s=user_dur_s + wait_s, sr=sr)

    return {
        "sample_id": f"ta_wait_{uuid.uuid4().hex[:10]}",
        "user_text": user_text,
        "asst_text": asst_text,
        "wait_s": wait_s,
        "user_voice": user_voice,
        "asst_voice": asst_voice,
        "user_audio": user_full,
        "asst_audio": asst_full,
        "sr": sr,
        "category": "time_aware_wait",
    }


def build_self_init_scenario(rng: random.Random, backend: str) -> dict:
    asst_text, silence_s = rng.choice(SELF_INIT_SCENARIOS)
    asst_voice = rng.choice(KOKORO_VOICES_MALE)
    sr = 24000

    asst_clip, _ = synth(asst_text, voice=asst_voice, backend=backend)
    total_s = silence_s + len(asst_clip) / sr + 1.0

    user_full = silent_track(total_s, sr=sr)
    asst_full = silent_track(total_s, sr=sr)
    asst_full = overlay_at(asst_full, asst_clip, at_s=silence_s, sr=sr)

    return {
        "sample_id": f"ta_init_{uuid.uuid4().hex[:10]}",
        "user_text": "",
        "asst_text": asst_text,
        "wait_s": silence_s,
        "user_voice": None,
        "asst_voice": asst_voice,
        "user_audio": user_full,
        "asst_audio": asst_full,
        "sr": sr,
        "category": "time_aware_self_init",
    }


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("--out", type=Path, required=True)
    p.add_argument("--n", type=int, default=100)
    p.add_argument("--backend", choices=["kokoro", "melotts", "gtts"], default="kokoro")
    p.add_argument("--encode-mimi", action="store_true")
    p.add_argument("--device", default="cuda")
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--self-init-frac", type=float, default=0.3)
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
            if rng.random() < args.self_init_frac:
                s = build_self_init_scenario(rng, args.backend)
            else:
                s = build_wait_scenario(rng, args.backend)
        except Exception as e:
            print(f"skip {i}: {e}")
            continue
        sid = s["sample_id"]
        sf.write(raw_dir / f"{sid}_user.wav", s["user_audio"], s["sr"])
        sf.write(raw_dir / f"{sid}_asst.wav", s["asst_audio"], s["sr"])

        meta = {
            "sample_id": sid,
            "category": s["category"],
            "user_text": s["user_text"],
            "asst_text": s["asst_text"],
            "wait_s": s["wait_s"],
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
                    "wait_s": s["wait_s"],
                    "user_text": s["user_text"],
                    "asst_text": s["asst_text"],
                },
                sample_id=sid,
                out_root=enc_dir,
                duration_s=meta["duration_s"],
            )

    (args.out / "samples.jsonl").write_text("\n".join(json.dumps(m) for m in metas))
    print(f"wrote {len(metas)} time-aware samples → {args.out}")


if __name__ == "__main__":
    main()
