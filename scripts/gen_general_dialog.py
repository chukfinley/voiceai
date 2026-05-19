"""Generate general turn-taking dialog samples — the bulk of training data.

This is the "talk like a human" backbone dataset. Diverse topics, natural
turn-taking, no special interaction tricks. Other generators (concurrent,
backchannel, time-aware) provide the specialty data on top of this base.

Source: scripted dialog templates expanded with simple variations. For
higher quality, swap to Claude API generation later (see TODO).
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


DIALOG_TEMPLATES = [
    [
        ("user", "Hey, how are you doing today?"),
        ("asst", "I'm doing well, thanks. How about you?"),
        ("user", "Pretty good. Just got back from a long walk."),
        ("asst", "Nice. Where did you go?"),
        ("user", "Down by the river. The weather was perfect."),
    ],
    [
        ("user", "What's the capital of France?"),
        ("asst", "Paris."),
        ("user", "And what's it known for?"),
        ("asst", "The Eiffel Tower, the Louvre, and great food."),
    ],
    [
        ("user", "I'm thinking about learning to cook."),
        ("asst", "That's a great skill. What kind of food do you want to make?"),
        ("user", "Probably Italian. I love pasta."),
        ("asst", "Start with a simple tomato sauce. It's forgiving and delicious."),
    ],
    [
        ("user", "Can you recommend a good book?"),
        ("asst", "What genre do you usually enjoy?"),
        ("user", "Mystery and thrillers mostly."),
        ("asst", "Try 'The Silent Patient' by Alex Michaelides. It's gripping."),
    ],
    [
        ("user", "I had a really stressful day at work."),
        ("asst", "I'm sorry to hear that. Want to talk about it?"),
        ("user", "We had a deadline that got moved up by a week."),
        ("asst", "That's tough. Do you have what you need to make it?"),
        ("user", "Maybe, if I work weekends."),
    ],
    [
        ("user", "What time is it?"),
        ("asst", "Around three in the afternoon."),
        ("user", "Already? I've gotten nothing done."),
        ("asst", "Don't worry about it. The day's not over yet."),
    ],
    [
        ("user", "I'm planning a vacation to Japan."),
        ("asst", "Exciting. When are you going?"),
        ("user", "Next spring. Probably April."),
        ("asst", "Cherry blossom season. You'll love it."),
    ],
    [
        ("user", "Do you know any good jokes?"),
        ("asst", "Why don't scientists trust atoms? Because they make up everything."),
        ("user", "Ha, that's terrible."),
        ("asst", "I have more if you want."),
    ],
    [
        ("user", "Explain quantum physics to me in one sentence."),
        ("asst", "Tiny particles behave weirdly and don't follow normal rules."),
        ("user", "That's it?"),
        ("asst", "That's the heart of it, yeah."),
    ],
    [
        ("user", "My internet keeps cutting out."),
        ("asst", "Have you tried restarting the router?"),
        ("user", "Twice already."),
        ("asst", "Then it might be an ISP issue. Worth calling them."),
    ],
]


def build_one(rng: random.Random, backend: str) -> dict:
    template = rng.choice(DIALOG_TEMPLATES)
    user_voice = rng.choice(KOKORO_VOICES_FEMALE)
    asst_voice = rng.choice(KOKORO_VOICES_MALE)

    # synth each turn separately
    turns = []
    for role, text in template:
        voice = user_voice if role == "user" else asst_voice
        audio, sr = synth(text, voice=voice, backend=backend)
        turns.append({"role": role, "text": text, "audio": audio, "sr": sr})

    # arrange end-to-end with small gaps
    gap_s = 0.4
    sr = turns[0]["sr"]
    total = sum(len(t["audio"]) for t in turns) + int(gap_s * sr * (len(turns) - 1))

    user_track = np.zeros(total, dtype=np.float32)
    asst_track = np.zeros(total, dtype=np.float32)
    cursor = 0
    transcript = []
    for t in turns:
        n = len(t["audio"])
        if t["role"] == "user":
            user_track[cursor : cursor + n] = t["audio"]
        else:
            asst_track[cursor : cursor + n] = t["audio"]
        transcript.append({"role": t["role"], "text": t["text"], "start_s": cursor / sr, "end_s": (cursor + n) / sr})
        cursor += n + int(gap_s * sr)

    return {
        "sample_id": f"gd_{uuid.uuid4().hex[:10]}",
        "transcript": transcript,
        "user_voice": user_voice,
        "asst_voice": asst_voice,
        "user_audio": user_track,
        "asst_audio": asst_track,
        "sr": sr,
        "category": "general_dialog",
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
        sid = s["sample_id"]
        sf.write(raw_dir / f"{sid}_user.wav", s["user_audio"], s["sr"])
        sf.write(raw_dir / f"{sid}_asst.wav", s["asst_audio"], s["sr"])

        meta = {
            "sample_id": sid,
            "category": s["category"],
            "transcript": s["transcript"],
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
                aux={"category": s["category"], "transcript": s["transcript"]},
                sample_id=sid,
                out_root=enc_dir,
                duration_s=meta["duration_s"],
            )

    (args.out / "samples.jsonl").write_text("\n".join(json.dumps(m) for m in metas))
    print(f"wrote {len(metas)} general-dialog samples → {args.out}")


if __name__ == "__main__":
    main()
