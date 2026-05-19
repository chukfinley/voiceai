"""Generate barge-in training samples.

Assistant starts speaking, user interrupts mid-sentence. Assistant must
detect, stop talking, and acknowledge.
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
    save_dual_stream_sample,
    silent_track,
)
from voiceai.training.data.tts_util import (
    KOKORO_VOICES_FEMALE,
    KOKORO_VOICES_MALE,
    synth,
)


SCENARIOS = [
    # (user_initial, asst_long_response, user_barge, asst_recovery)
    (
        "Tell me about the history of Rome.",
        "Rome was founded according to legend in 753 BC by Romulus and Remus, two brothers raised by a she-wolf, and grew from a small settlement on the Tiber river into one of the largest empires the world has ever seen, eventually stretching from",
        "Wait, just give me the short version.",
        "Sure, sorry. Rome was a city-state that became an empire spanning the Mediterranean from around 750 BC to 476 AD.",
    ),
    (
        "How do I make pasta?",
        "Making pasta is a wonderful process that starts with selecting good flour, ideally a high-protein flour like semolina or 00 flour, which gives the pasta its perfect texture, and then you mix it with eggs in a roughly two-to-one ratio by weight, and",
        "Just tell me the simplest way.",
        "Right. Boil salted water, drop pasta in, cook eight minutes, drain. Done.",
    ),
    (
        "What's the best programming language?",
        "There really isn't a single best programming language because each one has strengths and weaknesses depending on what you're trying to build, for example Python is great for data science and machine learning because of its libraries like NumPy and",
        "Just pick one.",
        "Python. For most use cases.",
    ),
    (
        "Explain how the internet works.",
        "The internet is a global network of interconnected computers that communicate using a set of standardized protocols, most importantly TCP/IP, where TCP stands for Transmission Control Protocol and IP stands for Internet Protocol, and the way it works is that data",
        "Hang on, way too technical.",
        "Got it. Computers send messages to each other through cables and wireless signals. Same as the postal system.",
    ),
    (
        "What do you think about climate change?",
        "Climate change is a complex global phenomenon driven primarily by human emissions of greenhouse gases, particularly carbon dioxide and methane, which trap heat in the atmosphere and cause average global temperatures to rise, leading to a cascade of effects including",
        "Just yes or no, is it real?",
        "Yes.",
    ),
]


def build_one(rng: random.Random, backend: str) -> dict:
    user_init, asst_long, user_barge, asst_recovery = rng.choice(SCENARIOS)
    user_voice = rng.choice(KOKORO_VOICES_FEMALE)
    asst_voice = rng.choice(KOKORO_VOICES_MALE)
    sr = 24000

    u_init_audio, _ = synth(user_init, voice=user_voice, backend=backend)
    asst_long_audio, _ = synth(asst_long, voice=asst_voice, backend=backend)
    u_barge_audio, _ = synth(user_barge, voice=user_voice, backend=backend)
    asst_rec_audio, _ = synth(asst_recovery, voice=asst_voice, backend=backend)

    # Layout:
    #   0..u_init_dur            -- user speaks initial question
    #   gap 0.4s
    #   then user is silent, assistant starts long answer
    #   barge_in_frac (0.4..0.7 of asst_long) -- user starts barge speech
    #   assistant audio truncated 200ms after barge_in begins (= reaction time)
    #   gap after barge ends, then assistant says recovery
    gap_s = 0.4
    u_init_n = len(u_init_audio)
    asst_long_n = len(asst_long_audio)
    barge_frac = rng.uniform(0.4, 0.7)
    barge_start_in_asst = int(asst_long_n * barge_frac)
    reaction_n = int(0.2 * sr)
    asst_truncated_n = barge_start_in_asst + reaction_n

    pre_gap = int(gap_s * sr)
    barge_n = len(u_barge_audio)
    post_gap = int(gap_s * sr)
    rec_n = len(asst_rec_audio)
    tail = int(0.5 * sr)

    total = u_init_n + pre_gap + max(asst_truncated_n, barge_start_in_asst + barge_n) + post_gap + rec_n + tail
    user_track = np.zeros(total, dtype=np.float32)
    asst_track = np.zeros(total, dtype=np.float32)

    # user init
    user_track[0:u_init_n] = u_init_audio
    # asst long (truncated)
    asst_start = u_init_n + pre_gap
    asst_track[asst_start : asst_start + asst_truncated_n] = asst_long_audio[:asst_truncated_n]
    # user barge starts at barge_start_in_asst into the asst speech
    barge_start_in_user = asst_start + barge_start_in_asst
    user_track[barge_start_in_user : barge_start_in_user + barge_n] = u_barge_audio
    # asst recovery
    rec_start = max(asst_start + asst_truncated_n, barge_start_in_user + barge_n) + post_gap
    asst_track[rec_start : rec_start + rec_n] = asst_rec_audio

    return {
        "sample_id": f"bg_{uuid.uuid4().hex[:10]}",
        "user_init": user_init,
        "asst_long": asst_long,
        "asst_truncated_at_word": int(asst_truncated_n / sr * 3),  # rough word count
        "user_barge": user_barge,
        "asst_recovery": asst_recovery,
        "barge_start_s": float(barge_start_in_user / sr),
        "reaction_s": 0.2,
        "user_voice": user_voice,
        "asst_voice": asst_voice,
        "user_audio": user_track,
        "asst_audio": asst_track,
        "sr": sr,
        "category": "barge_in",
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
            "user_init": s["user_init"],
            "asst_long": s["asst_long"],
            "user_barge": s["user_barge"],
            "asst_recovery": s["asst_recovery"],
            "barge_start_s": s["barge_start_s"],
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
                    "barge_start_s": s["barge_start_s"],
                    "user_init": s["user_init"],
                    "user_barge": s["user_barge"],
                },
                sample_id=sid,
                out_root=enc_dir,
                duration_s=meta["duration_s"],
            )

    (args.out / "samples.jsonl").write_text("\n".join(json.dumps(m) for m in metas))
    print(f"wrote {len(metas)} barge-in samples → {args.out}")


if __name__ == "__main__":
    main()
