"""Generate constraint-following samples.

User sets a constraint at session start, then assistant must adhere
throughout. Examples of constraints:

  - "Only answer with one word."
  - "Answer with yes or no only."
  - "Don't speak unless I say 'help'."
  - "Repeat back exactly what I say."
  - "Don't mention <topic> at all."
  - "Stop talking after 30 seconds total."

This teaches the model to RESPECT operating-mode instructions instead of
falling back to defaults.
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


CONSTRAINT_SCENARIOS = [
    {
        "instruction": "Just answer my questions with one word. No elaboration.",
        "ack": "Okay.",
        "turns": [
            ("What's the capital of Italy?", "Rome."),
            ("Largest planet?", "Jupiter."),
            ("Five times six?", "Thirty."),
            ("Author of Hamlet?", "Shakespeare."),
            ("Year humans landed on the moon?", "Nineteen sixty nine."),
        ],
    },
    {
        "instruction": "Only respond with yes or no. Nothing else.",
        "ack": "Understood.",
        "turns": [
            ("Is Paris in France?", "Yes."),
            ("Are dogs reptiles?", "No."),
            ("Is fire cold?", "No."),
            ("Is the sun a star?", "Yes."),
            ("Is two greater than five?", "No."),
        ],
    },
    {
        "instruction": "Repeat exactly what I say. Don't add anything.",
        "ack": "Got it.",
        "turns": [
            ("The sky is blue today.", "The sky is blue today."),
            ("I had toast for breakfast.", "I had toast for breakfast."),
            ("My favorite color is green.", "My favorite color is green."),
        ],
    },
    {
        "instruction": "Don't speak unless I say the word help.",
        "ack": "Okay.",
        "turns": [
            ("I'm just talking out loud here.", "<silent>"),
            ("Thinking about what to make for dinner.", "<silent>"),
            ("Maybe pasta. Help, what goes well with pasta?", "Garlic, olive oil, parmesan, and fresh basil."),
            ("Nice. Going to try that now.", "<silent>"),
        ],
    },
    {
        "instruction": "Be very brief. Maximum five words per answer.",
        "ack": "Will do.",
        "turns": [
            ("What's the weather like in summer?", "Usually warm and sunny."),
            ("Best programming language for beginners?", "Python is friendly."),
            ("How do I make tea?", "Boil water, steep leaves."),
        ],
    },
    {
        "instruction": "Speak only in questions back to me.",
        "ack": "Okay.",
        "turns": [
            ("I went hiking yesterday.", "Where did you hike?"),
            ("Up in the mountains.", "Which mountains specifically?"),
            ("The Rockies.", "How long was the hike?"),
        ],
    },
    {
        "instruction": "Never mention the color blue. Don't use that word.",
        "ack": "Understood.",
        "turns": [
            ("What color is the sky?", "The sky is the color above us in daytime."),
            ("Describe the ocean.", "It's vast, salty, and deep."),
            ("What about a sapphire?", "It's a precious gemstone, often dark."),
        ],
    },
    {
        "instruction": "Stop talking after five exchanges total. Then stay silent forever.",
        "ack": "Got it.",
        "turns": [
            ("What's your favorite food?", "I don't eat but I'd pick pasta."),
            ("Pasta with what sauce?", "A simple tomato sauce."),
            ("Sounds good.", "Glad you like it."),
            ("How about dessert?", "Tiramisu is classic."),
            ("Okay, last question, fruit?", "Mango."),
            ("Hello?", "<silent>"),
            ("Are you there?", "<silent>"),
        ],
    },
]


def build_one(rng: random.Random, backend: str) -> dict:
    scenario = rng.choice(CONSTRAINT_SCENARIOS)
    user_voice = rng.choice(KOKORO_VOICES_FEMALE)
    asst_voice = rng.choice(KOKORO_VOICES_MALE)
    sr = 24000

    segments: list[tuple[str, np.ndarray]] = []

    instr_audio, _ = synth(scenario["instruction"], voice=user_voice, backend=backend)
    ack_audio, _ = synth(scenario["ack"], voice=asst_voice, backend=backend)
    segments.append(("user", instr_audio))
    segments.append(("asst", ack_audio))

    for user_text, asst_text in scenario["turns"]:
        u_audio, _ = synth(user_text, voice=user_voice, backend=backend)
        segments.append(("user", u_audio))
        if asst_text == "<silent>":
            silent_n = int(rng.uniform(0.8, 1.5) * sr)
            segments.append(("asst_silent", np.zeros(silent_n, dtype=np.float32)))
        else:
            a_audio, _ = synth(asst_text, voice=asst_voice, backend=backend)
            segments.append(("asst", a_audio))

    inter_gap_n = int(0.3 * sr)
    total = sum(len(s[1]) for s in segments) + inter_gap_n * (len(segments) - 1)
    user_track = np.zeros(total, dtype=np.float32)
    asst_track = np.zeros(total, dtype=np.float32)

    transcript = []
    cursor = 0
    for role, audio in segments:
        n = len(audio)
        if role == "user":
            user_track[cursor : cursor + n] = audio
        elif role == "asst":
            asst_track[cursor : cursor + n] = audio
        # asst_silent: leave as zeros
        transcript.append({"role": role, "start_s": cursor / sr, "end_s": (cursor + n) / sr})
        cursor += n + inter_gap_n

    return {
        "sample_id": f"cn_{uuid.uuid4().hex[:10]}",
        "instruction": scenario["instruction"],
        "transcript": transcript,
        "user_audio": user_track,
        "asst_audio": asst_track,
        "sr": sr,
        "category": "constraint_following",
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
                aux={"category": s["category"], "instruction": s["instruction"]},
                sample_id=sid,
                out_root=enc_dir,
                duration_s=meta["duration_s"],
            )

    (args.out / "samples.jsonl").write_text("\n".join(json.dumps(m) for m in metas))
    print(f"wrote {len(metas)} constraint-following samples → {args.out}")


if __name__ == "__main__":
    main()
