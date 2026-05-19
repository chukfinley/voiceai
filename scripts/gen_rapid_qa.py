"""Generate rapid-fire QA samples — TML demo scenario.

Scenario: instruction sets the role ("just give the answer, no elaboration"),
then multiple speakers fire questions in quick succession. Assistant gives
short direct answers to each. Optional 30s time cap.

Teaches:
  - direct-answer mode without preamble
  - handling back-to-back questions
  - distinguishing between different voices (speaker-agnostic but recognizes
    new utterances)
  - respecting time-bounded sessions
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
    save_dual_stream_sample,
)
from voiceai.training.data.tts_util import (
    KOKORO_VOICES_FEMALE,
    KOKORO_VOICES_MALE,
    synth,
)


QA_BANK = [
    ("Capital of France?", "Paris."),
    ("Capital of Germany?", "Berlin."),
    ("Two plus two?", "Four."),
    ("Color of the sky?", "Blue."),
    ("Largest planet?", "Jupiter."),
    ("Speed of light?", "About three hundred thousand kilometers per second."),
    ("Who wrote Hamlet?", "Shakespeare."),
    ("Year World War Two ended?", "Nineteen forty five."),
    ("Tallest mountain?", "Mount Everest."),
    ("Longest river?", "The Nile."),
    ("Inventor of the light bulb?", "Edison."),
    ("How many continents are there?", "Seven."),
    ("Boiling point of water in Celsius?", "One hundred."),
    ("Currency of Japan?", "The yen."),
    ("Author of nineteen eighty four?", "George Orwell."),
    ("Largest ocean?", "The Pacific."),
    ("Year humans landed on the moon?", "Nineteen sixty nine."),
    ("Square root of sixty four?", "Eight."),
    ("Element with symbol O?", "Oxygen."),
    ("Smallest country?", "Vatican City."),
    ("Who painted the Mona Lisa?", "Leonardo da Vinci."),
    ("Capital of Australia?", "Canberra."),
    ("Number of bones in the human body?", "Two hundred six."),
    ("How many sides does a hexagon have?", "Six."),
    ("Hottest planet?", "Venus."),
    ("How many notes in an octave?", "Eight."),
    ("Fastest land animal?", "Cheetah."),
    ("Largest desert?", "The Sahara."),
    ("Capital of Canada?", "Ottawa."),
    ("Who discovered penicillin?", "Alexander Fleming."),
]


SYSTEM_INSTRUCTIONS = [
    "Just answer my questions directly with no elaboration. Short answers only.",
    "I'm going to ask several quick questions. Give me a one-word or one-sentence answer each time.",
    "We'll do a rapid Q and A. Keep your replies very brief.",
    "Be concise. Just the answer. No filler.",
]


def build_one(rng: random.Random, backend: str) -> dict:
    n_questions = rng.randint(3, 7)
    questions = rng.sample(QA_BANK, n_questions)
    n_speakers = rng.randint(1, 3)
    voices = rng.sample(KOKORO_VOICES_FEMALE + KOKORO_VOICES_MALE, n_speakers)
    asst_voice = rng.choice(KOKORO_VOICES_MALE)

    instruction = rng.choice(SYSTEM_INSTRUCTIONS)
    sr = 24000

    instr_audio, _ = synth(instruction, voice=voices[0], backend=backend)
    asst_ack, _ = synth("Okay.", voice=asst_voice, backend=backend)

    segments: list[tuple[str, np.ndarray]] = []
    segments.append(("user", instr_audio))
    segments.append(("asst", asst_ack))

    for q_text, a_text in questions:
        v = rng.choice(voices)
        q_audio, _ = synth(q_text, voice=v, backend=backend)
        a_audio, _ = synth(a_text, voice=asst_voice, backend=backend)
        segments.append(("user", q_audio))
        segments.append(("asst", a_audio))

    inter_gap_s = 0.25
    inter_gap_n = int(inter_gap_s * sr)
    total = sum(len(s[1]) for s in segments) + inter_gap_n * (len(segments) - 1)

    user_track = np.zeros(total, dtype=np.float32)
    asst_track = np.zeros(total, dtype=np.float32)
    transcript = []
    cursor = 0
    for role, audio in segments:
        n = len(audio)
        if role == "user":
            user_track[cursor : cursor + n] = audio
        else:
            asst_track[cursor : cursor + n] = audio
        transcript.append({"role": role, "start_s": cursor / sr, "end_s": (cursor + n) / sr})
        cursor += n + inter_gap_n

    return {
        "sample_id": f"rqa_{uuid.uuid4().hex[:10]}",
        "instruction": instruction,
        "questions": questions,
        "n_speakers": n_speakers,
        "asst_voice": asst_voice,
        "user_audio": user_track,
        "asst_audio": asst_track,
        "sr": sr,
        "transcript": transcript,
        "category": "rapid_qa",
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
            "questions": s["questions"],
            "n_speakers": s["n_speakers"],
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
                    "instruction": s["instruction"],
                    "n_speakers": s["n_speakers"],
                    "questions": s["questions"],
                },
                sample_id=sid,
                out_root=enc_dir,
                duration_s=meta["duration_s"],
            )

    (args.out / "samples.jsonl").write_text("\n".join(json.dumps(m) for m in metas))
    print(f"wrote {len(metas)} rapid-qa samples → {args.out}")


if __name__ == "__main__":
    main()
