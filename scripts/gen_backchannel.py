"""Generate backchannel training samples.

Backchanneling = assistant emits short acknowledgments ("mhm", "yeah",
"I see", "okay") DURING the user's turn without taking it over.

We synthesize:
  1. User narrative (multi-sentence story, ~10-30s)
  2. At natural pause points (commas, sentence ends, mid-clause), assistant
     drops one of the backchannel tokens.
  3. Save as dual-stream paired sample.
"""
from __future__ import annotations

import argparse
import json
import random
import re
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


BACKCHANNELS = [
    "mm-hmm", "yeah", "uh-huh", "right", "okay", "I see", "hmm",
    "oh", "go on", "really", "exactly", "of course", "got it",
]


STORY_TEMPLATES = [
    "So yesterday I went to the grocery store, and I was looking for some apples, but they were all out, which was really annoying because I needed them for a pie.",
    "I started a new project last week, and it's been going pretty well, but I keep running into this one bug that just won't go away no matter what I try.",
    "My cat did the funniest thing this morning, she jumped onto the table, looked at me for a second, then knocked my coffee cup right off the edge.",
    "I've been thinking about taking a trip somewhere this summer, maybe to the mountains or the coast, I haven't decided yet, but I really need a break.",
    "The weather has been so weird this week, one day it's sunny and warm, the next day there's snow on the ground, I don't even know what to wear anymore.",
    "I tried that new restaurant downtown last night, the one everyone has been talking about, and honestly the food was amazing but the service was kind of slow.",
    "My friend Sarah is moving across the country next month, which is sad because we've known each other for like ten years, but she got a great job offer.",
    "I've been learning Spanish on Duolingo for about six months now, and I can finally hold a basic conversation, although my accent is still pretty rough.",
    "We had this huge meeting at work today that lasted almost three hours, and honestly nothing got decided, so I'm not sure what the point of it was.",
    "I started reading this really interesting book about ancient Rome, and it's amazing how much we still don't know about that period of history.",
]


def pick_backchannel_times(text: str, audio: np.ndarray, sr: int) -> list[tuple[float, str]]:
    """Find natural pause points and assign backchannel words.

    We look at commas + sentence boundaries in the text, estimate their
    time positions linearly from word count. Drop a backchannel at ~40%
    of the pauses (not every single one).
    """
    words = re.findall(r"\S+", text)
    if not words:
        return []
    total_s = len(audio) / sr
    per_word_s = total_s / len(words)

    pause_indices = []
    running = 0
    for i, w in enumerate(words):
        running += 1
        if w.endswith((",", ".", "!", "?", ";")):
            pause_indices.append(i)

    rng = random.Random(hash(text) & 0xFFFFFFFF)
    chosen = [i for i in pause_indices if rng.random() < 0.4]
    out = []
    for idx in chosen:
        t = (idx + 1) * per_word_s + rng.uniform(0.1, 0.35)
        word = rng.choice(BACKCHANNELS)
        out.append((t, word))
    return out


def build_one(
    rng: random.Random,
    tmp_dir: Path,
    backend: str,
) -> dict:
    story = rng.choice(STORY_TEMPLATES)
    user_voice = rng.choice(KOKORO_VOICES_FEMALE)
    asst_voice = rng.choice(KOKORO_VOICES_MALE)

    user_audio, sr = synth(story, voice=user_voice, backend=backend)
    bcs = pick_backchannel_times(story, user_audio, sr)
    asst_audio = silent_track(len(user_audio) / sr, sr=sr)

    overlaid = []
    for t, word in bcs:
        clip, _ = synth(word, voice=asst_voice, backend=backend)
        asst_audio = overlay_at(asst_audio, clip * 0.7, t, sr=sr)
        overlaid.append({"word": word, "time_s": float(t)})

    return {
        "sample_id": f"bc_{uuid.uuid4().hex[:10]}",
        "story": story,
        "backchannels": overlaid,
        "user_voice": user_voice,
        "asst_voice": asst_voice,
        "user_audio": user_audio,
        "asst_audio": asst_audio,
        "sr": sr,
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
    tmp_dir = args.out / "_tmp"
    tmp_dir.mkdir(exist_ok=True)

    rng = random.Random(args.seed)
    metas = []

    mimi = None
    if args.encode_mimi:
        from voiceai.model.mimi_utils import load_mimi

        mimi = load_mimi(device=args.device, dtype=torch.bfloat16)

    import soundfile as sf

    for i in tqdm(range(args.n)):
        try:
            s = build_one(rng, tmp_dir, args.backend)
        except Exception as e:
            print(f"skip {i}: {e}")
            continue
        sid = s["sample_id"]
        sf.write(raw_dir / f"{sid}_user.wav", s["user_audio"], s["sr"])
        sf.write(raw_dir / f"{sid}_asst.wav", s["asst_audio"], s["sr"])
        meta = {
            "sample_id": sid,
            "story": s["story"],
            "backchannels": s["backchannels"],
            "user_voice": s["user_voice"],
            "asst_voice": s["asst_voice"],
            "duration_s": float(len(s["user_audio"]) / s["sr"]),
            "category": "backchannel",
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
                aux={"category": "backchannel", "backchannels": s["backchannels"], "story": s["story"]},
                sample_id=sid,
                out_root=enc_dir,
                duration_s=meta["duration_s"],
            )

    (args.out / "samples.jsonl").write_text("\n".join(json.dumps(m) for m in metas))
    print(f"wrote {len(metas)} backchannel samples → {args.out}")

    for f in tmp_dir.glob("*"):
        f.unlink(missing_ok=True)
    tmp_dir.rmdir()


if __name__ == "__main__":
    main()
