"""Generate concurrent-commentary training samples (panda-counter style).

For each sample:
  1. Pick a narrative template with embedded "triggers" (animals, colors, etc.).
  2. TTS the user narrative (voice A).
  3. TTS each trigger word's response (voice B) as a short clip.
  4. Overlay clips onto a silent track at trigger timestamps + 150ms reaction.
  5. Encode both tracks with Mimi → DualStreamSample → save.

This is the data TML clearly trained on. Nobody else has it. We synthesize.

TTS backends (all local, all Apache):
  - kokoro   (default)  — Kokoro-82M, multiple English voices, RTF 0.03
  - qwen-tts            — Qwen3-TTS, multilingual, larger but higher quality
  - melotts             — MeloTTS, very fast, multilingual
  - gtts                — gTTS (Google online), fallback if no GPU

NOTE: this is DATA GENERATION only. Our runtime voiceai model itself IS the
TTS + STT + LLM in one. These TTS backends are used offline to build the
training set; they do not appear at inference time.

Run:
    uv run python scripts/gen_concurrent_commentary.py \\
        --out data/concurrent_commentary --n 1000 --backend kokoro
"""
from __future__ import annotations

import argparse
import json
import os
import random
import re
import uuid
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import torch


ANIMALS = [
    "panda", "giraffe", "tiger", "elephant", "monkey", "lion", "zebra",
    "kangaroo", "penguin", "dolphin", "wolf", "fox", "bear", "rhinoceros",
    "hippopotamus", "crocodile", "eagle", "owl", "parrot", "squirrel",
]
COLORS = ["red", "blue", "green", "yellow", "orange", "purple", "pink", "white", "black"]
ACTIONS = ["jumped", "ran", "swam", "climbed", "danced", "slept", "ate", "drank"]


@dataclass
class Trigger:
    word: str
    response: str   # "one", "two", "red" — what assistant says
    time_s: float   # when in user audio the trigger word ends


def build_narrative_animals(n_animals: int, rng: random.Random) -> tuple[str, list[Trigger]]:
    triggers = []
    chosen = rng.sample(ANIMALS, n_animals)
    parts = ["I went to the zoo and"]
    for i, animal in enumerate(chosen):
        connector = "saw" if i == 0 else rng.choice([", then", ", and after that", ", followed by"])
        parts.append(f"{connector} a {animal}")
        # placeholder for timing (will fill after TTS)
        triggers.append(Trigger(word=animal, response=str(i + 1), time_s=0.0))
    narrative = " ".join(parts) + "."
    return narrative, triggers


def build_narrative_colors(n: int, rng: random.Random) -> tuple[str, list[Trigger]]:
    triggers = []
    parts = ["I painted my room"]
    chosen = rng.sample(COLORS, n)
    for i, c in enumerate(chosen):
        if i == 0:
            parts.append(c)
        else:
            parts.append(rng.choice([", then", ", and"]) + " " + c)
        triggers.append(Trigger(word=c, response=c, time_s=0.0))
    narrative = " ".join(parts) + "."
    return narrative, triggers


def build_narrative_counting(n: int, rng: random.Random) -> tuple[str, list[Trigger]]:
    triggers = []
    quantities = [rng.randint(1, 10) for _ in range(n)]
    items = rng.sample(["apples", "books", "cars", "stars", "coins"], n)
    parts = ["I bought"]
    total = 0
    for i, (q, item) in enumerate(zip(quantities, items)):
        total += q
        connector = "" if i == 0 else rng.choice([", then", ", plus", ", and"])
        parts.append(f"{connector} {q} {item}")
        triggers.append(Trigger(word=str(q) + " " + item, response=str(total), time_s=0.0))
    narrative = " ".join(parts) + "."
    return narrative, triggers


BUILDERS = [
    ("animals", build_narrative_animals, (2, 6)),
    ("colors", build_narrative_colors, (2, 5)),
    ("counting", build_narrative_counting, (2, 4)),
]


# ---------------------------------------------------------------------------
# TTS backends — all local, all Apache. Loaded lazily.
# ---------------------------------------------------------------------------

_TTS_CACHE: dict = {}


def synth_audio(text: str, voice: str, out_path: Path, backend: str = "kokoro") -> float:
    """Synthesize speech to a 24kHz mono WAV. Return duration seconds."""
    out_path.parent.mkdir(parents=True, exist_ok=True)
    if backend == "kokoro":
        return _synth_kokoro(text, voice, out_path)
    if backend == "qwen-tts":
        return _synth_qwen_tts(text, voice, out_path)
    if backend == "melotts":
        return _synth_melotts(text, voice, out_path)
    if backend == "gtts":
        return _synth_gtts(text, out_path)
    raise ValueError(f"unknown TTS backend: {backend}")


def _synth_kokoro(text: str, voice: str, out_path: Path) -> float:
    """Kokoro-82M — Apache 2.0, multiple English voices.

    Voice names:
        female: af_alloy, af_aoede, af_bella, af_jessica, af_kore,
                af_nicole, af_nova, af_river, af_sarah, af_sky
        male:   am_adam, am_echo, am_eric, am_fenrir, am_liam,
                am_michael, am_onyx, am_puck, am_santa
    """
    import soundfile as sf

    if "kokoro" not in _TTS_CACHE:
        from kokoro import KPipeline

        _TTS_CACHE["kokoro"] = KPipeline(lang_code="a")

    pipeline = _TTS_CACHE["kokoro"]
    audio_segments = []
    for _, _, audio in pipeline(text, voice=voice or "af_bella"):
        audio_segments.append(audio.cpu().numpy() if hasattr(audio, "cpu") else audio)
    import numpy as np

    audio = np.concatenate(audio_segments) if audio_segments else np.zeros(1000, dtype=np.float32)
    sf.write(out_path, audio, 24000)
    return len(audio) / 24000


def _synth_qwen_tts(text: str, voice: str, out_path: Path) -> float:
    """Qwen3-TTS via transformers."""
    import soundfile as sf
    import torch

    if "qwen-tts" not in _TTS_CACHE:
        from transformers import AutoProcessor, AutoModel

        model_id = "Qwen/Qwen3-TTS"
        proc = AutoProcessor.from_pretrained(model_id, trust_remote_code=True)
        model = AutoModel.from_pretrained(
            model_id, torch_dtype=torch.bfloat16, device_map="auto", trust_remote_code=True
        )
        _TTS_CACHE["qwen-tts"] = (proc, model)
    proc, model = _TTS_CACHE["qwen-tts"]
    inputs = proc(text=text, voice=voice or "ethan", return_tensors="pt").to(model.device)
    with torch.no_grad():
        audio = model.generate(**inputs)
    audio_np = audio[0].float().cpu().numpy()
    sf.write(out_path, audio_np, 24000)
    return len(audio_np) / 24000


def _synth_melotts(text: str, voice: str, out_path: Path) -> float:
    """MeloTTS — MIT, very fast, English/multilingual."""
    import soundfile as sf

    if "melotts" not in _TTS_CACHE:
        from melo.api import TTS

        _TTS_CACHE["melotts"] = TTS(language="EN", device="auto")
    tts = _TTS_CACHE["melotts"]
    speaker_ids = tts.hps.data.spk2id
    speaker = voice if voice in speaker_ids else "EN-US"
    audio = tts.tts_to_file(
        text, speaker_id=speaker_ids[speaker], output_path=None, speed=1.0
    )
    sf.write(out_path, audio, 24000)
    return len(audio) / 24000


def _synth_gtts(text: str, out_path: Path) -> float:
    """gTTS fallback — needs internet, low quality, no GPU."""
    from gtts import gTTS

    tts = gTTS(text=text, lang="en")
    tmp_mp3 = out_path.with_suffix(".mp3")
    tts.save(str(tmp_mp3))
    import subprocess

    subprocess.run(
        ["ffmpeg", "-y", "-i", str(tmp_mp3), "-ar", "24000", "-ac", "1", str(out_path)],
        check=True,
        stdout=subprocess.DEVNULL,
        stderr=subprocess.DEVNULL,
    )
    tmp_mp3.unlink(missing_ok=True)
    import soundfile as sf

    audio, sr = sf.read(out_path)
    return len(audio) / sr


def force_align_triggers(
    narrative: str, audio_path: Path, triggers: list[Trigger]
) -> list[Trigger]:
    """Locate each trigger word's end-time in the audio.

    Lightweight approach: split narrative by whitespace, assume words are
    uniformly distributed in time. Good enough for synth data; we can swap
    to whisper-large-v3 forced-alignment later for accuracy.
    """
    import soundfile as sf

    audio, sr = sf.read(audio_path)
    total_s = len(audio) / sr
    words = re.findall(r"\w+", narrative.lower())
    per_word_s = total_s / max(1, len(words))

    for tr in triggers:
        target = tr.word.split()[-1].lower()
        try:
            idx = words.index(target)
            tr.time_s = (idx + 1) * per_word_s
        except ValueError:
            tr.time_s = total_s / 2
    return triggers


def mix_overlay(user_audio: np.ndarray, sr: int, triggers: list[Trigger], voice_b: str, tmp_dir: Path, backend: str) -> np.ndarray:
    """Build assistant track: silence with response clips overlaid at trigger times."""
    asst = np.zeros_like(user_audio)
    for tr in triggers:
        clip_path = tmp_dir / f"clip_{uuid.uuid4().hex[:8]}.wav"
        synth_audio(tr.response, voice_b, clip_path, backend=backend)
        import soundfile as sf

        clip, _ = sf.read(clip_path)
        start_sample = int((tr.time_s + 0.15) * sr)
        end_sample = start_sample + len(clip)
        if end_sample > len(asst):
            clip = clip[: len(asst) - start_sample]
            end_sample = len(asst)
        if start_sample < 0 or start_sample >= len(asst):
            continue
        asst[start_sample:end_sample] += clip
        clip_path.unlink(missing_ok=True)
    return np.clip(asst, -1.0, 1.0)


VOICE_USER_DEFAULTS = {
    "kokoro": "af_bella",
    "qwen-tts": "ethan",
    "melotts": "EN-US",
    "gtts": "en",
}
VOICE_ASST_DEFAULTS = {
    "kokoro": "am_michael",
    "qwen-tts": "chelsie",
    "melotts": "EN-Default",
    "gtts": "en",
}


def build_one_sample(rng: random.Random, tmp_dir: Path, backend: str) -> dict:
    name, builder, (lo, hi) = rng.choice(BUILDERS)
    n = rng.randint(lo, hi)
    narrative, triggers = builder(n, rng)

    sample_id = f"cc_{name}_{uuid.uuid4().hex[:8]}"
    user_wav = tmp_dir / f"{sample_id}_user.wav"
    synth_audio(
        narrative,
        voice=VOICE_USER_DEFAULTS[backend],
        out_path=user_wav,
        backend=backend,
    )
    triggers = force_align_triggers(narrative, user_wav, triggers)

    import soundfile as sf

    user_audio, sr = sf.read(user_wav)
    asst_audio = mix_overlay(
        user_audio,
        sr,
        triggers,
        voice_b=VOICE_ASST_DEFAULTS[backend],
        tmp_dir=tmp_dir,
        backend=backend,
    )

    return {
        "sample_id": sample_id,
        "narrative": narrative,
        "triggers": [{"word": t.word, "response": t.response, "time_s": t.time_s} for t in triggers],
        "user_audio": user_audio.astype(np.float32),
        "asst_audio": asst_audio.astype(np.float32),
        "sr": sr,
        "category": name,
    }


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("--out", type=Path, required=True)
    p.add_argument("--n", type=int, default=100)
    p.add_argument("--seed", type=int, default=0)
    p.add_argument(
        "--backend",
        choices=["kokoro", "qwen-tts", "melotts", "gtts"],
        default="kokoro",
    )
    p.add_argument("--encode-mimi", action="store_true", help="encode to Mimi codes immediately")
    p.add_argument("--device", default="cuda")
    args = p.parse_args()

    args.out.mkdir(parents=True, exist_ok=True)
    raw_dir = args.out / "raw"
    raw_dir.mkdir(exist_ok=True)
    tmp_dir = args.out / "_tmp"
    tmp_dir.mkdir(exist_ok=True)

    rng = random.Random(args.seed)
    samples_meta = []

    mimi = None
    if args.encode_mimi:
        from voiceai.model.mimi_utils import load_mimi

        mimi = load_mimi(device=args.device, dtype=torch.bfloat16)

    import soundfile as sf
    from tqdm.auto import tqdm

    for i in tqdm(range(args.n)):
        try:
            s = build_one_sample(rng, tmp_dir, backend=args.backend)
        except Exception as e:
            print(f"skip {i}: {e}")
            continue
        sid = s["sample_id"]
        sf.write(raw_dir / f"{sid}_user.wav", s["user_audio"], s["sr"])
        sf.write(raw_dir / f"{sid}_asst.wav", s["asst_audio"], s["sr"])
        meta = {
            "sample_id": sid,
            "narrative": s["narrative"],
            "triggers": s["triggers"],
            "duration_s": float(len(s["user_audio"]) / s["sr"]),
            "category": s["category"],
        }
        samples_meta.append(meta)

        if mimi is not None:
            _encode_and_save(s, mimi, args.out)

    (args.out / "samples.jsonl").write_text("\n".join(json.dumps(m) for m in samples_meta))
    print(f"wrote {len(samples_meta)} samples → {args.out}")

    for f in tmp_dir.glob("*.wav"):
        f.unlink()
    tmp_dir.rmdir()


def _encode_and_save(sample: dict, mimi, out_root: Path) -> None:
    from voiceai.model.mimi_utils import mimi_encode, resample_to_mimi
    from voiceai.training.data.dual_stream import DualStreamSample, save_sample

    u = torch.from_numpy(sample["user_audio"]).unsqueeze(0).unsqueeze(0)
    a = torch.from_numpy(sample["asst_audio"]).unsqueeze(0).unsqueeze(0)
    u = resample_to_mimi(u, sample["sr"]).to(next(mimi.parameters()).device).bfloat16()
    a = resample_to_mimi(a, sample["sr"]).to(next(mimi.parameters()).device).bfloat16()
    u_codes = mimi_encode(mimi, u)[0].cpu().numpy()
    a_codes = mimi_encode(mimi, a)[0].cpu().numpy()

    ds_sample = DualStreamSample(
        user_codes=u_codes,
        asst_codes=a_codes,
        text_ids=np.array([], dtype=np.int32),
        text_align=np.array([], dtype=np.int32),
        aux={
            "category": sample["category"],
            "triggers": sample["triggers"],
            "narrative": sample["narrative"],
            "concurrent_commentary": True,
        },
        sample_id=sample["sample_id"],
        duration_s=float(len(sample["user_audio"]) / sample["sr"]),
    )
    save_sample(ds_sample, out_root / "encoded")


if __name__ == "__main__":
    main()
