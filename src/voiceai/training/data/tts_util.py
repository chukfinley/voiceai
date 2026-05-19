"""Shared TTS helpers used by all data generators.

Backends are loaded lazily and cached. Each generator imports `synth` and
gets a unified (text, voice, out_path) -> duration_s interface.
"""
from __future__ import annotations

from pathlib import Path

import numpy as np

_TTS_CACHE: dict = {}

KOKORO_VOICES_FEMALE = [
    "af_alloy", "af_aoede", "af_bella", "af_jessica", "af_kore",
    "af_nicole", "af_nova", "af_river", "af_sarah", "af_sky",
]
KOKORO_VOICES_MALE = [
    "am_adam", "am_echo", "am_eric", "am_fenrir", "am_liam",
    "am_michael", "am_onyx", "am_puck", "am_santa",
]


def synth(text: str, voice: str = "af_bella", out_path: Path | None = None, backend: str = "kokoro") -> tuple[np.ndarray, int]:
    """Return (audio float32, sample_rate). If out_path given, also write WAV."""
    if backend == "kokoro":
        audio, sr = _synth_kokoro(text, voice)
    elif backend == "melotts":
        audio, sr = _synth_melotts(text, voice)
    elif backend == "qwen-tts":
        audio, sr = _synth_qwen(text, voice)
    elif backend == "gtts":
        audio, sr = _synth_gtts(text)
    else:
        raise ValueError(f"unknown backend: {backend}")

    if out_path is not None:
        import soundfile as sf

        out_path.parent.mkdir(parents=True, exist_ok=True)
        sf.write(out_path, audio, sr)
    return audio, sr


def _synth_kokoro(text: str, voice: str) -> tuple[np.ndarray, int]:
    if "kokoro" not in _TTS_CACHE:
        from kokoro import KPipeline

        _TTS_CACHE["kokoro"] = KPipeline(lang_code="a")
    pipe = _TTS_CACHE["kokoro"]
    chunks = []
    for _, _, audio in pipe(text, voice=voice or "af_bella"):
        a = audio.cpu().numpy() if hasattr(audio, "cpu") else np.asarray(audio)
        chunks.append(a.astype(np.float32))
    if not chunks:
        return np.zeros(1000, dtype=np.float32), 24000
    return np.concatenate(chunks), 24000


def _synth_melotts(text: str, voice: str) -> tuple[np.ndarray, int]:
    if "melotts" not in _TTS_CACHE:
        from melo.api import TTS

        _TTS_CACHE["melotts"] = TTS(language="EN", device="auto")
    tts = _TTS_CACHE["melotts"]
    speaker_ids = tts.hps.data.spk2id
    speaker = voice if voice in speaker_ids else "EN-US"
    audio = tts.tts_to_file(text, speaker_id=speaker_ids[speaker], output_path=None, speed=1.0)
    return np.asarray(audio, dtype=np.float32), 24000


def _synth_qwen(text: str, voice: str) -> tuple[np.ndarray, int]:
    raise NotImplementedError("qwen-tts not wired yet; use kokoro/melotts")


def _synth_gtts(text: str) -> tuple[np.ndarray, int]:
    import subprocess
    import tempfile

    from gtts import gTTS
    import soundfile as sf

    with tempfile.NamedTemporaryFile(suffix=".mp3") as f_mp3, tempfile.NamedTemporaryFile(suffix=".wav") as f_wav:
        gTTS(text=text, lang="en").save(f_mp3.name)
        subprocess.run(
            ["ffmpeg", "-y", "-i", f_mp3.name, "-ar", "24000", "-ac", "1", f_wav.name],
            check=True,
            stdout=subprocess.DEVNULL,
            stderr=subprocess.DEVNULL,
        )
        audio, sr = sf.read(f_wav.name)
    return audio.astype(np.float32), sr
