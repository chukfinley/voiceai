"""Audio mixing + Mimi-encoding helpers for synthetic dual-stream samples."""
from __future__ import annotations

from pathlib import Path

import numpy as np
import torch


def silent_track(duration_s: float, sr: int = 24000) -> np.ndarray:
    return np.zeros(int(duration_s * sr), dtype=np.float32)


def overlay_at(target: np.ndarray, clip: np.ndarray, at_s: float, sr: int = 24000) -> np.ndarray:
    """Add `clip` into `target` starting at `at_s`. Returns clipped result."""
    start = int(at_s * sr)
    if start < 0:
        clip = clip[-start:]
        start = 0
    end = start + len(clip)
    if end > len(target):
        clip = clip[: len(target) - start]
        end = len(target)
    if start >= len(target):
        return target
    target[start:end] = target[start:end] + clip
    return np.clip(target, -1.0, 1.0)


def pad_or_trim(audio: np.ndarray, target_s: float, sr: int = 24000) -> np.ndarray:
    n = int(target_s * sr)
    if len(audio) >= n:
        return audio[:n]
    pad = np.zeros(n - len(audio), dtype=np.float32)
    return np.concatenate([audio, pad])


def add_echo_bleed(
    user_audio: np.ndarray,
    asst_audio: np.ndarray,
    sr: int,
    gain: float = 0.1,
    delay_ms: float = 80.0,
) -> np.ndarray:
    """Mix attenuated, delayed assistant audio into the user channel.

    Simulates the residual echo AEC leaves behind (speaker → room → mic),
    so the model learns "quiet, delayed copy of my own voice on the user
    channel = echo, not the user talking" instead of barging in on itself.
    """
    delay = int(sr * delay_ms / 1000)
    bleed = np.zeros_like(user_audio)
    src = asst_audio[: max(0, len(user_audio) - delay)]
    bleed[delay : delay + len(src)] = src * gain
    return (user_audio + bleed).astype(np.float32)


def encode_dual_stream(
    user_audio: np.ndarray,
    asst_audio: np.ndarray,
    mimi,
    sr: int = 24000,
    device: str = "cuda",
    echo_bleed: float = 0.0,
    echo_delay_ms: float = 80.0,
) -> tuple[np.ndarray, np.ndarray]:
    """Encode both streams with Mimi. Returns (user_codes, asst_codes) [K, T].

    echo_bleed > 0 applies add_echo_bleed to the user channel before encoding
    (recommended for a random subset of training samples, e.g. gain 0.05-0.2).
    """
    from ...model.mimi_utils import mimi_encode, resample_to_mimi

    if echo_bleed > 0:
        user_audio = add_echo_bleed(user_audio, asst_audio, sr, gain=echo_bleed, delay_ms=echo_delay_ms)

    n = max(len(user_audio), len(asst_audio))
    user = np.pad(user_audio, (0, n - len(user_audio))) if len(user_audio) < n else user_audio
    asst = np.pad(asst_audio, (0, n - len(asst_audio))) if len(asst_audio) < n else asst_audio

    u = torch.from_numpy(user).unsqueeze(0).unsqueeze(0).to(device)
    a = torch.from_numpy(asst).unsqueeze(0).unsqueeze(0).to(device)
    u = resample_to_mimi(u, sr)
    a = resample_to_mimi(a, sr)

    dtype = next(mimi.parameters()).dtype
    u = u.to(dtype)
    a = a.to(dtype)

    with torch.no_grad():
        u_codes = mimi_encode(mimi, u)[0].cpu().numpy()
        a_codes = mimi_encode(mimi, a)[0].cpu().numpy()
    return u_codes, a_codes


def save_dual_stream_sample(
    user_codes: np.ndarray,
    asst_codes: np.ndarray,
    text_ids: np.ndarray,
    text_align: np.ndarray,
    aux: dict,
    sample_id: str,
    out_root: Path,
    duration_s: float,
) -> Path:
    from .dual_stream import DualStreamSample, save_sample as _save

    s = DualStreamSample(
        user_codes=user_codes,
        asst_codes=asst_codes,
        text_ids=text_ids,
        text_align=text_align,
        aux=aux,
        sample_id=sample_id,
        duration_s=duration_s,
    )
    return _save(s, out_root)
