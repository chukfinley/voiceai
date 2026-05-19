"""Acoustic Echo Cancellation — strips own TTS output from mic input.

Required for simultaneous-speech: assistant talks while user talks. Without
AEC the ASR re-recognizes our own TTS output.

Three options:
  1. webrtc-audio-processing (BSD, mature) — preferred
  2. speexdsp (LGPL) — fallback
  3. DeepFilterNet (MIT, neural) — heavier but better in noisy rooms

We expose a single AECLoop that subscribes to AUDIO_IN_CHUNK + TTS_OUT_CHUNK
and republishes a cleaned AUDIO_IN_CHUNK_AEC event (or rewrites in-place).
"""
from __future__ import annotations

import asyncio
from collections import deque

import numpy as np

from .core import EventBus
from .events import EvType


class AECPassthrough:
    """No-op for dev. Real impl uses webrtc-audio-processing."""

    def process(self, mic: np.ndarray, ref: np.ndarray | None) -> np.ndarray:
        return mic


class AECWebRTC:
    """Wraps py-webrtc-audio-processing. Lazy import."""

    def __init__(self, frame_ms: int = 10) -> None:
        try:
            from webrtc_audio_processing import AudioProcessingModule  # type: ignore
        except ImportError as e:
            raise RuntimeError(
                "Install webrtc-audio-processing: "
                "pip install webrtc-audio-processing"
            ) from e
        self.apm = AudioProcessingModule(aec_type=2, enable_ns=True)
        self.apm.set_stream_format(16000, 1)
        self.apm.set_reverse_stream_format(16000, 1)
        self.frame_ms = frame_ms

    def process(self, mic: np.ndarray, ref: np.ndarray | None) -> np.ndarray:
        # WebRTC APM works on 10ms frames of int16. Re-chunk on the fly.
        if ref is not None:
            self.apm.process_reverse_stream(_f32_to_int16(ref).tobytes())
        out = self.apm.process_stream(_f32_to_int16(mic).tobytes())
        return _int16_bytes_to_f32(out)


def _f32_to_int16(x: np.ndarray) -> np.ndarray:
    return np.clip(x * 32767.0, -32768, 32767).astype(np.int16)


def _int16_bytes_to_f32(b: bytes) -> np.ndarray:
    return np.frombuffer(b, dtype=np.int16).astype(np.float32) / 32768.0


class AECLoop:
    """Aligns TTS reference and mic input by ~30ms, runs AEC, republishes."""

    def __init__(self, backend: str = "passthrough") -> None:
        self.backend = backend
        self._ref_buf: deque[np.ndarray] = deque(maxlen=8)  # ~1.6s
        if backend == "webrtc":
            self.aec = AECWebRTC()
        else:
            self.aec = AECPassthrough()

    async def run(self, bus: EventBus) -> None:
        q = bus.subscribe()
        while True:
            ev = await q.get()
            if ev.type == EvType.TTS_OUT_CHUNK:
                self._ref_buf.append(ev.data)
            elif ev.type == EvType.AUDIO_IN_CHUNK:
                ref = self._ref_buf.popleft() if self._ref_buf else None
                cleaned = self.aec.process(ev.data, ref)
                # rewrite event in-place: emit a new event that downstream ASR uses
                await bus.emit(EvType.AUDIO_IN_CHUNK, data=cleaned, aec=True)
