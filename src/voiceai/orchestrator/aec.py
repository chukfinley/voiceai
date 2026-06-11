"""Acoustic Echo Cancellation — strips own speaker output from mic input.

The self-hearing problem has THREE layers of defense in this stack:

  1. AEC (this module): subtract the known speaker signal from the mic
     signal before it ever reaches Mimi encoding. Classic DSP, handles the
     bulk of the echo.
  2. Model input design: the model hears its own stream through the
     DEDICATED asst input channel (audio_in_asst), so "what I said" and
     "what the user said" are architecturally separate. AEC only has to
     keep the user channel clean, not tell the model what it said.
  3. Training robustness: echo-bleed augmentation (see
     training/data/mixing.add_echo_bleed) mixes attenuated assistant audio
     into the user channel during data generation, so the model tolerates
     the residual echo that AEC always leaves behind.

Backends:
  - webrtc (webrtc-audio-processing, BSD) — real AEC, preferred
  - passthrough — dev/no-op

AECLoop subscribes to AUDIO_IN_CHUNK + TTS_OUT_CHUNK and republishes cleaned
AUDIO_IN_CHUNK events tagged meta={"aec": True}. Downstream consumers (ASR,
Mimi encoder, speaker-ID) should prefer aec-tagged chunks when an AECLoop is
running.
"""
from __future__ import annotations

import numpy as np

from .core import EventBus
from .events import EvType

SR = 16000
FRAME_10MS = SR // 100  # WebRTC APM operates on 10 ms frames


class AECPassthrough:
    """No-op for dev. Real impl uses webrtc-audio-processing."""

    def process(self, mic: np.ndarray, ref: np.ndarray | None) -> np.ndarray:
        return mic


class AECWebRTC:
    """Wraps webrtc-audio-processing. Lazy import.

    The APM consumes 10 ms int16 frames: for each mic frame we first feed the
    time-aligned reference (reverse stream), then process the mic frame.
    """

    def __init__(self) -> None:
        try:
            from webrtc_audio_processing import AudioProcessingModule  # type: ignore
        except ImportError as e:
            raise RuntimeError(
                "Install webrtc-audio-processing: pip install webrtc-audio-processing"
            ) from e
        self.apm = AudioProcessingModule(aec_type=2, enable_ns=True)
        self.apm.set_stream_format(SR, 1)
        self.apm.set_reverse_stream_format(SR, 1)

    def process(self, mic: np.ndarray, ref: np.ndarray | None) -> np.ndarray:
        if ref is None:
            ref = np.zeros_like(mic)
        n = len(mic)
        if len(ref) < n:
            ref = np.pad(ref, (0, n - len(ref)))
        out = np.empty(n, dtype=np.float32)
        # re-chunk into the 10 ms frames the APM actually accepts
        for i in range(0, n - n % FRAME_10MS, FRAME_10MS):
            self.apm.process_reverse_stream(
                _f32_to_int16(ref[i : i + FRAME_10MS]).tobytes()
            )
            cleaned = self.apm.process_stream(
                _f32_to_int16(mic[i : i + FRAME_10MS]).tobytes()
            )
            out[i : i + FRAME_10MS] = _int16_bytes_to_f32(cleaned)
        tail = n % FRAME_10MS
        if tail:
            out[n - tail :] = mic[n - tail :]  # <10 ms leftover passes through
        return out


def _f32_to_int16(x: np.ndarray) -> np.ndarray:
    return np.clip(x * 32767.0, -32768, 32767).astype(np.int16)


def _int16_bytes_to_f32(b: bytes) -> np.ndarray:
    return np.frombuffer(b, dtype=np.int16).astype(np.float32) / 32768.0


class _RefBuffer:
    """Sample-accurate ring buffer of everything sent to the speaker.

    `read(n)` returns the n reference samples assumed to be coming out of the
    speaker right now, compensating the output-path delay (audio callback
    latency + DAC). WebRTC's internal delay search covers the rest.
    """

    def __init__(self, delay_ms: float = 60.0, max_seconds: float = 4.0):
        self.delay_samples = int(SR * delay_ms / 1000)
        self.max_samples = int(SR * max_seconds)
        self._buf = np.zeros(self.delay_samples, dtype=np.float32)

    def write(self, chunk: np.ndarray) -> None:
        self._buf = np.concatenate([self._buf, np.asarray(chunk, dtype=np.float32)])
        if len(self._buf) > self.max_samples:
            self._buf = self._buf[-self.max_samples :]

    def read(self, n: int) -> np.ndarray:
        if len(self._buf) >= n:
            out, self._buf = self._buf[:n], self._buf[n:]
            return out
        out = np.pad(self._buf, (0, n - len(self._buf)))
        self._buf = np.zeros(0, dtype=np.float32)
        return out


class AECLoop:
    """Subscribes to mic + speaker streams, republishes cleaned mic chunks."""

    def __init__(self, backend: str = "passthrough", delay_ms: float = 60.0) -> None:
        self.backend = backend
        self._ref = _RefBuffer(delay_ms=delay_ms)
        if backend == "webrtc":
            self.aec = AECWebRTC()
        else:
            self.aec = AECPassthrough()

    async def run(self, bus: EventBus) -> None:
        q = bus.subscribe()
        while True:
            ev = await q.get()
            if ev.type == EvType.TTS_OUT_CHUNK:
                self._ref.write(ev.data)
            elif ev.type == EvType.AUDIO_IN_CHUNK:
                if ev.meta.get("aec"):
                    continue  # our own republished event — never re-process
                mic = np.asarray(ev.data, dtype=np.float32)
                ref = self._ref.read(len(mic))
                cleaned = self.aec.process(mic, ref)
                await bus.emit(EvType.AUDIO_IN_CHUNK, data=cleaned, aec=True)
