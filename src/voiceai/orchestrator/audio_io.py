"""Audio in/out loops. 16kHz mono throughout the stack."""
from __future__ import annotations

import asyncio
from collections.abc import AsyncIterator

import numpy as np
import sounddevice as sd

from .core import EventBus
from .events import EvType

SAMPLE_RATE = 16000
CHUNK_MS = 200
CHUNK_SAMPLES = SAMPLE_RATE * CHUNK_MS // 1000


class MicLoop:
    """Captures 200ms PCM chunks from default mic, emits AUDIO_IN_CHUNK."""

    def __init__(self, device: int | None = None) -> None:
        self.device = device

    async def run(self, bus: EventBus) -> None:
        loop = asyncio.get_running_loop()
        q: asyncio.Queue[np.ndarray] = asyncio.Queue()

        def cb(indata, frames, time_info, status):
            if status:
                # underruns happen; we don't care, downstream can interpolate
                pass
            loop.call_soon_threadsafe(q.put_nowait, indata.copy().reshape(-1))

        with sd.InputStream(
            samplerate=SAMPLE_RATE,
            channels=1,
            dtype="float32",
            blocksize=CHUNK_SAMPLES,
            device=self.device,
            callback=cb,
        ):
            while True:
                chunk = await q.get()
                await bus.emit(EvType.AUDIO_IN_CHUNK, data=chunk)


class SpeakerLoop:
    """Plays TTS_OUT_CHUNK events. Implements duck/mute via TTS_DUCK/TTS_MUTE."""

    def __init__(self, device: int | None = None) -> None:
        self.device = device
        self._volume = 1.0
        self._muted = False

    async def run(self, bus: EventBus) -> None:
        q = bus.subscribe()
        out_q: asyncio.Queue[np.ndarray] = asyncio.Queue()
        loop = asyncio.get_running_loop()

        def cb(outdata, frames, time_info, status):
            try:
                chunk = out_q.get_nowait()
                outdata[:] = chunk.reshape(-1, 1)
            except asyncio.QueueEmpty:
                outdata.fill(0.0)

        with sd.OutputStream(
            samplerate=SAMPLE_RATE,
            channels=1,
            dtype="float32",
            blocksize=CHUNK_SAMPLES,
            device=self.device,
            callback=cb,
        ):
            while True:
                ev = await q.get()
                if ev.type == EvType.TTS_OUT_CHUNK:
                    if self._muted:
                        continue
                    pcm = (ev.data * self._volume).astype("float32")
                    out_q.put_nowait(pcm)
                elif ev.type == EvType.TTS_DUCK:
                    self._volume = float(ev.data or 0.2)
                elif ev.type == EvType.TTS_MUTE:
                    self._muted = bool(ev.data)
