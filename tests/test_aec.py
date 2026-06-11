"""AEC loop + echo-bleed augmentation tests (no audio hardware, no webrtc dep).

    uv run pytest tests/test_aec.py -v
"""
from __future__ import annotations

import asyncio

import numpy as np
import pytest

from voiceai.orchestrator.aec import AECLoop, _RefBuffer
from voiceai.orchestrator.core import EventBus
from voiceai.orchestrator.events import EvType
from voiceai.training.data.mixing import add_echo_bleed


def test_ref_buffer_delay_and_alignment():
    rb = _RefBuffer(delay_ms=10.0)  # 160 samples of initial silence
    rb.write(np.ones(320, dtype=np.float32))
    first = rb.read(160)
    assert (first == 0).all()  # delay compensation: silence first
    second = rb.read(320)
    assert (second == 1).all()
    # underrun → zero-padded
    third = rb.read(100)
    assert third.shape == (100,) and (third == 0).all()


def test_aec_loop_no_self_reprocessing():
    """The cleaned AUDIO_IN_CHUNK (aec=True) must not be processed again."""

    async def scenario():
        bus = EventBus()
        loop = AECLoop(backend="passthrough")
        collector = bus.subscribe()
        task = asyncio.create_task(loop.run(bus))
        await asyncio.sleep(0)  # let the loop subscribe before emitting

        await bus.emit(EvType.TTS_OUT_CHUNK, data=np.ones(3200, dtype=np.float32))
        await bus.emit(EvType.AUDIO_IN_CHUNK, data=np.zeros(3200, dtype=np.float32))
        await asyncio.sleep(0.1)
        task.cancel()

        cleaned = []
        while not collector.empty():
            ev = collector.get_nowait()
            if ev.type == EvType.AUDIO_IN_CHUNK and ev.meta.get("aec"):
                cleaned.append(ev)
        return cleaned

    cleaned = asyncio.run(scenario())
    # exactly ONE cleaned event per raw chunk — old code recursed forever here
    assert len(cleaned) == 1
    assert cleaned[0].data.shape == (3200,)


def test_add_echo_bleed():
    sr = 16000
    user = np.zeros(sr, dtype=np.float32)
    asst = np.ones(sr, dtype=np.float32)
    out = add_echo_bleed(user, asst, sr, gain=0.1, delay_ms=100.0)
    delay = int(sr * 0.1)
    assert (out[:delay] == 0).all()              # nothing before the delay
    assert np.allclose(out[delay:], 0.1)          # attenuated asst after
    assert out.shape == user.shape
