"""Option A foreground: cascade Qwen3-ASR → Qwen3-0.6B → Qwen3-TTS.

Streaming-aware: ASR emits partials, LLM starts generating speculatively on
partials, cancels on revisions. TTS streams sentence-by-sentence so first
audio arrives before LLM finishes.

This is the bootstrap stack. Cheap, fast to ship, generates training data
for the native Qwen3-Omni path.
"""
from __future__ import annotations

import asyncio
import re
from dataclasses import dataclass

import numpy as np

from ..orchestrator.core import EventBus
from ..orchestrator.events import EvType


@dataclass
class CascadeConfig:
    asr_model: str = "Qwen/Qwen3-ASR-Flash"
    llm_model: str = "Qwen/Qwen3-0.6B-Instruct"
    tts_model: str = "Qwen/Qwen3-TTS-Flash"
    llm_max_new_tokens: int = 256
    speculative_partials: bool = True
    sentence_split_re: str = r"(?<=[.!?])\s+"


class CascadeForeground:
    """Cascade pipeline orchestrating ASR/LLM/TTS as three sub-loops.

    Sub-loops share state via the bus. ASR commits flush LLM context;
    LLM tokens stream to TTS sentence-buffer; TTS chunks emit to speaker.
    """

    def __init__(self, cfg: CascadeConfig | None = None) -> None:
        self.cfg = cfg or CascadeConfig()
        self._llm_task: asyncio.Task | None = None
        self._tts_task: asyncio.Task | None = None
        self._history: list[dict] = [
            {"role": "system", "content": _SYSTEM_PROMPT},
        ]

    async def run(self, bus: EventBus) -> None:
        # Lazy import: model loads happen here so the orchestrator can start
        # even without GPUs (tests, CI).
        from .backends import qwen_asr, qwen_llm, qwen_tts

        self.asr = qwen_asr.StreamingASR(self.cfg.asr_model)
        self.llm = qwen_llm.StreamingLLM(self.cfg.llm_model)
        self.tts = qwen_tts.StreamingTTS(self.cfg.tts_model)

        # ASR consumes mic chunks, emits USER_PARTIAL / USER_COMMIT
        async def asr_loop():
            q = bus.subscribe()
            async for partial, committed in self.asr.stream(_audio_chunks(q)):
                if committed is not None:
                    await bus.emit(EvType.USER_COMMIT, data=committed)
                elif partial is not None:
                    await bus.emit(EvType.USER_PARTIAL, data=partial)

        # LLM consumes commits (and optionally partials), produces text deltas
        async def llm_loop():
            q = bus.subscribe()
            while True:
                ev = await q.get()
                if ev.type == EvType.USER_COMMIT:
                    await self._maybe_cancel_speculation()
                    await self._kick_llm(ev.data, bus)
                elif ev.type == EvType.USER_PARTIAL and self.cfg.speculative_partials:
                    # only speculate if user paused — heuristic: partial unchanged ≥300ms
                    pass  # TODO: add speculation logic
                elif ev.type == EvType.BARGE_IN:
                    await self._maybe_cancel_speculation()
                    await bus.emit(EvType.TTS_MUTE, data=True)

        # TTS consumes text deltas, emits PCM chunks
        async def tts_loop():
            q = bus.subscribe()
            buffer = ""
            async for ev in _stream(q, EvType.LLM_TEXT_DELTA, EvType.LLM_DONE):
                if ev.type == EvType.LLM_DONE:
                    if buffer.strip():
                        async for pcm in self.tts.synthesize(buffer):
                            await bus.emit(EvType.TTS_OUT_CHUNK, data=pcm)
                    buffer = ""
                    continue
                buffer += ev.data
                # flush on sentence boundary for low first-audio latency
                parts = re.split(self.cfg.sentence_split_re, buffer)
                if len(parts) > 1:
                    *complete, buffer = parts
                    for sent in complete:
                        async for pcm in self.tts.synthesize(sent):
                            await bus.emit(EvType.TTS_OUT_CHUNK, data=pcm)

        async with asyncio.TaskGroup() as tg:
            tg.create_task(asr_loop(), name="asr")
            tg.create_task(llm_loop(), name="llm")
            tg.create_task(tts_loop(), name="tts")

    async def _maybe_cancel_speculation(self) -> None:
        if self._llm_task and not self._llm_task.done():
            self._llm_task.cancel()

    async def _kick_llm(self, user_text: str, bus: EventBus) -> None:
        self._history.append({"role": "user", "content": user_text})
        assistant = ""

        async def gen():
            nonlocal assistant
            try:
                async for tok in self.llm.stream(
                    self._history, max_new_tokens=self.cfg.llm_max_new_tokens
                ):
                    assistant += tok
                    await bus.emit(EvType.LLM_TEXT_DELTA, data=tok)
                await bus.emit(EvType.LLM_DONE)
                self._history.append({"role": "assistant", "content": assistant})
            except asyncio.CancelledError:
                # partial response: still record what we said
                if assistant:
                    self._history.append(
                        {"role": "assistant", "content": assistant + " [cut]"}
                    )
                raise

        self._llm_task = asyncio.create_task(gen())


_SYSTEM_PROMPT = """You are a helpful voice assistant. English only.

Rules:
- Keep replies SHORT. Usually 1-2 sentences.
- You receive time markers <t:Ns> showing elapsed seconds. React if relevant.
- On <visual:event=X> a visual event was detected — fold it in if useful.
- For complex questions emit <background_query>question</background_query>
  which will be answered async by a larger model.
- Emit <silent/> when there is nothing useful to say.
"""


async def _audio_chunks(q: asyncio.Queue):
    """Adapter: subscribe to AUDIO_IN_CHUNK events as an async iterable."""
    while True:
        ev = await q.get()
        if ev.type == EvType.AUDIO_IN_CHUNK:
            yield ev.data


async def _stream(q: asyncio.Queue, *types: EvType):
    wanted = set(types)
    while True:
        ev = await q.get()
        if ev.type in wanted:
            yield ev
