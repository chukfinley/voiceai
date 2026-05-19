"""Option B foreground: Qwen3-Omni-30B-A3B native end-to-end.

Single model: audio+video in, speech+text out, thinker-talker streaming.
We expose the same Loop interface as CascadeForeground so the orchestrator
doesn't care which is plugged in.

Key streaming bits (from Qwen3-Omni tech report):
  - AuT encoder produces audio tokens at ~12.5Hz
  - Thinker MoE (3B active) emits text tokens
  - Talker MoE consumes Thinker hidden states + emits multi-codebook audio tokens
  - MTP head outputs residual codebooks each frame
  - Code2Wav incrementally synthesizes 200ms waveform chunks

Our wrapper:
  1. Subscribes to AUDIO_IN_CHUNK (+ optional VISUAL_FRAME).
  2. Maintains a streaming session in the model.
  3. Injects <t:Ns> time markers and <visual:event=...> on tick / visual events.
  4. Emits LLM_TEXT_DELTA + LLM_AUDIO_DELTA → TTS_OUT_CHUNK.
  5. Detects <background_query>...</background_query> tokens in text stream
     and dispatches them to the Background loop.
"""
from __future__ import annotations

import asyncio
import re
from dataclasses import dataclass, field

import numpy as np

from ..orchestrator.core import EventBus
from ..orchestrator.events import EvType


_BG_QUERY_RE = re.compile(r"<background_query>(.*?)</background_query>", re.DOTALL)


@dataclass
class OmniConfig:
    model_id: str = "Qwen/Qwen3-Omni-30B-A3B-Instruct"
    dtype: str = "bfloat16"
    device: str = "cuda"
    # tick → token injection
    inject_tick: bool = True
    inject_visual: bool = True
    # streaming
    frame_ms: int = 200
    # background bridge
    background_enabled: bool = True
    # voice
    voice: str = "ethan"


class OmniForeground:
    def __init__(self, cfg: OmniConfig | None = None) -> None:
        self.cfg = cfg or OmniConfig()
        self._session = None
        self._text_acc = ""
        self._pending_visual: list[dict] = []

    async def run(self, bus: EventBus) -> None:
        from .backends.qwen_omni_backend import Qwen3OmniStreamer

        self._session = Qwen3OmniStreamer(
            model_id=self.cfg.model_id,
            dtype=self.cfg.dtype,
            device=self.cfg.device,
            voice=self.cfg.voice,
        )
        await self._session.start()

        async with asyncio.TaskGroup() as tg:
            tg.create_task(self._input_loop(bus), name="omni_in")
            tg.create_task(self._output_loop(bus), name="omni_out")

    async def _input_loop(self, bus: EventBus) -> None:
        q = bus.subscribe()
        while True:
            ev = await q.get()
            if ev.type == EvType.AUDIO_IN_CHUNK:
                await self._session.feed_audio(ev.data)
            elif ev.type == EvType.VISUAL_FRAME and self.cfg.inject_visual:
                await self._session.feed_image(ev.data)
            elif ev.type == EvType.VISUAL_EVENT and self.cfg.inject_visual:
                self._pending_visual.append(ev.data)
            elif ev.type == EvType.TICK and self.cfg.inject_tick:
                # Inject <t:Ns> as a control token. Cheap.
                await self._session.feed_text(f"<t:{ev.t:.1f}>")
                # Drain pending visual classifier events as tokens
                while self._pending_visual:
                    v = self._pending_visual.pop(0)
                    await self._session.feed_text(
                        f"<visual:{v['event']},conf={v.get('confidence', 0):.2f}>"
                    )
            elif ev.type == EvType.BG_RESULT:
                # background returned — inject as system observation
                await self._session.feed_text(
                    f"<bg_result id={ev.meta.get('id')}>{ev.data}</bg_result>"
                )

    async def _output_loop(self, bus: EventBus) -> None:
        async for piece in self._session.stream_out():
            if piece.text:
                self._text_acc += piece.text
                await bus.emit(EvType.LLM_TEXT_DELTA, data=piece.text)
                await self._maybe_dispatch_background(bus)
            if piece.audio is not None:
                # Pass audio straight to speaker — Talker already produced PCM.
                await bus.emit(EvType.TTS_OUT_CHUNK, data=piece.audio)

    async def _maybe_dispatch_background(self, bus: EventBus) -> None:
        if not self.cfg.background_enabled:
            return
        match = _BG_QUERY_RE.search(self._text_acc)
        if match:
            query = match.group(1).strip()
            self._text_acc = self._text_acc[match.end():]
            await bus.emit(EvType.BG_QUERY, data=query, id=id(match))
