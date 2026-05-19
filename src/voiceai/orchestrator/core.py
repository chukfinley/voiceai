"""Central orchestrator. Async event bus + loop scheduler.

Design follows TML interaction-model pattern at a coarse level:
  - foreground emits speech tokens at ~5Hz (200ms micro-turn)
  - background runs heavy reasoning async
  - shared context window with frame-aligned tick markers

Stack-agnostic. Plug-in foreground (cascade or qwen-omni), background, visual.
"""
from __future__ import annotations

import asyncio
import time
from collections.abc import AsyncIterator
from typing import Protocol

from .events import Event, EvType


class Loop(Protocol):
    """Any component that produces/consumes events implements run(bus)."""

    async def run(self, bus: "EventBus") -> None: ...


class EventBus:
    """Pub/sub with per-subscriber unbounded queues.

    Keep it dumb. Backpressure is the subscriber's problem — they drop or
    coalesce as needed.
    """

    def __init__(self) -> None:
        self._subs: list[asyncio.Queue[Event]] = []
        self._t0 = time.monotonic()

    def now(self) -> float:
        return time.monotonic() - self._t0

    def subscribe(self) -> asyncio.Queue[Event]:
        q: asyncio.Queue[Event] = asyncio.Queue()
        self._subs.append(q)
        return q

    async def emit(self, type_: EvType, data=None, **meta) -> None:
        ev = Event(type=type_, t=self.now(), data=data, meta=meta)
        for q in self._subs:
            q.put_nowait(ev)

    async def stream(self, *types: EvType) -> AsyncIterator[Event]:
        """Convenience: yield only events of given types."""
        q = self.subscribe()
        wanted = set(types) if types else None
        while True:
            ev = await q.get()
            if wanted is None or ev.type in wanted:
                yield ev


class Orchestrator:
    """Runs loops concurrently and waits for them to finish (or one to fail)."""

    def __init__(self) -> None:
        self.bus = EventBus()
        self.loops: list[Loop] = []

    def add(self, loop: Loop) -> "Orchestrator":
        self.loops.append(loop)
        return self

    async def run(self) -> None:
        async with asyncio.TaskGroup() as tg:
            for loop in self.loops:
                tg.create_task(loop.run(self.bus), name=type(loop).__name__)


class TickLoop:
    """Emits TICK every `period` seconds with elapsed time.

    Used by foreground LLM to know wall-clock time without external state.
    Matches TML's time-aware design: tokens like <t:Ns> get injected into
    the model context by the foreground wrapper.
    """

    def __init__(self, period: float = 0.5) -> None:
        self.period = period

    async def run(self, bus: EventBus) -> None:
        while True:
            await bus.emit(EvType.TICK, data=bus.now())
            await asyncio.sleep(self.period)
