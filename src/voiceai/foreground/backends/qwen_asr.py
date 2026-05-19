"""Streaming ASR wrapper around Qwen3-ASR-Flash.

Production swap: use Qwen3-ASR-Flash via Alibaba Model Studio API if real-time
GPU not available; or Parakeet-TDT-0.6B-v3 (NeMo) which is faster on consumer
hardware. Both produce (partial, committed) tuples.

Interface (everywhere in this project):
    async def stream(chunks: AsyncIterator[np.ndarray])
        -> AsyncIterator[tuple[partial | None, committed | None]]
"""
from __future__ import annotations

import asyncio
from collections.abc import AsyncIterator

import numpy as np


class StreamingASR:
    def __init__(self, model_id: str) -> None:
        self.model_id = model_id
        # TODO: load model. Choice tree:
        #   - "Qwen/Qwen3-ASR-Flash"  → via DashScope or transformers
        #   - "nvidia/parakeet-tdt-0.6b-v3" → via nemo_toolkit (fastest local)
        #   - "openai/whisper-large-v3-turbo" → faster-whisper for streaming
        # For dev we keep a stub.

    async def stream(
        self, chunks: AsyncIterator[np.ndarray]
    ) -> AsyncIterator[tuple[str | None, str | None]]:
        """Yields (partial, None) for partials, (None, text) on commit."""
        # Stub: pretends to commit every 2s of audio.
        n = 0
        async for chunk in chunks:
            n += len(chunk)
            if n >= 16000 * 2:
                n = 0
                yield (None, "[committed audio]")
            else:
                yield ("...", None)
