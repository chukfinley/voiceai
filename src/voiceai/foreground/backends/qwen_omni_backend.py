"""Qwen3-Omni streaming backend.

Two implementation paths:
  A) `transformers` Qwen3OmniMoeForConditionalGeneration with a custom
     streaming pump. Works for dev but not optimized.
  B) vLLM AsyncLLMEngine with audio-input + audio-output streaming.
     Production target. Requires vLLM >= 0.6 with multimodal audio support.

We pick (A) by default for the skeleton and leave a TODO for (B).
"""
from __future__ import annotations

import asyncio
from collections.abc import AsyncIterator
from dataclasses import dataclass

import numpy as np


@dataclass
class OmniPiece:
    text: str | None = None
    audio: np.ndarray | None = None  # 16kHz mono float32


class Qwen3OmniStreamer:
    """Persistent streaming session.

    Maintains: model, processor, KV cache, audio-token buffer in-flight.

    Real impl skeleton (transformers path):

        from transformers import Qwen3OmniMoeForConditionalGeneration, Qwen3OmniProcessor
        self.model = Qwen3OmniMoeForConditionalGeneration.from_pretrained(
            model_id, torch_dtype=dtype, device_map=device
        )
        self.processor = Qwen3OmniProcessor.from_pretrained(model_id)
        # Use the model's `streaming_chat` or manual interleaved generate loop.

    Talker output: codec tokens decoded by Code2Wav incrementally.
    """

    def __init__(
        self,
        model_id: str,
        dtype: str = "bfloat16",
        device: str = "cuda",
        voice: str = "ethan",
    ) -> None:
        self.model_id = model_id
        self.dtype = dtype
        self.device = device
        self.voice = voice
        self._in_q: asyncio.Queue = asyncio.Queue()
        self._out_q: asyncio.Queue[OmniPiece] = asyncio.Queue()

    async def start(self) -> None:
        # TODO: load model + processor; start background pump task.
        pass

    async def feed_audio(self, pcm: np.ndarray) -> None:
        await self._in_q.put(("audio", pcm))

    async def feed_image(self, img) -> None:
        await self._in_q.put(("image", img))

    async def feed_text(self, text: str) -> None:
        await self._in_q.put(("text", text))

    async def stream_out(self) -> AsyncIterator[OmniPiece]:
        # Stub: yields silence
        while True:
            await asyncio.sleep(0.2)
            yield OmniPiece(audio=np.zeros(3200, dtype=np.float32))
