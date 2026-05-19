"""Streaming LLM wrapper for foreground (Qwen3-0.6B-Instruct).

Foreground stays small for low TTFT. Heavy reasoning is delegated to the
background model via tool-call protocol — see voiceai.background.
"""
from __future__ import annotations

from collections.abc import AsyncIterator


class StreamingLLM:
    def __init__(self, model_id: str) -> None:
        self.model_id = model_id
        # TODO: transformers TextIteratorStreamer or vLLM AsyncLLMEngine.
        # Quantize FP8 / 4-bit AWQ for speed.

    async def stream(
        self, messages: list[dict], max_new_tokens: int = 256
    ) -> AsyncIterator[str]:
        # Stub
        yield "Hi. "
        yield "How can I help?"
