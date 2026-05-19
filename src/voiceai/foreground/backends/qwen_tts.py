"""Streaming TTS wrapper. Default Qwen3-TTS, swap for CosyVoice2/Orpheus.

Yields 200ms PCM (16kHz mono float32) chunks aligned to orchestrator tick.
"""
from __future__ import annotations

from collections.abc import AsyncIterator

import numpy as np


class StreamingTTS:
    def __init__(self, model_id: str) -> None:
        self.model_id = model_id
        # TODO: load. Recommended for prod:
        #   - Qwen/Qwen3-TTS-Flash (multilingual, fast)
        #   - FunAudioLLM/CosyVoice2 (clone voice, streaming)
        #   - canopylabs/orpheus-tts-0.1-finetune-prod (emotion tags)

    async def synthesize(self, text: str) -> AsyncIterator[np.ndarray]:
        # Stub: 200ms of silence per call
        yield np.zeros(3200, dtype=np.float32)
