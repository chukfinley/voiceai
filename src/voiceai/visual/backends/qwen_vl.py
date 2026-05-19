"""Qwen3-VL wrapper. Single-frame classify interface for the watcher."""
from __future__ import annotations

import numpy as np


class StreamingVLM:
    def __init__(self, model_id: str) -> None:
        self.model_id = model_id

    async def start(self) -> None:
        # TODO: load via transformers Qwen3VLForConditionalGeneration
        pass

    async def classify(self, frame: np.ndarray, prompt: str) -> str:
        # Stub
        return '{"event": null, "confidence": 0.0, "urgency": 0}'
