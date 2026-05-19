"""Visual proactivity watcher.

Runs a small VLM (Qwen3-VL-2B or Proact-VL) at 2 fps on the camera stream.
Outputs structured events that the foreground interleaves into its context.

Two strategies:
  (a) Classifier prompt — VLM emits JSON with event/confidence/urgency.
  (b) Streaming-EOS objective (VideoLLM-online): the VLM is trained to predict
      "speak or stay silent" at each frame. Better but requires our own SFT.

Strategy (a) for bootstrap; (b) when we own the model.
"""
from __future__ import annotations

import asyncio
import json
from dataclasses import dataclass

from ..orchestrator.core import EventBus
from ..orchestrator.events import EvType

WATCH_PROMPT = """Analyze this single camera frame.

Reply with JSON: {"event": "string or null", "confidence": 0.0-1.0, "urgency": 0-3}

Possible events: user_waved, user_pointed, user_holding_object,
user_looking_away, user_confused, environment_changed, danger_detected,
task_completed, object_count_changed, gesture_thumbs_up, gesture_stop.

Reply with null if nothing relevant happened.
"""


@dataclass
class WatcherConfig:
    model_id: str = "Qwen/Qwen3-VL-2B-Instruct"
    fps: float = 2.0
    confidence_threshold: float = 0.6
    camera_device: int = 0


class VisualWatcher:
    def __init__(self, cfg: WatcherConfig | None = None) -> None:
        self.cfg = cfg or WatcherConfig()

    async def run(self, bus: EventBus) -> None:
        from .backends.qwen_vl import StreamingVLM
        from .camera import CameraSource

        self.vlm = StreamingVLM(self.cfg.model_id)
        self.cam = CameraSource(device=self.cfg.camera_device)
        interval = 1.0 / self.cfg.fps

        await self.vlm.start()
        async for frame in self.cam.stream():
            await bus.emit(EvType.VISUAL_FRAME, data=frame)
            try:
                raw = await self.vlm.classify(frame, WATCH_PROMPT)
                event = json.loads(raw)
            except Exception:
                event = None
            if (
                event
                and event.get("event")
                and event.get("confidence", 0) >= self.cfg.confidence_threshold
            ):
                await bus.emit(EvType.VISUAL_EVENT, data=event)
            await asyncio.sleep(interval)
