"""Camera source. cv2 wrapper, falls back to file source for tests."""
from __future__ import annotations

import asyncio
from collections.abc import AsyncIterator

import numpy as np


class CameraSource:
    def __init__(self, device: int = 0, width: int = 640, height: int = 480) -> None:
        self.device = device
        self.width = width
        self.height = height

    async def stream(self) -> AsyncIterator[np.ndarray]:
        try:
            import cv2
        except ImportError:
            # No camera available — yield black frames @ 2fps
            while True:
                await asyncio.sleep(0.5)
                yield np.zeros((self.height, self.width, 3), dtype=np.uint8)
            return

        cap = cv2.VideoCapture(self.device)
        try:
            while True:
                ok, frame = cap.read()
                if not ok:
                    await asyncio.sleep(0.5)
                    continue
                yield frame
                await asyncio.sleep(0.05)
        finally:
            cap.release()
