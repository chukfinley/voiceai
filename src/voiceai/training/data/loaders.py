"""Dataset loaders. HF datasets-compatible.

Three loaders matching three training phases:

  - PairedDialogDataset   : (audio_user_24khz, audio_asst_24khz, transcript, timing)
                            for Phase-2 dual-stream SFT. Built from Fisher,
                            CallHome, CANDOR, synthetic Qwen3-Omni × Qwen3-Omni.

  - TimeAwareSFTDataset   : (frames as flat token list, masked targets)
                            for LoRA tuning on <t:>, <wait:>, <silent>, etc.

  - VisualProactiveDataset : (video frames, frame-aligned speak/silent labels)
                            for Streaming-EOS objective (VideoLLM-online style).
"""
from __future__ import annotations

from collections.abc import Iterator
from dataclasses import dataclass
from pathlib import Path

import numpy as np

from .format import Frame, flatten


@dataclass
class DialogSample:
    user_audio: np.ndarray            # 16kHz mono float32
    asst_audio: np.ndarray            # 16kHz mono float32, same length, both aligned
    asst_text: str
    timing: list[tuple[float, float]] # word-level [(start_s, end_s)]


class PairedDialogDataset:
    """Iterable dataset over (user, assistant) paired audio.

    Expected directory layout:
        <root>/
          <id>/user.wav
          <id>/asst.wav
          <id>/asst.json       # {"text": "...", "timing": [[s, e], ...]}
    """

    def __init__(self, root: str | Path) -> None:
        self.root = Path(root)

    def __iter__(self) -> Iterator[DialogSample]:
        import json
        import soundfile as sf

        for sub in sorted(self.root.iterdir()):
            if not sub.is_dir():
                continue
            try:
                u, _ = sf.read(sub / "user.wav", dtype="float32")
                a, _ = sf.read(sub / "asst.wav", dtype="float32")
                meta = json.loads((sub / "asst.json").read_text())
            except FileNotFoundError:
                continue
            yield DialogSample(
                user_audio=u,
                asst_audio=a,
                asst_text=meta["text"],
                timing=meta.get("timing", []),
            )


class TimeAwareSFTDataset:
    """Wraps a frame-list jsonl file: each line = list[Frame] as JSON.

    Yields flat token-string sequences ready for tokenizer.encode().
    """

    def __init__(self, path: str | Path) -> None:
        self.path = Path(path)

    def __iter__(self) -> Iterator[list[str]]:
        import json

        with self.path.open() as f:
            for line in f:
                frames_data = json.loads(line)
                frames = [Frame(**fd) for fd in frames_data]
                yield flatten(frames)


class VisualProactiveDataset:
    """Frames + binary speak/silent labels per frame for Streaming-EOS.

    Directory:
        <root>/<id>/frames/000001.jpg ...
        <root>/<id>/labels.npy        # shape [T], 1=speak, 0=silent
    """

    def __init__(self, root: str | Path) -> None:
        self.root = Path(root)

    def __iter__(self) -> Iterator[tuple[list[np.ndarray], np.ndarray]]:
        try:
            import cv2
        except ImportError as e:
            raise RuntimeError("Install opencv-python for visual dataset") from e

        for sub in sorted(self.root.iterdir()):
            if not sub.is_dir():
                continue
            labels = np.load(sub / "labels.npy")
            frames_dir = sub / "frames"
            frames = []
            for p in sorted(frames_dir.glob("*.jpg")):
                frames.append(cv2.imread(str(p)))
            if len(frames) != len(labels):
                continue
            yield frames, labels
