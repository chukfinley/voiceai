"""Dual-stream dataset: paired (user_audio, asst_audio) -> Mimi-coded tensors.

A sample for Stage 2/3 is:

    user_codes:  [num_codebooks, T]  Mimi codes for user channel
    asst_codes:  [num_codebooks, T]  Mimi codes for assistant channel
    text_ids:    [T_text]            tokenized Inner-Monologue text (assistant)
    text_align:  [T_text]            frame index each text token aligns to
    aux:         dict — visual events, bg_results, control flags by frame

Stored on disk as a directory tree:

    <root>/
      <sample_id>.npz       # numpy archive: user_codes, asst_codes, text_ids, text_align
      <sample_id>.json      # aux metadata, source attribution, duration

This shape lets us shuffle / shard cheaply during training.
"""
from __future__ import annotations

import json
from collections.abc import Iterator
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import torch
from torch.utils.data import Dataset, IterableDataset


@dataclass
class DualStreamSample:
    user_codes: np.ndarray         # [K, T] int32, K=num_codebooks
    asst_codes: np.ndarray         # [K, T] int32
    text_ids: np.ndarray           # [T_text] int32 (token ids, backbone vocab)
    text_align: np.ndarray         # [T_text] int32 (frame index)
    aux: dict                      # {"visual_events":[...], "barge_at":[...], ...}
    sample_id: str = ""
    duration_s: float = 0.0


def save_sample(sample: DualStreamSample, root: Path) -> Path:
    root = Path(root)
    root.mkdir(parents=True, exist_ok=True)
    out = root / f"{sample.sample_id}.npz"
    np.savez_compressed(
        out,
        user_codes=sample.user_codes.astype(np.int32),
        asst_codes=sample.asst_codes.astype(np.int32),
        text_ids=sample.text_ids.astype(np.int32),
        text_align=sample.text_align.astype(np.int32),
    )
    (root / f"{sample.sample_id}.json").write_text(
        json.dumps(
            {"aux": sample.aux, "duration_s": sample.duration_s, "sample_id": sample.sample_id}
        )
    )
    return out


def load_sample(path: Path | str) -> DualStreamSample:
    path = Path(path)
    if path.suffix == ".json":
        path = path.with_suffix(".npz")
    arr = np.load(path)
    meta = json.loads(path.with_suffix(".json").read_text())
    return DualStreamSample(
        user_codes=arr["user_codes"],
        asst_codes=arr["asst_codes"],
        text_ids=arr["text_ids"],
        text_align=arr["text_align"],
        aux=meta.get("aux", {}),
        sample_id=meta.get("sample_id", path.stem),
        duration_s=meta.get("duration_s", 0.0),
    )


class DualStreamDataset(Dataset):
    """Map-style dataset over a directory of .npz/.json samples."""

    def __init__(self, root: str | Path, pad_to_frames: int | None = None):
        self.root = Path(root)
        self.files = sorted(self.root.glob("*.npz"))
        if not self.files:
            raise FileNotFoundError(f"no .npz samples found in {self.root}")
        self.pad_to = pad_to_frames

    def __len__(self) -> int:
        return len(self.files)

    def __getitem__(self, idx: int) -> dict[str, torch.Tensor]:
        s = load_sample(self.files[idx])
        u = torch.from_numpy(s.user_codes).long()
        a = torch.from_numpy(s.asst_codes).long()
        t = torch.from_numpy(s.text_ids).long()
        ta = torch.from_numpy(s.text_align).long()
        if self.pad_to is not None:
            u = _pad_codes(u, self.pad_to)
            a = _pad_codes(a, self.pad_to)
        return {
            "user_codes": u,
            "asst_codes": a,
            "text_ids": t,
            "text_align": ta,
            "aux": s.aux,
        }


def _pad_codes(codes: torch.Tensor, target_T: int, pad_id: int = 0) -> torch.Tensor:
    K, T = codes.shape
    if T >= target_T:
        return codes[:, :target_T]
    pad = torch.full((K, target_T - T), pad_id, dtype=codes.dtype)
    return torch.cat([codes, pad], dim=1)


# ---------------------------------------------------------------------------
# Collate
# ---------------------------------------------------------------------------
def dual_stream_collate(batch: list[dict]) -> dict[str, torch.Tensor]:
    """Pad-collate. All tensors padded to max T in batch.

    We build:
      - input_text_ids: [B, T_frames]      placeholder text per frame (mostly <a:silent>)
      - user_codes:     [B, K, T_frames]   Mimi codes for user
      - labels_asst_audio: [B, K, T_frames]  shifted asst codes (next-token)
      - labels_user_audio: [B, K, T_frames]  shifted user codes
      - labels_text:    [B, T_frames]       text monologue targets per frame
      - attention_mask: [B, T_frames]       1 where real, 0 where pad
    """
    Tmax = max(item["user_codes"].shape[1] for item in batch)
    K = batch[0]["user_codes"].shape[0]
    B = len(batch)

    user_codes = torch.zeros(B, K, Tmax, dtype=torch.long)
    asst_codes = torch.zeros(B, K, Tmax, dtype=torch.long)
    attn = torch.zeros(B, Tmax, dtype=torch.long)
    text_per_frame = torch.full((B, Tmax), -100, dtype=torch.long)

    for i, item in enumerate(batch):
        T = item["user_codes"].shape[1]
        user_codes[i, :, :T] = item["user_codes"]
        asst_codes[i, :, :T] = item["asst_codes"]
        attn[i, :T] = 1
        for tok_id, frame_idx in zip(item["text_ids"].tolist(), item["text_align"].tolist()):
            if 0 <= frame_idx < Tmax:
                text_per_frame[i, frame_idx] = tok_id

    return {
        "user_codes": user_codes,
        "asst_codes": asst_codes,
        "attention_mask": attn,
        "labels_text": text_per_frame,
        "labels_user_audio": user_codes.clone(),
        "labels_asst_audio": asst_codes.clone(),
    }


# ---------------------------------------------------------------------------
# Streaming iterable variant for big datasets
# ---------------------------------------------------------------------------
class StreamingDualStreamDataset(IterableDataset):
    def __init__(self, root: str | Path):
        self.root = Path(root)
        self.files = sorted(self.root.glob("*.npz"))

    def __iter__(self) -> Iterator[dict[str, torch.Tensor]]:
        for f in self.files:
            s = load_sample(f)
            yield {
                "user_codes": torch.from_numpy(s.user_codes).long(),
                "asst_codes": torch.from_numpy(s.asst_codes).long(),
                "text_ids": torch.from_numpy(s.text_ids).long(),
                "text_align": torch.from_numpy(s.text_align).long(),
                "aux": s.aux,
            }
