"""Tests for dual-stream data format and collate.

    uv run pytest tests/test_dual_stream_data.py -v
"""
from __future__ import annotations

import tempfile
from pathlib import Path

import numpy as np

from voiceai.training.data.dual_stream import (
    DualStreamDataset,
    DualStreamSample,
    dual_stream_collate,
    load_sample,
    save_sample,
)


def _make_sample(sid: str, T: int, K: int = 8) -> DualStreamSample:
    return DualStreamSample(
        user_codes=np.random.randint(0, 2048, size=(K, T)),
        asst_codes=np.random.randint(0, 2048, size=(K, T)),
        text_ids=np.array([1, 2, 3, 4], dtype=np.int32),
        text_align=np.array([2, 5, 8, 11], dtype=np.int32),
        aux={"category": "test", "concurrent_commentary": True},
        sample_id=sid,
        duration_s=float(T) / 12.5,
    )


def test_save_load_roundtrip():
    with tempfile.TemporaryDirectory() as td:
        s = _make_sample("s0", T=50)
        save_sample(s, Path(td))
        loaded = load_sample(Path(td) / "s0.npz")
        np.testing.assert_array_equal(loaded.user_codes, s.user_codes)
        np.testing.assert_array_equal(loaded.asst_codes, s.asst_codes)
        np.testing.assert_array_equal(loaded.text_ids, s.text_ids)
        assert loaded.aux == s.aux
        assert loaded.sample_id == "s0"


def test_dataset_pad():
    with tempfile.TemporaryDirectory() as td:
        for i in range(3):
            save_sample(_make_sample(f"s{i}", T=20 + i * 5), Path(td))
        ds = DualStreamDataset(td, pad_to_frames=32)
        assert len(ds) == 3
        for i in range(3):
            x = ds[i]
            assert x["user_codes"].shape == (8, 32)
            assert x["asst_codes"].shape == (8, 32)


def test_collate_shapes():
    with tempfile.TemporaryDirectory() as td:
        for i in range(4):
            save_sample(_make_sample(f"s{i}", T=20 + i * 3), Path(td))
        ds = DualStreamDataset(td)
        batch = [ds[i] for i in range(4)]
        out = dual_stream_collate(batch)
        B = 4
        Tmax = max(b["user_codes"].shape[1] for b in batch)
        assert out["user_codes"].shape == (B, 8, Tmax)
        assert out["asst_codes"].shape == (B, 8, Tmax)
        assert out["attention_mask"].shape == (B, Tmax)
        assert out["labels_text"].shape == (B, Tmax)


def test_text_alignment_in_collate():
    with tempfile.TemporaryDirectory() as td:
        save_sample(_make_sample("s0", T=20), Path(td))
        ds = DualStreamDataset(td)
        batch = dual_stream_collate([ds[0]])
        labels = batch["labels_text"][0]
        for tok_id, frame in zip([1, 2, 3, 4], [2, 5, 8, 11]):
            assert labels[frame].item() == tok_id
