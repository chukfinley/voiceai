"""Tests for dual-stream data format and collate.

    uv run pytest tests/test_dual_stream_data.py -v
"""
from __future__ import annotations

import tempfile
from pathlib import Path

import numpy as np

import torch

from voiceai.training.data.dual_stream import (
    ACOUSTIC_BOS,
    DualStreamDataset,
    DualStreamSample,
    apply_acoustic_delay,
    dual_stream_collate,
    load_sample,
    remove_acoustic_delay,
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
        # token aligned to frame f is predicted from hidden state at f-1
        for tok_id, frame in zip([1, 2, 3, 4], [2, 5, 8, 11]):
            assert labels[frame - 1].item() == tok_id


def test_acoustic_delay_roundtrip():
    codes = torch.randint(0, 2048, (8, 20))
    delayed = apply_acoustic_delay(codes, delay=2)
    # semantic codebook untouched
    assert (delayed[0] == codes[0]).all()
    # acoustic books shifted right, front filled with BOS
    assert (delayed[1:, :2] == ACOUSTIC_BOS).all()
    assert (delayed[1:, 2:] == codes[1:, :-2]).all()
    # roundtrip restores alignment (minus the truncated tail)
    restored = remove_acoustic_delay(delayed, delay=2)
    assert (restored[0] == codes[0, :18]).all()
    assert (restored[1:] == codes[1:, :18]).all()
    # delay=0 is a no-op
    assert (apply_acoustic_delay(codes, 0) == codes).all()


def test_collate_with_acoustic_delay():
    with tempfile.TemporaryDirectory() as td:
        save_sample(_make_sample("s0", T=12), Path(td))
        ds = DualStreamDataset(td)
        item = ds[0]
        out = dual_stream_collate([item], acoustic_delay=1)
        delayed = apply_acoustic_delay(item["user_codes"], 1)
        assert (out["user_codes"][0] == delayed).all()
        # labels are the delayed stream shifted one frame (next-frame target)
        assert (out["labels_user_audio"][0, :, :11] == delayed[:, 1:]).all()


def test_collate_labels_are_next_frame():
    with tempfile.TemporaryDirectory() as td:
        save_sample(_make_sample("s0", T=10), Path(td))
        save_sample(_make_sample("s1", T=15), Path(td))
        ds = DualStreamDataset(td)
        batch = [ds[0], ds[1]]
        out = dual_stream_collate(batch)
        for i, item in enumerate(batch):
            T = item["user_codes"].shape[1]
            # label at t is the input code at t+1 (next-frame prediction)
            assert (out["labels_user_audio"][i, :, : T - 1] == item["user_codes"][:, 1:]).all()
            assert (out["labels_asst_audio"][i, :, : T - 1] == item["asst_codes"][:, 1:]).all()
            # last real frame and all padding are masked
            assert (out["labels_user_audio"][i, :, T - 1 :] == -100).all()
            assert (out["labels_asst_audio"][i, :, T - 1 :] == -100).all()
