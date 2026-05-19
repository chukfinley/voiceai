"""Unit tests for audio adapters. Run on CPU, no GPU needed.

    uv run pytest tests/test_audio_adapter.py -v
"""
from __future__ import annotations

import torch

from voiceai.model.audio_adapter import AudioAdapter, MimiOutputHeads


def test_audio_adapter_shape():
    d_model = 256
    K = 8
    V = 100
    adapter = AudioAdapter(d_model=d_model, num_codebooks=K, codebook_size=V)
    codes = torch.randint(0, V, (2, K, 12))
    out = adapter(codes)
    assert out.shape == (2, 12, d_model)
    assert out.dtype == torch.float32


def test_audio_adapter_rejects_bad_input():
    adapter = AudioAdapter(d_model=64, num_codebooks=8)
    try:
        adapter(torch.zeros(2, 12, dtype=torch.long))
        raise AssertionError("should have raised")
    except ValueError:
        pass

    try:
        codes = torch.randint(0, 100, (2, 4, 12))
        adapter(codes)
        raise AssertionError("should have raised on K mismatch")
    except ValueError:
        pass


def test_mimi_output_heads_logits_shape():
    d = 128
    K = 8
    V = 256
    head = MimiOutputHeads(d_model=d, num_codebooks=K, codebook_size=V)
    h = torch.randn(2, 20, d)
    logits = head(h)
    assert logits.shape == (2, K, 20, V + 1)


def test_mimi_output_heads_loss():
    d = 64
    K = 8
    V = 16
    head = MimiOutputHeads(d_model=d, num_codebooks=K, codebook_size=V)
    h = torch.randn(2, 10, d)
    targets = torch.randint(0, V, (2, K, 10))
    loss = head.loss(h, targets)
    assert loss.dim() == 0
    assert torch.isfinite(loss)


def test_audio_adapter_grad_flow():
    adapter = AudioAdapter(d_model=64, num_codebooks=8, codebook_size=64)
    head = MimiOutputHeads(d_model=64, num_codebooks=8, codebook_size=64)
    codes_in = torch.randint(0, 64, (1, 8, 12))
    codes_out = torch.randint(0, 64, (1, 8, 12))
    embeds = adapter(codes_in)
    loss = head.loss(embeds, codes_out)
    loss.backward()
    has_grad = sum(int(p.grad is not None and p.grad.abs().sum() > 0) for p in adapter.parameters())
    assert has_grad > 0
