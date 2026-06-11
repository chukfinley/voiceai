"""Unit tests for audio adapters. Run on CPU, no GPU needed.

    uv run pytest tests/test_audio_adapter.py -v
"""
from __future__ import annotations

import torch

from voiceai.model.audio_adapter import AudioAdapter, MimiDepthTransformer, MimiOutputHeads


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


def test_depth_transformer_loss_and_grad():
    d, K, V = 64, 8, 16
    head = MimiDepthTransformer(d_model=d, num_codebooks=K, codebook_size=V, depth_dim=32, num_layers=1, num_heads=2)
    h = torch.randn(2, 10, d, requires_grad=True)
    targets = torch.randint(0, V, (2, K, 10))
    targets[1, :, 7:] = -100  # masked tail
    loss = head.loss(h, targets)
    assert loss.dim() == 0 and torch.isfinite(loss)
    loss.backward()
    assert h.grad is not None and h.grad.abs().sum() > 0


def test_depth_transformer_fully_masked_frame():
    d, K, V = 32, 4, 8
    head = MimiDepthTransformer(d_model=d, num_codebooks=K, codebook_size=V, depth_dim=16, num_layers=1, num_heads=2)
    h = torch.randn(1, 5, d)
    targets = torch.full((1, K, 5), -100, dtype=torch.long)
    loss = head.loss(h, targets)
    assert float(loss) == 0.0


def test_depth_transformer_sample():
    d, K, V = 32, 4, 8
    head = MimiDepthTransformer(d_model=d, num_codebooks=K, codebook_size=V, depth_dim=16, num_layers=1, num_heads=2)
    h = torch.randn(3, d)
    codes = head.sample(h, temperature=0.0)  # greedy
    assert codes.shape == (3, K)
    assert codes.min() >= 0 and codes.max() <= V  # V = the extra BOS/EOS class


def test_depth_transformer_causality():
    """Logits for codebook k must not depend on codebooks >= k."""
    d, K, V = 32, 4, 8
    head = MimiDepthTransformer(d_model=d, num_codebooks=K, codebook_size=V, depth_dim=16, num_layers=1, num_heads=2)
    head.eval()
    h = torch.randn(1, d)
    codes_a = torch.randint(0, V, (1, K))
    codes_b = codes_a.clone()
    codes_b[0, 2] = (codes_b[0, 2] + 1) % V  # change codebook 2
    with torch.no_grad():
        la = head._logits(h, codes_a)
        lb = head._logits(h, codes_b)
    # positions 0..2 see only codebooks < 2 (embedding of code k feeds pos k+1)
    torch.testing.assert_close(la[:, :3], lb[:, :3])
    assert not torch.allclose(la[:, 3], lb[:, 3])


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
