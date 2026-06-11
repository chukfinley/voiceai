"""Streaming engine tests with a tiny random backbone (no downloads, no Mimi).

    uv run pytest tests/test_streaming.py -v
"""
from __future__ import annotations

import os

import pytest
import torch

from voiceai.training.data.dual_stream import ACOUSTIC_BOS

TINY = os.environ.get("VOICEAI_TINY_BACKBONE", "hf-internal-testing/tiny-random-LlamaForCausalLM")


@pytest.fixture(scope="module")
def model():
    from voiceai.model.voiceai_lm import VoiceAIConfig, VoiceAILM

    cfg = VoiceAIConfig(
        backbone=TINY,
        freeze_backbone=True,
        dtype="float32",
        depth_dim=32,
        depth_layers=1,
    )
    return VoiceAILM(cfg).eval()


def _engine(model, **kw):
    from voiceai.inference.streaming import StreamingEngine

    return StreamingEngine(model, mimi=None, device="cpu", **kw)


def test_step_shapes_and_feedback(model):
    eng = _engine(model)
    K = model.cfg.num_codebooks
    user = torch.randint(0, model.cfg.codebook_size, (K,))
    out = eng.step(user)
    assert out.asst_codes.shape == (K,)
    assert isinstance(out.text_token, int)
    # feedback: next frame's own-stream input is what was just emitted
    assert (eng.prev_asst.view(-1) == out.asst_codes).all()
    assert eng.frames == 1
    out2 = eng.step(user)
    assert eng.frames == 2
    assert out2.asst_codes.shape == (K,)


def test_mute_forces_bos_silence(model):
    eng = _engine(model)
    K = model.cfg.num_codebooks
    user = torch.randint(0, model.cfg.codebook_size, (K,))
    eng.mute()
    out = eng.step(user)
    assert (out.asst_codes == ACOUSTIC_BOS).all()
    assert out.audio is None


def test_sliding_window_reprefill(model):
    eng = _engine(model, max_frames=8, window_frames=4)
    K = model.cfg.num_codebooks
    user = torch.randint(0, model.cfg.codebook_size, (K,))
    for _ in range(12):  # crosses max_frames → re-prefill must trigger
        out = eng.step(user)
        assert out.asst_codes.shape == (K,)
    # after re-prefill the cache restarts from window + new frames, < max+1
    assert eng.frames <= 8 + 1


def test_kv_cache_matches_full_forward(model):
    """Incremental (cached) hidden states must equal a full forward pass."""
    eng = _engine(model, max_frames=1000, window_frames=500)
    K = model.cfg.num_codebooks
    torch.manual_seed(0)
    frames = [torch.randint(0, model.cfg.codebook_size, (K,)) for _ in range(4)]

    embeds = []
    hiddens_inc = []
    for fr in frames:
        embeds.append(eng._frame_embed(fr.view(1, K, 1)))
        hiddens_inc.append(eng._forward(embeds[-1]))
        # freeze feedback so both paths see identical inputs
        eng.prev_asst = torch.full((1, K, 1), ACOUSTIC_BOS, dtype=torch.long)
        eng.prev_text = torch.tensor([[eng._silent_id]])

    with torch.no_grad():
        full = model.backbone(
            inputs_embeds=torch.cat(embeds, dim=1),
            output_hidden_states=True,
            return_dict=True,
        ).hidden_states[-1]
    for t, h in enumerate(hiddens_inc):
        torch.testing.assert_close(h, full[:, t], rtol=1e-4, atol=1e-4)
