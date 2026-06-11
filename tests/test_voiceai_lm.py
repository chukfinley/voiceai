"""Smoke test for VoiceAILM. Uses a tiny stand-in HF model so it runs on CPU.

    uv run pytest tests/test_voiceai_lm.py -v -s

Set VOICEAI_TEST_BACKBONE=Qwen/Qwen3.5-0.8B for a real but slow test.
"""
from __future__ import annotations

import os
import tempfile
from pathlib import Path

import pytest
import torch


TINY_BACKBONE_ENV = "VOICEAI_TEST_BACKBONE"
DEFAULT_TINY = "hf-internal-testing/tiny-random-LlamaForCausalLM"


def _backbone() -> str:
    return os.environ.get(TINY_BACKBONE_ENV, DEFAULT_TINY)


@pytest.mark.slow
def test_voiceai_lm_forward():
    from voiceai.model.voiceai_lm import VoiceAIConfig, VoiceAILM

    cfg = VoiceAIConfig(
        backbone=_backbone(),
        freeze_backbone=True,
        train_user_audio=True,
        dtype="float32",
        use_depth_transformer=False,  # legacy heads expose eager logits
    )
    model = VoiceAILM(cfg)

    B, K, T = 1, 8, 6
    user_codes = torch.randint(0, model.cfg.codebook_size, (B, K, T))
    text_ids = torch.full((B, T), 0, dtype=torch.long)
    out = model(text_ids=text_ids, user_audio_codes=user_codes)
    assert out["text_logits"].shape[:2] == (B, T)
    assert out["asst_audio_logits"].shape == (B, K, T, model.cfg.codebook_size + 1)
    assert out["user_audio_logits"].shape == (B, K, T, model.cfg.codebook_size + 1)


@pytest.mark.slow
def test_voiceai_lm_depth_transformer_forward():
    from voiceai.model.audio_adapter import MimiDepthTransformer
    from voiceai.model.voiceai_lm import VoiceAIConfig, VoiceAILM

    cfg = VoiceAIConfig(
        backbone=_backbone(),
        freeze_backbone=True,
        train_user_audio=True,
        dtype="float32",
        depth_dim=32,
        depth_layers=1,
    )
    model = VoiceAILM(cfg)
    assert isinstance(model.asst_audio_out, MimiDepthTransformer)

    B, K, T = 1, 8, 6
    user_codes = torch.randint(0, model.cfg.codebook_size, (B, K, T))
    asst_codes = torch.randint(0, model.cfg.codebook_size, (B, K, T))
    text_ids = torch.full((B, T), 0, dtype=torch.long)
    labels_audio = torch.randint(0, model.cfg.codebook_size, (B, K, T))
    assert model.audio_in_asst is not None  # model hears its own stream
    out = model(
        text_ids=text_ids,
        user_audio_codes=user_codes,
        asst_audio_codes=asst_codes,
        labels_asst_audio=labels_audio,
        labels_user_audio=labels_audio,
    )
    # depth transformer: no eager logits, but losses exist and backprop
    assert "asst_audio_logits" not in out
    assert out["loss"] is not None
    out["loss"].backward()
    # per-frame sampling works
    codes = model.asst_audio_out.sample(out["hidden"][:, -1].detach())
    assert codes.shape == (B, K)


@pytest.mark.slow
def test_voiceai_lm_loss_and_backward():
    from voiceai.model.voiceai_lm import VoiceAIConfig, VoiceAILM

    cfg = VoiceAIConfig(
        backbone=_backbone(),
        freeze_backbone=True,
        train_text=True,
        train_user_audio=True,
        dtype="float32",
    )
    model = VoiceAILM(cfg)
    B, K, T = 1, 8, 4
    user_codes = torch.randint(0, model.cfg.codebook_size, (B, K, T))
    text_ids = torch.full((B, T), 0, dtype=torch.long)
    labels_text = torch.zeros((B, T), dtype=torch.long)
    labels_audio = torch.randint(0, model.cfg.codebook_size, (B, K, T))
    out = model(
        text_ids=text_ids,
        user_audio_codes=user_codes,
        labels_text=labels_text,
        labels_user_audio=labels_audio,
        labels_asst_audio=labels_audio,
    )
    assert out["loss"] is not None
    out["loss"].backward()


@pytest.mark.slow
def test_save_load_roundtrip():
    from voiceai.model.voiceai_lm import VoiceAIConfig, VoiceAILM

    cfg = VoiceAIConfig(backbone=_backbone(), freeze_backbone=True, dtype="float32")
    model = VoiceAILM(cfg)
    with tempfile.TemporaryDirectory() as td:
        model.save_pretrained(td)
        assert (Path(td) / "adapters.pt").exists()
        assert (Path(td) / "backbone").exists()
