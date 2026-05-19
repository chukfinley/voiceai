"""Mimi codec helpers.

Mimi (Kyutai) is a streaming neural audio codec.
- 24 kHz audio in
- 12.5 Hz token rate (8 RVQ codebooks per frame)
- 1024 codes per codebook
- Fully causal — no future context needed for encoding
- CC-BY-4.0 weights

We load it via the official `moshi` package which provides
`moshi.models.loaders.get_mimi`.
"""
from __future__ import annotations

from functools import lru_cache
from typing import TYPE_CHECKING

import torch

if TYPE_CHECKING:
    from moshi.models.compression import MimiModel


MIMI_HF_REPO = "kyutai/moshiko-pytorch-bf16"
MIMI_WEIGHTS_FILE = "tokenizer-e351c8d8-checkpoint125.safetensors"
MIMI_FRAME_HZ = 12.5
MIMI_NUM_CODEBOOKS = 8
MIMI_CARD = 2048  # vocab per codebook (real Mimi is 2048)
MIMI_SAMPLE_RATE = 24000


@lru_cache(maxsize=1)
def load_mimi(device: str = "cuda", dtype: torch.dtype | None = None):
    """Load and cache the Mimi codec.

    Returns a `MimiModel` from the `moshi` package, ready for streaming
    encode/decode. The model is frozen — we never train Mimi itself.

    Auto-picks float32 on CPU (bfloat16 on CPU is slow/buggy in older torch).
    """
    from huggingface_hub import hf_hub_download
    from moshi.models.loaders import get_mimi

    if dtype is None:
        dtype = torch.float32 if device == "cpu" else torch.bfloat16

    ckpt_path = hf_hub_download(MIMI_HF_REPO, MIMI_WEIGHTS_FILE)

    mimi = get_mimi(ckpt_path, device=device)
    mimi = mimi.to(dtype=dtype)
    mimi.eval()
    for p in mimi.parameters():
        p.requires_grad_(False)
    return mimi


@torch.no_grad()
def mimi_encode(mimi, audio: torch.Tensor) -> torch.Tensor:
    """Encode PCM audio → discrete Mimi codes.

    Args:
        mimi: loaded Mimi model
        audio: shape [batch, 1, samples] @ 24kHz, float

    Returns:
        codes: shape [batch, num_codebooks=8, frames]
    """
    if audio.dim() == 2:
        audio = audio.unsqueeze(1)
    return mimi.encode(audio)


@torch.no_grad()
def mimi_decode(mimi, codes: torch.Tensor) -> torch.Tensor:
    """Decode Mimi codes → PCM audio.

    Args:
        codes: shape [batch, num_codebooks=8, frames]

    Returns:
        audio: shape [batch, 1, samples] @ 24kHz
    """
    return mimi.decode(codes)


def resample_to_mimi(audio: torch.Tensor, src_sr: int) -> torch.Tensor:
    """Resample PCM to Mimi's required 24kHz."""
    if src_sr == MIMI_SAMPLE_RATE:
        return audio
    import torchaudio.functional as F

    return F.resample(audio, src_sr, MIMI_SAMPLE_RATE)
