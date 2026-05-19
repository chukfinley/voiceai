"""Audio adapter modules: Mimi tokens <-> Qwen embedding space.

Two trainable modules:

  AudioAdapter:
      input  = Mimi codes [B, num_codebooks, T]  (T = frame index)
      output = embeddings [B, T, d_model]
      Each codebook has its own embedding table; per-frame embeds are summed
      (Moshi-style residual aggregation).

  MimiOutputHeads:
      input  = hidden states [B, T, d_model]
      output = logits [B, num_codebooks, T, codebook_size]
      Predicts each codebook's next token independently (RQ-Transformer
      style; the depth-transformer is fused into linear heads for speed).

Both are tiny (~50M total) — these are what we actually train in Stage 1.
"""
from __future__ import annotations

import torch
import torch.nn as nn

from .mimi_utils import MIMI_CARD, MIMI_NUM_CODEBOOKS


class AudioAdapter(nn.Module):
    """Project Mimi codes into the backbone's embedding space."""

    def __init__(
        self,
        d_model: int,
        num_codebooks: int = MIMI_NUM_CODEBOOKS,
        codebook_size: int = MIMI_CARD,
    ):
        super().__init__()
        self.num_codebooks = num_codebooks
        self.codebook_size = codebook_size
        self.embeds = nn.ModuleList(
            [nn.Embedding(codebook_size + 1, d_model) for _ in range(num_codebooks)]
        )
        self.proj = nn.Linear(d_model, d_model, bias=False)
        nn.init.normal_(self.proj.weight, std=0.02)

    def forward(self, codes: torch.Tensor) -> torch.Tensor:
        """codes: [B, num_codebooks, T] long -> embeds: [B, T, d_model]"""
        if codes.dim() != 3:
            raise ValueError(f"expected [B, K, T], got {codes.shape}")
        B, K, T = codes.shape
        if K != self.num_codebooks:
            raise ValueError(f"got {K} codebooks, expected {self.num_codebooks}")
        out = torch.zeros(B, T, self.embeds[0].embedding_dim, device=codes.device, dtype=self.embeds[0].weight.dtype)
        for i, emb in enumerate(self.embeds):
            out = out + emb(codes[:, i, :])
        return self.proj(out)


class MimiOutputHeads(nn.Module):
    """Predict next Mimi codes from backbone hidden states.

    One linear head per codebook. Each head outputs logits over
    `codebook_size + 1` classes (last class = EOS / silent).
    """

    def __init__(
        self,
        d_model: int,
        num_codebooks: int = MIMI_NUM_CODEBOOKS,
        codebook_size: int = MIMI_CARD,
    ):
        super().__init__()
        self.num_codebooks = num_codebooks
        self.codebook_size = codebook_size
        self.heads = nn.ModuleList(
            [nn.Linear(d_model, codebook_size + 1) for _ in range(num_codebooks)]
        )
        for h in self.heads:
            nn.init.normal_(h.weight, std=0.02)
            nn.init.zeros_(h.bias)

    def forward(self, hidden: torch.Tensor) -> torch.Tensor:
        """hidden: [B, T, d_model] -> logits: [B, K, T, V]"""
        head_dtype = self.heads[0].weight.dtype
        if hidden.dtype != head_dtype:
            hidden = hidden.to(head_dtype)
        return torch.stack([h(hidden) for h in self.heads], dim=1)

    def loss(
        self,
        hidden: torch.Tensor,
        target_codes: torch.Tensor,
        ignore_index: int = -100,
        reduction: str = "mean",
    ) -> torch.Tensor:
        """Cross-entropy over all codebooks summed.

        target_codes: [B, K, T] long, ignore_index for masked positions.
        """
        logits = self.forward(hidden)  # [B, K, T, V]
        B, K, T, V = logits.shape
        return nn.functional.cross_entropy(
            logits.reshape(-1, V).float(),
            target_codes.reshape(-1),
            ignore_index=ignore_index,
            reduction=reduction,
        )
