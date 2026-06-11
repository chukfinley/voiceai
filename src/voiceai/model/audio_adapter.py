"""Audio adapter modules: Mimi tokens <-> Qwen embedding space.

Trainable modules:

  AudioAdapter:
      input  = Mimi codes [B, num_codebooks, T]  (T = frame index)
      output = embeddings [B, T, d_model]
      Each codebook has its own embedding table; per-frame embeds are summed
      (Moshi-style residual aggregation).

  MimiOutputHeads (legacy):
      input  = hidden states [B, T, d_model]
      output = logits [B, num_codebooks, T, codebook_size]
      Predicts each codebook independently from the same hidden state.
      Fast, but residual codebooks are NOT independent — independent
      sampling produces inconsistent code combinations (audio artifacts).

  MimiDepthTransformer (default):
      Moshi/RQ-Transformer-style: per frame, a small causal transformer
      runs across the K codebook positions, so codebook k is predicted
      conditioned on the backbone hidden state AND codebooks < k of the
      same frame. Fixes the independence problem; this is what makes
      sampled audio coherent.

Both are tiny — these are what we actually train in Stage 1.
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


class MimiDepthTransformer(nn.Module):
    """Autoregressive depth transformer over codebooks (Moshi/RQ-Transformer).

    Per frame: position 0 sees only the (projected) backbone hidden state and
    predicts codebook 0; position k additionally sees the embedding of
    codebook k-1, so the residual structure of Mimi codes is modeled instead
    of ignored.

    Training uses teacher forcing with the target codes (`loss`); inference
    samples codebook-by-codebook per frame (`sample`).
    """

    def __init__(
        self,
        d_model: int,
        num_codebooks: int = MIMI_NUM_CODEBOOKS,
        codebook_size: int = MIMI_CARD,
        depth_dim: int = 512,
        num_layers: int = 4,
        num_heads: int = 8,
    ):
        super().__init__()
        self.num_codebooks = num_codebooks
        self.codebook_size = codebook_size
        self.depth_dim = depth_dim

        self.hidden_proj = nn.Linear(d_model, depth_dim)
        # embedding of codebook k feeds position k+1 (teacher forcing / sampled)
        self.code_embeds = nn.ModuleList(
            [nn.Embedding(codebook_size + 1, depth_dim) for _ in range(num_codebooks - 1)]
        )
        self.pos_embed = nn.Parameter(torch.zeros(num_codebooks, depth_dim))
        layer = nn.TransformerEncoderLayer(
            d_model=depth_dim,
            nhead=num_heads,
            dim_feedforward=depth_dim * 4,
            batch_first=True,
            norm_first=True,
            activation="gelu",
        )
        self.transformer = nn.TransformerEncoder(layer, num_layers=num_layers)
        self.heads = nn.ModuleList(
            [nn.Linear(depth_dim, codebook_size + 1) for _ in range(num_codebooks)]
        )
        nn.init.normal_(self.pos_embed, std=0.02)
        for h in self.heads:
            nn.init.normal_(h.weight, std=0.02)
            nn.init.zeros_(h.bias)

    def _causal_mask(self, device: torch.device) -> torch.Tensor:
        K = self.num_codebooks
        return torch.triu(torch.full((K, K), float("-inf"), device=device), diagonal=1)

    def _logits(self, hidden_flat: torch.Tensor, codes_flat: torch.Tensor) -> torch.Tensor:
        """hidden_flat: [N, d_model], codes_flat: [N, K] (teacher codes, >=0).
        Returns logits [N, K, V]."""
        N = hidden_flat.shape[0]
        # backbone hidden may be bf16 while this module is fp32 (or vice versa);
        # match the projection's weight dtype so the matmul doesn't error.
        hidden_flat = hidden_flat.to(self.hidden_proj.weight.dtype)
        h = self.hidden_proj(hidden_flat)  # [N, depth]
        seq = [h]
        for k in range(self.num_codebooks - 1):
            seq.append(h + self.code_embeds[k](codes_flat[:, k]))
        x = torch.stack(seq, dim=1) + self.pos_embed[None, :, :]  # [N, K, depth]
        x = self.transformer(x, mask=self._causal_mask(x.device))
        return torch.stack([self.heads[k](x[:, k]) for k in range(self.num_codebooks)], dim=1)

    def loss(
        self,
        hidden: torch.Tensor,
        target_codes: torch.Tensor,
        ignore_index: int = -100,
        reduction: str = "mean",
    ) -> torch.Tensor:
        """Teacher-forced CE. hidden: [B, T, d_model], target_codes: [B, K, T]."""
        B, T, D = hidden.shape
        K = self.num_codebooks
        targets = target_codes.permute(0, 2, 1).reshape(B * T, K)  # [N, K]
        # frames that are fully masked contribute nothing; drop them up front
        valid = (targets != ignore_index).any(dim=1)
        if not bool(valid.any()):
            return hidden.sum() * 0.0
        targets = targets[valid]
        hidden_flat = hidden.reshape(B * T, D)[valid]
        teacher = targets.clamp(min=0)  # masked positions: dummy embed, loss ignored
        logits = self._logits(hidden_flat, teacher)  # [N, K, V]
        return nn.functional.cross_entropy(
            logits.reshape(-1, logits.shape[-1]).float(),
            targets.reshape(-1),
            ignore_index=ignore_index,
            reduction=reduction,
        )

    @torch.no_grad()
    def sample(
        self,
        hidden: torch.Tensor,
        temperature: float = 0.8,
        top_k: int = 64,
    ) -> torch.Tensor:
        """Sample one frame of codes. hidden: [B, d_model] → codes [B, K]."""
        B = hidden.shape[0]
        codes = torch.zeros(B, self.num_codebooks, dtype=torch.long, device=hidden.device)
        for k in range(self.num_codebooks):
            logits = self._logits(hidden, codes)[:, k]  # [B, V]
            if temperature <= 0:
                codes[:, k] = logits.argmax(dim=-1)
                continue
            logits = logits / temperature
            if top_k > 0:
                kth = torch.topk(logits, top_k, dim=-1).values[:, -1:]
                logits = logits.masked_fill(logits < kth, float("-inf"))
            probs = torch.softmax(logits, dim=-1)
            codes[:, k] = torch.multinomial(probs, 1).squeeze(-1)
        return codes
