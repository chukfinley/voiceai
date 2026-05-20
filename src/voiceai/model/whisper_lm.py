"""Whisper-encoder + Qwen-LLM bridge for ASR.

Skips Mimi entirely. Audio → Whisper-encoder (frozen, pretrained, 680k h) →
small MLP projector (TRAINED) → Qwen embed space → Qwen LLM (frozen) → text.

Prefix-LM ASR: concat audio embeds + text embeds, supervise text logits at
positions [T_audio-1 : T_audio-1+T_text).
"""
from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path

import torch
import torch.nn as nn
import torch.nn.functional as F


@dataclass
class WhisperLMConfig:
    whisper_id: str = "openai/whisper-small"
    llm_id: str = "Qwen/Qwen3-1.7B"
    dtype: str = "bfloat16"


class WhisperLM(nn.Module):
    def __init__(self, cfg: WhisperLMConfig):
        super().__init__()
        self.cfg = cfg
        dtype = getattr(torch, cfg.dtype)

        from transformers import (
            AutoModelForCausalLM,
            AutoTokenizer,
            WhisperFeatureExtractor,
            WhisperModel,
        )

        whisper = WhisperModel.from_pretrained(cfg.whisper_id, dtype=dtype)
        self.audio_encoder = whisper.encoder
        for p in self.audio_encoder.parameters():
            p.requires_grad_(False)

        self.feature_extractor = WhisperFeatureExtractor.from_pretrained(cfg.whisper_id)

        self.backbone = AutoModelForCausalLM.from_pretrained(cfg.llm_id, dtype=dtype, trust_remote_code=True)
        for p in self.backbone.parameters():
            p.requires_grad_(False)

        self.tokenizer = AutoTokenizer.from_pretrained(cfg.llm_id, trust_remote_code=True)
        if self.tokenizer.pad_token_id is None:
            self.tokenizer.pad_token = self.tokenizer.eos_token

        whisper_dim = self.audio_encoder.config.d_model
        llm_dim = self.backbone.config.hidden_size

        self.bridge = nn.Sequential(
            nn.Linear(whisper_dim, llm_dim),
            nn.GELU(),
            nn.Linear(llm_dim, llm_dim),
        ).to(dtype=dtype)

    @torch.no_grad()
    def _whisper_encode(self, audio_features: torch.Tensor) -> torch.Tensor:
        """audio_features: [B, 80, 3000] mel — direct output of WhisperFeatureExtractor.
        Returns [B, T_enc, whisper_dim]. T_enc is typically 1500 for 30s of audio.
        """
        return self.audio_encoder(audio_features).last_hidden_state

    def audio_to_embeds(self, audio_features: torch.Tensor) -> torch.Tensor:
        enc = self._whisper_encode(audio_features)
        return self.bridge(enc)

    def forward(
        self,
        audio_features: torch.Tensor,
        text_ids: torch.Tensor,
        text_attn: torch.Tensor | None = None,
    ) -> dict:
        """Prefix-LM ASR forward.

        audio_features: [B, 80, 3000] mel spectrograms.
        text_ids: [B, T_text] tokenized transcript.
        text_attn: [B, T_text] 1=real, 0=pad. If None inferred from pad_token_id.

        Returns: {'logits': [B, T_text, V], 'loss': scalar if labels provided}
        """
        device = audio_features.device
        pad_id = self.tokenizer.pad_token_id
        if text_attn is None:
            text_attn = (text_ids != pad_id).long()

        audio_e = self.audio_to_embeds(audio_features)  # [B, T_a, D]
        B, T_a, D = audio_e.shape
        T_t = text_ids.shape[1]
        bb_dtype = self.backbone.get_input_embeddings().weight.dtype

        if T_t > 1:
            text_in = self.backbone.get_input_embeddings()(text_ids[:, :-1])
            embeds = torch.cat([audio_e.to(bb_dtype), text_in], dim=1)
            audio_mask = torch.ones(B, T_a, device=device, dtype=torch.long)
            attn = torch.cat([audio_mask, text_attn[:, :-1]], dim=1)
        else:
            embeds = audio_e.to(bb_dtype)
            attn = torch.ones(B, T_a, device=device, dtype=torch.long)

        bb_out = self.backbone(
            inputs_embeds=embeds,
            attention_mask=attn,
            output_hidden_states=True,
            return_dict=True,
        )
        hidden = bb_out.hidden_states[-1]
        text_hidden = hidden[:, T_a - 1:T_a - 1 + T_t, :]
        text_logits = self.backbone.lm_head(text_hidden)

        target = text_ids.clone()
        target[text_attn == 0] = -100
        loss = F.cross_entropy(
            text_logits.reshape(-1, text_logits.size(-1)).float(),
            target.reshape(-1),
            ignore_index=-100,
        )
        return {"logits": text_logits, "loss": loss}

    @torch.no_grad()
    def generate(
        self,
        audio_features: torch.Tensor,
        max_new_tokens: int = 100,
    ) -> list[int]:
        device = audio_features.device
        eos_id = self.tokenizer.eos_token_id
        bb_dtype = self.backbone.get_input_embeddings().weight.dtype

        audio_e = self.audio_to_embeds(audio_features).to(bb_dtype)
        embeds = audio_e
        out: list[int] = []
        for _ in range(max_new_tokens):
            T = embeds.shape[1]
            attn = torch.ones(1, T, device=device, dtype=torch.long)
            bb_out = self.backbone(
                inputs_embeds=embeds,
                attention_mask=attn,
                output_hidden_states=True,
                return_dict=True,
            )
            last = bb_out.hidden_states[-1][:, -1:, :]
            logits = self.backbone.lm_head(last)
            nxt = int(logits.argmax(dim=-1).item())
            if eos_id is not None and nxt == eos_id:
                break
            out.append(nxt)
            next_e = self.backbone.get_input_embeddings()(torch.tensor([[nxt]], device=device))
            embeds = torch.cat([embeds, next_e.to(bb_dtype)], dim=1)
        return out

    def save_pretrained(self, path: str | Path) -> None:
        path = Path(path)
        path.mkdir(parents=True, exist_ok=True)
        torch.save(
            {"bridge": self.bridge.state_dict(), "cfg": self.cfg.__dict__},
            path / "bridge.pt",
        )

    @classmethod
    def from_pretrained(cls, path: str | Path) -> "WhisperLM":
        path = Path(path)
        data = torch.load(path / "bridge.pt", map_location="cpu")
        cfg = WhisperLMConfig(**data["cfg"])
        m = cls(cfg)
        m.bridge.load_state_dict(data["bridge"])
        return m

    def trainable_param_count(self) -> int:
        return sum(p.numel() for p in self.parameters() if p.requires_grad)
