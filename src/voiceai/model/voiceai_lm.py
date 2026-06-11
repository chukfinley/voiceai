"""VoiceAI dual-stream language model.

Wraps a Qwen3.5-0.8B (or compatible HF causal LM) backbone with:
  - AudioAdapter for input audio tokens
  - MimiOutputHeads for assistant audio output
  - Dual-stream sequence layout: per-frame interleave [user_audio, asst_text?, asst_audio]
  - Text-token monologue head (uses the backbone's existing lm_head)
  - Special tokens for time/visual/barge/background-query

Forward pass returns:
  - text_logits: [B, T, V_text]
  - asst_audio_logits: [B, K, T, V_audio]
  - (optionally) user_audio_logits when supervising both streams

Training stages:
  Stage 1: only AudioAdapter + MimiOutputHeads trained, backbone frozen.
  Stage 2: + LoRA on backbone for dual-stream conversational behavior.
  Stage 3: + LoRA fine-tune on capability dataset (time, visual, etc.)
"""
from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path

import torch
import torch.nn as nn
from transformers import AutoModelForCausalLM, AutoTokenizer

from .audio_adapter import AudioAdapter, MimiDepthTransformer, MimiOutputHeads
from .mimi_utils import MIMI_CARD, MIMI_NUM_CODEBOOKS


# Emotion vocabulary for the inner monologue. Used in BOTH directions:
#   recognized user emotion:  "<u:emo> angry </u:emo>" style tags in the text stream
#   intended assistant emotion: "<a:emo> warm </a:emo>" before speaking
# Discrete tags keep it trainable from labeled data (MELD/CREMA-D/ESD) without
# a separate conditioning pathway; the audio heads learn the acoustic side.
EMOTIONS = [
    "neutral", "happy", "sad", "angry", "surprised", "fearful",
    "disgusted", "stressed", "calm", "excited", "whispering", "shouting",
    "laughing", "crying", "sarcastic",
]

SPECIAL_TOKENS = [
    "<silent>",
    "<u:silent>",
    "<a:silent>",
    "<barge>",
    "<thinking>",
    "<ack>",
    "<background_query>",
    "</background_query>",
    "<bg_result>",
    "</bg_result>",
    "<visual_event>",
    "</visual_event>",
    "<wait>",
    "</wait>",
    "<frame>",
    "</frame>",
    "<audio>",
    "</audio>",
    "<u_stream>",
    "</u_stream>",
    "<a_stream>",
    "</a_stream>",
    "<u:emo>",
    "</u:emo>",
    "<a:emo>",
    "</a:emo>",
    "<speaker>",
    "</speaker>",
    # mode switching in the monologue, e.g. "<task> interpret de->es </task>"
    # (live interpreting inverts barge-in semantics: user talking = input,
    # not interruption)
    "<task>",
    "</task>",
] + [f"<emo:{e}>" for e in EMOTIONS]


@dataclass
class VoiceAIConfig:
    backbone: str = "Qwen/Qwen3.5-0.8B"
    num_codebooks: int = MIMI_NUM_CODEBOOKS
    codebook_size: int = MIMI_CARD
    train_user_audio: bool = True
    train_asst_audio: bool = True
    train_text: bool = True
    user_audio_loss_weight: float = 0.5
    asst_audio_loss_weight: float = 1.0
    text_loss_weight: float = 1.0
    freeze_backbone: bool = False
    dtype: str = "bfloat16"
    load_in_4bit: bool = False
    # Depth transformer (Moshi-style) instead of independent linear heads.
    # Old checkpoints (trained with linear heads) load with this set to False.
    use_depth_transformer: bool = True
    depth_dim: int = 512
    depth_layers: int = 4
    # Feed the assistant's own audio stream back as input (Moshi-style: the
    # model hears itself). Separate embedding tables from the user stream.
    asst_audio_input: bool = True


class VoiceAILM(nn.Module):
    def __init__(self, cfg: VoiceAIConfig):
        super().__init__()
        self.cfg = cfg

        kwargs = {
            "torch_dtype": getattr(torch, cfg.dtype),
            "trust_remote_code": True,
        }
        if cfg.load_in_4bit:
            from transformers import BitsAndBytesConfig

            kwargs["quantization_config"] = BitsAndBytesConfig(
                load_in_4bit=True,
                bnb_4bit_quant_type="nf4",
                bnb_4bit_compute_dtype=getattr(torch, cfg.dtype),
                bnb_4bit_use_double_quant=True,
            )

        self.backbone = AutoModelForCausalLM.from_pretrained(cfg.backbone, **kwargs)
        self.tokenizer = AutoTokenizer.from_pretrained(cfg.backbone, trust_remote_code=True)

        added = self.tokenizer.add_tokens(SPECIAL_TOKENS, special_tokens=True)
        if added:
            self.backbone.resize_token_embeddings(len(self.tokenizer))

        d_model = self.backbone.config.hidden_size

        self.audio_in = AudioAdapter(
            d_model=d_model,
            num_codebooks=cfg.num_codebooks,
            codebook_size=cfg.codebook_size,
        )
        self.audio_in_asst = (
            AudioAdapter(
                d_model=d_model,
                num_codebooks=cfg.num_codebooks,
                codebook_size=cfg.codebook_size,
            )
            if cfg.asst_audio_input
            else None
        )
        def _make_audio_head():
            if cfg.use_depth_transformer:
                return MimiDepthTransformer(
                    d_model=d_model,
                    num_codebooks=cfg.num_codebooks,
                    codebook_size=cfg.codebook_size,
                    depth_dim=cfg.depth_dim,
                    num_layers=cfg.depth_layers,
                )
            return MimiOutputHeads(
                d_model=d_model,
                num_codebooks=cfg.num_codebooks,
                codebook_size=cfg.codebook_size,
            )

        self.asst_audio_out = _make_audio_head()
        self.user_audio_out = _make_audio_head() if cfg.train_user_audio else None

        if cfg.freeze_backbone:
            for p in self.backbone.parameters():
                p.requires_grad_(False)

    # ------------------------------------------------------------------
    # Forward
    # ------------------------------------------------------------------
    def forward(
        self,
        text_ids: torch.Tensor | None = None,
        user_audio_codes: torch.Tensor | None = None,
        asst_audio_codes: torch.Tensor | None = None,
        text_audio_mask: torch.Tensor | None = None,
        attention_mask: torch.Tensor | None = None,
        labels_text: torch.Tensor | None = None,
        labels_user_audio: torch.Tensor | None = None,
        labels_asst_audio: torch.Tensor | None = None,
    ) -> dict:
        """Compute hidden states and (optionally) losses.

        text_ids: [B, T] token IDs at frame positions (mostly silent placeholders).
        user_audio_codes: [B, K, T] Mimi codes for the user audio stream.
        asst_audio_codes: [B, K, T] Mimi codes of the assistant's OWN stream
                          (what it already said) — summed into the input so the
                          model hears itself. Requires cfg.asst_audio_input.
        text_audio_mask: [B, T] bool — True where the position is an audio token
                         (replace text-embed with audio-adapter embed).
        labels_*: matching tensors for loss; ignore_index = -100.

        Returns dict with hidden, logits, and loss components.
        """
        device = next(self.parameters()).device
        if text_ids is None and user_audio_codes is None:
            raise ValueError("provide at least text_ids or user_audio_codes")

        text_embed_layer = self.backbone.get_input_embeddings()
        bb_dtype = text_embed_layer.weight.dtype

        if text_ids is not None:
            embeds = text_embed_layer(text_ids)
        else:
            B = user_audio_codes.shape[0]
            T = user_audio_codes.shape[2]
            embeds = torch.zeros(B, T, self.backbone.config.hidden_size, device=device, dtype=bb_dtype)

        if user_audio_codes is not None:
            audio_embeds = self.audio_in(user_audio_codes).to(dtype=bb_dtype)
            if text_audio_mask is None:
                embeds = embeds + audio_embeds
            else:
                mask = text_audio_mask.unsqueeze(-1).to(embeds.dtype)
                embeds = embeds * (1 - mask) + audio_embeds * mask

        if asst_audio_codes is not None and self.audio_in_asst is not None:
            embeds = embeds + self.audio_in_asst(asst_audio_codes).to(dtype=bb_dtype)

        embeds = embeds.to(dtype=bb_dtype)
        outputs = self.backbone(
            inputs_embeds=embeds,
            attention_mask=attention_mask,
            output_hidden_states=True,
            return_dict=True,
        )
        hidden = outputs.hidden_states[-1]

        result: dict = {"hidden": hidden, "loss": None, "loss_parts": {}}

        text_logits = self.backbone.lm_head(hidden)
        result["text_logits"] = text_logits
        # Depth transformer is autoregressive over codebooks: full logits only
        # exist under teacher forcing (inside .loss) or via .sample() per frame.
        if not self.cfg.use_depth_transformer:
            result["asst_audio_logits"] = self.asst_audio_out(hidden)
            if self.user_audio_out is not None:
                result["user_audio_logits"] = self.user_audio_out(hidden)

        total_loss = None
        if labels_text is not None and self.cfg.train_text:
            text_loss = nn.functional.cross_entropy(
                text_logits.reshape(-1, text_logits.size(-1)).float(),
                labels_text.reshape(-1),
                ignore_index=-100,
            )
            result["loss_parts"]["text"] = text_loss.detach()
            total_loss = (total_loss or 0) + self.cfg.text_loss_weight * text_loss

        if labels_asst_audio is not None and self.cfg.train_asst_audio:
            a_loss = self.asst_audio_out.loss(hidden, labels_asst_audio)
            result["loss_parts"]["asst_audio"] = a_loss.detach()
            total_loss = (total_loss or 0) + self.cfg.asst_audio_loss_weight * a_loss

        if (
            labels_user_audio is not None
            and self.user_audio_out is not None
            and self.cfg.train_user_audio
        ):
            u_loss = self.user_audio_out.loss(hidden, labels_user_audio)
            result["loss_parts"]["user_audio"] = u_loss.detach()
            total_loss = (total_loss or 0) + self.cfg.user_audio_loss_weight * u_loss

        result["loss"] = total_loss
        return result

    # ------------------------------------------------------------------
    # Save / load
    # ------------------------------------------------------------------
    def save_pretrained(self, path: str | Path, save_backbone: bool = True) -> None:
        path = Path(path)
        path.mkdir(parents=True, exist_ok=True)
        torch.save(
            {
                "audio_in": self.audio_in.state_dict(),
                "audio_in_asst": (
                    self.audio_in_asst.state_dict() if self.audio_in_asst else None
                ),
                "asst_audio_out": self.asst_audio_out.state_dict(),
                "user_audio_out": (
                    self.user_audio_out.state_dict() if self.user_audio_out else None
                ),
                "cfg": self.cfg.__dict__,
            },
            path / "adapters.pt",
        )
        if save_backbone:
            self.backbone.save_pretrained(path / "backbone")
            self.tokenizer.save_pretrained(path / "backbone")

    @classmethod
    def from_pretrained(cls, path: str | Path) -> "VoiceAILM":
        path = Path(path)
        adapters = torch.load(path / "adapters.pt", map_location="cpu")
        # ckpts from before the depth transformer used independent linear heads
        adapters["cfg"].setdefault("use_depth_transformer", False)
        cfg = VoiceAIConfig(**adapters["cfg"])
        backbone_dir = path / "backbone"
        if backbone_dir.exists():
            cfg.backbone = str(backbone_dir)
        # else: keep cfg.backbone as the original HF id (intermediate ckpt
        # without full backbone snapshot)
        model = cls(cfg)
        model.audio_in.load_state_dict(adapters["audio_in"])
        model.asst_audio_out.load_state_dict(adapters["asst_audio_out"])
        if adapters.get("user_audio_out") and model.user_audio_out:
            model.user_audio_out.load_state_dict(adapters["user_audio_out"])
        if model.audio_in_asst is not None:
            if adapters.get("audio_in_asst"):
                model.audio_in_asst.load_state_dict(adapters["audio_in_asst"])
            else:
                # warm-start the new asst-stream adapter from the trained
                # user-stream adapter (stage 1 ckpts predate it)
                model.audio_in_asst.load_state_dict(adapters["audio_in"])
        return model

    # ------------------------------------------------------------------
    # Param utilities for staged training
    # ------------------------------------------------------------------
    def freeze_backbone(self) -> None:
        for p in self.backbone.parameters():
            p.requires_grad_(False)

    def unfreeze_backbone(self) -> None:
        for p in self.backbone.parameters():
            p.requires_grad_(True)

    def trainable_param_count(self) -> int:
        return sum(p.numel() for p in self.parameters() if p.requires_grad)

    def total_param_count(self) -> int:
        return sum(p.numel() for p in self.parameters())
