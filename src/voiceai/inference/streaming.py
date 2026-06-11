"""Real-time streaming inference for the dual-stream VoiceAILM.

Drives the model frame-by-frame at Mimi's 12.5 Hz (80 ms per frame):

    user mic ──Mimi.encode──► user codes [K] ─┐
    own previous output codes [K] ────────────┼─► embed sum ─► backbone(KV-cache)
    previous monologue token ─────────────────┘        │
                                                       ▼
                          lm_head → next monologue token (text stream)
                          depth_transformer.sample → next asst codes [K]
                                                       │
            acoustic-delay re-alignment ──► Mimi.decode ──► speaker PCM

Context handling: when the KV cache reaches `max_frames`, the engine
re-prefills from the last `window_frames` input embeddings (positions reset).
Costs one prefill spike every (max_frames - window_frames) frames; between
spikes every step is a single-token forward.

Barge-in is the ORCHESTRATOR's call: it watches the user stream / VAD and
calls `mute()` — the engine then feeds silence as the assistant's own stream
(the model also learns to go silent on its own via <barge> training data).
"""
from __future__ import annotations

from collections import deque
from dataclasses import dataclass

import torch

from ..model.mimi_utils import MIMI_NUM_CODEBOOKS
from ..model.voiceai_lm import VoiceAILM
from ..training.data.dual_stream import ACOUSTIC_BOS


@dataclass
class StreamStep:
    """Result of one 80 ms frame step."""
    asst_codes: torch.Tensor       # [K] codes the model emitted this frame
    text_token: int                # inner-monologue token id
    audio: torch.Tensor | None     # [1, 1, samples] PCM @24 kHz (None while delay buffer fills)


class StreamingEngine:
    def __init__(
        self,
        model: VoiceAILM,
        mimi,
        device: str = "cuda",
        acoustic_delay: int = 1,
        max_frames: int = 4096,
        window_frames: int = 2048,
        temperature: float = 0.8,
        text_temperature: float = 0.0,
    ):
        if not model.cfg.use_depth_transformer:
            raise ValueError("streaming needs the depth transformer head (use_depth_transformer=True)")
        self.model = model.eval()
        self.mimi = mimi
        self.device = device
        self.delay = acoustic_delay
        self.max_frames = max_frames
        self.window_frames = min(window_frames, max_frames - 1)
        self.temperature = temperature
        self.text_temperature = text_temperature

        self.K = model.cfg.num_codebooks
        self._silent_id = model.tokenizer.convert_tokens_to_ids("<a:silent>")
        self.reset()

    # ------------------------------------------------------------------
    def reset(self) -> None:
        self.past = None
        self.frames = 0
        # model hasn't said anything yet: BOS codes on its own stream
        self.prev_asst = torch.full((1, self.K, 1), ACOUSTIC_BOS, dtype=torch.long, device=self.device)
        self.prev_text = torch.tensor([[self._silent_id]], dtype=torch.long, device=self.device)
        # re-prefill buffer of recent input embeds (sliding window)
        self._embed_buf: deque[torch.Tensor] = deque(maxlen=self.window_frames)
        # delay re-alignment: semantic emitted at t pairs with acoustics from t+delay
        self._sem_buf: deque[torch.Tensor] = deque()
        self._muted = False
        self._mimi_stream = None

    def close(self) -> None:
        if self._mimi_stream is not None:
            self._mimi_stream.__exit__(None, None, None)
            self._mimi_stream = None

    def mute(self) -> None:
        """Barge-in: orchestrator decided the assistant shuts up NOW.
        From the next frame the model hears silence on its own stream."""
        self._muted = True

    def unmute(self) -> None:
        self._muted = False

    # ------------------------------------------------------------------
    @torch.no_grad()
    def _frame_embed(self, user_codes: torch.Tensor) -> torch.Tensor:
        """user_codes: [1, K, 1] → input embed [1, 1, D]."""
        m = self.model
        bb_dtype = m.backbone.get_input_embeddings().weight.dtype
        e = m.backbone.get_input_embeddings()(self.prev_text)
        e = e + m.audio_in(user_codes).to(bb_dtype)
        if m.audio_in_asst is not None:
            e = e + m.audio_in_asst(self.prev_asst).to(bb_dtype)
        return e.to(bb_dtype)

    @torch.no_grad()
    def _forward(self, embed: torch.Tensor) -> torch.Tensor:
        """One frame through the backbone with cache; returns hidden [1, D]."""
        if self.frames >= self.max_frames:
            # sliding window: rebuild cache from the recent embeds
            self.past = None
            window = torch.cat(list(self._embed_buf), dim=1)
            out = self.model.backbone(
                inputs_embeds=window, use_cache=True,
                output_hidden_states=True, return_dict=True,
            )
            self.past = out.past_key_values
            self.frames = window.shape[1]
        out = self.model.backbone(
            inputs_embeds=embed,
            past_key_values=self.past,
            use_cache=True,
            output_hidden_states=True,
            return_dict=True,
        )
        self.past = out.past_key_values
        self.frames += 1
        return out.hidden_states[-1][:, -1]

    def _sample_text(self, hidden: torch.Tensor) -> int:
        logits = self.model.backbone.lm_head(hidden)[0]
        if self.text_temperature <= 0:
            return int(logits.argmax().item())
        probs = torch.softmax(logits / self.text_temperature, dim=-1)
        return int(torch.multinomial(probs, 1).item())

    @torch.no_grad()
    def _decode_frame(self, codes: torch.Tensor) -> torch.Tensor | None:
        """Re-align acoustic delay and stream-decode through Mimi.

        codes: [K] raw model output of this frame. Returns PCM or None while
        the delay buffer is still filling / codes contain BOS.
        """
        if self.mimi is None:  # benchmark mode: skip vocoding
            return None
        self._sem_buf.append(codes)
        if len(self._sem_buf) <= self.delay:
            return None
        past = self._sem_buf.popleft()
        frame = torch.empty(1, self.K, 1, dtype=torch.long, device=self.device)
        frame[0, 0, 0] = past[0]            # semantic from t
        frame[0, 1:, 0] = codes[1:]         # acoustics emitted at t+delay
        if int(frame.max()) >= ACOUSTIC_BOS:
            return None  # BOS/EOS frame — nothing decodable yet
        if self._mimi_stream is None and hasattr(self.mimi, "streaming"):
            self._mimi_stream = self.mimi.streaming(1)
            self._mimi_stream.__enter__()
        return self.mimi.decode(frame)

    # ------------------------------------------------------------------
    @torch.no_grad()
    def step(self, user_codes: torch.Tensor) -> StreamStep:
        """Advance one 80 ms frame.

        user_codes: [K] or [1, K, 1] Mimi codes of the user's mic frame.
        """
        if user_codes.dim() == 1:
            user_codes = user_codes.view(1, -1, 1)
        user_codes = user_codes.to(self.device)

        embed = self._frame_embed(user_codes)
        self._embed_buf.append(embed)
        hidden = self._forward(embed)

        text_token = self._sample_text(hidden)
        asst_codes = self.model.asst_audio_out.sample(
            hidden, temperature=self.temperature
        )[0]  # [K]

        if self._muted:
            asst_codes = torch.full_like(asst_codes, ACOUSTIC_BOS)

        audio = self._decode_frame(asst_codes)

        # feedback for the next frame: the model hears what it just said
        self.prev_asst = asst_codes.view(1, self.K, 1)
        self.prev_text = torch.tensor([[text_token]], dtype=torch.long, device=self.device)
        return StreamStep(asst_codes=asst_codes, text_token=text_token, audio=audio)
