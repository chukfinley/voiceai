"""Speaker identification for the orchestrator (system-level, not in-model).

Keeps a registry of enrolled voices (name → embedding). On incoming user
audio, computes a speaker embedding and matches against the registry; the
orchestrator can then inject "<speaker> jack </speaker>" into the model's
inner-monologue context, or gate responses ("only talk to me").

Backend: SpeechBrain ECAPA-TDNN (small, robust, runs on CPU). Alternative
stacks worth knowing: WhisperX (word timestamps + pyannote diarization for
"who spoke when" in multi-party audio), wespeaker, resemblyzer.

Usage:
    sid = SpeakerID()
    sid.enroll("dietrich", ref_audio_16k)        # one-time, few seconds of audio
    name, score = sid.identify(chunk_16k)         # per utterance/chunk
"""
from __future__ import annotations

import json
from pathlib import Path

import numpy as np


class SpeakerID:
    def __init__(self, threshold: float = 0.25, device: str = "cpu"):
        """threshold: min cosine similarity to accept a match (ECAPA scale:
        same speaker typically 0.4-0.7, different 0.0-0.2)."""
        self.threshold = threshold
        self.device = device
        self._model = None
        self.registry: dict[str, np.ndarray] = {}

    def _ensure_model(self):
        if self._model is not None:
            return
        try:
            from speechbrain.inference.speaker import EncoderClassifier
        except ImportError as e:
            raise ImportError(
                "speaker id needs speechbrain: uv add speechbrain"
            ) from e
        self._model = EncoderClassifier.from_hparams(
            source="speechbrain/spkrec-ecapa-voxceleb",
            run_opts={"device": self.device},
        )

    def embed(self, audio_16k: np.ndarray) -> np.ndarray:
        """audio_16k: mono float32 [T] at 16 kHz → L2-normalized embedding."""
        import torch

        self._ensure_model()
        wav = torch.from_numpy(np.asarray(audio_16k, dtype=np.float32))[None]
        emb = self._model.encode_batch(wav).squeeze().cpu().numpy()
        return emb / (np.linalg.norm(emb) + 1e-8)

    def enroll(self, name: str, audio_16k: np.ndarray) -> None:
        emb = self.embed(audio_16k)
        if name in self.registry:  # running average over enrollments
            emb = (self.registry[name] + emb) / 2
            emb = emb / (np.linalg.norm(emb) + 1e-8)
        self.registry[name] = emb

    def identify(self, audio_16k: np.ndarray) -> tuple[str | None, float]:
        """Returns (best_name, cosine) or (None, best_cosine) below threshold."""
        if not self.registry:
            return None, 0.0
        emb = self.embed(audio_16k)
        best_name, best = None, -1.0
        for name, ref in self.registry.items():
            score = float(np.dot(emb, ref))
            if score > best:
                best_name, best = name, score
        if best < self.threshold:
            return None, best
        return best_name, best

    # ------------------------------------------------------------------
    # Persistence (voice prints survive restarts)
    # ------------------------------------------------------------------
    def save(self, path: str | Path) -> None:
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        np.savez(path.with_suffix(".npz"), **self.registry)
        path.with_suffix(".json").write_text(
            json.dumps({"threshold": self.threshold, "names": list(self.registry)})
        )

    def load(self, path: str | Path) -> None:
        path = Path(path)
        arr = np.load(path.with_suffix(".npz"))
        self.registry = {k: arr[k] for k in arr.files}


class SpeakerIDLoop:
    """Orchestrator loop: buffers user audio between VAD start/end, identifies
    the speaker on segment end, emits SPEAKER_ID.

    The foreground wrapper turns these into "<speaker> jack </speaker>" tokens
    in the model context. `only_listen_to` makes the loop tag everyone else's
    segments with name=None so the foreground can ignore them ("rede nur mit
    mir").
    """

    def __init__(self, sid: SpeakerID, only_listen_to: str | None = None):
        self.sid = sid
        self.only_listen_to = only_listen_to
        self._buf: list = []
        self._active = False

    async def run(self, bus) -> None:
        import numpy as np

        from .events import EvType

        async for ev in bus.stream(
            EvType.AUDIO_IN_CHUNK, EvType.USER_VAD_START, EvType.USER_VAD_END
        ):
            if ev.type == EvType.USER_VAD_START:
                self._active = True
                self._buf = []
            elif ev.type == EvType.AUDIO_IN_CHUNK and self._active:
                self._buf.append(np.asarray(ev.data, dtype=np.float32))
            elif ev.type == EvType.USER_VAD_END and self._active:
                self._active = False
                if not self._buf:
                    continue
                audio = np.concatenate(self._buf)
                if len(audio) < 8000:  # <0.5 s @16k — too short for a stable print
                    continue
                name, score = self.sid.identify(audio)
                if self.only_listen_to and name != self.only_listen_to:
                    name = None
                await bus.emit(EvType.SPEAKER_ID, data={"name": name, "score": score})
