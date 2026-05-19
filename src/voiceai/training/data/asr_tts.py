"""Stage 1 dataset: ASR + TTS samples for audio adapter pretraining.

Two objectives sharing the same model:

  ASR:  input = audio tokens (user), labels = transcript tokens (lm_head loss)
  TTS:  input = transcript (text_ids), labels = audio tokens (mimi heads loss)

We multiplex them in a single dataset; each sample has a "task" flag.

Source corpora (PoC):
  - LibriSpeech-clean-360 (English, CC-BY-4.0)
  - Common Voice 17 English (CC0)

The HF `datasets` library handles downloading + caching.
"""
from __future__ import annotations

import random
from dataclasses import dataclass

import numpy as np
import torch
from torch.utils.data import IterableDataset


@dataclass
class ASRTTSSample:
    task: str  # "asr" | "tts"
    audio_codes: np.ndarray   # [K, T_frames]
    text_ids: np.ndarray      # [T_text]


class ASRTTSDataset(IterableDataset):
    """Streams (audio_codes, text_ids, task) tuples on the fly.

    Uses a pre-built jsonl manifest where each line is:
        {"audio": "/path/to/clip.wav", "text": "transcript", "duration": 4.2}

    The Mimi encoding happens lazily — we keep audio as path on disk and
    encode in a worker. Saves ~10x disk vs storing tokens.
    """

    def __init__(
        self,
        manifest_path: str,
        tokenizer,
        mimi,
        sample_rate: int = 24000,
        mix: tuple[float, float] = (0.5, 0.5),  # P(asr), P(tts)
        max_audio_s: float = 20.0,
        seed: int = 0,
    ):
        self.manifest_path = manifest_path
        self.tokenizer = tokenizer
        self.mimi = mimi
        self.sample_rate = sample_rate
        self.mix = mix
        self.max_audio_s = max_audio_s
        self.seed = seed
        with open(manifest_path) as f:
            self.lines = [line.strip() for line in f if line.strip()]

    def __len__(self) -> int:
        return len(self.lines)

    def __iter__(self):
        import json
        import soundfile as sf
        from ...model.mimi_utils import resample_to_mimi

        rng = random.Random(self.seed)
        order = list(range(len(self.lines)))
        rng.shuffle(order)
        for idx in order:
            try:
                meta = json.loads(self.lines[idx])
                audio, sr = sf.read(meta["audio"], dtype="float32")
            except Exception:
                continue
            if audio.ndim > 1:
                audio = audio.mean(axis=1)
            if len(audio) / sr > self.max_audio_s:
                start = rng.randint(0, len(audio) - int(self.max_audio_s * sr))
                audio = audio[start : start + int(self.max_audio_s * sr)]
            t = torch.from_numpy(audio).unsqueeze(0).unsqueeze(0)
            t = resample_to_mimi(t, sr)
            mimi_param = next(self.mimi.parameters())
            t = t.to(device=mimi_param.device, dtype=mimi_param.dtype)
            with torch.no_grad():
                codes = self.mimi.encode(t)
            codes = codes[0].cpu().numpy()
            text_ids = np.array(
                self.tokenizer.encode(meta["text"], add_special_tokens=False),
                dtype=np.int32,
            )
            task = "asr" if rng.random() < self.mix[0] else "tts"
            yield ASRTTSSample(task=task, audio_codes=codes, text_ids=text_ids)


def asr_tts_collate(batch: list[ASRTTSSample], pad_token_id: int = 0) -> dict:
    """Pad batch to max length. Return dict suitable for VoiceAILM.forward()."""
    Tmax_audio = max(s.audio_codes.shape[1] for s in batch)
    Tmax_text = max(s.text_ids.shape[0] for s in batch)
    K = batch[0].audio_codes.shape[0]
    B = len(batch)

    audio_codes = np.zeros((B, K, Tmax_audio), dtype=np.int64)
    text_ids = np.full((B, Tmax_text), pad_token_id, dtype=np.int64)
    tasks: list[str] = []

    for i, s in enumerate(batch):
        audio_codes[i, :, : s.audio_codes.shape[1]] = s.audio_codes
        text_ids[i, : s.text_ids.shape[0]] = s.text_ids
        tasks.append(s.task)

    return {
        "audio_codes": torch.from_numpy(audio_codes),
        "text_ids": torch.from_numpy(text_ids),
        "tasks": tasks,
    }
