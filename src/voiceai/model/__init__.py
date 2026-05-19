"""voiceai model components.

Architecture (Phase 3 target):

    Mimi encoder (frozen) ──► audio tokens 8×12.5Hz
                                │
                                ▼
                       AudioAdapter (trained)
                                │
                                ▼
                  Qwen3.5-0.8B backbone (LoRA-trained)
                  + dual-stream wrapper (trained)
                                │
                                ▼
                  MimiOutputHeads (trained, 8 codebooks)
                                │
                                ▼
                  Mimi decoder (frozen) ──► PCM
"""
from .audio_adapter import AudioAdapter, MimiOutputHeads
from .mimi_utils import load_mimi, mimi_encode, mimi_decode
from .voiceai_lm import VoiceAILM, VoiceAIConfig

__all__ = [
    "AudioAdapter",
    "MimiOutputHeads",
    "VoiceAILM",
    "VoiceAIConfig",
    "load_mimi",
    "mimi_encode",
    "mimi_decode",
]
