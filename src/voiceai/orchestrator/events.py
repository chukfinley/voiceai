"""Event types for the central bus.

All loops (audio_in, asr, llm, tts, visual, tick) emit events; orchestrator
routes them. Keeping this typed and small.
"""
from __future__ import annotations

from dataclasses import dataclass, field
from enum import Enum
from typing import Any


class EvType(str, Enum):
    AUDIO_IN_CHUNK = "audio_in_chunk"          # raw mic PCM 200ms
    USER_PARTIAL = "user_partial"              # asr partial transcript
    USER_COMMIT = "user_commit"                # asr committed segment
    USER_VAD_START = "user_vad_start"
    USER_VAD_END = "user_vad_end"

    VISUAL_FRAME = "visual_frame"              # raw camera frame
    VISUAL_EVENT = "visual_event"              # watcher classification

    TICK = "tick"                              # periodic time marker

    LLM_TEXT_DELTA = "llm_text_delta"          # streaming token from FG llm
    LLM_AUDIO_DELTA = "llm_audio_delta"        # streaming audio chunk from FG
    LLM_DONE = "llm_done"

    BG_QUERY = "bg_query"                      # foreground asks background
    BG_RESULT = "bg_result"                    # background returns

    BARGE_IN = "barge_in"                      # user interrupted
    BACKCHANNEL = "backchannel"                # mhm/ja signal

    TTS_OUT_CHUNK = "tts_out_chunk"            # PCM to speaker
    TTS_DUCK = "tts_duck"                      # lower volume
    TTS_MUTE = "tts_mute"


@dataclass
class Event:
    type: EvType
    t: float                                   # monotonic seconds since session start
    data: Any = None
    meta: dict[str, Any] = field(default_factory=dict)
