"""Synthetic data generator for interaction training.

Strategies:
  1. Two Qwen3-Omni instances chat → record both audio streams + transcripts
     with exact timing. Outputs ready-to-flatten Frame lists.
  2. Time-aware augmentation: insert silences, paraphrase responses to use
     <wait:Ns> when relevant.
  3. Barge-in augmentation: take normal dialog, insert user interruption
     mid-assistant-turn → assistant must cut + acknowledge.
  4. Backchannel augmentation: inject "mhm", "ja" tokens during user turn.
  5. Simultaneous-speech: live-translation pairs — assistant speaks in
     target language while user still speaking in source.
  6. Visual scripted scenes: pre-generate camera frame + WATCH_PROMPT-style
     event sequences with Qwen3-VL pseudo-labels.

This file is the skeleton. Real generation runs as a script, not a library.
"""
from __future__ import annotations

import random
from dataclasses import dataclass

from .format import Frame, FRAMES_PER_SEC


@dataclass
class DialogSpec:
    duration_s: float = 30.0
    user_turns: int = 3
    barge_in_prob: float = 0.15
    backchannel_prob: float = 0.10
    visual_event_prob: float = 0.05
    long_pause_prob: float = 0.10
    simul_speech_prob: float = 0.0          # only for translation runs


def script_dialog(spec: DialogSpec, seed: int = 0) -> list[Frame]:
    """Produce a *scripted* (no model call) frame list with the right shape.

    Replace each turn placeholder later with real audio tokens emitted by
    running Qwen3-Omni twice (one as user, one as assistant).
    """
    rng = random.Random(seed)
    total_frames = int(spec.duration_s * FRAMES_PER_SEC)
    frames = [Frame(idx=i) for i in range(total_frames)]

    cursor = 5  # start with 1s silence
    for _ in range(spec.user_turns):
        # user speaks for 1.5-4s
        u_dur = rng.randint(8, 20)
        for j in range(u_dur):
            if cursor + j >= total_frames:
                break
            frames[cursor + j].user_audio = [rng.randint(1, 2047) for _ in range(8)]
        cursor += u_dur

        # 200ms-800ms gap
        cursor += rng.randint(1, 4)

        # assistant responds: text then audio for 1-3s
        a_dur = rng.randint(5, 15)
        a_text = _placeholder_response()
        for j in range(a_dur):
            if cursor + j >= total_frames:
                break
            frames[cursor + j].asst_audio = [rng.randint(1, 2047) for _ in range(8)]
        if cursor < total_frames:
            frames[cursor].asst_text = a_text

        # barge-in?
        if rng.random() < spec.barge_in_prob and cursor + 2 < total_frames:
            barge_at = cursor + rng.randint(1, max(1, a_dur - 1))
            if barge_at < total_frames:
                frames[barge_at].user_audio = [rng.randint(1, 2047) for _ in range(8)]
                frames[barge_at].control.append("<barge>")
                # assistant stops generating audio after barge
                for k in range(barge_at + 1, cursor + a_dur):
                    if k < total_frames:
                        frames[k].asst_audio = []
        cursor += a_dur + rng.randint(2, 5)

        # backchannels during user turn
        if rng.random() < spec.backchannel_prob and cursor + 1 < total_frames:
            bc_at = cursor - rng.randint(1, 3)
            if 0 <= bc_at < total_frames:
                frames[bc_at].asst_text = '"mhm"'

        # visual event injection
        if rng.random() < spec.visual_event_prob and cursor < total_frames:
            frames[cursor].visual = {"event": "user_pointed", "confidence": 0.8}

        if cursor >= total_frames:
            break

    return frames


_RESPONSES = [
    "Sure, one moment.",
    "Got it.",
    "Depends — do you mean X or Y?",
    "Let me check.",
    "Yes, exactly.",
    "Hmm, let me think.",
]


def _placeholder_response() -> str:
    return random.choice(_RESPONSES)
