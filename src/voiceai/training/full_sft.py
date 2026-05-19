"""Full (non-LoRA) SFT for Phase 2 — dual-stream training.

Differs from lora_sft.py:
  - trains all attention + MLP weights of the Thinker
  - keeps Talker + Code2Wav frozen (they're tightly-coupled and brittle)
  - dual-stream sample: both user and assistant audio token streams supervised
  - loss combines text + dual audio streams with relative weights

Required: 8× H100 with FSDP, gradient checkpointing on, bf16.

Run:
    uv run python -m voiceai.training.full_sft \\
        --backbone Qwen/Qwen3-Omni-30B-A3B-Instruct \\
        --paired-data data/paired/ \\
        --output runs/full_v1
"""
from __future__ import annotations

import argparse
from pathlib import Path


def build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser()
    p.add_argument("--backbone", required=True)
    p.add_argument("--paired-data", required=True, type=Path)
    p.add_argument("--output", required=True, type=Path)
    p.add_argument("--text-loss-weight", type=float, default=1.0)
    p.add_argument("--user-audio-loss-weight", type=float, default=0.5)
    p.add_argument("--asst-audio-loss-weight", type=float, default=1.0)
    p.add_argument("--freeze-talker", action="store_true", default=True)
    p.add_argument("--freeze-audio-encoder", action="store_true", default=True)
    p.add_argument("--lr", type=float, default=2e-5)
    p.add_argument("--batch-size", type=int, default=2)
    p.add_argument("--grad-accum", type=int, default=16)
    p.add_argument("--epochs", type=int, default=1)
    return p


def main() -> None:
    args = build_parser().parse_args()
    # See lora_sft.py for token registration. Then build a custom loop that:
    #   1. Tokenizes paired dialog as flat 200ms frame sequence
    #   2. Constructs targets where both user_audio[t+1] and asst_audio[t+1]
    #      are supervised (Moshi-style).
    #   3. Weights losses per stream with --*-loss-weight.
    #   4. Uses FSDP + activation checkpointing.
    #
    # This is a multi-day implementation effort; skeleton only.
    raise NotImplementedError("see PLAN.md Phase 2 — implementation pending")


if __name__ == "__main__":
    main()
