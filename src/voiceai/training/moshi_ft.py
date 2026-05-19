"""Option C training entry: fine-tune Kyutai Moshi.

Uses nu-dialogue/moshi-finetune as upstream. We wrap their CLI with our
data adapter that converts the 200ms frame format → Moshi's expected
dual-stream parquet shards (Mimi token ids per 80ms frame).

Steps:
  1. Convert paired dialog audio → Mimi tokens (encode with kyutai/mimi).
  2. Pack into 80ms-frame multi-codebook tensor [T, 8] per stream.
  3. Run moshi-finetune trainer with our shards.

Output: a Moshi LoRA or full-ft checkpoint that handles German + our
interaction tasks (time-aware, visual-aware via separate side-channel —
visual is NOT in Moshi natively, would need extra encoder).
"""
from __future__ import annotations

import argparse
from pathlib import Path


def build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser()
    p.add_argument("--moshi-base", default="kyutai/moshiko-pytorch-bf16")
    p.add_argument("--paired-data", required=True, type=Path)
    p.add_argument("--output", required=True, type=Path)
    p.add_argument("--frame-rate-hz", type=int, default=12)
    p.add_argument("--codebooks", type=int, default=8)
    return p


def main() -> None:
    args = build_parser().parse_args()
    raise NotImplementedError(
        "Pipeline: see refs/moshi for Mimi encode; "
        "use https://github.com/nu-dialogue/moshi-finetune as trainer."
    )


if __name__ == "__main__":
    main()
