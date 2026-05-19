"""Generate synthetic interaction-SFT data.

Writes JSONL where each line is a list of Frame dicts. Use as input to
voiceai.training.lora_sft via data/loaders.TimeAwareSFTDataset.

Usage:
    uv run python scripts/gen_synth_data.py --n 10000 --out data/interaction_sft.jsonl
"""
from __future__ import annotations

import argparse
import dataclasses
import json
from pathlib import Path

from voiceai.training.data.synth import DialogSpec, script_dialog


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("--n", type=int, default=1000)
    p.add_argument("--out", type=Path, required=True)
    p.add_argument("--duration", type=float, default=30.0)
    p.add_argument("--barge-prob", type=float, default=0.15)
    p.add_argument("--vis-prob", type=float, default=0.05)
    args = p.parse_args()

    args.out.parent.mkdir(parents=True, exist_ok=True)
    spec = DialogSpec(
        duration_s=args.duration,
        barge_in_prob=args.barge_prob,
        visual_event_prob=args.vis_prob,
    )
    with args.out.open("w") as f:
        for i in range(args.n):
            frames = script_dialog(spec, seed=i)
            f.write(
                json.dumps([dataclasses.asdict(fr) for fr in frames])
                + "\n"
            )
    print(f"wrote {args.n} samples → {args.out}")


if __name__ == "__main__":
    main()
