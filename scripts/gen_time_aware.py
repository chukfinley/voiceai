"""Generate time-aware training samples.

Each sample teaches the model one or more of:
  - Long silences with self-initiated speech ("if no response in 5s, ask")
  - Explicit <wait:Ns> usage ("wait 3 seconds then answer")
  - Time-of-elapsed awareness ("how long have we been talking?")

We synthesize narratives where the assistant must keep silent for measured
durations and then speak.
"""
from __future__ import annotations

import argparse
import json
import random
import uuid
from pathlib import Path

import numpy as np


SCENARIOS = [
    ("wait_explicit", "Wait three seconds then say hello.", [(3.0, "Hello.")]),
    ("wait_explicit", "Wait five seconds and then count to three.", [(5.0, "One. Two. Three.")]),
    ("self_initiate", "Are you still there?", [(8.0, "Yes, I'm here.")]),
    ("silence_then_ack", "Just listen for ten seconds.", [(10.0, "Okay, I listened.")]),
    ("elapsed_query", "How long have we been talking?", [(0.5, "About thirty seconds.")]),
]


def build_one(rng: random.Random) -> dict:
    name, user_text, asst_schedule = rng.choice(SCENARIOS)
    sample_id = f"ta_{name}_{uuid.uuid4().hex[:8]}"

    duration_s = max(a_t for a_t, _ in asst_schedule) + 5.0
    return {
        "sample_id": sample_id,
        "category": name,
        "user_text": user_text,
        "asst_schedule": asst_schedule,
        "duration_s": duration_s,
    }


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("--out", type=Path, required=True)
    p.add_argument("--n", type=int, default=100)
    p.add_argument("--seed", type=int, default=0)
    args = p.parse_args()
    args.out.mkdir(parents=True, exist_ok=True)

    rng = random.Random(args.seed)
    meta = [build_one(rng) for _ in range(args.n)]
    (args.out / "samples.jsonl").write_text("\n".join(json.dumps(m) for m in meta))
    print(f"wrote {args.n} time-aware spec samples → {args.out}/samples.jsonl")
    print("next step: synthesize audio + encode with Mimi via gen_synthesize_audio.py")


if __name__ == "__main__":
    main()
