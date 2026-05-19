"""Generate diverse dialog samples by prompting an LLM API.

For each sample:
  1. Pick a scenario from the catalog.
  2. Prompt an LLM (any OpenAI-compat endpoint) to write a fresh dialog.
  3. Parse the JSON-formatted output into turns.
  4. TTS each turn with two voices (user vs assistant).
  5. Mix into dual-stream audio with realistic gaps.
  6. Encode with Mimi → save sample.

This produces 10-100× more variety than hand-written templates and covers
the "human-like conversation" generalization gap.

Usage:
    export OPENAI_API_KEY=sk-...
    uv run python scripts/gen_diverse_dialogs.py \\
        --out data/diverse --n 5000 --provider openai --model gpt-4.1-mini \\
        --encode-mimi

For cheap generation use a small open model:
    --provider dashscope --model qwen3-max
    --provider vllm  (own local 32B serving)
"""
from __future__ import annotations

import argparse
import asyncio
import json
import os
import random
import uuid
from pathlib import Path

import numpy as np
import torch
from tqdm.auto import tqdm

from voiceai.background import get_bridge
from voiceai.training.data.mixing import encode_dual_stream, save_dual_stream_sample
from voiceai.training.data.scenario_catalog import SCENARIOS, Scenario
from voiceai.training.data.tts_util import (
    KOKORO_VOICES_FEMALE,
    KOKORO_VOICES_MALE,
    synth,
)


GENERATE_PROMPT = """{description}

{scenario_prompt}

Output STRICT JSON only, no commentary, in this exact shape:
{{
  "title": "short description",
  "turns": [
    {{"role": "user", "text": "..."}},
    {{"role": "assistant", "text": "..."}}
  ]
}}

Constraints:
- {n_turns_min}-{n_turns_max} turns total
- Alternate roles starting with "user"
- Each turn 1-30 words, natural spoken English
- No stage directions, no parentheticals
- No emojis
- This will be spoken aloud, so write conversationally
"""


async def generate_one_script(bridge, scenario: Scenario, max_retries: int = 2) -> dict | None:
    prompt = GENERATE_PROMPT.format(
        description=scenario.description,
        scenario_prompt=scenario.prompt,
        n_turns_min=scenario.n_turns_range[0],
        n_turns_max=scenario.n_turns_range[1],
    )
    for attempt in range(max_retries + 1):
        try:
            raw = await bridge.query(
                prompt,
                system="You are a dialogue writer producing strict JSON only. No surrounding text.",
                max_tokens=900,
                temperature=0.95,
            )
            raw = raw.strip()
            start = raw.find("{")
            end = raw.rfind("}")
            if start < 0 or end < 0:
                continue
            data = json.loads(raw[start : end + 1])
            if "turns" not in data or not data["turns"]:
                continue
            return data
        except Exception:
            if attempt == max_retries:
                return None
    return None


def render_dialog(
    data: dict, rng: random.Random, backend: str
) -> dict:
    """TTS turns → dual-stream audio."""
    user_voice = rng.choice(KOKORO_VOICES_FEMALE)
    asst_voice = rng.choice(KOKORO_VOICES_MALE)
    sr = 24000

    rendered = []
    for t in data["turns"]:
        role = t.get("role", "user")
        text = t.get("text", "").strip()
        if not text:
            continue
        voice = user_voice if role == "user" else asst_voice
        audio, _ = synth(text, voice=voice, backend=backend)
        rendered.append({"role": role, "text": text, "audio": audio.astype(np.float32)})

    if not rendered:
        return {}

    gap_min_s = 0.25
    gap_max_s = 0.7
    user_track = []
    asst_track = []
    transcript = []
    cursor = 0
    for t in rendered:
        n = len(t["audio"])
        if t["role"] == "user":
            user_track.append((cursor, t["audio"]))
        else:
            asst_track.append((cursor, t["audio"]))
        transcript.append(
            {"role": t["role"], "text": t["text"], "start_s": cursor / sr, "end_s": (cursor + n) / sr}
        )
        cursor += n + int(rng.uniform(gap_min_s, gap_max_s) * sr)

    total = cursor + sr  # tail
    user_audio = np.zeros(total, dtype=np.float32)
    asst_audio = np.zeros(total, dtype=np.float32)
    for start, a in user_track:
        user_audio[start : start + len(a)] = a
    for start, a in asst_track:
        asst_audio[start : start + len(a)] = a

    return {
        "user_audio": user_audio,
        "asst_audio": asst_audio,
        "user_voice": user_voice,
        "asst_voice": asst_voice,
        "transcript": transcript,
        "sr": sr,
        "title": data.get("title", ""),
    }


async def amain() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("--out", type=Path, required=True)
    p.add_argument("--n", type=int, default=500)
    p.add_argument("--provider", default="openai")
    p.add_argument("--model", default=None)
    p.add_argument("--base-url", default=None)
    p.add_argument("--tts-backend", default="kokoro", choices=["kokoro", "melotts", "gtts"])
    p.add_argument("--encode-mimi", action="store_true")
    p.add_argument("--device", default="cuda")
    p.add_argument("--concurrency", type=int, default=8)
    p.add_argument("--scenarios", nargs="+", default=None, help="filter to specific scenarios")
    p.add_argument("--seed", type=int, default=0)
    args = p.parse_args()

    args.out.mkdir(parents=True, exist_ok=True)
    raw_dir = args.out / "raw"
    raw_dir.mkdir(exist_ok=True)
    enc_dir = args.out / "encoded"
    enc_dir.mkdir(exist_ok=True)
    scripts_dir = args.out / "scripts"
    scripts_dir.mkdir(exist_ok=True)

    rng = random.Random(args.seed)

    bridge_kwargs = {}
    if args.model:
        bridge_kwargs["model"] = args.model
    if args.base_url:
        bridge_kwargs["base_url"] = args.base_url
    bridge = get_bridge(args.provider, **bridge_kwargs)

    pool = [s for s in SCENARIOS if not args.scenarios or s.name in args.scenarios]
    weights = [s.weight for s in pool]

    mimi = None
    if args.encode_mimi:
        from voiceai.model.mimi_utils import load_mimi

        mimi = load_mimi(device=args.device, dtype=torch.bfloat16)

    import soundfile as sf

    sem = asyncio.Semaphore(args.concurrency)

    async def one_script_job(scenario: Scenario):
        async with sem:
            return await generate_one_script(bridge, scenario)

    print(f"[1/2] generating {args.n} dialog scripts via {args.provider}…")
    chosen = rng.choices(pool, weights=weights, k=args.n)
    script_tasks = [one_script_job(s) for s in chosen]
    scripts = []
    for coro in tqdm(asyncio.as_completed(script_tasks), total=len(script_tasks)):
        data = await coro
        if data is not None:
            scripts.append(data)

    print(f"got {len(scripts)} valid scripts. Saving to {scripts_dir}…")
    for i, sc in enumerate(scripts):
        (scripts_dir / f"script_{i:06d}.json").write_text(json.dumps(sc, indent=2))

    print(f"[2/2] rendering audio + encoding Mimi…")
    metas = []
    for i, sc in enumerate(tqdm(scripts)):
        try:
            r = render_dialog(sc, rng, args.tts_backend)
        except Exception as e:
            print(f"render fail {i}: {e}")
            continue
        if not r:
            continue
        sid = f"div_{uuid.uuid4().hex[:10]}"
        sf.write(raw_dir / f"{sid}_user.wav", r["user_audio"], r["sr"])
        sf.write(raw_dir / f"{sid}_asst.wav", r["asst_audio"], r["sr"])

        meta = {
            "sample_id": sid,
            "category": "diverse_dialog",
            "title": r["title"],
            "transcript": r["transcript"],
            "user_voice": r["user_voice"],
            "asst_voice": r["asst_voice"],
            "duration_s": float(len(r["user_audio"]) / r["sr"]),
            "scenario": chosen[i].name if i < len(chosen) else "unknown",
        }
        metas.append(meta)

        if mimi is not None:
            try:
                u_codes, a_codes = encode_dual_stream(
                    r["user_audio"], r["asst_audio"], mimi, sr=r["sr"], device=args.device
                )
                save_dual_stream_sample(
                    user_codes=u_codes,
                    asst_codes=a_codes,
                    text_ids=np.array([], dtype=np.int32),
                    text_align=np.array([], dtype=np.int32),
                    aux={
                        "category": meta["category"],
                        "scenario": meta["scenario"],
                        "transcript": r["transcript"],
                        "title": r["title"],
                    },
                    sample_id=sid,
                    out_root=enc_dir,
                    duration_s=meta["duration_s"],
                )
            except Exception as e:
                print(f"encode fail {sid}: {e}")

    (args.out / "samples.jsonl").write_text("\n".join(json.dumps(m) for m in metas))
    print(f"wrote {len(metas)} diverse-dialog samples → {args.out}")


def main() -> None:
    asyncio.run(amain())


if __name__ == "__main__":
    main()
