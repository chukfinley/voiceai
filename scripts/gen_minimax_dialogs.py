#!/usr/bin/env python
"""Generate diverse short dialogs via the Minimax API (or any OpenAI-compatible LLM).

Combinatorial prompts (topic × persona × style × scenario) force variety instead
of relying on the model to spontaneously not repeat. Writes dialogs.jsonl where
each line is a list of turns: [{"speaker":"user","text":...}, {"speaker":"asst",...}, ...].

Downstream: scripts/gen_general_dialog.py --dialogs-file dialogs.jsonl renders
these to audio (kokoro TTS) + Mimi-encodes them into dual-stream training samples.

Credentials via env (Minimax = free for the user):
    MINIMAX_API_KEY=...            (required)
    MINIMAX_BASE_URL=https://api.minimaxi.chat/v1   (OpenAI-compatible endpoint)
    MINIMAX_MODEL=MiniMax-Text-01

    uv run python scripts/gen_minimax_dialogs.py --n 5000 --out data/dialogs.jsonl
"""
from __future__ import annotations

import argparse
import json
import os
import random
import sys
from pathlib import Path

TOPICS = [
    "the weather", "cooking dinner", "a movie you watched", "weekend plans",
    "a new phone", "getting a dog", "running late for work", "a vacation to the coast",
    "fixing a flat tire", "learning guitar", "a noisy neighbor", "buying groceries",
    "a job interview", "the best pizza topping", "missing the bus", "a science fact",
    "planting a garden", "a board game night", "a power outage", "trying a new cafe",
    "a long hike", "losing your keys", "a birthday party", "the price of coffee",
    "fixing a leaky faucet", "a soccer match", "moving to a new apartment",
    "a strange dream", "the best way to make tea", "a thunderstorm last night",
]
PERSONAS = [
    "a curious child", "a tired office worker", "a cheerful grandmother",
    "a sarcastic teenager", "a polite stranger", "an excited tourist",
    "a calm scientist", "a busy chef", "a friendly neighbor", "a sleepy student",
]
STYLES = [
    "ask a question and get a short helpful answer",
    "make a request and get a confirmation",
    "share news and get a reaction",
    "ask for an opinion and get one",
    "have a quick casual back-and-forth",
    "ask for directions and get them",
]


def build_prompt(rng: random.Random, k: int) -> str:
    items = []
    for _ in range(k):
        items.append(
            f"- topic: {rng.choice(TOPICS)}; one speaker is {rng.choice(PERSONAS)}; "
            f"the exchange should {rng.choice(STYLES)}"
        )
    spec = "\n".join(items)
    return (
        "Write short spoken-style English dialogs between USER and ASSISTANT. "
        "2 to 4 turns each, natural and conversational, each turn one short sentence. "
        f"Produce exactly {k} dialogs, one per spec below:\n{spec}\n\n"
        'Return ONLY a JSON array. Each element is an array of turns; each turn is '
        '{"speaker":"user"|"asst","text":"..."}. No markdown, no commentary.'
    )


def call_llm(prompt: str, base_url: str, key: str, model: str) -> str:
    import httpx

    r = httpx.post(
        f"{base_url.rstrip('/')}/chat/completions",
        headers={"Authorization": f"Bearer {key}", "Content-Type": "application/json"},
        json={
            "model": model,
            "messages": [{"role": "user", "content": prompt}],
            "temperature": 1.0,
            "max_tokens": 6000,  # room for M3 reasoning + a batch of dialogs
        },
        timeout=120,
    )
    r.raise_for_status()
    return r.json()["choices"][0]["message"]["content"]


def parse_dialogs(text: str) -> list:
    import re

    # M3 is a reasoning model: strip its <think>...</think> block
    text = re.sub(r"<think>.*?</think>", "", text, flags=re.DOTALL).strip()
    # pull the first JSON array out of whatever's left
    if "```" in text:
        m = re.search(r"```(?:json)?\s*(.*?)```", text, flags=re.DOTALL)
        if m:
            text = m.group(1).strip()
    if not text.startswith("["):
        m = re.search(r"\[.*\]", text, flags=re.DOTALL)
        if m:
            text = m.group(0)
    try:
        data = json.loads(text)
    except Exception:
        return []
    out = []
    for d in data if isinstance(data, list) else []:
        turns = [
            {"speaker": "asst" if str(t.get("speaker", "")).lower().startswith("a") else "user",
             "text": str(t.get("text", "")).strip()}
            for t in d if str(t.get("text", "")).strip()
        ]
        if 2 <= len(turns) <= 6:
            out.append(turns)
    return out


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("--n", type=int, default=5000, help="target number of dialogs")
    p.add_argument("--per-call", type=int, default=10, help="dialogs requested per API call")
    p.add_argument("--out", type=Path, default=Path("data/dialogs.jsonl"))
    p.add_argument("--seed", type=int, default=0)
    a = p.parse_args()

    key = os.environ.get("MINIMAX_API_KEY")
    base = os.environ.get("MINIMAX_BASE_URL", "https://api.minimaxi.chat/v1")
    model = os.environ.get("MINIMAX_MODEL", "MiniMax-Text-01")
    if not key:
        sys.exit("set MINIMAX_API_KEY (and optionally MINIMAX_BASE_URL, MINIMAX_MODEL)")

    a.out.parent.mkdir(parents=True, exist_ok=True)
    rng = random.Random(a.seed)
    seen: set[str] = set()
    written = 0
    with a.out.open("w") as f:
        calls = 0
        while written < a.n and calls < a.n:  # safety bound
            calls += 1
            try:
                raw = call_llm(build_prompt(rng, a.per_call), base, key, model)
            except Exception as e:
                print(f"[warn] call {calls} failed: {e}", file=sys.stderr)
                continue
            for turns in parse_dialogs(raw):
                key_h = "|".join(t["text"] for t in turns)
                if key_h in seen:
                    continue
                seen.add(key_h)
                f.write(json.dumps(turns) + "\n")
                written += 1
            if calls % 10 == 0:
                print(f"  {written}/{a.n} dialogs ({calls} calls)")
    print(f"wrote {written} dialogs -> {a.out}")


if __name__ == "__main__":
    main()
