# voiceai

Open-source competitor to [Thinking Machines Interaction Model](https://thinkingmachines.ai/blog/interaction-models/).

Native end-to-end speech LLM. Listens **while** it speaks. Time-aware. Visual-proactive. Tool-capable via background LLM.

## Architecture

| Component | Choice | Trainable | Size |
|-----------|--------|-----------|------|
| Backbone | Qwen3.5-0.8B (Apache 2.0) | LoRA in Stage 2+ | 873M |
| Audio codec | Kyutai Mimi (CC-BY-4.0) | frozen | 145M |
| Audio adapter | own | yes | ~50M |
| Output heads | own (8 Mimi codebooks) | yes | ~50M |
| Backend LLM | pluggable (Claude/GPT/Qwen-Max API) | n/a | external |

Total trainable ≈ 920M ("close to 1B"). English-only PoC, single voice.

## Status

Production code skeleton complete. Ready for first training runs.

- **`PROJECT_STATE.md` — single source of truth (read this first)**
- `PLAN.md` — strategy
- `TRAINING_IDEAS.md` — research + ideas for the real training run
- `OSS_LANDSCAPE.md` — comprehensive 2026 OSS inventory
- `COMPUTE.md` — cheap/free training options
- `HOW_TO_TRAIN.md` — concrete commands, Stage-by-Stage

## Setup

```bash
uv sync
# optional: uv sync --extra train --extra infer --extra eval
```

## Komponenten

```
src/voiceai/
├── orchestrator/   # async event bus, audio I/O, AEC, tick
├── foreground/     # interaction model — cascade (Option A) oder Qwen3-Omni (Option B)
├── background/     # heavy LLM bridge — Qwen3-Max API oder lokal
├── visual/         # streaming VLM watcher für visual proactivity
├── training/       # LoRA SFT, Full SFT, Moshi-FT, dual-stream Daten
└── eval/           # FD-bench, TimeSpeak, etc.
```

## Quick demo (Option A, half-duplex cascade)

```bash
uv run python scripts/bootstrap.py --option a
```

## Mit Background-Reasoning

```bash
export DASHSCOPE_API_KEY=sk-...
uv run python scripts/bootstrap.py --option a --bg
```

## Mit visueller Proactivity

```bash
uv run python scripts/bootstrap.py --option a --vis
```

## Synthetische Trainingsdaten generieren

```bash
uv run python scripts/gen_synth_data.py --n 10000 --out data/interaction_sft.jsonl
```

## LoRA SFT

```bash
uv run python -m voiceai.training.lora_sft \
    --backbone Qwen/Qwen3-Omni-30B-A3B-Instruct \
    --data data/interaction_sft.jsonl \
    --output runs/lora_v1
```

## Reference repos

- `refs/moshi/` — Kyutai Moshi (dual-stream full-duplex, Mimi codec)
- `refs/ultravox/` — Fixie Ultravox (audio adapter to LLM, frozen-LLM training)

## Lizenz

Apache-2.0 (Code). Modelle behalten ihre eigenen Lizenzen.
