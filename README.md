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

## Our goal: TRUE full-duplex — and the model landscape

**End goal:** a speech model as good as OpenAI Advanced Voice Mode, but **genuinely
full-duplex**. Most "voice AI" is NOT full-duplex — it just *feels* fast.

**Definition we hold ourselves to.** *True full-duplex* = the model jointly models the
**user audio stream AND its own assistant stream every single frame**, so it can start
speaking, backchannel ("mhm"), or interrupt **mid-user-utterance** — no external VAD, no
turn-taking gate. *Turn-based* = you finish → it replies (optionally with a bolted-on VAD
that cuts playback to fake an interrupt). **We want the first kind.**

### A. TRUE full-duplex (joint dual-stream, native barge-in) — the real targets

| Model | Backbone / codec | License | Train code? | Notes |
|---|---|---|---|---|
| **Moshi** (Kyutai) | Helium-7B + Mimi, dual-stream RQ-Transformer | CC-BY-4.0 | ✅ `moshi-finetune` | The reference. ~40GB at default finetune config; 24GB unverified. |
| **PersonaPlex** (NVIDIA) | Moshi-based + voice/role control | code MIT / weights NVIDIA OML | ⚠️ inference+eval only | Barge-in formally evaluated (FullDuplexBench). Cloned in `refs/personaplex/`. |
| **Hertz-dev** (Standard Intelligence) | 8.5B + Hertz-Codec @8Hz | Apache-2.0 | ❌ inference only | Real joint dual-stream, 120ms on a 4090. Train-it-yourself. |
| **BayLing-Duplex** | single AR LLM over GLM-4-Voice | CC-BY-NC-ND 4.0 | ❌ inference only | 100% interrupt success, no external VAD. No-derivatives license = hard block. |
| **dGSLM** (Meta/fairseq) | dual-tower, HuBERT units | MIT | ✅ fairseq | Unit-based, telephone-quality, old; proves the dual-tower idea. |
| **SyncLLM** | frame-sync interleaved streams | — | ❌ no repo found | Paper only (arXiv 2409.15594). |
| **SALMONN-omni** (Bytedance) | codec-free, embedding streams + "thinking gate" | research | ❌ no usable open code | Claims +30% over prior open full-duplex. |
| **Thinking Machines Interaction Model** | dMel continuous + flow-matching, from scratch | **closed** | ❌ | Our conceptual target. No code/weights/recipe. **Note: their design is the opposite of ours** (continuous dMel + from-scratch, not frozen-LLM + discrete codec). |
| **OpenAI GPT-4o voice / xAI Grok voice** | undisclosed | **closed** | ❌ | Black boxes. Only latency + safety published. The quality bar we chase. |

### B. Turn-based / half-duplex / pure TTS — **NOT our goal**

These are useful as components, warm-starts, or data tools — but **none of them is
full-duplex.** Listed here so we never mistake one for the target.

| Model | What it actually is | License | Why it's not full-duplex |
|---|---|---|---|
| **Ultravox** (Fixie) | audio→text adapter, **input only, no audio out** | MIT | Turn-based, can't speak. (Closest precedent for our *frozen-LLM + projector* input bridge, nothing more.) |
| **Sesame CSM-1B** | contextual TTS (text+context → Mimi codes) | Apache-2.0 | Single-stream, turn-based; "begins again on next request." Maya's barge-in is unreleased external glue. |
| **Mini-Omni / Mini-Omni2** | Qwen2-0.5B speech chat | MIT | Interruptible *turn-taking* via `irq` token — not joint co-modeling. |
| **GLM-4-Voice** | end-to-end zh/en speech LLM | Apache code / restricted weights | Half-duplex + external VAD. |
| **LLaMA-Omni / LLaMA-Omni2** | streaming speech chat | non-commercial weights | Streaming TTS + VAD, turn-based. (Stage-2 pattern — separate speech head on **frozen** LLM hidden states — is worth stealing for our frozen-backbone audio-out.) |
| **Step-Audio / Step-Audio 2 mini** | speech chat + tool calls | mixed / Apache (2-mini) | Turn-based, external VAD. |
| **VITA-Audio** | Qwen2.5-7B speech | non-commercial | Turn-based MCTP. |
| **Qwen2.5-Omni / Qwen3-Omni** | omni (audio+vision+text) | Apache-2.0 | Half-duplex Thinker-Talker. |
| **Freeze-Omni** (Tencent) | frozen-LLM S2S | non-commercial | Partial: chunk-state classifier but **VAD-gated entry**; inference only. Steal its 0/1/2 "when to speak" head, not the duplex claim. |
| **Kokoro / Orpheus / CosyVoice 2 / F5-TTS / XTTS / Parler / VoxCPM** | pure TTS | various | Text→speech only. We use Kokoro as a **data generator**, never at inference. |

**One-line rule for this repo:** if a model needs an external VAD to "interrupt," or
restarts on each request, it is **turn-based** — it is a component or a warm-start, not
the architecture. Our model co-models both streams natively. Full inventory + codecs +
datasets in [`OSS_LANDSCAPE.md`](OSS_LANDSCAPE.md).

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
