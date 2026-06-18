# CLAUDE.md — voiceai

Read this first. For full detail, **`PROJECT_STATE.md` is the single source of truth**
(if any doc conflicts, PROJECT_STATE wins — the rest are older).

## What this is

Open-source competitor to [Thinking Machines' Interaction Model](https://thinkingmachines.ai/blog/interaction-models/).
A **native, end-to-end full-duplex speech LLM**: listens **while** it speaks. NOT a
ASR→LLM→TTS cascade — two parallel audio streams (user + assistant) as Mimi codes,
interleaved frame-by-frame through the LLM (Moshi-style, 12.5 Hz). Plus: time-aware,
visual-proactive (separate VLM watcher), tool-capable via a background LLM.

Target capabilities: simultaneous speech, barge-in, backchannel ("mhm"), time sense,
live commentary, background reasoning with bridging, inner monologue (thinks in text,
speaks in audio), <~200ms latency. No existing OSS model does all of these — we use
existing ones as warm-starts/experiments, then train our own.

## Architecture (decided)

| Component | Choice | Trainable |
|---|---|---|
| Backbone | Qwen3 (Apache 2.0) — 1.7B for PoC (0.6B/4B alts) | frozen Stage 1, LoRA Stage 2+ |
| Audio codec | Kyutai **Mimi** (CC-BY-4.0), 12.5Hz, 8 codebooks | **frozen** |
| Audio adapter + Depth-Transformer (Moshi/RQ-style) | own | yes |
| Output heads (8 Mimi codebooks) | own | yes |
| Background LLM | pluggable (Claude/GPT/Qwen-Max API or local) | external |

Backbone is Qwen, NOT Moshi's Helium-7B (Helium = English-heavy, 2024 knowledge, weak
reasoning, not swappable). Qwen gives intelligence + multilingual for free. Mimi stays —
don't reinvent the codec.

## THE one open problem: grounding (read before proposing anything)

Measured empirically (2026-06-11), not theory:

| Input approach | Held-out WER | Grounds? |
|---|---|---|
| Mimi-only prefix-LM → frozen Qwen | 176% | **NO** |
| Mimi + LoRA on backbone | 209% / 299% (worse) | **NO** |
| **Whisper-encoder + frame-stacking bridge (SLAM-ASR)** | **10.5%** | **YES** |

Raw Mimi RVQ codes are a weak ASR representation; prefix-LM + teacher-forcing + frozen
LLM lets the LLM ignore audio (language prior). **Always measure held-out WER, never
trust teacher-forced loss.**

**Resolution = HYBRID INPUT** (what Kimi-Audio / GLM-4-Voice / Step-Audio do): feed
**both** in parallel — continuous Whisper features (for understanding/grounding) **plus**
Mimi codes (for emotion/prosody/self-hearing/audio-output). NOT either/or:
- Whisper: understands ✓, emotion ✗, audio-out ✗
- Mimi: understands ✗ (raw), emotion ✓, audio-out ✓

The deleted `whisper_lm.py` (frame-stacking with `downsample_k=5`) is the template —
in git history / branch `lora-grounding-fix` / HF `chukfinley/voiceai-asr-whisper`.
**Re-attaching the Whisper strand alongside Mimi is the #1 priority.** Without it the
model is deaf and everything else is decoration.

## Current state (PAUSED 2026-06-12)

Full-duplex pipeline runs mechanically end-to-end: audio in → model → audio +
inner-monologue text out, live in a Gradio web UI (`scripts/webui.py`). **Proven: it
trains and speaks.** Output is babble (`etch irmahood ~ ~ 娘家`) — cause is purely
data/training (only 1500 synthetic dialogs, short Stage-2, empty inner-monologue text),
NOT an architecture defect. **Open: quality needs budget for more data + longer training.**

Paused pending budget for a real run.

## Reference models / the real competitive landscape

- **Moshi** (Kyutai) — the full-duplex base (Helium-7B + Mimi + dual-stream RQ-Transformer).
  CC-BY-4.0. `refs/moshi/`. The architecture we reimplement on Qwen.
- **PersonaPlex** (NVIDIA, `refs/personaplex/`) — Moshi finetune with voice/role control.
  **Code MIT (public), weights NVIDIA Open Model License** (`nvidia/personaplex-7b-v1`).
  Trained on synthetic + Fisher convs. This is "Track A" — fast demo + builds the RL infra.
- **KAME** (Sakana, `refs/kame/`) — tandem oracle/background-LLM pattern.
- **Ultravox** (Fixie, `refs/ultravox/`) — adapter-only training (Whisper-encoder + projector
  + frozen LLM); the input-adapter intuition.
- **moshi-finetune** (nu-dialogue/Kyutai) — the finetune framework for Track A/B.
- Full OSS inventory: `OSS_LANDSCAPE.md`. Kyutai GRPO post-training recipe (Seamless
  Interaction data, 4 reward axes) = our planned **Stage 4**.

When someone proposes a model/codec, check it against these docs first — the choices
(Mimi, Qwen, Moshi-style dual-stream, hybrid Whisper+Mimi input) are already reasoned out.
CSM/SNAC/mel-vocoder are NOT the path here.

## Training pipeline (code state)

One-command recipes: `scripts/train_recipe.py --recipe poc|full` (`--dry-run` for plan;
`--phase` to resume). Phases: download → encode → stage1 → **gate1** → synth → stage2 →
stage3 → bench.

- **Stage 1** (`stage1_adapter.py`) — audio adapter + depth-transformer on ASR+TTS, backbone frozen.
- **gate1** — measures WER on strictly-separate `librispeech_dev` (exact dir-name match, leak-safe).
  **<15% great, 15-30% usable, >30% STOP** (more Stage-1 data before spending on Stage 2).
- **Stage 2** (`stage2_dualstream.py`) — dual-stream LoRA; model hears its own stream; acoustic-delay.
- **Stage 3** (`stage3_capabilities.py`) — time / barge-in / background / emotion.
- **Stage 4** (planned, infra TODO) — GRPO, Kyutai recipe. `moshi_ft.py` is a stub.

PoC scope: **English only, single voice, ~$100-250 cloud.** Do NOT push production
(multilingual, voice cloning, bigger LLM, emotion depth) into the PoC.

## Workflow / gotchas

- **Setup:** `uv sync` (`--extra train --extra infer --extra eval --extra datagen` as needed).
- **Smoke first, always:** `PYTHONNOUSERSITE=1 .venv/bin/python scripts/smoke_test.py` →
  `ALL SMOKE TESTS PASSED`. If it fails, spend no cloud money. Tests: `uv run pytest tests/ -q`.
- **GPU: rent RTX 3090, NOT 4090** — 4090 secure-cloud pods (EUR-IS-2, image
  `runpod/pytorch:0.7.0-cu1241-torch260`) never got a runtime assigned; 3090 boots fine.
- **Runpod SSH (from Claude):** proxy needs `-tt`, ignores argv → pipe via stdin
  (`printf 'cmd\nexit\n' | ssh -tt -F ~/.ssh/config_runpod runpod`); strip ANSI with
  `sed 's/\x1b\[[0-9;?]*[a-zA-Z]//g'`; PTY echoes input — don't grep for markers that
  are in your command. Loop: `git push` → ssh pod → `git pull` → relaunch in tmux.
- **Web UI on RunPod:** serve via uvicorn + `gr.mount_gradio_app`, NOT `demo.launch()`
  (localhost self-check fails in container → 502). Run with `--with "gradio<5"
  --with "huggingface-hub<1.0"`. Use the HTTP-proxy (port 7860), not `share=True`.
- **Mimi weights:** `kyutai/moshiko-pytorch-bf16/tokenizer-*.safetensors`, NOT `kyutai/mimi/model.safetensors`.
- **Dtype:** audio adapter outputs float32, backbone bf16 → cast embeddings to backbone dtype.
- **DataLoader:** `num_workers=0` (CUDA Mimi.encode forks badly); rglob not glob for nested encoded dirs.
- **NEVER commit secrets** — GitHub secret-scanning rejected a push with an HF token literal.
  Ask the user for literal keys.

## Repos & artifacts

- GitHub: `github.com/chukfinley/voiceai` (public). Branch with grounding work: `lora-grounding-fix`.
- HF (consolidated): `chukfinley/voiceai` (all checkpoints, deduped). Key:
  `chukfinley/voiceai-asr-whisper` (10.5% WER ASR), `chukfinley/voiceai-duplex-demo` (the babbling demo).
- Source layout: `src/voiceai/{orchestrator,foreground,background,visual,training,model,eval,server}/`.
  Key files in `RUNBOOK.md` / `SESSION_HANDOFF.md` file tables.

## License

Code Apache-2.0. Models keep their own licenses (Mimi/Moshi CC-BY-4.0 attribution;
PersonaPlex weights NVIDIA OML — check commercial). Our trained model ships Apache-2.0.
