# RUNBOOK.md — Today's launch sequence

What to do **right now** to launch the first training run this afternoon.

## TWO-PHASE FLOW (saves ~50% GPU cost)

If you have a CPU-only server with lots of RAM that can run for days,
use the CPU/GPU split:

**PHASE 1 (CPU server, days, $0):** all data prep
**PHASE 2 (GPU rental, ~10 days, ~$110):** only training

See "CPU-server data prep" section below for the full command.

## What's already done

- ✅ Skeleton + tests (9 passing, smoke OK)
- ✅ Mimi codec integration
- ✅ VoiceAILM with Qwen3.5-0.8B backbone (873M, Apache 2.0)
- ✅ All three training stages coded
- ✅ **Ten data generators**:
  - general_dialog (turn-taking baseline)
  - diverse_dialogs (LLM-generated across 33 scenarios) ⭐
  - concurrent_commentary (panda counting)
  - backchannel (mhm/yeah during user speech)
  - time_aware_audio (wait/self-init)
  - barge_in (user interrupts)
  - rapid_qa (multi-speaker QA)
  - sound_recognition (ESC-50 non-speech)
  - constraints (only-X behaviors)
  - time_limited (30s session cap)
- ✅ **Unified OpenAI-compat background bridge** (supports OpenAI/Anthropic/DashScope/Gemini/vLLM/Ollama/LMStudio + streaming)
- ✅ **HF dataset downloader** (LibriSpeech, CV, SpokenWOZ, IntrinsicVoice, etc.)
- ✅ Eval harness (ASR WER, TimeSpeak, ConcurrentCommentary)
- ✅ One-command launchers for Stage 1 + Stage 2
- ✅ **WebSocket inference server** (Moshi-style browser UI)
- ✅ GitHub Actions CI

## Step-by-step

### 1. Local verification (you, 5 minutes, $0)

```bash
cd /home/user/git/voiceai
PYTHONNOUSERSITE=1 .venv/bin/python -m pytest tests/ -q -m "not slow"
PYTHONNOUSERSITE=1 .venv/bin/python scripts/smoke_test.py
```

Expect: `9 passed`, `ALL SMOKE TESTS PASSED`.

### 2. Push to GitHub (you, 2 minutes)

```bash
git init && git add -A
git commit -m "voiceai: full skeleton, all stages, all generators"
gh repo create voiceai --private --source=. --push
```

CI will smoke-test on every push.

### 3. Get HF token + accept Qwen license (you, 5 minutes)

1. Sign in at https://huggingface.co
2. Settings → Access Tokens → create token (Read)
3. Visit https://huggingface.co/Qwen/Qwen3.5-0.8B → accept license
4. Visit https://huggingface.co/kyutai/mimi → accept license

```bash
export HF_TOKEN=hf_...
```

### 4. Get wandb token (you, 2 minutes, free)

1. Sign up at https://wandb.ai (free tier is fine)
2. Settings → API keys → copy

```bash
export WANDB_API_KEY=...
```

### 5. Rent Runpod RTX 3090 (you, 5 minutes, $0.46/h)

1. Sign in at https://runpod.io
2. Deploy → Community Cloud → RTX 3090 24GB
3. Template: PyTorch 2.5 / CUDA 12.4
4. Disk: 100 GB persistent
5. SSH into instance

### 6. Setup on Runpod (10 minutes)

```bash
git clone https://github.com/YOUR_USER/voiceai
cd voiceai
curl -LsSf https://astral.sh/uv/install.sh | sh
~/.local/bin/uv sync --extra train --extra dev --extra datagen
export HF_TOKEN=hf_...
export WANDB_API_KEY=...
huggingface-cli login --token $HF_TOKEN

# Verify smoke still passes on this box
uv run python scripts/smoke_test.py
```

Expect: `ALL SMOKE TESTS PASSED`. **If this fails on Runpod, do not start training.**

### 7. Launch Stage 1 — Audio Adapter Pretrain

This is THE FIRST REAL TRAINING RUN. ~3 days, ~$33.

```bash
uv run python scripts/launch_stage1.py \
    --output runs/stage1 \
    --steps 30000 \
    --hours 100
```

What it does:
1. Checks GPU + env
2. Re-runs smoke test
3. Downloads ~100h of LibriSpeech + CommonVoice English (~30 GB, ~30 min)
4. Launches training, tagged in wandb

Monitor at https://wandb.ai/YOUR_USER/voiceai.

Expected loss curve: starts ~9-12, drops below 4 by step 5000, below 2 by step 30000.

If loss > 6 after step 5000 — stop and inspect; something is wrong.

### 8. While Stage 1 trains: prepare Stage 2 data

In a second SSH session on the same instance (no extra GPU cost since data gen
uses spare CPU/GPU):

```bash
uv run python scripts/launch_stage2.py \
    --stage1 runs/stage1/final \
    --skip-data=false \
    --mult 0.3 \
    --steps 100   # tiny smoke run first
```

`--mult 0.3` = generate ~30% of full dataset to start (so we have something to
train Stage 2 on quickly). Total samples: ~900 across all categories. Takes ~3h
on the RTX 3090's spare cycles.

### 9. When Stage 1 finishes (3 days)

```bash
# Verify Stage 1 quality
uv run python -m voiceai.eval.asr_quality \
    --model runs/stage1/final \
    --manifest data/stage1/manifest.jsonl \
    --n 100

# Launch real Stage 2
uv run python scripts/launch_stage2.py \
    --stage1 runs/stage1/final \
    --skip-data \
    --steps 80000
```

### 10. Stage 3 — capabilities

After Stage 2 finishes (5 days):

```bash
uv run python -m voiceai.training.stage3_capabilities \
    --stage2 runs/stage2/final \
    --data-root data/stage2/combined \
    --output runs/stage3 \
    --steps 15000
```

### 11. Final eval

```bash
uv run python -m voiceai.eval.concurrent_commentary \
    --model runs/stage3/final \
    --data data/stage2/concurrent
uv run python -m voiceai.eval.timespeak \
    --model runs/stage3/final \
    --data data/stage2/time_aware
```

## Budget tracking

| Step | $ spent | $ remaining (of $1000) |
|------|---------|------------------------|
| 1-6 setup | 0 | 1000 |
| 7 Stage 1 train (3 days @ $0.46/h) | 33 | 967 |
| 8 Stage 2 data gen (cheap GPU work) | ~5 | 962 |
| 9 Stage 2 train (5 days) | 55 | 907 |
| 10 Stage 3 train (2 days) | 22 | 885 |
| 11 Eval | 2 | 883 |
| **Total** | **$117** | **$883 reserve** |

## CPU-server data prep (recommended)

If you have a weak-CPU but big-RAM server you can leave running for days:

```bash
# Setup once
git clone <your_repo>; cd voiceai
curl -LsSf https://astral.sh/uv/install.sh | sh
~/.local/bin/uv sync --extra train --extra datagen --extra dev
VIRTUAL_ENV=.venv uv pip install --reinstall torch torchaudio \
    --index-url https://download.pytorch.org/whl/cpu

# Set API key for diverse-dialog generation
export OPENAI_API_KEY=sk-...      # or DASHSCOPE_API_KEY / ANTHROPIC_API_KEY

# Smoke test (5 min)
PYTHONNOUSERSITE=1 .venv/bin/python scripts/smoke_test.py

# Kick off full prep in background (runs for ~2 days on weak CPU)
nohup .venv/bin/python scripts/prep_data_cpu.py \
    --out data/full \
    --provider openai \
    --diverse-n 3000 \
    --tts-backend kokoro \
    --encode-mimi \
    > prep.log 2>&1 &

# Monitor
tail -f prep.log
```

The script is **resumable** — kill and restart any time. Each step writes
a `.NN_step.done` marker so it skips finished work.

What it produces:
- `data/full/hf/` — LibriSpeech + Common Voice audio for Stage 1
- `data/full/hf/adapter_manifest.jsonl` — combined Stage 1 manifest
- `data/full/<category>/encoded/*.npz` — Mimi-encoded dual-stream samples
- `data/full/stage2_combined/` — symlinks combining all encoded into one dir

Estimated runtime on a weak server (~4 cores, slow CPU):
- HF download:                ~2-4 h  (network bound)
- TTS for all generators:     ~24-48 h
- Mimi encoding (CPU):        ~12-24 h
- Diverse dialogs API + TTS:  ~6-12 h
- **Total:** ~2-4 days

When done, rsync `data/full/` to your Runpod RTX 3090 and start training
WITHOUT paying for data-prep GPU hours.

```bash
# On runpod after rsync
uv run python -m voiceai.training.stage1_adapter \
    --manifest data/full/hf/adapter_manifest.jsonl \
    --output runs/stage1 --steps 30000

uv run python -m voiceai.training.stage2_dualstream \
    --stage1 runs/stage1/final \
    --data-root data/full/stage2_combined \
    --output runs/stage2 --steps 80000
```

Cost: only $33+$55+$22 = **$110 GPU**, vs $130-200 with on-GPU prep.

## Troubleshooting

**Smoke test fails on Runpod:**
- Check `nvidia-smi` shows the GPU
- Check `PYTHONNOUSERSITE=1` prefix is set
- Check uv venv created (.venv/ in repo root)

**Stage 1 loss flat:**
- Increase LR to 5e-4
- Verify manifest entries are valid (not all silence/empty)

**OOM on RTX 3090:**
- Reduce `--batch-size 4` (from default 8)
- Increase `--grad-accum 8` (from default 4)
- Add `--load-in-4bit` to use QLoRA

**Mimi download fails:**
- Accept license at https://huggingface.co/kyutai/mimi
- `huggingface-cli login`

**wandb not logging:**
- Check `WANDB_API_KEY` set
- Or add `--wandb-disable` to skip tracking

## Files reference

| File | What |
|------|------|
| `src/voiceai/model/voiceai_lm.py` | Main model: Qwen3.5-0.8B + Mimi heads + dual-stream |
| `src/voiceai/training/stage1_adapter.py` | Stage 1: ASR+TTS adapter pretrain |
| `src/voiceai/training/stage2_dualstream.py` | Stage 2: dual-stream conversational LoRA |
| `src/voiceai/training/stage3_capabilities.py` | Stage 3: capability fine-tune |
| `scripts/launch_stage1.py` | One-shot Stage 1 launcher |
| `scripts/launch_stage2.py` | One-shot Stage 2 launcher (incl. data gen) |
| `scripts/download_hf_datasets.py` | HF dataset puller (LibriSpeech, CV, SpokenWOZ, ...) |
| `scripts/gen_diverse_dialogs.py` | LLM-generated dialog scripts → TTS → Mimi |
| `scripts/gen_*.py` | 10 capability-specific generators |
| `src/voiceai/eval/*.py` | FD-bench / TimeSpeak / CC evals |
| `src/voiceai/background/openai_compat.py` | Single OpenAI-compat bridge (any provider) |
| `src/voiceai/server/app.py` | FastAPI WebSocket inference server |

## Quick reference: API calls for diverse-dialog gen

```bash
# OpenAI
export OPENAI_API_KEY=sk-...
uv run python scripts/gen_diverse_dialogs.py --out data/diverse --n 5000 --provider openai --model gpt-4.1-mini

# Anthropic (also OpenAI-compat now)
export ANTHROPIC_API_KEY=sk-ant-...
uv run python scripts/gen_diverse_dialogs.py --out data/diverse --n 5000 --provider anthropic --model claude-opus-4-7

# Cheap: DashScope Qwen-Max
export DASHSCOPE_API_KEY=sk-...
uv run python scripts/gen_diverse_dialogs.py --out data/diverse --n 5000 --provider dashscope

# Free local: Ollama
ollama serve &
ollama pull qwen3.5:0.8b
uv run python scripts/gen_diverse_dialogs.py --out data/diverse --n 5000 --provider ollama
```

## Quick reference: inference server

```bash
uv sync --extra infer
uv run python -m voiceai.server.app --model runs/stage3/final
# open http://YOUR_HOST:8765 in browser, click "start"
```
