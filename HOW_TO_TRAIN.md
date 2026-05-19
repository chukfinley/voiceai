# HOW_TO_TRAIN.md — Concrete commands

End-to-end recipe to take this skeleton to a working voiceai model.

## 0. Smoke test (free, 5 min, CPU or any GPU)

Validates the whole stack with a tiny stand-in model — no real weights downloaded.

```bash
uv sync --extra train --extra dev
PYTHONNOUSERSITE=1 .venv/bin/python scripts/smoke_test.py
```

Expect: `ALL SMOKE TESTS PASSED`. If this fails, do not spend any cloud money.

Note: `PYTHONNOUSERSITE=1` is needed if your `~/.local/` has a stale torch
install. Skip this prefix on a clean machine or in Colab.

If you don't have CUDA installed locally but uv pulled the GPU torch:

```bash
VIRTUAL_ENV=.venv uv pip install nvidia-cuda-runtime-cu12 nvidia-cublas-cu12
```

This adds CUDA libs as pip wheels so torch can at least import on CPU dev boxes.

## 1. Colab smoke (free, 15 min, T4 16GB)

Validates with real Qwen3.5-0.8B + Mimi weights on a real GPU.

1. Open `notebooks/colab_smoke.ipynb` in Colab.
2. Set HF token in cell 4.
3. Run all cells.
4. Last cell should print `text_logits: torch.Size([1, X, ...])` shapes without OOM.

## 2. Download Stage 1 data (free, 1-3h CPU)

```bash
uv run python scripts/download_data.py --out data/stage1 --librispeech --commonvoice --max-hours 200
```

Pulls ~200h of English speech (LibriSpeech + Common Voice) and builds a manifest.
Total disk: ~20-30 GB.

## 3. Stage 1 — Audio Adapter Pretrain

**Local smoke (1 min, CPU):**

```bash
uv run python -m voiceai.training.stage1_adapter \
    --manifest data/stage1/manifest.jsonl \
    --output runs/stage1_smoke \
    --backbone hf-internal-testing/tiny-random-LlamaForCausalLM \
    --smoke --device cpu --dtype float32 --wandb-disable
```

**Real run on RTX 3090 (3 days, $33 Runpod):**

```bash
export WANDB_API_KEY=...
uv run python -m voiceai.training.stage1_adapter \
    --manifest data/stage1/manifest.jsonl \
    --output runs/stage1 \
    --backbone Qwen/Qwen3.5-0.8B \
    --steps 30000 \
    --batch-size 8 \
    --grad-accum 4 \
    --lr 3e-4 \
    --warmup 500 \
    --ckpt-every 2000
```

**On Colab Pro+ A100 (≈40h spread over 2 sessions):**

Save checkpoints to Drive; resume with `--stage1 runs/stage1/step_NNNN` when reconnecting.

## 4. Generate concurrent-commentary data (1-2h, $0, all local OSS)

```bash
uv sync --extra datagen
uv run python scripts/gen_concurrent_commentary.py \
    --out data/concurrent --n 2000 \
    --backend kokoro \
    --encode-mimi
```

TTS backends (all Apache, all local):
  - `kokoro` (default) — 82M, multiple English voices, RTF 0.03
  - `qwen-tts` — Qwen3-TTS, larger but higher quality
  - `melotts` — very fast
  - `gtts` — online fallback, low quality

**Note:** TTS here is **data-generation only**. At runtime our voiceai model
is itself the TTS+STT+LLM. These TTS tools never appear at inference time.

Outputs:
- `data/concurrent/raw/*.wav` — raw audio
- `data/concurrent/encoded/*.npz` — Mimi-encoded dual-stream samples
- `data/concurrent/samples.jsonl` — metadata

## 5. Get Stage 2 paired dialog data

Three options, pick one (or combine):

### Option A — CANDOR Corpus (free, request access)

Apply at https://betterup-data-requests.herokuapp.com/. After approval you get a download script. Place audio under `data/candor/` then convert:

```bash
uv run python scripts/encode_dual_stream.py \
    --in-dir data/candor --out-dir data/stage2_candor
```

### Option B — Fisher Corpus ($200-500, LDC)

Order from https://catalog.ldc.upenn.edu/LDC2004S13. Highest quality channel-separated dialog.

### Option C — Pure synthetic (free, ~$10 in API calls)

```bash
uv run python scripts/gen_synthetic_dialogs.py --out data/stage2_synth --n 5000
```

(synthetic dialog generator script — uses Claude API to write scripts, then our TTS to render two voices alternating)

## 6. Stage 2 — Dual-Stream Conversational SFT

**Smoke (1 min, CPU):**

```bash
uv run python -m voiceai.training.stage2_dualstream \
    --stage1 runs/stage1_smoke/final \
    --data-root data/concurrent/encoded \
    --output runs/stage2_smoke \
    --smoke --device cpu --dtype float32 --wandb-disable
```

**Real run on RTX 3090 (5 days, $55):**

```bash
uv run python -m voiceai.training.stage2_dualstream \
    --stage1 runs/stage1/final \
    --data-root data/stage2_combined \
    --output runs/stage2 \
    --steps 80000 \
    --batch-size 4 \
    --grad-accum 8 \
    --lr 1e-4 \
    --lora-rank 64 \
    --pad-frames 512
```

## 7. Generate Stage 3 capability data

```bash
uv run python scripts/gen_concurrent_commentary.py --out data/cap_cc --n 5000 --encode-mimi
uv run python scripts/gen_time_aware.py --out data/cap_ta --n 5000
# (similar for backchannel, barge-in, background-query, visual)
```

Combine all into `data/stage3_capabilities/encoded/`.

## 8. Stage 3 — Capability Fine-Tune

```bash
uv run python -m voiceai.training.stage3_capabilities \
    --stage2 runs/stage2/final \
    --data-root data/stage3_capabilities/encoded \
    --output runs/stage3 \
    --steps 15000 \
    --batch-size 4 \
    --grad-accum 8 \
    --lr 5e-5
```

RTX 3090, 2 days, $22.

## 9. Eval

```bash
uv run python -m voiceai.eval.fd_bench --model runs/stage3/final
uv run python -m voiceai.eval.timespeak --model runs/stage3/final
uv run python -m voiceai.eval.concurrent_commentary --model runs/stage3/final
```

## 10. Demo

```bash
uv run python scripts/bootstrap.py --option b --voiceai-model runs/stage3/final
```

Mic in, speaker out, full duplex.

---

## Cost summary

| Step | Where | Time | $ |
|------|-------|------|---|
| 0 Smoke local | laptop | 10m | 0 |
| 1 Colab smoke | Colab free | 15m | 0 |
| 2 Download data | laptop | 3h | 0 |
| 3 Stage 1 train | Runpod RTX 3090 | 3 days | $33 |
| 4 CC data | laptop + TTS API | 2h | 3 |
| 5 Stage 2 data | varies | 1-7 days | 0-500 |
| 6 Stage 2 train | Runpod RTX 3090 | 5 days | $55 |
| 7 Stage 3 data | laptop | 4h | 5 |
| 8 Stage 3 train | Runpod RTX 3090 | 2 days | $22 |
| 9 Eval | Runpod RTX 3090 | 4h | 2 |
| **Total (no Fisher)** | | **~2 weeks** | **~$120** |

Budget headroom: $1000 - $120 = $880 for re-runs, ablations, and Stage-2 full SFT if LoRA underperforms.
