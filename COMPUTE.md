# COMPUTE.md — Cheap & Free Training Paths

We can't afford a $15k H100 week. This doc maps every viable cheap/free path
for the PoC and beyond.

## Why training needs GPUs (TL;DR)

Inference = 1× forward pass.
Training = forward + backward + optimizer step. Backward stores activations
for every layer, so memory grows ~3-5×. Full training of a 3B model needs
~60-120GB VRAM.

**LoRA cuts trainable params 1000×.** Only small adapter matrices update;
the base model stays frozen. VRAM stays close to inference.

**QLoRA = LoRA + 4-bit base.** Base model loaded in 4-bit (NF4) — another
4× memory cut. A 3B-param model fits in ~6GB VRAM for training.

| Method | VRAM (3B model) | Quality vs full |
|--------|-----------------|------------------|
| Full SFT | ~60 GB | 100% |
| LoRA bf16 | ~12 GB | ~98% |
| QLoRA 4-bit | ~6 GB | ~95% |
| LoRA + grad-ckpt | ~8 GB | ~98% |

PoC target: QLoRA on a free or near-free GPU.

## Free tier — what actually works

### 1. Kaggle (best free option)
- 2× T4 16GB, 30h/week, 12h session limit
- Persistent /kaggle/working storage
- Internet enabled (download HF weights, push runs)
- **Verdict**: best for PoC. QLoRA Qwen3-0.6B easily fits.

### 2. Google Colab free
- T4 16GB, ~3h sessions, frequent disconnects
- Drive mount for checkpoints
- **Verdict**: backup. Use for prototyping notebooks.

### 3. HF Spaces ZeroGPU
- A100 80GB on demand for short bursts
- Free tier limited to small jobs
- **Verdict**: only for demos / eval, not multi-hour training

### 4. Lightning AI Studio
- ~$25 free credits on signup
- Mix of T4/V100/A100 on-demand
- **Verdict**: stretches to a small LoRA run

### 5. Modal Labs
- $30/month free credit ongoing
- Spot A100 ~$1.10/h
- **Verdict**: production-grade serverless GPU

### 6. Vast.ai
- Spot market RTX 3090 24GB: **$0.20-0.30/h**
- Spot A100 40GB: ~$0.50-0.80/h
- **Verdict**: cheapest per VRAM, but instances can vanish

### 7. Runpod Community
- RTX 4090: $0.34/h
- A100 80GB: ~$1.19/h
- **Verdict**: reliable spot tier

### 8. SaladCloud
- Consumer RTX 3090/4090 distributed
- $0.10-0.30/h
- **Verdict**: very cheap, queues during peak

### 9. Lambda Cloud reserved
- H100: $2-2.50/h
- A100 80GB: $1.10/h
- **Verdict**: full-SFT later when needed

### 10. Free startup credits (apply)
- AWS Activate: up to $5k–$100k
- Google for Startups: up to $200k
- Azure for Startups: up to $150k
- NVIDIA Inception: discounted DGX
- **Verdict**: worth applying — kept for later phases

## Concrete training paths for our PoC

### Phase 0: Cascade — no training needed
Just use off-the-shelf Qwen3-ASR + Qwen3-0.6B + Qwen3-TTS. **$0.**

### Phase 1: LoRA on Qwen3-0.6B (time/visual/barge tokens)
- Backbone: Qwen3-0.6B-Instruct (584M params)
- 4-bit QLoRA, rank 32
- 10k synthetic samples
- **Hardware**: Kaggle T4 16GB
- **Time**: ~4-6 hours
- **Cost**: **$0**

### Phase 2: LoRA on Qwen3-Omni-30B-A3B Thinker
- Trainable: ~80M LoRA params (Thinker attention)
- 4-bit QLoRA, rank 32
- Active params 3B → ~22GB peak with grad-ckpt
- **Hardware**: 1× RTX 4090 24GB or A100 40GB
- **Time**: 24-48h for 50k samples
- **Cost**: $20-50 on Vast.ai spot RTX 4090

### Phase 3: Full SFT on Thinker only (Talker frozen)
- 3B trainable
- FSDP across 4-8 GPUs
- **Hardware**: 4-8× A100 80GB or 4× H100
- **Time**: 5-7 days for full dataset
- **Cost**: $2k-15k on Lambda/Runpod
- **When**: after Phase 1+2 prove product-market fit

## Tricks to reduce cost further

1. **Use bfloat16** if hardware supports — no quality loss vs fp32.
2. **Gradient checkpointing** — trades ~30% speed for ~50% VRAM.
3. **Smaller LoRA rank** — rank 8 is 4× cheaper than rank 32, often within 2%.
4. **Flash Attention 2** — same memory, 2-3× faster.
5. **Liger Kernel** — drop-in 20-40% speedup for Llama/Qwen.
6. **Sequence packing** — pack short samples to max length, no wasted compute.
7. **DoRA / VeRA** — newer LoRA variants, smaller still.
8. **Train on subsampled data first** — 1k samples to validate, then scale.
9. **Distillation from a free API** — use Qwen3-Max API to generate teacher
   data, then student is small. Costs are inference-side ($/M tokens) not GPU-hours.

## Recommended starting point

```bash
# On Kaggle T4 — fully free
uv run python -m voiceai.training.lora_sft \
    --backbone Qwen/Qwen3-0.6B-Instruct \
    --data data/interaction_sft.jsonl \
    --output /kaggle/working/lora_v1 \
    --quant 4bit \
    --lora-rank 16 \
    --batch-size 1 \
    --grad-accum 16 \
    --max-seq-len 2048
```

That should fit comfortably in 12GB VRAM, finish in ~4h, cost **$0**.
