# Training Run Log — for cost/time extrapolation

Living benchmark log. Goal: measure exact time + cost on this run so we can
predict cost/duration for bigger GPUs, more steps, larger models later.

How to extrapolate later:
- **time/step** scales ~inversely with GPU throughput (bf16 TFLOPS / bandwidth),
  but small models are often dataloader/encode-bound → real speedup < raw ratio.
- **cost = wall_clock_hours × $/hr**. Fewer hours on a faster card can cost the
  same or less even at higher $/hr.
- record actual it/s per GPU here; that's the ground truth for scaling.

---

## Run #1 — RTX 4090, ASR mini (Stage-1)

### Hardware
- GPU: **NVIDIA GeForce RTX 4090**, 24 GB, driver 570.195.03
- Provider: RunPod **SECURE**, **$0.69/hr**
- Image: `runpod/pytorch:0.7.0-cu1241-torch260-ubuntu2004` (cu124; cu128 image failed — machine had CUDA 12.4)
- Pod id: `hlea7kfez7bgjc` · 60 GB disk · 128 vCPU
- Note: 1st attempt cu128 image → "Minimum CUDA version not met (machine 12.4)". Use cu124 images on 4090 secure.

### Config
- Backbone: **Qwen/Qwen3-0.6B** (34.6M trainable adapter / 630M total)
- Data: 12 000 real LibriSpeech clips streamed (16 held out for eval)
- Steps: **20 000**, batch 4, grad-accum 2 (eff. batch 8), lr 3e-4, mix-asr 0.8
- ckpt every 4000 steps · dtype bf16
- Script: `scripts/colab_asr_mini.py --clips 12000 --steps 20000 --push-to-hub`

### Timeline (UTC, 2026-06-11)
| time (UTC) | event |
|------------|-------|
| 03:14:28 | pod rented (RUNNING) |
| 03:18:48 | run.sh start (uv install + clone) |
| 03:19:55 | uv venv built — 116 pkgs (~1 min incl. torch download) |
| ~03:20:55 | data stream done + training step 0 begins (run #1, default repo) |
| 03:22:11 | step 202/20000, loss 15.3→8.0, **~2.9 it/s** |
| 03:28:19 | **clean restart** (run2.sh, setsid) → dedicated repo `chukfinley/voiceai-asr-12k` to avoid collision with the parallel Colab run on `voiceai-asr-mini`. Data cached, no re-download. it/s ~3.1. New step-0 baseline. |
| 05:17:52 | training done — run2 ran 03:28:19→05:17:52 = **1h49m33s** for 20k steps |
| ~05:18 | HF upload done → https://huggingface.co/chukfinley/voiceai-asr-12k |
| 05:23 | pod terminated (DELETE 204) |

### Measurements (FINAL)
- it/s @ 4090: **~3.04 it/s avg** (range 2.5–4.2) @ batch 4 / 0.6B / bf16
- s/step: **0.329 s/step** (6573s / 20000 steps, effective batch 8)
- training wall-clock (20k steps): **1h49m33s**
- total pod wall-clock (rent 03:14:28 → terminate ~05:23): **~2h09m**
- **total cost: ~$1.48** (2.14h × $0.69/hr)
- **held-out WER: 176.9%** ⚠️ (see Result below)

### Extrapolation cheatsheet (from this 4090 data point)
- 4090 @ 0.6B, eff-batch 8: **~0.34 s/step** → 20k steps ≈ 1.9h.
- Scale to N steps: `hours ≈ N × 0.34 / 3600`. Cost ≈ hours × $/hr.
- Bigger backbone roughly linear in params for compute-bound part (0.6B→1.7B ≈ 2.8×
  params; expect ~2–2.5× slower if not dataloader-bound).
- Faster card (H100 ~3–4× 4090 raw) → maybe ~2× real here (small model, partly
  encode/dataloader bound). Confirm by running same script on H100 and logging it/s.

### Notes
- bf16 on 4090 native (unlike T4 which emulates). Compare later vs T4 1.46 s/it on 0.6B (different config: 4000 steps/3000 clips).

## Progress checkpoints (live)
- 03:37:21 UTC — step 1714/20000, ~3.38 it/s, loss 4.02, GPU 8.0GB (ETA ~05:08 UTC)
- 03:52:22 UTC — step 4474/20000 (22%), ~3.07 it/s, loss 5.12 (ckpt@4000 on disk)
- 04:07:18 UTC — step 7115/20000 (36%), ~2.87 it/s, loss 4.69
- 04:22:19 UTC — step 9845/20000 (49%), ~3.34 it/s, loss 4.91
- 04:37:20 UTC — step 12678/20000 (63%), ~3.05 it/s, loss 3.42
- 04:52:20 UTC — step 15454/20000 (77%), ~2.84 it/s, loss 3.32
- 05:07:22 UTC — step 18268/20000 (91%), ~3.1 it/s, loss 3.30 (nearly done)

## RESULT & finding (Run #1)

**Held-out WER 176.9% — the model does NOT transcribe unseen audio.**
On the 16 never-trained clips the HYP is fluent English but unrelated to the
audio (loops like "the wind was so strong and the wind was so strong…", runs of
"iiiii…"). The frozen Qwen language prior dominates; the audio adapter did not
learn strong acoustic grounding at this scale.

Why (hypotheses):
- Frozen backbone + only a 34.6M `audio_in` adapter trainable, prefix-LM setup:
  training loss (→3.3) can drop via LM prior + partial memorization without
  robust audio→text grounding. On unseen audio it falls back to free-generation.
- 12k clips / 20k steps is still tiny for from-scratch audio grounding
  (SLAM-ASR-scale needs ~hundreds of h; the recipe `poc` uses 960h / 60k steps).
- Possibly the adapter/conditioning is too weak, or needs the backbone partially
  unfrozen, or more steps, or CTC/alignment aux loss.

Next experiments to try (cheap, on 4090, measured against this baseline):
1. More data + steps (recipe `poc` scale: 960h, 60k steps) — main lever.
2. Unfreeze top-N backbone layers (or LoRA on attn) so it can adapt to audio.
3. Stronger audio conditioning / longer warmup / lower lr.
4. Sanity: does training-set WER ~0? (yes loss low) → confirms it's a
   generalization/grounding gap, not a code bug.

**Takeaway for cost planning:** one 4090-hour ≈ $0.69 buys ~11k steps @ 0.6B.
A `poc`-scale Stage-1 (60k steps) ≈ ~5.5 GPU-h ≈ ~$3.8 on a 4090 (training only;
+ data download/encode time). That's the next real test of whether grounding
emerges with scale.

## ✅ SOLVED — Whisper-encoder ASR works (2026-06-11)

Root cause (across 3 failed Mimi runs): **prefix-LM + teacher forcing + frozen
LLM** lets the LLM predict each next token from the previous GROUND-TRUTH tokens
(language prior), so the audio is only needed for the FIRST token → it gets
ignored. Mimi acoustic codes made it worse (weak ASR representation). LoRA made
it worse still (more LM-prior memorization).

**Fix = SLAM-ASR:** frozen Whisper-small encoder (features ≈ transcript) → small
trained bridge MLP → frozen Qwen. PLUS the critical detail: **frame stacking**
(concatenate k=5 whisper frames → 1500 sparse frames become 300 dense tokens).
Without stacking: loss plateaus at LM-prior (~4), identical degenerate output for
every clip (audio ignored). WITH stacking: loss breaks through to <0.2.

Run (whisper-small + Qwen3-0.6B, 2000 clips, 3000 steps, batch 8, ~9 min, ~$0.15):
- loss 4 → 0.19
- **TRAIN WER 1.9%**, **HELD-OUT WER 10.5%** (real ASR on unseen clips)
- model: https://huggingface.co/chukfinley/voiceai-asr-whisper (bridge.pt only, ~frozen rest)

Takeaway: only the ~4M bridge is trained; Whisper+Qwen frozen. Cheap, fast, works.
Branch: lora-grounding-fix.

### Polished run (3000 clips, 6000 steps, 24 held-out)
- loss → 0.03 · TRAIN WER 1.4% · **HELD-OUT WER 11.3%**
- ~same held-out as the 3000-step run (10.5%) → converged at ~11% on 3000 clips.
  More STEPS overfit train; to lower held-out WER add more DATA (clips), not steps.
- final model: HF chukfinley/voiceai-asr-whisper + local runs/asr_whisper_final/bridge.pt (9.5MB)
- test it: `uv run --extra train python scripts/transcribe.py <file.wav>`
