# SESSION_HANDOFF.md — Full state for next Claude session

Project: **voiceai** — open-source competitor to Thinking Machines Interaction Model.

Last updated: long initial session, lots of bug-fix-launch-fail cycles, Stage 1 still
trying to start. Latest fix pushed but not yet tested. See "WHERE WE ARE RIGHT NOW".

---

## tl;dr

End-to-end speech LLM. Qwen3-1.7B backbone (originally targeted Qwen3.5-0.8B but
transformers doesn't support it yet) + Kyutai Mimi codec + dual-stream wrapper.

Pod is rented and running. Code on github. Stage 1 data extracted. Training
keeps failing on dtype/CUDA-context issues — most recent fix is for dtype mismatch
between audio adapter (float32) and backbone (bf16).

## WHERE WE ARE RIGHT NOW

1. Pod = Runpod RTX 3090, $0.46/h, SSH as `l840jy1v2aiwh1-64410b32@ssh.runpod.io`
2. Stage 1 data extracted to `/workspace/voiceai/data/stage1/`:
   - 7908 LibriSpeech-clean-100 .flac files (~26h)
   - `manifest.jsonl` with text+duration per file
3. Stage 1 training was launched 4 times so far, each crashed with different bug.
   **Latest fix just pushed (see below) — not yet tested.**
4. Test loop pattern: `git push` → ssh pod → `git pull` → relaunch train in tmux

## Pod info

- **Provider:** Runpod
- **GPU:** RTX 3090 (24GB)
- **CPU:** allocated 8 vCPU on AMD Threadripper host, 46-62GB RAM
- **Cost:** $0.46/h GPU + storage = ~$0.71/h with 200GB persistent
- **Image:** Runpod Pytorch 2.8.0 (CUDA 12.8, torch 2.8, Ubuntu 24.04)
- **Persistent storage:** 200 GB at `/workspace`
- **SSH proxy:** `ssh l840jy1v2aiwh1-64410b32@ssh.runpod.io -i ~/.ssh/id_ed25519`
- **SSH direct TCP:** `ssh root@213.192.2.90 -p 40896 -i ~/.ssh/id_ed25519` (from user's machine only — firewalled from Claude container)

## GitHub

**https://github.com/chukfinley/voiceai** (public, user is `chukfinley`)
- Branch: master
- Pod clones via HTTPS (no auth needed)
- Local user's gh CLI is auth'd as chukfinley
- **Don't commit secrets** — GitHub secret scanning rejected one push that had HF_TOKEN literal

## SSH from Claude — gotchas (CRITICAL)

Runpod proxy SSH (`ssh.runpod.io`):
- **Requires PTY** (`-tt` flag mandatory) — without it: `Error: Your SSH client doesn't support PTY`
- **Ignores command argv** — drops to interactive shell, args go to /dev/null
- **Workaround**: pipe commands via stdin: `printf 'cmd1\ncmd2\nexit\n' | ssh -tt ...`
- **Output capture**: works, but contains ANSI escape sequences from PTY. Clean with `sed 's/\x1b\[[0-9;?]*[a-zA-Z]//g'`
- **For long scripts**: base64-encode locally, decode on remote:
  ```bash
  B64=$(base64 -w0 script.sh)
  printf 'echo %s | base64 -d > /tmp/s.sh && bash /tmp/s.sh\nexit\n' "$B64" | ssh -tt ...
  ```

ControlMaster (SSH multiplexing) at `~/.ssh/config_runpod`:
```
Host runpod
    HostName ssh.runpod.io
    User l840jy1v2aiwh1-64410b32
    IdentityFile ~/.ssh/id_ed25519
    ControlMaster auto
    ControlPath ~/.ssh/cm/%r@%h:%p
    ControlPersist 30m
    RequestTTY yes
```
Usage: `printf 'cmd\nexit\n' | ssh -tt -F ~/.ssh/config_runpod runpod`. ~1.5s per call.

## Polling gotcha (PTY echo)

Runpod's PTY echoes back your input. If you grep ssh output for a marker that's
IN the command, grep matches the echoed input.

**Wrong:**
```bash
until ssh -tt ... 'test -f /file && echo READY' | grep -q READY; do ...
```
This always matches "READY" from echoed `echo READY`.

**Right:**
```bash
until ssh -tt ... 'ls /file' 2>/dev/null | grep -q "/file$"; do ...
```
Match on actual file path output, or string that's specific to real result.

Even better — use grep for stuff that ONLY appears on actual outcome:
```bash
| grep -qE "loss=[0-9.]+|RuntimeError|Traceback"
```

## What's running NOW (pod state)

```
/workspace/voiceai/                                  # cloned repo
├── data/stage1/
│   ├── librispeech/ls_NNNNNNN.flac (7908 files)
│   └── manifest.jsonl  (7908 lines)
├── .venv/                                           # uv venv ~12GB
└── ...

/workspace/.cache/huggingface/
├── hub/datasets--openslr--librispeech_asr/...       # 14 parquet shards (~6GB)
└── hub/models--Qwen--Qwen3-1.7B/...                 # ~3.4GB
└── hub/models--kyutai--moshiko-pytorch-bf16/...     # Mimi weights ~150MB

/workspace/train.log                                 # latest train output
/workspace/dl.log                                    # extract log (done)
```

Tmux sessions:
- `train` — should have Stage 1 process if last relaunch worked
- `dl` — completed (extract done)

Active processes: check with `printf 'ps -ef | grep stage1_adapter | grep -v grep\n' | ssh -tt -F ~/.ssh/config_runpod runpod`

## All bugs hit + fixes (in chronological order)

1. **`hf_transfer` env var set but package missing** → `uv pip install hf_transfer`
2. **`torchcodec` requires libavutil.so.56, Ubuntu has .58** → bypass torchcodec, use `Audio(decode=False)` + soundfile direct decode
3. **Mimi `get_mimi("kyutai/mimi")` interpreted as filepath** → use `hf_hub_download` first
4. **`load_dataset(num_proc=8)` hangs on initial config** → switched to `snapshot_download` + direct parquet read
5. **fast_download.py crashes silently mid-extract** → rewrote `extract_librispeech.py`:
   - Raw FLAC bytes copy (no decode/encode, 30x faster)
   - Batch read columns to_pylist per shard
   - Incremental manifest writes
   - Resumable
   - Verbose progress
6. **`qwen3_5` model_type unknown in transformers 4.57** → upgraded to git main (`5.8.0.dev0`)
7. **Qwen3.5-0.8B STILL not in transformers 5.8.0.dev** → switched to **Qwen3-1.7B** (regular Qwen3, fully supported)
8. **`PYTHONNOUSERSITE=1` needed locally for tests** (no impact on pod)
9. **GitHub push rejected — HF token literal in markdown** → redacted to `hf_***REDACTED***`
10. **Mimi `state_dict` mismatch** → wrong checkpoint file. Use `kyutai/moshiko-pytorch-bf16/tokenizer-e351c8d8-checkpoint125.safetensors` NOT `kyutai/mimi/model.safetensors`
11. **DataLoader workers fork + CUDA Mimi.encode** → `num_workers=0` (Mimi encode in main process, slower but stable)
12. **`Input type (float) and bias type (BFloat16)` in Mimi conv** → cast audio tensor to mimi dtype in asr_tts.py
13. **`mat1 mat2 dtype mismatch` in Qwen attention** → AudioAdapter outputs float32, backbone bf16. Fix: cast embeddings to `bb_dtype` (text_embed_layer.weight.dtype) before passing to backbone. **This is the latest fix.**

## Stage 1 launch command (current)

```bash
tmux new-session -d -s train "cd /workspace/voiceai && HF_HUB_ENABLE_HF_TRANSFER=1 uv run python -m voiceai.training.stage1_adapter \
  --manifest data/stage1/manifest.jsonl \
  --output runs/stage1 \
  --backbone Qwen/Qwen3-1.7B \
  --steps 30000 \
  --batch-size 4 \
  --grad-accum 8 \
  --lr 3e-4 \
  --warmup 500 \
  --ckpt-every 2000 \
  --wandb-project voiceai \
  > /workspace/train.log 2>&1"
```

Expected behavior if all fixes work:
1. Load model (~5s) — see "Loading checkpoint shards"
2. Load Mimi (~3s)
3. wandb init (~2s) — see "Tracking run with wandb version..."
4. Start data loader (Mimi encode first batch ~10s)
5. **First loss line should appear: `loss=N.NNN`**
6. tqdm bar progresses through 30000 steps

Each step: ~1-2s on RTX 3090 with batch=4 grad-accum=8 = 30000 steps × 1.5s = ~12h
With num_workers=0 it's a bit slower — maybe 1-2 days.

## Token env vars (already set in /root/.bashrc on pod)

- `HF_TOKEN` — chukfinley's HF token (set already, expires never)
- `WANDB_API_KEY` — wandb token
- `HF_HUB_ENABLE_HF_TRANSFER=1` — pre-set in Runpod image
- **NOT YET:** OpenAI/Anthropic/DashScope key for Stage 2 diverse-dialog gen

For literal values, ask the user. NEVER commit them.

## Next steps (priority order)

### 1. Verify latest fix works

```bash
printf 'cd /workspace/voiceai && tmux kill-session -t train 2>/dev/null; git pull -q\ntmux new-session -d -s train "cd /workspace/voiceai && HF_HUB_ENABLE_HF_TRANSFER=1 uv run python -m voiceai.training.stage1_adapter --manifest data/stage1/manifest.jsonl --output runs/stage1 --backbone Qwen/Qwen3-1.7B --steps 30000 --batch-size 4 --grad-accum 8 --lr 3e-4 --warmup 500 --ckpt-every 2000 --wandb-project voiceai > /workspace/train.log 2>&1"\nexit\n' | ssh -tt -F ~/.ssh/config_runpod runpod
```

Wait 2-3 min, then check:
```bash
printf 'tail -15 /workspace/train.log\nnvidia-smi --query-gpu=utilization.gpu,memory.used --format=csv,noheader\nexit\n' | ssh -tt -F ~/.ssh/config_runpod runpod
```

Look for either `loss=N.N` (good) or `Traceback` (next bug to fix).

### 2. If next bug

Common remaining suspects:
- TTS path (text → audio) has similar dtype mismatch (line ~152 in stage1_adapter.py also uses model forward with audio)
- Mimi heads dtype (may need to match too)
- Resampling sample_rate mismatch (LibriSpeech is 16kHz, Mimi wants 24kHz)

### 3. If training runs

- Let it cook for ~12-24h
- Check wandb dashboard regularly: https://wandb.ai/chukfinley2-chuk-development/voiceai
- After 5000 steps: loss should be below 6
- After 15000: below 4
- After 30000: below 2 ideal

### 4. After Stage 1

- Run eval: `uv run python -m voiceai.eval.asr_quality --model runs/stage1/final --manifest data/stage1/manifest.jsonl --n 100`
- WER target: <30% for Stage 2 to be worthwhile
- If WER good: launch Stage 2 (dual-stream training on synth data)

## Key files

```
src/voiceai/
├── model/
│   ├── voiceai_lm.py        # ⭐ updated with bb_dtype casts (latest fix)
│   ├── audio_adapter.py
│   └── mimi_utils.py        # ⭐ uses correct kyutai/moshiko-pytorch-bf16 repo
├── training/
│   ├── stage1_adapter.py    # ⭐ num_workers=0
│   ├── stage2_dualstream.py
│   ├── stage3_capabilities.py
│   └── data/
│       ├── asr_tts.py       # ⭐ casts audio to mimi dtype
│       ├── dual_stream.py
│       ├── format.py
│       ├── mixing.py
│       ├── tts_util.py
│       └── scenario_catalog.py
├── background/openai_compat.py   # unified OpenAI-compat bridge
├── eval/                    # FD-bench, TimeSpeak etc.
└── server/app.py            # FastAPI+WebSocket browser demo

scripts/
├── extract_librispeech.py   # ⭐ current data extractor (DONE)
├── fast_download.py
├── download_data.py
├── download_hf_datasets.py
├── gen_diverse_dialogs.py   # uses LLM API for dialog scripts
├── gen_general_dialog.py
├── gen_concurrent_commentary.py
├── gen_backchannel.py
├── gen_time_aware_audio.py
├── gen_barge_in.py
├── gen_rapid_qa.py
├── gen_sound_recognition.py
├── gen_constraints.py
├── gen_time_limited.py
├── launch_stage1.py
├── launch_stage2.py
├── prep_data_cpu.py
├── encode_all_mimi.py
└── smoke_test.py            # local CPU validation (passes)
```

## Quick commands cheatsheet

```bash
# Status check (~1.5s)
printf 'tmux ls; ps -ef | grep -E "python.*stage" | grep -v grep | head; tail -5 /workspace/train.log; nvidia-smi --query-gpu=utilization.gpu,memory.used --format=csv,noheader\nexit\n' | ssh -tt -F ~/.ssh/config_runpod runpod | sed 's/\x1b\[[0-9;?]*[a-zA-Z]//g'

# Kill + restart train
printf 'tmux kill-session -t train 2>/dev/null; pkill -9 -f stage1_adapter 2>/dev/null; cd /workspace/voiceai && git pull -q\ntmux new-session -d -s train "cd /workspace/voiceai && HF_HUB_ENABLE_HF_TRANSFER=1 uv run python -m voiceai.training.stage1_adapter --manifest data/stage1/manifest.jsonl --output runs/stage1 --backbone Qwen/Qwen3-1.7B --steps 30000 --batch-size 4 --grad-accum 8 --lr 3e-4 --warmup 500 --ckpt-every 2000 --wandb-project voiceai > /workspace/train.log 2>&1"\nexit\n' | ssh -tt -F ~/.ssh/config_runpod runpod

# Read full error trace
printf 'cat /workspace/train.log | tail -80\nexit\n' | ssh -tt -F ~/.ssh/config_runpod runpod | sed 's/\x1b\[[0-9;?]*[a-zA-Z]//g'

# Attach to running training (interactive — for user)
ssh -t l840jy1v2aiwh1-64410b32@ssh.runpod.io -i ~/.ssh/id_ed25519
# then: tmux attach -t train  (detach with Ctrl-B then D)

# Push latest code from user's local machine
cd /home/user/git/voiceai
git add -A
git -c user.email=claude@chuk.dev -c user.name=chukfinley commit -m "..." -q
git push -q
```

## Budget tracker

- Spent so far: ~$1.50-2 (pod has been running ~3-4 hours during setup/debug)
- Stage 1 training est.: $15-30
- Stage 2/3 est.: $50-100
- Remaining of $1000: ~$870+

## Tasks state

```
#35 ✅ SSH into Runpod + setup pod
#36 ✅ Run data prep on pod
#37 🟡 Launch Stage 1 training — keeps failing on dtype bugs, latest fix being tested
#38 ✅ Write SESSION_HANDOFF.md (this file)
```

## Backbone alternatives if Qwen3-1.7B has issues

Best alternatives (all Apache 2.0, in transformers):
- **Qwen3-0.6B** — smaller, faster, similar architecture (`Qwen/Qwen3-0.6B`)
- **Qwen3-4B** — bigger but matches Qwen2.5-72B on bench (`Qwen/Qwen3-4B`)
- **Llama-3.2-3B** — alternative architecture (`meta-llama/Llama-3.2-3B`)
- **Granite-3.0-2B** — IBM alternative (`ibm-granite/granite-3.0-2b-base`)

Qwen3.5-0.8B / Qwen3.6-* are NEW and not yet in transformers 5.8.dev (as of 2026-05-19).
