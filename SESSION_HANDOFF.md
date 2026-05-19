# SESSION_HANDOFF.md — Pickup notes for next Claude session

Last updated: end of long initial session, pod running, Stage 1 data extraction underway.

## tl;dr

We're building **voiceai** — an open-source end-to-end speech LLM (clone of Thinking Machines Interaction Model). Qwen3.5-0.8B + Mimi codec + dual-stream wrapper. Pod is rented and running. Code is on github. Stage 1 data extraction is the current step.

## Pod info

- **Provider:** Runpod
- **GPU:** RTX 3090 (24GB)
- **CPU:** AMD Threadripper 24-core, 62GB RAM
- **Cost:** $0.46/h GPU + $0.005/h network volume = ~$0.71/h with 200GB persistent
- **Image:** Runpod Pytorch 2.8.0 (CUDA 12.8, torch 2.8, Ubuntu 24.04)
- **Persistent storage:** 200 GB at `/workspace`
- **SSH proxy:** `ssh l840jy1v2aiwh1-64410b32@ssh.runpod.io -i ~/.ssh/id_ed25519`
- **SSH direct TCP:** `ssh root@213.192.2.90 -p 40896 -i ~/.ssh/id_ed25519` (timed out from previous Claude container — firewall — but works from user's machine for SCP/SFTP)

## SSH from Claude — gotchas

Runpod proxy SSH (`ssh.runpod.io`):
- **Requires PTY** (`-tt` flag mandatory) — without it: `Error: Your SSH client doesn't support PTY`
- **Ignores command argv** — drops to interactive shell, command goes to /dev/null
- **Workaround**: pipe commands via stdin: `printf 'cmd1\ncmd2\nexit\n' | ssh -tt ...`
- **Output capture**: works, but contains ANSI escape sequences from PTY (use `sed 's/\x1b\[[0-9;?]*[a-zA-Z]//g'` to clean)
- **For long scripts**: base64-encode locally, decode on remote: `B64=$(base64 -w0 script.sh); printf 'echo %s | base64 -d > /tmp/s.sh && bash /tmp/s.sh\nexit\n' "$B64" | ssh -tt ...`

ControlMaster (SSH multiplexing) is set up at `~/.ssh/config_runpod`:
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

Usage: `printf 'cmd\nexit\n' | ssh -tt -F ~/.ssh/config_runpod runpod`. Reuses connection = ~1.5s per call instead of 5-10s.

## Polling gotcha (BUG to avoid)

Runpod's PTY **echoes back** the input you send. If you grep ssh output for a marker that's IN the command, grep matches the echoed input, **not** the actual remote response.

**Wrong:**
```bash
until ssh -tt ... 'test -f /file && echo READY' | grep -q READY; do ...
```
This always matches "READY" from the echoed `echo READY` text.

**Right:**
```bash
until ssh -tt ... 'ls /file' 2>/dev/null | grep -q "/file$"; do ...
```
Match on actual file path presence in `ls` output.

## What's running NOW

```
tmux on pod:
  dl  → cd /workspace/voiceai && HF_HUB_ENABLE_HF_TRANSFER=1 \
        uv run python scripts/extract_librispeech.py \
        --out data/stage1 --max-hours 100 \
        > /workspace/dl.log 2>&1
```

Background bash on local (Claude controller):
- Polling task `blaghb4ob` waiting for "[DONE] N files" line in dl.log

When extract finishes, manifest will be at `/workspace/voiceai/data/stage1/manifest.jsonl`.

## GitHub repo

**https://github.com/chukfinley/voiceai** (public)

- master branch
- Git user: chukfinley (already authenticated on user's machine)
- Pod git clone via HTTPS (no auth needed since public)

## Key tokens (set as env vars on pod)

- `HF_TOKEN=hf_***REDACTED***` — already set in pod `/root/.bashrc` (ask user for value)
- `WANDB_API_KEY=wandb_v1_***REDACTED***` — already set in pod `/root/.bashrc` (ask user)
- `HF_HUB_ENABLE_HF_TRANSFER=1` — pre-set in pod env (must install hf_transfer pkg)
- **NOT YET:** OpenAI/Anthropic key for diverse-dialog generation

## Bugs hit + fixes already applied

1. **`hf_transfer` missing**: `HF_HUB_ENABLE_HF_TRANSFER=1` was set in pod env but package missing. Fixed with `uv pip install hf_transfer`.

2. **`torchcodec` libavutil.so.56 mismatch**: HF datasets default audio decoder needs ffmpeg 4.x; Ubuntu 24.04 has ffmpeg 6.x (libavutil.so.58). Fixed by bypassing torchcodec: use `Audio(decode=False)` + decode raw bytes via `soundfile`.

3. **Mimi loader passed "kyutai/mimi" string as filepath**: `moshi.models.loaders.get_mimi(model_path)` expects local file path, not HF repo. Fixed in `src/voiceai/model/mimi_utils.py` to use `huggingface_hub.hf_hub_download` first.

4. **`load_dataset(num_proc=8)` hangs on initial config resolve**: never produced output. Switched to `huggingface_hub.snapshot_download` + direct parquet read (faster, more reliable).

5. **fast_download.py crashed silently mid-extract**: rewrote as `extract_librispeech.py` with:
   - Raw FLAC bytes copy (no decode/encode = ~30x faster)
   - Batch-read parquet columns (one `.to_pylist()` per shard, not per row)
   - Incremental manifest writes
   - Resumable
   - Verbose progress every 500 files

6. **`qwen3_5` model type unknown in transformers 4.57**: upgraded to transformers from git (`5.8.0.dev0`). Qwen3.5 added in this dev version.

7. **PTY echo poll bug**: see "Polling gotcha" above.

## Current state of code

```
src/voiceai/
├── model/
│   ├── voiceai_lm.py        # Qwen3.5-0.8B + dual-stream wrapper (works after transformers upgrade)
│   ├── audio_adapter.py     # Mimi → Qwen embed projection
│   └── mimi_utils.py        # ⭐ uses hf_hub_download (fixed)
├── training/                # Stage 1/2/3 trainers
│   ├── stage1_adapter.py
│   ├── stage2_dualstream.py
│   ├── stage3_capabilities.py
│   └── data/                # Dataset loaders + format
├── background/
│   └── openai_compat.py     # Single bridge for OpenAI/Anthropic/DashScope/etc.
├── eval/                    # FD-bench, TimeSpeak, ConcurrentCommentary
└── server/app.py            # FastAPI+WebSocket browser demo

scripts/
├── smoke_test.py            # End-to-end CPU smoke (PASSES)
├── extract_librispeech.py   # ⭐ current data prep (running on pod)
├── fast_download.py         # earlier attempt, kept for reference
├── download_data.py         # original (still works after torchcodec fix)
├── download_hf_datasets.py  # SpokenWOZ + others
├── gen_diverse_dialogs.py   # LLM-API-driven dialog gen
├── gen_general_dialog.py    # template baseline
├── gen_concurrent_commentary.py
├── gen_backchannel.py
├── gen_time_aware_audio.py
├── gen_barge_in.py
├── gen_rapid_qa.py
├── gen_constraints.py
├── gen_time_limited.py
├── gen_sound_recognition.py
├── launch_stage1.py
├── launch_stage2.py
└── prep_data_cpu.py         # full orchestrator for CPU server

docs/
├── PLAN.md
├── OSS_LANDSCAPE.md
├── COMPUTE.md
├── HOW_TO_TRAIN.md
├── RUNBOOK.md
└── SESSION_HANDOFF.md       # this file
```

## Pod filesystem layout

```
/workspace/
├── voiceai/                                  # cloned repo
│   ├── .venv/                                # uv-managed virtualenv (~12GB)
│   ├── data/stage1/
│   │   ├── librispeech/                      # ls_*.flac files (growing)
│   │   └── manifest.jsonl                    # appears when extract done
│   └── runs/                                 # checkpoints (none yet)
├── .cache/huggingface/                       # HF cache (parquet shards + model weights)
│   ├── hub/datasets--openslr--librispeech_asr/snapshots/.../clean/train.100/
│   │   └── 0000.parquet ... 0013.parquet     # 14 shards
│   └── hub/models--Qwen--Qwen3.5-0.8B/       # already cached
├── dl.log                                    # extract progress
├── prefetch.log
├── sync.log
└── watchdog.sh
```

## Next steps (priority order)

1. **Wait for extract DONE** (~few more minutes). Background polling task `blaghb4ob` will notify.

2. **Verify manifest**:
   ```bash
   printf 'wc -l /workspace/voiceai/data/stage1/manifest.jsonl\nhead -1 /workspace/voiceai/data/stage1/manifest.jsonl\nexit\n' | ssh -tt -F ~/.ssh/config_runpod runpod
   ```

3. **Re-verify Qwen3.5 load works** with upgraded transformers:
   ```bash
   printf 'cd /workspace/voiceai && HF_HUB_ENABLE_HF_TRANSFER=1 uv run python -c "from voiceai.model.voiceai_lm import VoiceAILM, VoiceAIConfig; m = VoiceAILM(VoiceAIConfig()).cuda(); print(m.trainable_param_count()/1e6, \"M trainable\")"\nexit\n' | ssh -tt -F ~/.ssh/config_runpod runpod
   ```

4. **Launch Stage 1 training** in tmux:
   ```bash
   tmux new-session -d -s train "cd /workspace/voiceai && HF_HUB_ENABLE_HF_TRANSFER=1 uv run python -m voiceai.training.stage1_adapter --manifest data/stage1/manifest.jsonl --output runs/stage1 --backbone Qwen/Qwen3.5-0.8B --steps 30000 --batch-size 8 --grad-accum 4 --lr 3e-4 --warmup 500 --ckpt-every 2000 --wandb-project voiceai > /workspace/train.log 2>&1"
   ```

5. **Monitor wandb**: https://wandb.ai/chukfinley/voiceai (after first log)

6. **Parallel during Stage 1 (CPU available)**: generate capability data via `scripts/gen_*.py` with `--encode-mimi --device cuda` (uses GPU when not training? need to test sharing). OR pause data gen during training.

7. **When Stage 1 done (~3 days RTX 3090)**:
   - Verify ASR WER: `uv run python -m voiceai.eval.asr_quality --model runs/stage1/final --manifest data/stage1/manifest.jsonl --n 100`
   - If WER < 30%: proceed to Stage 2 launch
   - If WER > 30%: investigate before spending more

## Things to do better in next session

1. Use `tmux send-keys` or `expect` for remote control instead of stdin-piping (cleaner).
2. Maybe set up a tiny HTTP server on pod for status JSON to poll without ssh-per-call.
3. Get user's OpenAI/DashScope key BEFORE Stage 2 (needed for diverse-dialog generation).
4. Consider running CommonVoice extract while Stage 1 trains (currently skipped, only LibriSpeech 100h going).

## Tasks state

```
#35 ✅ SSH into Runpod + setup pod
#36 🟡 Run data prep on pod (Threadripper 24c)  — extract running
#37 ⏳ Launch Stage 1 training                  — pending data
#38 🟡 Write SESSION_HANDOFF.md                 — this doc
```

Other completed: build skeleton, write all 10 data generators, write OpenAI-compat bridge, write eval harness, write inference server, set up GitHub repo + CI.

## Budget tracker

- Spent so far: ~$1 (1.5h pod runtime during setup + debugging)
- Stage 1 training est.: $33
- Stage 2 + 3 est.: $77
- Remaining of $1000: ~$889

## Quick reference commands

```bash
# Quick status check
printf 'tmux ls; ps -ef | grep -E "python.*scripts" | grep -v grep | head; tail -3 /workspace/dl.log; tail -3 /workspace/train.log 2>/dev/null; nvidia-smi --query-gpu=utilization.gpu,memory.used --format=csv,noheader\nexit\n' | ssh -tt -F ~/.ssh/config_runpod runpod | sed 's/\x1b\[[0-9;?]*[a-zA-Z]//g'

# Attach to running tmux on pod (interactive)
ssh -t l840jy1v2aiwh1-64410b32@ssh.runpod.io -i ~/.ssh/id_ed25519
# then: tmux attach -t dl  OR  tmux attach -t train

# Pod stop (preserves persistent storage, stops GPU billing)
# → use Runpod UI
```
