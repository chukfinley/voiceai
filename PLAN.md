# voiceai — Plan

Open-source competitor to [Thinking Machines Interaction Model](https://thinkingmachines.ai/blog/interaction-models/).

**End goal:** OUR OWN trained speech-LLM that does **all** the following natively:

1. Full-duplex (simultaneous speech)
2. Sub-300ms speech-to-speech latency
3. Time-awareness (knows elapsed time, can self-initiate, can wait)
4. Visual proactivity (frame-by-frame video, decides when to speak)
5. Background-LLM split (small foreground + heavy backend)
6. Tool use
7. Barge-in
8. Streaming output

**No existing OSS model does all of these.** We use existing models as *experiments* and *warm starts*, then train our own.

## PoC Scope

- **English only** (in + out)
- **Single fixed voice**
- Total compute budget: **<$100 cloud** to working trained model

Everything else (multilingual, voice cloning, multi-voice) is post-PoC.

---

## Capability gap matrix

| Model | Full-Dup | <300ms | Time-aware | Visual proac | BG-split | Tools | License |
|-------|---------|--------|-----------|-------------|---------|-------|---------|
| Moshi (Kyutai) | ✅ | ✅ 200ms | ❌ | ❌ | ❌ | ❌ | CC-BY-4.0 |
| Hertz-dev | ✅ | ✅ 120ms | ❌ | ❌ | ❌ | ❌ | Apache |
| KAME (Sakana) | ✅ | ✅ 150ms | ❌ | ❌ | ✅ | ✅ via BG | MIT/Moshi |
| PersonaPlex 7B (NV) | ✅ | ✅ 150ms | ❌ | ❌ | ❌ | ❌ | NVIDIA OML |
| Qwen3-Omni-30B-A3B | ❌ | ✅ 250ms | ❌ | ✅ video | ❌ | ⚠️ | Apache 2.0 |
| Qwen3.5-Omni | ✅ ARIA | ✅ | partial | partial | ❌ | partial | API only |
| MiniCPM-o 4.5 | ✅ | ? | ❌ | ✅ | ❌ | ❌ | Apache |
| SALMONN-omni | ✅ codec-free | ? | ❌ | ❌ | ❌ | ❌ | Apache |
| Step-Audio 2 mini | ⚠️ | ? | ❌ | ❌ | ❌ | ✅ | Apache |
| VideoLLM-online | n/a | streaming | ❌ | ✅ EOS | ❌ | ❌ | research |
| Proact-VL | n/a | streaming | ❌ | ✅ | ❌ | ❌ | research |
| **OUR TARGET** | ✅ | ✅ | ✅ | ✅ | ✅ | ✅ | Apache |

---

## Architecture decision (working draft)

```
                     ┌──────────────────────────┐
mic + camera in  →   │  Frontend Interaction    │  → speaker out
                     │  ───────────────────     │
                     │  Backbone: Moshi-style   │  Native full-duplex
                     │  (dual stream, Mimi)     │  Sub-300ms
                     │  ~3-7B params            │
                     │                          │
                     │  + Time tokens <t:N>     │  Time-aware
                     │  + Visual frame tokens   │  Visual proactivity
                     │  + Barge control tokens  │  Barge-in
                     │  + Oracle slot tokens    │  BG split
                     └──────────┬───────────────┘
                                │ <background_query>...</background_query>
                                ▼
                     ┌──────────────────────────┐
                     │  Backend LLM (pluggable) │
                     │  Claude/GPT/Qwen-Max     │  Tool use, reasoning
                     │  Local Qwen-32B/Llama-70B│
                     └──────────────────────────┘
```

**Key design choices:**

1. **Backbone:** start from Moshi weights (CC-BY-4.0 attribution OK). Helium-7B text backbone + Mimi codec + dual-stream RQ-Transformer. Already has full-duplex + sub-300ms latency for free.

2. **Audio codec:** Mimi (Kyutai). 12.5Hz, 8 codebooks, fully causal streaming. Don't reinvent the wheel — and we can fall back to retraining a SpeechTokenizer-style codec later.

3. **Frame format:** 200ms micro-turns matching TML's design. Inside each frame: tick token, user-audio tokens, asst-text monologue (Moshi style), asst-audio tokens, optional visual-event, optional oracle-result, optional control tokens. See `src/voiceai/training/data/format.py`.

4. **Visual proactivity:** add a Qwen3-VL-2B frame encoder to the input stream. At 2 fps the bandwidth is manageable. Train with VideoLLM-online's Streaming-EOS objective so the model learns when to interrupt.

5. **Background split:** model emits `<background_query>...</background_query>`. Orchestrator dispatches async to a backend LLM (any provider). Result returns as `<bg_result id=X>...</bg_result>` and gets folded into the next assistant frame. This is KAME's oracle pattern.

6. **Training:** fine-tune Moshi backbone with extended vocabulary + extended dual-stream format. Use [nu-dialogue/moshi-finetune](https://github.com/nu-dialogue/moshi-finetune) as starting point.

---

## Phases

### Phase 0 — Skeleton (DONE)

- uv project, async event bus, audio I/O, AEC interface
- Stubbed backends for cascade and Moshi/Qwen-Omni
- Plan + landscape + compute docs

### Phase 1 — Experimentation (2 weeks, ~$10)

Run each candidate locally on a Runpod RTX 3090 ($0.46/h) to learn what each gives us and where they fail:

| Experiment | Question | Time |
|-----------|----------|------|
| Run Moshi unmodified | What's its latency on consumer GPU? Tool use? | 2h |
| Run KAME oracle path | How does async knowledge injection feel? | 2h |
| Run PersonaPlex | Voice quality, role control, barge-in | 2h |
| Run Qwen3-Omni-A3B | Visual input quality, half-duplex limits | 4h |
| Run MiniCPM-o 4.5 | Visual + duplex combo viability | 2h |
| Run VideoLLM-online | Streaming-EOS objective in practice | 4h |
| Run Ultravox v0.7 | Adapter-only training intuition | 2h |
| Run Step-Audio 2 mini | Tool-call quality in audio | 2h |

Output: experiment notes in `experiments/` per model. Identify what we keep, what we drop.

### Phase 2 — Data generation (1-2 weeks, ~$5)

Generate the interaction-format dataset. We need:

1. **Time-awareness pairs** (~10k) — dialog with explicit `<t:>`, `<wait:>`, silences, self-initiated speech. Synthesize with Claude/GPT-4 as data labelers, render to audio with Kokoro/PersonaPlex.
2. **Barge-in pairs** (~5k) — extract from Fisher Corpus + synthesize new. Mark exact barge timestamps.
3. **Backchannel pairs** (~5k) — "mhm/yeah" during user turn.
4. **Visual proactivity scenes** (~3k) — scripted videos with annotated speak/silent labels. Generate via VideoLLM-online's data pipeline.
5. **Simultaneous-speech translation pairs** (~2k) — VoxPopuli interpretation pairs.
6. **Background query pairs** (~5k) — dialog where assistant emits `<background_query>` mid-conversation.

Total ~30k interaction samples. Storage ~50GB after Mimi-encoding.

### Phase 3 — Training (4-6 weeks, ~$50)

Two training tracks:

**Track A — LoRA over Moshi backbone** (fast iteration)
- Add new special tokens to Moshi vocab (`<t:>`, `<visual:>`, `<background_query>`, etc.)
- QLoRA-4bit, rank 32, on RTX 3090 ($0.46/h)
- ~50 GPU-hours for 30k samples → $23
- Validate on FD-bench reimplementation

**Track B — Full fine-tune** (if LoRA hits a ceiling)
- Use `nu-dialogue/moshi-finetune` framework, bf16
- A100 80GB ($1.39/h) for 5 days → $167
- Only run when Track A's quality justifies it

Decision gate: if Track A scores ≥80% of GPT-realtime-2 on FD-bench, ship that. Otherwise spend on Track B.

### Phase 4 — Backend integration (1 week, ~$5)

Implement `voiceai.background.*` bridges:
- `openai_api.py` (GPT-4.1)
- `claude_api.py` (Claude Opus)
- `dashscope_api.py` (Qwen3-Max)
- `local_vllm.py` (when we have local GPU)

Bridge protocol: model emits `<background_query>X</background_query>` → orchestrator dispatches → backend streams response → inject as `<bg_result id=N>...</bg_result>` at next frame boundary.

### Phase 5 — Visual proactivity (2 weeks, ~$20)

Two approaches in parallel:

**Approach X — external watcher (cheap, ships fast)**
- Qwen3-VL-2B running parallel at 2fps
- Emits structured events into the foreground context
- No training needed for PoC

**Approach Y — native frame tokens (target)**
- Add visual token stream to the Mimi-style dual-stream layout
- Frame embeddings injected at 2fps
- Train with Streaming-EOS objective
- Requires Track B full-FT — defer until budget allows

### Phase 6 — Eval + polish (1 week, ~$5)

- Re-implement FD-bench v1/v3, TimeSpeak, CueSpeak, RepCount-A
- Compare against KAME, Moshi, GPT-realtime-2, Hertz-dev
- AEC integration (WebRTC AEC3) for simul-speech
- Blog post + demo video

---

## Total budget

| Phase | What | $ |
|------|------|---|
| 0 | Skeleton | 0 |
| 1 | Experiments on RTX 3090 | 10 |
| 2 | Synth data (API calls + GPU encoding) | 5 |
| 3a | LoRA training | 25 |
| 3b | Full fine-tune (conditional) | +170 |
| 4 | Backend integration | 5 |
| 5 | Visual proactivity (external + native) | 20 |
| 6 | Eval + polish | 5 |
| **Total without Track B** | | **~$70** |
| **Total with Track B** | | **~$240** |

---

## Repository layout

```
voiceai/
├── PLAN.md
├── OSS_LANDSCAPE.md
├── COMPUTE.md
├── pyproject.toml
├── refs/                       # cloned reference repos
│   ├── moshi/                  # base, full-duplex, Mimi codec
│   ├── kame/                   # tandem oracle pattern
│   ├── personaplex/            # voice/role control on Moshi
│   └── ultravox/               # adapter approach
├── src/voiceai/
│   ├── orchestrator/           # event bus, audio I/O, AEC
│   ├── foreground/
│   │   ├── cascade.py          # Option A baseline
│   │   ├── moshi_wrapper.py    # Phase 1 — drive Moshi server
│   │   ├── kame_wrapper.py     # Phase 1 — KAME oracle
│   │   ├── personaplex.py      # Phase 1 — PersonaPlex
│   │   └── voiceai.py          # Phase 3 — OUR OWN trained model
│   ├── background/             # pluggable LLM bridges
│   ├── visual/                 # streaming VLM watcher (Phase 5 X)
│   ├── training/
│   │   ├── data/               # 200ms frame format, synth, loaders
│   │   ├── lora_sft.py         # Track A
│   │   ├── full_sft.py         # Track B
│   │   └── moshi_ft.py         # Wraps nu-dialogue trainer
│   └── eval/                   # FD-bench, TimeSpeak, etc.
├── experiments/                # per-model run logs from Phase 1
├── scripts/
│   ├── bootstrap.py
│   ├── gen_synth_data.py
│   ├── run_moshi.py
│   ├── run_kame.py
│   └── run_personaplex.py
└── tests/
```

---

## License & weights tracking

| Component | License | PoC OK? |
|----------|---------|---------|
| Moshi weights | CC-BY-4.0 | ✅ attribution |
| KAME code | MIT | ✅ |
| PersonaPlex code | MIT | ✅ |
| PersonaPlex weights | NVIDIA Open Model License | ⚠️ check for commercial |
| Mimi codec | CC-BY-4.0 | ✅ |
| Helium-7B (Moshi backbone) | CC-BY-4.0 | ✅ |
| Qwen3-VL-2B | Apache 2.0 | ✅ |
| WebRTC AEC3 | BSD | ✅ |

Our final trained model: **Apache 2.0**, with attribution to Moshi/Mimi as required.

---

## Anti-goals

- ❌ Use a finished model as-is — we want to learn and train our own
- ❌ Multilingual in PoC
- ❌ Voice cloning / multi-voice in PoC
- ❌ Train Qwen3-Omni or any official Qwen-Omni model
- ❌ Spend on H100s in PoC phase (RTX 3090/4090 only)
- ❌ Cascade as final architecture
