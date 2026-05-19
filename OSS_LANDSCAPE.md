# OSS_LANDSCAPE.md — Open-Source Real-Time Speech LLM Stack (May 2026)

Komplett-Inventar OSS-Komponenten für native Speech-LLMs. Quelle: research agent run.

## 1. Full-Duplex / Streaming Speech LLMs

- **Moshi (Kyutai)** — Helium-7B backbone + dual-stream + Inner Monologue, native full-duplex. 7B params, ~200ms latency, CC-BY-4.0 / Apache. [github.com/kyutai-labs/moshi](https://github.com/kyutai-labs/moshi) · [paper](https://arxiv.org/abs/2410.00037)
- **GLM-4-Voice (Zhipu)** — End-to-end zh/en speech LLM; GLM-4-9B + Whisper-VQ tokenizer (12.5Hz) + CosyVoice flow-matching decoder. 9B, streamable from 10 tokens. [github.com/zai-org/GLM-4-Voice](https://github.com/zai-org/GLM-4-Voice)
- **MiniCPM-o 2.6 / 4.5 (OpenBMB)** — SigLip + Whisper-medium + ChatTTS-200M + Qwen2.5-7B; 4.5 (9B) adds true full-duplex with non-blocking streams. Apache-2.0. [github.com/OpenBMB/MiniCPM-o](https://github.com/OpenBMB/MiniCPM-o)
- **LLaMA-Omni / LLaMA-Omni2 (ICTNLP)** — v1: Llama-3.1-8B + speech adapter + streaming HiFi-GAN-style decoder, 226ms. v2: 0.5B–14B on Qwen2.5 with AR streaming decoder. Apache-2.0. [github.com/ictnlp/LLaMA-Omni2](https://github.com/ictnlp/LLaMA-Omni2) · [arxiv 2505.02625](https://arxiv.org/abs/2505.02625)
- **Step-Audio 2 mini (StepFun)** — Qwen2.5-7B backbone, multimodal pretrain on 5M hours, tool-calling + multimodal RAG. Apache-2.0. [github.com/stepfun-ai/Step-Audio2](https://github.com/stepfun-ai/Step-Audio2)
- **Freeze-Omni (Tencent/VITA)** — Frozen-LLM speech-to-speech; preserves text intelligence by training adapters only. [github.com/VITA-MLLM/Freeze-Omni](https://github.com/VITA-MLLM/Freeze-Omni)
- **VITA-1.5 (Tencent)** — Vision+speech multimodal, ~1.5s e2e latency, no separate ASR/TTS. [github.com/VITA-MLLM/VITA](https://github.com/VITA-MLLM/VITA)
- **SLAM-Omni** — Single-stage training, zero-shot timbre via semantic-token + vocoder decoupling. [slam-omni.github.io](https://slam-omni.github.io/)
- **Westlake-Omni** — Chinese emotional speech LLM with discrete unified speech/text. [github.com/xinchen-ai/Westlake-Omni](https://github.com/xinchen-ai/Westlake-Omni)
- **IntrinsicVoice** — Multi-turn S2S, <100ms latency, releases IntrinsicVoice-500k dataset. [arxiv 2410.08035](https://arxiv.org/abs/2410.08035)
- **SyncLLM** — Interleaved input/output stream tokens for full-duplex.
- **Mini-Omni / Mini-Omni2 (gpt-omni)** — Qwen2-0.5B base; v2 adds vision + command-based interruption. [github.com/gpt-omni/mini-omni2](https://github.com/gpt-omni/mini-omni2) · [arxiv 2408.16725](https://arxiv.org/abs/2408.16725)
- **Baichuan-Audio** — End-to-end zh/en duplex; Baichuan tokenizer + LLM + flow-matching decoder. [github.com/baichuan-inc/Baichuan-Audio](https://github.com/baichuan-inc/Baichuan-Audio)
- **OpenS2S (CASIA)** — Fully open empathetic S2S on Qwen3-8B-Instruct. [github.com/CASIA-LM/OpenS2S](https://github.com/CASIA-LM/OpenS2S)
- **OmniFlatten (Alibaba)** — End-to-end GPT for seamless voice convo; concurrent text+audio token output. [arxiv 2410.17799](https://arxiv.org/abs/2410.17799)
- **SALMONN-omni (Bytedance)** — Codec-free full-duplex; embedding-based with dynamic thinking gate, +30% vs prior open. [arxiv 2505.17060](https://arxiv.org/abs/2505.17060) · [github.com/bytedance/SALMONN](https://github.com/bytedance/SALMONN)
- **Hertz-dev (Standard Intelligence)** — First full-duplex base model, 8.5B, Hertz-Codec @8Hz/1kbps, 120ms real-world latency on RTX 4090. [github.com/Standard-Intelligence/hertz-dev](https://github.com/Standard-Intelligence/hertz-dev)
- **Spirit-LM (Meta)** — Llama-2 continually pretrained with interleaved speech+text; BASE and EXPRESSIVE variants. [github.com/facebookresearch/spiritlm](https://github.com/facebookresearch/spiritlm)
- **Ming-Lite-Omni v1.5 (Ant Group)** — MoE 22B total / 3B active; image+audio+text+video gen. [github.com/inclusionAI/Ming](https://github.com/inclusionAI/Ming) · [HF](https://huggingface.co/inclusionAI/Ming-Lite-Omni)
- **Ming-UniAudio (Ant)** — Unified speech understand/gen/edit. [github.com/inclusionAI/Ming-UniAudio](https://github.com/inclusionAI/Ming-UniAudio)
- **Ultravox (Fixie)** — Adapter-only training; Whisper-encoder + projector + frozen Llama-3.3-70B; v0.7 supports GLM-4.6 backbone. MIT. [github.com/fixie-ai/ultravox](https://github.com/fixie-ai/ultravox)

## 2. Audio Codecs (streaming-decode capable)

- **Mimi (Kyutai)** — Semantic+acoustic via SpeechTokenizer-style distillation; 12.5Hz, 1.1kbps, fully causal. Reference codec for LLM speech. [HF kyutai/mimi](https://huggingface.co/kyutai/mimi)
- **SNAC (Hubert Siuzdak)** — Multi-scale RVQ; hierarchical tokens at lower bitrate than DAC. [github.com/hubertsiuzdak/snac](https://github.com/hubertsiuzdak/snac)
- **EnCodec (Meta)** — 24/48kHz, RVQ, streaming variant; pure neural acoustic (no semantic distillation).
- **DAC (Descript Audio Codec)** — 44.1kHz, high-fidelity RVQ acoustic codec; not natively streaming-causal.
- **WavTokenizer** — Single-quantizer at 40/75 tokens/sec, contains semantic info. [github.com/jishengpeng/WavTokenizer](https://github.com/jishengpeng/WavTokenizer)
- **X-Codec** — Adds semantic supervision to acoustic codec for LLM use. [arxiv 2408.17175](https://arxiv.org/abs/2408.17175)
- **BigCodec** — Single-VQ, low-bitrate speech codec. [arxiv 2409.05377](https://arxiv.org/abs/2409.05377)
- **Hertz-Codec** — Convolutional VAE @8Hz/1kbps, mono 16kHz. Inside Hertz-dev repo.
- **Survey hub**: [github.com/ga642381/speech-trident](https://github.com/ga642381/speech-trident)

Semantic+acoustic: Mimi, WavTokenizer, X-Codec, SpeechTokenizer. Pure acoustic: EnCodec, DAC, SNAC, BigCodec, Hertz-Codec.

## 3. Audio Encoders (LLM input)

- **Whisper-large-v3 / turbo (OpenAI)** — 1.5B; v3-turbo is de-facto encoder for omni models. MIT.
- **Distil-Whisper (HF)** — 5.8× faster, 51% smaller, within 1% WER. MIT. [arxiv 2311.00430](https://arxiv.org/abs/2311.00430)
- **Moonshine v2 (Useful Sensors)** — Ergodic streaming encoder ASR, smallest 27MB. MIT. [arxiv 2602.12241](https://arxiv.org/abs/2602.12241)
- **SenseVoice-Small (Alibaba FunAudioLLM)** — Multilingual ASR+SER+AED. [github.com/FunAudioLLM/SenseVoice](https://github.com/FunAudioLLM/SenseVoice)
- **FunASR (Alibaba)** — Toolkit with Paraformer/SeACo, streaming-friendly. [github.com/modelscope/FunASR](https://github.com/modelscope/FunASR)
- **Parakeet-TDT 0.6B v2/v3 (NVIDIA NeMo)** — FastConformer, TDT decoder, RTFx 3386, top of HF OpenASR leaderboard; v3 supports 25 EU languages. CC-BY-4.0. [HF](https://huggingface.co/nvidia/parakeet-tdt-0.6b-v3)
- **Canary-Qwen-2.5B (NVIDIA)** — SALM: FastConformer + Qwen3-1.7B w/ LoRA; 5.63% WER. CC-BY-4.0. [HF](https://huggingface.co/nvidia/canary-qwen-2.5b)
- **Qwen3-ASR / Qwen-AuT** — Qwen3 audio encoder, used inside Qwen3-Omni.

## 4. Streaming TTS (cascade fallback)

- **Kokoro-82M** — 82M params, RTF ~0.03 on A100, sub-300ms TTFA, Apache-2.0. [HF hexgrad/Kokoro-82M](https://huggingface.co/hexgrad/Kokoro-82M)
- **Orpheus 3B (Canopy)** — Llama-based Speech-LLM; 150M/400M/1B/3B sizes; zero-shot clone, emotion tags, streaming. Apache-2.0.
- **CosyVoice 2 (Alibaba)** — Unified streaming/non-streaming, voice clone, multilingual, LLM-based. Apache-2.0. [arxiv 2412.10117](https://arxiv.org/abs/2412.10117)
- **F5-TTS** — Flow-matching, 7×–33× RT, zero-shot clone but non-AR → poor for streaming TTFA. [arxiv 2410.06885](https://arxiv.org/abs/2410.06885)
- **XTTS-v2 (Coqui)** — Streaming, ~200ms TTFA, multilingual clone, CPML license.
- **StyleTTS2** — High-quality non-AR diffusion, not natively streaming.
- **MeloTTS (MyShot)** — Very fast, sub-second TTFA, multilingual. MIT.
- **Parler-TTS (HF)** — Text-prompt-controlled voices, AR streaming-friendly. Apache-2.0.
- **VoxCPM 2 (OpenBMB)** — Tokenizer-free, MiniCPM-4 backbone, AudioVAE V2, 48kHz; RTF 0.17. [github.com/OpenBMB/VoxCPM](https://github.com/OpenBMB/VoxCPM)
- **OpenVoice v2 (MyShell)** — Multilingual instant cloning, MIT.
- **Sesame CSM-1B** — 1B Llama backbone + 100M decoder; ~150ms TTFA, context-aware prosody. Apache-2.0. [github.com/SesameAILabs/csm](https://github.com/SesameAILabs/csm)
- **Qwen3-TTS** — Alibaba's Qwen-aligned multilingual TTS.

## 5. Training Frameworks

- **Moshi/Helium training stack**: Helium-7B text backbone → train Mimi codec → Inner Monologue (joint text+audio token prediction, text leads audio by ~80ms) → multi-stream RQ-Transformer modeling user + Moshi streams in parallel.
- **Ultravox training**: only adapter (LLM + Whisper-encoder frozen in v0.4; encoder fine-tuned in v0.5). Single-stage SFT on ASR + speech-instruction pairs.
- **NeMo SALM (Canary-Qwen)**: encoder + linear projection + LoRA on LLM, two-stage curriculum.
- **LLaMA-Omni2 recipe**: 200K multi-turn S2S samples; AR streaming decoder on Qwen2.5.
- **OmniFlatten**: progressive interleaving curriculum (text→speech-text→full-duplex).
- **SALMONN-omni**: codec-free, embedding streams with "thinking" gate that learns listen↔speak transitions.
- **SLAM-Omni**: single-stage training, no codec injection; timbre vocoder decoupled.

## 6. Public Datasets for Full-Duplex Training

- **Fisher Corpus (LDC2004S13, LDC2005S13)** — ~2000h English telephone dyads, channel-separated. Gold standard.
- **CallHome (LDC97S42)** — Conversational telephone speech, 6 languages, channel-separated.
- **CANDOR Corpus** — 1656 video-chat convs, 850+h, 7M words, multimodal. [science.org/doi/10.1126/sciadv.adf3197](https://www.science.org/doi/10.1126/sciadv.adf3197)
- **Spotify Podcasts** — 100k episodes, ~47k h transcribed.
- **VoxPopuli** — 100k h unlabelled, 1.8k labelled (16 langs), 17.3k h simultaneous interpretation pairs. [arxiv 2101.00390](https://arxiv.org/abs/2101.00390)
- **GigaSpeech** — 10k h multi-domain English.
- **CHiME-5/6/7** — Far-field multi-talker dinner-party conversation.
- **IntrinsicVoice-500k** — 500k S2S turns synthetic+real.
- **Common Voice, LibriSpeech, MLS** — read-speech ASR baselines.
- **MOSEL, Emilia, MagicData-RAMC** — large open multilingual TTS/dialog corpora.

## 7. Inference Frameworks

- **vLLM** — Best multimodal support (Qwen3-Omni, MiniCPM-o, Ultravox); audio-token streaming; broadest hardware. Apache-2.0. [github.com/vllm-project/vllm](https://github.com/vllm-project/vllm)
- **SGLang** — ~29% throughput edge in TBS; RadixAttention prefix reuse.
- **TensorRT-LLM** — Peak NVIDIA throughput; CosyVoice 2 + F5-TTS TRT engines exist.
- **llama.cpp** — Now supports audio+vision for Qwen2.5-Omni & Qwen3-Omni GGUF; [tc-mb/llama.cpp-omni](https://github.com/tc-mb/llama.cpp-omni) fork = first OSS full-duplex omni streaming engine. MIT.
- **MLC-LLM** — TVM compile; on-device focus (iOS/Android/web).
- **Moshi-mlx / moshi-rs / moshi-server** — Production-grade Mimi + Moshi serving.

## 8. Acoustic Echo Cancellation (OSS)

- **WebRTC AEC3** — Production-quality; pulled from Chromium's `modules/audio_processing/aec3`. BSD. Standard.
- **SpeexDSP** — Classic DSP-based AEC; Xiph; unmaintained.
- **RNNoise (Mozilla)** — DSP + tiny RNN suppression; unmaintained.
- **DeepFilterNet 2/3 (FAU Erlangen)** — DNN suppression, real-time on CPU. MIT. [github.com/Rikorose/DeepFilterNet](https://github.com/Rikorose/DeepFilterNet)
- **PJSIP AEC** wrapper for SpeexDSP/WebRTC.
- **PortAudio + voice-engine/ec** — Linux ALSA front-end.

## 9. Visual Proactivity / Streaming VLM

- **VideoLLM-online (showlab, CVPR'24)** — LIVE framework; Streaming-EOS objective predicts "speak or stay silent." 5-15 FPS. [github.com/showlab/videollm-online](https://github.com/showlab/videollm-online)
- **StreamingVLM (MIT Han Lab)** — Infinite streams via compact KV cache + training/inference alignment. [github.com/mit-han-lab/streaming-vlm](https://github.com/mit-han-lab/streaming-vlm) · [arxiv 2510.09608](https://arxiv.org/abs/2510.09608)
- **Proact-VL** — Proactive VideoLLM; built on Qwen2-VL/2.5-VL/3-VL; Live Gaming Benchmark. [proact-vl.github.io](https://proact-vl.github.io/)
- **VideoLLaMA 3 (DAMO)** — Vision encoder + compressor + projector + LLM. [github.com/DAMO-NLP-SG/VideoLLaMA3](https://github.com/DAMO-NLP-SG/VideoLLaMA3)
- **LLaVA-Video** — 178K synth instruction set, SlowFast frame selection. [llava-vl.github.io/blog/2024-09-30-llava-video](https://llava-vl.github.io/blog/2024-09-30-llava-video/)
- **LiveVLM** — KV-cache compression for online VQA. [arxiv 2505.15269](https://arxiv.org/abs/2505.15269)
- **VST (Streaming Thinking)** — Watch-and-think simultaneously. [github.com/1ranGuan/VST](https://github.com/1ranGuan/VST)
- **NVIDIA Live-VLM-WebUI** — WebRTC reference frontend.
- **Awesome list**: [github.com/sotayang/Awesome-Streaming-Video-Understanding](https://github.com/sotayang/Awesome-Streaming-Video-Understanding)

## 10. Closed Competitors (context)

- **OpenAI GPT-realtime / gpt-realtime-2 / gpt-realtime-translate / gpt-realtime-whisper** — Native S2S over WebRTC/WebSocket; ~320ms median latency; function-calling. GA exited beta May 2026. Pricing: $32/M audio-in, $64/M audio-out tokens.
- **Google Gemini 3.1 Flash Live (Mar 2026)** — Native audio in/out, 128K context, audio+image+video+text in single pass.
- **Sesame CSM (Maya/Miles)** — 1B base is open; production "Maya" closed and larger.
- **Hume EVI 3** — Native speech-LM, emotion modeling, sub-300ms e2e, voice clone <30s.
- **Thinking Machines Interaction Model** — Split foreground + background; simultaneous speech, time-aware, visual proactivity. **Unser target.**

---

## Build-Empfehlung für Qwen-basiertes Interaction-Modell

**Nächste OSS-Architektur:** Qwen3-Omni-30B-A3B (MoE 3B aktiv) als Foreground + Qwen3-235B-A22B-Thinking als Background. **Mimi @ 12.5Hz** als Codec (license-check für kommerziell) oder eigener SpeechTokenizer-style trainen. **Moshi's dual-stream + Inner Monologue** Training-Recipe übernehmen. **VideoLLM-online Streaming-EOS** Objective für visuelle Proactivity. Cascade-Fallback TTS: **CosyVoice 2** oder **Orpheus-3B**. ASR-only: **Parakeet-TDT-0.6B-v3**. AEC: **WebRTC AEC3** + **DeepFilterNet 3**. Inference: **vLLM** oder **llama.cpp-omni** für edge. Data: **Fisher + CallHome + CANDOR** für duplex; **VoxPopuli interpretation pairs** für simultaneous speech; **Emilia + IntrinsicVoice-500k** für S2S.
