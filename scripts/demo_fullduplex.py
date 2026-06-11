#!/usr/bin/env python
"""Full-duplex demo: feed user audio, get the assistant's spoken response.

    uv run --extra train python scripts/demo_fullduplex.py --model runs/stage2/final

If --user-wav is omitted, a default question is synthesised with kokoro TTS so
the demo is self-contained. Drives the StreamingEngine frame by frame and writes
the assistant's audio to --out. Quality will be rough on a small training run —
the point is that the full loop runs and produces audio.
"""
from __future__ import annotations

import sys
from pathlib import Path

REPO = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO / "src"))


def main() -> None:
    import argparse

    p = argparse.ArgumentParser()
    p.add_argument("--model", default="runs/stage2/final", help="stage2 checkpoint dir")
    p.add_argument("--user-wav", default=None, help="user mic wav (default: TTS a question)")
    p.add_argument("--question", default="Hey, how are you doing today?")
    p.add_argument("--out", default="assistant_response.wav")
    p.add_argument("--respond-frames", type=int, default=125, help="silent frames to let it answer (~10s)")
    a = p.parse_args()

    import numpy as np
    import soundfile as sf
    import torch

    from voiceai.inference.streaming import StreamingEngine
    from voiceai.model.mimi_utils import load_mimi, mimi_encode, resample_to_mimi
    from voiceai.model.voiceai_lm import VoiceAILM

    device = "cuda" if torch.cuda.is_available() else "cpu"
    dtype = torch.bfloat16 if device == "cuda" else torch.float32

    model = VoiceAILM.from_pretrained(a.model).to(device).eval()
    mimi = load_mimi(device=device, dtype=dtype)
    engine = StreamingEngine(model, mimi, device=device)

    # --- user audio -> mimi codes ---
    if a.user_wav:
        arr, sr = sf.read(a.user_wav, dtype="float32")
        if arr.ndim > 1:
            arr = arr.mean(axis=1)
    else:
        from voiceai.training.data.tts_util import synth
        arr, sr = synth(a.question, voice="af_bella", backend="kokoro")
        print(f"[demo] user says: {a.question!r}")
    t = torch.from_numpy(np.asarray(arr, dtype="float32"))[None, None].to(device)
    t = resample_to_mimi(t, sr).to(dtype)
    with torch.no_grad():
        user_codes = mimi_encode(mimi, t)  # [1, K, T]
    silence = mimi_encode(mimi, torch.zeros(1, 1, 24000, device=device, dtype=dtype))  # [1,K,Ts]

    Tu = user_codes.shape[2]
    audio_chunks, text_tokens = [], []

    def run_frame(codes_frame):
        s = engine.step(codes_frame)
        if s.audio is not None:
            audio_chunks.append(s.audio.squeeze().float().cpu().numpy())
        text_tokens.append(s.text_token)

    with torch.no_grad():
        for i in range(Tu):                       # user speaking
            run_frame(user_codes[0, :, i])
        for i in range(a.respond_frames):         # user silent -> let it answer
            run_frame(silence[0, :, i % silence.shape[2]])

    out = np.concatenate(audio_chunks) if audio_chunks else np.zeros(2400, dtype="float32")
    sf.write(a.out, out, 24000)
    inner = model.tokenizer.decode([t for t in text_tokens if t is not None], skip_special_tokens=True)
    print(f"[demo] assistant inner-monologue text: {inner[:200]!r}")
    print(f"[demo] wrote {len(out)/24000:.1f}s of assistant audio -> {a.out}")


if __name__ == "__main__":
    main()
