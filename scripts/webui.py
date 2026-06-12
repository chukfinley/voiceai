#!/usr/bin/env python
"""Tiny web UI to test the full-duplex model in a browser.

Record (or upload) audio -> the model produces a spoken response you can play.
Quality is rough (small training run) — this is for a quick listen/test.

    uv run --extra train --extra datagen --with gradio python scripts/webui.py

Loads the model from HuggingFace (chukfinley/voiceai-duplex-demo) by default.
Opens a public gradio.live URL (share=True).
"""
from __future__ import annotations

import sys
from pathlib import Path

REPO = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO / "src"))


def main() -> None:
    import argparse

    p = argparse.ArgumentParser()
    p.add_argument("--model", default="hub", help="local stage2 dir, or 'hub' to pull from HF")
    p.add_argument("--hub-repo", default="chukfinley/voiceai-duplex-demo")
    p.add_argument("--respond-frames", type=int, default=125, help="silent frames to let it answer (~10s)")
    p.add_argument("--port", type=int, default=7860)
    p.add_argument("--share", action="store_true", default=False,
                   help="gradio.live tunnel (buggy on gradio 4.x); prefer RunPod http-proxy instead")
    a = p.parse_args()

    import gradio as gr
    import numpy as np
    import torch

    from voiceai.inference.streaming import StreamingEngine
    from voiceai.model.mimi_utils import load_mimi, mimi_encode, resample_to_mimi
    from voiceai.model.voiceai_lm import VoiceAILM

    device = "cuda" if torch.cuda.is_available() else "cpu"
    dtype = torch.bfloat16 if device == "cuda" else torch.float32

    model_dir = a.model
    if a.model == "hub" or not (Path(a.model) / "adapters.pt").exists():
        from huggingface_hub import snapshot_download
        model_dir = snapshot_download(repo_id=a.hub_repo, repo_type="model")

    print(f"[webui] loading model from {model_dir} on {device}…")
    model = VoiceAILM.from_pretrained(model_dir).to(device).eval()
    mimi = load_mimi(device=device, dtype=dtype)
    silence = mimi_encode(mimi, torch.zeros(1, 1, 24000, device=device, dtype=dtype))

    @torch.no_grad()
    def respond(audio):
        if audio is None:
            return None, "Nimm zuerst was auf 🎤"
        sr, arr = audio
        arr = np.asarray(arr, dtype="float32")
        if arr.ndim > 1:
            arr = arr.mean(axis=1)
        if np.abs(arr).max() > 1.0:        # int16 -> float
            arr = arr / 32768.0
        t = torch.from_numpy(arr)[None, None].to(device)
        t = resample_to_mimi(t, sr).to(dtype)
        codes = mimi_encode(mimi, t)

        engine = StreamingEngine(model, mimi, device=device)
        chunks, toks = [], []

        def frame(cf):
            s = engine.step(cf)
            if s.audio is not None:
                chunks.append(s.audio.squeeze().float().cpu().numpy())
            toks.append(s.text_token)

        for i in range(codes.shape[2]):
            frame(codes[0, :, i])
        for i in range(a.respond_frames):
            frame(silence[0, :, i % silence.shape[2]])

        out = np.concatenate(chunks) if chunks else np.zeros(2400, dtype="float32")
        inner = model.tokenizer.decode([x for x in toks if x is not None], skip_special_tokens=True)
        return (24000, out), f"inner-monologue: {inner[:160]}"

    demo = gr.Interface(
        fn=respond,
        inputs=gr.Audio(sources=["microphone", "upload"], type="numpy", label="Du sprichst"),
        outputs=[gr.Audio(label="Modell antwortet"), gr.Textbox(label="Modell-Text (inner monologue)")],
        title="voiceai — Vollduplex-Demo (roh)",
        description="Nimm was auf oder lad ein Wav hoch. Das Modell antwortet mit Audio. "
                    "Qualität ist grob (winziges Training) — es geht ums Testen dass es läuft.",
    )
    # show_api=False avoids a gradio 4.x json-schema bug (bool not iterable) that
    # otherwise crashes share-tunnel setup before the public URL is printed.
    demo.launch(share=a.share, server_name="0.0.0.0", server_port=a.port, show_api=False)


if __name__ == "__main__":
    main()
