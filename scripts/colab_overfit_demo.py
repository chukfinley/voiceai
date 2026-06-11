#!/usr/bin/env python
"""Tiny OVERFIT demo — see the model actually learn audio -> text.

Trains the Stage-1 audio adapter HARD on a handful of real LibriSpeech clips,
keeps the model in memory, then greedily transcribes those SAME clips. If the
adapter overfit, the printed HYP should match the REF — a visible proof the
pipeline learns. (Not a useful general model; a couple of clips memorised.)

Runs in ~2 min on a free Colab T4:
    uv run --extra train python scripts/colab_overfit_demo.py
"""
from __future__ import annotations

import io
import os
import sys
from pathlib import Path

REPO = Path(__file__).resolve().parent.parent
if str(REPO / "src") not in sys.path:
    sys.path.insert(0, str(REPO / "src"))
os.chdir(REPO)

import torch
import torch.nn.functional as F


def main() -> None:
    import argparse

    p = argparse.ArgumentParser()
    p.add_argument("--backbone", default="Qwen/Qwen3-0.6B")
    p.add_argument("--clips", type=int, default=6)
    p.add_argument("--steps", type=int, default=600)
    p.add_argument("--lr", type=float, default=5e-4)
    a = p.parse_args()

    import soundfile as sf
    from datasets import Audio, load_dataset

    from voiceai.model.mimi_utils import load_mimi, mimi_encode, resample_to_mimi
    from voiceai.model.voiceai_lm import VoiceAIConfig, VoiceAILM

    device = "cuda" if torch.cuda.is_available() else "cpu"
    dtype = torch.bfloat16 if device == "cuda" else torch.float32

    # --- real clips -------------------------------------------------------
    ds = load_dataset(
        "hf-internal-testing/librispeech_asr_dummy", "clean", split="validation"
    ).cast_column("audio", Audio(decode=False))
    raw = []
    for ex in ds:
        if len(raw) >= a.clips:
            break
        au = ex["audio"]
        arr, sr = (
            sf.read(io.BytesIO(au["bytes"])) if au.get("bytes") else sf.read(au["path"])
        )
        txt = (ex.get("text") or "").strip()
        if txt:
            raw.append((arr, sr, txt))
    print(f"[demo] {len(raw)} real clips on {device}")

    # --- model ------------------------------------------------------------
    cfg = VoiceAIConfig(
        backbone=a.backbone, freeze_backbone=True, train_text=True,
        train_asst_audio=True, dtype=("bfloat16" if device == "cuda" else "float32"),
    )
    model = VoiceAILM(cfg).to(device)
    mimi = load_mimi(device=device, dtype=dtype)
    tok = model.tokenizer
    if tok.pad_token_id is None:
        tok.pad_token = tok.eos_token
    emb = model.backbone.get_input_embeddings()
    bb_dtype = emb.weight.dtype
    eos = tok.eos_token_id

    # --- pre-encode audio + tokenize text --------------------------------
    data = []
    for arr, sr, txt in raw:
        if getattr(arr, "ndim", 1) > 1:
            arr = arr.mean(axis=1)
        t = torch.from_numpy(arr.astype("float32"))[None, None].to(device)
        t = resample_to_mimi(t, sr).to(dtype)
        with torch.no_grad():
            codes = mimi_encode(mimi, t)
        ids = tok(txt, add_special_tokens=False, return_tensors="pt").input_ids.to(device)
        ids = torch.cat([ids, torch.tensor([[eos]], device=device)], dim=1)
        data.append((codes, ids, txt))

    # --- train (overfit) --------------------------------------------------
    opt = torch.optim.AdamW([q for q in model.parameters() if q.requires_grad], lr=a.lr)
    model.train()
    for step in range(a.steps):
        codes, ids, _ = data[step % len(data)]
        audio_e = model.audio_in(codes).to(bb_dtype)
        Ta, Tt = audio_e.shape[1], ids.shape[1]
        embeds = torch.cat([audio_e, emb(ids[:, :-1])], dim=1)
        out = model.backbone(inputs_embeds=embeds, output_hidden_states=True, return_dict=True)
        h = out.hidden_states[-1][:, Ta - 1: Ta - 1 + Tt, :]
        logits = model.backbone.lm_head(h)
        loss = F.cross_entropy(logits.reshape(-1, logits.size(-1)).float(), ids.reshape(-1))
        opt.zero_grad()
        loss.backward()
        opt.step()
        if step % 100 == 0 or step == a.steps - 1:
            print(f"  step {step:4d}  loss {loss.item():.3f}")

    # --- transcribe the same clips ---------------------------------------
    model.eval()
    print("\n[demo] transcribe the SAME clips (overfit -> HYP should match REF):\n")
    ok = 0
    for codes, ids, ref in data:
        with torch.no_grad():
            embeds = model.audio_in(codes).to(bb_dtype)
            gen: list[int] = []
            for _ in range(80):
                out = model.backbone(inputs_embeds=embeds, output_hidden_states=True, return_dict=True)
                nid = int(model.backbone.lm_head(out.hidden_states[-1][:, -1:, :]).argmax(-1).item())
                if nid == eos:
                    break
                gen.append(nid)
                embeds = torch.cat([embeds, emb(torch.tensor([[nid]], device=device)).to(bb_dtype)], dim=1)
        hyp = tok.decode(gen, skip_special_tokens=True).strip()
        match = hyp.lower() == ref.lower()
        ok += match
        print(f"  REF: {ref}")
        print(f"  HYP: {hyp}   {'OK' if match else ''}\n")
    print(f"[demo] {ok}/{len(data)} clips reproduced exactly.")


if __name__ == "__main__":
    main()
