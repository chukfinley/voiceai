"""Eval: emotion recognition accuracy from audio.

Feeds Mimi-coded audio as a prefix and checks whether the model's FIRST
predicted monologue token is the correct <emo:x> tag (the download pipeline
bakes tags in first position: "<emo:angry> transcript...").

Usage:
    uv run python -m voiceai.eval.emotion_recognition \
        --model runs/stage1/final \
        --manifest data/hf/crema_d/manifest.jsonl --max-samples 500
"""
from __future__ import annotations

import argparse
import json
from collections import Counter
from pathlib import Path

import torch


@torch.no_grad()
def predict_emotion_token(model, codes: torch.Tensor) -> str:
    """codes: [1, K, T] Mimi codes → decoded first text token string."""
    bb_dtype = model.backbone.get_input_embeddings().weight.dtype
    audio_e = model.audio_in(codes).to(bb_dtype)
    out = model.backbone(
        inputs_embeds=audio_e, output_hidden_states=True, return_dict=True
    )
    logits = model.backbone.lm_head(out.hidden_states[-1][:, -1])
    tok = int(logits.argmax(dim=-1).item())
    return model.tokenizer.convert_ids_to_tokens(tok)


def evaluate(model, mimi, manifest: Path, device: str, max_samples: int = 500) -> dict:
    import soundfile as sf

    from ..model.mimi_utils import mimi_encode, resample_to_mimi

    total, correct = 0, 0
    confusion: Counter = Counter()
    for line in manifest.read_text().splitlines():
        if total >= max_samples:
            break
        if not line.strip():
            continue
        meta = json.loads(line)
        emo = meta.get("emotion")
        if not emo:
            continue
        try:
            audio, sr = sf.read(meta["audio"], dtype="float32")
        except Exception:
            continue
        if audio.ndim > 1:
            audio = audio.mean(axis=1)
        wav = resample_to_mimi(torch.from_numpy(audio)[None, None], sr)
        codes = mimi_encode(mimi, wav).long().to(device)
        pred = predict_emotion_token(model, codes)
        expected = f"<emo:{emo}>"
        total += 1
        correct += int(pred == expected)
        confusion[(expected, pred)] += 1

    acc = correct / max(1, total)
    return {"accuracy": acc, "n": total, "confusion": confusion}


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("--model", required=True, type=Path)
    p.add_argument("--manifest", required=True, type=Path)
    p.add_argument("--max-samples", type=int, default=500)
    p.add_argument("--device", default="cuda")
    args = p.parse_args()

    from ..model.mimi_utils import load_mimi
    from ..model.voiceai_lm import VoiceAILM

    device = args.device if torch.cuda.is_available() else "cpu"
    model = VoiceAILM.from_pretrained(args.model).to(device).eval()
    mimi = load_mimi(device=device)

    res = evaluate(model, mimi, args.manifest, device, args.max_samples)
    print(f"emotion accuracy: {res['accuracy']:.3f} on {res['n']} samples")
    for (exp, pred), n in res["confusion"].most_common(20):
        marker = "✓" if exp == pred else "✗"
        print(f"  {marker} {exp:20s} → {pred:20s} ×{n}")


if __name__ == "__main__":
    main()
