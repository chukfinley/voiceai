"""Stage-1 sanity eval: ASR WER on a held-out LibriSpeech subset.

After Stage 1 the model should be able to go (audio → text) at reasonable
accuracy. This eval verifies the audio adapter actually learned ASR.
"""
from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path

import torch

from .runner import EvalResult


@dataclass
class ASRConfig:
    manifest: Path
    max_samples: int = 50
    sample_rate: int = 24000
    max_text_len: int = 100


def evaluate(runner, cfg: ASRConfig) -> EvalResult:
    try:
        from jiwer import wer
    except ImportError:
        wer = None

    from ..model.mimi_utils import load_mimi, mimi_encode, resample_to_mimi

    model = runner.load()
    mimi = load_mimi(device=runner.device, dtype=torch.bfloat16)
    runner._mimi = mimi

    with cfg.manifest.open() as f:
        lines = [json.loads(line) for line in f if line.strip()]
    lines = lines[: cfg.max_samples]

    refs = []
    hyps = []
    import soundfile as sf

    for ex in lines:
        try:
            audio, sr = sf.read(ex["audio"], dtype="float32")
        except Exception:
            continue
        if audio.ndim > 1:
            audio = audio.mean(axis=1)
        t = torch.from_numpy(audio).unsqueeze(0).unsqueeze(0).to(runner.device)
        t = resample_to_mimi(t, sr).to(torch.bfloat16)
        with torch.no_grad():
            codes = mimi_encode(mimi, t)
        T = codes.shape[2]
        attn = torch.ones(1, T, device=runner.device, dtype=torch.long)
        with torch.no_grad():
            out = model(text_ids=None, user_audio_codes=codes, attention_mask=attn)
        text_pred_ids = out["text_logits"].argmax(dim=-1).squeeze(0).tolist()
        pred = model.tokenizer.decode(text_pred_ids, skip_special_tokens=True)
        refs.append(ex["text"].strip().lower())
        hyps.append(pred.strip().lower()[: cfg.max_text_len])

    if wer is not None and refs:
        wer_score = float(wer(refs, hyps))
    else:
        wer_score = 1.0

    return EvalResult(
        name="asr_wer",
        score=1.0 - wer_score,
        n_samples=len(refs),
        details={"wer": wer_score, "sample_hyps": hyps[:5], "sample_refs": refs[:5]},
    )


def main() -> None:
    import argparse

    from .runner import EvalRunner

    p = argparse.ArgumentParser()
    p.add_argument("--model", required=True, type=Path)
    p.add_argument("--manifest", required=True, type=Path)
    p.add_argument("--out", type=Path, default=None)
    p.add_argument("--device", default="cuda")
    p.add_argument("--n", type=int, default=50)
    args = p.parse_args()

    runner = EvalRunner(args.model, device=args.device)
    cfg = ASRConfig(manifest=args.manifest, max_samples=args.n)
    r = evaluate(runner, cfg)
    print(f"asr WER: {1 - r.score:.3f} ({r.n_samples} samples)")
    if args.out:
        args.out.parent.mkdir(parents=True, exist_ok=True)
        args.out.write_text(json.dumps({"name": r.name, "score": r.score, "n": r.n_samples, "details": r.details}, indent=2))


if __name__ == "__main__":
    main()
