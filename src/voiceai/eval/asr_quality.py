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

    text_embed_layer = model.backbone.get_input_embeddings()
    bb_dtype = text_embed_layer.weight.dtype
    eos_id = model.tokenizer.eos_token_id
    if model.tokenizer.pad_token_id is None:
        model.tokenizer.pad_token = model.tokenizer.eos_token
    pad_id = model.tokenizer.pad_token_id

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

        with torch.no_grad():
            # Audio prefix embeddings
            audio_e = model.audio_in(codes).to(dtype=bb_dtype)
            embeds = audio_e
            generated: list[int] = []
            for _ in range(cfg.max_text_len):
                T = embeds.shape[1]
                attn = torch.ones(1, T, device=runner.device, dtype=torch.long)
                bb_out = model.backbone(
                    inputs_embeds=embeds,
                    attention_mask=attn,
                    output_hidden_states=True,
                    return_dict=True,
                )
                last_hidden = bb_out.hidden_states[-1][:, -1:, :]
                next_logits = model.backbone.lm_head(last_hidden)
                next_id = int(next_logits.argmax(dim=-1).item())
                if eos_id is not None and next_id == eos_id:
                    break
                if next_id == pad_id:
                    break
                generated.append(next_id)
                next_emb = text_embed_layer(torch.tensor([[next_id]], device=runner.device))
                embeds = torch.cat([embeds, next_emb.to(dtype=bb_dtype)], dim=1)
        pred = model.tokenizer.decode(generated, skip_special_tokens=True)
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
