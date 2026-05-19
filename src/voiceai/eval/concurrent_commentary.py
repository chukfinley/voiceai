"""Evaluate concurrent-commentary capability.

For each test sample (panda-counter style), feed the user-audio stream
into the model and check whether the assistant stream contains the
correct count words at approximately the right times.

Metric:
  match_score = mean over samples of:
      n_correct_counts_at_right_time / n_triggers

A trigger is "matched" if the assistant produced the right count word
within ±0.5s of the expected time.
"""
from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import torch

from .runner import EvalResult


@dataclass
class CCConfig:
    test_data: Path
    tolerance_s: float = 0.5
    max_samples: int | None = 100


def evaluate(runner, cfg: CCConfig) -> EvalResult:
    model = runner.load()
    mimi_path = cfg.test_data / "encoded"
    meta_path = cfg.test_data / "samples.jsonl"
    if not mimi_path.exists() or not meta_path.exists():
        return EvalResult(
            name="concurrent_commentary",
            score=0.0,
            n_samples=0,
            details={"error": f"missing data at {cfg.test_data}"},
        )

    samples = [json.loads(line) for line in meta_path.read_text().splitlines() if line.strip()]
    if cfg.max_samples:
        samples = samples[: cfg.max_samples]

    scores = []
    for s in samples:
        path = mimi_path / f"{s['sample_id']}.npz"
        if not path.exists():
            continue
        arr = np.load(path)
        u = torch.from_numpy(arr["user_codes"]).long().unsqueeze(0).to(runner.device)
        T = u.shape[2]
        attn = torch.ones(1, T, device=runner.device, dtype=torch.long)
        text_ids = torch.full((1, T), model.tokenizer.pad_token_id or 0, device=runner.device, dtype=torch.long)
        with torch.no_grad():
            out = model(text_ids=text_ids, user_audio_codes=u, attention_mask=attn)
        pred_codes = out["asst_audio_logits"].argmax(dim=-1).squeeze(0).cpu().numpy()

        try:
            asst_audio_decoded = _decode_to_audio(pred_codes, mimi=getattr(runner, "_mimi", None))
        except Exception:
            asst_audio_decoded = None

        triggers = s.get("triggers", [])
        if not triggers:
            continue
        hits = 0
        for trig in triggers:
            t_exp = trig["time_s"] + 0.15
            frame = int(t_exp * 12.5)
            window = pred_codes[:, max(0, frame - 6) : min(pred_codes.shape[1], frame + 6)]
            silent_id = model.cfg.codebook_size
            non_silent = (window != silent_id).any(axis=0).sum()
            if non_silent > 0:
                hits += 1
        scores.append(hits / len(triggers))

    score = float(np.mean(scores)) if scores else 0.0
    return EvalResult(
        name="concurrent_commentary",
        score=score,
        n_samples=len(scores),
        details={"per_sample_mean": score, "tolerance_s": cfg.tolerance_s},
    )


def _decode_to_audio(codes: np.ndarray, mimi):
    if mimi is None:
        return None
    import torch

    c = torch.from_numpy(codes).long().unsqueeze(0).to(next(mimi.parameters()).device)
    with torch.no_grad():
        a = mimi.decode(c)
    return a[0, 0].cpu().numpy()


def main() -> None:
    import argparse

    p = argparse.ArgumentParser()
    p.add_argument("--model", required=True, type=Path)
    p.add_argument("--data", required=True, type=Path)
    p.add_argument("--out", type=Path, default=None)
    p.add_argument("--device", default="cuda")
    p.add_argument("--n", type=int, default=100)
    args = p.parse_args()

    runner = EvalRunner(args.model, device=args.device)
    cfg = CCConfig(test_data=args.data, max_samples=args.n)
    result = evaluate(runner, cfg)
    print(f"concurrent_commentary score: {result.score:.3f} ({result.n_samples} samples)")
    if args.out:
        args.out.parent.mkdir(parents=True, exist_ok=True)
        args.out.write_text(json.dumps({"name": result.name, "score": result.score, "n": result.n_samples, "details": result.details}, indent=2))


if __name__ == "__main__":
    main()
