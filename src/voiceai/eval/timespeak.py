"""TimeSpeak benchmark — re-implementation of TML's time-awareness eval.

For each sample: given an instruction like "wait N seconds then say X",
measure when the model produces non-silent assistant audio. Score = how
close the timing is to N (within tolerance).

Macro accuracy = % samples where model spoke within ±tolerance_s of the
expected wait time.
"""
from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import torch

from .runner import EvalResult


@dataclass
class TimeSpeakConfig:
    test_data: Path
    tolerance_s: float = 1.0
    max_samples: int | None = 100


def evaluate(runner, cfg: TimeSpeakConfig) -> EvalResult:
    model = runner.load()
    meta_path = cfg.test_data / "samples.jsonl"
    enc_path = cfg.test_data / "encoded"
    if not meta_path.exists():
        return EvalResult(name="timespeak", score=0.0, n_samples=0, details={"error": "no data"})

    samples = [json.loads(line) for line in meta_path.read_text().splitlines() if line.strip()]
    if cfg.max_samples:
        samples = samples[: cfg.max_samples]

    correct = 0
    per_sample = []
    for s in samples:
        path = enc_path / f"{s['sample_id']}.npz"
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
        silent_id = model.cfg.codebook_size
        non_silent = (pred_codes != silent_id).any(axis=0)
        first_idx = int(np.argmax(non_silent)) if non_silent.any() else -1
        predicted_t_s = first_idx / 12.5 if first_idx >= 0 else None
        expected_t_s = s.get("wait_s")

        per_sample.append({"sample_id": s["sample_id"], "predicted_s": predicted_t_s, "expected_s": expected_t_s})

        if predicted_t_s is None or expected_t_s is None:
            continue
        if abs(predicted_t_s - expected_t_s) <= cfg.tolerance_s:
            correct += 1

    n = len(per_sample)
    score = correct / max(1, n)
    return EvalResult(
        name="timespeak",
        score=score,
        n_samples=n,
        details={"per_sample": per_sample[:10], "tolerance_s": cfg.tolerance_s},
    )


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
    cfg = TimeSpeakConfig(test_data=args.data, max_samples=args.n)
    r = evaluate(runner, cfg)
    print(f"timespeak score: {r.score:.3f} ({r.n_samples} samples)")
    if args.out:
        args.out.parent.mkdir(parents=True, exist_ok=True)
        args.out.write_text(json.dumps({"name": r.name, "score": r.score, "n": r.n_samples}, indent=2))


from .runner import EvalRunner

if __name__ == "__main__":
    main()
