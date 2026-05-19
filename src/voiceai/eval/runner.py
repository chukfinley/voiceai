"""Eval runner: loads a model checkpoint, runs each benchmark, writes report."""
from __future__ import annotations

import json
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Callable


@dataclass
class EvalResult:
    name: str
    score: float
    n_samples: int
    details: dict


class EvalRunner:
    def __init__(self, model_path: str | Path, device: str = "cuda"):
        self.model_path = Path(model_path)
        self.device = device
        self._model = None

    def load(self):
        if self._model is None:
            from ..model.voiceai_lm import VoiceAILM

            self._model = VoiceAILM.from_pretrained(self.model_path).to(self.device).eval()
        return self._model

    def run(self, benchmarks: dict[str, Callable[["EvalRunner"], EvalResult]], out: Path | None = None) -> list[EvalResult]:
        results = []
        for name, fn in benchmarks.items():
            print(f"running {name}…")
            r = fn(self)
            results.append(r)
            print(f"  → {r.score:.3f} ({r.n_samples} samples)")
        if out is not None:
            out.parent.mkdir(parents=True, exist_ok=True)
            out.write_text(json.dumps([asdict(r) for r in results], indent=2))
        return results
