"""Shared eval metrics."""
from __future__ import annotations

from collections.abc import Iterable


def accuracy(preds: Iterable, targets: Iterable) -> float:
    preds = list(preds)
    targets = list(targets)
    if not preds:
        return 0.0
    return sum(int(p == t) for p, t in zip(preds, targets)) / len(preds)


def mean(values: Iterable[float]) -> float:
    values = list(values)
    return sum(values) / max(1, len(values))


def count_matches(text: str, words: Iterable[str]) -> int:
    text_lower = text.lower()
    return sum(text_lower.count(w.lower()) for w in words)


def normalized_levenshtein(a: str, b: str) -> float:
    """1.0 = identical, 0.0 = totally different."""
    if not a and not b:
        return 1.0
    m, n = len(a), len(b)
    dp = list(range(n + 1))
    for i in range(1, m + 1):
        prev = dp[0]
        dp[0] = i
        for j in range(1, n + 1):
            cur = dp[j]
            cost = 0 if a[i - 1] == b[j - 1] else 1
            dp[j] = min(dp[j] + 1, dp[j - 1] + 1, prev + cost)
            prev = cur
    return 1 - dp[n] / max(m, n)


def first_speech_latency_ms(asst_audio_codes, silent_token_id: int = 2048) -> float | None:
    """Time of first non-silent assistant frame, in ms (12.5Hz → 80ms/frame)."""
    import numpy as np

    arr = asst_audio_codes if hasattr(asst_audio_codes, "shape") else None
    if arr is None:
        return None
    nonsilent = (arr != silent_token_id).any(axis=0) if arr.ndim == 2 else (arr != silent_token_id)
    nonzero = np.where(nonsilent)[0]
    if len(nonzero) == 0:
        return None
    return float(nonzero[0]) * 80.0
