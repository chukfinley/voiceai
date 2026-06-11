"""Shared LR schedules for the training stages."""
from __future__ import annotations

import math

import torch


def build_scheduler(optimizer, warmup: int, total_opt_steps: int, min_lr_ratio: float):
    """Linear warmup → cosine decay to min_lr_ratio of peak."""

    def lr_lambda(step: int) -> float:
        if step < warmup:
            return step / max(1, warmup)
        if min_lr_ratio >= 1.0:
            return 1.0
        progress = (step - warmup) / max(1, total_opt_steps - warmup)
        progress = min(1.0, progress)
        return min_lr_ratio + (1 - min_lr_ratio) * 0.5 * (1 + math.cos(math.pi * progress))

    return torch.optim.lr_scheduler.LambdaLR(optimizer, lr_lambda=lr_lambda)
