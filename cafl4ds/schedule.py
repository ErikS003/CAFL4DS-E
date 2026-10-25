"""Learning-rate schedules for the training loops.

Phase-0 P0.1/P0.2 ran a **flat** learning rate over a single ~25-step pass. P0.2.1 found that
a genuine healthy SimSiam baseline (a RankMe that recovers and holds, rather than decaying
indistinguishably from a forced collapse) needs a *fair* training regime: multiple epochs with
a **warmup + cosine-decay** schedule. As the LR anneals, the intact model's effective rank
re-expands from the early-training dip — the healthy U-shape that separates it from the
collapsed arm. This module supplies that schedule as a plain, Hydra-instantiable factory.
"""

from __future__ import annotations

import math

from torch import optim
from torch.optim.lr_scheduler import LambdaLR


def warmup_cosine_schedule(
    optimizer: optim.Optimizer,
    total_steps: int,
    warmup_frac: float = 0.1,
    min_lr_frac: float = 0.0,
) -> LambdaLR:
    """Linear warmup then cosine decay, expressed as a multiplier on the base LR.

    The returned scheduler is stepped **once per optimizer step** (not per epoch). The
    multiplier ramps linearly from ``0`` to ``1`` over the first ``warmup_frac`` of
    ``total_steps``, then follows a half-cosine down to ``min_lr_frac`` at the final step.

    Args:
        optimizer: The optimizer whose base LR(s) are scaled.
        total_steps: Total number of optimizer steps over the whole run
            (``epochs * batches_per_epoch``).
        warmup_frac: Fraction of ``total_steps`` spent warming up, in ``[0, 1)``.
        min_lr_frac: Final LR as a fraction of the base LR (``0`` decays fully to zero).

    Returns:
        A :class:`~torch.optim.lr_scheduler.LambdaLR` implementing the schedule.

    Raises:
        ValueError: If ``total_steps`` is not positive or ``warmup_frac`` is out of range.
    """
    if total_steps <= 0:
        raise ValueError(f"total_steps must be positive, got {total_steps}.")
    if not 0.0 <= warmup_frac < 1.0:
        raise ValueError(f"warmup_frac must be in [0, 1), got {warmup_frac}.")
    warmup_steps = max(1, int(round(warmup_frac * total_steps)))

    def lr_lambda(step: int) -> float:
        if step < warmup_steps:
            return (step + 1) / warmup_steps
        progress = (step - warmup_steps) / max(1, total_steps - warmup_steps)
        cosine = 0.5 * (1.0 + math.cos(math.pi * min(1.0, progress)))
        return min_lr_frac + (1.0 - min_lr_frac) * cosine

    return LambdaLR(optimizer, lr_lambda)
