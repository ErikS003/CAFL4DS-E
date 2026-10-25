"""Known-answer tests for the warmup+cosine LR schedule (P0.2.1).

The schedule is a multiplier on the base LR, stepped once per optimizer step: it ramps
linearly to the base LR over the warmup fraction, then follows a half-cosine down to
``min_lr_frac`` at the final step. These tests pin the shape at the boundaries.
"""

import pytest
import torch

from cafl4ds.schedule import warmup_cosine_schedule

_BASE_LR = 0.1


def _lrs(total_steps: int, warmup_frac: float, min_lr_frac: float = 0.0) -> list[float]:
    """Return the LR seen at each of ``total_steps`` steps (stepping after each)."""
    param = torch.nn.Parameter(torch.zeros(1))
    opt = torch.optim.SGD([param], lr=_BASE_LR)
    sched = warmup_cosine_schedule(opt, total_steps, warmup_frac=warmup_frac, min_lr_frac=min_lr_frac)
    seen = []
    for _ in range(total_steps):
        seen.append(opt.param_groups[0]["lr"])
        opt.step()  # no grads -> a no-op, but keeps the optimizer-before-scheduler call order
        sched.step()
    return seen


def test_warmup_ramps_then_cosine_decays_to_zero() -> None:
    """LR is small at the start, peaks at ~base LR after warmup, and decays toward 0."""
    lrs = _lrs(total_steps=100, warmup_frac=0.1)
    assert lrs[0] < lrs[10] <= _BASE_LR + 1e-9  # warming up
    assert lrs[10] == pytest.approx(_BASE_LR, rel=1e-6)  # peak at end of warmup
    assert lrs[-1] < 0.05 * _BASE_LR  # cosine has decayed nearly to zero
    # Monotone non-increasing after the warmup peak.
    post = lrs[10:]
    assert all(a >= b - 1e-9 for a, b in zip(post, post[1:], strict=False))


def test_min_lr_frac_is_the_floor() -> None:
    """With a non-zero ``min_lr_frac`` the decay bottoms out at that fraction of base LR."""
    lrs = _lrs(total_steps=50, warmup_frac=0.0, min_lr_frac=0.2)
    assert min(lrs) >= 0.2 * _BASE_LR - 1e-9
    assert lrs[-1] == pytest.approx(0.2 * _BASE_LR, abs=1e-3)


def test_no_warmup_starts_at_base_lr() -> None:
    """``warmup_frac=0`` starts at (approximately) the full base LR on the first step."""
    lrs = _lrs(total_steps=20, warmup_frac=0.0)
    assert lrs[0] == pytest.approx(_BASE_LR, rel=1e-6)


def test_invalid_args_raise() -> None:
    """Non-positive ``total_steps`` or out-of-range ``warmup_frac`` is rejected."""
    param = torch.nn.Parameter(torch.zeros(1))
    opt = torch.optim.SGD([param], lr=_BASE_LR)
    with pytest.raises(ValueError):
        warmup_cosine_schedule(opt, 0)
    with pytest.raises(ValueError):
        warmup_cosine_schedule(opt, 10, warmup_frac=1.0)
