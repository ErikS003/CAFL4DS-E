"""Image-independent update masks and causal current-only optimization."""

from __future__ import annotations

import torch

from cafl4ds.ssl.base import SSLMethod

MASK_SEED_OFFSET = 62003


def update_mask(policy: str, seed: int, steps: int = 576) -> list[bool]:
    """Build masks with a private generator; never consume the learner RNG."""
    if steps <= 0 or steps % 144:
        raise ValueError("Expected complete 144-arrival blocks")
    if policy == "always":
        return [True] * steps
    if policy == "fixed_half":
        return [(index + 1) % 2 == 0 for index in range(steps)]
    if policy != "random_half":
        raise ValueError(f"Unknown policy: {policy}")
    generator = torch.Generator().manual_seed(seed + MASK_SEED_OFFSET)
    mask = [False] * steps
    for start in range(0, steps, 144):
        for index in torch.randperm(144, generator=generator)[:72].tolist():
            mask[start + index] = True
    return mask


def optimize_current(
    model: SSLMethod, optimizer: torch.optim.Optimizer, current: torch.Tensor, accepted: bool
) -> tuple[float, float] | None:
    """Rejected arrivals leave model, optimizer and SSL RNG completely untouched."""
    if not accepted:
        return None
    if len(current) != 32:
        raise ValueError("Expected exactly 32 current images")
    model.train()
    optimizer.zero_grad()
    loss = model.training_step(current)
    if not torch.isfinite(loss):
        raise RuntimeError("Nonfinite SSL loss")
    torch.autograd.backward(loss)
    norm = torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0, error_if_nonfinite=True)
    optimizer.step()
    return float(loss.detach()), float(norm)
