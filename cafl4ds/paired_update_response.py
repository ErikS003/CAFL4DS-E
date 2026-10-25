"""Restore isolated causal SSL branches from complete saved decision states."""

from __future__ import annotations

import copy
import time
from typing import Any

import torch
from hydra.utils import instantiate
from omegaconf import DictConfig

from cafl4ds.ssl.base import SSLMethod
from cafl4ds.update_filtering import optimize_current


def identical(left: Any, right: Any) -> bool:  # noqa: ANN401 - recursive serialized optimizer structures
    """Compare complete optimizer/model trees exactly, including tensor-valued clocks."""
    if isinstance(left, torch.Tensor):
        return isinstance(right, torch.Tensor) and left.dtype == right.dtype and torch.equal(left, right)
    if isinstance(left, dict):
        return (
            isinstance(right, dict)
            and left.keys() == right.keys()
            and all(identical(value, right[key]) for key, value in left.items())
        )
    if isinstance(left, (list, tuple)):
        return (
            type(left) is type(right)
            and len(left) == len(right)
            and all(identical(a, b) for a, b in zip(left, right, strict=True))
        )
    return bool(left == right)


def restore(initial: SSLMethod, config: DictConfig, state: dict[str, Any]) -> tuple[SSLMethod, torch.optim.Optimizer]:
    """Avoid shared optimizer tensors across branches and restore the saved learning RNG."""
    model = copy.deepcopy(initial)
    model.load_state_dict(state["model"], strict=True)
    optimizer = instantiate(config.optim, params=[p for p in model.parameters() if p.requires_grad])
    optimizer.load_state_dict(copy.deepcopy(state["optimizer"]))
    model.train(state["model_training"])
    torch.set_rng_state(state["torch_rng"].clone())
    if not identical(model.state_dict(), state["model"]) or not identical(optimizer.state_dict(), state["optimizer"]):
        raise ValueError("Restored state differs from parent")
    return model, optimizer


def run_window(
    model: SSLMethod,
    optimizer: torch.optim.Optimizer,
    images: torch.Tensor,
    order: torch.Tensor,
    offset: int,
    parent_updates: int,
    policy: str,
    horizon: int = 36,
) -> dict[str, Any]:
    """Advance the same arrival window under full or even-index half optimization."""
    if policy not in {"full", "half"} or horizon <= 0 or offset + horizon > len(order) // 32:
        raise ValueError("Invalid branch window/action")
    ledger: list[dict[str, Any]] = []
    losses: list[dict[str, Any]] = []
    updates = 0
    for arrival in range(offset + 1, offset + horizon + 1):
        tick = time.perf_counter()
        ids = order[(arrival - 1) * 32 : arrival * 32]
        current = images[ids]
        accepted = policy == "full" or arrival % 2 == 0
        observed = optimize_current(model, optimizer, current, accepted)
        if observed is not None:
            updates += 1
            losses.append(
                {
                    "step": arrival,
                    "optimizer_step": parent_updates + updates,
                    "loss": observed[0],
                    "grad_norm": observed[1],
                }
            )
        row: dict[str, Any] = {
            "step": arrival,
            "current": ids.tolist(),
            "accepted": accepted,
            "branch_updates": updates,
            "optimizer_steps": parent_updates + updates,
        }
        row["online_seconds"] = time.perf_counter() - tick
        ledger.append(row)
    return {
        "ledger": ledger,
        "losses": losses,
        "updates": updates,
        "optimizer_images": updates * 32,
        "arrivals": horizon * 32,
        "online_seconds": sum(row["online_seconds"] for row in ledger),
    }
