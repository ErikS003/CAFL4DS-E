"""Complete-state restoration and predeclared source/seed replication summaries."""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np

from cafl4ds.reservoir_action_screen import opportunity_screen
from cafl4ds.short_buffer import capture as basic_capture
from cafl4ds.short_buffer import restore as basic_restore


def guard_learning(study_root, out, replay):
    """Reject repeated stages and any launch exceeding the physical SSL budget."""
    prefix = "rerun_" if replay else "collect_"
    if any((Path(out) / "attempts").glob(prefix + "*.json")):
        raise RuntimeError("Learning stage already attempted; preserve it rather than repeat updates")
    consumed = 0
    for path in Path(study_root).glob("block*-seed*/attempts/*.json"):
        consumed += json.loads(path.read_text(encoding="utf-8")).get("ssl_started", 0)
    if consumed + 3072 > 55296:
        raise RuntimeError("The physical SSL-start budget cannot complete this fixed stage")


def capture(model, optimizer):
    """Retain gradients as well as the inherited learner, Adam, modes and RNG."""
    state = basic_capture(model, optimizer)
    state["gradients"] = {n: None if p.grad is None else p.grad.detach().clone() for n, p in model.named_parameters()}
    return state


def restore(initial, optim_config, state):
    """Restore an independent learner including any saved parameter gradients."""
    model, optimizer = basic_restore(initial, optim_config, state)
    gradients = state.get("gradients", {})
    for name, parameter in model.named_parameters():
        value = gradients.get(name)
        parameter.grad = None if value is None else value.to(parameter.device).clone()
    return model, optimizer


def aggregate(effects):
    """Apply the declared gates to block, SSL seed, parent, support draw, component."""
    values = np.asarray(effects, dtype=float)
    if values.shape != (3, 3, 8, 3, 3) or not np.isfinite(values).all():
        raise ValueError("Require 3 blocks x 3 SSL draws x 8 parents x 3 support draws x 3 components")
    runs = values.mean(axis=(2, 3))
    blocks = runs.mean(1)
    domain_by_draw = values[..., 1:].mean(axis=(1, 2))
    gates = {
        "overall_balanced_at_least_half_pp": bool(blocks[:, 0].mean() >= 0.5),
        "every_block_positive": bool((blocks[:, 0] > 0).all()),
        "two_positive_seeds_per_block": bool(((runs[:, :, 0] > 0).sum(1) >= 2).all()),
        "block_draw_domain_average_budget": bool((domain_by_draw >= -1).all()),
    }
    recovery, normal, screens = [], [], []
    for block in values:
        cases = [opportunity_screen(seed) for seed in block]
        a = np.asarray([c["robust_recovery"] for c in cases])
        b = np.asarray([c["robust_base"] for c in cases])
        recovery.append((a.sum(0) >= 2) & ~b.any(0))
        normal.append((b.sum(0) >= 2) & ~a.any(0))
        screens.append(cases)
    recovery, normal = np.asarray(recovery), np.asarray(normal)
    opportunity_blocks = (recovery.sum(1) >= 2) & (normal.sum(1) >= 2)
    return {
        "benefit_passed": all(gates.values()),
        "benefit_gates": gates,
        "opportunity_passed": bool(opportunity_blocks.sum() >= 2),
        "opportunity_blocks": opportunity_blocks,
        "stable_recovery": recovery,
        "stable_normal": normal,
        "per_run_effects": runs,
        "per_block_effects": blocks,
        "overall_effect": blocks.mean(0),
        "block_draw_domain_effects": domain_by_draw,
        "per_run_screens": screens,
        "worst_domain_effect": float(values[..., 1:].min()),
        "fraction_domain_effects_below_minus_one": float((values[..., 1:] < -1).mean()),
        "controller_qualified": False,
    }


def protection(curve, warm):
    """Grid-based per-draw/per-domain branch deficits at 0,32,64,128."""
    a, anchor = np.asarray(curve, dtype=float), np.asarray(warm, dtype=float)
    assert a.shape == (4, 3, 3) and anchor.shape == (3, 3)
    dd = np.maximum.accumulate(a, axis=0) - a
    return {
        "endpoint_minus_M0": a[-1] - anchor,
        "maximum_drawdown": dd.max(0),
        "maximum_loss_below_M0": np.maximum(anchor[None] - a, 0).max(0),
        "arrival_weighted_drawdown": np.trapezoid(dd, x=[0, 32, 64, 128], axis=0) / 128,
    }
