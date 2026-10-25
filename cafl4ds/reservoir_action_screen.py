"""Fixed paired actions and conservative opportunity summaries; no online selector."""

from __future__ import annotations

import numpy as np


def allocation(offset, arm):
    """Return current/history counts at a future arrival relative to a fixed parent."""
    if offset not in range(1, 129) or arm not in ("base", "recovery"):
        raise ValueError("Require base/recovery and offsets 1..128")
    current = 24 if arm == "base" or offset > 64 else (8 if offset <= 32 else 16)
    return current, 32 - current


def opportunity_screen(effects):
    """Describe material opposite preferences robust to the three inherited support draws."""
    values = np.asarray(effects, dtype=float)
    if values.shape != (8, 3, 3) or not np.isfinite(values).all():
        raise ValueError("Require eight parents, three draws and balanced/day/night effects")
    pooled = values.mean(1)
    recovery = (pooled[:, 0] >= 0.5) & (pooled[:, 1:].min(1) >= -1)
    base = (pooled[:, 0] <= -0.5) & (pooled[:, 1:].max(1) <= 1)
    robust_recovery = recovery & (values[:, :, 0] > 0).all(1) & (values[:, :, 1:] >= -1).all((1, 2))
    robust_base = base & (values[:, :, 0] < 0).all(1) & (values[:, :, 1:] <= 1).all((1, 2))
    gates = {
        "at_least_two_robust_recovery_parents": int(robust_recovery.sum()) >= 2,
        "at_least_two_robust_base_parents": int(robust_base.sum()) >= 2,
    }
    return {
        "passed": all(gates.values()),
        "gates": gates,
        "pooled_recovery_count": int(recovery.sum()),
        "pooled_base_count": int(base.sum()),
        "robust_recovery_count": int(robust_recovery.sum()),
        "robust_base_count": int(robust_base.sum()),
        "robust_recovery": robust_recovery,
        "robust_base": robust_base,
        "mean_effect": pooled.mean(0),
        "mean_minimum_domain_effect": pooled[:, 1:].min(1).mean(),
        "optimistic_balanced_headroom_over_best_static": float(
            np.maximum(pooled[:, 0], 0).mean() - max(float(pooled[:, 0].mean()), 0)
        ),
        "verdict": "replicate_action_opportunity" if all(gates.values()) else "action_opportunity_not_established",
        "controller_qualified": False,
    }
