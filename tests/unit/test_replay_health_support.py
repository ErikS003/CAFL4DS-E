"""Group isolation and fitting-only normalization for support-trained health control."""
from __future__ import annotations

import copy
from typing import Any

import numpy as np
import torch

from cafl4ds.replay_health_support import precise_probe, train_choices
from cafl4ds.replay_local import RECIPES


def test_heldout_pool_targets_cannot_change_health_choices() -> None:
    """Changing every held-out target leaves operational predictions identical."""
    fields = sorted(set().union(*map(set, RECIPES.values())))
    features: list[dict[str, Any]] = []
    rows: list[dict[str, Any]] = []
    for group in range(3):
        for state in range(2):
            for action in (8, 16, 24):
                r = {"base_state": f"{group}_{state}", "pool_group": str(group), "action": action,
                     "features": {k: float(group + state + action / 32) for k in fields}}
                features.append(r)
                if action != 16:
                    rows.append({**r, "target_name": "accuracy", "target": (action - 16) / 1000})
    before, models = train_choices(rows, features, "2", "accuracy")
    changed = copy.deepcopy(rows)
    for r in changed:
        if r["pool_group"] == "2":
            r["target"] = 1000000
    after, other = train_choices(changed, features, "2", "accuracy")
    assert before == after
    assert models == other
    assert all(not s.startswith("2_") for c in before for s in c["training_states"])


def test_precision_probe_excludes_scoring_features_and_labels() -> None:
    """A scoring-panel perturbation cannot affect the fitted head."""
    generator = torch.Generator().manual_seed(71021)
    z = torch.randn(36, 4, generator=generator)
    labels = torch.arange(36) % 3
    fitting = torch.arange(24)
    a = precise_probe(z, labels, fitting, 10.0)
    changed_z, changed_labels = z.clone(), labels.clone()
    changed_z[24:] += 1000
    changed_labels[24:] = 2
    b = precise_probe(changed_z, changed_labels, fitting, 10.0)
    np.testing.assert_array_equal(a.mean, b.mean)
    np.testing.assert_array_equal(a.scale, b.scale)
    np.testing.assert_array_equal(a.classifier.coef_, b.classifier.coef_)
