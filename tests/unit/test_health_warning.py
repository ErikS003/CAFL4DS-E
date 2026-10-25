"""Check future-effect definition and isolation across source histories."""

import numpy as np
import pytest

from cafl4ds.health_warning import fit, remaining_effect


def test_future_damage_excludes_damage_already_incurred():
    """An impaired landmark can nevertheless benefit from continuing."""
    effect = remaining_effect({"balanced": 50, "A": 45, "B": 55}, {"balanced": 52, "A": 46, "B": 58})
    assert effect["damage"] == -1
    assert effect["balanced_damage"] == -2


def test_worst_domain_preserves_tradeoff():
    """A balanced gain must not hide a domain loss."""
    effect = remaining_effect({"balanced": 50, "A": 50, "B": 50}, {"balanced": 51, "A": 55, "B": 47})
    assert effect["balanced"] == 1
    assert effect["damage"] == 3


def test_held_history_targets_do_not_change_its_predictions():
    """Labels in the held group cannot enter its fitted forecast."""
    points = [
        {"stream": str(g), "updates": g * 10 + j, "recent_loss": j + g / 10, "cka_drift": (j + g) / 10, "damage": g - j}
        for g in range(6)
        for j in range(3)
    ]
    fields = ("updates", "recent_loss", "cka_drift")
    first = fit(points, fields)
    altered = [{**p, "damage": p["damage"] + (100 if p["stream"] == "0" else 0)} for p in points]
    second = fit(altered, fields)
    np.testing.assert_array_equal(first["predictions"][:3], second["predictions"][:3])


def test_group_population_is_enforced():
    """Optimizer repeats cannot substitute for six histories."""
    with pytest.raises(ValueError, match="Six histories"):
        fit([{"stream": "one", "damage": 1}] * 18, ())
