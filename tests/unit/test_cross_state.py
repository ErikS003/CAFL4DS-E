import pytest

from cafl4ds.cross_state import CASES, STATES, paired_interval, preference


def test_design_and_family_intervals():
    assert len(CASES) == 20 and set(STATES.values()) == {4224, 8448}
    ordinary = paired_interval([1, 2, 3, 4, 5])
    adjusted = paired_interval([1, 2, 3, 4, 5], True)
    assert ordinary["mean_pp"] == 3 and adjusted["ci95_pp"][0] < ordinary["ci95_pp"][0]
    with pytest.raises(ValueError):
        paired_interval([1, 2, 3])


def test_joint_domain_guards_prevent_balanced_only_preference():
    contrasts = {m: {"mean_pp": 2, "ci95_pp": [1, 3]} for m in ("balanced", "A", "B")}
    assert preference(contrasts) == {"full": True, "hold": False}
    contrasts["B"] = {"mean_pp": -2, "ci95_pp": [-3, -1]}
    assert preference(contrasts) == {"full": False, "hold": False}
