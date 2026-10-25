"""Reader nonlinearities, history exclusion, matched work and robustness decisions."""

import copy

import numpy as np
import pytest

from cafl4ds.dnd_warning_robustness import (
    COHORTS,
    WarningRobustness,
    metrics,
    reader_targets,
    recalibrated_age_risk,
    robustness_screen,
    sign_check,
)


def test_pooling_scores_before_maximum_is_not_average_reader_damage() -> None:
    """Different reader loss dates can be suppressed by pooled readout scores."""
    panels = np.full((2, 3, 3, 3), 50.0)
    panels[:, 1, 0, 1] = 48
    panels[:, 2, 1, 2] = 48
    args = ((140, 148, 156), np.array([140]), 16)
    first = reader_targets(panels, (0,), *args)
    second = reader_targets(panels, (1,), *args)
    pooled = reader_targets(panels, (0, 1), *args)
    np.testing.assert_array_equal((first[..., 0] + second[..., 0]) / 2, 2)
    np.testing.assert_array_equal(pooled[..., 0], 1)
    np.testing.assert_array_equal(pooled, reader_targets(panels, (1, 0), *args))


def test_recipient_and_omitted_source_cannot_calibrate_recipient_age() -> None:
    """Age control predictions exclude both the recipient and a removed source."""
    rng = np.random.default_rng(11)
    targets = rng.uniform(size=(8, 5, 4))
    kept = COHORTS["omit_07"]
    before = recalibrated_age_risk(targets, kept)
    targets[7] = 1000
    np.testing.assert_array_equal(before, recalibrated_age_risk(targets, kept))
    targets[2] = 500
    after = recalibrated_age_risk(targets, kept)
    np.testing.assert_array_equal(before[2], after[2])
    assert not np.array_equal(before[0], after[0])


def test_costs_keep_both_domain_opportunities_and_undefined_empty_precision() -> None:
    """Balanced cancellation cannot hide day or night learning forgone."""
    truth = np.array([[[1, 0, 2, -2], [1, 0, -2, 2]]], float)
    stats = metrics(truth, np.ones((1, 2)))
    np.testing.assert_array_equal(stats["positive_learning_per_decision_pp"], [[0, 1, 1]])
    empty = metrics(truth, np.zeros((1, 2)))
    assert empty["harm_precision"] is None
    np.testing.assert_array_equal(empty["positive_learning_per_decision_pp"], 0)


def good_screen_cells() -> list[dict]:
    """Make a complete synthetic grid with beneficial means and clear intervals."""
    return [
        {
            "horizon": 32,
            "instrument": "combined",
            "reader": reader,
            "cohort": cohort,
            "sign_pass": True,
            "comparisons": {
                name: {"loss_advantage_pp": {"ci95": [0.1, 0.3]}} for name in ("age_recalibrated", "uniform")
            },
        }
        for reader in ("draw_61001", "draw_61007", "draw_61013", "pooled")
        for cohort in COHORTS
    ]


def test_pooled_success_cannot_rescue_a_failed_single_reader_source_cross() -> None:
    """A favorable average must not bypass the declared reader/source stresses."""
    cells = good_screen_cells()
    assert robustness_screen(cells)["passed"]
    cells[1]["sign_pass"] = False
    result = robustness_screen(cells)
    assert not result["passed"] and result["sign_cells_passed"] == 35
    assert result["singleton_full_sign_pass"] and result["pooled_omission_sign_pass"]


def test_missing_stress_or_uncertain_uniform_advantage_cannot_pass() -> None:
    """The full grid and a positive uniform-control interval are necessary."""
    cells = good_screen_cells()
    with pytest.raises(ValueError, match="36-cell"):
        robustness_screen(cells[:-1])
    changed = copy.deepcopy(cells)
    next(row for row in changed if row["reader"] == "pooled" and row["cohort"] == "all")["comparisons"]["uniform"][
        "loss_advantage_pp"
    ]["ci95"][0] = -0.1
    assert not robustness_screen(changed)["passed"]


def test_both_control_comparisons_are_required_for_beneficial_signs() -> None:
    """A missing baseline must not silently pass an empty comparison."""
    with pytest.raises(ValueError, match="Both matched controls"):
        sign_check({})


def test_every_reader_and_source_omission_preserves_warning_counts() -> None:
    """Full analysis keeps per-history budgets and cannot select omitted histories."""
    points = tuple(range(140, 189, 8))
    panels = np.full((8, len(points), 3, 3), 50.0)
    warning = np.arange(3)[None] < (np.arange(8) % 4)[:, None]
    alarms = {32: np.repeat(warning[..., None], 4, axis=-1)}
    frozen = {32: np.repeat(warning[None], 4, axis=0)}
    summary, arrays = WarningRobustness().analyze(panels, points, {32: np.array(points[:3])}, alarms, frozen, [])
    for row in summary["cells"]:
        ids = list(row["histories"])
        for name in ("health", "age_recalibrated", "age_frozen", "uniform"):
            np.testing.assert_allclose(row["metrics"][name]["warning_count"], warning[ids].sum())
    selections = arrays["age_selections_32"]
    for c, ids in enumerate(COHORTS.values()):
        omitted = [i for i in range(8) if i not in ids]
        if omitted:
            assert not selections[:, c, :, omitted].any()
    assert not summary["robustness_screen"]["passed"]
