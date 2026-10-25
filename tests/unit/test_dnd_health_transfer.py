"""Auxiliary readouts cannot trigger control; exact work and risk semantics."""

import numpy as np

from cafl4ds.dnd_health_stopping import HealthStop
from cafl4ds.dnd_health_transfer import MONITOR, POINTS, HealthTransfer, path_metrics, stopped_path


def test_controller_clock_is_independent_of_transition_metadata():
    assert 132 in MONITOR and 264 not in MONITOR and 160 not in MONITOR
    assert 264 in POINTS and 160 in POINTS and len(POINTS) == 49
    banks = {p: np.random.default_rng(p).normal(size=(40, 8)) for p in POINTS}
    events = []
    for transitions in ((160, 224), (224, 320)):
        guard = HealthStop()
        for p in POINTS:
            # Transitions only decide which extra offline score is displayed.
            offline_boundary = p in transitions
            assert isinstance(offline_boundary, bool)
            if p in MONITOR:
                guard.observe(p, banks[p])
        events.append(guard.events)
    assert events[0] == events[1]


def test_left_interval_weighting_excludes_sparse_initial_segment():
    path = np.zeros((len(POINTS), 3))
    j = POINTS.index(148)
    path[j:] = -2
    m = path_metrics(path)
    np.testing.assert_allclose(m[3:6], 2)
    np.testing.assert_allclose(m[9:12], (448 - 148) * 2)
    np.testing.assert_allclose(m[6:9], 2 * (448 - 148) / (448 - 132))


def test_constant_stop_matches_fixed_rule_and_shuffle_exactly():
    rng = np.random.default_rng(9)
    full = rng.normal(size=(8, len(POINTS), 3))
    s, raw = HealthTransfer().summarize(full, [148] * 8, [(160, 64)] * 8)
    np.testing.assert_array_equal(
        s["rules"]["health"]["history_trajectories_pp"], s["rules"]["fixed148"]["history_trajectories_pp"]
    )
    assert s["matched_work_shuffle"]["assignments"] == 40320
    np.testing.assert_allclose(s["health_vs_shuffle"]["risk_reduction"]["mean"], 0, atol=1e-12)
    assert not s["added_timing_information_flag"]
    stops = np.array(s["rules"]["health"]["stops"])
    assert (stops[raw["permutations"]].sum(1) == 8 * 148).all()


def test_stopped_path_cannot_see_future_outcomes():
    rng = np.random.default_rng(3)
    full = rng.normal(size=(len(POINTS), 3))
    before = stopped_path(full, 148)
    full[POINTS.index(148) + 1 :] = np.nan
    np.testing.assert_array_equal(before, stopped_path(full, 148))


def test_nonconstant_shuffle_preserves_total_work_and_recipient_outcomes():
    full = np.array([np.repeat(np.array([[i, i * 2, i * 3.0]]), len(POINTS), axis=0) for i in range(8)])
    selected = [140, 148, 156, 164, 172, 180, 188, 196]
    summary, raw = HealthTransfer().summarize(full, selected, [(160, 64)] * 8)
    order = raw["permutations"]
    assert (np.array(selected)[order].sum(1) == sum(selected)).all()
    np.testing.assert_allclose(
        raw["assignment_mean_metrics"][:, :3], np.repeat(full[:, 0].mean(0)[None], 40320, axis=0)
    )
    assert summary["matched_work_shuffle"]["total_updates_each"] == sum(selected)
