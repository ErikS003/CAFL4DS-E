"""Tests for the P0.2.2 collapse-suite envelope reporting layer.

Confirms that :func:`metric_envelope` orients each instrument's separation in its own collapse
direction (ratio vs. gap), fires only for standalone detectors that separate the right way,
detects the projector (``_proj``) surface as its own row, and never fires on the non-standalone
alignment metric — and that the table renderer handles all of that without crashing.
"""

from typing import Any

from cafl4ds.metric_envelope import metric_envelope, quiet_verdict, render_envelope_table

# Two checkpoints per arm; only the FINAL value drives the reading, so the first row is filler.
# Finals chosen so every standalone instrument separates in its collapse direction on `backbone`
# but rankme's `proj` surface stays quiet (a non-firing case), and alignment "looks aligned"
# under collapse (small gap) yet must never fire because it is not a standalone detector.
_FILLER = {
    "step": 0.0,
    "rankme": 6.0,
    "rankme_proj": 4.0,
    "mean_feature_var": 0.5,
    "offdiag_cov": 0.5,
    "uniformity": -2.0,
    "alignment": 0.5,
}
_HEALTHY = [
    _FILLER,
    {
        "step": 1.0,
        "rankme": 5.0,
        "rankme_proj": 4.0,
        "mean_feature_var": 0.8,
        "offdiag_cov": 0.1,
        "uniformity": -3.0,
        "alignment": 0.6,
    },
]
_PC = [
    _FILLER,
    {
        "step": 1.0,
        "rankme": 2.0,
        "rankme_proj": 3.5,
        "mean_feature_var": 0.1,
        "offdiag_cov": 0.9,
        "uniformity": -0.5,
        "alignment": 0.05,
    },
]


def _by(rows: list[dict[str, Any]], metric: str, surface: str) -> dict[str, Any]:
    """The single row for a given (metric, surface)."""
    (row,) = [r for r in rows if r["metric"] == metric and r["surface"] == surface]
    return row


def test_ratio_and_gap_orientation() -> None:
    """Each instrument's separation is oriented in its own collapse direction."""
    rows = metric_envelope(_PC, _HEALTHY)
    # rankme (down, ratio healthy/PC): 5.0 / 2.0 = 2.5.
    assert _by(rows, "rankme", "backbone")["separation"] == 2.5
    # mean_feature_var (down, ratio healthy/PC): 0.8 / 0.1 = 8.0.
    assert _by(rows, "mean_feature_var", "backbone")["separation"] == 8.0
    # offdiag_cov (up, ratio PC/healthy): 0.9 / 0.1 = 9.0.
    assert _by(rows, "offdiag_cov", "backbone")["separation"] == 9.0
    # uniformity (up, gap PC-healthy): -0.5 - (-3.0) = 2.5.
    assert _by(rows, "uniformity", "backbone")["separation"] == 2.5


def test_fires_only_for_standalone_separating_the_right_way() -> None:
    """`fires` is True for standalone detectors past their bar, False otherwise."""
    rows = metric_envelope(_PC, _HEALTHY, min_ratio=2.0, min_gap=0.5)
    assert _by(rows, "rankme", "backbone")["fires"] is True
    assert _by(rows, "mean_feature_var", "backbone")["fires"] is True
    assert _by(rows, "offdiag_cov", "backbone")["fires"] is True
    assert _by(rows, "uniformity", "backbone")["fires"] is True
    # rankme on the projector surface does not separate (4.0 / 3.5 ~= 1.14 < 2.0): quiet.
    assert _by(rows, "rankme", "proj")["fires"] is False
    # alignment is not a standalone detector -> never fires, whatever the gap.
    align = _by(rows, "alignment", "backbone")
    assert align["standalone"] is False
    assert align["fires"] is False


def test_projector_surface_detected_as_its_own_row() -> None:
    """A `_proj` key yields a distinct `surface='proj'` row alongside the backbone one."""
    rows = metric_envelope(_PC, _HEALTHY)
    surfaces = {r["surface"] for r in rows if r["metric"] == "rankme"}
    assert surfaces == {"backbone", "proj"}


def test_absent_metric_yields_no_row() -> None:
    """Instruments with no key present in the records are simply omitted (no crash)."""
    healthy = [{"step": 0.0, "rankme": 5.0}]
    pc = [{"step": 0.0, "rankme": 2.0}]
    rows = metric_envelope(pc, healthy)
    assert {r["metric"] for r in rows} == {"rankme"}


def test_quiet_verdict_fails_when_any_instrument_fires() -> None:
    """P0.2.4 quiet verdict: an atypical arm that separates past a bar is a false fire (not passed)."""
    # _PC in the pc slot separates every standalone instrument on backbone -> false fires.
    rows = metric_envelope(_PC, _HEALTHY, min_ratio=2.0, min_gap=0.5)
    verdict = quiet_verdict(rows)
    assert verdict["mode"] == "quiet"
    assert verdict["passed"] is False
    assert verdict["num_fired"] == len(verdict["fired"]) > 0
    fired = {(f["metric"], f["surface"]) for f in verdict["fired"]}
    assert ("rankme", "backbone") in fired  # a firing standalone detector is reported
    assert ("alignment", "backbone") not in fired  # non-standalone never fires


def test_quiet_verdict_passes_when_nothing_fires() -> None:
    """When the atypical arm matches the reference, no instrument fires -> the quiet side holds."""
    verdict = quiet_verdict(metric_envelope(_HEALTHY, _HEALTHY))
    assert verdict["passed"] is True
    assert verdict["num_fired"] == 0
    assert verdict["fired"] == []


def test_render_table_runs_and_marks_paired() -> None:
    """The renderer produces a header + one line per row, marking alignment as '(paired)'."""
    rows = metric_envelope(_PC, _HEALTHY)
    table = render_envelope_table(rows)
    lines = table.splitlines()
    assert lines[0].split()[:2] == ["metric", "surface"]  # header
    assert len(lines) == 2 + len(rows)  # header + separator + one per row
    assert "(paired)" in table  # alignment's non-standalone marker
