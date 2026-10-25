"""Unit tests for the INSTABILITY / DIVERGENCE gate reductions (audit P0.4 §E, E1).

The shared streaming-loop machinery — the pre-clip grad-norm capture and the divergence stop — is
already guarded by ``test_loop.py`` (a real pathological-LR run). But the *gate arithmetic* that turns
a per-step trace into the two-sided divergence verdict had **no** regression guard, which the P0.4
code-validity audit (``docs/experiments/audits/P0.4.md`` §E, E1) flagged: a future mis-wire (a wrong
window, a broken ``min``-over-firing-arms selection, a sign error in the lead time) would still pass
CI. These drive the pure reductions on hand-built synthetic traces — **no** harness run — so each
load-bearing property is asserted directly:

* ``_summarize`` reads the steady band, excluding the no-warmup init transient (``init_skip``), and
  flags the first non-finite step;
* ``_lead_time`` yields a *positive* lead when the grad-norm spike precedes the loss NaN, and ``None``
  when there is no runaway;
* ``_evaluate_instability_gate`` computes ``separation = min(firing-arm peaks) / healthy_peak`` — the
  ``min`` taken over *firing* arms only (the A3 truncation property) — and passes iff the healthy arm
  is stable + learns and the top-LR PC diverges by ``≥ min_separation``.
"""

from __future__ import annotations

import importlib.util
import math
from pathlib import Path
from types import ModuleType
from typing import Any

import pytest
from hydra import compose, initialize_config_dir
from hydra.core.global_hydra import GlobalHydra
from omegaconf import DictConfig

_REPO_ROOT = Path(__file__).resolve().parents[2]
_CONFIG_DIR = _REPO_ROOT / "cafl4ds" / "configs"


def _load_harness() -> ModuleType:
    """Import ``scripts/positive_control_instability.py`` as a module (it is a script, not a package)."""
    path = _REPO_ROOT / "scripts" / "positive_control_instability.py"
    spec = importlib.util.spec_from_file_location("positive_control_instability_harness", path)
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


@pytest.fixture(scope="module")
def h() -> ModuleType:
    """The imported instability-harness module (shared across the test module)."""
    return _load_harness()


@pytest.fixture(scope="module")
def gate_config() -> DictConfig:
    """The composed ``positive_control_instability`` config — for its ``gate`` thresholds."""
    GlobalHydra.instance().clear()  # isolate from any other hydra-using test
    with initialize_config_dir(version_base=None, config_dir=str(_CONFIG_DIR)):
        return compose(config_name="positive_control_instability")


def _trace(losses: list[float], gnorms: list[float]) -> list[dict[str, Any]]:
    """Build a per-step loss trace; ``finite`` is derived from the loss + grad norm being finite."""
    return [
        {
            "series": "loss",
            "step": i,
            "loss": loss,
            "grad_norm": gnorm,
            "finite": math.isfinite(loss) and math.isfinite(gnorm),
        }
        for i, (loss, gnorm) in enumerate(zip(losses, gnorms, strict=True))
    ]


def _summary(lr: float, *, peak: float, nonfinite: int | None) -> dict[str, Any]:
    """A minimal arm summary sufficient for the gate reduction (the fields it reads)."""
    return {
        "lr": lr,
        "peak_grad_norm": peak,
        "median_grad_norm": min(peak, 0.1),
        "first_nonfinite_step": nonfinite,
        "init_loss": 1.0,
        "final_loss": 0.5 if nonfinite is None else float("nan"),
        "n_steps": 8,
    }


def test_summarize_excludes_init_transient(h: ModuleType) -> None:
    """The no-warmup first-gradient spike (steps 0-2) must not count as the steady-state peak."""
    # A big decaying init transient, then a low steady band — a HEALTHY arm, not a runaway.
    trace = _trace(losses=[1.0] * 8, gnorms=[9.0, 5.0, 2.0, 0.1, 0.12, 0.09, 0.11, 0.1])
    summ = h._summarize(trace, init_skip=3)
    assert summ["first_nonfinite_step"] is None
    assert summ["peak_grad_norm"] == pytest.approx(0.12)  # steady band, NOT the 9.0 init spike
    assert summ["init_loss"] == pytest.approx(1.0)
    assert summ["final_loss"] == pytest.approx(1.0)


def test_summarize_flags_first_nonfinite(h: ModuleType) -> None:
    """A trace that goes non-finite mid-run records the *first* such step as the divergence event."""
    trace = _trace(losses=[1.0, 1.0, 2.0, float("nan"), 1.0], gnorms=[0.1, 0.2, 50.0, float("inf"), 0.1])
    summ = h._summarize(trace, init_skip=3)
    assert summ["first_nonfinite_step"] == 3


def test_lead_time_positive_when_spike_precedes_nan(h: ModuleType) -> None:
    """The payoff read: a grad-norm blow-up crossing the level *before* the loss NaN gives lead > 0."""
    # steady ~0.1 through init_skip, crosses threshold=5 at step 4, loss NaN at step 7 -> lead 3.
    trace = _trace(
        losses=[1.0, 1.0, 1.0, 1.0, 1.2, 2.0, 4.0, float("nan")],
        gnorms=[8.0, 3.0, 0.1, 0.2, 6.0, 30.0, 200.0, float("inf")],
    )
    lead = h._lead_time(trace, threshold=5.0, init_skip=3)
    assert lead["spike_step"] == 4  # first cross of 5.0 at/after init_skip (the step-0 8.0 is skipped)
    assert lead["nonfinite_step"] == 7
    assert lead["lead"] == 3


def test_lead_time_none_without_runaway(h: ModuleType) -> None:
    """A quiet arm that never diverges has no lead time."""
    trace = _trace(losses=[1.0] * 6, gnorms=[0.1, 0.1, 0.1, 0.1, 0.1, 0.1])
    lead = h._lead_time(trace, threshold=5.0, init_skip=3)
    assert lead["nonfinite_step"] is None
    assert lead["lead"] is None


def test_gate_separation_is_min_over_firing_arms(h: ModuleType, gate_config: DictConfig) -> None:
    """``separation = min(firing peaks) / healthy peak`` — the ``min`` over *firing* arms only (A3).

    A non-firing arm with a large peak must be excluded from the ``min``; an arm that NaNs early with a
    small recorded peak is exactly what the ``min`` picks (the truncation property the audit confirmed).
    """
    healthy = _summary(1.0, peak=0.4, nonfinite=None)
    sweep = [
        _summary(0.1, peak=999.0, nonfinite=None),  # a big peak but does NOT fire -> excluded
        _summary(3.0, peak=6.0e8, nonfinite=8),  # fires, large peak
        _summary(10.0, peak=5.0e5, nonfinite=5),  # fires, SMALLER peak (NaN'd sooner) -> the min
    ]
    gate = h._evaluate_instability_gate(gate_config, healthy, sweep)
    assert gate["min_diverging_peak_grad_norm"] == pytest.approx(5.0e5)  # min over FIRING arms
    assert gate["grad_norm_separation"] == pytest.approx(5.0e5 / 0.4)
    assert gate["threshold_lr"] == 3.0  # smallest LR that diverges


def test_gate_passes_two_sided(h: ModuleType, gate_config: DictConfig) -> None:
    """Healthy stable + learns, top PC diverges, separation ≥ min_separation -> PASS."""
    healthy = _summary(1.0, peak=0.4, nonfinite=None)
    healthy["final_loss"], healthy["init_loss"] = 0.5, 1.0  # descends past 0.9x
    sweep = [_summary(1.0, peak=0.5, nonfinite=None), _summary(30.0, peak=4.0e12, nonfinite=4)]
    gate = h._evaluate_instability_gate(gate_config, healthy, sweep)
    assert gate["checks"] == {
        "healthy_stable": True,
        "healthy_learns": True,
        "pc_diverges": True,
        "grad_norm_separates": True,
    }
    assert gate["passed"] is True


def test_gate_fails_when_healthy_does_not_learn(h: ModuleType, gate_config: DictConfig) -> None:
    """A quiet-but-non-learning healthy arm is a degenerate baseline — the gate must not pass."""
    healthy = _summary(1.0, peak=0.4, nonfinite=None)
    healthy["final_loss"], healthy["init_loss"] = 0.98, 1.0  # barely moves -> not learning
    sweep = [_summary(30.0, peak=4.0e12, nonfinite=4)]
    gate = h._evaluate_instability_gate(gate_config, healthy, sweep)
    assert gate["checks"]["healthy_learns"] is False
    assert gate["passed"] is False


def test_gate_fails_when_separation_below_threshold(h: ModuleType, gate_config: DictConfig) -> None:
    """A diverging arm whose recorded peak barely clears the healthy peak fails ``grad_norm_separates``.

    This is the AdamW-pole shape (``adamw_seed0``): a one-step NaN records a tiny pre-blow-up peak, so
    the separation stays ~1x and the gate correctly does not certify a two-sided divergence fire.
    """
    healthy = _summary(1.0, peak=1.5, nonfinite=None)
    healthy["final_loss"], healthy["init_loss"] = 0.87, 1.34
    sweep = [_summary(100.0, peak=1.94, nonfinite=1)]  # NaN@1, tiny recorded peak
    gate = h._evaluate_instability_gate(gate_config, healthy, sweep)
    assert gate["checks"]["pc_diverges"] is True  # the non-finite event still fires
    assert gate["checks"]["grad_norm_separates"] is False  # but the separation does not
    assert gate["passed"] is False
