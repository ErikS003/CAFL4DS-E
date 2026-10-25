"""Integration regression test for the collapse positive control (audit item E1).

The unit tests (``test_ssl.py``, ``test_barlow.py``) validate the collapse *mechanism* —
the predictor is bypassed, the stop-gradient detaches the target branch, the redundancy
term is dropped — but they explicitly **delegate** the *behavioural* contrast ("a
forced-collapse run actually collapses relative to a healthy arm") to
``scripts/positive_control.py`` (see the note in ``test_ssl.py``). The P0.2 code-validity
audit (``docs/experiments/audits/P0.2.md`` §E) flagged that this leaves the load-bearing
behaviour without a regression guard: a future edit that silently mis-wired an arm (both
arms ``anti_collapse=True``, a swapped ``pc``/``healthy`` slot, or a "fixed" predictor that
no longer collapses) would still pass CI.

This test closes that gap. It drives the **real harness** (`_run_arm`, the streaming loop,
the monitor, and the instruments) for both toggles on the fast synthetic source and asserts
the forced-collapse arm leaves its behavioural fingerprint:

* its SSL loss drives toward the ``-1`` constant-solution optimum while the healthy arm's
  stays near 0 — the discriminator at a *toy horizon*, where RankMe does not yet separate
  (that separation needs the P0.2.1 40-epoch IID regime; this is a wiring guard, not a
  re-run of the calibration);
* its **projector**-surface RankMe ends *below* the healthy arm's (directional collapse at
  the surface the suite is read on).

Fixed ``seed=1`` gives a decisive, deterministic margin (the loss-floor separation and the
projector-RankMe ordering hold across every seed tried; seed 1 is simply the cleanest).
"""

from __future__ import annotations

import importlib.util
from pathlib import Path
from types import ModuleType
from typing import Any

import pytest
from hydra import compose, initialize_config_dir
from hydra.core.global_hydra import GlobalHydra
from omegaconf import DictConfig

_REPO_ROOT = Path(__file__).resolve().parents[2]
_CONFIG_DIR = _REPO_ROOT / "cafl4ds" / "configs"

# Fast, network-free, deterministic: the synthetic source + a tiny image size + a short
# horizon; probes off (kNN/linear add wall-clock for no wiring value here). seed=1 gives the
# clearest forced-collapse loss dive (~-0.54 vs the healthy arm's ~-0.02).
_OVERRIDES = [
    "data=synthetic",
    "img_size=16",
    "epochs=4",
    "seed=1",
    "stream.support_per_class=8",
    "stream.query_per_class=8",
    "stream.era_eval_per_class=5",
    "monitor.run_knn=false",
    "monitor.run_linear=false",
]


def _load_harness() -> ModuleType:
    """Import ``scripts/positive_control.py`` as a module (it is a script, not a package)."""
    path = _REPO_ROOT / "scripts" / "positive_control.py"
    spec = importlib.util.spec_from_file_location("positive_control_harness", path)
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def _compose() -> DictConfig:
    """Compose the ``positive_control`` config with the fast-smoke overrides."""
    GlobalHydra.instance().clear()  # isolate from any other hydra-using test
    with initialize_config_dir(version_base=None, config_dir=str(_CONFIG_DIR)):
        return compose(config_name="positive_control", overrides=_OVERRIDES)


def _final(records: list[dict[str, Any]], key: str) -> float:
    """Last logged value of ``key`` across the health series."""
    return float([r[key] for r in records if key in r][-1])


@pytest.fixture(scope="module")
def arms(tmp_path_factory: pytest.TempPathFactory) -> dict[str, Any]:
    """Run both arms once (the collapse contrast) and share the result across assertions."""
    harness = _load_harness()
    config = _compose()
    out_dir = tmp_path_factory.mktemp("positive_control")
    # Mirror the harness's own arm mapping: anti_collapse=True is healthy, False is the PC.
    hc, hc_floor = harness._run_arm(config, anti_collapse=True, run_name="hc", out_dir=out_dir)
    pc, pc_floor = harness._run_arm(config, anti_collapse=False, run_name="pc", out_dir=out_dir)
    gate = harness._evaluate_point_collapse_gate(config, pc, hc, pc_floor, hc_floor)
    return {"hc": hc, "pc": pc, "hc_floor": hc_floor, "pc_floor": pc_floor, "gate": gate}


def test_forced_collapse_arm_drives_to_constant_solution(arms: dict[str, Any]) -> None:
    """The PC's loss rides toward the -1 optimum; the healthy arm's stays near 0."""
    pc_floor, hc_floor = arms["pc_floor"], arms["hc_floor"]
    assert pc_floor < -0.15, f"forced-collapse arm did not drive negative (loss floor {pc_floor:.3f})"
    assert hc_floor > -0.10, f"healthy arm unexpectedly collapsed (loss floor {hc_floor:.3f})"
    # The arms genuinely differ, in the right direction — guards swapped slots / same toggle.
    assert hc_floor - pc_floor > 0.15, f"arms did not separate on loss (gap {hc_floor - pc_floor:.3f})"


def test_projector_rankme_lower_for_collapse_arm(arms: dict[str, Any]) -> None:
    """Directional collapse at the projector surface: PC RankMe@proj < healthy RankMe@proj."""
    pc_rm = _final(arms["pc"], "rankme_proj")
    hc_rm = _final(arms["hc"], "rankme_proj")
    assert pc_rm < hc_rm, f"collapse arm's projector RankMe not below healthy ({pc_rm:.2f} vs {hc_rm:.2f})"


def test_gate_machinery_reports_consistent_separation(arms: dict[str, Any]) -> None:
    """The real gate function runs and reports the loss separation with the right sign."""
    gate = arms["gate"]
    assert gate["mode"] == "point_collapse"
    # loss_separation = healthy_floor - pc_floor; positive means the PC is nearer the -1 floor.
    assert gate["loss_separation"] > 0.0
