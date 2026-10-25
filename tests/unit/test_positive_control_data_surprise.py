"""Unit tests for the DATA-SURPRISE / SHOCK gate reductions (audit P0.4 §E, E1 + B1).

The P0.4.1 harness reuses the P0.4.0-instrumented ``StreamingLoop`` (guarded by ``test_loop.py``), but
its *change-point reductions* — the paired splice, the cross-arm contrasts, and the two-sided gate —
had no regression guard (audit ``docs/experiments/audits/P0.4.md`` §E, E1). These drive the pure
reductions on hand-built synthetic traces (no harness run) and assert:

* ``_shock_sequence`` indexes the base steps **identically across arms** regardless of ``inject`` — only
  the burst-window *content* differs (the paired design; a broken splice would desync the counterfactual);
* ``_summarize`` reads the three windows (pre-burst base, burst, tail) and the **within-arm** ratio;
* ``_cross_ratios`` are **cross-arm** (shock vs. control at the same steps);
* ``_evaluate_surprise_gate`` fires on the **cross-arm** ``fire_ratio`` and reads flatness off the
  **within-arm** ratio — the two must not be conflated (the B1 name-collision the audit flagged): a
  training-progress spike that lifts *both* arms has a high within-arm ratio yet a ~1x cross-arm ratio,
  and must **not** fire.
"""

from __future__ import annotations

import importlib.util
from pathlib import Path
from types import ModuleType
from typing import Any

import pytest
import torch
from hydra import compose, initialize_config_dir
from hydra.core.global_hydra import GlobalHydra
from omegaconf import DictConfig

from cafl4ds.data.streams import StreamBatch

_REPO_ROOT = Path(__file__).resolve().parents[2]
_CONFIG_DIR = _REPO_ROOT / "cafl4ds" / "configs"


def _load_harness() -> ModuleType:
    """Import ``scripts/positive_control_data_surprise.py`` as a module (it is a script, not a package)."""
    path = _REPO_ROOT / "scripts" / "positive_control_data_surprise.py"
    spec = importlib.util.spec_from_file_location("positive_control_data_surprise_harness", path)
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


@pytest.fixture(scope="module")
def h() -> ModuleType:
    """The imported data-surprise-harness module (shared across the test module)."""
    return _load_harness()


@pytest.fixture(scope="module")
def gate_config() -> DictConfig:
    """A composed config with a short window shape, for the ``_summarize`` / gate reductions."""
    GlobalHydra.instance().clear()  # isolate from any other hydra-using test
    with initialize_config_dir(version_base=None, config_dir=str(_CONFIG_DIR)):
        return compose(
            config_name="positive_control_data_surprise",
            overrides=["warm_steps=4", "burst_steps=2", "tail_steps=3", "init_skip=1"],
        )


def _trace(gnorms: list[float], losses: list[float]) -> list[dict[str, Any]]:
    """Build a per-step trace (all finite) for the window reductions."""
    return [
        {"series": "loss", "step": i, "loss": loss, "grad_norm": gnorm, "finite": True}
        for i, (gnorm, loss) in enumerate(zip(gnorms, losses, strict=True))
    ]


def _arm(*, base_peak: float, burst_peak: float, burst_loss: float, base_loss: float = 0.06) -> dict[str, Any]:
    """A minimal arm summary with the fields the gate + cross-ratios read."""
    return {
        "base_peak_grad_norm": base_peak,
        "burst_peak_grad_norm": burst_peak,
        "base_mean_loss": base_loss,
        "burst_mean_loss": burst_loss,
        "base_init_loss": 0.2,
        "base_final_loss": base_loss,  # descends 0.2 -> base_loss
        "within_arm_ratio": (burst_peak / base_peak) if base_peak else float("inf"),
        "recover_ratio": 0.5,
        "tail_min_grad_norm": base_peak,
        "first_nonfinite_step": None,
    }


def test_shock_sequence_identical_base_across_arms(h: ModuleType) -> None:
    """Only the burst-window content differs between arms; every base step is bit-identical."""
    base = [torch.full((1, 3, 4, 4), float(i)) for i in range(5)]
    ood = [torch.full((1, 3, 4, 4), 99.0) for _ in range(2)]
    warm, burst, tail = 4, 2, 3
    shock = h._shock_sequence(base, ood, warm=warm, burst=burst, tail=tail, inject=True)
    control = h._shock_sequence(base, ood, warm=warm, burst=burst, tail=tail, inject=False)

    assert len(shock) == len(control) == warm + burst + tail
    for step, (s, c) in enumerate(zip(shock, control, strict=True)):
        assert isinstance(s, StreamBatch) and s.step == c.step == step
        if warm <= step < warm + burst:  # burst window: shock=OOD, control=base
            assert torch.equal(s.images, ood[(step - warm) % len(ood)])
            assert torch.equal(c.images, base[step % len(base)])
            assert s.era == 1  # OOD burst era on the injected arm only
        else:  # warm / tail: identical base batch on both arms
            assert torch.equal(s.images, c.images)
            assert torch.equal(s.images, base[step % len(base)])
            assert s.era == c.era == (2 if step >= warm + burst else 0)


def test_summarize_windows_and_within_arm_ratio(h: ModuleType, gate_config: DictConfig) -> None:
    """The within-arm ratio is burst peak / pre-burst base peak, over the configured windows."""
    # steps: 0 init (skipped), [1,4) base band, [4,6) burst, [6,9) tail.
    gnorms = [9.0, 0.1, 0.2, 0.1, 8.0, 6.0, 0.15, 0.1, 0.12]
    losses = [0.20, 0.08, 0.07, 0.06, 2.0, 1.5, 0.06, 0.06, 0.06]
    summ = h._summarize(gate_config, _trace(gnorms, losses))
    assert summ["base_peak_grad_norm"] == pytest.approx(0.2)  # step-0 9.0 excluded by init_skip=1
    assert summ["burst_peak_grad_norm"] == pytest.approx(8.0)
    assert summ["within_arm_ratio"] == pytest.approx(8.0 / 0.2)
    assert summ["base_init_loss"] == pytest.approx(0.08) and summ["base_final_loss"] == pytest.approx(0.06)
    assert summ["first_nonfinite_step"] is None


def test_cross_ratios_are_cross_arm(h: ModuleType) -> None:
    """``fire_ratio`` / ``surprise_ratio`` compare the shock to the control at the SAME steps."""
    shock = _arm(base_peak=0.1, burst_peak=8.0, burst_loss=2.0)
    control = _arm(base_peak=0.1, burst_peak=0.2, burst_loss=0.06)
    x = h._cross_ratios(shock, control)
    assert x["fire_ratio"] == pytest.approx(8.0 / 0.2)  # cross-arm grad peaks
    assert x["surprise_ratio"] == pytest.approx(2.0 / 0.06)  # cross-arm burst losses


def test_surprise_gate_fires_two_sided(h: ModuleType, gate_config: DictConfig) -> None:
    """A saturating shock: cross-arm fire + real surprise + a flat control -> two-sided PASS."""
    shock = _arm(base_peak=0.1, burst_peak=8.0, burst_loss=2.0)
    control = _arm(base_peak=0.1, burst_peak=0.12, burst_loss=0.06)
    gate = h._evaluate_surprise_gate(gate_config, shock, control)
    assert gate["checks"] == {
        "base_competent": True,
        "surprise_real": True,
        "shock_fires": True,
        "control_flat": True,
    }
    assert gate["passed"] is True
    assert gate["fire_ratio"] == pytest.approx(8.0 / 0.12)


def test_gate_keys_on_cross_arm_not_within_arm(h: ModuleType, gate_config: DictConfig) -> None:
    """B1 guard: a training-progress spike that lifts BOTH arms is high within-arm but ~1x cross-arm.

    The within-arm ratio (each arm's own burst / base band) is large — training progress makes any
    burst-window batch look harder than the earlier base band — yet nothing is *shock-specific*, so the
    cross-arm ``fire_ratio`` is ~1x and the gate must **not** fire. This is exactly why the gate keys on
    the cross-arm quantity; conflating the two (the audit's B1 name collision) would false-fire here.
    """
    shock = _arm(base_peak=0.1, burst_peak=8.0, burst_loss=2.0)  # within-arm ratio 80x
    control = _arm(base_peak=0.1, burst_peak=8.0, burst_loss=2.0)  # ALSO spiked (same progress) -> within 80x
    x = h._cross_ratios(shock, control)
    assert x["fire_ratio"] == pytest.approx(1.0)  # cross-arm: nothing shock-specific
    assert shock["within_arm_ratio"] == pytest.approx(80.0)  # within-arm is large regardless
    gate = h._evaluate_surprise_gate(gate_config, shock, control)
    assert gate["checks"]["shock_fires"] is False  # keys on cross-arm fire_ratio, not within-arm
    assert gate["passed"] is False
