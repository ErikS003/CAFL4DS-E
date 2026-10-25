"""Unit tests for the SHORTCUT / stable-loss representation-degradation harness (audit P0.5 §E, C4/E1).

The candidate readers themselves are already guarded by ``test_measurements.py`` (clusterability
separates structure from noise, ``mean_attention_distance`` known-answer, ``alignment`` identity/rise)
and ``test_monitor.py``. What had **no** regression guard — and the P0.5 code-validity audit
(``docs/experiments/audits/P0.5.md`` §E, C4/E1) flagged it — is the load-bearing *harness* arithmetic
that turns two arms' health records into the mode verdict: the sign-normalized ``_reader_verdict``, the
four-condition ``_evaluate_shortcut_gate`` (RankMe-frac precondition, kNN-primary crater orientation,
loss-does-not-alarm), and the P0.3.9 warm-up-recovery diagnostic. A future mis-wire (a flipped reader
sign, a swapped arm, a wrong probe orientation, a reader accidentally gating ``passed``) would still
pass CI. These drive the pure reductions on hand-built health records — **no** harness run — plus one
fast, network-free paired-init check that the seed reset leaves both arms' encoders bit-identical
(so the lever is genuinely the only variable). Each load-bearing property is asserted directly:

* ``_reader_verdict`` sign-normalizes ``sign * (shortcut - healthy)`` with the correct +1/-1 map per
  reader (uniformity/alignment/alignment_strong climb when worse; clusterability/attn_distance drop),
  fires per its own scale-specific threshold, skips unmeasured readers, and never gates ``passed``;
* ``_evaluate_shortcut_gate`` orients the crater as ``healthy - shortcut`` on kNN, ``rankme_frac`` as
  ``shortcut / healthy`` (a large drop is collapse, not this mode), and ``loss_does_not_alarm`` the
  right way; ``passed`` reads only the four induction conditions;
* the warm-up-recovery boolean is inert for from-scratch (``well_baseline=None``) and flags a
  non-recovering healthy arm for the pretrained-continue vehicle.
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
from hydra.utils import instantiate
from omegaconf import DictConfig, OmegaConf

_REPO_ROOT = Path(__file__).resolve().parents[2]
_CONFIG_DIR = _REPO_ROOT / "cafl4ds" / "configs"


def _load_harness() -> ModuleType:
    """Import ``scripts/positive_control_shortcut.py`` as a module (it is a script, not a package)."""
    path = _REPO_ROOT / "scripts" / "positive_control_shortcut.py"
    spec = importlib.util.spec_from_file_location("positive_control_shortcut_harness", path)
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


@pytest.fixture(scope="module")
def h() -> ModuleType:
    """The imported shortcut-harness module (shared across the test module)."""
    return _load_harness()


@pytest.fixture(scope="module")
def gate_config() -> DictConfig:
    """The composed ``positive_control_shortcut`` config — for its ``gate`` thresholds."""
    GlobalHydra.instance().clear()  # isolate from any other hydra-using test
    with initialize_config_dir(version_base=None, config_dir=str(_CONFIG_DIR)):
        return compose(config_name="positive_control_shortcut")


def _reader_gaps(gate_config: DictConfig) -> tuple[float, dict[str, float]]:
    """The default + per-reader fire thresholds as plain python (what the harness reads)."""
    g = gate_config.gate
    per_reader = OmegaConf.to_container(g.reader_min_gaps, resolve=True)
    assert isinstance(per_reader, dict)
    return float(g.reader_min_gap), {str(k): float(v) for k, v in per_reader.items()}


def _hrec(**kw: float) -> list[dict[str, Any]]:
    """A one-element health series (the reductions read ``[-1]``); defaults give a healthy full-rank arm."""
    rec: dict[str, Any] = {"rankme": 10.0, "knn_acc": 0.40, "linear_acc": 0.55}
    rec.update(kw)
    return [rec]


# --------------------------------------------------------------------------- #
# _reader_verdict — the sign-normalized DELIVERABLE (reported, never gated).
# --------------------------------------------------------------------------- #


def test_reader_sign_map_fires_when_shortcut_worse_each_direction(h: ModuleType, gate_config: DictConfig) -> None:
    """Every reader fires when the shortcut arm is worse in that reader's OWN worsening direction.

    +1 readers (uniformity/alignment/alignment_strong) climb when worse; -1 readers
    (clusterability/attn_distance) drop when worse. A worse-in-direction shortcut must yield a
    POSITIVE ``degrade_gap`` for all five, past each reader's scale-specific threshold.
    """
    default_gap, per_reader = _reader_gaps(gate_config)
    healthy = _hrec(uniformity=-3.0, alignment=0.10, alignment_strong=0.10, clusterability=0.30, attn_distance=3.0)
    shortcut = _hrec(  # +1: higher; -1: lower — each past its threshold
        uniformity=-2.7,  # +0.30 >= 0.20
        alignment=0.35,  # +0.25 >= 0.20
        alignment_strong=0.20,  # +0.10 >= 0.05
        clusterability=0.20,  # drop 0.10 -> gap +0.10 >= 0.05
        attn_distance=2.4,  # drop 0.60 -> gap +0.60 >= 0.50
    )
    verdict = h._reader_verdict(shortcut, healthy, default_gap, per_reader)
    gaps = {r["reader"]: r["degrade_gap"] for r in verdict["candidates"]}
    assert gaps["uniformity"] == pytest.approx(0.30)
    assert gaps["alignment_strong"] == pytest.approx(0.10)
    assert gaps["clusterability"] == pytest.approx(0.10)  # -1 map: lower shortcut => positive gap
    assert gaps["attn_distance"] == pytest.approx(0.60)
    assert set(verdict["fired"]) == {"uniformity", "alignment", "alignment_strong", "clusterability", "attn_distance"}
    assert verdict["any_reader_fired"] is True


def test_reader_sign_map_quiet_when_shortcut_better(h: ModuleType, gate_config: DictConfig) -> None:
    """A shortcut BETTER in each reader's direction gives negative gaps and fires nothing.

    Guards the +1/-1 map from a flip: for the -1 readers a HIGHER shortcut must read as *better*
    (negative gap), not worse.
    """
    default_gap, per_reader = _reader_gaps(gate_config)
    healthy = _hrec(uniformity=-3.0, alignment=0.30, alignment_strong=0.30, clusterability=0.20, attn_distance=2.0)
    shortcut = _hrec(uniformity=-3.5, alignment=0.10, alignment_strong=0.10, clusterability=0.40, attn_distance=3.0)
    verdict = h._reader_verdict(shortcut, healthy, default_gap, per_reader)
    assert verdict["fired"] == []
    assert verdict["any_reader_fired"] is False
    gaps = {r["reader"]: r["degrade_gap"] for r in verdict["candidates"]}
    assert gaps["clusterability"] == pytest.approx(-0.20)  # higher shortcut => negative (better)
    assert all(g < 0 for g in gaps.values())


def test_reader_per_reader_thresholds(h: ModuleType, gate_config: DictConfig) -> None:
    """The same +0.10 gap fires ``alignment_strong`` (0.05 bar) but not ``uniformity`` (0.20 bar).

    Pins the A4 wiring: readers live on different scales, so the per-reader thresholds are load-bearing
    — a single default bar would mis-classify one of these two.
    """
    default_gap, per_reader = _reader_gaps(gate_config)
    healthy = _hrec(uniformity=-3.0, alignment_strong=0.10)
    shortcut = _hrec(uniformity=-2.9, alignment_strong=0.20)  # both +0.10
    verdict = h._reader_verdict(shortcut, healthy, default_gap, per_reader)
    fires = {r["reader"]: r["fires"] for r in verdict["candidates"]}
    assert fires["alignment_strong"] is True  # 0.10 >= 0.05
    assert fires["uniformity"] is False  # 0.10 <  0.20


def test_reader_skips_unmeasured(h: ModuleType, gate_config: DictConfig) -> None:
    """A reader absent from the records is excluded from the panel (no KeyError, not a false quiet)."""
    default_gap, per_reader = _reader_gaps(gate_config)
    # Only alignment_strong measured (e.g. a method exposing one positive pair).
    healthy = _hrec(alignment_strong=0.10)
    shortcut = _hrec(alignment_strong=0.20)
    verdict = h._reader_verdict(shortcut, healthy, default_gap, per_reader)
    assert [r["reader"] for r in verdict["candidates"]] == ["alignment_strong"]
    assert verdict["fired"] == ["alignment_strong"]


# --------------------------------------------------------------------------- #
# _evaluate_shortcut_gate — the four-condition mode-induction gate.
# --------------------------------------------------------------------------- #


def test_gate_orientation_and_pass(h: ModuleType, gate_config: DictConfig) -> None:
    """Crater = healthy - shortcut on kNN; rankme_frac = shortcut / healthy; a clean induction PASSES."""
    healthy = _hrec(rankme=10.0, knn_acc=0.40, linear_acc=0.55)
    shortcut = _hrec(rankme=10.0, knn_acc=0.30, linear_acc=0.45)  # full-rank, cratered probe
    gate = h._evaluate_shortcut_gate(gate_config, shortcut, healthy, shortcut_loss_floor=0.30, healthy_loss_floor=0.35)
    assert gate["mode"] == "shortcut"
    assert gate["probe_gap"] == pytest.approx(0.10)  # healthy - shortcut
    assert gate["probe_ratio"] == pytest.approx(0.40 / 0.30)
    assert gate["rankme_frac"] == pytest.approx(1.0)  # shortcut / healthy
    assert gate["checks"] == {
        "healthy_has_headroom": True,
        "rankme_preserved": True,
        "probe_craters": True,
        "loss_does_not_alarm": True,
        "contrast": True,
    }
    assert gate["passed"] is True


def test_gate_rankme_precondition_blocks_collapse(h: ModuleType, gate_config: DictConfig) -> None:
    """A big RankMe drop is P0.2 collapse, not this mode: ``rankme_preserved`` fails -> not passed.

    Even with a deep probe crater, a shortcut rank at half the healthy arm's (frac 0.5 < 0.85) must
    NOT certify the full-rank degradation mode.
    """
    healthy = _hrec(rankme=10.0, knn_acc=0.40)
    shortcut = _hrec(rankme=5.0, knn_acc=0.28)  # deep crater but rank collapsed
    gate = h._evaluate_shortcut_gate(gate_config, shortcut, healthy, shortcut_loss_floor=0.30, healthy_loss_floor=0.35)
    assert gate["rankme_frac"] == pytest.approx(0.5)
    assert gate["checks"]["rankme_preserved"] is False
    assert gate["checks"]["probe_craters"] is True  # the crater itself is real
    assert gate["passed"] is False


def test_gate_loss_alarm_orientation(h: ModuleType, gate_config: DictConfig) -> None:
    """``loss_does_not_alarm`` fails only when the shortcut loss floor RISES past healthy + margin."""
    healthy = _hrec(rankme=10.0, knn_acc=0.40)
    shortcut = _hrec(rankme=10.0, knn_acc=0.30)
    # Shortcut loss well ABOVE healthy -> the loss does NOT lie -> the anti-instrument condition fails.
    alarmed = h._evaluate_shortcut_gate(
        gate_config, shortcut, healthy, shortcut_loss_floor=0.50, healthy_loss_floor=0.35
    )
    assert alarmed["checks"]["loss_does_not_alarm"] is False
    assert alarmed["passed"] is False
    # Shortcut loss at/below healthy (the loss lies) -> condition holds.
    quiet = h._evaluate_shortcut_gate(gate_config, shortcut, healthy, shortcut_loss_floor=0.34, healthy_loss_floor=0.35)
    assert quiet["checks"]["loss_does_not_alarm"] is True


def test_gate_contrast_below_threshold_fails(h: ModuleType, gate_config: DictConfig) -> None:
    """A crater too shallow to clear ``min_probe_ratio`` (1.15x) fails ``contrast`` even if the gap clears."""
    healthy = _hrec(rankme=10.0, knn_acc=0.40)
    shortcut = _hrec(rankme=10.0, knn_acc=0.36)  # gap 0.04 < 0.05 and ratio 1.11 < 1.15
    gate = h._evaluate_shortcut_gate(gate_config, shortcut, healthy, shortcut_loss_floor=0.30, healthy_loss_floor=0.35)
    assert gate["checks"]["probe_craters"] is False
    assert gate["checks"]["contrast"] is False
    assert gate["passed"] is False


def test_reader_verdict_never_gates_passed(h: ModuleType, gate_config: DictConfig) -> None:
    """The reader verdict is REPORTED, not gated: a fire cannot pass a failed gate, nor a null fail a clean one."""
    # (a) Clean induction, NO reader fires -> still passed (a null is a legitimate result).
    healthy = _hrec(rankme=10.0, knn_acc=0.40, alignment_strong=0.10)
    shortcut_quiet = _hrec(rankme=10.0, knn_acc=0.30, alignment_strong=0.10)  # reader gap 0
    clean = h._evaluate_shortcut_gate(
        gate_config, shortcut_quiet, healthy, shortcut_loss_floor=0.30, healthy_loss_floor=0.35
    )
    assert clean["reader_verdict"]["any_reader_fired"] is False
    assert clean["passed"] is True
    # (b) Failed induction (no crater), a reader fires -> still NOT passed.
    shortcut_fires = _hrec(rankme=10.0, knn_acc=0.40, alignment_strong=0.30)  # no crater, reader gap +0.20
    failed = h._evaluate_shortcut_gate(
        gate_config, shortcut_fires, healthy, shortcut_loss_floor=0.30, healthy_loss_floor=0.35
    )
    assert failed["checks"]["probe_craters"] is False
    assert failed["reader_verdict"]["any_reader_fired"] is True
    assert failed["passed"] is False


# --------------------------------------------------------------------------- #
# Warm-up-recovery diagnostic (P0.3.9) — reported, inert for from-scratch.
# --------------------------------------------------------------------------- #


def test_warmup_guard_inert_for_from_scratch(h: ModuleType, gate_config: DictConfig) -> None:
    """``well_baseline=None`` (from-scratch) -> no baseline reported, recovery trivially True."""
    healthy = _hrec(rankme=10.0, knn_acc=0.40)
    shortcut = _hrec(rankme=10.0, knn_acc=0.30)
    gate = h._evaluate_shortcut_gate(
        gate_config, shortcut, healthy, shortcut_loss_floor=0.30, healthy_loss_floor=0.35, well_baseline=None
    )
    assert gate["well_baseline_probe"] is None
    assert gate["healthy_drop_from_well"] is None
    assert gate["healthy_recovers_baseline"] is True


def test_warmup_guard_flags_non_recovering_healthy(h: ModuleType, gate_config: DictConfig) -> None:
    """Pretrained-continue: a healthy arm that does not climb back to the pristine well is flagged."""
    healthy = _hrec(rankme=10.0, knn_acc=0.40)
    shortcut = _hrec(rankme=10.0, knn_acc=0.30)
    well = {"knn_acc": 0.42, "linear_acc": 0.55, "rankme": 10.0}
    # healthy 0.40 vs well 0.42 -> drop 0.02 <= recovery_margin 0.05 -> recovers.
    recovered = h._evaluate_shortcut_gate(
        gate_config, shortcut, healthy, shortcut_loss_floor=0.30, healthy_loss_floor=0.35, well_baseline=well
    )
    assert recovered["well_baseline_probe"] == pytest.approx(0.42)
    assert recovered["healthy_drop_from_well"] == pytest.approx(0.02)
    assert recovered["healthy_recovers_baseline"] is True
    # A far-below-well healthy arm -> NOT recovered (its headroom may be warm-up damage).
    damaged_healthy = _hrec(rankme=10.0, knn_acc=0.30)
    damaged = h._evaluate_shortcut_gate(
        gate_config, shortcut, damaged_healthy, shortcut_loss_floor=0.30, healthy_loss_floor=0.35, well_baseline=well
    )
    assert damaged["healthy_recovers_baseline"] is False


# --------------------------------------------------------------------------- #
# Paired-init invariant — the seed reset makes the lever the ONLY variable.
# --------------------------------------------------------------------------- #


def test_seed_reset_gives_bit_identical_encoders_across_arms(h: ModuleType, gate_config: DictConfig) -> None:
    """Both arms build a bit-identical encoder (the P0.5.0 mask lever touches only mask/decoder).

    Mirrors ``_run_arm``'s init (``torch.manual_seed(seed)`` then instantiate) for the healthy and
    shortcut overrides, and asserts the encoders match parameter-for-parameter — so any downstream
    divergence is attributable to the lever alone, not the init.
    """
    overrides_h = h._arm_overrides(gate_config, "healthy")
    overrides_s = h._arm_overrides(gate_config, "shortcut")
    assert overrides_h["mask_ratio"] == pytest.approx(0.75)
    assert overrides_s["mask_ratio"] == pytest.approx(0.1)  # the lever genuinely differs

    torch.manual_seed(gate_config.seed)
    method_h = instantiate(gate_config.ssl, encoder=instantiate(gate_config.encoder), **overrides_h)
    torch.manual_seed(gate_config.seed)
    method_s = instantiate(gate_config.ssl, encoder=instantiate(gate_config.encoder), **overrides_s)

    sd_h, sd_s = method_h.encoder.state_dict(), method_s.encoder.state_dict()
    assert sd_h.keys() == sd_s.keys()
    assert all(torch.equal(sd_h[k], sd_s[k]) for k in sd_h), "seed reset did not yield identical encoder inits"
