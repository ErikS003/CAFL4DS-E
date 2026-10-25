"""Unit tests for the P1.0.2 Phase-0 calibration trust map (``cafl4ds.health_trust``)."""

from __future__ import annotations

import re

import pytest

from cafl4ds.data.attributes import SyntheticAttributeSource
from cafl4ds.data.regime import RegimeStream
from cafl4ds.health_trust import (
    CANARY_SIGNALS,
    DEFAULT_LABEL_FREE_SIGNALS,
    LOAD_BEARING_KEYS,
    TRUST_MAP,
    Backbone,
    FailureMode,
    TrustStatus,
    annotate,
    backbone_family,
    trust_for,
)
from cafl4ds.models.vit import TinyViTEncoder
from cafl4ds.monitor import HealthMonitor
from cafl4ds.ssl.factory import build_mae, build_simsiam

_P0_ID = re.compile(r"^P0\.\d+(\.\d+)?$")


def _monitor_emitted_keys(family: Backbone) -> set[str]:
    """The signal keys the deploy-default monitor actually emits for a backbone family.

    Builds the tiny backbone + the monitor with the deploy-default flags (kNN / linear / alignment
    on; ``drift_surfaces`` on for the projector-bearing JE, off for MAE) and reads one checkpoint —
    so the load-bearing key list is checked against the monitor's *real* output, not a hand copy.
    """
    source = SyntheticAttributeSource(num_regimes=2, num_canary_classes=3, per_cell=16, img_size=16, long_tail=False)
    stream = RegimeStream(source, batch_size=8, support_per_canary=6, query_per_canary=6)
    encoder = TinyViTEncoder(img_size=16, patch_size=8, in_chans=3, embed_dim=32, depth=2, num_heads=2, mlp_ratio=2.0)
    if family is Backbone.JE:
        method = build_simsiam(encoder=encoder, proj_hidden=32, proj_dim=16, pred_hidden=16)
        drift_surfaces = True
    else:
        method = build_mae(encoder=encoder, decoder_dim=32, decoder_depth=1, decoder_heads=2, decoder_mlp_ratio=2.0)
        drift_surfaces = False
    monitor = HealthMonitor(
        eval_sets=stream.eval_sets,
        knn_k=3,
        run_knn=True,
        run_linear=True,
        run_alignment=True,
        align_seed=0,
        drift_surfaces=drift_surfaces,
    )
    return {k for k in monitor.measure(method, 0) if k != "step"}


@pytest.mark.parametrize("family", list(Backbone))
def test_load_bearing_keys_are_emitted_by_the_monitor(family: Backbone) -> None:
    """The curated load-bearing key list is checked against the monitor's emitted menu — B2.

    Guards the realistic regression: if a load-bearing signal were renamed or dropped from the
    monitor, its key would no longer appear here and this test would catch it (rather than the trust
    map silently defaulting it to ``uncalibrated``).
    """
    emitted = _monitor_emitted_keys(family)
    missing = set(LOAD_BEARING_KEYS[family]) - emitted
    assert not missing, f"{family.value}: load-bearing keys not emitted by the monitor: {sorted(missing)}"


def test_backbone_family_resolves_known_methods() -> None:
    """Each SSL method name maps to its calibration family (JE covers SimSiam / Barlow)."""
    assert backbone_family("mae") is Backbone.MAE
    assert backbone_family("simsiam") is Backbone.JE
    assert backbone_family("simsiam_collapse") is Backbone.JE
    assert backbone_family("barlow") is Backbone.JE
    assert backbone_family("barlow_collapse") is Backbone.JE


def test_backbone_family_rejects_unknown_method() -> None:
    """An unmapped method name raises rather than silently guessing a family."""
    with pytest.raises(ValueError, match="no backbone family"):
        backbone_family("resnet")


@pytest.mark.parametrize("family", list(Backbone))
def test_load_bearing_keys_are_calibrated_with_provenance(family: Backbone) -> None:
    """Every load-bearing signal carries a non-UNCALIBRATED verdict pinned to a Phase-0 substudy."""
    for key in LOAD_BEARING_KEYS[family]:
        entries = trust_for(family, key)
        assert entries, f"{family.value}/{key} has no verdict"
        assert all(e.status is not TrustStatus.UNCALIBRATED for e in entries), (
            f"{family.value}/{key} is UNCALIBRATED but load-bearing"
        )
        assert all(_P0_ID.match(e.provenance) for e in entries), f"{family.value}/{key} has a malformed provenance"


def test_unknown_signal_defaults_to_uncalibrated_never_raises() -> None:
    """A signal with no Phase-0 verdict is marked UNCALIBRATED, not treated as trusted or crashed."""
    entries = trust_for(Backbone.MAE, "some_new_instrument")
    assert len(entries) == 1
    assert entries[0].status is TrustStatus.UNCALIBRATED
    assert entries[0].provenance == ""


def test_uniformity_proj_is_a_trap_for_je_forgetting() -> None:
    """The load-bearing trap: a collapse-calibrated signal must be flagged where it mis-signals."""
    modes = {e.mode: e.status for e in trust_for(Backbone.JE, "uniformity_proj")}
    assert modes[FailureMode.COLLAPSE] is TrustStatus.CANDIDATE
    assert modes[FailureMode.FORGETTING] is TrustStatus.TRAP


def test_mae_quality_geometry_readers_are_traps() -> None:
    """On MAE, the P0.2 geometry suite mis-signals for quality — must not read as trusted."""
    for key in ("uniformity", "alignment", "mean_feature_var", "offdiag_cov"):
        statuses = {e.mode: e.status for e in trust_for(Backbone.MAE, key)}
        assert statuses.get(FailureMode.QUALITY) is TrustStatus.TRAP, key


def test_canary_probes_need_labels_both_families() -> None:
    """The labelled canary probes are flagged NEEDS_LABELS on both backbones."""
    for family in Backbone:
        for key in CANARY_SIGNALS:
            statuses = {e.status for e in trust_for(family, key)}
            assert TrustStatus.NEEDS_LABELS in statuses, f"{family.value}/{key}"


def test_default_label_free_signals_are_annotated_and_exclude_the_canary() -> None:
    """The default logged set is calibrated (annotatable) and keeps the canary a separate channel."""
    for family, signals in DEFAULT_LABEL_FREE_SIGNALS.items():
        assert signals, family
        for key in signals:
            assert key not in CANARY_SIGNALS, f"{family.value}: {key} is a canary, not label-free"
            assert trust_for(family, key), f"{family.value}/{key} unannotated"


def test_annotate_builds_serializable_block_defaulting_unknowns() -> None:
    """The stamped block is all plain strings, one verdict-list per key, unknowns UNCALIBRATED."""
    keys = ["rankme_proj", "uniformity_proj", "totally_unknown"]
    block = annotate(keys, Backbone.JE)
    assert set(block) == set(keys)
    # every verdict is a plain-string dict (JSON-serializable), with the four expected fields
    for verdicts in block.values():
        for v in verdicts:
            assert set(v) == {"mode", "status", "caveat", "provenance"}
            assert all(isinstance(x, str) for x in v.values())
    # the unknown key is present and flagged uncalibrated
    assert block["totally_unknown"][0]["status"] == TrustStatus.UNCALIBRATED.value


def test_registry_keys_are_well_formed() -> None:
    """Every registry entry is keyed by a real family and cites a Phase-0 id (or empty for none)."""
    for (family, _key), entries in TRUST_MAP.items():
        assert isinstance(family, Backbone)
        assert entries
        assert all(_P0_ID.match(e.provenance) for e in entries)
