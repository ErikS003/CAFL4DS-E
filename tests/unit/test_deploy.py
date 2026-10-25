"""Unit + integration tests for the P1.0.2 deployment harness (``cafl4ds.deploy``)."""

from __future__ import annotations

from pathlib import Path
from typing import Any

import torch

from cafl4ds.data.attributes import SyntheticAttributeSource
from cafl4ds.data.regime import RegimeStream
from cafl4ds.deploy import (
    build_deploy_report,
    channel_completeness,
    emitted_signals,
    project_health,
    split_channels,
    tier_b_report,
)
from cafl4ds.filters.accept_all import AcceptAll
from cafl4ds.harness import Arm, run_stream_arm
from cafl4ds.health_trust import CANARY_SIGNALS, DEFAULT_LABEL_FREE_SIGNALS, Backbone
from cafl4ds.models.vit import TinyViTEncoder
from cafl4ds.monitor import HealthMonitor
from cafl4ds.ssl.factory import build_simsiam


def _health(step: int, era: int, **signals: float) -> dict[str, Any]:
    """A synthetic health record (a monitor checkpoint)."""
    return {"run": "x", "series": "health", "step": float(step), "era": era, "loss": None, **signals}


def _je_signals() -> dict[str, float]:
    """A minimal set of JE default label-free + canary signals for a checkpoint."""
    return {
        "rankme_proj": 8.0,
        "mean_feature_var_proj": 0.1,
        "offdiag_cov_proj": 0.01,
        "alignment_proj": 0.5,
        "cosine_drift_proj": 0.2,
        "rankme": 7.0,
        "knn_acc": 0.9,
        "linear_acc": 0.85,
    }


def _je_arm(role: str = "live", n: int = 3) -> Arm:
    """A fabricated JE arm carrying the full default signal set at every checkpoint."""
    return Arm(name=f"{role}_arm", role=role, records=[_health(s * 3, s % 2, **_je_signals()) for s in range(n)])


def _expected() -> list[str]:
    """The JE Tier-A expected signals (default label-free + canary)."""
    return [*DEFAULT_LABEL_FREE_SIGNALS[Backbone.JE], *CANARY_SIGNALS]


def test_emitted_signals_unions_and_drops_bookkeeping() -> None:
    """The signal set is the union of instrument keys, without step/era/loss/run/series."""
    health = [_health(0, 0, rankme=1.0), _health(3, 1, rankme=2.0, knn_acc=0.5)]
    assert emitted_signals(health) == ["knn_acc", "rankme"]


def test_split_channels_separates_the_labelled_canary() -> None:
    """KNN / linear probes land in the canary channel; the rest is label-free."""
    channels = split_channels(["rankme_proj", "knn_acc", "cosine_drift_proj", "linear_acc"])
    assert channels["canary"] == ["knn_acc", "linear_acc"]
    assert channels["label_free"] == ["cosine_drift_proj", "rankme_proj"]


def test_project_health_passthrough_and_restrict() -> None:
    """keep=None logs everything; a list restricts to those signals but keeps bookkeeping."""
    health = [_health(0, 0, rankme=1.0, uniformity=2.0, knn_acc=0.5)]
    assert project_health(health, None) == health
    restricted = project_health(health, ["rankme"])
    assert set(restricted[0]) == {"run", "series", "step", "era", "loss", "rankme"}


def test_channel_completeness_flags_a_missing_signal() -> None:
    """A checkpoint missing an expected signal fails completeness and is localized by step."""
    health = [_health(0, 0, rankme=1.0, knn_acc=0.5), _health(3, 1, rankme=2.0)]
    report = channel_completeness(health, ["rankme", "knn_acc"])
    assert report["complete"] is False
    assert report["missing"] == {"3": ["knn_acc"]}


def test_build_report_passes_when_wired_and_stamps_trust() -> None:
    """A complete JE arm passes Tier-A, populates both channels, and carries the trust block."""
    report = build_deploy_report(
        config_header={"backbone": "je", "seed": 0},
        family=Backbone.JE,
        live=_je_arm(),
        expected_signals=_expected(),
    )
    val = report["validation"]
    assert val["passed"] and val["finite"] and val["channels_present"]
    assert val["channel_complete"] and val["trust_complete"]
    assert report["channels"]["canary"] == ["knn_acc", "linear_acc"]
    # the trust block annotates every emitted signal, and rankme_proj is a calibrated collapse reader
    assert "rankme_proj" in report["trust"]
    assert any(v["status"] == "calibrated" for v in report["trust"]["rankme_proj"])
    assert report["backbone_family"] == "je"


def test_build_report_fails_when_a_signal_is_missing() -> None:
    """Dropping an expected signal fails channel-completeness and the overall verdict."""
    arm = Arm(
        name="live_arm",
        role="live",
        records=[_health(0, 0, **{k: 1.0 for k in _je_signals() if k != "cosine_drift_proj"})],
    )
    report = build_deploy_report(config_header={}, family=Backbone.JE, live=arm, expected_signals=_expected())
    assert report["validation"]["channel_complete"] is False
    assert report["validation"]["passed"] is False


def test_build_report_fails_without_a_canary_channel() -> None:
    """No labelled probe means the canary channel is empty — Tier-A must not pass."""
    arm = Arm(
        name="live_arm",
        role="live",
        records=[_health(0, 0, rankme_proj=8.0, cosine_drift_proj=0.1)],
    )
    report = build_deploy_report(
        config_header={}, family=Backbone.JE, live=arm, expected_signals=["rankme_proj", "cosine_drift_proj"]
    )
    assert report["validation"]["channels_present"] is False
    assert report["validation"]["passed"] is False


def test_build_report_folds_in_the_leak_verdict() -> None:
    """A non-flat memory trace fails the verdict even when everything else is wired."""
    report = build_deploy_report(
        config_header={},
        family=Backbone.JE,
        live=_je_arm(),
        expected_signals=_expected(),
        leak={"flat": False, "tail_spread_frac": 0.4},
    )
    assert report["validation"]["memory_flat"] is False
    assert report["validation"]["passed"] is False
    assert report["leak_check"]["flat"] is False


def test_build_report_co_logs_optional_gate_arms() -> None:
    """PC/B5 arms, when present, are co-logged under their roles in the corpus."""
    report = build_deploy_report(
        config_header={},
        family=Backbone.JE,
        live=_je_arm("live"),
        expected_signals=_expected(),
        pc=_je_arm("pc"),
        b5=_je_arm("b5"),
    )
    assert set(report["arms"]) == {"live", "pc", "b5"}


def test_build_report_respects_log_signals_restriction() -> None:
    """log_signals prunes the logged corpus to the chosen signals (canary can be dropped)."""
    report = build_deploy_report(
        config_header={},
        family=Backbone.JE,
        live=_je_arm(),
        expected_signals=["rankme_proj"],
        log_signals=["rankme_proj"],
    )
    assert report["channels"]["label_free"] == ["rankme_proj"]
    assert report["channels"]["canary"] == []
    # the pruned health records carry only the kept signal (+ bookkeeping)
    kept = set(report["arms"]["live"]["health"][0])
    assert "knn_acc" not in kept and "rankme_proj" in kept


def test_tier_b_passes_on_a_plausible_series() -> None:
    """A plausible trajectory clears the loose sanity floor: RankMe in range, drift up, canary > chance."""
    tb = tier_b_report(_je_arm().health, canary_chance=1 / 3)
    assert tb["passed"]
    assert tb["checks"] == {"rankme_in_range": True, "drift_accumulated": True, "canary_above_chance": True}
    # the projector current-stream drift is preferred over the backbone drift when both are logged
    assert tb["reported"]["drift_key"] == "cosine_drift_proj"


def test_tier_b_flags_collapsed_rankme_flat_drift_and_chance_canary() -> None:
    """Each criterion fails independently on an implausible series (collapse / no drift / chance canary)."""
    bad = {**_je_signals(), "rankme": 0.0, "cosine_drift_proj": 0.0, "knn_acc": 1 / 3}
    health = [_health(s, s % 2, **bad) for s in range(3)]
    tb = tier_b_report(health, canary_chance=1 / 3)
    assert tb["passed"] is False
    assert tb["checks"] == {"rankme_in_range": False, "drift_accumulated": False, "canary_above_chance": False}


def test_build_report_attaches_tier_b_separately_from_tier_a() -> None:
    """Tier B rides in the report only when chance is given, and never touches the Tier-A wiring verdict."""
    plausible = build_deploy_report(
        config_header={}, family=Backbone.JE, live=_je_arm(), expected_signals=_expected(), canary_chance=1 / 3
    )
    assert plausible["tier_b"]["passed"] is True
    # a wired-but-implausible series still passes Tier A (wiring) while failing Tier B (sanity)
    bad = {**_je_signals(), "cosine_drift_proj": 0.0, "knn_acc": 1 / 3}
    arm = Arm(name="live", role="live", records=[_health(s, s % 2, **bad) for s in range(3)])
    report = build_deploy_report(
        config_header={}, family=Backbone.JE, live=arm, expected_signals=_expected(), canary_chance=1 / 3
    )
    assert report["validation"]["passed"] is True
    assert report["tier_b"]["passed"] is False
    # with no chance supplied, Tier B is simply not computed
    no_chance = build_deploy_report(config_header={}, family=Backbone.JE, live=_je_arm(), expected_signals=_expected())
    assert "tier_b" not in no_chance


def test_tier_b_thresholds_are_overridable() -> None:
    """A stricter canary margin can fail a series the default (loose) floor would pass."""
    health = _je_arm().health  # knn_acc = 0.9, chance 1/3
    assert tier_b_report(health, canary_chance=1 / 3)["passed"]
    strict = tier_b_report(health, canary_chance=1 / 3, thresholds={"canary_margin": 0.6})
    assert strict["checks"]["canary_above_chance"] is False


def test_end_to_end_regime_stream_feeds_the_monitor_and_validates(tmp_path: Path) -> None:
    """The whole data→backbone→monitor→corpus path runs on synthetic data and passes Tier-A."""
    source = SyntheticAttributeSource(num_regimes=3, num_canary_classes=3, per_cell=16, img_size=16, long_tail=False)
    stream = RegimeStream(source, batch_size=8, support_per_canary=6, query_per_canary=6, seed=0)
    encoder = TinyViTEncoder(img_size=16, patch_size=8, in_chans=3, embed_dim=32, depth=2, num_heads=2, mlp_ratio=2.0)
    method = build_simsiam(encoder=encoder, proj_hidden=32, proj_dim=16, pred_hidden=16)
    live = run_stream_arm(
        name="it_live",
        role="live",
        method=method,
        stream=stream,
        optimizer=torch.optim.AdamW(method.parameters(), lr=1e-3),
        selection_filter=AcceptAll(),
        monitor=HealthMonitor(eval_sets=stream.eval_sets, drift_surfaces=True),
        out_dir=tmp_path,
        eval_every=2,
    )
    report = build_deploy_report(
        config_header={"backbone": "je"},
        family=Backbone.JE,
        live=live,
        expected_signals=[*DEFAULT_LABEL_FREE_SIGNALS[Backbone.JE], *CANARY_SIGNALS],
    )
    assert report["validation"]["passed"], report["completeness"]["missing"]
    assert report["channels"]["canary"] == ["knn_acc", "linear_acc"]
    assert "rankme_proj" in report["channels"]["label_free"]
