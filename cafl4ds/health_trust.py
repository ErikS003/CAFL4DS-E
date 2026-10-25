"""Phase-0 calibration trust map for the deployment-prototype harness (P1.0.2).

Distils the Phase-0 Failure-Modes scoreboard (``docs/experiments/phase0/index.md``) into a typed,
provenance-pinned registry: for each ``(backbone family, health signal)``, what Phase 0 established
about *reading* it — ``calibrated`` / ``candidate`` / ``trap`` / ``needs-two-sided-control`` /
``needs-labels`` — and for which failure mode. The deployment harness stamps this into every logged
artifact, so the multivariate health traces are **self-describing**: a consumer never fuses a signal
that is a known *trap* in the mode it cares about (e.g. ``uniformity_proj``, calibrated to collapse,
*mis-signals* on forgetting — P0.6.0).

This is **fixed reference truth**, not a runtime knob — hence code, enum-typed, and unit-tested: a
curated ``LOAD_BEARING_KEYS`` list is checked both for calibrated provenance *and* to be actually
emitted by the deploy-default monitor (``test_health_trust.py``), so a load-bearing signal renamed
or dropped from the monitor is caught rather than silently defaulting to ``uncalibrated``. Each entry
cites the Phase-0 substudy that earned it, which
binds the registry to ``phase0/index.md`` under the accuracy mandate: if a Phase-0 verdict changes,
both move together.

The registry annotates the **label-free** geometry/drift signals plus the **labelled canary** probes
(``knn_acc`` / ``linear_acc``). ``grad_norm`` (the instability reader) lives in the loss series, not
the monitor's health dict, so it is annotated for reference but excluded from the health-menu
completeness contract.
"""

from __future__ import annotations

from dataclasses import dataclass
from enum import Enum


class Backbone(str, Enum):
    """The backbone family a signal's calibration is native to."""

    MAE = "mae"
    JE = "je"  # joint-embedding (SimSiam / Barlow Twins), read at the projector


class FailureMode(str, Enum):
    """The failure mode a trust verdict is *about* (a signal can differ across modes)."""

    COLLAPSE = "collapse"
    FORGETTING = "forgetting"
    QUALITY = "quality"
    INSTABILITY = "instability"


class TrustStatus(str, Enum):
    """How far a signal can be trusted to read a mode, as Phase 0 found it."""

    CALIBRATED = "calibrated"  # PC fires + healthy stays quiet, seed-stable — trust it
    CANDIDATE = "candidate"  # separates, but reads the *regime* not the magnitude (reader upside)
    TRAP = "trap"  # mis-signals here (wrong direction, or calibrated to a *different* mode)
    NEEDS_CONTROL = "needs-two-sided-control"  # reads change, not the mode — needs a matched control
    NEEDS_LABELS = "needs-labels"  # trustworthy ground truth, but requires labels (canary only)
    UNCALIBRATED = "uncalibrated"  # no Phase-0 verdict for this (backbone, signal, mode)


@dataclass(frozen=True)
class TrustEntry:
    """One Phase-0 verdict about reading a signal for a mode, with its source."""

    mode: FailureMode
    status: TrustStatus
    caveat: str
    provenance: str  # the Phase-0 substudy id that earned this verdict

    def as_dict(self) -> dict[str, str]:
        """JSON-serializable form (enum values as plain strings) for the artifact header."""
        return {
            "mode": self.mode.value,
            "status": self.status.value,
            "caveat": self.caveat,
            "provenance": self.provenance,
        }


_JE_METHODS = frozenset({"simsiam", "simsiam_collapse", "barlow", "barlow_collapse"})


def backbone_family(method_name: str) -> Backbone:
    """Resolve an SSL method's ``name`` to the backbone family its calibration is keyed on.

    Args:
        method_name: The method's ``name`` (e.g. ``"mae"``, ``"simsiam"``, ``"barlow_collapse"``).

    Returns:
        The :class:`Backbone` family (MAE, or JE for any joint-embedding method).

    Raises:
        ValueError: If the method name maps to no known family.
    """
    if method_name == "mae":
        return Backbone.MAE
    if method_name in _JE_METHODS:
        return Backbone.JE
    raise ValueError(f"no backbone family for method name {method_name!r}")


def _e(mode: FailureMode, status: TrustStatus, caveat: str, provenance: str) -> TrustEntry:
    """Terse constructor to keep the registry table below readable."""
    return TrustEntry(mode=mode, status=status, caveat=caveat, provenance=provenance)


# The registry: (backbone family, emitted metric key) -> the Phase-0 verdicts for that signal.
# Keyed on the *emitted* key (surface-suffixed), because the verdict is surface-specific — the
# geometry suite earns trust at the projector (`_proj`), not the backbone (P0.2.4).
TRUST_MAP: dict[tuple[Backbone, str], tuple[TrustEntry, ...]] = {
    # --- Joint-embedding, projector surface — where the collapse suite is calibrated --------------
    (Backbone.JE, "rankme_proj"): (
        _e(
            FailureMode.COLLAPSE,
            TrustStatus.CALIBRATED,
            "universal collapse detector; but thins to ~1 marginal, LR-conditional reader (~2.17x, on "
            "the 2.0 bar) in the correlated single-pass corner",
            "P0.2.2",
        ),
        _e(
            FailureMode.FORGETTING,
            TrustStatus.NEEDS_CONTROL,
            "collapse-specificity guard: full-rank throughout means forgetting, not collapse — not "
            "itself a forgetting reader",
            "P0.6.0",
        ),
    ),
    (Backbone.JE, "mean_feature_var_proj"): (
        _e(
            FailureMode.COLLAPSE, TrustStatus.CALIBRATED, "point-vs-redundancy discriminator at the projector", "P0.2.2"
        ),
    ),
    (Backbone.JE, "offdiag_cov_proj"): (
        _e(
            FailureMode.COLLAPSE,
            TrustStatus.CALIBRATED,
            "redundancy-collapse detector (fires ~13-20x on whitening-ablated Barlow); a principled "
            "null under point collapse",
            "P0.2.3",
        ),
    ),
    (Backbone.JE, "uniformity_proj"): (
        _e(
            FailureMode.COLLAPSE,
            TrustStatus.CANDIDATE,
            "long-horizon-only — quiet single-pass, rho~0.92 redundant with rankme_proj",
            "P0.2.2",
        ),
        _e(
            FailureMode.FORGETTING,
            TrustStatus.TRAP,
            "calibrated to collapse; mis-signals on forgetting (flips direction, separates 3/5)",
            "P0.6.0",
        ),
    ),
    (Backbone.JE, "alignment_proj"): (
        _e(FailureMode.COLLAPSE, TrustStatus.CANDIDATE, "a non-standalone qualifier", "P0.2.2"),
        _e(
            FailureMode.FORGETTING,
            TrustStatus.CANDIDATE,
            "regime-level 2nd reader: co-separates 5/5 but reads the far-shift regime, not forgetting magnitude",
            "P0.6.0",
        ),
    ),
    (Backbone.JE, "cosine_drift_proj"): (
        _e(
            FailureMode.FORGETTING,
            TrustStatus.CALIBRATED,
            "past-data-free reader: projector frame-churn separates two-sided 5/5, keeps no task-A "
            "data (needs drift_surfaces)",
            "P0.6.1",
        ),
    ),
    (Backbone.JE, "cka_drift_proj"): (
        _e(
            FailureMode.FORGETTING,
            TrustStatus.TRAP,
            "backbone content-drift does not port past-data-free at the projector (3/5, inverts on a seed)",
            "P0.6.1",
        ),
    ),
    # --- Joint-embedding, backbone surface — reference only (genuine in-regime blind spots) --------
    (Backbone.JE, "rankme"): (
        _e(
            FailureMode.COLLAPSE,
            TrustStatus.NEEDS_CONTROL,
            "backbone has genuine in-regime blind spots — read collapse at the projector, keep the "
            "backbone as reference only",
            "P0.2.4",
        ),
    ),
    (Backbone.JE, "mean_feature_var"): (
        _e(FailureMode.COLLAPSE, TrustStatus.NEEDS_CONTROL, "read at the projector, not the backbone", "P0.2.4"),
    ),
    (Backbone.JE, "offdiag_cov"): (
        _e(FailureMode.COLLAPSE, TrustStatus.NEEDS_CONTROL, "read at the projector, not the backbone", "P0.2.4"),
    ),
    (Backbone.JE, "uniformity"): (
        _e(FailureMode.COLLAPSE, TrustStatus.NEEDS_CONTROL, "read at the projector, not the backbone", "P0.2.4"),
    ),
    (Backbone.JE, "cka_drift"): (
        _e(
            FailureMode.FORGETTING,
            TrustStatus.NEEDS_CONTROL,
            "canary A1 reader (task-A content drift) — trustworthy but canary-bound (needs retained "
            "task-A data) and reads change, so needs a two-sided control",
            "P0.6.0",
        ),
    ),
    (Backbone.JE, "cosine_drift"): (
        _e(
            FailureMode.FORGETTING,
            TrustStatus.NEEDS_CONTROL,
            "drift reads representation change, not forgetting; needs a matched control to subtract benign plasticity",
            "P0.6.1",
        ),
    ),
    # --- MAE, backbone surface — only the *loud* failures are readable label-free ------------------
    (Backbone.MAE, "rankme"): (
        _e(
            FailureMode.COLLAPSE,
            TrustStatus.CALIBRATED,
            "rank-collapse — a loud failure MAE internals do catch",
            "P0.5.0",
        ),
        _e(
            FailureMode.QUALITY,
            TrustStatus.NEEDS_CONTROL,
            "blind to quality by construction (full-rank-but-useless); kept only as the full-rank "
            "precondition / collapse guard",
            "P0.5.1",
        ),
    ),
    (Backbone.MAE, "cka_drift"): (
        _e(
            FailureMode.FORGETTING,
            TrustStatus.TRAP,
            "does not transfer as an MAE forgetting reader (demoted: drift ~ backbone movement, only "
            "loosely aligned with forgetting)",
            "P0.3.1",
        ),
    ),
    (Backbone.MAE, "cosine_drift"): (
        _e(
            FailureMode.FORGETTING,
            TrustStatus.NEEDS_CONTROL,
            "drift reads change, not forgetting; MAE resists, so the signal sits near the floor",
            "P0.3.1",
        ),
    ),
    (Backbone.MAE, "uniformity"): (
        _e(
            FailureMode.QUALITY,
            TrustStatus.TRAP,
            "mis-signals on MAE quality — the degraded rep looks *healthier*",
            "P0.5.1",
        ),
    ),
    (Backbone.MAE, "alignment"): (
        _e(
            FailureMode.QUALITY,
            TrustStatus.TRAP,
            "alignment(_strong) reads the shortcut-decoder regime, not quality — refuted as a quality reader",
            "P0.5.3",
        ),
    ),
    (Backbone.MAE, "mean_feature_var"): (
        _e(
            FailureMode.QUALITY,
            TrustStatus.TRAP,
            "the P0.2 collapse-geometry suite mis-signals for MAE quality",
            "P0.5.1",
        ),
    ),
    (Backbone.MAE, "offdiag_cov"): (
        _e(
            FailureMode.QUALITY,
            TrustStatus.TRAP,
            "the P0.2 collapse-geometry suite mis-signals for MAE quality",
            "P0.5.1",
        ),
    ),
    # --- Labelled canary probes (both families) — trustworthy ground truth, needs labels ----------
    (Backbone.MAE, "knn_acc"): (
        _e(
            FailureMode.QUALITY,
            TrustStatus.NEEDS_LABELS,
            "ground-truth canary — the only trustworthy MAE quality/forgetting read, but needs labels",
            "P0.5.2",
        ),
    ),
    (Backbone.MAE, "linear_acc"): (
        _e(
            FailureMode.QUALITY,
            TrustStatus.NEEDS_LABELS,
            "ground-truth canary — the only trustworthy MAE quality/forgetting read, but needs labels",
            "P0.5.2",
        ),
    ),
    (Backbone.JE, "knn_acc"): (
        _e(
            FailureMode.FORGETTING,
            TrustStatus.NEEDS_LABELS,
            "ground-truth canary probe — trustworthy but needs labels",
            "P0.6.0",
        ),
    ),
    (Backbone.JE, "linear_acc"): (
        _e(
            FailureMode.FORGETTING,
            TrustStatus.NEEDS_LABELS,
            "ground-truth canary probe — trustworthy but needs labels",
            "P0.6.0",
        ),
    ),
    # --- Instability (loss-series field, annotated for reference; not in the health menu) ----------
    (Backbone.MAE, "grad_norm"): (
        _e(
            FailureMode.INSTABILITY,
            TrustStatus.CALIBRATED,
            "fires ~5 steps ahead of the loss NaN — but only under raw-MSE + SGD; vanishes to a "
            "coincident alarm under AdamW + norm_pix",
            "P0.4.0",
        ),
    ),
    (Backbone.JE, "grad_norm"): (
        _e(
            FailureMode.INSTABILITY,
            TrustStatus.CALIBRATED,
            "fires ~5 steps ahead of the loss NaN — but only under raw-MSE + SGD; vanishes to a "
            "coincident alarm under AdamW + norm_pix",
            "P0.4.0",
        ),
    ),
}

# The label-free signals the deployment harness logs *by default* per family — the Phase-0-informed
# assignment (any signal is log/un-loggable via config). The labelled canary (`knn_acc`,
# `linear_acc`) is logged as a separate channel, not listed here.
DEFAULT_LABEL_FREE_SIGNALS: dict[Backbone, tuple[str, ...]] = {
    Backbone.JE: (
        "rankme_proj",
        "mean_feature_var_proj",
        "offdiag_cov_proj",
        "alignment_proj",
        "cosine_drift_proj",
        "rankme",  # backbone reference
    ),
    Backbone.MAE: (
        "rankme",
        "cosine_drift",
    ),
}

CANARY_SIGNALS: tuple[str, ...] = ("knn_acc", "linear_acc")

# The load-bearing keys the completeness test guards: every one must carry a non-UNCALIBRATED
# verdict with a Phase-0 provenance. New emitted keys default to UNCALIBRATED via `annotate`
# (self-describing, never crashing), but these must stay explicitly calibrated.
LOAD_BEARING_KEYS: dict[Backbone, tuple[str, ...]] = {
    Backbone.JE: (
        "rankme_proj",
        "mean_feature_var_proj",
        "offdiag_cov_proj",
        "uniformity_proj",
        "alignment_proj",
        "cosine_drift_proj",
        "knn_acc",
        "linear_acc",
    ),
    Backbone.MAE: (
        "rankme",
        "uniformity",
        "alignment",
        "knn_acc",
        "linear_acc",
    ),
}


def trust_for(family: Backbone, metric_key: str) -> tuple[TrustEntry, ...]:
    """The Phase-0 verdict(s) for one emitted signal, or an explicit UNCALIBRATED default.

    Never raises on an unknown key: an un-annotated signal is marked ``uncalibrated`` so the
    artifact stays self-describing and a newly-added instrument cannot silently look *trusted*.

    Args:
        family: The backbone family the run uses.
        metric_key: The emitted (surface-suffixed) metric key (e.g. ``"rankme_proj"``).

    Returns:
        The tuple of verdicts, or a single UNCALIBRATED entry for an unknown signal.
    """
    entry = TRUST_MAP.get((family, metric_key))
    if entry is not None:
        return entry
    return (
        TrustEntry(
            mode=FailureMode.COLLAPSE,  # placeholder mode; the status is the load-bearing field
            status=TrustStatus.UNCALIBRATED,
            caveat="no Phase-0 verdict for this signal on this backbone",
            provenance="",
        ),
    )


def annotate(metric_keys: list[str], family: Backbone) -> dict[str, list[dict[str, str]]]:
    """Build the JSON-serializable trust block stamped into an artifact header.

    Args:
        metric_keys: The emitted metric keys the run actually logged.
        family: The backbone family the run used.

    Returns:
        ``{metric_key: [verdict, ...]}`` — one list of serialized verdicts per key.
    """
    return {key: [e.as_dict() for e in trust_for(family, key)] for key in metric_keys}
