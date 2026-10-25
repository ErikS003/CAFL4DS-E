"""The P1.0.2 deployment-prototype harness — multivariate, trust-annotated health logging.

Where the P1.0.0 harness (:mod:`cafl4ds.harness`) logs a *directional* smoke on toy data, the
deployment prototype logs the **full multivariate health vector** of one config-selected backbone on
the BDD regime diet, partitioned into two self-describing channels — a **label-free** vector (what a
real deployment would see) and a **labelled canary** (kNN / linear probes, ground truth available
because we control the eval set) — with every signal annotated by its Phase-0 calibrated status
(:mod:`cafl4ds.health_trust`) so the corpus is safe to mine without the code.

It reads **no degradation**. The acceptance bar is a **two-tier validation**: Tier A (the anyone-can-
run wiring check) asserts the pipeline is wired — every configured signal present at every checkpoint,
both channels populated, the trust map complete, nothing non-finite, memory flat; Tier B (our
sanity-of-readings pass) confirms the readings are plausible. This module owns the report the harness
writes; the arm execution reuses :mod:`cafl4ds.harness` untouched.
"""

from __future__ import annotations

import math
from typing import Any

from cafl4ds.harness import Arm, all_finite
from cafl4ds.health_trust import CANARY_SIGNALS, Backbone, annotate

# The deploy-corpus schema version (independent of the P1.0.0 comparison schema).
SCHEMA_VERSION = 1

# Health-record bookkeeping fields that are not instrument signals.
_BOOKKEEPING = frozenset({"step", "era", "loss", "run", "series"})

# Drift keys in preference order — the first present in the series is the Tier-B drift read (the
# projector current-stream reader when logged, else the backbone drift).
_TIER_B_DRIFT_KEYS = ("cosine_drift_proj", "cosine_drift", "cka_drift_proj", "cka_drift")

# The Tier-B plausibility floor — loose by design (see :func:`tier_b_report`).
DEFAULT_TIER_B_THRESHOLDS: dict[str, float] = {
    "rankme_lo": 0.5,  # RankMe must stay above near-collapse ...
    "rankme_hi": 1000.0,  # ... and bounded (not blown up to a degenerate value)
    "min_final_drift": 0.0,  # representation drift must accumulate off zero
    "canary_margin": 0.0,  # the mean canary must clear chance by at least this
}


def _finite_series(health: list[dict[str, Any]], key: str) -> list[float]:
    """The finite numeric values of one signal across a health series (bools/NaN excluded)."""
    out: list[float] = []
    for record in health:
        value = record.get(key)
        if isinstance(value, bool) or not isinstance(value, int | float):
            continue
        if math.isfinite(float(value)):
            out.append(float(value))
    return out


def tier_b_report(
    live_health: list[dict[str, Any]],
    *,
    canary_chance: float,
    thresholds: dict[str, float] | None = None,
    canary_key: str = "knn_acc",
    rankme_key: str = "rankme",
) -> dict[str, Any]:
    """The Tier-B sanity-of-readings verdict — a reproducible, *loose* plausibility floor.

    Not a degradation read, and not the Tier-A wiring gate: it confirms the logged trajectory is
    *plausible* the way P1.0.2's acceptance bar describes — RankMe stays in a sane range, drift
    accumulates off zero, and the labelled canary sits above chance. Deliberately loose (floors on
    *aggregates*, never a per-checkpoint gate): the canary is the thin channel — its mean clears
    chance by only a few points and *individual* checkpoints dip below chance — so a per-step canary
    gate would flap ``passed=false`` on a perfectly healthy drive (RankMe stays comfortably in range
    and the projector drift accumulates fine; the canary is the one that needs the aggregate floor —
    audit P1.0 B1 / C2). Downstream studies can tighten the thresholds as their diets bite harder.

    Args:
        live_health: The live arm's (projected) health records.
        canary_chance: Random-guess accuracy for the canary probe (``1 / #canary classes``).
        thresholds: Overrides for :data:`DEFAULT_TIER_B_THRESHOLDS` (``rankme_lo`` / ``rankme_hi`` /
            ``min_final_drift`` / ``canary_margin``).
        canary_key: The labelled-probe key averaged for the above-chance check.
        rankme_key: The RankMe key range-checked (the backbone surface, logged for both families).

    Returns:
        The Tier-B report: per-criterion booleans, the reported aggregates, the thresholds, and the
        overall ``passed``.
    """
    t = {**DEFAULT_TIER_B_THRESHOLDS, **(thresholds or {})}
    rankme = _finite_series(live_health, rankme_key)
    rankme_in_range = bool(rankme) and all(t["rankme_lo"] <= v <= t["rankme_hi"] for v in rankme)

    drift_key, drift_final = None, None
    for key in _TIER_B_DRIFT_KEYS:
        series = _finite_series(live_health, key)
        if series:
            drift_key, drift_final = key, series[-1]
            break
    drift_accumulated = drift_final is not None and drift_final > t["min_final_drift"]

    canary = _finite_series(live_health, canary_key)
    canary_mean = (sum(canary) / len(canary)) if canary else None
    canary_above_chance = canary_mean is not None and canary_mean > canary_chance + t["canary_margin"]

    return {
        "passed": bool(rankme_in_range and drift_accumulated and canary_above_chance),
        "checks": {
            "rankme_in_range": rankme_in_range,
            "drift_accumulated": bool(drift_accumulated),
            "canary_above_chance": bool(canary_above_chance),
        },
        "reported": {
            "rankme_min": min(rankme) if rankme else None,
            "rankme_max": max(rankme) if rankme else None,
            "drift_key": drift_key,
            "final_drift": drift_final,
            "canary_key": canary_key,
            "canary_mean": canary_mean,
            "canary_chance": canary_chance,
        },
        "thresholds": t,
    }


def emitted_signals(health: list[dict[str, Any]]) -> list[str]:
    """The instrument signals present across a health series (bookkeeping fields excluded).

    Args:
        health: The arm's health records (one per checkpoint).

    Returns:
        The sorted union of signal keys logged over the checkpoints.
    """
    keys: set[str] = set()
    for record in health:
        keys |= {k for k in record if k not in _BOOKKEEPING}
    return sorted(keys)


def split_channels(signals: list[str]) -> dict[str, list[str]]:
    """Partition signals into the label-free vector and the labelled canary channel.

    Args:
        signals: The emitted signal keys.

    Returns:
        ``{"label_free": [...], "canary": [...]}`` — the canary holds the labelled probes
        (``knn_acc`` / ``linear_acc``), the label-free vector holds the rest.
    """
    canary = sorted(s for s in signals if s in CANARY_SIGNALS)
    label_free = sorted(s for s in signals if s not in CANARY_SIGNALS)
    return {"label_free": label_free, "canary": canary}


def project_health(health: list[dict[str, Any]], keep: list[str] | None) -> list[dict[str, Any]]:
    """Restrict each health record to the configured signals (plus bookkeeping), or pass through.

    This is the log/un-log lever: ``keep=None`` logs everything the monitor emitted; a list keeps
    only those signals (the bookkeeping fields are always retained).

    Args:
        health: The arm's health records.
        keep: The signals to keep, or ``None`` to keep all.

    Returns:
        The (possibly projected) health records.
    """
    if keep is None:
        return health
    allowed = _BOOKKEEPING | set(keep)
    return [{k: v for k, v in record.items() if k in allowed} for record in health]


def channel_completeness(health: list[dict[str, Any]], expected: list[str]) -> dict[str, Any]:
    """Check every expected signal is present (non-``None``) at every checkpoint.

    Args:
        health: The (projected) health records.
        expected: The signals that must appear at every checkpoint.

    Returns:
        The completeness report: the ``complete`` verdict, the expected set, and any per-step gaps.
    """
    missing: dict[str, list[str]] = {}
    for record in health:
        absent = [s for s in expected if record.get(s) is None]
        if absent:
            missing[str(int(record["step"]))] = absent
    return {
        "complete": not missing,
        "expected": sorted(expected),
        "missing": missing,
        "num_checkpoints": len(health),
    }


def build_deploy_report(
    *,
    config_header: dict[str, Any],
    family: Backbone,
    live: Arm,
    expected_signals: list[str],
    pc: Arm | None = None,
    b5: Arm | None = None,
    leak: dict[str, Any] | None = None,
    log_signals: list[str] | None = None,
    canary_chance: float | None = None,
    tier_b_thresholds: dict[str, float] | None = None,
) -> dict[str, Any]:
    """Assemble the deploy corpus — the config, the trust-annotated channels, the arms, the validation.

    Args:
        config_header: The run's factor levels (backbone, diet, seed, device, …).
        family: The backbone family (drives the trust annotation).
        live: The live arm (the system under test — its health series is the corpus).
        expected_signals: The signals Tier-A completeness requires at every checkpoint.
        pc: Optional manufactured-pathology gate arm.
        b5: Optional frozen-floor gate arm.
        leak: Optional long-horizon leak report (adds the flat-memory check to the verdict).
        log_signals: Optional restriction of the logged signals (``None`` logs all emitted).
        canary_chance: Random-guess accuracy for the canary probe (``1 / #canary classes``). When
            given, the reproducible **Tier-B** sanity-of-readings verdict is computed and attached
            under ``tier_b`` — separate from the Tier-A ``validation.passed`` wiring gate.
        tier_b_thresholds: Optional overrides for the loose Tier-B plausibility floor.

    Returns:
        The deploy-corpus dict, ready to serialize.
    """
    arms = [a for a in (live, pc, b5) if a is not None]
    projected = {a.role: project_health(a.health, log_signals) for a in arms}
    live_health = projected[live.role]

    signals = emitted_signals(live_health)
    channels = split_channels(signals)
    trust = annotate(signals, family)
    completeness = channel_completeness(live_health, expected_signals)

    finite = all_finite(*arms)
    channels_present = bool(channels["label_free"] and channels["canary"])
    trust_complete = all(any(v["status"] != "uncalibrated" for v in trust.get(s, [])) for s in expected_signals)

    checks = [finite, channels_present, completeness["complete"], trust_complete]
    if leak is not None:
        checks.append(bool(leak["flat"]))
    validation = {
        "finite": finite,
        "channels_present": channels_present,
        "channel_complete": completeness["complete"],
        "trust_complete": trust_complete,
        "memory_flat": (bool(leak["flat"]) if leak is not None else None),
        "passed": all(checks),
    }

    report: dict[str, Any] = {
        "schema_version": SCHEMA_VERSION,
        "study": "P1.0.2",
        "config": config_header,
        "backbone_family": family.value,
        "channels": channels,
        "trust": trust,
        "arms": {
            role: {"run_name": next(a.name for a in arms if a.role == role), "health": h}
            for role, h in projected.items()
        },
        "completeness": completeness,
        "validation": validation,
    }
    if leak is not None:
        report["leak_check"] = leak
    if canary_chance is not None:
        report["tier_b"] = tier_b_report(live_health, canary_chance=canary_chance, thresholds=tier_b_thresholds)
    return report
