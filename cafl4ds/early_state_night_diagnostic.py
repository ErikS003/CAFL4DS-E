"""Descriptive cached diagnostics; no model construction or predictive fitting."""

from __future__ import annotations

import numpy as np

COMPONENTS = ("balanced", "day", "night", "day_city", "day_highway", "night_city", "night_highway")
FIELDS = ("rank_drop", "variance_drop", "covariance_rise", "cka_drift", "cosine_drift")
SIGNALS = tuple(f"{b}_{f}" for b in ("monitor", "audit") for f in FIELDS) + tuple(
    f"{window}_{f}" for f in ("loss", "gradient", "step_l2") for window in ("cumulative", "trailing")
)
ASSAYS = (
    (16, 512, "terminal"),
    (128, 128, "contemporaneous"),
    (128, 512, "terminal"),
    (128, 512, "remaining"),
    (512, 512, "contemporaneous"),
)


def masks(labels, domains):
    labels, domains = np.asarray(labels), np.asarray(domains)
    result = [(domains == d) & (labels == c) for d in (0, 1) for c in (0, 1)]
    if any(not m.any() for m in result):
        raise ValueError("All domain/class cells are required")
    return result


def components(values, labels, domains):
    cells = [np.asarray(values)[..., m].mean(-1) for m in masks(labels, domains)]
    day, night = (cells[0] + cells[1]) / 2, (cells[2] + cells[3]) / 2
    return np.stack([(day + night) / 2, day, night, *cells], axis=-1)


def transitions(before, after):
    a, b = np.asarray(before), np.asarray(after)
    if a.shape != b.shape or not np.isin(a, [0, 1]).all() or not np.isin(b, [0, 1]).all():
        raise ValueError("Aligned binary correctness arrays required")
    return (a == 1) & (b == 0), (a == 0) & (b == 1)


def average_ranks(x):
    _, inverse, counts = np.unique(np.asarray(x), return_inverse=True, return_counts=True)
    end = np.cumsum(counts)
    return (end - (counts - 1) / 2)[inverse]


def correlation(x, y, ranked=False):
    x, y = np.asarray(x, float), np.asarray(y, float)
    if x.ndim != 1 or x.shape != y.shape or len(x) < 3 or not np.isfinite([x, y]).all():
        raise ValueError("Finite aligned vectors of length>=3 required")
    if np.ptp(x) == 0 or np.ptp(y) == 0:
        return None
    if ranked:
        x, y = average_ranks(x), average_ranks(y)
    a, b = x - x.mean(), y - y.mean()
    return float(np.clip(np.dot(a, b) / np.sqrt(np.dot(a, a) * np.dot(b, b)), -1, 1))


def signal_values(record, origin, trailing=16):
    trace = record["trace"][:origin]
    if len(trace) != origin or origin < 1:
        raise ValueError("Signal window must end at an observed origin")
    values = {
        f"{bank}_{field}": record["readings"][str(origin)][bank]["change"][field]
        for bank in ("monitor", "audit")
        for field in FIELDS
    }
    for field, raw in (
        ("loss", [r["response"][0] for r in trace]),
        ("gradient", [r["response"][1] for r in trace]),
        ("step_l2", [r["encoder_step_l2"] for r in trace]),
    ):
        values[f"cumulative_{field}"] = float(np.mean(raw))
        values[f"trailing_{field}"] = float(np.mean(raw[-trailing:]))
    values["encoder_path_length"] = float(sum(r["encoder_step_l2"] for r in trace))
    return values


def association_rows(x, y, sources, futures):
    """Group deletion is sensitivity, never a held-out prediction."""
    sources, futures = np.asarray(sources), np.asarray(futures)
    groups = [("all", 0, np.ones(len(x), dtype=bool))]
    groups += [("leave_source", int(s), sources != s) for s in np.unique(sources)]
    groups += [("leave_future", int(f), futures != f) for f in np.unique(futures)]
    return [
        {
            "deletion": kind,
            "group": group,
            "n_cases": int(keep.sum()),
            "pearson": correlation(np.asarray(x)[keep], np.asarray(y)[keep]),
            "spearman": correlation(np.asarray(x)[keep], np.asarray(y)[keep], True),
        }
        for kind, group, keep in groups
    ]


def query_summary(lost_counts, gained_counts, draws):
    a, b = np.asarray(lost_counts), np.asarray(gained_counts)
    net = a.astype(float) - b
    positive = np.maximum(net, 0)
    denominator = positive.sum()
    take = max(1, int(np.ceil(0.1 * len(a))))
    return {
        "queries": len(a),
        "net_loss_query_fraction": float(np.mean(net > 0)),
        "persistent_lost_queries": int(np.sum(a >= 0.75 * draws)),
        "persistent_gained_queries": int(np.sum(b >= 0.75 * draws)),
        "top_decile_positive_net_loss_share": None
        if denominator == 0
        else float(np.sort(positive)[-take:].sum() / denominator),
    }
