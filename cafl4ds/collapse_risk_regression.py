"""Causal ridge forecasts and source-separated collapse-warning calibration."""

from __future__ import annotations

import numpy as np
from scipy.stats import t

ARMS = ("healthy", "heavy", "lowdiv", "blend005", "blend015", "blend100", "soft015", "soft050", "soft100")
METRICS = ("rank", "variance", "cka")
FEATURES = ("log_rank", "rank_increment", "previous_increment", "log_variance", "cka", "recent_loss", "age")


def geometry(z, reference):
    """Float64 raw-spectrum RankMe, normalized variance and centered linear CKA."""
    z, reference = np.asarray(z, float), np.asarray(reference, float)
    if z.ndim != 2 or z.shape != reference.shape or len(z) <= z.shape[1] or not np.isfinite(z).all():
        raise ValueError("Matched finite banks larger than their width required")
    spectrum = np.linalg.svd(z, compute_uv=False)
    p = spectrum / max(float(spectrum.sum()), 1e-30)
    rank = float(np.exp(-(p[p > 0] * np.log(p[p > 0])).sum()))
    normalized = z / np.maximum(np.linalg.norm(z, axis=1, keepdims=True), 1e-30)
    variance = float(normalized.var(axis=0, ddof=1).mean())
    a, b = z - z.mean(0), reference - reference.mean(0)
    denominator = np.linalg.norm(a.T @ a) * np.linalg.norm(b.T @ b)
    cka = float(np.linalg.norm(a.T @ b) ** 2 / max(float(denominator), 1e-30))
    return [rank, variance, float(np.clip(1 - cka, 0, 1))]


def ridge_fit(x, y, query, penalty=10.0):
    """Train-only scaling and an unpenalized intercept; support vector/multioutput targets."""
    x, y, query = np.asarray(x, float), np.asarray(y, float), np.asarray(query, float)
    if not (np.isfinite(x).all() and np.isfinite(y).all() and np.isfinite(query).all()):
        raise ValueError("Finite regressions required")
    center, scale = x.mean(0), x.std(0)
    scale[scale < 1e-12] = 1
    z = (x - center) / scale
    intercept = y.mean(0)
    beta = np.linalg.solve(z.T @ z + penalty * np.eye(x.shape[1]), z.T @ (y - intercept))
    prediction = intercept + (query - center) / scale @ beta
    return prediction, {
        "center": center.tolist(),
        "scale": scale.tolist(),
        "intercept": intercept.tolist(),
        "beta": beta.tolist(),
    }


def feature(health, losses, origin):
    """Require only bank readings through origin and losses strictly before its decision."""
    if origin < 16 or origin % 8:
        raise ValueError("Origin needs two eight-update increments")
    j = origin // 8
    h = np.asarray(health, float)[: j + 1]
    if len(h) != j + 1 or len(losses) < origin:
        raise ValueError("Missing observed prefix")
    logrank = np.log(h[:, 0] / h[0, 0])
    return np.r_[
        logrank[-1],
        np.diff(logrank)[-1],
        np.diff(logrank)[-2],
        np.log(max(h[-1, 1], 1e-30) / max(h[0, 1], 1e-30)),
        h[-1, 2],
        np.mean(losses[origin - 8 : origin]),
        origin / 128,
    ]


def interval(values, family=1):
    """Cohort-level t interval; overlapping origins/seeds never inflate replication."""
    v = np.asarray(values, float)
    half = float(t.ppf(1 - 0.05 / (2 * family), len(v) - 1) * v.std(ddof=1) / np.sqrt(len(v)))
    mean = float(v.mean())
    return {"mean": mean, "low": mean - half, "high": mean + half, "cohort_values": v.tolist()}


def observations(records):
    """Keep identity metadata apart from the prospective regression matrix."""
    points = []
    for i, row in enumerate(records):
        h = np.asarray(row["health"])
        for origin in range(16, 121, 8):
            j = origin // 8
            points.append(
                {
                    "record": i,
                    "group": row["group"],
                    "seed": row["seed"],
                    "arm": row["arm"],
                    "origin": origin,
                    "x": feature(h, row["losses"], origin).tolist(),
                    "y": float(np.log(h[j + 1, 0] / h[j, 0])),
                    "current_logrank": float(np.log(h[j, 0] / h[0, 0])),
                }
            )
    return points


def crossfit(records):
    """Hold out complete sources and never fit the second collapse mechanism."""
    points = observations(records)
    x, y = np.array([p["x"] for p in points]), np.array([p["y"] for p in points])
    groups = sorted({r["group"] for r in records})
    predicted = np.zeros(len(points))
    folds = {}
    for group in groups:
        fit = np.array([p["group"] != group and not p["arm"].startswith("soft") for p in points])
        test = np.array([p["group"] == group for p in points])
        prediction, model = ridge_fit(x[fit], y[fit], x[test])
        predicted[test] = prediction
        ratios = {
            arm: [r["health"][-1][0] / r["health"][0][0] for r in records if r["group"] != group and r["arm"] == arm]
            for arm in ("healthy", "blend100")
        }
        healthy, forced = (float(np.median(ratios[a])) for a in ("healthy", "blend100"))
        boundary = float(np.sqrt(healthy * forced))
        folds[group] = {
            "fit_points": np.flatnonzero(fit).tolist(),
            "test_points": np.flatnonzero(test).tolist(),
            "fit_groups": [g for g in groups if g != group],
            "model": model,
            "healthy_ratio": healthy,
            "forced_ratio": forced,
            "boundary": boundary,
        }
    for i, p in enumerate(points):
        p["predicted"] = float(predicted[i])
        p["persistence"] = 0.0
        p["trend"] = float(x[i, 1])
        p["boundary"] = folds[p["group"]]["boundary"]
    return {"features": FEATURES, "points": points, "folds": folds}


def warning_report(records, fitted, buffer=0.0):
    """Episode lead time and separately measured nominal false warnings."""
    points = fitted["points"]
    events, nominal = [], []
    for i, row in enumerate(records):
        selected = [p for p in points if p["record"] == i]
        cutoff = np.log(selected[0]["boundary"])
        rank = np.log(np.array(row["health"])[:, 0] / row["health"][0][0])
        below = rank <= cutoff
        starts = np.flatnonzero(below[:-1] & below[1:])
        first = int(starts[0] * 8) if len(starts) else None
        if row["arm"].startswith("soft") and first is not None:
            eligible = 24 <= first <= 120
            match = next((p for p in selected if p["origin"] == first - 8), None)
            warned = bool(
                eligible
                and match is not None
                and match["current_logrank"] > cutoff
                and match["current_logrank"] + match["predicted"] <= cutoff + buffer
            )
            events.append(
                {
                    "group": row["group"],
                    "seed": row["seed"],
                    "arm": row["arm"],
                    "first_event": first,
                    "eligible": eligible,
                    "warned": warned,
                    "lead_updates": 8 if warned else 0,
                }
            )
        for p in selected:
            j = p["origin"] // 8
            if row["arm"] == "healthy" and rank[j] > cutoff and rank[j + 1] > cutoff:
                nominal.append(
                    {
                        "group": row["group"],
                        "record": i,
                        "origin": p["origin"],
                        "warning": bool(rank[j] + p["predicted"] <= cutoff + buffer),
                    }
                )
    eligible = [e for e in events if e["eligible"]]
    rates = {
        g: float(np.mean([p["warning"] for p in nominal if p["group"] == g]))
        if any(p["group"] == g for p in nominal)
        else None
        for g in fitted["folds"]
    }
    return {
        "buffer": buffer,
        "events": events,
        "eligible_events": len(eligible),
        "all_events": len(events),
        "eligible_fraction": len(eligible) / len(events) if events else 0.0,
        "episode_recall": sum(e["warned"] for e in eligible) / len(eligible) if eligible else 0.0,
        "nominal_opportunities": len(nominal),
        "false_warnings": sum(p["warning"] for p in nominal),
        "false_warning_rate": float(np.mean([p["warning"] for p in nominal])) if nominal else None,
        "cohort_false_warning_rate": rates,
        "nominal": nominal,
    }


def summarize(records, parents, fitted):
    """Evaluate the frozen sequential prerequisites without a secondary rescue."""
    groups = sorted(fitted["folds"])
    endpoints = []
    for parent in parents:
        rows = {r["arm"]: r for r in records if r["group"] == parent["group"] and r["seed"] == parent["seed"]}
        h, c = rows["healthy"], rows["blend100"]
        endpoints.append(
            {
                "group": parent["group"],
                "seed": parent["seed"],
                "rank_separation": h["health"][-1][0] / c["health"][-1][0],
                "healthy_rank_ratio": h["health"][-1][0] / h["health"][0][0],
                "healthy_minus_forced": float(np.mean(h["scores"]) - np.mean(c["scores"])),
                "healthy_minus_warm": float(np.mean(h["scores"]) - np.mean(parent["warm_scores"])),
            }
        )
    anchor = interval([np.mean([e["healthy_minus_forced"] for e in endpoints if e["group"] == g]) for g in groups])
    beneficial = interval([np.mean([e["healthy_minus_warm"] for e in endpoints if e["group"] == g]) for g in groups])
    soft = [p for p in fitted["points"] if p["arm"].startswith("soft")]
    errors = {
        name: {g: float(np.mean([(p[name] - p["y"]) ** 2 for p in soft if p["group"] == g])) for g in groups}
        for name in ("predicted", "persistence", "trend")
    }
    mse = {k: float(np.mean(list(v.values()))) for k, v in errors.items()}
    improvements = {
        base: interval([errors[base][g] - errors["predicted"][g] for g in groups], family=2)
        for base in ("persistence", "trend")
    }
    skill = {
        base: 100 * (1 - mse["predicted"] / mse[base]) if mse[base] > 1e-30 else None
        for base in ("persistence", "trend")
    }
    warnings = warning_report(records, fitted)
    rates = list(warnings["cohort_false_warning_rate"].values())
    gates = {
        "instrument": bool(
            min(e["rank_separation"] for e in endpoints) >= 2
            and np.median([e["healthy_rank_ratio"] for e in endpoints]) >= 0.8
        ),
        "downstream_anchor": bool(anchor["mean"] >= 1 and anchor["low"] > 0),
        "beneficial_control": bool(beneficial["low"] > 0),
        "forecast": bool(all(skill[b] is not None and skill[b] >= 10 and improvements[b]["low"] > 0 for b in skill)),
        "warning": bool(
            warnings["eligible_events"] >= 6
            and warnings["episode_recall"] >= 0.8
            and warnings["eligible_fraction"] >= 0.8
            and warnings["false_warning_rate"] is not None
            and warnings["false_warning_rate"] <= 0.05
            and all(r is not None and r <= 0.1 for r in rates)
        ),
    }
    return {
        "gates": gates,
        "ready_for_separate_controller_experiment": all(gates.values()),
        "live_filter_qualified": False,
        "endpoints": endpoints,
        "downstream_anchor": anchor,
        "beneficial_control": beneficial,
        "soft_mechanism_mse": mse,
        "cohort_mse": errors,
        "mse_improvement": improvements,
        "skill_percent": skill,
        "warnings": warnings,
        "secondary_buffers": [warning_report(records, fitted, b) for b in (0.1, 0.2)],
        "scientific_updates": len(parents) * 320 + len(records) * 128,
        "replay_updates": len(parents) * 320 + len(records) * 128,
        "population_independence_established": False,
    }
