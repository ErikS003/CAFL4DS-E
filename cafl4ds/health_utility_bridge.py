"""Frozen matched-batch menus, anchored utility models and diagnostic selectors."""

from __future__ import annotations

from typing import Any

import numpy as np
import torch
from scipy.stats import t

from cafl4ds.health_timeseries import METRICS, TRAIN_ORIGINS, feature, linear_predict

ACTIONS = ("uniform", "rank_low", "rank_high", "cka_low", "cka_high")
RECIPES = ("action", "history", "bridge")
DRAWS = (73003, 73009)
DOMAINS = ("balanced", "A", "B")


def candidate_menu(parent_index: int) -> list[list[int]]:
    """Sixteen distinct chronological32-of64 menus, independent of model/outcomes."""
    generator = torch.Generator().manual_seed(113003 + parent_index)
    menus = [list(range(1, 64, 2))]
    while len(menus) < 16:
        selected = torch.randperm(64, generator=generator)[:32].sort().values.tolist()
        if selected not in menus:
            menus.append(selected)
    return menus


def extrema(previews: list[dict[str, Any]]) -> dict[str, int]:
    """Choose both response directions without calling either direction helpful."""
    return {
        "uniform": 0,
        "rank_low": min(range(16), key=lambda i: (previews[i]["predicted"]["rank"], i)),
        "rank_high": min(range(16), key=lambda i: (-previews[i]["predicted"]["rank"], i)),
        "cka_low": min(range(16), key=lambda i: (previews[i]["predicted"]["cka"], i)),
        "cka_high": min(range(16), key=lambda i: (-previews[i]["predicted"]["cka"], i)),
    }


def contrast_features(observation: dict[str, Any], action: str, recipe: str) -> np.ndarray:
    """Every column is zero for uniform; health features are predicted responses only."""
    indicator = np.array([float(action == a) for a in ACTIONS[1:]])
    if recipe == "action":
        return indicator
    age = observation["age"]
    h = np.array(observation["history"])
    dr, dc = h[-1, 2] - h[-2, 2], h[-1, 3] - h[-2, 3]
    a, b = observation["previews"][action], observation["previews"]["uniform"]
    cheap = np.array([a[k] - b[k] for k in ("loss", "rgb_mean", "rgb_std")])
    base = np.r_[indicator, indicator * age, indicator * dr, indicator * dc, cheap]
    if recipe == "history":
        return base
    if recipe != "bridge":
        raise ValueError("Unknown frozen utility recipe")
    response = np.array([a["predicted"][m] - b["predicted"][m] for m in METRICS])
    return np.r_[base, response, response * dr, response * dc]


def temporal_context(series: list[dict[str, Any]], index: int, excluded: set[str]) -> dict[str, Any]:
    """Nested source exclusion prevents held future health entering training features."""
    target = series[index]
    output, fits = {}, {}
    for k, name in enumerate(METRICS):
        x, y = [], []
        for s in series:
            if s["stream"] in excluded:
                continue
            values = np.array(s["health"])
            for origin in TRAIN_ORIGINS:
                x.append(feature(s, k, origin, "delta_ar2"))
                y.append(values[origin + 4, k] - values[origin, k])
        prediction, fit = linear_predict(np.array(x), np.array(y), feature(target, k, 4, "delta_ar2")[None])
        output[name], fits[name] = float(prediction[0]), fit
    return {"predicted": output, "fits": fits, "excluded": sorted(excluded)}


def net_features(observation: dict[str, Any], context: dict[str, Any]) -> np.ndarray:
    """Seven low-capacity state/context features for uniform versus holding."""
    h = np.array(observation["history"])
    return np.array(
        [
            observation["age"],
            h[-1, 2],
            h[-1, 3],
            h[-1, 2] - h[-2, 2],
            h[-1, 3] - h[-2, 3],
            context["predicted"]["rank"],
            context["predicted"]["cka"],
        ]
    )


def fit_contrast(x: np.ndarray, y: np.ndarray) -> dict[str, Any]:
    """Uncentered RMS scaling and no intercept preserve the reference exactly."""
    scale = np.sqrt(np.mean(x * x, axis=0))
    scale[scale < 1e-12] = 1
    z = x / scale
    beta = np.linalg.solve(z.T @ z + 10 * np.eye(x.shape[1]), z.T @ y)
    return {"scale": scale.tolist(), "beta": beta.tolist(), "fit_rows": len(y)}


def predict_contrast(model: dict[str, Any], x: np.ndarray) -> float:
    """Evaluate an anchored frozen model."""
    return float((x / np.array(model["scale"])) @ np.array(model["beta"]))


def predict_net(model: dict[str, Any], x: np.ndarray) -> float:
    """Evaluate the fitting-only standardized uniform-net model."""
    return float(
        model["intercept"] + ((x - np.array(model["center"])) / np.array(model["scale"])) @ np.array(model["beta"])
    )


def choose(increment: dict[str, dict[str, float]], uniform_net: dict[str, float], stop: bool = False) -> str:
    """Diagnostic margins are fixed; uncertainty is not claimed calibrated."""
    if stop and max(uniform_net["balanced"] + increment[a]["balanced"] for a in ACTIONS) <= -0.25:
        return "hold"
    allowed = [
        a
        for a in ACTIONS[1:]
        if increment[a]["balanced"] >= 0.25
        and uniform_net["balanced"] + increment[a]["balanced"] >= 0.25
        and all(increment[a][d] >= -1 and uniform_net[d] + increment[a][d] >= -1 for d in ("A", "B"))
    ]
    return min(allowed, key=lambda a: (-increment[a]["balanced"], ACTIONS.index(a))) if allowed else "uniform"


def paired_interval(values: list[float], family: int = 8) -> dict[str, Any]:
    """Six history means, not candidates or support panels, set uncertainty."""
    a = np.array(values)
    width = float(t.ppf(1 - 0.05 / (2 * family), len(a) - 1) * a.std(ddof=1) / np.sqrt(len(a)))
    return {"mean": float(a.mean()), "adjusted_95_interval": [float(a.mean() - width), float(a.mean() + width)]}
