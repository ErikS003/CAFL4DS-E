"""A frozen surrogate-improvement admission rule for singleton SSL."""

from __future__ import annotations

import numpy as np
from hydra.utils import instantiate

from cafl4ds.single_image_forecast import HEALTH


def objective(values, reference, settings):
    """Lower reconstruction plus explicit warm-relative rank/spread penalties."""
    values, reference = np.asarray(values), np.asarray(reference)
    r, v, loss = [HEALTH.index(k) for k in ("rank", "variance", "bank_reconstruction")]
    if np.any(reference[[r, v, loss]] <= 0) or not np.isfinite(values).all():
        raise ValueError("Finite health and positive objective references required")
    return (
        values[..., loss] / reference[loss]
        + settings["rank_weight"] * np.maximum(0, 1 - values[..., r] / reference[r])
        + settings["variance_weight"] * np.maximum(0, 1 - values[..., v] / reference[v])
    )


class FrozenRegressor:
    """Persistable prefix scaling and one fixed-capacity estimator."""

    def __init__(self, family, x, y, settings):
        self.center, self.scale = x.mean(0), x.std(0)
        self.scale[self.scale < 1e-12] = 1
        self.minimum, self.maximum = x.min(0), x.max(0)
        self.ycenter, self.yscale = float(y.mean()), max(float(y.std()), 1e-12)
        self.model = instantiate(settings["regressors"][family])
        self.model.fit((x - self.center) / self.scale, (y - self.ycenter) / self.yscale)

    def predict(self, x):
        return self.ycenter + self.yscale * self.model.predict((np.atleast_2d(x) - self.center) / self.scale)

    def outside(self, x):
        x = np.atleast_2d(x)
        return np.any((x < self.minimum) | (x > self.maximum), axis=1)


def fit_forecasters(a, reference, settings):
    """Choose families using old-source early-prefix labels only."""
    y = (objective(a["after"], reference, settings) - objective(a["before"], reference, settings)[:, None]).mean(1)
    prefix = a["origins"] + 1 <= settings["prefix_end"]
    inner = a["origins"] + 1 <= settings["inner_end"]
    valid = prefix & ~inner
    models, info = {}, {}
    for design in ("state", "image"):
        x = a[design]
        losses = {}
        for family in ("linear", "nonlinear"):
            p = np.zeros(valid.sum())
            for seed in settings["trajectory_seeds"]:
                fit, ask = inner & (a["seeds"] == seed), valid & (a["seeds"] == seed)
                m = FrozenRegressor(family, x[fit], y[fit], settings)
                p[a["seeds"][valid] == seed] = m.predict(x[ask])
            losses[family] = float(np.mean((p - y[valid]) ** 2))
        winner = min(losses, key=losses.get)
        models[design] = {
            int(seed): FrozenRegressor(
                winner, x[prefix & (a["seeds"] == seed)], y[prefix & (a["seeds"] == seed)], settings
            )
            for seed in settings["trajectory_seeds"]
        }
        info[design] = dict(selected=winner, validation_mse=losses)
    return models, dict(
        selection=info,
        prefix_rows=int(prefix.sum()),
        validation_rows=int(valid.sum()),
        objective_target="mean actual J(after singleton update) - J(before)",
    )


def matched_random(accepted, origins, period, seed):
    """Offline uniform identities with exactly the gate's count in every clip."""
    if len(accepted) != len(origins):
        raise ValueError("Aligned decisions and arrival indices required")
    rng, result = np.random.default_rng(seed), np.zeros(len(origins), bool)
    for clip in np.unique(origins // period):
        indices = np.flatnonzero(origins // period == clip)
        result[rng.choice(indices, int(np.asarray(accepted)[indices].sum()), replace=False)] = True
    return result


def features(before, lag, optimizer, origin, period, image=None):
    state = np.r_[before, lag, optimizer, origin, origin % period == 0, origin % period]
    return state if image is None else np.r_[state, image]
