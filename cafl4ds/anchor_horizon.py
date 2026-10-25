"""Fixed-anchor budgets and separate-horizon marginal-image diagnostics."""

from __future__ import annotations

import numpy as np

from cafl4ds.prediction_benchmark import correlation


class AnchorHorizonStudy:
    """Retrospective diagnostics; never an authorized online admission certificate."""

    def __init__(
        self,
        gain_scale_pp=0.5,
        minimum_gain_pp=0.25,
        domain_budget_pp=1.0,
        rank_ratio_floor=0.6,
        variance_ratio_floor=0.5,
        precision_half_width_pp=0.25,
        required_precision_fraction=0.8,
    ):
        if min(gain_scale_pp, domain_budget_pp, precision_half_width_pp) <= 0:
            raise ValueError("Positive gain scale, domain budget and precision required")
        self.gain_scale_pp = gain_scale_pp
        self.minimum_gain_pp = minimum_gain_pp
        self.domain_budget_pp = domain_budget_pp
        self.rank_ratio_floor = rank_ratio_floor
        self.variance_ratio_floor = variance_ratio_floor
        self.precision_half_width_pp = precision_half_width_pp
        self.required_precision_fraction = required_precision_fraction

    def shadow(self, marginal, anchor, geometry_ratios):
        """Check marginal benefit and absolute anchor floors without exchanging budgets."""
        gain = float(marginal["crossed"]["lower"][0])
        losses = -np.asarray(anchor["crossed"]["lower"], float)[1:]
        if not np.isfinite(np.r_[gain, losses, list(geometry_ratios.values())]).all():
            raise ValueError("Finite estimates required")
        checks = {
            "marginal_gain": bool(gain >= self.minimum_gain_pp),
            "day_budget": bool(losses[0] <= self.domain_budget_pp),
            "night_budget": bool(losses[1] <= self.domain_budget_pp),
            "rank_floor": bool(geometry_ratios["rank"] >= self.rank_ratio_floor),
            "variance_floor": bool(geometry_ratios["variance"] >= self.variance_ratio_floor),
        }
        return {
            "reward": float(np.tanh(gain / self.gain_scale_pp)),
            "checks": checks,
            "retrospective_shadow_accept": all(checks.values()),
            "upper_anchor_loss_pp": losses,
            "remaining_budget_pp": self.domain_budget_pp - losses,
            "online_decision": "unknown_unqualified",
        }

    def summarize(self, records, health):
        """Keep every horizon and source visible; repeat observations are not new sources."""
        horizons = sorted({r["horizon"] for r in records})
        result = {}
        for horizon in horizons:
            rows = [r for r in records if r["horizon"] == horizon]
            widths = np.array([r["marginal"]["crossed"]["half_width"][1:] for r in rows])
            fraction = float((widths <= self.precision_half_width_pp).mean())
            result[str(horizon)] = {
                "contrasts": len(rows),
                "precision_fraction": fraction,
                "marginal_measurement_resolved": fraction >= self.required_precision_fraction,
                "median_crossed_half_width_pp": np.median(widths, axis=0),
                "source_mean_gain_pp": {
                    source: np.mean([r["marginal"]["mean_gain_pp"] for r in rows if r["source"] == source], axis=0)
                    for source in sorted({r["source"] for r in rows})
                },
                "shadow_acceptances": sum(r["shadow"]["retrospective_shadow_accept"] for r in rows),
            }
        correspondence = {}
        for horizon in horizons[1:]:
            pairs = []
            for row in records:
                if row["horizon"] != horizon:
                    continue
                first = next(
                    r for r in records if (r["source"], r["slot"], r["horizon"]) == (row["source"], row["slot"], 1)
                )
                pairs.append((first["marginal"]["mean_gain_pp"], row["marginal"]["mean_gain_pp"]))
            a, b = np.array(pairs).transpose(1, 0, 2)
            correspondence[str(horizon)] = {
                "capability_correlation_vs_h1": [correlation(a[:, d], b[:, d]) for d in range(3)],
                "nonzero_sign_flip_count_vs_h1": ((a * b) < 0).sum(0),
                "mean_absolute_gain_pp": np.abs(b).mean(0),
            }
        forecasts = {}
        for horizon in horizons:
            rows = [r for r in health if r["horizon"] == horizon]
            metrics = {}
            for metric in ("rank", "variance", "covariance", "cka", "cosine"):
                actual = np.array([r["effect"][metric] for r in rows])
                persistence = np.array([r["paid_h1_persistence"][metric] for r in rows])
                zero_mse, persistent_mse = float(np.mean(actual**2)), float(np.mean((actual - persistence) ** 2))
                metrics[metric] = {
                    "zero_mse": zero_mse,
                    "paid_h1_persistence_mse": persistent_mse,
                    "persistence_skill_vs_zero": 1 - persistent_mse / zero_mse if zero_mse > 1e-30 else None,
                    "source_mse": {
                        source: float(np.mean((actual - persistence)[[r["source"] == source for r in rows]] ** 2))
                        for source in sorted({r["source"] for r in rows})
                    },
                }
            forecasts[str(horizon)] = metrics
        return {
            "horizons": result,
            "cross_horizon": correspondence,
            "health_forecasts": forecasts,
            "independent_sources": len({r["source"] for r in records}),
            "scope": "reused_history_marginal_inclusion_development",
            "controller_activated": False,
            "health_predictor_fitted": False,
            "filter_qualified": False,
            "decision": "separate_protocol_required_before_health_fitting_or_admission",
        }
