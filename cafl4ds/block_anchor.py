"""Separate block benefit from original and acquired capability budgets."""

from __future__ import annotations

import numpy as np


class BlockAnchorStudy:
    """Conditional measurement and retrospective shadows, never an online certificate."""

    def __init__(
        self,
        minimum_gain_pp=0.25,
        gain_scale_pp=0.5,
        domain_budget_pp=1.0,
        support_half_width_pp=0.5,
        crossed_half_width_pp=1.0,
        required_fraction=0.8,
        rank_ratio_floor=0.6,
        variance_ratio_floor=0.5,
    ):
        if min(gain_scale_pp, domain_budget_pp, support_half_width_pp, crossed_half_width_pp) <= 0:
            raise ValueError("Positive scales and budgets required")
        self.minimum_gain_pp = minimum_gain_pp
        self.gain_scale_pp = gain_scale_pp
        self.domain_budget_pp = domain_budget_pp
        self.support_half_width_pp = support_half_width_pp
        self.crossed_half_width_pp = crossed_half_width_pp
        self.required_fraction = required_fraction
        self.rank_ratio_floor = rank_ratio_floor
        self.variance_ratio_floor = variance_ratio_floor

    def precision(self, measurements):
        if not measurements:
            raise ValueError("Nonempty measurement population required")
        result = {}
        for kind, threshold in (
            ("support", self.support_half_width_pp),
            ("query", self.crossed_half_width_pp),
            ("crossed", self.crossed_half_width_pp),
        ):
            widths = np.array([r[kind]["half_width"][1:] for r in measurements], float)
            if not np.isfinite(widths).all():
                raise ValueError("Finite widths required")
            fraction = float((widths <= threshold + 1e-8).mean())
            result[kind] = {
                "fraction": fraction,
                "passed": fraction >= self.required_fraction,
                "median_half_width_pp": np.median(widths, axis=0),
                "domain_comparisons": int(widths.size),
            }
        return result

    def budgets(self, measurement):
        lower = np.asarray(measurement["crossed"]["lower"], float)[1:]
        upper = np.asarray(measurement["crossed"]["upper"], float)[1:]
        if not np.isfinite(np.r_[lower, upper]).all() or np.any(lower > upper):
            raise ValueError("Finite ordered budget ranges required")
        return [
            "within_conditional_range"
            if lo >= -self.domain_budget_pp
            else "resolved_violation"
            if hi < -self.domain_budget_pp
            else "unknown"
            for lo, hi in zip(lower, upper, strict=True)
        ]

    def shadow(self, benefit, anchor, ratios):
        lower = float(benefit["crossed"]["lower"][0])
        anchor_status, acquired_status = self.budgets(anchor), self.budgets(benefit)
        checks = {
            "balanced_gain": lower >= self.minimum_gain_pp,
            "warm_day_budget": anchor_status[0] == "within_conditional_range",
            "warm_night_budget": anchor_status[1] == "within_conditional_range",
            "rank_floor": ratios["rank"] >= self.rank_ratio_floor,
            "variance_floor": ratios["variance"] >= self.variance_ratio_floor,
        }
        original = bool(all(checks.values()))
        return {
            "reward": float(np.tanh(lower / self.gain_scale_pp)),
            "checks": checks,
            "warm_budget_status": anchor_status,
            "acquired_budget_status": acquired_status,
            "warm_shadow_accept": original,
            "warm_and_acquired_shadow_accept": original
            and all(s == "within_conditional_range" for s in acquired_status),
            "online_decision": "unknown_unqualified",
        }

    def summarize(self, block, anchor, panels, final_panels):
        by_horizon = {}
        for horizon in sorted({r["horizon"] for r in block}):
            rows = [r for r in block if r["horizon"] == horizon]
            by_horizon[str(horizon)] = {
                "precision": self.precision([r["measurement"] for r in rows]),
                "mean_gain_pp": np.mean([r["measurement"]["mean_gain_pp"] for r in rows], axis=0),
                "warm_shadow_acceptances": sum(r["shadow"]["warm_shadow_accept"] for r in rows),
                "warm_and_acquired_shadow_acceptances": sum(
                    r["shadow"]["warm_and_acquired_shadow_accept"] for r in rows
                ),
            }
        pooled_block = self.precision([r["measurement"] for r in block])
        pooled_anchor = self.precision([r["measurement"] for r in anchor])
        query_gate = pooled_block["query"]["passed"] and pooled_anchor["query"]["passed"]
        groups = {"warm_anchor": pooled_anchor, **{h: r["precision"] for h, r in by_horizon.items()}}
        precision_pass = all(r[k]["passed"] for r in groups.values() for k in ("support", "crossed"))
        useful, harmful = set(), set()
        for row in block:
            m = row["measurement"]
            if (
                m["mean_gain_pp"][0] >= self.minimum_gain_pp
                and m["crossed"]["lower"][0] > 0
                and all(s == "within_conditional_range" for s in self.budgets(m))
            ):
                useful.add(row["source"])
            if "resolved_violation" in self.budgets(m):
                harmful.add(row["source"])
        final_stage = panels == final_panels
        return {
            "panels": panels,
            "horizons": by_horizon,
            "warm_anchor_precision": pooled_anchor,
            "pooled_block_precision": pooled_block,
            "interim_query_gate": bool(query_gate),
            "final_precision_pass": bool(final_stage and precision_pass),
            "useful_sources": sorted(useful),
            "harmful_sources": sorted(harmful),
            "two_sided_opportunity": len(useful) >= 2 and len(harmful) >= 2,
            "measurement_ready": bool(final_stage and precision_pass and len(useful) >= 2 and len(harmful) >= 2),
            "source_groups": len({r["source"] for r in block}),
            "filter_qualified": False,
            "health_model_fitted": False,
            "controller_activated": False,
            "scope": "reused_development_conditional_measurement_not_source_confirmation",
        }
