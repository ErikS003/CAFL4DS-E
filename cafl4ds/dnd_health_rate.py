"""A causal quarter-rate admission actuator with gradual recovery and repeat alarms."""

from __future__ import annotations

from typing import Any

import numpy as np

from cafl4ds.dnd_health_stopping import reading
from cafl4ds.dnd_health_transfer import MONITOR, path_metrics, source_interval


class HealthRate:
    """Reduce image admission on alarms while preserving a nonzero training rate."""

    def __init__(self, threshold: float = 0.1, patience: int = 2, ramp_arrivals: int = 32) -> None:
        """Keep the initial experiment's fixed alarm and recovery schedule."""
        if (threshold, patience, ramp_arrivals) != (0.1, 2, 32):
            raise ValueError("This experiment freezes threshold, persistence and recovery")
        self.threshold, self.patience, self.ramp_arrivals = threshold, patience, ramp_arrivals
        self.reference: np.ndarray | None = None
        self.last_alarm: int | None = None
        self.streak = self.credit = 0
        self.events: list[dict] = []

    def observe(self, step: int, features: np.ndarray) -> bool:
        """Read only the fixed monitor clock; an alarm affects subsequent arrivals."""
        if step not in MONITOR:
            return False
        if step == 132:
            if self.reference is not None:
                raise ValueError("The reference cannot be rebased")
            self.reference = np.array(features, copy=True)
            return False
        if self.reference is None:
            raise ValueError("Missing fixed monitor reference")
        values = reading(self.reference, features)
        unhealthy = bool((values >= self.threshold - 1e-12).any())
        self.streak = self.streak + 1 if unhealthy else 0
        fired = self.streak >= self.patience
        self.events.append(
            {"step": step, "values": values, "unhealthy": unhealthy, "streak": self.streak, "alarm": fired}
        )
        if fired:
            self.last_alarm, self.streak, self.credit = step, 0, 0
        return fired

    def admit(self, step: int) -> tuple[bool, int]:
        """Consume an arriving batch, returning admission and rate in quarter units."""
        if self.last_alarm is None or step <= self.last_alarm:
            numerator = 4
        else:
            numerator = min(4, 1 + (step - self.last_alarm - 1) // self.ramp_arrivals)
        self.credit += numerator
        accepted = self.credit >= 4
        if accepted:
            self.credit -= 4
        return accepted, numerator


def static_mask(fork: int, accepted_suffix: int | None = None) -> np.ndarray:
    """Quarter admission or exact-count even pacing after the common first alarm."""
    remaining = 448 - fork
    count = remaining // 4 if accepted_suffix is None else accepted_suffix
    if not 0 <= count <= remaining:
        raise ValueError("An admissible suffix work count is required")
    mask = np.ones(448, dtype=bool)
    mask[fork:] = [((i + 1) * count // remaining) > (i * count // remaining) for i in range(remaining)]
    return mask


class RateStudy:
    """Compare physical admission trajectories under the unchanged capability reader."""

    def summarize(self, curves: dict[str, np.ndarray], updates: dict[str, list[int]]) -> dict[str, Any]:
        """Keep source-paired tradeoffs, work and exploratory flags separate."""
        metrics, rules = {}, {}
        warm = np.asarray(curves["full"])[:, 0]
        for name, source_curves in curves.items():
            curve = np.asarray(source_curves)
            value = np.array([path_metrics(path) for path in curve])
            metrics[name] = value
            rules[name] = {
                "updates": updates[name],
                "total_updates": sum(updates[name]),
                "mean_updates": np.mean(updates[name]),
                "curves": curve,
                "mean_curve": curve.mean(0),
                "metrics": source_interval(value),
                "final_gain_over_warm": source_interval(value[:, :3] - warm),
            }
        contrasts, screens = {}, {}
        for name in ("full", "permanent", "quarter", "paced"):
            risk = metrics[name][:, -1] - metrics["adaptive"][:, -1]
            quality = metrics["adaptive"][:, :3] - metrics[name][:, :3]
            reduction = source_interval(risk)
            fractional = float(risk.mean() / max(metrics[name][:, -1].mean(), 1e-30))
            contrasts[name] = {
                "risk_reduction_pp": reduction,
                "fractional_risk_reduction": fractional,
                "final_difference_pp": source_interval(quality),
            }
            if name in ("full", "paced"):
                screens[name] = bool(
                    risk.mean() >= 0.5
                    and fractional >= 0.2
                    and reduction["ci95"][0] > 0
                    and quality.mean(0)[0] >= -0.5
                    and (quality.mean(0)[1:] >= -1).all()
                )
        return {
            "rules": rules,
            "contrasts": contrasts,
            "exploratory_screens": screens,
            "exact_adaptive_paced_work": updates["adaptive"] == updates["paced"],
            "source_histories": 8,
            "exploratory_reuse": True,
            "controller_qualified": False,
        }
