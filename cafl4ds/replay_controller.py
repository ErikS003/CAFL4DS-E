"""Feature-driven replay allocation with a training-pool disagreement fallback."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

from cafl4ds.filters.replay_ratio import ReplayRatio
from cafl4ds.replay_response import ACTIONS, ResponseModel, choose


@dataclass
class ReplayAllocationController:
    """Choose from predecision features; outcome labels never enter this interface."""

    objective_models: tuple[ResponseModel, ...]
    night_models: tuple[ResponseModel, ...]
    baseline: int
    gate: str = "agreement_guard"

    def decide(self, rows: list[dict[str, Any]]) -> dict[str, Any]:
        """Require agreement about improvement and retention relative to the same constant."""
        if len(rows) != 3 or {r["action"] for r in rows} != set(ACTIONS) or self.baseline not in ACTIONS:
            raise ValueError("Require one observation per replay action and a valid baseline")
        if len({r["state"] for r in rows}) != 1:
            raise ValueError("Observations must refer to one decision state")
        if self.gate not in ("best_gain", "agreement_guard"):
            raise ValueError("Unknown gate")
        if len(self.objective_models) != 3 or len(self.night_models) != 3:
            raise ValueError("Require full and two single-pool models")
        actions = [r["action"] for r in rows]
        forecasts = [dict(zip(actions, m.predict(rows), strict=True)) for m in self.objective_models]
        night = [dict(zip(actions, m.predict(rows), strict=True)) for m in self.night_models]
        proposed = choose(forecasts[0])
        accepted = (
            self.gate == "best_gain"
            or proposed == self.baseline
            or (
                all(p[proposed] - p[self.baseline] > 1e-12 for p in forecasts)
                and all(p[proposed] - p[self.baseline] >= -1e-12 for p in night)
            )
        )
        action = proposed if accepted else self.baseline
        return {
            "action": action,
            "proposed": proposed,
            "baseline": self.baseline,
            "accepted": accepted,
            "objective_predictions": forecasts,
            "night_predictions": night,
        }

    def apply(self, selector: ReplayRatio, rows: list[dict[str, Any]]) -> dict[str, Any]:
        """Connect the frozen feature decision to the actual allocation actuator."""
        decision = self.decide(rows)
        selector.set_action(decision["action"])
        return decision
