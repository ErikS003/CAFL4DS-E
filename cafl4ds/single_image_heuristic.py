"""Heuristic scheduling; learner transactions remain the runner's responsibility."""

from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np


@dataclass
class HeuristicController:
    """All counters advance in offered arrivals, independently of learner rollback."""

    mode: str = "normal"
    warning_count: int = 0
    clear_count: int = 0
    caution_checks: int = 0
    recovery_until: int = 0
    blocked_until: int = 0
    burst_remaining: int = 0
    last_burst: int = -256
    recent_commits: list[int] = field(default_factory=list)
    bad_task_count: int = 0
    good_task_count: int = 0

    def choose(self, i, prediction, clipped, threshold, draw, cfg):
        """Return probability, rate, reason; consume one external RNG draw every arrival."""
        self.recent_commits = [j for j in self.recent_commits if j >= i - cfg["burst"]["lookback_arrivals"]]
        if i < self.blocked_until:
            return 0.0, 0.0, "cooldown"
        if self.mode == "recovery" and i >= self.recovery_until:
            self.mode, self.clear_count, self.caution_checks = "caution", 0, 0
        if self.mode == "recovery":
            p, rate, reason = cfg["recovery"]["update_probability"], cfg["recovery"]["lr_factor"], "recovery"
        else:
            burst = cfg["burst"]
            if (
                not self.burst_remaining
                and i >= burst["lookback_arrivals"]
                and len(self.recent_commits) < burst["trigger_commits_below"]
                and i - self.last_burst >= burst["minimum_start_spacing_arrivals"]
            ):
                self.burst_remaining, self.last_burst = burst["updates"], i
            if self.burst_remaining:
                self.burst_remaining -= 1
                p, rate, reason = 1.0, burst["lr_factor"], "burst"
            else:
                settings = cfg["prediction"]
                if not np.isfinite(prediction) or clipped > settings["clipped_fraction_fallback_above"]:
                    p, reason = settings["fallback_probability"], "fallback"
                elif prediction < threshold:
                    p, reason = settings["probability_good"], "good"
                elif prediction < 0:
                    p, reason = settings["probability_borderline"], "borderline"
                else:
                    p, reason = settings["probability_unfavorable"], "unfavorable"
                rate = cfg["health"][f"{self.mode}_lr_factor"]
        return p, rate if draw < p else 0.0, reason

    def health(self, current, anchor, cfg):
        """Covariance alone never changes mode or vetoes a step."""
        if self.mode == "recovery":
            return False
        indices = [0, 1, 2, 6]
        a, v = np.asarray(anchor)[indices], np.asarray(current)[indices]
        if not np.isfinite(np.r_[a, v]).all() or np.any(a <= 1e-12):
            raise ValueError("Invalid fixed-bank health measurement")
        r, variance, covariance, reconstruction = v / a
        h = cfg["health"]
        warning = bool(
            min(r, variance) < h["rank_or_variance_ratio_below"]
            or (covariance > h["covariance_ratio_above"] and reconstruction > h["reconstruction_ratio_above"])
        )
        self.warning_count = self.warning_count + 1 if warning else 0
        self.clear_count = 0 if warning else self.clear_count + 1
        if self.mode == "normal" and self.warning_count >= h["warning_checks"]:
            self.mode, self.caution_checks = "caution", 0
        elif self.mode == "caution":
            self.caution_checks += 1
            if self.caution_checks >= 2 and self.clear_count >= h["clear_checks"]:
                self.mode, self.warning_count = "normal", 0
        return warning

    def task(self, scores, backup, original, cfg):
        """Scores are fractions. Return a command without reading final-query outcomes."""
        s, b, o = np.asarray(scores)[:3] * 100, np.asarray(backup)[:3] * 100, np.asarray(original)[:3] * 100
        if not np.isfinite(np.r_[s, b, o]).all():
            raise ValueError("Nonfinite development task measurement")
        t = cfg["task_brake"]
        bad = np.any(b - s > t["rollback_loss_pp"])
        severe = np.any(b - s > t["immediate_rollback_loss_pp"])
        self.bad_task_count = self.bad_task_count + 1 if bad else 0
        good = bool(
            s[0] - b[0] >= t["backup_balanced_gain_pp"]
            and np.all(s[1:] >= b[1:] - t["backup_subgroup_loss_pp"])
            and np.all(s[1:] >= o[1:] - t["backup_subgroup_loss_vs_original_pp"])
        )
        self.good_task_count = self.good_task_count + 1 if good else 0
        if severe or self.bad_task_count >= t["rollback_confirmation_readouts"]:
            self.bad_task_count = self.good_task_count = 0
            return "rollback"
        if self.good_task_count >= t["backup_confirmation_readouts"]:
            self.bad_task_count = self.good_task_count = 0
            return "backup"
        return "continue"

    def recover(self, next_arrival, cfg, numerical=False):
        """Do not rewind opportunity history or randomness after restoring weights."""
        self.mode, self.burst_remaining = "recovery", 0
        self.blocked_until = next_arrival + (cfg["numerical_cooldown_arrivals"] if numerical else 0)
        self.recovery_until = self.blocked_until + cfg["recovery"]["arrivals"]
        self.warning_count = self.clear_count = self.caution_checks = 0
