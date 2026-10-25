"""Fixed-clock observation and source-level risk/timing comparisons on new histories."""

from __future__ import annotations

import itertools

import numpy as np
from scipy.stats import t

MONITOR = (132, *range(140, 445, 8))
POINTS = tuple(sorted({0, *MONITOR, 448, 148, 264, 160, 192, 224, 256, 288, 320}))


def class_accuracy(probabilities, labels, domains):
    correct = np.asarray(probabilities).argmax(-1) == np.asarray(labels)
    day, night = [100 * np.mean([correct[(domains == d) & (labels == c)].mean() for c in (0, 1)]) for d in (0, 1)]
    return np.array([(day + night) / 2, day, night])


def stopped_path(full, stop):
    indices = np.minimum(np.arange(len(POINTS)), POINTS.index(int(stop)))
    return np.asarray(full)[indices]


def path_metrics(path):
    """Final scores, drawdown and left-observation weighted post-burn-in deficits."""
    path = np.asarray(path, float)
    if path.shape != (len(POINTS), 3):
        raise ValueError("The universal observation grid and three metrics are required")
    deficit = np.maximum.accumulate(path, axis=0) - path
    weights = np.diff(POINTS).astype(float)
    weights[np.array(POINTS[:-1]) < 132] = 0
    mean_deficit = (deficit[:-1] * weights[:, None]).sum(0) / weights.sum()
    seconds_below = ((deficit[:-1] > 1) * weights[:, None]).sum(0) * 2
    drawdown = deficit.max(0)
    return np.r_[path[-1], drawdown, mean_deficit, seconds_below, max(drawdown[1:])]


METRICS = (
    "final_balanced",
    "final_day",
    "final_night",
    "drawdown_balanced",
    "drawdown_day",
    "drawdown_night",
    "mean_deficit_balanced",
    "mean_deficit_day",
    "mean_deficit_night",
    "seconds_below_balanced",
    "seconds_below_day",
    "seconds_below_night",
    "worst_domain_drawdown",
)


def source_interval(values):
    values = np.asarray(values, float)
    mean = values.mean(0)
    half = t.ppf(0.975, 7) * values.std(0, ddof=1) / np.sqrt(8)
    return {"mean": mean, "ci95": np.stack([mean - half, mean + half], axis=-1), "history_values": values}


class HealthTransfer:
    def summarize(self, full, stops, phase_lengths):
        full = np.asarray(full, float)
        stops = np.asarray(stops, int)
        if full.shape != (8, len(POINTS), 3) or stops.shape != (8,):
            raise ValueError("Eight complete fresh histories required")
        choices = {
            "frozen": [0] * 8,
            "full": [448] * 8,
            "fixed148": [148] * 8,
            "fixed264": [264] * 8,
            "health": stops.tolist(),
        }
        rules = {}
        metrics = {}
        for name, selected in choices.items():
            paths = np.stack([stopped_path(path, stop) for path, stop in zip(full, selected, strict=True)])
            value = np.stack([path_metrics(path) for path in paths])
            metrics[name] = value
            rules[name] = {
                "stops": selected,
                "total_updates": int(sum(selected)),
                "trajectory_mean_pp": paths.mean(0),
                "history_trajectories_pp": paths,
                "metrics": {key: source_interval(value[:, i]) for i, key in enumerate(METRICS)},
                "terminal_minus_warm": source_interval(value[:, :3] - full[:, 0]),
            }
        # A donor contributes a stopping time, never another history's capability outcome.
        donor_cube = np.array([[path_metrics(stopped_path(full[i], stop)) for stop in stops] for i in range(8)])
        permutations = np.array(list(itertools.permutations(range(8))), dtype=np.int16)
        assignment_means = donor_cube[np.arange(8), permutations].mean(1)
        expected_by_history = donor_cube.mean(1)
        random_mean = assignment_means.mean(0)
        actual = metrics["health"]
        risk_index = METRICS.index("worst_domain_drawdown")
        risk_improvement = expected_by_history[:, risk_index] - actual[:, risk_index]
        risk_range = source_interval(risk_improvement)
        quality_difference = actual[:, :3] - expected_by_history[:, :3]
        quality_full = actual[:, :3] - metrics["full"][:, :3]
        reduction = 1 - actual[:, risk_index].mean() / max(random_mean[risk_index], 1e-30)
        information_flag = bool(
            reduction >= 0.2
            and risk_improvement.mean() >= 0.5
            and risk_range["ci95"][0] > 0
            and quality_difference.mean(0)[0] >= -0.5
            and (quality_difference.mean(0)[1:] >= -1).all()
        )
        contrasts = {}
        for name in ("full", "fixed148", "fixed264"):
            contrasts[name] = {
                "terminal_difference_pp": source_interval(actual[:, :3] - metrics[name][:, :3]),
                "worst_drawdown_reduction_pp": source_interval(metrics[name][:, risk_index] - actual[:, risk_index]),
            }
        phases = []
        for stop, (day, night) in zip(stops, phase_lengths, strict=True):
            phases.append("day1" if stop <= day else "night" if stop <= day + night else "day2")
        dominators = []
        for name in ("fixed148", "fixed264"):
            r = metrics[name].mean(0)
            health = actual.mean(0)
            weak = r[risk_index] <= health[risk_index] and r[0] >= health[0] and sum(choices[name]) <= stops.sum()
            strict = r[risk_index] < health[risk_index] or r[0] > health[0] or sum(choices[name]) < stops.sum()
            if weak and strict:
                dominators.append(name)
        return {
            "rules": rules,
            "metric_order": list(METRICS),
            "points": list(POINTS),
            "matched_work_shuffle": {
                "assignments": len(permutations),
                "total_updates_each": int(stops.sum()),
                "mean_metrics": dict(zip(METRICS, random_mean, strict=True)),
                "history_expected_metrics": expected_by_history,
                "assignment_quantiles": np.quantile(assignment_means, [0.025, 0.5, 0.975], axis=0),
                "fraction_assignments_with_no_larger_drawdown": float(
                    (assignment_means[:, risk_index] <= actual[:, risk_index].mean() + 1e-12).mean()
                ),
            },
            "health_vs_shuffle": {
                "risk_reduction": risk_range,
                "fractional_risk_reduction": reduction,
                "terminal_difference_pp": source_interval(quality_difference),
            },
            "contrasts": contrasts,
            "stop_phase": phases,
            "distinct_stop_counts": len(set(stops)),
            "added_timing_information_flag": information_flag,
            "full_quality_tolerances_met": bool(
                quality_full.mean(0)[0] >= -0.5 and (quality_full.mean(0)[1:] >= -1).all()
            ),
            "fixed_controls_dominating_health": dominators,
            "source_histories": 8,
            "ssl_seed": 5003,
            "support_draws": 3,
            "controller_qualified": False,
            "capture_sessions_known": False,
        }, {"permutations": permutations, "assignment_mean_metrics": assignment_means}
