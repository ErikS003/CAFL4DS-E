"""Post-hoc diagnosis of task observability, health alarms and cached action utility."""

from __future__ import annotations

import numpy as np
from scipy.stats import spearmanr

from cafl4ds.functional_retention import plain

READERS = ("finetune", "linear", "knn")
DOMAINS = ("day", "night")


def direction(values, epsilon=1e-8):
    values = np.asarray(values, float)
    return np.where(values > epsilon, 1, np.where(values < -epsilon, -1, 0))


def agreement(values, epsilon=1e-8):
    signs = direction(values, epsilon)
    if np.all(signs > 0):
        return "all_improved"
    if np.all(signs < 0):
        return "all_worsened"
    if np.all(signs == 0):
        return "all_zero"
    return "mixed"


def correlation(x, y):
    x, y = np.asarray(x, float), np.asarray(y, float)
    if len(x) < 3 or np.ptp(x) < 1e-12 or np.ptp(y) < 1e-12:
        return {"n": len(x), "pearson": None, "spearman": None}
    return {"n": len(x), "pearson": float(np.corrcoef(x, y)[0, 1]), "spearman": float(spearmanr(x, y).statistic)}


def confusion(predicted, harmful, eligible):
    predicted, harmful, eligible = (np.asarray(v, bool) for v in (predicted, harmful, eligible))
    tp, fp, fn, tn = (
        int(np.sum(v)) for v in (predicted & harmful, predicted & ~harmful, ~predicted & harmful, ~predicted & ~harmful)
    )
    return {
        "n": len(predicted),
        "tp": tp,
        "fp": fp,
        "fn": fn,
        "tn": tn,
        "precision": tp / (tp + fp) if tp + fp else None,
        "recall": tp / (tp + fn) if tp + fn else None,
        "eligible": int(eligible.sum()),
        "coverage": float(eligible.mean()) if len(eligible) else None,
        "missed_unreadable": int(np.sum(harmful & ~predicted & ~eligible)),
        "missed_eligible": int(np.sum(harmful & ~predicted & eligible)),
    }


class HealthHarmAudit:
    def __init__(self, health_budget=0.05, minimum_gap=0.25, material_harm_pp=1.0, epsilon_pp=1e-8):
        self.budget, self.gap, self.harm, self.epsilon = health_budget, minimum_gap, material_harm_pp, epsilon_pp

    def episode(self, identity, landmarks, health, gaps, accuracies):
        h, g = np.asarray(health, float), np.asarray(gaps, float)
        if h.shape != (5, 2) or g.shape != h.shape or list(landmarks) != [0, 4, 8, 16, 32]:
            raise ValueError("Complete frozen five-landmark health path required")
        if not np.isfinite(h).all() or not np.isfinite(g).all():
            raise ValueError("Finite readings required")
        values = {reader: np.asarray(accuracies[reader], float) for reader in READERS}
        if any(v.shape != (5, 3, 3) or not np.isfinite(v).all() for v in values.values()):
            raise ValueError("Five checkpoints, three support panels and three domain scores required")
        active = g[0] >= self.gap
        best = np.where(active, h[0], np.inf)
        best_index = np.where(active, 0, -1)
        rows, first_alarm = [], None
        for endpoint in range(1, 5):
            excess = h[endpoint] - best
            predicted = active & (excess > self.budget)
            if first_alarm is None and predicted.any():
                first_alarm = landmarks[endpoint]
            prefix = first_alarm is None or landmarks[endpoint] <= first_alarm
            row = {
                **identity,
                "start": landmarks[endpoint - 1],
                "end": landmarks[endpoint],
                "global_start": identity["parent_age"] + landmarks[endpoint - 1],
                "global_end": identity["parent_age"] + landmarks[endpoint],
                "eligible_prior": active.copy(),
                "health_increment": h[endpoint] - h[endpoint - 1],
                "health_excess": [float(v) if np.isfinite(v) else None for v in excess],
                "alarm_candidates": predicted,
                "actual_prefix": prefix,
                "first_alarm": landmarks[endpoint] == first_alarm,
                "remaining_updates": 32 - landmarks[endpoint],
                "health_start": h[endpoint - 1],
                "health_end": h[endpoint],
                "gap_start": g[endpoint - 1],
                "gap_end": g[endpoint],
                "best_health_step": [landmarks[int(i)] if i >= 0 else None for i in best_index],
            }
            row["panel_retention_damage_pp"] = {
                reader: [
                    values[reader][int(best_index[j]), :, j + 1] - values[reader][endpoint, :, j + 1]
                    if active[j]
                    else None
                    for j in range(2)
                ]
                for reader in READERS
            }
            row["retention_damage_pp"] = {
                reader: [float(np.mean(v)) if v is not None else None for v in row["panel_retention_damage_pp"][reader]]
                for reader in READERS
            }
            for field, before, after in (
                ("immediate_damage", endpoint - 1, endpoint),
                ("subsequent_damage", endpoint, 4),
                ("rollback_advantage", endpoint - 1, 4),
            ):
                row[f"{field}_pp"] = {
                    reader: (values[reader][before] - values[reader][after]).mean(0) for reader in READERS
                }
                row[f"panel_{field}_pp"] = {
                    reader: values[reader][before] - values[reader][after] for reader in READERS
                }
            rows.append(row)
            newly = ~active & (g[endpoint] >= self.gap)
            best_index[newly | (active & (h[endpoint] < best))] = endpoint
            active |= newly
            best[active] = np.minimum(best[active], h[endpoint, active])
            row["active_after"] = active.copy()
        alarm = next((r for r in rows if r["first_alarm"]), None)
        ever_prefix = np.logical_or.reduce([g[0] >= self.gap] + [r["active_after"] for r in rows if r["actual_prefix"]])
        episode = {
            **identity,
            "first_alarm_step": first_alarm,
            "ever_active": ever_prefix,
            "alarm_domains": alarm["alarm_candidates"] if alarm else np.zeros(2, bool),
            "full_damage_pp": {reader: (values[reader][0] - values[reader][4]).mean(0) for reader in READERS},
            "panel_full_damage_pp": {reader: values[reader][0] - values[reader][4] for reader in READERS},
            "rollback_advantage_pp": alarm["rollback_advantage_pp"]
            if alarm
            else {reader: np.zeros(3) for reader in READERS},
            "panel_rollback_advantage_pp": alarm["panel_rollback_advantage_pp"]
            if alarm
            else {reader: np.zeros((3, 3)) for reader in READERS},
            "committed_updates": alarm["start"] if alarm else 32,
        }
        return plain(rows), plain(episode)

    def summarize(self, rows, episodes):
        prefix = [r for r in rows if r["actual_prefix"]]
        alarms = [r for r in prefix if r["first_alarm"]]
        sources = sorted({r["source"] for r in episodes})
        tables = {}
        for threshold, cutoff in (("any_drop", 0.0), ("material", self.harm)):
            tables[threshold] = {}
            for j, domain in enumerate(DOMAINS):

                def table(selected):
                    return confusion(
                        [r["first_alarm"] and r["alarm_candidates"][j] for r in selected],
                        [r["immediate_damage_pp"]["finetune"][j + 1] > cutoff + self.epsilon for r in selected],
                        [r["eligible_prior"][j] for r in selected],
                    )

                tables[threshold][domain] = {
                    "all_prefix": table(prefix),
                    "eligible_prefix": table([r for r in prefix if r["eligible_prior"][j]]),
                    "by_source": {s: table([r for r in prefix if r["source"] == s]) for s in sources},
                }
        associations = {}
        for reader in READERS:
            associations[reader] = {}
            for j, domain in enumerate(DOMAINS):
                associations[reader][domain] = {}
                for endpoint in ("immediate", "subsequent", "retention"):
                    applicable = rows if endpoint == "immediate" else [r for r in rows if r["remaining_updates"] > 0]
                    if endpoint == "retention":
                        applicable = [r for r in rows if r["eligible_prior"][j]]
                    for feature in ("increment", "excess"):
                        select = (
                            applicable if feature == "increment" else [r for r in applicable if r["eligible_prior"][j]]
                        )

                        def association(selected):
                            return correlation(
                                [r[f"health_{feature}"][j] for r in selected],
                                [
                                    r[f"{endpoint}_damage_pp"][reader][j if endpoint == "retention" else j + 1]
                                    for r in selected
                                ],
                            )

                        associations[reader][domain][f"{endpoint}_{feature}"] = {
                            "pooled_descriptive": association(select),
                            "by_source": {s: association([r for r in select if r["source"] == s]) for s in sources},
                        }
        source_effects = {
            s: np.mean([r["rollback_advantage_pp"]["finetune"] for r in episodes if r["source"] == s], axis=0)
            for s in sources
        }
        panel_agreement = {}
        for j, domain in enumerate(("balanced", *DOMAINS)):
            mean_effect = [r["rollback_advantage_pp"]["finetune"][j] for r in alarms]
            patterns = [
                agreement(np.array(r["panel_rollback_advantage_pp"]["finetune"])[:, j], self.epsilon) for r in alarms
            ]
            panel_agreement[domain] = {
                "mean_improved": int(np.sum(direction(mean_effect, self.epsilon) > 0)),
                "mean_worsened": int(np.sum(direction(mean_effect, self.epsilon) < 0)),
                "mean_zero": int(np.sum(direction(mean_effect, self.epsilon) == 0)),
                "patterns": {p: patterns.count(p) for p in ("all_improved", "all_worsened", "all_zero", "mixed")},
                "all_panels_material_improvement": sum(
                    np.all(np.array(r["panel_rollback_advantage_pp"]["finetune"])[:, j] > self.harm + self.epsilon)
                    for r in alarms
                ),
                "all_panels_material_worsening": sum(
                    np.all(np.array(r["panel_rollback_advantage_pp"]["finetune"])[:, j] < -self.harm - self.epsilon)
                    for r in alarms
                ),
            }
        episode_tables = {}
        for j, domain in enumerate(DOMAINS):
            episode_tables[domain] = confusion(
                [r["first_alarm_step"] is not None and r["alarm_domains"][j] for r in episodes],
                [r["full_damage_pp"]["finetune"][j + 1] > self.harm + self.epsilon for r in episodes],
                [r["ever_active"][j] for r in episodes],
            )
        retention_tables = {}
        for j, domain in enumerate(DOMAINS):
            selected = [r for r in prefix if r["eligible_prior"][j]]
            retention_tables[domain] = {
                "undefined_unreadable_prefix_blocks": len(prefix) - len(selected),
                "eligible_prefix": confusion(
                    [r["first_alarm"] and r["alarm_candidates"][j] for r in selected],
                    [r["retention_damage_pp"]["finetune"][j] > self.harm + self.epsilon for r in selected],
                    [True] * len(selected),
                ),
            }
        return plain(
            {
                "scope": "posthoc_reused_histories_queries_descriptive_only",
                "filter_qualified": False,
                "populations": {
                    "histories": len(sources),
                    "episodes": len(episodes),
                    "full_path_blocks": len(rows),
                    "actual_prefix_blocks": len(prefix),
                    "counterfactual_after_stop_blocks": len(rows) - len(prefix),
                    "first_alarms": len(alarms),
                    "alarms_with_remaining_work": sum(r["remaining_updates"] > 0 for r in alarms),
                    "zero_future_horizon_alarms": sum(r["remaining_updates"] == 0 for r in alarms),
                },
                "immediate_confusion": tables,
                "episode_material_harm_confusion": episode_tables,
                "retention_material_harm_confusion": retention_tables,
                "associations": associations,
                "alarm_panel_agreement": panel_agreement,
                "source_rollback_advantages_pp": source_effects,
                "mean_rollback_advantage_pp": np.mean(list(source_effects.values()), axis=0),
                "leave_one_source_out_descriptive_means_pp": {
                    s: np.mean([v for k, v in source_effects.items() if k != s], axis=0) if len(sources) > 1 else None
                    for s in sources
                },
                "support_unanimous_material_harm": {
                    domain: {
                        "blocks": sum(
                            np.all(
                                np.asarray(r["panel_immediate_damage_pp"]["finetune"])[:, j + 1]
                                > self.harm + self.epsilon
                            )
                            for r in prefix
                        )
                    }
                    for j, domain in enumerate(DOMAINS)
                },
            }
        )
