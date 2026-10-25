"""Source-held-out bridge fitting and frozen diagnostic policy evaluation."""

from __future__ import annotations

from typing import Any

import numpy as np

from cafl4ds.health_timeseries import linear_predict
from cafl4ds.health_utility_bridge import (
    ACTIONS,
    DOMAINS,
    RECIPES,
    choose,
    contrast_features,
    fit_contrast,
    net_features,
    paired_interval,
    predict_contrast,
    predict_net,
    temporal_context,
)
from cafl4ds.prediction_benchmark import correlation, improvement, scores


def fit_models(
    observations: list[dict[str, Any]],
    outcomes: list[dict[str, Any]],
    holds: list[dict[str, Any]],
    series: list[dict[str, Any]],
) -> dict[str, Any]:
    """Exclude held histories from both response fitting and nested temporal context."""
    lookup = {(r["parent"], r["action"]): r for r in outcomes}
    models = {}
    for group in sorted({o["stream"] for o in observations}):
        fitting = [o for o in observations if o["stream"] != group]
        contexts = {str(o["parent"]): temporal_context(series, o["parent"], {group, o["stream"]}) for o in observations}
        recipes = {}
        for recipe in RECIPES:
            recipes[recipe] = {}
            x = np.array([contrast_features(o, a, recipe) for o in fitting for a in ACTIONS[1:]])
            for domain in DOMAINS:
                y = np.array(
                    [
                        lookup[o["parent"], a]["scores"]["finetune"][domain]
                        - lookup[o["parent"], "uniform"]["scores"]["finetune"][domain]
                        for o in fitting
                        for a in ACTIONS[1:]
                    ]
                )
                recipes[recipe][domain] = fit_contrast(x, y)
        x = np.array([net_features(o, contexts[str(o["parent"])]) for o in fitting])
        net = {}
        for domain in DOMAINS:
            y = np.array(
                [
                    lookup[o["parent"], "uniform"]["scores"]["finetune"][domain]
                    - holds[o["parent"]]["scores"]["finetune"][domain]
                    for o in fitting
                ]
            )
            _, net[domain] = linear_predict(x, y, x[:1])
        models[group] = {
            "recipes": recipes,
            "net": net,
            "contexts": contexts,
            "training_parents": [o["parent"] for o in fitting],
            "held_history": group,
        }
    return {"folds": models, "training_draw": 0}


def frozen_decisions(observations: list[dict[str, Any]], models: dict[str, Any]) -> list[dict[str, Any]]:
    """Use draw1 observations and frozen coefficients before any draw1 readout."""
    decisions = []
    for o in observations:
        fold = models["folds"][o["stream"]]
        xnet = net_features(o, fold["contexts"][str(o["parent"])])
        net = {d: predict_net(fold["net"][d], xnet) for d in DOMAINS}
        recipes = {}
        for recipe in RECIPES:
            increment = {
                a: {d: predict_contrast(fold["recipes"][recipe][d], contrast_features(o, a, recipe)) for d in DOMAINS}
                for a in ACTIONS
            }
            recipes[recipe] = {
                "predictions": increment,
                "content_choice": choose(increment, net),
                "stop_choice": choose(increment, net, stop=True),
            }
        decisions.append({"parent": o["parent"], "stream": o["stream"], "uniform_net": net, "recipes": recipes})
    return decisions


def analyze(
    outcomes: list[dict[str, Any]], holds: list[dict[str, Any]], decisions: list[dict[str, Any]]
) -> dict[str, Any]:
    """Keep forecast skill, selected utility, stop value and retention as separate results."""
    lookup = {(r["parent"], r["draw"], r["action"]): r for r in outcomes}
    groups = sorted({d["stream"] for d in decisions})
    parents = {g: [d["parent"] for d in decisions if d["stream"] == g] for g in groups}

    def quality(parent: int, draw: int, action: str, domain: str, reader: str = "finetune") -> float:
        return (
            holds[parent]["scores"][reader][domain]
            if action == "hold"
            else lookup[parent, draw, action]["scores"][reader][domain]
        )

    rows = [(d, a) for d in decisions for a in ACTIONS[1:]]
    y = np.array(
        [quality(d["parent"], 1, a, "balanced") - quality(d["parent"], 1, "uniform", "balanced") for d, a in rows]
    )
    predicted = {
        "zero": np.zeros(len(rows)),
        **{r: np.array([d["recipes"][r]["predictions"][a]["balanced"] for d, a in rows]) for r in RECIPES},
    }
    forecast_scores = scores(y, predicted, np.array([d["stream"] for d, _ in rows]))
    forecast_contrasts = {
        r: improvement(forecast_scores["bridge"], forecast_scores[r], 8) for r in ("action", "history")
    }
    choices = {
        f"{r}_{kind}": {d["parent"]: d["recipes"][r][kind + "_choice"] for d in decisions}
        for r in RECIPES
        for kind in ("content", "stop")
    }
    choices.update({a: {d["parent"]: a for d in decisions} for a in (*ACTIONS, "hold")})
    # This oracle sees draw0 labels and is explicitly an optimistic offline opportunity check.
    oracle = {}
    for d in decisions:
        p = d["parent"]
        allowed = [
            a
            for a in ACTIONS
            if quality(p, 0, a, "balanced") - quality(p, 0, "uniform", "balanced") >= 0.25
            and quality(p, 0, a, "balanced") - quality(p, 0, "hold", "balanced") >= 0.25
            and all(
                quality(p, 0, a, domain) - quality(p, 0, control, domain) >= -1
                for domain in ("A", "B")
                for control in ("uniform", "hold")
            )
        ]
        oracle[p] = (
            min(allowed, key=lambda a: (-quality(p, 0, a, "balanced"), ACTIONS.index(a))) if allowed else "uniform"
        )
    choices["draw0_label_oracle"] = oracle
    policy_scores = {}
    for name, choice in choices.items():
        readers = {}
        for reader in ("finetune", "linear", "knn"):
            domains = {}
            for domain in DOMAINS:
                domains[domain] = {
                    control: {
                        g: float(
                            np.mean(
                                [
                                    quality(p, 1, choice[p], domain, reader) - quality(p, 1, control, domain, reader)
                                    for p in parents[g]
                                ]
                            )
                        )
                        for g in groups
                    }
                    for control in ("uniform", "hold")
                }
            readers[reader] = domains
        policy_scores[name] = {
            "choices": {str(p): a for p, a in choice.items()},
            "readers": readers,
            "nonuniform_choices": sum(
                a != "hold" and lookup[p, 1, a]["selected_slots"] != lookup[p, 1, "uniform"]["selected_slots"]
                for p, a in choice.items()
            ),
            "hold_choices": sum(a == "hold" for a in choice.values()),
        }
    primary = policy_scores["bridge_content"]["readers"]["finetune"]
    history = policy_scores["history_content"]["readers"]["finetune"]
    policy_contrasts = {
        "balanced_vs_uniform": paired_interval(list(primary["balanced"]["uniform"].values())),
        "balanced_vs_history": paired_interval(
            [primary["balanced"]["uniform"][g] - history["balanced"]["uniform"][g] for g in groups]
        ),
    }
    for domain in ("A", "B"):
        for control in ("uniform", "hold"):
            policy_contrasts[domain + "_vs_" + control] = paired_interval(list(primary[domain][control].values()))
    gates = {"forecast_vs_" + name: c["passes"] for name, c in forecast_contrasts.items()}
    for name, c in policy_contrasts.items():
        gates[name] = (
            bool(c["mean"] >= 0.25 and c["adjusted_95_interval"][0] > 0)
            if name.startswith("balanced")
            else bool(c["adjusted_95_interval"][0] >= -1)
        )
    gates["content_switches"] = policy_scores["bridge_content"]["nonuniform_choices"] > 0
    draw0 = [quality(d["parent"], 0, a, "balanced") - quality(d["parent"], 0, "uniform", "balanced") for d, a in rows]
    panel_gains = {}
    for panel in (51011, 51017, 51023):
        gains = []
        for d in decisions:
            p, choice = d["parent"], choices["bridge_content"][d["parent"]]
            gains.append(
                100
                * (
                    lookup[p, 1, choice]["panels"][str(panel)]["finetune"]["balanced"]
                    - lookup[p, 1, "uniform"]["panels"][str(panel)]["finetune"]["balanced"]
                )
            )
        panel_gains[str(panel)] = float(np.mean(gains))
    return {
        "role": "reused_history_new_batch_interventions_development",
        "forecast_scores": forecast_scores,
        "forecast_contrasts": forecast_contrasts,
        "policy_scores": policy_scores,
        "policy_contrasts": policy_contrasts,
        "gates": gates,
        "bridge_qualified": all(gates.values()),
        "cross_draw_utility_correlation": correlation(draw0, y),
        "bridge_content_panel_gains_pp": panel_gains,
        "closed_loop_rollouts_launched": False,
        "independent_source_confirmation": False,
    }
