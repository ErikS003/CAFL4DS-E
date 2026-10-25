"""Source-level inference for a frozen fresh-history health guard comparison."""

from __future__ import annotations

import numpy as np

from cafl4ds.collapse_risk_regression import interval
from cafl4ds.functional_retention import plain


def validate_sources(intake, excluded):
    streams, clips = intake["streams"], intake["clips"]
    if len(streams) != 6 or len(clips) != 144:
        raise ValueError("Six complete new histories required")
    lookup = {c["clip_id"]: c for c in clips}
    if len(lookup) != 144 or set(lookup).intersection(excluded):
        raise ValueError("Fresh clips must be unique and excluded from previous sources")
    used = []
    for stream in streams:
        if len(stream["periods"]) != 3:
            raise ValueError("Three visits required")
        for visit, identifiers in enumerate(stream["periods"]):
            if len(identifiers) != 8:
                raise ValueError("Eight clips per visit required")
            for position, identity in enumerate(identifiers):
                attrs = lookup[identity]["keyframe_attributes"]
                light = "night" if visit == 1 else "daytime"
                scene = "highway" if position % 2 else "city street"
                if attrs["timeofday"] != light or attrs["scene"] != scene:
                    raise ValueError("Frozen enrollment strata changed")
            used.extend(identifiers)
    if len(set(used)) != 144 or set(used) != set(lookup):
        raise ValueError("Histories share or omit a clip")


def age_cutoff(age, config):
    return config.age_early_cutoff if age < config.age_boundary else config.age_mature_cutoff


def summarize(cases, config):
    sources = sorted({r["source"] for r in cases})
    if len(sources) != 6 or len(cases) != 54:
        raise ValueError("Complete six-history, three-context, three-seed population required")
    contexts = tuple(config.parents)
    frequencies = {}
    for context in contexts:
        same = [r for r in cases if r["context"] == context]
        frequencies[context] = {
            str(step): sum(r["policies"]["rollback"]["committed_updates"] == step for r in same) / len(same)
            for step in config.landmarks
        }
    gains, outcomes = {}, {}
    for row in cases:
        identity = f"{row['source']}/{row['seed']}/{row['context']}"
        baseline = np.asarray(row["checkpoints"]["0"]["panel_accuracy_pp"])
        values = {
            mode: np.asarray(row["checkpoints"][str(policy["committed_updates"])]["panel_accuracy_pp"])
            for mode, policy in row["policies"].items()
        }
        values["hold"] = baseline
        values["matched_random"] = sum(
            frequencies[row["context"]][str(step)] * np.asarray(row["checkpoints"][str(step)]["panel_accuracy_pp"])
            for step in config.landmarks
        )
        outcomes[identity] = values
        gains[identity] = {mode: (value - baseline).mean(0) for mode, value in values.items()}
    policies = tuple(next(iter(gains.values())))
    source_means = {
        mode: {
            source: np.mean([v[mode] for key, v in gains.items() if key.startswith(source + "/")], axis=0)
            for source in sources
        }
        for mode in policies
    }
    comparisons = {}
    for mode in ("full", "fixed8", "age", "matched_random"):
        comparisons[mode] = {
            domain: interval(
                [source_means["rollback"][s][j] - source_means[mode][s][j] for s in sources],
                family=config.primary_family,
            )
            for j, domain in enumerate(("balanced", "day", "night"))
        }
    population = []
    for source in sources:
        for context in contexts:
            values = [
                gains[f"{r['source']}/{r['seed']}/{r['context']}"]["full"]
                for r in cases
                if r["source"] == source and r["context"] == context
            ]
            population.append(np.mean(values, axis=0))
    observable = [
        s
        for s in sources
        if any(
            any(o["decision"]["active"])
            for r in cases
            if r["source"] == s
            for o in r["policies"]["rollback"]["observations"]
        )
    ]
    positive = sum(v[0] >= 0.5 and min(v[1:]) >= -1 for v in population)
    negative = sum(-min(v[1:]) > 1 for v in population)
    gates = {
        "two_sided_population": bool(positive >= 2 and negative >= 2),
        "observable_sources": len(observable) >= 4,
        "balanced_efficacy": all(
            v["balanced"]["mean"] >= config.minimum_gain_pp and v["balanced"]["low"] > 0 for v in comparisons.values()
        ),
        "retention_and_acquisition": all(
            v[d]["low"] >= -config.domain_budget_pp for v in comparisons.values() for d in ("day", "night")
        ),
        "source_stability": sum(source_means["rollback"][s][0] - source_means["full"][s][0] >= 0 for s in sources) >= 5,
    }
    by_context = {
        context: {
            mode: np.mean(
                [gains[f"{r['source']}/{r['seed']}/{r['context']}"][mode] for r in cases if r["context"] == context],
                axis=0,
            )
            for mode in policies
        }
        for context in contexts
    }
    budget_checks = {
        context: {
            "guard_mean_committed": np.mean(
                [r["policies"]["rollback"]["committed_updates"] for r in cases if r["context"] == context]
            ),
            "random_mean_committed": sum(float(step) * weight for step, weight in frequencies[context].items()),
        }
        for context in contexts
    }
    return plain(
        {
            "gates": gates,
            "fresh_source_lead": all(gates.values()),
            "deployment_qualified": False,
            "source_means_pp": source_means,
            "mean_gains_pp": {mode: np.mean(list(v.values()), axis=0) for mode, v in source_means.items()},
            "primary_comparisons": comparisons,
            "by_context_mean_gains_pp": by_context,
            "population": {
                "beneficial_parents": positive,
                "harmful_parents": negative,
                "observable_sources": observable,
            },
            "random_cutoff_frequencies": frequencies,
            "matched_committed_budget": budget_checks,
            "scope": "fresh_clip_histories_reused_reference_and_validation",
            "capture_sessions_known": False,
        }
    )
