"""Frozen admission sweeps and explicit allocation controls for long streams."""

import numpy as np

GATES = ("gate_original", "gate_half", "gate_zero")


def policy_specs(settings, original_threshold):
    """Declare every comparison before observing continuation outcomes."""
    if not np.isfinite(original_threshold) or original_threshold >= 0:
        raise ValueError("This sweep requires the original negative delta-J threshold")
    specs = {
        name: dict(kind="gate", threshold=float(original_threshold * factor))
        for name, factor in zip(GATES, settings["threshold_factors"], strict=True)
    }
    for p in settings["independent_probabilities"]:
        if not 0 < p < 1:
            raise ValueError("Random admission probabilities must be interior")
        specs[f"independent_{int(p * 100):02d}"] = dict(kind="independent", probability=float(p))
    specs.update(full=dict(kind="full"), hold=dict(kind="hold"))
    for gate in GATES:
        for kind in ("global", "clip"):
            specs[f"{kind}_{gate}"] = dict(kind=kind, gate=gate, retrospective=True)
    return specs


def independent_schedule(length, probability, seed):
    """An arrival-independent Bernoulli policy with no gate or clip inputs."""
    return np.random.default_rng(seed).random(length) < probability


def global_schedule(accepted, seed):
    """Match only the total realized budget, without copying clip allocation."""
    accepted = np.asarray(accepted, dtype=bool)
    if accepted.ndim != 1:
        raise ValueError("Require a one-dimensional admission trace")
    result = np.zeros(len(accepted), dtype=bool)
    chosen = np.random.default_rng(seed).choice(len(accepted), int(accepted.sum()), replace=False)
    result[chosen] = True
    return result


def finite_metrics(metrics):
    """Refuse invalid scientific observations rather than writing NaN JSON."""
    if not metrics or not np.isfinite(list(metrics.values())).all():
        raise ValueError("All recorded metrics must be finite")
