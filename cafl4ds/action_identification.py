"""Declared action sequences and compact pre-decision response features."""

from __future__ import annotations

SCHEDULES = {
    "constant_0": (0.0, 0.0),
    "constant_half": (0.5, 0.5),
    "constant_1": (1.0, 1.0),
    "zero_to_one": (0.0, 1.0),
    "one_to_zero": (1.0, 0.0),
    "half_to_one": (0.5, 1.0),
    "one_to_half": (1.0, 0.5),
}
STATE_FEATURES = (
    "health_reconstruction",
    "health_cka_drift",
    "health_rankme",
    "health_occupancy_tv",
    "arrival_reconstruction",
    "arrival_coverage",
    "arrival_embedding_drift",
    "arrival_cluster_tv",
    "previous_alpha",
)
INTERACTIONS = ("health_reconstruction", "health_cka_drift", "health_occupancy_tv", "arrival_coverage")


def action_at(schedule: str, offset: int) -> float:
    """Switch precisely after six updates; reject undefined exposure positions."""
    if not 0 <= offset < 12:
        raise ValueError("Action schedule has exactly twelve updates")
    return SCHEDULES[schedule][int(offset >= 6)]


def response_features(schedule: str, state: dict[str, float], health: bool) -> dict[str, float]:
    """Use requested action and whitelisted pre-decision state; exclude outcome/post-action fields."""
    first, last = SCHEDULES[schedule]
    features = {
        "action_first": first,
        "action_last": last,
        "action_mean": (first + last) / 2,
        "action_change": last - first,
    }
    if health:
        features.update({key: state[key] for key in STATE_FEATURES})
        for key in INTERACTIONS:
            features[f"{key}_first"] = state[key] * first
            features[f"{key}_last"] = state[key] * last
    return features
