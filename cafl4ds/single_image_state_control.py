"""Frozen image-versus-state admission ablation, using previously fitted models."""

from __future__ import annotations

from cafl4ds.single_image_filter import features
from cafl4ds.single_image_signal import optimizer_features

MAIN = ("image", "state")
ARMS = (*MAIN, "random_image", "random_state", "full", "hold")
DESIGNS = ("state", "base")
TARGETS = ("accuracy",)


def state_vector(model, optimizer, observations, origin, settings):
    """No incoming image, embedding, reconstruction or query label is accessed."""
    return features(
        observations[-1][0],
        observations[-1][0] - observations[-2][0],
        optimizer_features(model, optimizer),
        origin,
        settings["frames_per_clip"],
    )
