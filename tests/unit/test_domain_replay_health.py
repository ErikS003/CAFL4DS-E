"""Domain contrasts use metadata only and controls retain their original forecasts."""

import numpy as np
import torch

from cafl4ds.domain_replay_health import SIGNALS, domain_sensors
from cafl4ds.domain_replay_health import tune as domain_tune
from cafl4ds.replay_response import tune as pooled_tune


def test_swapping_monitor_domains_reverses_contrasts() -> None:
    """The same image measurements reverse sign if day/night metadata is exchanged."""
    generator = torch.Generator().manual_seed(91)
    warm = torch.randn(96, 8, generator=generator)
    z = warm + 0.2 * torch.randn(96, 8, generator=generator)
    reconstruction = torch.linspace(0.1, 0.9, 96)
    groups = torch.cat((torch.zeros(48, dtype=torch.long), torch.ones(48, dtype=torch.long)))
    before = z.clone()
    observed = domain_sensors(z, warm, reconstruction, groups)
    swapped = domain_sensors(z, warm, reconstruction, 1 - groups)
    for signal in SIGNALS:
        np.testing.assert_allclose(observed[f"night_minus_day_{signal}"], -swapped[f"night_minus_day_{signal}"])
        np.testing.assert_allclose(observed[f"day_health_{signal}"], swapped[f"night_health_{signal}"])
    assert torch.equal(before, z)


def test_original_recipe_fits_remain_exact() -> None:
    """Extending the sensor registry must not alter the existing action/history controls."""
    rows = [
        {
            "stream": stream,
            "action": action,
            "features": {"step": float(step), "history_replay_fraction": 0.5},
            "target": (action - 16) * 0.0001 * step,
        }
        for stream in ("s1", "s2")
        for step in (4, 12)
        for action in (8, 24)
    ]
    original, old_cv = pooled_tune(rows, "action_history")
    extended, new_cv = domain_tune(rows, "action_history")
    np.testing.assert_array_equal(original.weight, extended.weight)
    np.testing.assert_array_equal(original.predict(rows), extended.predict(rows))
    assert old_cv == new_cv
