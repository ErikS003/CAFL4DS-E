"""Check complete source isolation in fixed predictive calibration."""

from cafl4ds.anchor_signal_statistics import grouped_predictions


def test_held_group_labels_never_change_its_predictions() -> None:
    """Perturbing a held group's targets leaves its predictions exactly fixed."""
    points = [
        {
            "stream": f"s{g}",
            "updates": g * 10 + k,
            "incoming_loss": g + k / 7,
            "relative_anchor_change": (g - k) / 100,
            "target": g / 5 + k / 3,
        }
        for g in range(6)
        for k in range(3)
    ]
    original = grouped_predictions(points, "target")
    altered = [{**p, "target": p["target"] + (1000 if p["stream"] == "s0" else 0)} for p in points]
    changed = grouped_predictions(altered, "target")
    for name in original:
        assert original[name]["predictions"][:3] == changed[name]["predictions"][:3]
