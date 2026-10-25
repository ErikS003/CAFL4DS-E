"""Paired summaries must retain exact parents, repeats and support disagreements."""

from pathlib import Path

import pandas as pd
import pytest

from scripts.run_linear_probe_reliability import METRICS, summarize


def test_summary_gain_direction_and_agreement(tmp_path: Path) -> None:
    """Check a known reversal and tie instead of assuming agreement means learning."""
    rows = []
    for panel in ("support1", "support2", "support3", "pooled"):
        base = {"panel": panel, "role": "holdout", "recipe": "test", "horizon": 12}
        for state, value in (
            ("warm", 0.6),
            ("real_77021", 0.4),
            ("real_77027", 0.4),
            ("shuffled_77021", 0.5),
            ("shuffled_77027", 0.5),
        ):
            rows.append(
                {
                    **base,
                    "kind": "warm" if state == "warm" else "control",
                    "state": state,
                    "repeat": -1,
                    "alpha": -1,
                    **dict.fromkeys(METRICS, value),
                }
            )
        rows.append(
            {**base, "kind": "parent", "state": "parent", "repeat": -1, "alpha": -1, **dict.fromkeys(METRICS, 0.7)}
        )
        for repeat in (0, 1):
            gain = {"support1": 0.1, "support2": -0.1, "support3": 0.0, "pooled": 0.1 if repeat == 0 else -0.1}[panel]
            for alpha in (0.0, 0.5, 1.0):
                value = 0.5 if alpha == 1 else 0.5 - gain
                rows.append(
                    {
                        **base,
                        "kind": "branch",
                        "state": "parent",
                        "repeat": repeat,
                        "alpha": alpha,
                        **dict.fromkeys(METRICS, value),
                    }
                )
    summarize(pd.DataFrame(rows), tmp_path)
    control = pd.read_csv(tmp_path / "control_gains.csv")
    assert control.warm_gain_linear_error_balanced.tolist() == pytest.approx([0.2] * 8)
    assert control.shuffled_gain_linear_error_balanced.tolist() == pytest.approx([0.1] * 8)
    variability = pd.read_csv(tmp_path / "variability.csv")
    assert (variability.linear_error_support_sign_agreement == 0).all()
    assert (variability.linear_error_support_tie_cells == 1).all()
    assert (variability.linear_error_repeat_sign_agreement == 0).all()
    assert variability.linear_error_mean_abs_repeat_difference.tolist() == pytest.approx([0.2, 0.2])
    learning = pd.read_csv(tmp_path / "learning_summary.csv")
    assert learning.linear_error_balanced.tolist() == pytest.approx([0.2, 0.2, 0.2])
