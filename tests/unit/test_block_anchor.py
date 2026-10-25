"""Block opportunity, staged precision and separate reference protection."""

import copy

import numpy as np
import pytest

from cafl4ds.block_anchor import BlockAnchorStudy


def measured(mean=(1, 1, 1), lower=(0.5, 0.5, 0.5), upper=(1.5, 1.5, 1.5), width=0.1):
    return {
        "mean_gain_pp": list(mean),
        **{
            k: {"lower": list(lower), "upper": list(upper), "half_width": [width] * 3}
            for k in ("support", "query", "crossed")
        },
    }


def test_original_floor_does_not_automatically_protect_acquired_gain():
    study = BlockAnchorStudy()
    benefit = measured(lower=(0.5, -1.2, 2.2), upper=(1, -1.1, 3.1))
    result = study.shadow(benefit, measured(), {"rank": 1, "variance": 1})
    assert result["warm_shadow_accept"]
    assert not result["warm_and_acquired_shadow_accept"]
    assert result["acquired_budget_status"][0] == "resolved_violation"


def test_point_harm_with_unresolved_range_is_unknown():
    result = BlockAnchorStudy().budgets(measured(mean=(-2, -2, -2), lower=(-3, -3, -3), upper=(0, 0, 0)))
    assert result == ["unknown", "unknown"]


def test_other_domain_gain_cannot_purchase_warm_budget_violation():
    study = BlockAnchorStudy()
    result = study.shadow(measured(), measured(lower=(3, -2, 8), upper=(4, -1.5, 9)), {"rank": 1, "variance": 1})
    assert not result["warm_shadow_accept"]
    assert result["warm_budget_status"][0] == "resolved_violation"


def test_interim_precision_never_promotes_final_measurement():
    study = BlockAnchorStudy()
    rows = []
    for source in ("a", "b"):
        for horizon in (8, 32):
            m = measured()
            rows.append(
                {
                    "source": source,
                    "horizon": horizon,
                    "measurement": m,
                    "shadow": study.shadow(m, m, {"rank": 1, "variance": 1}),
                }
            )
    anchor = [{"measurement": measured()}]
    result = study.summarize(rows, anchor, 64, 256)
    assert result["interim_query_gate"]
    assert not result["final_precision_pass"]
    assert not result["measurement_ready"]
    assert result["source_groups"] == 2
    assert not result["controller_activated"]


def test_anchor_query_failure_cannot_hide_inside_precise_block_population():
    study = BlockAnchorStudy()
    m = measured()
    rows = [{"source": "a", "horizon": 8, "measurement": m, "shadow": study.shadow(m, m, {"rank": 1, "variance": 1})}]
    wide = copy.deepcopy(m)
    wide["query"]["half_width"] = [2, 2, 2]
    result = study.summarize(rows, [{"measurement": wide}], 64, 256)
    assert result["pooled_block_precision"]["query"]["passed"]
    assert not result["interim_query_gate"]


def test_invalid_budgets_and_ranges_fail():
    with pytest.raises(ValueError):
        BlockAnchorStudy(domain_budget_pp=0)
    with pytest.raises(ValueError):
        BlockAnchorStudy().budgets(measured(lower=(np.nan, np.nan, np.nan)))
