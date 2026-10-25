"""Timing and population tests for descriptive health/harm auditing."""

import numpy as np
import pytest

from cafl4ds.health_harm_audit import HealthHarmAudit, READERS, agreement, confusion, correlation


def path(health=None, gaps=None):
    h = np.array(health if health is not None else [[1, 0.6], [1, 0.61], [1, 0.68], [1, 0.72], [1, 0.74]])
    g = np.array(gaps if gaps is not None else [[0.1, 0.4]] * 5)
    scores = {reader: np.tile(np.array([60, 60, 59, 58, 57])[:, None, None], (1, 3, 3)) for reader in READERS}
    return HealthHarmAudit().episode(
        {"source": "s0", "context": "day_end", "seed": 1, "parent_age": 128}, [0, 4, 8, 16, 32], h, g, scores
    )


def test_newly_readable_task_cannot_alarm_at_activation():
    rows, episode = path(gaps=[[0.1, 0.1], [0.1, 0.1], [0.1, 0.4], [0.1, 0.4], [0.1, 0.4]])
    assert not rows[1]["eligible_prior"][1]
    assert not rows[1]["first_alarm"]
    assert episode["first_alarm_step"] == 32


def test_post_stop_blocks_are_not_live_false_negatives():
    rows, episode = path()
    assert episode["first_alarm_step"] == 8
    assert [r["actual_prefix"] for r in rows] == [True, True, False, False]
    summary = HealthHarmAudit().summarize(rows, [episode])
    assert summary["populations"]["full_path_blocks"] == 4
    assert summary["populations"]["actual_prefix_blocks"] == 2


def test_latching_persists_after_readability_is_lost():
    rows, episode = path(gaps=[[0.1, 0.4], [0.1, 0.01], [0.1, 0.01], [0.1, 0.01], [0.1, 0.01]])
    assert rows[1]["eligible_prior"][1]
    assert episode["first_alarm_step"] == 8


def test_accuracy_gain_and_late_warning_are_distinct_endpoints():
    rows, episode = path(health=[[1, 0.6], [1, 0.6], [1, 0.6], [1, 0.6], [1, 0.7]])
    assert episode["first_alarm_step"] == 32
    assert rows[-1]["remaining_updates"] == 0
    assert rows[-1]["subsequent_damage_pp"]["finetune"] == [0, 0, 0]
    assert episode["rollback_advantage_pp"]["finetune"] == [1, 1, 1]


def test_unreadable_harm_is_counted_in_all_population_recall():
    result = confusion([False, True, False], [True, True, True], [False, True, True])
    assert result["recall"] == pytest.approx(1 / 3)
    assert result["missed_unreadable"] == result["missed_eligible"] == 1


def test_accumulated_health_uses_the_same_prior_checkpoint_as_accuracy():
    rows, episode = path(health=[[1, 0.6], [1, 0.58], [1, 0.61], [1, 0.64], [1, 0.65]])
    assert episode["first_alarm_step"] == 16
    assert rows[2]["health_increment"][1] < 0.05
    assert rows[2]["best_health_step"][1] == 4
    assert rows[2]["immediate_damage_pp"]["finetune"][2] == 1
    assert rows[2]["retention_damage_pp"]["finetune"][1] == 2
    assert rows[2]["retention_damage_pp"]["finetune"][0] is None


def test_exact_material_boundary_does_not_create_rounding_events():
    rows, episode = path()
    rows[0]["immediate_damage_pp"]["finetune"][2] = 1 + 5e-9
    rows[1]["immediate_damage_pp"]["finetune"][2] = 1 + 2e-8
    result = HealthHarmAudit().summarize(rows, [episode])
    assert result["immediate_confusion"]["material"]["night"]["all_prefix"]["tp"] == 1
    assert result["immediate_confusion"]["material"]["night"]["all_prefix"]["fn"] == 0


def test_panel_disagreement_is_not_unanimous_protection():
    assert agreement([1, 1, -0.2]) == "mixed"
    assert agreement([1, 2, 3]) == "all_improved"
    assert agreement([0, 0, 0]) == "all_zero"
    assert correlation([1, 1, 1], [1, 2, 3])["pearson"] is None


def test_source_means_average_correlated_repeats_before_concentration():
    rows, episode = path()
    other_rows, other = path()
    other["source"] = "s1"
    for row in other_rows:
        row["source"] = "s1"
    other["rollback_advantage_pp"]["finetune"] = [0, 0, 0]
    result = HealthHarmAudit().summarize(rows * 3 + other_rows, [episode] * 3 + [other])
    assert result["populations"]["histories"] == 2
    assert result["mean_rollback_advantage_pp"] == [1.5, 1.5, 1.5]
    assert result["leave_one_source_out_descriptive_means_pp"]["s0"] == [0, 0, 0]
