import numpy as np
from scipy.stats import spearmanr

from cafl4ds.early_state_night_diagnostic import (
    ASSAYS,
    FIELDS,
    SIGNALS,
    association_rows,
    components,
    correlation,
    query_summary,
    signal_values,
    transitions,
)


def test_equal_class_weights_are_not_image_frequency_weights():
    y = np.array([0, 0, 0, 1, 0, 1, 1, 1])
    d = np.array([0, 0, 0, 0, 1, 1, 1, 1])
    result = components(np.array([0, 0, 0, 100, 100, 0, 0, 0]), y, d)
    np.testing.assert_array_equal(result, [50, 50, 50, 0, 100, 100, 0])


def test_transition_accounting_and_persistent_query_counts():
    a = np.array([[1, 0, 1], [1, 0, 0]])
    b = np.array([[0, 1, 1], [0, 0, 1]])
    lost, gained = transitions(a, b)
    np.testing.assert_array_equal(gained.astype(int) - lost, b - a)
    result = query_summary([12, 0, 8, 0], [0, 10, 0, 0], 12)
    assert result["persistent_lost_queries"] == result["persistent_gained_queries"] == 1
    assert result["top_decile_positive_net_loss_share"] == 0.6
    assert query_summary([0, 0], [0, 0], 12)["top_decile_positive_net_loss_share"] is None


def test_tied_ranks_and_constant_signals():
    x, y = [1, 1, 4, 7, 7, 7], [6, 3, 2, 1, 1, 0]
    np.testing.assert_allclose(correlation(x, y, True), spearmanr(x, y).statistic, atol=1e-14)
    assert correlation([1, 1, 1], [2, 3, 4]) is None


def test_group_deletions_keep_case_units():
    rows = association_rows(np.arange(12), -np.arange(12), np.repeat([1, 2, 3], 4), np.tile([1, 2, 3, 4], 3))
    assert len(rows) == 8
    assert [r["n_cases"] for r in rows] == [12, 8, 8, 8, 9, 9, 9, 9]
    assert all(abs(r["spearman"] + 1) < 1e-14 for r in rows)


def test_signal_origin_cannot_look_ahead_and_path_is_not_net_distance():
    trace = [{"response": [i, 2 * i], "encoder_step_l2": 0.1 * i} for i in range(1, 34)]
    record = {
        "trace": trace,
        "readings": {"32": {bank: {"change": {f: 0.2 for f in FIELDS}} for bank in ("monitor", "audit")}},
    }
    result = signal_values(record, 32)
    assert result["cumulative_loss"] == 16.5
    assert result["trailing_loss"] == 24.5
    assert result["encoder_path_length"] == sum(0.1 * i for i in range(1, 33))
    assert len([f for f in result if f in SIGNALS]) == 16


def test_synthetic_report_and_saved_reconstruction(tmp_path, monkeypatch):
    from scripts import report_early_state_night_diagnostic as report
    from scripts.run_early_state_night_diagnostic import digest, read, write

    monkeypatch.setattr(report, "ROOT", tmp_path)
    out = tmp_path / "output"
    out.mkdir()
    study = {
        "output": "output",
        "notebook": "notebook40.ipynb",
        "ages": [512, 4096],
        "arms": ["current", "balanced"],
        "checkpoints": [128, 512],
        "minimum_free_mib": 0,
        "max_output_mib": 64,
    }
    # reserve uses its own ROOT, so bypass only that storage assertion in this
    # isolated rendering fixture. No source/model work is performed.
    monkeypatch.setattr(report, "reserve", lambda _: None)
    rng = np.random.default_rng(4)
    gains = rng.normal(size=(3, 2, 2, 3, 4, 2, 12, 7))
    focus = gains[0, 0, 1, :, 3]
    summary = {
        "exploratory": True,
        "new_SSL_updates": 0,
        "new_readout_fits": 0,
        "download_bytes": 0,
        "original_gate_unchanged": True,
        "future4_earlier_half_replay": {
            "means_pp": focus.mean((0, 2)).tolist(),
            "source_means_pp": focus.mean(2).tolist(),
            "leave_source_out_means_pp": [np.delete(focus, i, 0).mean((0, 2)).tolist() for i in range(3)],
            "negative_night_support_draws": [4, 5],
        },
    }
    write(out / "summary.json", summary)
    write(out / "verification.json", {"passed": True, "summary_sha256": digest(out / "summary.json")})
    np.savez_compressed(out / "decomposition.npz", gains=gains)
    signals, correlations = [], []
    for age in (512, 4096):
        for arm in ("current", "balanced"):
            for origin in (16, 128, 512):
                for source in (1, 2, 3):
                    for future in (1, 2, 3, 4):
                        signals.append(
                            {
                                "age": age,
                                "arm": arm,
                                "origin": origin,
                                "source": source,
                                "future": future,
                                **{f: float(rng.normal()) for f in SIGNALS},
                            }
                        )
            for origin, endpoint, assay in ASSAYS:
                for field in SIGNALS:
                    for deletion, group in [("all", 0), ("leave_source", 1)]:
                        correlations.append(
                            {
                                "age": age,
                                "arm": arm,
                                "origin": origin,
                                "endpoint": endpoint,
                                "assay": assay,
                                "component": "night",
                                "signal": field,
                                "deletion": deletion,
                                "group": group,
                                "pearson": 0.3,
                                "spearman": -0.2,
                            }
                        )
    monkeypatch.setattr(report, "read_csv", lambda path: signals if path.name == "signals.csv" else correlations)
    report.report(study)
    assert read(out / "delivery-verification.json")["saved_result_reconstruction_executed"]
    assert len(read(tmp_path / "notebook40.ipynb")["cells"]) == 8
    assert len(list((out / "figures").glob("*.png"))) == 6
