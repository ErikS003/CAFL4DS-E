import numpy as np
import pytest

from cafl4ds.early_state_adaptation import crossed_samples, diet, screen
from cafl4ds.target_alignment_measurement import resampling_counts
from scripts.prepare_early_state_adaptation import identities


def fixture():
    labels = np.tile([0, 1], 8)
    domains = np.repeat([0, 1], 8)
    p, q = resampling_counts(labels, domains, 12, 100, 193039)
    return labels, domains, p, q


def test_full_constant_positive_screen():
    y, d, p, q = fixture()
    result = screen(np.full((3, 4, 12, 16), 2.0), y, d, p, q)
    assert result["passes"]
    np.testing.assert_allclose(result["family_two_conditional_bounds_pp"], [[2, 2]] * 3)


def test_domain_guard_can_fail_despite_positive_balance():
    y, d, p, q = fixture()
    effect = np.where(d == 0, 6.0, -2.0) * np.ones((3, 4, 12, 16))
    result = screen(effect, y, d, p, q)
    assert result["mean_pp"][0] == 2
    assert not result["passes"]
    assert not result["requirements"]["every_future_domain_above_minus_one"]


def test_common_crossed_samples_match_explicit_resampling():
    y, d, p, q = fixture()
    effect = np.random.default_rng(1).normal(size=(3, 4, 12, 16))
    actual = crossed_samples(effect, y, d, p, q)
    expected = []
    for support, query in zip(p, q):
        means = []
        for domain in (0, 1):
            cell = []
            for label in (0, 1):
                selected = (d == domain) & (y == label)
                raw = effect[..., selected].mean((0, 1))
                cell.append((raw * support[:, None] * query[selected]).sum() / (12 * selected.sum()))
            means.append(np.mean(cell))
        expected.append([np.mean(means), *means])
    np.testing.assert_allclose(actual, expected, atol=1e-14)


def test_zero_effect_fails_and_invalid_resampling_stops():
    y, d, p, q = fixture()
    assert not screen(np.zeros((3, 4, 12, 16)), y, d, p, q)["passes"]
    with pytest.raises(ValueError):
        crossed_samples(np.zeros((3, 4, 12, 16)), y, d, p * 0, q)


def test_schedule_uses_same_current_subset_across_ages():
    early, late = list(range(256)), list(range(2000, 2256))
    a, b = diet(early, "balanced", 6), diet(late, "balanced", 6)
    for x, z in zip(a, b):
        assert x["current"] == z["current"] and len(x["past"]) == 16
        assert x["past"] == [v - 2000 for v in z["past"]]
    assert all(len(r["current"]) == 32 and not r["past"] for r in diet(early, "current", 6))


def test_nested_manifest_identity_inventory():
    assert identities(
        {
            "records": [{"path": "C:/data/abcdef12-12345678.jpg"}],
            "warm_clip_ids": ["99887766-abcdef01"],
            "unrelated": "other",
        }
    ) == {"abcdef12-12345678", "99887766-abcdef01"}


def test_synthetic_full_analysis_and_notebook_delivery(tmp_path, monkeypatch):
    from omegaconf import OmegaConf
    from scripts.analyze_early_state_adaptation import build_summary
    from scripts import report_early_state_adaptation as report
    from scripts.prepare_early_state_adaptation import write, digest, read

    config = tmp_path / "config.yaml"
    config.write_text("output: output\nnotebook: study.ipynb\n")
    monkeypatch.setattr(report, "ROOT", tmp_path)
    monkeypatch.setattr(report, "CONFIG", config)
    out = tmp_path / "output"
    y, d, _, _ = fixture()
    correct = np.random.default_rng(2).integers(0, 2, (3, 2, 3, 3, 4, 12, 16), dtype=np.uint8)
    summary, _, _ = build_summary(correct, y, d, OmegaConf.create({"bootstrap_draws": 20, "bootstrap_seed": 3}))
    summary["intermediate_effects_pp"] = np.zeros((3, 2, 2, 3, 4, 12, 3)).tolist()
    parents, health = [], []
    for source in (1, 2, 3):
        for age in (512, 4096):
            for future in (1, 2, 3, 4):
                pid = f"b{source}-a{age}-f{future}"
                parents.append({"id": pid, "age": age})
                for arm in ("current", "balanced"):
                    write(
                        out / "branches" / pid / arm / "record.json",
                        {"trace": [{"encoder_step_l2": 0.01, "response": [0.2, 0.4]}] * 512},
                    )
                    health.append(
                        {
                            "bank": "monitor",
                            "step": 512,
                            "age": age,
                            "arm": arm,
                            "health": [0.1] * 5,
                            "mean_encoder_step_l2": 0.01,
                            "capability_change_pp": [0.2, 0.1, 0.3],
                        }
                    )
    summary["health_descriptive"] = health
    write(out / "plan.json", {"parents": parents})
    write(out / "summary.json", summary)
    write(out / "verification.json", {"passed": True, "summary_sha256": digest(out / "summary.json")})
    report.report()
    delivery = read(out / "delivery-verification.json")
    assert delivery["passed"] and delivery["saved_result_reconstruction_executed"]
    assert len(list((out / "figures").glob("*.png"))) == 6
    assert len(read(tmp_path / "study.ipynb")["cells"]) == 8
