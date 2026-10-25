"""Unit tests for the P1.0.2 hand-off corpus emitter (``cafl4ds.deploy_corpus``)."""

from __future__ import annotations

import csv
import json
import sys
from pathlib import Path
from typing import Any

import pyarrow.parquet as pq
import pytest

from cafl4ds.deploy import build_deploy_report
from cafl4ds.deploy_corpus import (
    CORPUS_SCHEMA_VERSION,
    event_coordinates,
    health_rows,
    normalize_rows,
    rows_from_report,
    segment_map,
    write_corpus,
)
from cafl4ds.harness import Arm
from cafl4ds.health_trust import CANARY_SIGNALS, DEFAULT_LABEL_FREE_SIGNALS, Backbone


def _health(step: int, era: int, **signals: float) -> dict[str, Any]:
    """A synthetic health record (a monitor checkpoint)."""
    return {"run": "x", "series": "health", "step": float(step), "era": era, "loss": 0.5, **signals}


def _je_signals() -> dict[str, float]:
    """A minimal JE default label-free + canary signal set for a checkpoint."""
    return {
        "rankme_proj": 8.0,
        "mean_feature_var_proj": 0.1,
        "offdiag_cov_proj": 0.01,
        "alignment_proj": 0.5,
        "cosine_drift_proj": 0.2,
        "rankme": 7.0,
        "knn_acc": 0.9,
        "linear_acc": 0.85,
    }


def _contiguous_health() -> list[dict[str, Any]]:
    """A dense health series over three contiguous legs: eras [0,0,1,1,1,2]."""
    eras = [0, 0, 1, 1, 1, 2]
    return [_health(step=i, era=e, **_je_signals()) for i, e in enumerate(eras)]


def _je_arm(role: str = "live") -> Arm:
    """A fabricated JE arm carrying the full default signal set over contiguous legs."""
    return Arm(name=f"{role}_arm", role=role, records=_contiguous_health())


def _report(seed: int = 0, *, pc: Arm | None = None, b5: Arm | None = None) -> dict[str, Any]:
    """A JE deploy report over the given arms (defaults to a single live arm)."""
    expected = [*DEFAULT_LABEL_FREE_SIGNALS[Backbone.JE], *CANARY_SIGNALS]
    return build_deploy_report(
        config_header={"backbone": "je", "seed": seed},
        family=Backbone.JE,
        live=_je_arm("live"),
        expected_signals=expected,
        pc=pc,
        b5=b5,
    )


def test_event_coordinates_marks_transitions_and_leg_offsets() -> None:
    """Transitions open at each era change; steps_since_transition/leg_ordinal reset per leg."""
    coords = event_coordinates(_contiguous_health())
    assert [c["is_transition"] for c in coords] == [True, False, True, False, False, True]
    assert [c["steps_since_transition"] for c in coords] == [0, 1, 0, 1, 2, 0]
    assert [c["leg_ordinal"] for c in coords] == [0, 1, 0, 1, 2, 0]
    assert [c["era_index"] for c in coords] == [0, 0, 1, 1, 1, 2]


def test_health_rows_carry_context_and_every_signal() -> None:
    """Each row stamps backbone/seed/arm + event coords and keeps every instrument signal."""
    rows = health_rows(_contiguous_health(), family=Backbone.JE, seed=3, arm="live", era_names={1: "daytime·rain"})
    assert len(rows) == 6
    first, mid = rows[0], rows[2]
    assert first["backbone"] == "je" and first["seed"] == 3 and first["arm"] == "live"
    assert mid["era_name"] == "daytime·rain" and mid["is_transition"] is True
    assert first["era_name"] == "era0"  # falls back when unnamed
    assert first["rankme_proj"] == 8.0 and first["knn_acc"] == 0.9
    # bookkeeping is not duplicated as a signal column
    assert "series" not in first and "run" not in first


def test_rows_from_report_covers_every_arm_and_stamps_seed() -> None:
    """Every arm in the report contributes rows, each tagged with the seed."""
    report = _report(seed=7, pc=_je_arm("pc"), b5=_je_arm("b5"))
    rows = rows_from_report(report, seed=7)
    assert {r["arm"] for r in rows} == {"live", "pc", "b5"}
    assert all(r["seed"] == 7 for r in rows)
    assert len(rows) == 6 * 3


def test_segment_map_recovers_the_legs() -> None:
    """The leg map collapses the era sequence into contiguous blocks with correct spans."""
    segments = segment_map(_contiguous_health(), era_names={0: "daytime·clear"})
    assert [s["era_index"] for s in segments] == [0, 1, 2]
    assert segments[0] == {
        "era_index": 0,
        "era_name": "daytime·clear",
        "start_step": 0,
        "end_step": 1,
        "start_checkpoint": 0,
        "end_checkpoint": 1,
        "n_checkpoints": 2,
    }
    assert segments[1]["n_checkpoints"] == 3


def test_segment_map_merges_per_era_label_composition() -> None:
    """When composition is supplied, each leg is tagged with its image count + scene/category hists."""
    composition = {
        0: {"n_images": 40, "scene_hist": {"highway": 40}, "category_hist": {"car": 120}},
        1: {"n_images": 12, "scene_hist": {"city street": 12}, "category_hist": {}},
    }
    segments = segment_map(_contiguous_health(), era_names={0: "daytime·clear"}, composition=composition)
    era0 = next(s for s in segments if s["era_index"] == 0)
    assert era0["n_images"] == 40
    assert era0["scene_hist"] == {"highway": 40}
    assert era0["category_hist"] == {"car": 120}
    era1 = next(s for s in segments if s["era_index"] == 1)
    assert era1["n_images"] == 12
    assert "category_hist" not in era1  # empty histograms are not attached


def test_normalize_rows_unifies_heterogeneous_columns() -> None:
    """The column union is context-first + sorted signals; missing cells fill with None."""
    rows = [
        {"backbone": "mae", "seed": 0, "arm": "live", "rankme": 7.0},
        {"backbone": "je", "seed": 0, "arm": "live", "rankme_proj": 8.0, "knn_acc": 0.9},
    ]
    columns, normalized = normalize_rows(rows)
    assert columns[:3] == ["backbone", "seed", "arm"]
    assert columns[-3:] == ["knn_acc", "rankme", "rankme_proj"]
    assert normalized[0]["rankme_proj"] is None and normalized[1]["rankme"] is None


def test_write_corpus_writes_a_readable_ensemble(tmp_path: Path) -> None:
    """The writer emits a CSV + parquet table, the leg map, trust, and a provenance manifest."""
    reports = [(seed, _report(seed=seed)) for seed in (0, 1)]
    era_names = {0: "daytime·clear", 1: "daytime·rain", 2: "night·clear"}
    paths = write_corpus(tmp_path, reports=reports, era_names=era_names, manifest={"git_sha": "abc123"})

    # every artifact landed
    assert {p.name for p in paths.values()} == {
        "health.csv",
        "health.parquet",
        "segments.json",
        "trust.json",
        "manifest.json",
    }

    with paths["health_csv"].open(encoding="utf-8") as handle:
        table = list(csv.DictReader(handle))
    assert len(table) == 6 * 2  # 6 checkpoints × 2 seeds, single live arm
    assert {row["seed"] for row in table} == {"0", "1"}
    assert table[2]["era_name"] == "daytime·rain"

    manifest = json.loads(paths["manifest"].read_text(encoding="utf-8"))
    assert manifest["schema_version"] == CORPUS_SCHEMA_VERSION
    assert manifest["seeds"] == [0, 1] and manifest["backbone_family"] == "je"
    assert manifest["git_sha"] == "abc123"
    assert manifest["tier_a_passed"] == {"0": True, "1": True}
    # Tier-B was not computed for these reports (no canary_chance), so it is absent — no false verdict
    assert "tier_b_passed" not in manifest

    segments = json.loads(paths["segments"].read_text(encoding="utf-8"))["segments"]
    assert [s["era_name"] for s in segments] == ["daytime·clear", "daytime·rain", "night·clear"]


def test_write_corpus_records_tier_b_when_computed(tmp_path: Path) -> None:
    """When the seed reports carry a Tier-B block, the manifest records the per-seed sanity verdict."""
    reports = [
        (
            seed,
            build_deploy_report(
                config_header={"backbone": "je", "seed": seed},
                family=Backbone.JE,
                live=_je_arm("live"),
                expected_signals=[*DEFAULT_LABEL_FREE_SIGNALS[Backbone.JE], *CANARY_SIGNALS],
                canary_chance=1 / 3,
            ),
        )
        for seed in (0, 1)
    ]
    paths = write_corpus(tmp_path, reports=reports)
    manifest = json.loads(paths["manifest"].read_text(encoding="utf-8"))
    assert manifest["tier_b_passed"] == {"0": True, "1": True}


def test_write_corpus_threads_composition_into_segments(tmp_path: Path) -> None:
    """write_corpus passes the per-era composition through to the on-disk leg map."""
    composition = {
        0: {"n_images": 30, "scene_hist": {"highway": 30}, "category_hist": {"car": 90}},
        1: {"n_images": 20, "scene_hist": {"city street": 20}, "category_hist": {"person": 15}},
        2: {"n_images": 10, "scene_hist": {"tunnel": 10}, "category_hist": {}},
    }
    paths = write_corpus(tmp_path, reports=[(0, _report(seed=0))], composition=composition)
    segments = json.loads(paths["segments"].read_text(encoding="utf-8"))["segments"]
    assert segments[0]["n_images"] == 30 and segments[0]["category_hist"] == {"car": 90}
    assert segments[1]["scene_hist"] == {"city street": 20}


def test_write_corpus_parquet_roundtrips(tmp_path: Path) -> None:
    """The parquet table reads back with the same row count and the wide signal columns."""
    paths = write_corpus(tmp_path, reports=[(0, _report(seed=0))])
    table = pq.read_table(paths["health_parquet"])
    assert table.num_rows == 6
    assert "rankme_proj" in table.column_names and "knn_acc" in table.column_names
    assert "steps_since_transition" in table.column_names


def test_write_corpus_skips_parquet_without_pyarrow(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    """When pyarrow is unavailable the parquet twin is skipped, not fatal — CSV + JSON still land.

    This is the deployment case (the Gaudi runtime image ships no pyarrow): a missing optional dep
    must not discard an expensive drive's corpus at the final write.
    """
    monkeypatch.setitem(sys.modules, "pyarrow", None)  # force `import pyarrow` to raise ImportError
    monkeypatch.setitem(sys.modules, "pyarrow.parquet", None)
    paths = write_corpus(tmp_path, reports=[(0, _report(seed=0))])
    assert "health_parquet" not in paths  # gracefully skipped
    assert not (tmp_path / "health.parquet").exists()
    # the portable primary and the rest still land
    assert {p.name for p in paths.values()} == {"health.csv", "segments.json", "trust.json", "manifest.json"}
    assert (tmp_path / "health.csv").exists()


def test_write_corpus_propagates_study_name(tmp_path: Path) -> None:
    """A downstream study can reuse the corpus schema without being mislabeled P1.0.2."""
    paths = write_corpus(tmp_path, reports=[(0, _report())], manifest={"study": "P1.2"})
    manifest = json.loads(paths["manifest"].read_text(encoding="utf-8"))
    segments = json.loads(paths["segments"].read_text(encoding="utf-8"))
    assert manifest["study"] == "P1.2"
    assert segments["study"] == "P1.2"


def test_write_corpus_rejects_empty_reports(tmp_path: Path) -> None:
    """An empty ensemble is a caller error, not a silent empty corpus."""
    with pytest.raises(ValueError, match="at least one"):
        write_corpus(tmp_path, reports=[])
