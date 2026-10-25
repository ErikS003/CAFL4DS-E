"""End-to-end smoke tests for the Phase-1 entry-point scripts (audit P1.0 C1).

Each script's Hydra-free core is composed on the network-free synthetic source and run once into a
tmp dir, asserting the expected artifact lands with the right shape. This guards the glue the library
unit tests don't reach — the seed loop + manifest assembly (``run_deploy_harness``), the warm-well
save (``warm_well``), the divergence report (``measure_divergence``), and the canonical comparison
(``run_harness``) — from regressions the promoted one-shot runs would not catch.
"""

from __future__ import annotations

import importlib.util
import json
from pathlib import Path
from types import ModuleType

from hydra import compose, initialize_config_dir
from hydra.core.global_hydra import GlobalHydra
from omegaconf import DictConfig

_REPO_ROOT = Path(__file__).resolve().parents[2]
_CONFIG_DIR = _REPO_ROOT / "cafl4ds" / "configs"

# Fast, network-free synthetic overrides shared by the class-axis scripts (divergence + harness):
# a tiny image, a short horizon, small reservations, probes off (they add wall-clock for no wiring
# value here).
_SYNTHETIC_CLASS = [
    "data=synthetic",
    "img_size=16",
    "eval_every=3",
    "max_per_class=40",
    "max_train_per_class=8",
    "stream.support_per_class=6",
    "stream.query_per_class=6",
    "stream.era_eval_per_class=4",
    "monitor.run_knn=false",
    "monitor.run_linear=false",
]


def _load_script(name: str) -> ModuleType:
    """Import a ``scripts/<name>.py`` entry point as a module (it is a script, not a package)."""
    path = _REPO_ROOT / "scripts" / f"{name}.py"
    spec = importlib.util.spec_from_file_location(f"_script_{name}", path)
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def _compose(config_name: str, overrides: list[str]) -> DictConfig:
    """Compose a script config with fast-smoke overrides (isolated from other hydra-using tests)."""
    GlobalHydra.instance().clear()
    with initialize_config_dir(version_base=None, config_dir=str(_CONFIG_DIR)):
        return compose(config_name=config_name, overrides=overrides)


def test_run_deploy_harness_writes_the_ensemble_corpus(tmp_path: Path) -> None:
    """The deploy script drives a 2-seed ensemble and lands the four-part corpus + per-seed reports."""
    script = _load_script("run_deploy_harness")
    config = _compose(
        "deploy",
        [
            "data=attr_synthetic",
            "img_size=16",
            "eval_every=3",
            "stream.support_per_canary=6",
            "stream.query_per_canary=6",
            "monitor.knn_k=3",
            "seeds=[0,1]",
        ],
    )
    paths = script.run_deploy(config, tmp_path)
    for part in ("health_csv", "segments", "trust", "manifest"):
        assert paths[part].is_file(), part
    manifest = json.loads(paths["manifest"].read_text())
    assert manifest["n_seeds"] == 2
    reports = list((tmp_path / "reports").glob("*.json"))
    assert len(reports) == 2
    # both the Tier-A wiring verdict and the Tier-B sanity block are present per seed
    first = json.loads(reports[0].read_text())
    assert first["validation"]["passed"] is True
    assert "tier_b" in first


def test_warm_well_saves_a_resumable_well(tmp_path: Path) -> None:
    """The warm-well script plateaus on the stationary diet and writes a resumable well."""
    script = _load_script("warm_well")
    well_out = tmp_path / "je_well.pt"
    config = _compose(
        "warm_well",
        [
            "data=attr_synthetic",
            "img_size=16",
            "stream.support_per_canary=6",
            "stream.query_per_canary=6",
            "monitor.knn_k=3",
            "warmup.window=2",
            "warmup.min_steps=2",
            "warmup.max_steps=6",
            "warmup.max_passes=2",
            f"well_out={well_out}",
        ],
    )
    assert script.warm(config) == well_out
    assert well_out.is_file()


def test_measure_divergence_certifies_same_seed(tmp_path: Path) -> None:
    """The divergence script runs its four arms and certifies the same-seed rebuild bit-identically."""
    script = _load_script("measure_divergence")
    report = script._measure_divergence(_compose("divergence", _SYNTHETIC_CLASS), tmp_path)
    assert report["study"] == "P1.0.1"
    assert report["data_identity"]["bit_identical"] is True
    assert set(report["attribution"]) == {"stream_construction", "in_arm_rng_matched", "in_arm_rng_desynced"}


def test_run_harness_writes_the_canonical_comparison(tmp_path: Path) -> None:
    """The P1.0.0 harness co-logs the three arms into the canonical comparison schema."""
    script = _load_script("run_harness")
    comparison = script._run_smoke(_compose("harness", _SYNTHETIC_CLASS), tmp_path)
    assert comparison["schema_version"] == 1
    assert set(comparison["arms"]) == {"live", "pc", "b5"}
    assert "gate" in comparison and "directional" in comparison


def test_synthetic_smoke_profile_passes_collapse_gate(tmp_path: Path) -> None:
    """The advertised CPU smoke must trigger the PC gate, not merely write its schema."""
    script = _load_script("run_harness")
    comparison = script._run_smoke(_compose("harness_smoke", []), tmp_path)
    assert comparison["gate"]["passed"], comparison["gate"]
    assert comparison["finite"]
    assert comparison["directional"]["drift_accumulated"]
