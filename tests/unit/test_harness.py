"""Unit tests for the Phase-1 multi-arm harness (P1.0.0).

Split into fast pure-logic tests over crafted records (the gate, the schema, the finite/leak
checks, the alignment grid) and a light end-to-end that drives the real arms on the network-free
synthetic source — the live arm, the frozen B5 floor stamped onto the live grid, and the
manufactured-collapse PC — asserting the smoke's load-bearing wiring: the three arms co-log into
one canonical artifact, live and B5 share the step grid, and the anti-collapse-off PC leaves its
collapse fingerprint (loss rides toward -1, projector RankMe craters).
"""

from __future__ import annotations

import copy
import math
from collections.abc import Iterable
from pathlib import Path
from typing import Any

import torch

from cafl4ds import harness
from cafl4ds.data.sources import SyntheticSource
from cafl4ds.data.streams import EraStream, EvalSets, StreamBatch
from cafl4ds.filters.accept_all import AcceptAll
from cafl4ds.models.vit import TinyViTEncoder
from cafl4ds.monitor import HealthMonitor
from cafl4ds.ssl.factory import build_mae, build_simsiam


def _health(step: int, era: int, **metrics: float) -> dict[str, Any]:
    """A crafted health record (as :class:`~cafl4ds.run_log.RunLogger` writes them)."""
    return {"run": "x", "series": "health", "step": float(step), "era": era, "loss": None, **metrics}


def _loss(step: int, era: int, loss: float, *, finite: bool = True) -> dict[str, Any]:
    """A crafted loss record."""
    return {"run": "x", "series": "loss", "step": step, "era": era, "loss": loss, "finite": finite}


# --------------------------------------------------------------------------- pure-logic tests


def test_arm_properties_split_series_and_find_floor() -> None:
    """An arm exposes its health/loss series, its loss floor, and its divergence flag."""
    arm = harness.Arm(
        name="a",
        role="live",
        records=[_loss(0, 0, 0.5), _health(0, 0, rankme=5.0), _loss(1, 0, -0.2), _health(1, 0, rankme=4.0)],
    )
    assert len(arm.health) == 2
    assert len(arm.loss_records) == 2
    assert arm.loss_floor == -0.2
    assert arm.diverged is False


def test_arm_diverged_flags_non_finite_step() -> None:
    """A step logged ``finite=False`` (the P0.4.0 fingerprint) sets the arm's divergence flag."""
    arm = harness.Arm(name="a", role="live", records=[_loss(0, 0, 0.5), _loss(1, 0, float("nan"), finite=False)])
    assert arm.diverged is True


def test_health_grid_is_step_era_pairs() -> None:
    """The alignment grid is the arm's ``(step, era)`` health checkpoints."""
    arm = harness.Arm(name="a", role="live", records=[_health(0, 0, rankme=5.0), _health(3, 1, rankme=4.0)])
    assert harness.health_grid(arm) == [(0, 0), (3, 1)]


def test_all_finite_true_and_catches_nan() -> None:
    """``all_finite`` passes clean arms and catches a non-finite instrument value."""
    clean = harness.Arm(name="a", role="live", records=[_health(0, 0, rankme=5.0, cka_drift=0.1)])
    dirty = harness.Arm(name="b", role="pc", records=[_health(0, 0, rankme=float("inf"))])
    assert harness.all_finite(clean) is True
    assert harness.all_finite(clean, dirty) is False


def test_collapse_gate_fires_on_manufactured_fingerprint() -> None:
    """The gate passes iff the PC rode its loss floor down AND collapsed its projector RankMe."""
    fired = harness.Arm(
        name="pc",
        role="pc",
        records=[
            _loss(0, 0, 0.1),
            _health(0, 0, rankme_proj=10.0),
            _loss(1, 0, -0.7),
            _health(1, 0, rankme_proj=3.0),
        ],
    )
    gate = harness.collapse_gate(fired, pc_loss_floor_max=-0.3, pc_rankme_drop_frac=0.6)
    assert gate["passed"] is True
    assert gate["checks"] == {"loss_floor_fired": True, "proj_rankme_collapsed": True}
    assert math.isclose(gate["reported"]["pc_rankme_proj_drop_frac"], 0.3)


def test_collapse_gate_does_not_fire_without_both_conditions() -> None:
    """A negative loss floor alone (no RankMe collapse) does not pass the gate."""
    loss_only = harness.Arm(
        name="pc",
        role="pc",
        records=[_loss(0, 0, -0.7), _health(0, 0, rankme_proj=10.0), _health(1, 0, rankme_proj=9.8)],
    )
    gate = harness.collapse_gate(loss_only, pc_loss_floor_max=-0.3, pc_rankme_drop_frac=0.6)
    assert gate["passed"] is False
    assert gate["checks"] == {"loss_floor_fired": True, "proj_rankme_collapsed": False}


def test_directional_report_reads_movement() -> None:
    """The directional report flags a responding RankMe and accumulating drift."""
    live = harness.Arm(
        name="live",
        role="live",
        records=[_health(0, 0, rankme=25.0, cosine_drift=0.0), _health(3, 1, rankme=7.0, cosine_drift=0.4)],
    )
    report = harness.directional_report(live)
    assert report["rankme_responded"] is True
    assert report["drift_accumulated"] is True


def test_build_comparison_has_canonical_shape() -> None:
    """The artifact carries the schema version, header, gate, and per-role arm health series."""
    live = harness.Arm(name="live", role="live", records=[_health(0, 0, rankme=5.0)])
    pc = harness.Arm(name="pc", role="pc", records=[_health(0, 0, rankme_proj=3.0)])
    b5 = harness.Arm(name="b5", role="b5", records=[_health(0, 0, rankme=5.0)])
    comparison = harness.build_comparison(
        config_header={"C": "mae", "seed": 0},
        gate={"passed": True},
        arms=[live, pc, b5],
        directional={"rankme_responded": True},
        extensions={"finite": True},
    )
    assert comparison["schema_version"] == harness.SCHEMA_VERSION
    assert set(comparison["arms"]) == {"live", "pc", "b5"}
    assert comparison["arms"]["pc"]["run_name"] == "pc"
    assert comparison["finite"] is True


def test_leak_report_flat_vs_growing() -> None:
    """A settled-then-flat RSS trace passes; monotone growth past the budget reads as a leak."""
    flat = harness.leak_report([500.0, 440.0, 441.0, 442.0], growth_frac_max=0.10)
    grew = harness.leak_report([500.0, 440.0, 470.0, 520.0], growth_frac_max=0.10)
    assert flat["flat"] is True
    assert grew["flat"] is False


# ----------------------------------------------------------- P1.0.1 arm-alignment divergence


def _batch(step: int, era: int, fill: float) -> StreamBatch:
    """A crafted stream batch of constant-``fill`` images (for the data-identity checks)."""
    return StreamBatch(images=torch.full((2, 3, 4, 4), fill), era=era, step=step)


def test_data_identity_certifies_bit_identical_rebuild() -> None:
    """Two identical batch sequences (the oracle vs a matching rebuild) read bit-identical."""
    oracle = [_batch(0, 0, 0.5), _batch(1, 0, 0.7), _batch(2, 1, 0.2)]
    rebuilt = [_batch(0, 0, 0.5), _batch(1, 0, 0.7), _batch(2, 1, 0.2)]
    report = harness.data_identity(rebuilt, oracle)
    assert report["bit_identical"] is True
    assert report["max_abs_image_diff"] == 0.0
    assert report["num_steps"] == 3
    assert report["era_step_aligned"] is True


def test_data_identity_catches_a_pixel_disagreement() -> None:
    """A single differing pixel value breaks identity and is reported as the max abs gap."""
    oracle = [_batch(0, 0, 0.5), _batch(1, 0, 0.7)]
    rebuilt = [_batch(0, 0, 0.5), _batch(1, 0, 0.9)]  # step 1 differs by 0.2
    report = harness.data_identity(rebuilt, oracle)
    assert report["bit_identical"] is False
    assert math.isclose(report["max_abs_image_diff"], 0.2, rel_tol=1e-6)


def test_data_identity_catches_era_step_misalignment() -> None:
    """A mismatched ``(era, step)`` tag flags the sequences as not aligned (never bit-identical)."""
    oracle = [_batch(0, 0, 0.5), _batch(1, 0, 0.5)]
    rebuilt = [_batch(0, 0, 0.5), _batch(1, 1, 0.5)]  # era differs at step 1
    report = harness.data_identity(rebuilt, oracle)
    assert report["era_step_aligned"] is False
    assert report["bit_identical"] is False


def test_trajectory_divergence_zero_for_identical_arms() -> None:
    """Two arms with an identical health series read zero divergence, bit-identical."""
    records = [_health(0, 0, rankme=5.0, cosine_drift=0.0), _health(3, 1, rankme=4.0, cosine_drift=0.4)]
    a = harness.Arm(name="a", role="live", records=list(records))
    b = harness.Arm(name="b", role="live", records=list(records))
    div = harness.trajectory_divergence(a, b)
    assert div["bit_identical"] is True
    assert div["max_abs_divergence"] == 0.0
    assert div["step_grid_matched"] is True


def test_trajectory_divergence_localizes_the_worst_instrument_gap() -> None:
    """A per-instrument gap is localized and surfaced as the worst overall divergence."""
    a = harness.Arm(name="a", role="live", records=[_health(0, 0, rankme=5.0, cosine_drift=0.10)])
    b = harness.Arm(name="b", role="live", records=[_health(0, 0, rankme=6.5, cosine_drift=0.12)])
    div = harness.trajectory_divergence(a, b)
    assert div["bit_identical"] is False
    assert math.isclose(div["max_abs_divergence"], 1.5, rel_tol=1e-6)  # rankme gap dominates
    assert math.isclose(div["per_surface_max"]["rankme"], 1.5, rel_tol=1e-6)
    assert math.isclose(div["per_surface_max"]["cosine_drift"], 0.02, rel_tol=1e-6)


def test_build_divergence_report_certifies_only_when_identical() -> None:
    """The report certifies iff both data identity and the matched trajectory are bit-identical."""
    identity = {"max_abs_image_diff": 0.0, "bit_identical": True, "num_steps": 3}
    matched = {"max_abs_divergence": 0.0, "bit_identical": True}
    desynced = {"max_abs_divergence": 0.61, "bit_identical": False}
    report = harness.build_divergence_report(
        config_header={"C": "mae", "seed": 0}, identity=identity, matched=matched, desynced=desynced
    )
    assert report["study"] == "P1.0.1"
    assert report["schema_version"] == harness.SCHEMA_VERSION
    assert report["attribution"] == {
        "stream_construction": 0.0,
        "in_arm_rng_matched": 0.0,
        "in_arm_rng_desynced": 0.61,
    }
    assert report["verdict"]["same_seed_certified"] is True
    assert report["verdict"]["measurement_sensitive"] is True

    # A non-identical rebuild must not certify.
    broken = harness.build_divergence_report(
        config_header={"C": "mae", "seed": 0},
        identity={"max_abs_image_diff": 0.2, "bit_identical": False, "num_steps": 3},
        matched=matched,
        desynced=desynced,
    )
    assert broken["verdict"]["same_seed_certified"] is False


def test_same_seed_arm_matches_the_shared_iterator_oracle_end_to_end(tmp_path: Path) -> None:
    """The load-bearing certification, end to end: a same-seed rebuild reproduces the oracle exactly.

    Runs one MAE arm two ways on the network-free synthetic source — on its own freshly-built
    :class:`EraStream` (same-seed, Option B) and on a materialized shared buffer (the oracle,
    Option A) — with each re-seeded identically. The stream's own generator never touches the
    global RNG, so both arms start training from the same state: the data is bit-identical and the
    health trajectory does not diverge.
    """

    def run(iterable: Iterable[StreamBatch], eval_sets: EvalSets, name: str) -> harness.Arm:
        torch.manual_seed(0)
        encoder = TinyViTEncoder(img_size=16, patch_size=8, embed_dim=32, depth=2, num_heads=2)
        method = build_mae(encoder)
        monitor = HealthMonitor(eval_sets, knn_k=5, run_knn=False, run_linear=False)
        return harness.run_stream_arm(
            name=name,
            role="live",
            method=method,
            stream=iterable,
            optimizer=torch.optim.AdamW(method.parameters(), lr=1e-3),
            selection_filter=AcceptAll(),
            monitor=monitor,
            out_dir=tmp_path,
            eval_every=2,
        )

    same_seed_stream = _synthetic_stream()
    arm_ss = run(same_seed_stream, same_seed_stream.eval_sets, "same_seed")
    oracle_stream = _synthetic_stream()
    oracle = list(oracle_stream)
    arm_or = run(oracle, oracle_stream.eval_sets, "shared_iterator")

    identity = harness.data_identity(list(_synthetic_stream()), oracle)
    matched = harness.trajectory_divergence(arm_ss, arm_or)
    assert identity["bit_identical"] is True  # re-seeding reconstructs the oracle's data exactly
    assert matched["bit_identical"] is True  # and the trajectory does not diverge


# ---------------------------------------------------------------------- light end-to-end wiring


def _synthetic_stream() -> EraStream:
    """A tiny network-free class-blocked stream for the end-to-end arms."""
    return EraStream(
        SyntheticSource(num_classes=3, per_class=48, img_size=16),
        batch_size=12,
        support_per_class=8,
        query_per_class=8,
        era_eval_per_class=5,
    )


def test_live_and_b5_arms_share_the_step_grid(tmp_path: Path) -> None:
    """The frozen B5 floor is stamped onto the live arm's grid; it never steps and stays finite."""
    torch.manual_seed(0)
    encoder = TinyViTEncoder(img_size=16, patch_size=8, embed_dim=32, depth=2, num_heads=2)
    method = build_mae(encoder)
    b5_frozen = copy.deepcopy(method)  # snapshot BEFORE any gradient step
    stream = _synthetic_stream()
    monitor = HealthMonitor(stream.eval_sets, knn_k=5, run_knn=False, run_linear=False)

    live = harness.run_stream_arm(
        name="live",
        role="live",
        method=method,
        stream=stream,
        optimizer=torch.optim.AdamW(method.parameters(), lr=1e-3),
        selection_filter=AcceptAll(),
        monitor=monitor,
        out_dir=tmp_path,
        eval_every=2,
    )
    b5 = harness.run_frozen_arm(
        name="b5",
        frozen_method=b5_frozen,
        monitor=HealthMonitor(stream.eval_sets, knn_k=5, run_knn=False, run_linear=False),
        grid=harness.health_grid(live),
    )
    assert harness.health_grid(b5) == harness.health_grid(live)  # co-logged on one step index
    assert b5.loss_records == []  # measured only, never stepped
    assert harness.all_finite(live, b5)


def test_pc_arm_wires_projector_surface_over_multiple_epochs(tmp_path: Path) -> None:
    """The manufactured-collapse PC (SimSiam) runs multi-epoch, logging the projector surface.

    A *wiring* guard, not a collapse-strength assertion — re-proving that anti-collapse-off SimSiam
    craters is ``test_positive_control.py``'s job (and it needs the calibrated regime, not this
    tiny toy scale). Here we assert the harness drives the PC arm end to end: it produces a loss
    floor and a projector-RankMe health series (the surface :func:`~cafl4ds.harness.collapse_gate`
    reads), runs the full multi-epoch horizon, and stays finite.
    """
    torch.manual_seed(0)
    encoder = TinyViTEncoder(img_size=16, patch_size=8, embed_dim=32, depth=2, num_heads=2)
    method = build_simsiam(encoder, anti_collapse=False)
    stream = _synthetic_stream()
    monitor = HealthMonitor(stream.eval_sets, knn_k=5, run_knn=False, run_linear=False)

    epochs = 4
    pc = harness.run_stream_arm(
        name="pc",
        role="pc",
        method=method,
        stream=stream,
        optimizer=torch.optim.AdamW(method.parameters(), lr=1e-3),
        selection_filter=AcceptAll(),
        monitor=monitor,
        out_dir=tmp_path,
        epochs=epochs,
        eval_every=2,
    )
    assert pc.loss_floor is not None  # a loss series was logged (the collapse "right reason" read)
    assert all("rankme_proj" in r for r in pc.health)  # the projector surface the gate reads
    # Multi-epoch numbers steps globally, so the horizon exceeds a single pass over the stream.
    assert int(pc.health[-1]["step"]) >= len(stream)
    assert harness.all_finite(pc)
