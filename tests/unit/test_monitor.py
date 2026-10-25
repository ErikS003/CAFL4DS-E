"""Tests for the health monitor.

Confirms the monitor reports the expected metric surface, that drift is zero at the first
checkpoint and becomes positive once the representation moves, and that measuring never
leaves the method in eval mode (it must restore training).
"""

import torch

from cafl4ds.data.sources import SyntheticSource
from cafl4ds.data.streams import EraStream
from cafl4ds.models.vit import TinyViTEncoder
from cafl4ds.monitor import HealthMonitor
from cafl4ds.ssl.factory import build_mae, build_simsiam

# MAE exposes only the backbone surface (no projector → no `_proj` keys), but it now returns a
# positive pair from `make_views` (two augmentation draws, for the P0.5 alignment-on-trial read),
# so `alignment` is reported at the backbone.
_EXPECTED_KEYS = {
    "step",
    "rankme",
    "uniformity",
    "alignment",
    "offdiag_cov",
    "mean_feature_var",
    "cka_drift",
    "cosine_drift",
    "knn_acc",
    "linear_acc",
}


def _method_and_monitor() -> tuple[object, HealthMonitor]:
    """Build a tiny method and a monitor over a synthetic stream's eval sets."""
    torch.manual_seed(0)
    encoder = TinyViTEncoder(img_size=16, patch_size=8, embed_dim=32, depth=2, num_heads=2)
    method = build_mae(encoder)
    stream = EraStream(
        SyntheticSource(num_classes=3, per_class=40, img_size=16),
        support_per_class=8,
        query_per_class=8,
        era_eval_per_class=5,
    )
    return method, HealthMonitor(stream.eval_sets, knn_k=5)


def test_measure_reports_expected_metrics() -> None:
    """A measurement returns exactly the expected metric keys, all finite floats."""
    method, monitor = _method_and_monitor()
    metrics = monitor.measure(method, step=0)  # type: ignore[arg-type]
    assert set(metrics) == _EXPECTED_KEYS
    assert all(isinstance(v, float) and v == v for v in metrics.values())  # not NaN


def test_first_checkpoint_has_zero_drift_then_positive() -> None:
    """Drift is zero at the first checkpoint and positive after the encoder changes."""
    method, monitor = _method_and_monitor()
    first = monitor.measure(method, step=0)  # type: ignore[arg-type]
    assert first["cka_drift"] == 0.0 and first["cosine_drift"] == 0.0

    opt = torch.optim.AdamW(method.parameters(), lr=1e-2)  # type: ignore[attr-defined]
    x = SyntheticSource(num_classes=3, per_class=8, img_size=16).load()[0]
    for _ in range(10):
        opt.zero_grad()
        loss = method.training_step(x)  # type: ignore[attr-defined]
        loss.backward()
        opt.step()

    later = monitor.measure(method, step=10)  # type: ignore[arg-type]
    assert later["cosine_drift"] > 0.0


def test_measure_restores_training_mode() -> None:
    """The monitor must not leave the method stuck in eval mode."""
    method, monitor = _method_and_monitor()
    method.train()  # type: ignore[attr-defined]
    monitor.measure(method, step=0)  # type: ignore[arg-type]
    assert method.training  # type: ignore[attr-defined]


def _simsiam_and_monitor() -> tuple[object, HealthMonitor]:
    """Build a tiny SimSiam (two surfaces + positive pair) and a monitor over its eval sets."""
    torch.manual_seed(0)
    encoder = TinyViTEncoder(img_size=16, patch_size=8, embed_dim=32, depth=2, num_heads=2)
    method = build_simsiam(encoder, proj_hidden=64, proj_dim=32, pred_hidden=32)
    stream = EraStream(
        SyntheticSource(num_classes=3, per_class=40, img_size=16),
        support_per_class=8,
        query_per_class=8,
        era_eval_per_class=5,
    )
    return method, HealthMonitor(stream.eval_sets, knn_k=5)


def test_joint_embedding_reports_both_surfaces_and_alignment() -> None:
    """SimSiam backone and projector surfaces.

    SimSiam exposes backbone + projector surfaces and a positive pair, so the monitor emits
    the geometry suite at both surfaces (``_proj`` keys) plus alignment at both.
    """
    method, monitor = _simsiam_and_monitor()
    metrics = monitor.measure(method, step=0)  # type: ignore[arg-type]
    surfaced = {"rankme", "mean_feature_var", "offdiag_cov", "uniformity", "alignment"}
    expected = {"step", "cka_drift", "cosine_drift", "knn_acc", "linear_acc"}
    expected |= surfaced | {f"{m}_proj" for m in surfaced}
    assert set(metrics) == expected
    assert all(isinstance(v, float) and v == v for v in metrics.values())  # all finite


def test_drift_surfaces_adds_projector_drift() -> None:
    """P0.6.1: ``drift_surfaces`` tracks drift at the projector too, keeping the backbone keys.

    By default drift is backbone-only (unsuffixed ``cka_drift`` / ``cosine_drift``). With
    ``drift_surfaces=True`` a joint-embedding method additionally reports ``cka_drift_proj`` /
    ``cosine_drift_proj`` — the past-data-free reader's second surface — and the backbone keys are
    unchanged. Both are zero at the first checkpoint (no reference yet).
    """
    torch.manual_seed(0)
    encoder = TinyViTEncoder(img_size=16, patch_size=8, embed_dim=32, depth=2, num_heads=2)
    method = build_simsiam(encoder, proj_hidden=64, proj_dim=32, pred_hidden=32)
    stream = EraStream(
        SyntheticSource(num_classes=3, per_class=40, img_size=16),
        support_per_class=8,
        query_per_class=8,
        era_eval_per_class=5,
    )
    monitor = HealthMonitor(stream.eval_sets, knn_k=5, run_knn=False, run_linear=False, drift_surfaces=True)
    first = monitor.measure(method, step=0)
    assert {"cka_drift", "cosine_drift", "cka_drift_proj", "cosine_drift_proj"} <= set(first)
    assert first["cka_drift"] == 0.0 and first["cka_drift_proj"] == 0.0  # both surfaces zero at t0

    opt = torch.optim.AdamW(method.parameters(), lr=1e-2)
    x = SyntheticSource(num_classes=3, per_class=8, img_size=16).load()[0]
    for _ in range(10):
        opt.zero_grad()
        loss = method.training_step(x)
        loss.backward()
        opt.step()

    later = monitor.measure(method, step=10)
    assert later["cosine_drift"] > 0.0 and later["cosine_drift_proj"] > 0.0  # both surfaces move


def test_alignment_view_pair_is_fixed_across_checkpoints() -> None:
    """Alignment view across checkpoints.

    The alignment view-pair is drawn once and reused, so it is identical across calls and
    does not consume the global RNG (two measurements draw the same pair).
    """
    method, monitor = _simsiam_and_monitor()
    monitor.measure(method, step=0)  # type: ignore[arg-type]
    first = monitor._views  # cached pair
    assert first is not None
    monitor.measure(method, step=1)  # type: ignore[arg-type]
    assert monitor._views is first  # same cached object, not re-drawn


def test_p052_readers_reported_when_enabled_for_mae() -> None:
    """Enabling the P0.5.2 candidate readers adds their keys for an MAE backbone.

    Clusterability, mean attention distance, and alignment-under-stronger-aug are off by default
    (so existing runs are unchanged); switching them on must emit exactly those three extra finite
    keys and nothing else.
    """
    method, _ = _method_and_monitor()
    stream = EraStream(
        SyntheticSource(num_classes=3, per_class=40, img_size=16),
        support_per_class=8,
        query_per_class=8,
        era_eval_per_class=5,
    )
    monitor = HealthMonitor(
        stream.eval_sets,
        knn_k=5,
        run_clusterability=True,
        run_attn_distance=True,
        run_alignment_strong=True,
    )
    metrics = monitor.measure(method, step=0)  # type: ignore[arg-type]
    extra = {"clusterability", "attn_distance", "alignment_strong"}
    assert extra <= set(metrics)
    assert set(metrics) == _EXPECTED_KEYS | extra
    assert all(isinstance(metrics[k], float) and metrics[k] == metrics[k] for k in extra)
