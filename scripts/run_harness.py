"""Phase-1 multi-arm harness entry point (P1.0.0).

Consolidates the scattered Phase-0 pieces — the adapting backbone, the frozen floor B5 (measured
only at run-end in ``run_loop.py``), and a separate positive-control script per mode — into one
loop that co-logs the **live / PC / B5** arms into a single canonical ``comparison.json`` (schema
owned by :mod:`cafl4ds.harness`). Default alignment is Phase-0's sequential same-seed arms
(Option B); the shared-iterator oracle (Option A) is P1.0.1's.

Deliberately number-free: it proves the rig runs, the instruments move in the right direction, the
PC fires, and nothing goes non-finite — plus, under ``long_horizon=true``, that a resampled-image
run stays finite with flat memory.

Examples:
    The full C x I STL-10 smoke (live / PC / B5 co-logged), as a multirun::

        uv run python scripts/run_harness.py -m ssl=mae,simsiam init=from_scratch,pretrained

    Network-free CPU smoke with a gate-checked control horizon::

        uv run python scripts/run_harness.py --config-name harness_smoke

    The long-horizon leak check::

        uv run python scripts/run_harness.py data=synthetic long_horizon=true
"""

import copy
import json
import sys
from pathlib import Path
from typing import Any

import hydra
import torch
from hydra.core.hydra_config import HydraConfig
from hydra.utils import instantiate, to_absolute_path
from loguru import logger
from omegaconf import DictConfig

from cafl4ds import harness
from cafl4ds.data.sources import SyntheticSource
from cafl4ds.data.streams import EraStream
from cafl4ds.ssl.base import SSLMethod, apply_encoder_init

logger.remove()
logger.add(sys.stdout, level="INFO")


def _live_method(config: DictConfig) -> SSLMethod:
    """Instantiate the live SSL method and apply its initialization (the ``C`` × ``I`` cell)."""
    method: SSLMethod = instantiate(config.ssl, encoder=instantiate(config.encoder))
    checkpoint = config.init.checkpoint
    if config.init.mode == "pretrained" and not checkpoint:
        checkpoint = str(Path(to_absolute_path(config.pretrain_dir)) / f"{method.name}.pt")
    apply_encoder_init(method.encoder, config.init.mode, checkpoint)
    return method


def _run_pc_arm(config: DictConfig, out_dir: Path) -> harness.Arm:
    """Run the manufactured-collapse PC arm — SimSiam with anti-collapse off (a fixed control).

    Built from its own fresh encoder under the run seed (from-scratch — a PC never loads a warm
    start), on a same-seed stream, over the multi-epoch horizon that makes the collapse fire
    decisively at the smoke scale.

    Args:
        config: The composed harness config (its ``pc`` block selects the manufactured pathology).
        out_dir: Directory the arm's run log is written to.

    Returns:
        The PC :class:`~cafl4ds.harness.Arm`.
    """
    torch.manual_seed(config.seed)
    method = instantiate(config.pc.ssl, encoder=instantiate(config.encoder))
    apply_encoder_init(method.encoder, "from_scratch")
    stream = instantiate(config.stream)
    return harness.run_stream_arm(
        name=f"{method.name}_pc",
        role="pc",
        method=method,
        stream=stream,
        optimizer=instantiate(config.optim, params=method.parameters()),
        selection_filter=instantiate(config.filter),
        monitor=instantiate(config.monitor, eval_sets=stream.eval_sets),
        out_dir=out_dir,
        eval_every=config.eval_every,
        epochs=config.pc.epochs,
        device=config.device,
    )


def _run_smoke(config: DictConfig, out_dir: Path) -> dict[str, Any]:
    """Run the three co-logged arms, gate the PC, and assemble the canonical comparison.

    Args:
        config: The composed harness config.
        out_dir: Directory the arm logs + ``comparison.json`` are written to.

    Returns:
        The canonical ``comparison.json`` dict.
    """
    torch.manual_seed(config.seed)
    method = _live_method(config)
    b5_frozen = copy.deepcopy(method)  # the frozen floor: snapshot BEFORE any gradient step

    stream = instantiate(config.stream)
    run_name = config.run_name or f"{method.name}_{config.init.mode}"
    logger.info(f"harness '{run_name}': {stream.num_eras} eras, {len(stream)} batches, device={config.device}")

    live = harness.run_stream_arm(
        name=f"{run_name}_live",
        role="live",
        method=method,
        stream=stream,
        optimizer=instantiate(config.optim, params=method.parameters()),
        selection_filter=instantiate(config.filter),
        monitor=instantiate(config.monitor, eval_sets=stream.eval_sets),
        out_dir=out_dir,
        eval_every=config.eval_every,
        device=config.device,
    )
    # B5 shares the live arm's step grid (measured only, never stepped). A fresh monitor with the
    # same eval sets suffices — a frozen model does not drift.
    b5 = harness.run_frozen_arm(
        name=f"{run_name}_b5",
        frozen_method=b5_frozen,
        monitor=instantiate(config.monitor, eval_sets=stream.eval_sets),
        grid=harness.health_grid(live),
        device=config.device,
    )
    pc = _run_pc_arm(config, out_dir)

    gate = harness.collapse_gate(
        pc,
        pc_loss_floor_max=config.gate.pc_loss_floor_max,
        pc_rankme_drop_frac=config.gate.pc_rankme_drop_frac,
    )
    directional = harness.directional_report(live)
    finite = harness.all_finite(live, pc, b5)

    header = {
        "C": method.name,
        "I": config.init.mode,
        "F": config.stream.order,
        "A": _target_leaf(config.filter),
        "seed": int(config.seed),
        "alignment": config.alignment,
    }
    comparison = harness.build_comparison(
        config_header=header,
        gate=gate,
        arms=[live, pc, b5],
        directional=directional,
        extensions={"finite": finite},
    )
    _log_smoke_verdict(gate, directional, finite)
    return comparison


def _run_long_horizon(config: DictConfig, out_dir: Path) -> dict[str, Any]:
    """The no-blow-up / no-leak check — repeat the live arm over a resampled-image stream.

    Continually adapts *one* method over several passes, each a freshly-resampled synthetic stream
    (a long horizon without new data), sampling process RSS after each pass. Passes iff every value
    stays finite and memory stays flat (no leak).

    Args:
        config: The composed harness config (``long_horizon_*`` knobs).
        out_dir: Directory the per-pass logs are written to.

    Returns:
        The canonical ``comparison.json`` dict (a single live arm + the leak report).
    """
    torch.manual_seed(config.seed)
    method = _live_method(config)
    optimizer = instantiate(config.optim, params=method.parameters())

    passes: list[harness.Arm] = []
    rss_samples: list[float] = []
    for p in range(int(config.long_horizon_passes)):
        source = SyntheticSource(
            num_classes=config.data.get("num_classes", 4),
            per_class=int(config.long_horizon_per_class),
            img_size=config.img_size,
            seed=config.seed + p,  # resampled images each pass — the long horizon
        )
        stream = EraStream(
            source,
            batch_size=config.batch_size,
            order=config.stream.order,
            max_train_per_class=config.max_train_per_class,
            seed=config.seed + p,
        )
        arm = harness.run_stream_arm(
            name=f"long_horizon_p{p}",
            role="live",
            method=method,  # reused across passes: one model, long horizon
            stream=stream,
            optimizer=optimizer,
            selection_filter=instantiate(config.filter),
            monitor=instantiate(config.monitor, eval_sets=stream.eval_sets),
            out_dir=out_dir,
            eval_every=config.eval_every,
            device=config.device,
        )
        passes.append(arm)
        rss_samples.append(harness.rss_mb())
        logger.info(f"long-horizon pass {p}: {len(arm.health)} checkpoints, RSS {rss_samples[-1]:.1f} MiB")

    finite = harness.all_finite(*passes)
    leak = harness.leak_report(rss_samples, growth_frac_max=float(config.long_horizon_growth_frac_max))
    gate = {
        "mode": "long_horizon",
        "passed": bool(finite and leak["flat"]),
        "checks": {"finite": finite, "memory_flat": leak["flat"]},
    }
    # Only the last pass's series is co-logged as the representative live arm (the full per-pass
    # logs remain in the run dir); the leak report carries the cross-pass memory trace.
    comparison = harness.build_comparison(
        config_header={"C": method.name, "I": config.init.mode, "mode": "long_horizon", "seed": int(config.seed)},
        gate=gate,
        arms=[passes[-1]],
        extensions={"finite": finite, "leak_check": leak},
    )
    logger.info(
        f"long-horizon verdict: finite={finite}, memory_flat={leak['flat']} "
        f"(tail spread {leak['tail_spread_frac']:+.3f} vs {leak['growth_frac_max']:.3f})"
    )
    return comparison


def _target_leaf(node: DictConfig) -> str:
    """The filter's short name — the ``_target_`` class leaf (the ``A`` factor label)."""
    return str(node.get("_target_", "filter")).rsplit(".", 1)[-1]


def _log_smoke_verdict(gate: dict[str, Any], directional: dict[str, Any], finite: bool) -> None:
    """Log the human-readable smoke verdict (PC fired, instruments moved, nothing NaN'd)."""
    rep = gate["reported"]
    logger.info(
        "P1.0.0 harness smoke:\n"
        f"  PC fired (gate): {gate['passed']}  "
        f"[loss_floor={rep['pc_loss_floor']:.3f}, proj_rankme {rep['pc_rankme_proj_init']:.2f} -> "
        f"{rep['pc_rankme_proj_final']:.2f} (x{rep['pc_rankme_proj_drop_frac']:.2f})]\n"
        f"  live instruments moved: rankme {directional['rankme_init']:.2f} -> {directional['rankme_final']:.2f} "
        f"(responded={directional['rankme_responded']}), drift={directional['final_cosine_drift']:.3f} "
        f"(accumulated={directional['drift_accumulated']})\n"
        f"  all values finite: {finite}"
    )


@hydra.main(version_base=None, config_path="../cafl4ds/configs", config_name="harness")  # type: ignore[misc]
def main(config: DictConfig) -> None:
    """Run the P1.0.0 harness and write the canonical ``comparison.json``."""
    out_dir = Path(HydraConfig.get().runtime.output_dir)
    comparison = _run_long_horizon(config, out_dir) if config.long_horizon else _run_smoke(config, out_dir)

    (out_dir / config.comparison).write_text(json.dumps(comparison, indent=2), encoding="utf-8")
    logger.info(f"wrote canonical comparison to {out_dir / config.comparison}")
    if not comparison["gate"]["passed"]:
        logger.error("Gate did NOT pass — the harness apparatus is suspect; investigate before trusting it.")


if __name__ == "__main__":
    main()  # pylint: disable=no-value-for-parameter
