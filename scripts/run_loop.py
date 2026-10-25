"""Phase-0 streaming-loop entry point.

Wires the config-instantiated components in dependency order (the standard Hydra recipe when
there are runtime cross-references: the optimizer needs the method's parameters, the monitor
needs the stream's held-out eval sets), then runs the loop. Loss and health land in one run
log as separate series.

Examples:
    Single run (STL-10, MAE, from-scratch)::

        uv run python scripts/run_loop.py

    Exit matrix (C x I) as a multirun::

        uv run python scripts/run_loop.py -m ssl=mae,simsiam init=from_scratch,pretrained

    Fast network-free smoke::

        uv run python scripts/run_loop.py data=synthetic img_size=16 eval_every=3
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

from cafl4ds.data.streams import EraStream
from cafl4ds.eval import PerEraProbe, adaptation_report
from cafl4ds.run_log import RunLogger
from cafl4ds.ssl.base import SSLMethod, apply_encoder_init

logger.remove()
logger.add(sys.stdout, level="INFO")


def _fmt(x: float | None) -> str:
    """Format an optionally-undefined metric (``None`` for the single-era / IID case)."""
    return f"{x:+.4f}" if x is not None else "n/a (needs >= 2 eras)"


def _report_establish(
    config: DictConfig,
    method: SSLMethod,
    b5: SSLMethod | None,
    era_evaluator: PerEraProbe | None,
    stream: EraStream,
    out_dir: Path,
) -> None:
    """Print the three Phase-0 establishing questions and write ``establish.json``.

    Args:
        config: The composed config (its ``eval`` block selects the probe).
        method: The adapted SSL method after the loop.
        b5: The init-matched frozen backbone twin (baseline B5), or ``None`` if not requested.
        era_evaluator: The probe-on-past evaluator, or ``None`` if not requested.
        stream: The stream (supplies the held-out eval sets).
        out_dir: Directory the ``establish.json`` report is written to.
    """
    probe, knn_k = config.eval.probe, config.eval.knn_k
    report: dict[str, Any] = {}
    lines = ["PHASE-0 ESTABLISHING QUESTIONS (validating axis)"]

    if b5 is not None:
        adapt = adaptation_report(method.encode, b5.encode, stream.eval_sets, probe=probe, knn_k=knn_k)
        report["adaptation_vs_b5"] = adapt
        lines.append(
            f"  (a) adaptation vs B5 [{probe}]: adapted {adapt['adapted_acc']:.3f} vs "
            f"frozen {adapt['b5_acc']:.3f}  (gain {adapt['gain']:+.3f} -> "
            f"{'adaptation helps' if adapt['gain'] > 0 else 'no gain'})"
        )

    if era_evaluator is not None:
        summary = era_evaluator.summary()
        report["probe_on_past"] = {"matrix": era_evaluator.matrix, **summary}
        per_era = summary["per_era_final"]
        lines.append(
            f"  (b) degradation [{probe}]: BackwardTransfer={_fmt(summary['backward_transfer'])}  "
            f"ForgettingMeasure={_fmt(summary['forgetting_measure'])}  (over {summary['num_eras']} recorded eras)"
        )
        if per_era:
            lines.append("      per-era final acc: " + ", ".join(f"e{e}={a:.2f}" for e, a in sorted(per_era.items())))

    lines.append("  (c) knob moves health: compare the RankMe/drift health series across filter= runs")
    logger.info("\n".join(lines))
    (out_dir / "establish.json").write_text(json.dumps(report, indent=2), encoding="utf-8")
    logger.info(f"wrote establishing report to {out_dir / 'establish.json'}")


@hydra.main(version_base=None, config_path="../cafl4ds/configs", config_name="loop")  # type: ignore[misc]
def main(config: DictConfig) -> None:
    """Instantiate and run the Phase-0 streaming loop from the Hydra config."""
    torch.manual_seed(config.seed)

    encoder = instantiate(config.encoder)
    method = instantiate(config.ssl, encoder=encoder)

    checkpoint = config.init.checkpoint
    if config.init.mode == "pretrained" and not checkpoint:
        checkpoint = str(Path(to_absolute_path(config.pretrain_dir)) / f"{method.name}.pt")
    apply_encoder_init(method.encoder, config.init.mode, checkpoint)

    stream = instantiate(config.stream)
    optimizer = instantiate(config.optim, params=method.parameters())
    monitor = instantiate(config.monitor, eval_sets=stream.eval_sets)
    selection_filter = instantiate(config.filter)

    # Validating-axis evaluation (off by default). B5 is an init-matched frozen twin snapshot
    # taken BEFORE any gradient step; the era probe records forgetting as the stream advances.
    b5 = copy.deepcopy(method) if config.eval.compare_b5 else None
    era_evaluator = (
        PerEraProbe(stream.eval_sets, probe=config.eval.probe, knn_k=config.eval.knn_k)
        if config.eval.probe_on_past
        else None
    )

    run_name = config.run_name or f"{method.name}_{config.init.mode}"
    # Resolve the run log under Hydra's per-run output dir so multirun jobs never collide
    # (Hydra 1.3 defaults job.chdir=False, so a bare relative path lands in the shared cwd).
    run_log_path = Path(HydraConfig.get().runtime.output_dir) / config.run_log
    run_logger = RunLogger(run_log_path, run_name=run_name)
    logger.info(f"run '{run_name}': {stream.num_eras} eras, {len(stream)} batches, device={config.device}")

    loop = instantiate(
        config.loop,
        stream=stream,
        method=method,
        optimizer=optimizer,
        selection_filter=selection_filter,
        monitor=monitor,
        run_logger=run_logger,
        era_evaluator=era_evaluator,
    )
    loop.run()

    if b5 is not None or era_evaluator is not None:
        _report_establish(config, method, b5, era_evaluator, stream, Path(HydraConfig.get().runtime.output_dir))


if __name__ == "__main__":
    main()  # pylint: disable=no-value-for-parameter
