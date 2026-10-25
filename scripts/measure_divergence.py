"""Arm-alignment divergence measurement (P1.0.1) — same-seed vs. the shared-iterator oracle.

The Phase-0 comparison method (``positive_control.py``) aligns two arms by re-seeding each so it
rebuilds "the same" stream, then ``zip``-ing their step-indexed series post-hoc. That rests on an
unverified assumption: re-seeding reconstructs a **bit-identical** stream and keeps the arms in
lockstep. This script certifies it against the **shared-iterator oracle** — one materialized batch
buffer every arm draws from, so the data is *provably* identical by construction (Option A) — and
owns the number: the divergence magnitude, split into its two contributors.

It runs one fixed arm (a ``C`` x ``I`` cell) four ways and reads three measurements off them:

* **same-seed** — the arm on its own freshly-built :class:`~cafl4ds.data.streams.EraStream` (Option
  B, Phase-0's default), vs.
* **shared-iterator** — the same arm on the oracle's materialized buffer (Option A);
  their per-step **data identity** certifies the stream-construction half, and their
  **trajectory divergence** (``matched``) the reproduction half — both expected bit-identical.
* an **in-arm desync control** — two arms with *identical init* on the *identical* oracle buffer,
  but with the in-arm RNG (augmentation / MAE masking / dropout) deliberately desynced; their
  trajectory divergence is the in-arm lever's magnitude — what Phase-0's per-arm re-seed buys.

Examples:
    The 2x2 C x I STL-10 certification, as a multirun::

        uv run python scripts/measure_divergence.py -m ssl=mae,simsiam init=from_scratch,pretrained

    Fastest network-free run::

        uv run python scripts/measure_divergence.py data=synthetic img_size=16 eval_every=3
"""

import json
import sys
from collections.abc import Iterable
from pathlib import Path
from typing import Any

import hydra
import torch
from hydra.core.hydra_config import HydraConfig
from hydra.utils import instantiate, to_absolute_path
from loguru import logger
from omegaconf import DictConfig

from cafl4ds import harness
from cafl4ds.data.streams import EvalSets, StreamBatch
from cafl4ds.ssl.base import SSLMethod, apply_encoder_init

logger.remove()
logger.add(sys.stdout, level="INFO")


def _fresh_method(config: DictConfig) -> SSLMethod:
    """Reset the global RNG to ``seed`` and build the ``C`` x ``I`` method (identical init each call).

    Re-seeding here is what makes the arms comparable: the encoder init and every in-arm draw start
    from the same RNG state, so the *only* thing that can differ between two arms is how the stream
    is delivered (rebuilt vs. materialized) — exactly the variable under test.
    """
    torch.manual_seed(int(config.seed))
    method: SSLMethod = instantiate(config.ssl, encoder=instantiate(config.encoder))
    checkpoint = config.init.checkpoint
    if config.init.mode == "pretrained" and not checkpoint:
        checkpoint = str(Path(to_absolute_path(config.pretrain_dir)) / f"{method.name}.pt")
    apply_encoder_init(method.encoder, config.init.mode, checkpoint)
    return method


def _run_arm(
    config: DictConfig,
    method: SSLMethod,
    stream: Iterable[StreamBatch],
    eval_sets: EvalSets,
    name: str,
    out_dir: Path,
) -> harness.Arm:
    """Run one single-pass arm (RNG state controlled by the caller — this never re-seeds)."""
    return harness.run_stream_arm(
        name=name,
        role="live",
        method=method,
        stream=stream,
        optimizer=instantiate(config.optim, params=method.parameters()),
        selection_filter=instantiate(config.filter),
        monitor=instantiate(config.monitor, eval_sets=eval_sets),
        out_dir=out_dir,
        eval_every=config.eval_every,
        device=config.device,
    )


def _measure_divergence(config: DictConfig, out_dir: Path) -> dict[str, Any]:
    """Run the four arms, read the three measurements, and assemble the P1.0.1 report.

    Args:
        config: The composed divergence config.
        out_dir: Directory the arm logs + ``divergence.json`` are written to.

    Returns:
        The P1.0.1 divergence-report dict.
    """
    # same-seed arm (Option B): its own freshly-built stream. Building an EraStream draws from its
    # *own* generator, never the global RNG, so `_fresh_method` below still starts training from the
    # seeded state — the arm sees data + in-arm RNG both determined by `seed`.
    ss_stream = instantiate(config.stream)
    ss_method = _fresh_method(config)
    c_name = ss_method.name
    arm_ss = _run_arm(config, ss_method, ss_stream, ss_stream.eval_sets, "same_seed", out_dir)

    # shared-iterator oracle (Option A): materialize one buffer; every arm below draws from it, so
    # the data is provably identical by construction.
    ora_stream = instantiate(config.stream)
    oracle: list[StreamBatch] = list(ora_stream)
    arm_or = _run_arm(config, _fresh_method(config), oracle, ora_stream.eval_sets, "shared_iterator", out_dir)

    # in-arm desync control: two arms, identical init (each `_fresh_method` re-seeds) on the identical
    # oracle buffer — but the second burns `desync_draws` RNG values *after* init, so its in-arm draws
    # (augmentation / MAE masking / dropout) desync while the data stays pinned. Isolates the in-arm
    # RNG contributor.
    arm_base = _run_arm(config, _fresh_method(config), oracle, ora_stream.eval_sets, "inarm_base", out_dir)
    method_desync = _fresh_method(config)
    torch.rand(int(config.desync_draws))  # burn training RNG only (init already drawn) -> in-arm desync
    arm_desync = _run_arm(config, method_desync, oracle, ora_stream.eval_sets, "inarm_desync", out_dir)

    identity = harness.data_identity(list(instantiate(config.stream)), oracle)
    matched = harness.trajectory_divergence(arm_ss, arm_or)
    desynced = harness.trajectory_divergence(arm_base, arm_desync)

    header = {
        "C": c_name,
        "I": config.init.mode,
        "F": config.stream.order,
        "seed": int(config.seed),
        "alignment": "shared_iterator",
    }
    report = harness.build_divergence_report(
        config_header=header, identity=identity, matched=matched, desynced=desynced
    )
    _log_verdict(report)
    return report


def _log_verdict(report: dict[str, Any]) -> None:
    """Log the human-readable P1.0.1 verdict (data identity, matched + desynced trajectory gaps)."""
    identity, attribution, verdict = report["data_identity"], report["attribution"], report["verdict"]
    logger.info(
        "P1.0.1 arm-alignment divergence:\n"
        f"  data identity (rebuild vs oracle): bit_identical={identity['bit_identical']} "
        f"[max_abs_image_diff={identity['max_abs_image_diff']:.2e} over {identity['num_steps']} steps]\n"
        f"  stream construction contributor: {attribution['stream_construction']:.2e}\n"
        f"  in-arm RNG (matched re-seed):    {attribution['in_arm_rng_matched']:.2e}\n"
        f"  in-arm RNG (desynced control):   {attribution['in_arm_rng_desynced']:.3f} "
        f"(sensitive={verdict['measurement_sensitive']})\n"
        f"  same-seed alignment certified:   {verdict['same_seed_certified']}"
    )


@hydra.main(version_base=None, config_path="../cafl4ds/configs", config_name="divergence")  # type: ignore[misc]
def main(config: DictConfig) -> None:
    """Run the P1.0.1 divergence measurement and write ``divergence.json``."""
    out_dir = Path(HydraConfig.get().runtime.output_dir)
    report = _measure_divergence(config, out_dir)

    (out_dir / config.report).write_text(json.dumps(report, indent=2), encoding="utf-8")
    logger.info(f"wrote divergence report to {out_dir / config.report}")
    if not report["verdict"]["same_seed_certified"]:
        logger.error(
            "Same-seed alignment NOT certified — the Phase-0 comparison method's data/reproduction "
            "assumption is violated; investigate before trusting the `zip`-aligned PCs."
        )


if __name__ == "__main__":
    main()  # pylint: disable=no-value-for-parameter
