"""P1.0.2 warm-the-well step — produce a reusable "generally competent" backbone for the drive.

Trains a config-selected backbone on the regime stream's **stationary** all-regime diet until the SSL
loss plateaus (Phase 0's competence-plateau discipline), then saves the **whole method** — encoder and
head — as a well the deployment drive resumes intact (:mod:`cafl4ds.warmup`). Producing the well once
and pointing many drives at it (``well=<path>``) means every warm drive starts from the *identical*
competent state, so drive-to-drive comparisons carry no per-run warm-up variance.

Examples:
    Warm a synthetic well (fresh clone, no dataset)::

        uv run python scripts/warm_well.py data=attr_synthetic img_size=32 well_out=wells/je_well.pt

    Warm the default BDD well for the joint-embedding backbone::

        uv run python scripts/warm_well.py data=bdd bdd_root=/abs/path backbone=je well_out=wells/je_well.pt
"""

import sys
from pathlib import Path

import hydra
import torch
from hydra.utils import instantiate, to_absolute_path
from loguru import logger
from omegaconf import DictConfig

from cafl4ds import warmup
from cafl4ds.ssl.base import SSLMethod, apply_encoder_init

logger.remove()
logger.add(sys.stdout, level="INFO")


def warm(config: DictConfig) -> Path:
    """Warm a competent well on the stationary diet and save the full method (Hydra-free core).

    Kept separate from :func:`main` so it is unit-testable on the synthetic source.

    Args:
        config: The composed ``warm_well`` config.

    Returns:
        The path the warmed well was saved to.
    """
    torch.manual_seed(int(config.seed))
    method: SSLMethod = instantiate(config.ssl, encoder=instantiate(config.encoder))
    apply_encoder_init(method.encoder, config.init.mode, config.init.checkpoint)
    stream = instantiate(config.stream)
    optimizer = instantiate(config.optim, params=method.parameters())

    logger.info(f"warming {method.name} well on the stationary diet ({stream.num_eras} regimes), dev={config.device}")
    summary = warmup.warm_until_settled(
        method=method,
        stream=stream,
        optimizer=optimizer,
        plateau=warmup.PlateauCriterion(
            window=int(config.warmup.window), tol=float(config.warmup.tol), min_steps=int(config.warmup.min_steps)
        ),
        max_steps=int(config.warmup.max_steps),
        max_passes=int(config.warmup.max_passes),
        grad_clip=config.warmup.grad_clip,
        device=config.device,
    )

    # A competence read on the reserved canary probes — a sanity line, not a gate.
    health = instantiate(config.monitor, eval_sets=stream.eval_sets).measure(method, summary["steps"])
    logger.info(f"well competence: knn_acc={health.get('knn_acc')}, linear_acc={health.get('linear_acc')}")

    well_path = Path(to_absolute_path(str(config.well_out)))
    warmup.save_well(method, well_path)
    logger.info(f"warm well ready — drive it with `well={well_path}`")
    return well_path


@hydra.main(version_base=None, config_path="../cafl4ds/configs", config_name="warm_well")  # type: ignore[misc]
def main(config: DictConfig) -> None:
    """Warm a competent well on the stationary diet and save the full method."""
    warm(config)


if __name__ == "__main__":
    main()  # pylint: disable=no-value-for-parameter
