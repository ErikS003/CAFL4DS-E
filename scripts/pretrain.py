"""Produce the ``I=pretrained`` warm-start checkpoint via an IID SSL pre-pass.

Runs the chosen SSL method over the **shuffled** (IID) stream for a few epochs and saves the
encoder's ``state_dict`` to ``<pretrain_dir>/<method_name>.pt``. The IID ordering is the whole
point: the warm start must be a well-behaved reference, so stream correlation enters only in
the streaming phase (:mod:`scripts.run_loop`). This is a plain training loop — no monitor, no
filter — because it produces an artifact, not a health trajectory.

Examples:
    Both backbones::

        uv run python scripts/pretrain.py -m ssl=mae,simsiam

    Fast network-free smoke::

        uv run python scripts/pretrain.py data=synthetic img_size=16 epochs=1
"""

import sys
from pathlib import Path

import hydra
import torch
from hydra.utils import instantiate, to_absolute_path
from loguru import logger
from omegaconf import DictConfig

from cafl4ds.ssl.base import save_encoder_checkpoint
from cafl4ds.warmup import save_well

logger.remove()
logger.add(sys.stdout, level="INFO")

_MIN_BATCH = 2  # BatchNorm heads / per-patch stats need at least two samples.


@hydra.main(version_base=None, config_path="../cafl4ds/configs", config_name="pretrain")  # type: ignore[misc]
def main(config: DictConfig) -> None:
    """Run the IID pre-pass and save the encoder checkpoint."""
    torch.manual_seed(config.seed)
    device = torch.device(config.device)

    encoder = instantiate(config.encoder)
    method = instantiate(config.ssl, encoder=encoder).to(device)
    stream = instantiate(config.stream)
    optimizer = instantiate(config.optim, params=method.parameters())

    method.train()
    step = 0
    for epoch in range(config.epochs):
        for batch in stream:
            images = batch.images.to(device)
            if images.shape[0] < _MIN_BATCH:
                continue
            optimizer.zero_grad()
            loss = method.training_step(images)
            loss.backward()
            if config.grad_clip is not None:
                torch.nn.utils.clip_grad_norm_(method.parameters(), config.grad_clip)
            optimizer.step()
            if step % 10 == 0:
                logger.info(f"[pretrain {method.name}] epoch {epoch} step {step} loss={loss.item():.4f}")
            step += 1

    out = Path(to_absolute_path(config.pretrain_dir)) / f"{method.name}.pt"
    save_encoder_checkpoint(method.encoder, out)
    if config.get("full_method_path"):
        save_well(method, to_absolute_path(str(config.full_method_path)))


if __name__ == "__main__":
    main()  # pylint: disable=no-value-for-parameter
