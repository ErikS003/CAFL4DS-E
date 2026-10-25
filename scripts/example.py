"""Example CLI / function calling script."""

import sys

import hydra
from hydra.utils import instantiate
from loguru import logger
from omegaconf import DictConfig

logger.add(sys.stdout, format="{time} {level} {message}", filter="example", level="INFO")


@hydra.main(version_base=None, config_path="../cafl4ds/configs", config_name="pipeline")  # type: ignore[misc]
def main(config: DictConfig) -> None:
    """Instantiate the pipeline from the Hydra config and run it."""
    pipeline = instantiate(config, _convert_="all")
    pipeline.run()


if __name__ == "__main__":
    main()  # pylint: disable=no-value-for-parameter
