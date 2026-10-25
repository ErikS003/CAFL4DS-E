"""An example module: a tiny text-transform pipeline.

This is throwaway "replace-me" starter code. It exists to show how the project is
wired together -- in particular how Hydra's ``instantiate`` builds a *nested* object
graph from config: a ``Pipeline`` whose ``steps`` are themselves ``_target_`` objects
(see ``configs/pipeline.yaml``). Delete it and drop in your own package code.
"""

from abc import ABC, abstractmethod
from dataclasses import dataclass

from loguru import logger


class Transform(ABC):
    """A single pipeline step that maps a string to a string."""

    @abstractmethod
    def __call__(self, text: str) -> str:
        """Apply the transform to ``text`` and return the result."""


class Upper(Transform):
    """Upper-cases the text."""

    def __call__(self, text: str) -> str:
        """Return ``text`` upper-cased."""
        return text.upper()


class Repeat(Transform):
    """Repeats the text ``times`` times, space-separated."""

    def __init__(self, times: int = 2) -> None:
        """Store how many times to repeat the text."""
        self.times = times

    def __call__(self, text: str) -> str:
        """Return ``text`` repeated ``times`` times, space-separated."""
        return " ".join([text] * self.times)


@dataclass
class Pipeline:
    """Runs an ordered list of transforms over a starting message."""

    steps: list[Transform]
    message: str = "hello"

    def run(self) -> str:
        """Apply each transform in turn, log the result, and return it."""
        result = self.message
        for step in self.steps:
            result = step(result)
        logger.info(result)
        return result
