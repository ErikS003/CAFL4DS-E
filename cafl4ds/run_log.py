"""Run logging — one run log, two separate series (SSL loss and health).

The Phase-0 exit criterion is a single run log in which the **SSL loss** and the **health
metrics** (rankme, drift, probe) can be read *side by side over steps*. This module writes
one JSONL file per run, each line a record tagged with its ``series`` (``"loss"`` or
``"health"``), and mirrors a human-readable line to loguru. Every health record also carries
the most recent loss, so :func:`tabulate` can render the required side-by-side table from the
health series alone.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

from loguru import logger

# Columns rendered, in order, by :func:`tabulate` (only those present are shown).
_TABLE_COLUMNS = ("step", "era", "loss", "rankme", "cka_drift", "cosine_drift", "knn_acc", "linear_acc")


class RunLogger:
    """Append-only JSONL logger with distinct loss and health series."""

    def __init__(self, path: str | Path, run_name: str = "run") -> None:
        """Open (truncate) the run log.

        Args:
            path: Destination JSONL file (parent directories are created).
            run_name: A short name recorded on every line (for grouping across runs).
        """
        self.path = Path(path)
        self.run_name = run_name
        self.path.parent.mkdir(parents=True, exist_ok=True)
        self._file = self.path.open("w", encoding="utf-8")
        self._last_loss: float | None = None
        self._health: list[dict[str, Any]] = []

    def _write(self, record: dict[str, Any]) -> None:
        """Write one record as a JSON line and flush.

        Args:
            record: The record to serialize.
        """
        record = {"run": self.run_name, **record}
        self._file.write(json.dumps(record) + "\n")
        self._file.flush()

    def log_loss(
        self, step: int, era: int, loss: float, grad_norm: float | None = None, finite: bool | None = None
    ) -> None:
        """Record one point of the SSL-loss series.

        Args:
            step: Global step index.
            era: Current era (class block).
            loss: The SSL loss value at this step.
            grad_norm: Optional global pre-clip gradient L2 norm for the step (the P0.4.0
                divergence instrument); omitted from the record when ``None``.
            finite: Optional flag — whether the step's loss and grad norm are both finite;
                omitted when ``None``.
        """
        self._last_loss = loss
        record: dict[str, Any] = {"series": "loss", "step": step, "era": era, "loss": loss}
        if grad_norm is not None:
            record["grad_norm"] = grad_norm
        if finite is not None:
            record["finite"] = finite
        self._write(record)

    def log_health(self, step: int, era: int, metrics: dict[str, float]) -> None:
        """Record one point of the health series (with the most recent loss attached).

        Args:
            step: Global step index.
            era: Current era (class block).
            metrics: The metric dictionary from :meth:`cafl4ds.monitor.HealthMonitor.measure`.
        """
        record = {"series": "health", "step": step, "era": era, "loss": self._last_loss, **metrics}
        self._health.append(record)
        self._write(record)
        shown = " ".join(
            f"{k}={record[k]:.4f}" for k in _TABLE_COLUMNS if k in record and isinstance(record[k], int | float)
        )
        logger.info(f"[{self.run_name}] health {shown}")

    def close(self) -> None:
        """Close the underlying file."""
        self._file.close()

    def tabulate(self) -> str:
        """Render the health series (loss + metrics) as a side-by-side text table.

        Returns:
            A fixed-width table with one row per health checkpoint, or a placeholder string
            if no health was logged.
        """
        return tabulate(self._health)


def tabulate(health_records: list[dict[str, Any]]) -> str:
    """Render health records as a fixed-width side-by-side table.

    Args:
        health_records: Health-series records (as written by :meth:`RunLogger.log_health`).

    Returns:
        The formatted table, or a placeholder if there are no records.
    """
    if not health_records:
        return "(no health records)"
    cols = [c for c in _TABLE_COLUMNS if any(c in r and r[c] is not None for r in health_records)]
    header = "  ".join(f"{c:>12}" for c in cols)
    lines = [header, "  ".join("-" * 12 for _ in cols)]
    for r in health_records:
        cells = []
        for c in cols:
            v = r.get(c)
            cells.append(f"{v:>12.4f}" if isinstance(v, int | float) else f"{'-':>12}")
        lines.append("  ".join(cells))
    return "\n".join(lines)


def read_run(path: str | Path) -> list[dict[str, Any]]:
    """Read a run log back into a list of records.

    Args:
        path: Path to a JSONL run log.

    Returns:
        The records, in file order.
    """
    with Path(path).open(encoding="utf-8") as f:
        return [json.loads(line) for line in f if line.strip()]
