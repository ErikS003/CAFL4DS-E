"""Controlled support sampling, normalization and logistic penalty diagnostics."""

from __future__ import annotations

from typing import Any

import numpy as np
import torch
from sklearn.linear_model import LogisticRegression

from cafl4ds.image_utility import FixedProbe

SUPPORT_SEEDS = (283003, 284003, 285007, 286019)
REFERENCE_CS = (0.1, 10.0)


def balanced_supports(metadata: dict[str, Any]) -> dict[str, list[int]]:
    """Sample64 per domain/class without replacement from the labeled support union."""
    pool = metadata["supports"]["pooled"]
    labels, domains = metadata["labels"], metadata["domains"]
    result = {name: indices.tolist() for name, indices in metadata["supports"].items()}
    for draw, seed in enumerate(SUPPORT_SEEDS):
        generator = torch.Generator().manual_seed(seed)
        selected = []
        for domain in (0, 1):
            for label in (0, 1, 2):
                eligible = pool[(domains[pool] == domain) & (labels[pool] == label)]
                if len(eligible) < 64:
                    raise ValueError("Insufficient balanced support")
                selected.extend(eligible[torch.randperm(len(eligible), generator=generator)[:64]].tolist())
        result[f"sample{draw}"] = sorted(selected)
    return result


def configurations() -> list[dict[str, Any]]:
    """Twenty-four fixed readers isolate known settings; no outcome-dependent tuning."""
    rows = []
    for reference_c in REFERENCE_CS:
        for support in ("support1", "support2", "support3"):
            for normalization in ("panel", "common"):
                rows.append(
                    {
                        "support": support,
                        "normalization": normalization,
                        "reference_C": reference_c,
                        "actual_C": reference_c,
                        "penalty": "matched",
                        "n_support": 384,
                    }
                )
        for penalty in ("fixed_C", "matched"):
            rows.append(
                {
                    "support": "pooled",
                    "normalization": "common",
                    "reference_C": reference_c,
                    "actual_C": reference_c if penalty == "fixed_C" else reference_c / 3,
                    "penalty": penalty,
                    "n_support": 1152,
                }
            )
        for draw in range(4):
            rows.append(
                {
                    "support": f"sample{draw}",
                    "normalization": "common",
                    "reference_C": reference_c,
                    "actual_C": reference_c,
                    "penalty": "matched",
                    "n_support": 384,
                }
            )
    for index, row in enumerate(rows):
        row["reader"] = f"reader{index:02d}"
    return rows


def fit_reader(
    z: torch.Tensor,
    labels: torch.Tensor,
    fitting: list[int],
    statistics: list[int],
    c: float,
) -> FixedProbe:
    """Only fitting labels enter the fit; statistics use declared support features."""
    x = z.numpy().astype(np.float64)
    mean, scale = x[statistics].mean(0), x[statistics].std(0)
    scale[scale < 1e-8] = 1
    classifier = LogisticRegression(C=c, tol=1e-10, max_iter=10000, random_state=63000)
    classifier.fit((x[fitting] - mean) / scale, labels[fitting].numpy())
    if int(classifier.n_iter_.max()) >= 10000:
        raise ValueError("Head solver reached its iteration limit")
    return FixedProbe(mean, scale, classifier)
