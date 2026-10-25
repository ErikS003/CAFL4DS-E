"""Deterministic, label-free full-group replay choices for counterfactual studies."""

from __future__ import annotations

import numpy as np
import torch
import torch.nn.functional as F
from sklearn.cluster import KMeans
from threadpoolctl import threadpool_limits

RULES = ("uniform", "recent", "age_balanced", "cluster_balanced")
CLUSTER_SEED = 261017


def structured_choices(arrivals: list[int], embeddings: torch.Tensor) -> tuple[dict[str, list[int]], list[int]]:
    """Return sixteen shortlist positions per rule, with fixed membership ordering."""
    if len(arrivals) != 64 or len(set(arrivals)) != 64 or embeddings.ndim != 2 or len(embeddings) != 64:
        raise ValueError("Require64 distinct arrival times and64 embeddings")
    if not torch.isfinite(embeddings).all():
        raise ValueError("Nonfinite embeddings")
    by_age = sorted(range(64), key=lambda i: arrivals[i])
    recent = sorted(by_age[-16:])
    balanced = sorted(i for group in np.array_split(by_age, 4) for i in sorted(group.tolist())[:4])
    z = F.normalize(embeddings.double(), dim=1).cpu().numpy()
    with threadpool_limits(limits=1):
        model = KMeans(n_clusters=8, n_init=10, random_state=CLUSTER_SEED).fit(z)
    labels = model.labels_.tolist()
    groups = [
        sorted(
            [i for i in range(64) if labels[i] == c],
            key=lambda i: (float(np.sum((z[i] - model.cluster_centers_[c]) ** 2)), i),
        )
        for c in range(8)
    ]
    cluster: list[int] = []
    depth = 0
    while len(cluster) < 16:
        for group in groups:
            if depth < len(group) and len(cluster) < 16:
                cluster.append(group[depth])
        depth += 1
    return {
        "uniform": list(range(16)),
        "recent": recent,
        "age_balanced": balanced,
        "cluster_balanced": sorted(cluster),
    }, labels
