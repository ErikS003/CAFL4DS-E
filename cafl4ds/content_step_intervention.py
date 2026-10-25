"""Content-by-realized-encoder-step intervention and fixed contrasts."""

from __future__ import annotations

import numpy as np
import torch

DIETS = ("current", "balanced")
SCALES = (1.0, 0.5)
CONTRASTS = (
    "current_half_minus_full",
    "balanced_half_minus_full",
    "replay_minus_current_full",
    "replay_minus_current_half",
    "interaction",
)


def displacement(parameters, before):
    return float(
        torch.linalg.vector_norm(
            torch.cat([(p.detach().double() - a.double()).flatten() for p, a in zip(parameters, before, strict=True)])
        )
    )


@torch.no_grad()
def set_encoder_step(parameters, before, target):
    """Rescale the realized Adam displacement; do not rewrite Adam or SSL heads."""
    parameters = list(parameters)
    if not np.isfinite(target) or target <= 0:
        raise ValueError("Positive finite target required")
    raw = displacement(parameters, before)
    if not np.isfinite(raw) or raw <= 0:
        raise ValueError("Cannot normalize a zero/nonfinite proposed step")
    factor = target / raw
    for p, a in zip(parameters, before, strict=True):
        # Form in double precision, round only the final parameter to its dtype.
        p.copy_(a.double() + factor * (p.double() - a.double()))
    actual = displacement(parameters, before)
    assert abs(actual - target) <= max(1e-9, 2e-5 * target), (actual, target)
    return raw, actual, factor


def save_xor(path, base, state):
    if list(base) != list(state):
        raise ValueError("Aligned state keys required")
    arrays = {}
    for i, key in enumerate(base):
        a = base[key].detach().cpu().contiguous().numpy()
        b = state[key].detach().cpu().contiguous().numpy()
        if a.shape != b.shape or a.dtype != b.dtype:
            raise ValueError("Aligned state tensors required")
        arrays[str(i)] = np.bitwise_xor(a.view(np.uint8).reshape(-1), b.view(np.uint8).reshape(-1))
    np.savez_compressed(path, **arrays)


def load_xor(path, base):
    result = {}
    with np.load(path, allow_pickle=False) as z:
        assert len(z.files) == len(base)
        for i, (key, value) in enumerate(base.items()):
            a = value.detach().cpu().contiguous().numpy()
            raw = np.bitwise_xor(a.view(np.uint8).reshape(-1), z[str(i)])
            result[key] = torch.from_numpy(raw.view(a.dtype).reshape(a.shape).copy())
    return result


def contrasts(gains):
    """Input [procedure,source,context,seed,diet,magnitude,support,component]."""
    g = np.asarray(gains)
    if g.ndim != 8 or g.shape[4:6] != (2, 2):
        raise ValueError("Require crossed diet and magnitude factors")
    current = g[:, :, :, :, 0, 1] - g[:, :, :, :, 0, 0]
    replay = g[:, :, :, :, 1, 1] - g[:, :, :, :, 1, 0]
    full = g[:, :, :, :, 1, 0] - g[:, :, :, :, 0, 0]
    half = g[:, :, :, :, 1, 1] - g[:, :, :, :, 0, 1]
    return np.stack([current, replay, full, half, half - full])


def conditional_samples(delta, draws=2000, seed=194042):
    """Common support/query resampling; delta [contrast,procedure,support,2048]."""
    if delta.ndim != 4 or delta.shape[-2:] != (12, 2048):
        raise ValueError("Expected12 supports and2048 ordered queries")
    rng = np.random.default_rng(seed)
    sw = rng.multinomial(12, np.full(12, 1 / 12), size=draws) / 12
    qw = [rng.multinomial(512, np.full(512, 1 / 512), size=draws) / 512 for _ in range(4)]
    values = np.empty((draws, *delta.shape[:2]))
    for k in range(delta.shape[0]):
        for p in range(delta.shape[1]):
            values[:, k, p] = np.sum((qw[3] @ delta[k, p, :, 1536:].T) * sw, axis=1)
    return values, sw, qw[3]
