"""Controlled singleton responses and causally matured residual forecasts."""

from __future__ import annotations

import copy
import hashlib
from typing import Any

import numpy as np
import torch

from cafl4ds.prediction_benchmark import health_values
from cafl4ds.single_image_forecast import (
    HEALTH,
    PrefixRidge,
    encode,
    fine_tune,
    fit_probes,
    fixed_reconstruction,
    observer,
    single_step,
)


def training_seed(history: str, arrival: int) -> int:
    """Give each arrival its own random draw, independent of previous admissions."""
    value = hashlib.sha256(f"signal-5003-{history}-{arrival}".encode()).digest()
    return int.from_bytes(value[:4], "little") % (2**31 - 1)


def seeded_step(model: Any, optimizer: Any, image: torch.Tensor, seed: int) -> tuple[float, float]:
    """Use the declared arrival/view seed for one genuine singleton update."""
    torch.manual_seed(seed)
    return single_step(model, optimizer, image)


def zero_step(model: Any, optimizer: Any) -> None:
    """Advance momentum/decay with explicit zeros; never silently skip with None."""
    model.train()
    optimizer.zero_grad(set_to_none=True)
    for parameter in model.parameters():
        if parameter.requires_grad:
            parameter.grad = torch.zeros_like(parameter)
    optimizer.step()


def optimizer_features(model: Any, optimizer: Any) -> np.ndarray:
    """Summarize the pre-update state without labels or a hypothetical update."""
    parameters = [p for p in model.parameters() if p.requires_grad]
    states = [optimizer.state.get(p, {}) for p in parameters]
    weights = torch.cat([p.detach().flatten() for p in parameters])
    first = torch.cat(
        [s.get("exp_avg", torch.zeros_like(p)).flatten() for p, s in zip(parameters, states, strict=True)]
    )
    second = torch.cat(
        [s.get("exp_avg_sq", torch.zeros_like(p)).flatten() for p, s in zip(parameters, states, strict=True)]
    )
    steps = [float(s.get("step", 0)) for s in states]
    return np.array(
        [
            float(weights.norm()),
            float(first.norm()),
            float(second.norm()),
            float(second.mean()),
            float(first.abs().mean()),
            float(np.mean(steps)),
        ]
    )


def gradient_features(model: Any, optimizer: Any, image: torch.Tensor, seed: int) -> np.ndarray:
    """Pay for a pre-update backward pass on the actual training view, preserving .grad."""
    named = [(n, p) for n, p in model.named_parameters() if p.requires_grad]
    with observer(model, seed):
        model.train()
        loss = model.training_step(image)
        gradients = torch.autograd.grad(loss, [p for _, p in named], allow_unused=True)
        pieces = [(name, p, g.detach()) for (name, p), g in zip(named, gradients, strict=True) if g is not None]
        g = torch.cat([v.flatten() for _, _, v in pieces])
        m = torch.cat([optimizer.state.get(p, {}).get("exp_avg", torch.zeros_like(p)).flatten() for _, p, _ in pieces])
        v = torch.cat(
            [optimizer.state.get(p, {}).get("exp_avg_sq", torch.zeros_like(p)).flatten() for _, p, _ in pieces]
        )
        norm = g.norm()
        encoder_norm = torch.cat([v.flatten() for name, _, v in pieces if name.startswith("encoder.")]).norm()
        cosine = (g @ m) / (norm * m.norm()).clamp_min(1e-20)
        preconditioned = (g / (v.sqrt() + optimizer.param_groups[0]["eps"])).norm()
        return np.array(
            [
                float(loss.detach()),
                float(norm),
                float(cosine),
                float(preconditioned),
                float(encoder_norm / norm.clamp_min(1e-20)),
                float(g.abs().mean()),
            ]
        )


def performance64(probabilities: dict[str, np.ndarray], labels: np.ndarray, domains: np.ndarray) -> dict[str, float]:
    """Score exact correct counts and float64 probability losses in each domain."""
    result = {}
    for name, values in probabilities.items():
        p = np.asarray(values, dtype=np.float64)
        correct = p.argmax(1) == labels
        losses = -np.log(np.clip(p[np.arange(len(labels)), labels], 1e-12, 1))
        for domain, mask in (("balanced", np.ones(len(labels), bool)), ("day", domains == 0), ("night", domains == 1)):
            result[f"{name}_accuracy_{domain}"] = 100 * float(correct[mask].mean())
            result[f"{name}_ce_{domain}"] = float(losses[mask].mean())
    return result


def read_state(
    model: Any,
    bank: torch.Tensor,
    warm: np.ndarray,
    panel: dict,
    draws: list[int],
    steps: tuple[int, ...] = (100,),
    seed: int = 81231,
) -> tuple[dict, dict]:
    """Measure invariant geometry and independent copied readers, never the live optimizer."""
    z = encode(model, bank, seed)
    health = health_values(z, warm)
    health["feature_norm"] = float(np.linalg.norm(z.astype(float), axis=1).mean())
    reconstruction = fixed_reconstruction(model, bank, seed)
    health["bank_reconstruction"] = float(reconstruction.mean())
    # Keep inference batch composition independent of the requested panel count.
    query_features = encode(model, panel["query_images"], seed)
    metrics, probabilities = {}, {}
    for draw in draws:
        support_features = encode(model, panel["support_images"][draw], seed)
        p = fit_probes(support_features, panel["support_labels"][draw].numpy(), query_features)
        for count in steps:
            p[f"finetune{count}"], _ = fine_tune(
                model,
                panel["support_images"][draw],
                panel["support_labels"][draw],
                panel["query_images"],
                count,
                draw + 25,
            )
        metrics[str(draw)] = dict(health, **performance64(p, panel["query_labels"], panel["query_domains"]))
        probabilities.update({f"{draw}_{name}": a for name, a in p.items()})
    return metrics, {"bank_features": z, "bank_reconstruction": reconstruction, "probabilities": probabilities}


def feature_sets(
    before: np.ndarray,
    last_delta: np.ndarray,
    image: np.ndarray,
    optimizer: np.ndarray,
    gradient: np.ndarray,
    age: np.ndarray,
    metrics: list[str],
) -> dict[str, np.ndarray]:
    """All quantities refer to the parent state and currently available image only."""
    history = np.column_stack((before, last_delta, age))
    health = [metrics.index(m) for m in HEALTH]
    return {
        "history": history,
        "optimizer": np.column_stack((history, optimizer)),
        "image": np.column_stack((history, image)),
        "optimizer_image": np.column_stack((history, optimizer, image)),
        "paid_gradient": np.column_stack((history, optimizer, image, gradient)),
        "label_free_paid": np.column_stack((before[:, health], last_delta[:, health], age, optimizer, image, gradient)),
    }


def residual_forecasts(
    x: np.ndarray,
    y: np.ndarray,
    baseline: np.ndarray,
    origins: np.ndarray,
    ends: np.ndarray,
    held: np.ndarray,
    prefix: int = 512,
    test_start: int = 544,
    penalty: float = 10.0,
) -> tuple[dict, dict]:
    """Fit only matured labels; online outcomes enter before their next eligible origin."""
    calibration = ends <= prefix
    transfer = calibration & ~held
    test = held & (origins >= test_start)
    if not transfer.any() or not (held & calibration).any() or not test.any():
        raise ValueError("Development prefixes, held prefixes and later held tests required")
    residual = y - baseline
    frozen = PrefixRidge(x[calibration], residual[calibration], penalty)
    source = PrefixRidge(x[transfer], residual[transfer], penalty)
    online = copy.deepcopy(frozen)
    prediction = np.zeros_like(y)
    admitted = calibration.copy()
    counts = {}
    for t in sorted(set(origins[test])):
        mature = np.flatnonzero(held & ~admitted & (ends <= t))
        for row in sorted(mature, key=lambda i: (ends[i], i)):
            online.update(x[row], residual[row])
            admitted[row] = True
        selected = test & (origins == t)
        prediction[selected] = baseline[selected] + online.predict(x[selected])
        counts[str(int(t))] = online.seen
    return {
        "transfer": baseline[test] + source.predict(x[test]),
        "frozen": baseline[test] + frozen.predict(x[test]),
        "online": prediction[test],
    }, {
        "calibration_rows": int(calibration.sum()),
        "transfer_rows": int(transfer.sum()),
        "online_fit_rows_by_origin": counts,
        "test_indices": np.flatnonzero(test).tolist(),
    }


def centered_repeatability(effects: np.ndarray) -> dict[str, Any]:
    """Separate candidate contrasts from parent means before comparing draws/readers."""
    # parent, candidate, training draw, reader panel
    if effects.ndim != 4 or effects.shape[2:] != (3, 3):
        raise ValueError("Require three training draws and three reader panels")
    centered = effects - effects.mean(axis=1, keepdims=True)
    between = effects.mean(axis=2).var(axis=1, ddof=1).mean()
    within = effects.var(axis=2, ddof=1).mean()
    raw_signal = between - within / 3

    def corr(a, b):
        a, b = a.flatten(), b.flatten()
        return float(np.corrcoef(a, b)[0, 1]) if a.std() > 1e-14 and b.std() > 1e-14 else None

    draw_correlations = [
        corr(centered[:, :, d].mean(-1), np.delete(centered, d, axis=2).mean((2, 3))) for d in range(3)
    ]
    reader_correlations = [
        corr(centered[:, :, :, p].mean(-1), np.delete(centered, p, axis=3).mean((2, 3))) for p in range(3)
    ]
    return {
        "between_candidate_variance_of_draw_means": float(between),
        "within_candidate_draw_variance": float(within),
        "noise_corrected_candidate_variance": float(raw_signal),
        "leave_one_draw_out_correlations": draw_correlations,
        "leave_one_reader_out_correlations": reader_correlations,
    }
