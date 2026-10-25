"""Causal one-image MAE observations and prefix/online next-state regression."""

from __future__ import annotations

import copy
from contextlib import contextmanager
from typing import Any

import numpy as np
import torch
from sklearn.linear_model import LogisticRegression
from sklearn.neighbors import KNeighborsClassifier

from cafl4ds.prediction_benchmark import health_values

HEALTH = ("rank", "variance", "covariance", "cka", "cosine", "feature_norm", "bank_reconstruction")
READERS = ("linear", "knn", "finetune20")


@contextmanager
def observer(model: torch.nn.Module, seed: int):
    """Restore RNG and every module mode, including on exceptions."""
    modes = {n: m.training for n, m in model.named_modules()}
    device = next(model.parameters()).device
    devices = [device.index or 0] if device.type == "cuda" else []
    try:
        with torch.random.fork_rng(devices=devices):
            torch.manual_seed(seed)
            model.eval()
            yield
    finally:
        for n, m in model.named_modules():
            m.training = modes[n]


def single_step(model, optimizer, image: torch.Tensor) -> tuple[float, float]:
    """Exactly one image and one real optimizer update, without accumulation."""
    if image.ndim != 4 or len(image) != 1:
        raise ValueError("Require exactly one arriving image")
    model.train()
    optimizer.zero_grad(set_to_none=True)
    loss = model.training_step(image)
    if not torch.isfinite(loss):
        raise ValueError("Nonfinite single-image SSL loss")
    loss.backward()
    norm = torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0, error_if_nonfinite=True)
    optimizer.step()
    return float(loss.detach()), float(norm)


def fixed_reconstruction(model, images: torch.Tensor, seed: int) -> np.ndarray:
    """Use fixed pixels and masks; bypass training augmentation for a stable assay."""
    device = next(model.parameters()).device
    values = []
    with observer(model, seed), torch.no_grad():
        for batch in images.split(32):
            batch = batch.to(device)
            latent, mask, restore = model.encoder.forward_encoder(batch, mask_ratio=model.mask_ratio)
            prediction = model.decoder(latent, restore)
            target = model._reconstruction_target(batch)
            losses = (prediction - target).square().mean(-1)
            values.append(((losses * mask).sum(-1) / mask.sum(-1)).cpu())
    return torch.cat(values).double().numpy()


def encode(model, images: torch.Tensor, seed: int) -> np.ndarray:
    """Read unaugmented features without changing the live learner."""
    device = next(model.parameters()).device
    with observer(model, seed), torch.no_grad():
        return torch.cat([model.encoder.embed(b.to(device)).cpu() for b in images.split(64)]).numpy()


def fit_probes(support: np.ndarray, labels: np.ndarray, query: np.ndarray) -> dict[str, np.ndarray]:
    """Fit only on the labeled support; retain full query probabilities."""
    mean, scale = support.mean(0), support.std(0)
    scale[scale < 1e-12] = 1
    linear = LogisticRegression(C=1, max_iter=1000, random_state=0)
    linear.fit((support - mean) / scale, labels)
    knn = KNeighborsClassifier(n_neighbors=min(20, len(labels)), metric="cosine", weights="distance")
    knn.fit(support, labels)
    return {"linear": linear.predict_proba((query - mean) / scale), "knn": knn.predict_proba(query)}


def fine_tune(model, support, labels, query, steps: int, seed: int) -> tuple[np.ndarray, float]:
    """Fit a fresh independent encoder copy with the established fixed recipe."""
    device = next(model.parameters()).device
    with observer(model, seed):
        encoder = copy.deepcopy(model.encoder).requires_grad_(True).train()
        head = torch.nn.Linear(encoder.embed_dim, 2).to(device)
        generator = torch.Generator().manual_seed(seed)
        optimizer = torch.optim.AdamW(
            [{"params": encoder.parameters(), "lr": 1e-5}, {"params": head.parameters(), "lr": 1e-3}],
            weight_decay=0.05,
        )
        support, labels = support.to(device), labels.to(device)
        for _ in range(steps):
            ids = torch.randperm(len(labels), generator=generator)[:32].to(device)
            optimizer.zero_grad(set_to_none=True)
            loss = torch.nn.functional.cross_entropy(head(encoder.embed(support[ids])), labels[ids])
            loss.backward()
            torch.nn.utils.clip_grad_norm_([*encoder.parameters(), *head.parameters()], 1.0, error_if_nonfinite=True)
            optimizer.step()
        encoder.eval()
        with torch.no_grad():
            probabilities = torch.cat(
                [head(encoder.embed(b.to(device))).softmax(-1).cpu() for b in query.split(64)]
            ).numpy()
            displacement = sum(
                (p - q).square().sum() for p, q in zip(encoder.parameters(), model.encoder.parameters(), strict=True)
            ).sqrt()
        return probabilities, float(displacement)


def performance(probabilities: dict[str, np.ndarray], labels: np.ndarray, domains: np.ndarray) -> dict[str, float]:
    """Keep exact accuracy counts, with continuous CE and separate domains."""
    result = {}
    for reader, p in probabilities.items():
        if p.shape != (len(labels), 2) or not np.isfinite(p).all():
            raise ValueError("Invalid reader probabilities")
        correct = p.argmax(1) == labels
        losses = -np.log(np.clip(p[np.arange(len(labels)), labels], 1e-12, 1))
        for domain, mask in (("balanced", np.ones(len(labels), bool)), ("day", domains == 0), ("night", domains == 1)):
            result[f"{reader}_accuracy_{domain}"] = 100 * float(correct[mask].mean())
            result[f"{reader}_ce_{domain}"] = float(losses[mask].mean())
    if "finetune20" in probabilities:
        for domain in ("balanced", "day", "night"):
            result[f"finetune_gain_{domain}"] = (
                result[f"finetune20_accuracy_{domain}"] - result[f"linear_accuracy_{domain}"]
            )
    return result


def observe_state(model, bank, warm, panel, seed: int, full: bool = False) -> tuple[dict, dict]:
    """Measure fixed-bank health and copied-reader performance at one state."""
    z = encode(model, bank, seed)
    values = health_values(z, warm)
    values["feature_norm"] = float(np.linalg.norm(z.astype(float), axis=1).mean())
    values["bank_reconstruction"] = float(fixed_reconstruction(model, bank, seed).mean())
    images = torch.cat((panel["support_images"], panel["query_images"]))
    encoded = encode(model, images, seed)
    count = len(panel["support_labels"])
    probabilities = fit_probes(encoded[:count], panel["support_labels"].numpy(), encoded[count:])
    probabilities["finetune20"], displacement = fine_tune(
        model, panel["support_images"], panel["support_labels"], panel["query_images"], 20, 51036
    )
    values.update(performance(probabilities, panel["query_labels"], panel["query_domains"]))
    values["finetune_encoder_displacement"] = displacement
    full_metrics = None
    if full:
        probabilities["finetune100"], _ = fine_tune(
            model, panel["support_images"], panel["support_labels"], panel["query_images"], 100, 51036
        )
        full_metrics = performance(
            {"finetune100": probabilities["finetune100"]}, panel["query_labels"], panel["query_domains"]
        )
    return values, {"bank_features": z, "probabilities": probabilities, "full_metrics": full_metrics}


def image_features(model, image, previous, centroid: np.ndarray, projection: np.ndarray, seed: int):
    """Only current pixels and pre-update model information; no preview update."""
    x = image.cpu().numpy().astype(float)
    z = encode(model, image, seed)[0].astype(float)
    normalized = z / max(np.linalg.norm(z), 1e-12)
    means, deviations = x.mean((0, 2, 3)), x.std((0, 2, 3))
    edge = np.abs(np.diff(x, axis=2)).mean() + np.abs(np.diff(x, axis=3)).mean()
    loss = float(fixed_reconstruction(model, image, seed + 1)[0])
    values = np.r_[
        means,
        deviations,
        edge,
        0 if previous is None else np.abs(x - previous.cpu().numpy()).mean(),
        loss,
        np.linalg.norm(z),
        np.linalg.norm(normalized - centroid),
        normalized @ projection,
    ]
    return values, loss


class PrefixRidge:
    """Fixed prefix scaling and prequential recursive least squares, multi-output."""

    def __init__(self, x: np.ndarray, y: np.ndarray, penalty: float = 10.0):
        if x.ndim != 2 or y.ndim != 2 or len(x) != len(y) or len(x) < 2:
            raise ValueError("Require aligned two-dimensional training arrays")
        if not np.isfinite(x).all() or not np.isfinite(y).all() or penalty <= 0:
            raise ValueError("Finite data and positive ridge penalty required")
        self.center, self.scale = x.mean(0), x.std(0)
        self.scale[self.scale < 1e-12] = 1
        self.ycenter, self.yscale = y.mean(0), y.std(0)
        self.yscale[self.yscale < 1e-12] = 1
        a = self.design(x)
        prior = np.diag(np.r_[1e-8, np.full(x.shape[1], penalty)])
        self.inverse = np.linalg.inv(a.T @ a + prior)
        self.beta = self.inverse @ a.T @ ((y - self.ycenter) / self.yscale)
        self.seen = len(x)

    def design(self, x):
        return np.column_stack((np.ones(len(np.atleast_2d(x))), (np.atleast_2d(x) - self.center) / self.scale))

    def predict(self, x):
        return self.ycenter + (self.design(x) @ self.beta) * self.yscale

    def update(self, x, y):
        """Call only after recording the forecast and observing its target."""
        a = self.design(x)[0]
        v = self.inverse @ a
        gain = v / (1 + a @ v)
        residual = (np.asarray(y) - self.ycenter) / self.yscale - a @ self.beta
        self.beta += np.outer(gain, residual)
        self.inverse -= np.outer(gain, v)
        self.inverse = (self.inverse + self.inverse.T) / 2
        self.seen += 1


def designs(before: np.ndarray, after: np.ndarray, image: np.ndarray, metrics: list[str], training: np.ndarray):
    """At row t every historical difference ends at t-1, never at the target."""
    delta = after - before
    lag = np.vstack((np.zeros((1, delta.shape[1])), delta[:-1]))
    average = np.array([delta[max(0, i - 4) : i].mean(0) if i else np.zeros(delta.shape[1]) for i in range(len(delta))])
    previous_training = np.vstack((np.zeros((1, 2)), training[:-1]))
    age = np.arange(len(delta))[:, None]
    state = np.column_stack((before, age, previous_training))
    history = np.column_stack((state, lag, average))
    health = [metrics.index(m) for m in HEALTH]
    return {
        "state": state,
        "history": history,
        "history_image": np.column_stack((history, image)),
        "label_free_image": np.column_stack(
            (before[:, health], lag[:, health], average[:, health], age, previous_training, image)
        ),
    }


def forecasts(before, after, image, metrics, training, prefix: int = 64, penalty: float = 10.0):
    """Freeze on the first N transitions; online updates follow predictions."""
    if not 4 <= prefix < len(before):
        raise ValueError("A nonempty prefix and later test period are required")
    y = after - before
    inputs = designs(before, after, image, metrics, training)
    output = {
        "persistence": np.zeros_like(y[prefix:]),
        "last_change": y[prefix - 1 : -1].copy(),
        "prefix_mean": np.broadcast_to(y[:prefix].mean(0), y[prefix:].shape).copy(),
        "online_mean": np.array([y[:i].mean(0) for i in range(prefix, len(y))]),
    }
    for name, x in inputs.items():
        frozen = PrefixRidge(x[:prefix], y[:prefix], penalty)
        live = copy.deepcopy(frozen)
        output["offline_" + name] = frozen.predict(x[prefix:])
        rows = []
        for i in range(prefix, len(y)):
            rows.append(live.predict(x[i])[0])
            live.update(x[i], y[i])
        output["online_" + name] = np.array(rows)
    if not all(np.isfinite(p).all() for p in output.values()):
        raise ValueError("Nonfinite forecast")
    return y[prefix:], output


def error_summary(y: np.ndarray, prediction: np.ndarray, tolerance: float = 1e-10) -> dict[str, Any]:
    """Native-unit errors and directions; distinguish ties from decreases."""
    error = prediction - y
    actual_sign = np.where(np.abs(y) <= tolerance, 0, np.sign(y))
    predicted_sign = np.where(np.abs(prediction) <= tolerance, 0, np.sign(prediction))
    changed = actual_sign != 0
    classes = np.unique(actual_sign)
    return {
        "mse": float(np.mean(error**2)),
        "rmse": float(np.sqrt(np.mean(error**2))),
        "mae": float(np.mean(np.abs(error))),
        "direction_accuracy": float(np.mean(actual_sign == predicted_sign)),
        "changed_direction_accuracy": float(np.mean(actual_sign[changed] == predicted_sign[changed]))
        if changed.any()
        else None,
        "balanced_direction_recall": float(np.mean([np.mean(predicted_sign[actual_sign == k] == k) for k in classes])),
        "tie_fraction": float(np.mean(~changed)),
        "n": len(y),
    }
