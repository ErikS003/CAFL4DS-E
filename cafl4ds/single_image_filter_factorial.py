"""Task-informed singleton features and frozen objective-by-feature forecasts."""

from __future__ import annotations

import numpy as np
import torch
import torch.nn.functional as F
from sklearn.linear_model import LogisticRegression
from sklearn.neighbors import KNeighborsClassifier

from cafl4ds.single_image_filter import FrozenRegressor, features
from cafl4ds.single_image_forecast import encode, image_features, observer
from cafl4ds.single_image_signal import optimizer_features, performance64

TARGETS = ("J", "probe")
DESIGNS = ("state", "base", "rich")
MAIN = ("J_base", "J_rich", "probe_base", "probe_rich")
ARMS = (*MAIN, "J_state", "probe_state", *("random_" + arm for arm in MAIN), "full", "hold")


def linear_readout(model, panel, draws, seed):
    """Refit the established support-only probes; expose fixed heads for derivatives."""
    query = encode(model, panel["query_images"], seed)
    probabilities, heads, rows = [], [], []
    for draw in draws:
        z = encode(model, panel["support_images"][draw], seed)
        labels = panel["support_labels"][draw].numpy()
        center, scale = z.mean(0), z.std(0)
        scale[scale < 1e-12] = 1
        reader = LogisticRegression(C=1, max_iter=1000, random_state=0)
        reader.fit((z - center) / scale, labels)
        linear = reader.predict_proba((query - center) / scale)
        knn = KNeighborsClassifier(n_neighbors=min(20, len(labels)), metric="cosine", weights="distance")
        knn.fit(z, labels)
        near = knn.predict_proba(query)
        probabilities.append(np.stack([linear, near]))
        weight = reader.coef_[0] / scale
        bias = reader.intercept_[0] - weight @ center
        heads.append(np.r_[weight, bias])
        rows.append(performance64(dict(linear=linear, knn=near), panel["query_labels"], panel["query_domains"]))
    return dict(
        metrics={k: float(np.mean([r[k] for r in rows])) for k in rows[0]},
        panels=rows,
        probabilities=np.asarray(probabilities),
        heads=np.asarray(heads),
    )


def reference_gradient(model, panel, heads, seed):
    """Fixed-current-head CE gradient on 8 development images per scene/light cell."""
    indices = np.concatenate(
        [
            np.flatnonzero((panel["query_labels"] == label) & (panel["query_domains"] == domain))[:8]
            for label in (0, 1)
            for domain in (0, 1)
        ]
    )
    assert len(indices) == 32
    parameters = [p for p in model.parameters() if p.requires_grad]
    device = parameters[0].device
    head = torch.as_tensor(heads, device=device, dtype=parameters[0].dtype)
    with observer(model, seed):
        z = model.encoder.embed(panel["query_images"][indices].to(device))
        logits = z @ head[:, :-1].T + head[:, -1]
        labels = torch.as_tensor(panel["query_labels"][indices], device=device, dtype=z.dtype)
        loss = F.binary_cross_entropy_with_logits(logits, labels[:, None].expand_as(logits))
        gradients = torch.autograd.grad(loss, parameters, allow_unused=True)
    return torch.cat(
        [
            (torch.zeros_like(p) if g is None else g.detach()).flatten()
            for p, g in zip(parameters, gradients, strict=True)
        ]
    )


def adamw_displacement(optimizer, parameters, gradients, max_norm=1.0):
    """Predict the actual clipped AdamW displacement without changing any tensor."""
    available = [g for g in gradients if g is not None]
    norm = torch.linalg.vector_norm(torch.stack([torch.linalg.vector_norm(g.detach(), 2) for g in available]), 2)
    clip = torch.clamp(max_norm / (norm + 1e-6), max=1.0)
    lookup = {id(p): g for p, g in zip(parameters, gradients, strict=True)}
    displacements = {}
    for group in optimizer.param_groups:
        if group.get("amsgrad", False) or group.get("maximize", False):
            raise ValueError("Only the frozen ordinary AdamW recipe is supported")
        b1, b2 = group["betas"]
        for p in group["params"]:
            g = lookup[id(p)]
            if g is None:
                displacements[id(p)] = torch.zeros_like(p)
                continue
            g = g.detach() * clip
            state = optimizer.state.get(p, {})
            first = state.get("exp_avg", torch.zeros_like(p))
            second = state.get("exp_avg_sq", torch.zeros_like(p))
            step = float(state.get("step", 0)) + 1
            first_new = first.lerp(g, 1 - b1)
            second_new = second * b2 + g.square() * (1 - b2)
            denominator = second_new.sqrt() / np.sqrt(1 - b2**step) + group["eps"]
            new = p.detach() * (1 - group["lr"] * group["weight_decay"])
            new = new.addcdiv(first_new, denominator, value=-group["lr"] / (1 - b1**step))
            displacements[id(p)] = new - p.detach()
    return torch.cat([displacements[id(p)].flatten() for p in parameters]), norm


def update_features(model, optimizer, image, seed, task_gradient):
    """Pay for a singleton backward pass; never perform a hypothetical update."""
    named = [(n, p) for n, p in model.named_parameters() if p.requires_grad]
    parameters = [p for _, p in named]
    with observer(model, seed):
        model.train()
        loss = model.training_step(image.to(parameters[0].device))
        gradients = torch.autograd.grad(loss, parameters, allow_unused=True)
        delta, norm = adamw_displacement(optimizer, parameters, gradients)
        g = torch.cat(
            [
                (torch.zeros_like(p) if v is None else v.detach()).flatten()
                for p, v in zip(parameters, gradients, strict=True)
            ]
        )
        first = torch.cat(
            [optimizer.state.get(p, {}).get("exp_avg", torch.zeros_like(p)).flatten() for p in parameters]
        )
        second = torch.cat(
            [optimizer.state.get(p, {}).get("exp_avg_sq", torch.zeros_like(p)).flatten() for p in parameters]
        )
        encoder_square = sum(
            v.detach().square().sum()
            for (name, _), v in zip(named, gradients, strict=True)
            if name.startswith("encoder.") and v is not None
        )
        eps = optimizer.param_groups[0]["eps"]
        dot = task_gradient @ delta
        result = np.array(
            [
                float(loss.detach()),
                float(norm),
                float((g @ first) / (g.norm() * first.norm()).clamp_min(1e-20)),
                float((g / (second.sqrt() + eps)).norm()),
                float(encoder_square.sqrt() / norm.clamp_min(1e-20)),
                float(g.abs().mean()),
                float(delta.norm()),
                float(dot),
                float(dot / (delta.norm() * task_gradient.norm()).clamp_min(1e-20)),
            ]
        )
    return result, delta


def feature_vectors(
    model,
    optimizer,
    arrivals,
    origin,
    observations,
    centroid,
    projection,
    cluster_ids,
    settings,
    draw_seed,
    task_gradient=None,
    rich=False,
):
    current = observations[-1]
    lag = current[0] - observations[-2][0]
    pixel, _ = image_features(
        model,
        arrivals[origin : origin + 1],
        arrivals[origin - 1 : origin],
        centroid,
        projection,
        settings["feature_seed"],
    )
    state = features(current[0], lag, optimizer_features(model, optimizer), origin, settings["frames_per_clip"])
    result = dict(state=state, base=np.r_[state, pixel])
    if rich:
        assert task_gradient is not None
        z = encode(model, arrivals[origin : origin + 1], settings["feature_seed"])[0].astype(float)
        z /= max(np.linalg.norm(z), 1e-12)
        normalized = current[1].astype(float)
        normalized /= np.maximum(np.linalg.norm(normalized, axis=1, keepdims=True), 1e-12)
        distances, reconstruction = [], []
        for cluster in range(settings["clusters"]):
            take = cluster_ids == cluster
            center = normalized[take].mean(0)
            center /= max(np.linalg.norm(center), 1e-12)
            distances.append(1 - z @ center)
            reconstruction += [current[2][take].mean(), np.quantile(current[2][take], 0.9)]
        recent = encode(model, arrivals[max(0, origin - 8) : origin], settings["feature_seed"]).astype(float)
        recent /= np.maximum(np.linalg.norm(recent, axis=1, keepdims=True), 1e-12)
        similarities = recent @ z
        older = [observations[-2][0] - observations[-3][0], observations[-3][0] - observations[-4][0]]
        average = np.mean([lag, *older], axis=0)
        update, _ = update_features(model, optimizer, arrivals[origin : origin + 1], draw_seed, task_gradient)
        result["rich"] = np.r_[
            result["base"],
            distances,
            reconstruction,
            1 - similarities.max(),
            1 - similarities.mean(),
            np.concatenate(older),
            average,
            update,
        ]
    for value in result.values():
        assert np.isfinite(value).all()
    return result


def admission_threshold(predictions, rate):
    """Freeze a development-only threshold; never admit a predicted worsening."""
    p = np.asarray(predictions, float)
    if not len(p) or not np.isfinite(p).all() or not 0 < rate < 1:
        raise ValueError("Finite development predictions and interior budget required")
    return float(min(0, np.quantile(p, rate)))


def fit_factorial(a, settings):
    """Whole-development-history validation followed by frozen refitting."""
    valid = np.isin(a["histories"], settings["validation_histories"])
    assert valid.any() and (~valid).any()
    models, selection = {}, {}
    for ti, target in enumerate(TARGETS):
        y = a["targets"][:, ti]
        for design in DESIGNS:
            x = a[design]
            predictions, losses = {}, {}
            for family in settings["regressors"]:
                predicted = np.zeros(valid.sum())
                for seed in settings["trajectory_seeds"]:
                    train = ~valid & (a["seeds"] == seed)
                    ask = valid & (a["seeds"] == seed)
                    model = FrozenRegressor(family, x[train], y[train], settings)
                    predicted[a["seeds"][valid] == seed] = model.predict(x[ask])
                predictions[family] = predicted
                losses[family] = float(np.mean((predicted - y[valid]) ** 2))
            winner = min(losses, key=losses.get)
            p = predictions[winner]
            threshold = admission_threshold(p, settings["max_validation_admission"])
            key = target + "_" + design
            models[key] = {
                int(seed): FrozenRegressor(winner, x[a["seeds"] == seed], y[a["seeds"] == seed], settings)
                for seed in settings["trajectory_seeds"]
            }
            selection[key] = dict(
                family=winner,
                validation_mse=losses,
                threshold=threshold,
                validation_rmse=float(np.sqrt(np.mean((p - y[valid]) ** 2))),
                validation_mae=float(np.mean(np.abs(p - y[valid]))),
                zero_rmse=float(np.sqrt(np.mean(y[valid] ** 2))),
                zero_mae=float(np.mean(np.abs(y[valid]))),
                validation_admission=float(np.mean(p < threshold)),
                validation_predictions=p.tolist(),
                validation_targets=y[valid].tolist(),
                rows=len(y),
                validation_rows=int(valid.sum()),
            )
    return models, selection
