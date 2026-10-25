"""Paired image-loss interventions and read-only health derivatives for MAE."""

from __future__ import annotations

import torch
import torch.nn.functional as F


def image_step(model, optimizer, images, omit=None, reconstruct=False):
    """Keep every view/mask fixed and zero one loss without renormalizing others."""
    model.train()
    optimizer.zero_grad()
    losses, mask = model._masked_recon_loss(images)
    per_image = (losses * mask).sum(-1) / mask.sum(-1).clamp_min(1)
    weights = torch.ones_like(per_image)
    if omit is not None:
        weights[omit] = 0
    # Keep the ordinary MAE reduction order for the all-included control.
    objective = (losses * mask * weights[:, None]).sum() / mask.sum().clamp_min(1)
    if not torch.isfinite(objective):
        raise ValueError("Nonfinite image-weighted loss")
    if reconstruct:
        if omit is None:
            raise ValueError("Gradient subtraction control requires omitted slot")
        parameters = [p for p in model.parameters() if p.requires_grad]
        full = torch.autograd.grad(per_image.mean(), parameters, retain_graph=True, allow_unused=True)
        one = torch.autograd.grad(per_image[omit] / len(images), parameters, allow_unused=True)
        for p, a, b in zip(parameters, full, one, strict=True):
            p.grad = None if a is None else a - (b if b is not None else 0)
    else:
        objective.backward()
    norm = torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0, error_if_nonfinite=True)
    optimizer.step()
    return {
        "loss": float(objective.detach()),
        "gradient_norm": float(norm),
        "per_image_losses": per_image.detach().cpu().tolist(),
    }


def variance_gradient(model, bank):
    """Derivative of fixed-bank normalized variance; preserve modes, RNG and .grad."""
    parameters = [p for p in model.encoder.parameters() if p.requires_grad]
    modes = {n: m.training for n, m in model.named_modules()}
    device = next(model.parameters()).device
    try:
        with torch.random.fork_rng(devices=[device.index or 0] if device.type == "cuda" else []):
            model.eval()
            z = torch.cat([model.encoder.embed(b.to(device)) for b in bank.split(32)]).double()
            value = F.normalize(z, dim=1).var(0, unbiased=True).mean()
            gradients = torch.autograd.grad(value, parameters, allow_unused=True)
    finally:
        for name, module in model.named_modules():
            module.training = modes[name]
    return [
        torch.zeros_like(p) if g is None else g.detach() for p, g in zip(parameters, gradients, strict=True)
    ], float(value.detach())


def gradient_forecast(gradients, after, baseline, variance):
    """First-order signed variance response to a known optimizer displacement."""
    names = [n for n, p in after.encoder.named_parameters() if p.requires_grad]
    total = sum(
        float(
            (
                g.double()
                * (
                    dict(after.encoder.named_parameters())[n].detach().double()
                    - baseline["encoder." + n].to(g.device).double()
                )
            ).sum()
        )
        for n, g in zip(names, gradients, strict=True)
    )
    return total / variance
