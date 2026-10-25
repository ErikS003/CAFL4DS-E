"""Matched downstream intervention: zero encoder LR, full joint clipping."""

from __future__ import annotations

import copy
import hashlib

import numpy as np
import torch

from scripts.run_paired_update_response import fingerprint


def fit(encoder, images, labels, seed, encoder_lr=0.0):
    """Change only the backbone LR in the canonical 100-step recipe.

    Encoder gradients still affect joint clipping; this is deliberately not a
    conventional precomputed-feature linear probe. Trace reads follow each step.
    """
    original = fingerprint(encoder.state_dict())
    model = copy.deepcopy(encoder).requires_grad_(True).train()
    device = next(model.parameters()).device
    torch.manual_seed(seed)
    head = torch.nn.Linear(model.embed_dim, 2).to(device)
    initial_head = fingerprint(head.state_dict())
    generator = torch.Generator().manual_seed(seed)
    images, labels = images.to(device), labels.to(device)
    optimizer = torch.optim.AdamW(
        [{"params": model.parameters(), "lr": encoder_lr}, {"params": head.parameters(), "lr": 1e-3}],
        weight_decay=0.05,
    )
    trace, batches = [], hashlib.sha256()
    for _ in range(100):
        indices = torch.randperm(len(labels), generator=generator)[:32].to(device)
        batches.update(indices.cpu().numpy().tobytes())
        optimizer.zero_grad()
        loss = torch.nn.functional.cross_entropy(head(model.embed(images[indices])), labels[indices])
        loss.backward()
        norm = torch.nn.utils.clip_grad_norm_([*model.parameters(), *head.parameters()], 1.0, error_if_nonfinite=True)
        optimizer.step()
        trace.append([float(loss.detach()), float(norm)])
    delta = float(
        sum(
            (p.detach() - q.detach()).square().sum()
            for p, q in zip(model.parameters(), encoder.parameters(), strict=True)
        ).sqrt()
    )
    final = fingerprint(model.state_dict())
    assert fingerprint(encoder.state_dict()) == original
    assert np.isfinite(trace).all()
    if encoder_lr == 0:
        assert delta == 0 and final == original, "Encoder parameters or buffers changed"
    else:
        assert delta > 0
    return (
        model,
        head,
        {
            "steps": 100,
            "encoder_lr": encoder_lr,
            "backbone_delta_l2": delta,
            "encoder_before": original,
            "encoder_after": final,
            "initial_head": initial_head,
            "batch_order_sha256": batches.hexdigest(),
            "initial_loss": trace[0][0],
            "final_loss": trace[-1][0],
            "trace_loss_joint_gradient": trace,
        },
    )


def reference_fixed_fit(encoder, images, labels, seed):
    """Separate literal loop for exact-repeat audits (no trace machinery)."""
    model = copy.deepcopy(encoder)
    model.requires_grad_(True)
    model.train()
    device = next(model.parameters()).device
    torch.manual_seed(seed)
    head = torch.nn.Linear(model.embed_dim, 2).to(device)
    generator = torch.Generator().manual_seed(seed)
    optimizer = torch.optim.AdamW(
        [{"params": list(model.parameters()), "lr": 0.0}, {"params": list(head.parameters()), "lr": 0.001}],
        weight_decay=0.05,
    )
    x, y = images.to(device), labels.to(device)
    losses = []
    for _ in range(100):
        index = torch.randperm(y.numel(), generator=generator)[:32].to(device)
        optimizer.zero_grad()
        logits = head(model.embed(x[index]))
        value = torch.nn.functional.cross_entropy(logits, y[index])
        value.backward()
        torch.nn.utils.clip_grad_norm_(
            list(model.parameters()) + list(head.parameters()), max_norm=1.0, error_if_nonfinite=True
        )
        optimizer.step()
        losses.append(float(value.detach()))
    assert fingerprint(model.state_dict()) == fingerprint(encoder.state_dict())
    return model, head, losses


def effects(joint, fixed, labels, domains):
    """Correctness tensors [source,future,landmark(0,128,512),support,query]."""
    from cafl4ds.early_state_night_diagnostic import components

    if joint.shape != fixed.shape or joint.ndim != 5 or joint.shape[2] != 3:
        raise ValueError("Aligned parent/128/512 correctness tensors required")
    scores = np.stack([components(100.0 * v, labels, domains) for v in (joint, fixed)])
    gains = scores[:, :, :, 1:] - scores[:, :, :, :1]
    # Positive means joint fitting amplifies online-associated damage.
    interaction = gains[1] - gains[0]
    return scores, gains, interaction


def bootstrap_primary(delta, draws=2000, seed=194041):
    """Delta [joint/fixed/interaction,12supports,2048queries], cases averaged."""
    if delta.shape != (3, 12, 2048):
        raise ValueError("Expected three effects,12 supports, four512-query cells")
    rng = np.random.default_rng(seed)
    sw = rng.multinomial(12, np.full(12, 1 / 12), size=draws) / 12
    # Four class cells retained in the fixed randomization plan, primary is last.
    qw = [rng.multinomial(512, np.full(512, 1 / 512), size=draws) / 512 for _ in range(4)]
    sampled = np.stack([np.sum((qw[3] @ value[:, 1536:].T) * sw, axis=1) for value in delta], axis=1)
    return sampled, sw, qw[3]
