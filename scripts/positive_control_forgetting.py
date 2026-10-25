"""Phase-0 positive control (P0.3) — the FORGETTING-instrument calibration gate.

The collapse gate (``positive_control.py``, P0.2.x) says nothing about whether the *forgetting*
detectors work. This harness is its forgetting analogue: it induces a deliberate catastrophic-
forgetting event and checks that the forgetting instruments **fire** on it, while a matched
no-forgetting control stays **quiet** — the same two-sided discipline, a different failure mode.

Vehicle (P0.3.0): a **split-task** stream over MAE. STL-10's classes are partitioned into two
disjoint tasks, ``task_a_classes`` (era A) and ``task_b_classes`` (era B). Each arm trains in two
phases over the SAME data, differing only by a single **replay** toggle:

* **PC (forgets)** — phase A trains on task A only, then phase B trains on task B **only**. With
  no revisits to A, the representation respecializes to B and task-A probe accuracy craters.
* **healthy (holds)** — identical phase A, then phase B **replays A** (trains on A ∪ B). Task A
  is revisited, so its probe accuracy holds. The replay toggle is the *only* difference — the
  seed is reset before each arm so phase A is bit-identical, isolating forgetting as the cause.

The forgetting instruments are read from the two-phase accuracy matrix ``R[i][j]`` = accuracy on
task ``j`` after phase ``i`` (phase 0 = after A, phase 1 = after B), plus the label-free backbone
drift:

* **Backward Transfer** ``BWT = R[1][0] - R[0][0]`` (Lopez-Paz & Ranzato 2017) — negative ⇒ later
  learning eroded task A.
* **Forgetting Measure** ``FM = R[0][0] - R[1][0]`` (Chaudhry et al. 2018) — the drop from task
  A's best (post-A) to its final accuracy.
* **CKA / cosine drift** — how far the backbone moved *during phase B*, referenced to the post-A
  representation (the label-free leading indicator).

Both instruments (:func:`~cafl4ds.eval.backward_transfer`, :func:`~cafl4ds.eval.forgetting_measure`,
the :class:`~cafl4ds.monitor.HealthMonitor` drift) already exist from P0.1 — P0.3 *calibrates*
them, it does not build them.

Examples:
    Default (STL-10, CPU — small, for a plumbing check)::

        uv run python scripts/positive_control_forgetting.py

    Escalated MAE on the Gaudi HPU (the P0.3.0 fair regime; see docs/developing.md)::

        ./scripts/run_gaudi_dev.sh -m /mnt/stl10 gaudi-env-cafl4ds:latest 0 \
            python scripts/positive_control_forgetting.py device=hpu \
            encoder.embed_dim=256 encoder.depth=8 encoder.num_heads=8 img_size=64 \
            epochs_a=300 epochs_b=300
"""

import math
import sys
from collections.abc import Callable
from pathlib import Path
from typing import Any

import hydra
import torch
import torch.nn.functional as F  # noqa: N812 - conventional alias
from hydra.core.hydra_config import HydraConfig
from hydra.utils import instantiate
from loguru import logger
from omegaconf import DictConfig, OmegaConf

from cafl4ds.data.streams import EvalSet, EvalSets
from cafl4ds.eval import backward_transfer, forgetting_measure
from cafl4ds.jsonio import dumps_valid
from cafl4ds.measurements import knn_probe, linear_probe
from cafl4ds.monitor import HealthMonitor
from cafl4ds.ssl.augment import make_light_augment
from cafl4ds.ssl.base import apply_encoder_init
from cafl4ds.ssl.mae import MAE
from cafl4ds.ssl.supervised import SupervisedMethod

logger.remove()
logger.add(sys.stdout, level="INFO")


def _task_split(config: DictConfig) -> dict[str, Any]:
    """Partition the data source into two disjoint tasks with held-out probe sets.

    Loads the source once, then per class reserves a disjoint ``support`` + ``query`` split
    (labels used HERE ONLY — for the probes) before pooling the remaining images into that
    task's training set. Task A is ``config.task_a_classes``; task B is ``config.task_b_classes``.

    Args:
        config: The composed config (task class lists + per-class eval sizing + seed/batch).

    Returns:
        A dict of per-task training image batches (device-agnostic CPU tensors) and per-task
        held-out ``(support, query)`` :class:`~cafl4ds.data.streams.EvalSet` pairs.
    """
    img_a, lab_a = instantiate(config.data).load()
    data_b = config.get("data_b", None)
    if data_b is not None:  # cross-dataset split (P0.3.6): task A <- data, task B <- data_b
        img_b, lab_b = instantiate(data_b).load()
        logger.info("cross-dataset split: task A from `data`, task B from `data_b`")
    else:
        img_b, lab_b = img_a, lab_a
    per_task_source = {"A": (img_a, lab_a), "B": (img_b, lab_b)}
    gen = torch.Generator().manual_seed(config.seed)
    s, q = config.support_per_class, config.query_per_class

    tasks: dict[str, list[int]] = {"A": list(config.task_a_classes), "B": list(config.task_b_classes)}
    out: dict[str, Any] = {}
    for name, classes in tasks.items():
        images, labels = per_task_source[name]
        sup_idx, qry_idx, train_idx = [], [], []
        for cls in classes:
            idx = (labels == cls).nonzero(as_tuple=True)[0]
            idx = idx[torch.randperm(idx.numel(), generator=gen)]
            if idx.numel() <= s + q:
                raise ValueError(f"class {cls} has {idx.numel()} images but {s + q} reserved for eval.")
            sup_idx.append(idx[:s])
            qry_idx.append(idx[s : s + q])
            train_idx.append(idx[s + q :])
        train = torch.cat(train_idx)
        train = train[torch.randperm(train.numel(), generator=gen)]  # shuffle within-task (IID-in-A)
        bs = config.batch_size
        pairs = [(images[train[i : i + bs]], labels[train[i : i + bs]]) for i in range(0, train.numel(), bs)]
        pairs = [(im, lb) for im, lb in pairs if im.shape[0] >= 2]
        sup_i, qry_i = torch.cat(sup_idx), torch.cat(qry_idx)
        out[name] = {
            "batches": [im for im, _ in pairs],
            "label_batches": [lb for _, lb in pairs],  # aligned original labels (supervised mode only)
            "support": EvalSet(images[sup_i], labels[sup_i]),
            "query": EvalSet(images[qry_i], labels[qry_i]),
        }
    logger.info(f"task split: A={len(out['A']['batches'])} batches, B={len(out['B']['batches'])} batches")
    return out


def _probe(config: DictConfig, method: MAE, task: dict[str, Any]) -> float:
    """One task-specific probe: |task|-way accuracy within that task's own held-out classes."""
    was = method.training
    method.eval()
    try:
        sup, qry = task["support"], task["query"]
        if config.probe == "linear":
            return linear_probe(method.encode, (sup.images, sup.labels), (qry.images, qry.labels))
        return knn_probe(method.encode, (sup.images, sup.labels), (qry.images, qry.labels), k=config.knn_k)
    finally:
        method.train(was)


def _recon_loss(config: DictConfig, method: MAE, imgs: torch.Tensor, device: torch.device) -> float:
    """MAE-native forgetting readout: mean held-out reconstruction MSE on ``imgs``.

    Reads forgetting through MAE's *own* objective rather than a transfer probe — the signal
    that stays flat when the backbone-probe accuracy is dominated by init (P0.3.0 found the
    transfer probe has almost no headroom on toy STL-10). Augmentation is bypassed (identity) so
    the loss is measured on the actual held-out images, and the random mask is averaged over
    ``config.recon_masks`` draws for a stable estimate. Higher after phase B ⇒ the model got
    worse at reconstructing this (past) task ⇒ forgetting.

    Args:
        config: The composed config (``recon_masks`` = mask draws to average over).
        method: The MAE method (must expose ``per_sample_loss`` + a swappable ``augment``).
        imgs: The held-out images to score ``[M, C, H, W]``.
        device: Device to run the forward on.

    Returns:
        The mean reconstruction MSE over ``imgs`` (averaged across mask draws).
    """
    was = method.training
    method.eval()
    orig_augment = method.augment
    method.augment = torch.nn.Identity()  # score the real held-out images, no spatial aug
    try:
        # Resize to the encoder's native patch grid so the reconstruction target (patchify of the
        # un-augmented image) has the same patch count as the prediction. No-op for the toy path
        # (source already at that size); required when the encoder resizes internally (ViT-B, 224).
        side = int(round(method.encoder.num_patches**0.5)) * method.encoder.patch_size
        moved = imgs.to(device)
        if moved.shape[-1] != side or moved.shape[-2] != side:
            moved = F.interpolate(moved, size=side, mode="bilinear", align_corners=False)
        draws = [float(method.per_sample_loss(moved).mean().item()) for _ in range(config.recon_masks)]
        return sum(draws) / len(draws)
    finally:
        method.augment = orig_augment
        method.train(was)


def _projector_readers(health_a: dict[str, float], health_b: dict[str, float]) -> dict[str, float | None]:
    """P0.6.0/A1: the joint-embedding projector-surface label-free readers, post-A vs post-B.

    Read on the task-A canary, so a rise in ``alignment_proj`` — task-A positive pairs drifting apart at
    the projector — is a *label-free* forgetting signal, while ``rankme_proj`` is the collapse-specificity
    precondition (it must stay high; a crater there is collapse, the P0.2 mode, not forgetting). Each
    reader is reported as its absolute value at both checkpoints plus the post-A → post-B ``rise`` (delta);
    a value is ``None`` where the surface is absent (a non-JE method exposes no ``_proj`` surface).

    Args:
        health_a: The post-A :meth:`HealthMonitor.measure` dict (checkpoint 0).
        health_b: The post-B :meth:`HealthMonitor.measure` dict (checkpoint 1).

    Returns:
        ``{align_*, uniformity_*, rankme_*}`` — the projector readers behind the A1 verdict.
    """

    def rise(key: str) -> float | None:
        a, b = health_a.get(key), health_b.get(key)
        return None if a is None or b is None else b - a

    return {
        "align_after_a": health_a.get("alignment_proj"),
        "align_after_b": health_b.get("alignment_proj"),
        "align_rise": rise("alignment_proj"),
        "uniformity_after_a": health_a.get("uniformity_proj"),
        "uniformity_after_b": health_b.get("uniformity_proj"),
        "uniformity_rise": rise("uniformity_proj"),
        "rankme_after_a": health_a.get("rankme_proj"),
        "rankme_after_b": health_b.get("rankme_proj"),
    }


def _train(
    method: MAE,
    batches: list[torch.Tensor],
    epochs: int,
    optimizer: torch.optim.Optimizer,
    device: torch.device,
    grad_stats: dict[str, Any] | None = None,
) -> float:
    """Train ``method`` on ``batches`` for ``epochs`` passes; return the final-step loss.

    When ``grad_stats`` is passed it accumulates the **pre-clip** gradient-norm reads (the value
    ``clip_grad_norm_`` returns, measured before it rescales) into ``max_grad_norm`` / ``all_finite``.
    This is the P0.4 divergence instrument, read here off the forgetting vehicle for the cross-mode
    specificity check (grad norm must NOT spike during a slow forgetting event); default ``None`` is a
    no-op so existing callers are unaffected.
    """
    method.train()
    last = 0.0
    for _ in range(epochs):
        for imgs in batches:
            optimizer.zero_grad()
            loss = method.training_step(imgs.to(device))
            loss.backward()
            total_norm = float(torch.nn.utils.clip_grad_norm_(method.parameters(), 1.0))
            if grad_stats is not None:
                if math.isfinite(total_norm):
                    grad_stats["max_grad_norm"] = max(grad_stats["max_grad_norm"], total_norm)
                else:
                    grad_stats["all_finite"] = False
            optimizer.step()
            last = float(loss.item())
    return last


def _train_supervised(
    method: SupervisedMethod,
    img_batches: list[torch.Tensor],
    label_batches: list[torch.Tensor],
    class_map: dict[int, int],
    epochs: int,
    config: DictConfig,
    device: torch.device,
) -> float:
    """Train the backbone supervised (cross-entropy through a fresh per-task linear head).

    The head is created here, sized to ``len(class_map)``, trained jointly with the encoder, and
    discarded on return — only the backbone persists across tasks, so training a later task
    respecializes the shared representation and erodes an earlier one. Labels are remapped to
    task-local indices ``0..K-1`` via ``class_map`` (each task's own ``|task|``-way head).

    Args:
        method: The supervised method wrapping the shared encoder.
        img_batches: Per-batch images for this phase ``[B, C, H, W]``.
        label_batches: Per-batch original labels aligned to ``img_batches``.
        class_map: Original-class -> task-local head index (defines the head width).
        epochs: Passes over the batches.
        config: The composed config (``optim`` block builds the optimizer).
        device: Device to train on.

    Returns:
        The final-step cross-entropy loss.
    """
    head = torch.nn.Linear(method.encoder.embed_dim, len(class_map)).to(device)
    params = list(method.encoder.parameters()) + list(head.parameters())
    optimizer = instantiate(config.optim, params=params)
    # A from-scratch ViT memorises the few-thousand-image task in a few dozen epochs (train loss
    # -> 0), degrading the held-out probe that reads forgetting. Light spatial augmentation (the
    # same crop+flip MAE uses) regularises it so task A is genuinely learned and there is a real
    # well to crater; disable with supervised_augment=false to reproduce the no-aug overfit.
    augment = make_light_augment(config.img_size) if config.get("supervised_augment", True) else None
    # Precompute task-local label batches once (avoid a python remap every epoch).
    local_batches = [torch.tensor([class_map[int(y)] for y in lb.tolist()], device=device) for lb in label_batches]
    method.train()
    head.train()
    last = 0.0
    for _ in range(epochs):
        for imgs, local in zip(img_batches, local_batches, strict=False):
            optimizer.zero_grad()
            batch = imgs.to(device)
            feats = method.encoder.embed(augment(batch) if augment is not None else batch)  # grad-enabled features
            loss = F.cross_entropy(head(feats), local)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(params, 1.0)
            optimizer.step()
            last = float(loss.item())
    return last


def _recovery_curve(
    method: SupervisedMethod,
    split: dict[str, Any],
    class_map: dict[int, int],
    steps: int,
    config: DictConfig,
    device: torch.device,
) -> list[dict[str, float]]:
    """P0.3.5 recoverability guard: re-learn task A after phase B and log how fast it comes back.

    True catastrophic forgetting *destroys* the task-A structure, so relearning is slow; benign
    interference (apparent forgetting — the linear probe cannot read task A, but the information is
    retained) snaps back in a handful of steps. This runs a short task-A-only fine-tune from the
    post-phase-B backbone (a fresh ``|A|``-way head, no augmentation for a clean signal) and probes
    task A at a geometric checkpoint schedule, returning the ``(step, accuracy)`` recovery
    trajectory. The distinction separates a genuine fire from an apparent (recoverable) one.

    Args:
        method: The supervised method whose backbone has just finished phase B (mutated in place).
        split: The task split (task-A training batches + probe sets).
        class_map: Task-A original-class -> local head index (the ``|A|``-way recovery head).
        steps: Number of task-A mini-batch updates to run (0 disables; the caller gates on this).
        config: The composed config (``optim`` builds the recovery optimizer; ``probe`` reads A).
        device: Device to train/probe on.

    Returns:
        A list of ``{"step": s, "acc": task-A probe accuracy}`` from step 0 (= post-B, pre-recovery)
        through ``steps``, at geometric checkpoints ``0,1,2,4,8,…``.
    """
    img_batches, label_batches = split["A"]["batches"], split["A"]["label_batches"]
    local_batches = [torch.tensor([class_map[int(y)] for y in lb.tolist()], device=device) for lb in label_batches]
    head = torch.nn.Linear(method.encoder.embed_dim, len(class_map)).to(device)
    params = list(method.encoder.parameters()) + list(head.parameters())
    # Recovery LR. Default (``recovery_lr`` unset) ties it to the phase-B lr (``config.optim.lr``) —
    # the honest permanence test: can the *same* optimiser that dug the crater climb back out (run to
    # a plateau, not a fixed step cap)? Fixing ``recovery_lr`` across craters dug at *different* phase-B
    # lrs instead confounds depth with step-size (a deep crater relearned at a smaller lr looks less
    # recoverable for the wrong reason) — kept only as an explicit knob for a same-lr rate comparison.
    recovery_lr = config.get("recovery_lr", None)
    optimizer = (
        instantiate(config.optim, params=params, lr=recovery_lr)
        if recovery_lr is not None
        else instantiate(config.optim, params=params)
    )
    checkpoints = {0, steps}
    k = 1
    while k < steps:  # geometric schedule 1,2,4,8,... so early (fast) recovery is densely sampled
        checkpoints.add(k)
        k *= 2

    curve: list[dict[str, float]] = [{"step": 0.0, "acc": _probe(config, method, split["A"])}]
    done = 0
    while done < steps:
        for imgs, local in zip(img_batches, local_batches, strict=False):
            if done >= steps:
                break
            method.train()
            head.train()
            optimizer.zero_grad()
            feats = method.encoder.embed(imgs.to(device))
            loss = F.cross_entropy(head(feats), local)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(params, 1.0)
            optimizer.step()
            done += 1
            if done in checkpoints:
                curve.append({"step": float(done), "acc": _probe(config, method, split["A"])})
    return curve


def _guard_from_scratch_encoder(config: DictConfig, keep_weights: bool) -> None:
    """Fail loudly if a checkpoint-loading encoder would be silently kept under a from-scratch arm.

    ``apply_encoder_init(..., "from_scratch")`` is a documented **no-op** — it leaves the encoder's
    *instantiated* weights in place, which is correct for a from-scratch encoder (freshly random) but
    would SILENTLY RETAIN a loaded checkpoint if a checkpoint-loading encoder (one that declares a
    ``state_dict_path``, e.g. ``vit_b16_mae`` / ``vit_b16_pretrained``) were reached with
    ``keep_weights=False``. That is the same class of trap that once made the savings ``scratch`` pole
    a no-op (P0.3.7). No P0.3 config hits it — every checkpoint path sets ``pretrained_encoder`` /
    ``skip_phase_a`` — so this raises rather than train "from scratch" on a pretrained well by accident.

    Args:
        config: The composed config (``config.encoder`` is inspected for a ``state_dict_path`` key).
        keep_weights: Whether the arm keeps the instantiated weights (skip the from-scratch path).

    Raises:
        ValueError: If the arm is from-scratch yet the encoder declares a ``state_dict_path``.
    """
    if not keep_weights and "state_dict_path" in config.encoder:
        raise ValueError(
            "encoder declares a `state_dict_path` (a checkpoint-loading encoder) but the arm is "
            "from-scratch (neither `pretrained_encoder` nor `skip_phase_a` set): 'from_scratch' is a "
            "no-op and would silently keep the loaded checkpoint. Set pretrained_encoder=true (MAE) or "
            "skip_phase_a=true (supervised) to use the well, or select a from-scratch encoder."
        )


def _set_encoder_requires_grad(method: MAE, flag: bool) -> None:
    """Toggle ``requires_grad`` on every encoder parameter (the decoder-first phase-A warm-up lever)."""
    for p in method.encoder.parameters():
        p.requires_grad_(flag)


def _train_with_schedule(
    method: MAE,
    batches: list[torch.Tensor],
    epochs: int,
    optimizer: torch.optim.Optimizer,
    device: torch.device,
    *,
    warmup_epochs: int,
    probe_fn: Callable[[], float] | None,
    grad_stats: dict[str, Any] | None = None,
) -> tuple[float, list[float]]:
    """Train ``epochs`` passes, freezing the encoder for the first ``warmup_epochs`` (decoder-first warm-up).

    Stepping epoch-by-epoch is bit-identical to a single ``_train(epochs)`` call (same batch order, same
    optimizer, same RNG draw sequence). The encoder is frozen for the leading ``warmup_epochs`` so only the
    fresh decoder warms, then released at that boundary to consolidate jointly; if it was frozen through all
    ``epochs`` (the full-freeze endpoint) it is released at the end so later phases can train it. When
    ``probe_fn`` is given it is called after each epoch (RNG-neutral: an eval-mode + sklearn probe), building
    the per-epoch trajectory that separates the decoder-warming transient from genuine forgetting.

    Returns:
        ``(final-step loss, per-epoch probe trajectory)`` — the trajectory is empty when ``probe_fn`` is ``None``.
    """
    traj: list[float] = []
    if warmup_epochs > 0:
        _set_encoder_requires_grad(method, False)
    loss = float("nan")
    for ep in range(epochs):
        if warmup_epochs > 0 and ep == warmup_epochs:  # decoder warmed → release the encoder to consolidate
            _set_encoder_requires_grad(method, True)
        loss = _train(method, batches, 1, optimizer, device, grad_stats=grad_stats)
        if probe_fn is not None:
            traj.append(probe_fn())
    if warmup_epochs >= epochs:  # frozen through every epoch → release for any later phase
        _set_encoder_requires_grad(method, True)
    return loss, traj


def _current_stream_monitor(config: DictConfig, split: dict[str, Any]) -> HealthMonitor | None:
    """P0.6.1/C3: build the *past-data-free* current-stream drift monitor (or ``None`` when off).

    Where the canary monitor pins its drift reference to a RETAINED task-A batch (P0.6.0/A1 —
    trustworthy but canary-bound), this monitor pins it to a **current-stream** (phase-B) batch —
    ``split["B"]["query"]``, snapshotted at post-A (the A→B boundary). It stores only a fixed vector
    set (no task-A data), so it is the one reader a genuinely transient stream can run. Both arms
    reference the SAME phase-B batch, so PC-vs-healthy drift on it isolates the respecialisation that
    *causes* forgetting (the PC arm trains on this distribution and moves; the a-only healthy arm never
    does). Drift is read at every surface (backbone + the quality-tied ``_proj``) via ``drift_surfaces``.

    Args:
        config: The composed config (``current_stream_drift`` gates this on).
        split: The task split from :func:`_task_split` (its task-B held-out set is the reference).

    Returns:
        The current-stream :class:`HealthMonitor`, or ``None`` when ``current_stream_drift`` is off.
    """
    if not bool(config.get("current_stream_drift", False)):
        return None
    cs_eval = EvalSets(probe_support=split["B"]["support"], probe_query=split["B"]["query"])
    return HealthMonitor(
        eval_sets=cs_eval,
        run_knn=False,
        run_linear=False,
        run_alignment=False,
        knn_k=config.knn_k,
        drift_surfaces=True,
    )


def _run_arm(config: DictConfig, split: dict[str, Any], *, replay: bool, run_name: str) -> tuple[dict[str, Any], MAE]:
    """Run one two-phase arm (shared phase A; phase B = B-only for PC, else the healthy composition).

    The global seed is reset here so both arms start from a bit-identical encoder and train an
    identical phase A — the ``replay`` toggle in phase B is the only variable. Records the task
    accuracy matrix (post-A and post-B) and the backbone drift accrued *during phase B*.

    Args:
        config: The composed config.
        split: The task split from :func:`_task_split`.
        replay: Arm toggle — ``False`` = PC (phase B = B-only, forgets), ``True`` = healthy (phase B per
            ``healthy_replay_mode``: ``a_only`` no-shift oracle, or ``mixed`` rehearsal at ``replay_a_mult``).
        run_name: Label recorded on the returned record.

    Returns:
        A dict with the accuracy matrix, BWT / FM, phase-B drift, and per-phase task accuracies.
    """
    torch.manual_seed(config.seed)  # bit-identical init + identical phase A across arms
    device = torch.device(config.device)
    # `pretrained_encoder` keeps the instantiated (e.g. MAE-pretrained ViT-B, P0.3.6) weights as the
    # deep phase-A well instead of re-initialising from scratch; `skip_phase_a` additionally skips
    # phase-A training (the pretrained rep *is* phase A). For a pretrained MAE the encoder-only
    # checkpoint pairs with a FRESH decoder, so prefer `pretrained_encoder=true` WITH a short phase A
    # (warms the decoder + pins R00) over `skip_phase_a` (fresh-decoder gradients would hit phase B raw).
    skip_a = bool(config.get("skip_phase_a", False))
    keep_weights = skip_a or bool(config.get("pretrained_encoder", False))
    _guard_from_scratch_encoder(config, keep_weights)
    # D1 redo (audit P0.3): a DECODER-FIRST phase-A warm-up. Freeze the encoder for the first
    # `decoder_warmup_epochs` of phase A so the FRESH decoder warms against a fixed (deep) encoder
    # well, THEN unfreeze and joint-warm the remaining phase-A epochs so the encoder consolidates
    # into the trained system BEFORE R00 is measured. This preserves well-depth (the decoder no
    # longer blasts noise into the encoder at step 0) WITHOUT deferring the encoder-warming transient
    # into phase B — where the original full-freeze D1 mis-counted it as forgetting (the healthy replay
    # control then craters purely from the deferred warm-up, not from forgetting, and the gate breaks).
    #   * decoder_warmup_epochs = 0            -> plain joint-warm (the P0.3.6 path).
    #   * 0 < decoder_warmup_epochs < epochs_a -> decoder-first, then consolidate (the corrected D1).
    #   * decoder_warmup_epochs >= epochs_a    -> encoder frozen through ALL of phase A (the original,
    #                                             warming-confounded D1; `freeze_encoder_phase_a` alias).
    freeze_a = bool(config.get("freeze_encoder_phase_a", False))
    warmup_epochs = int(config.get("decoder_warmup_epochs", 0))
    if freeze_a:  # the flag is the full-freeze endpoint alias — force the whole of phase A frozen
        warmup_epochs = max(warmup_epochs, int(config.epochs_a))
    warmup_epochs = max(0, min(warmup_epochs, int(config.epochs_a)))
    if warmup_epochs > 0 and not keep_weights:
        raise ValueError(
            "decoder_warmup_epochs>0 / freeze_encoder_phase_a=true warms only the decoder while the "
            "encoder is frozen, which is meaningful ONLY with a pretrained encoder well "
            "(pretrained_encoder=true); on a from-scratch encoder it would leave the backbone at its "
            "random init with nothing learned to forget."
        )
    log_traj = bool(config.get("log_task_a_trajectory", False))
    encoder = instantiate(config.encoder)
    method = instantiate(config.ssl, encoder=encoder)
    if not keep_weights:
        apply_encoder_init(method.encoder, "from_scratch")
    method.to(device)
    optimizer = instantiate(config.optim, params=method.parameters())
    # A joint-embedding (JE) method (SimSiam / Barlow — the P0.6 vehicle) exposes a `proj` surface and a
    # positive pair, so it carries the projector-surface label-free readers (`alignment_proj` /
    # `uniformity_proj` / `rankme_proj`) MAE cannot. This flag routes the two vehicles: MAE reads its
    # recon gap, a JE method reads its projector; the shared BWT/FM probe path is identical for both.
    is_mae = isinstance(method, MAE)

    # Health monitor for the label-free readers. The query set is the task-A CANARY (P0.6.0/A1): reading
    # the P0.2-calibrated projector geometry on RETAINED task-A data turns the collapse suite into a
    # label-free FORGETTING reader (task-A structure lost ⇒ its positive pairs drift apart at the
    # projector). Drift is referenced to checkpoint 0 (post-A), so the post-B read is phase-B movement
    # alone. Alignment runs only for a JE method (MAE has no positive-pair projector surface).
    canary_eval = EvalSets(probe_support=split["A"]["support"], probe_query=split["A"]["query"])
    monitor = HealthMonitor(
        eval_sets=canary_eval, run_knn=False, run_linear=False, run_alignment=not is_mae, knn_k=config.knn_k
    )
    cs_monitor = _current_stream_monitor(config, split)  # P0.6.1/C3: past-data-free phase-B drift ref (or None)

    chance = 1.0 / len(config.task_a_classes)
    traj_a: list[float] = []  # per-epoch task-A probe across phase A (only when log_task_a_trajectory)
    a_probe = (lambda: _probe(config, method, split["A"])) if log_traj else None  # RNG-neutral (eval-mode)
    if skip_a:  # no phase-A training: the pretrained backbone IS phase A (no distinct pre-A probe)
        a_init = chance
        loss_a = float("nan")
    else:
        a_init = _probe(config, method, split["A"])  # from-scratch random-init floor, or pretrained transfer probe
        # Decoder-first schedule: freeze the encoder for the first `warmup_epochs` (decoder-only warm), then
        # release it so it consolidates jointly for the rest of phase A — the warm-up transient is paid
        # BEFORE R00 (`warmup_epochs=0` is plain joint-warm; `>= epochs_a` is the full-freeze endpoint).
        loss_a, traj_a = _train_with_schedule(
            method,
            split["A"]["batches"],
            int(config.epochs_a),
            optimizer,
            device,
            warmup_epochs=warmup_epochs,
            probe_fn=a_probe,
        )
    r00 = _probe(config, method, split["A"])  # R[0][0]: task A right after learning it
    # "Something to forget" is headroom ABOVE CHANCE. A pretrained backbone (skip_a or pretrained_encoder)
    # has no random-init floor — a short phase A only warms the fresh MAE decoder, it cannot lift an
    # already-strong probe — so gate the well against chance, not the pretrained transfer probe (which
    # would make the headroom structurally negative). Matches the supervised P0.3.4 convention.
    learn_floor = chance if keep_weights else a_init
    recon_a_after_a = _recon_loss(config, method, split["A"]["query"].images, device) if is_mae else float("nan")
    health_a = monitor.measure(method, 0)  # pin the drift reference + read the post-A canary geometry
    if cs_monitor is not None:  # P0.6.1: pin the past-data-free current-stream (phase-B) drift reference
        cs_monitor.measure(method, 0)

    # Healthy-arm phase-B composition (the PC arm, `replay=False`, is always B-only and forgets). Two modes:
    #   * "a_only"  (P0.6.0 primary): phase B = task A ONLY — the no-shift oracle. Task A is retained
    #     because the fine-tune distribution never changes, so the sole PC-vs-healthy variable is the
    #     phase-B distribution (A vs B) and both arms do a MATCHED step count (|A| ≈ |B| batches) — no
    #     arbitrary rehearsal ratio, and the healthy arm never touches a far phase-B domain that could
    #     destabilise it. What it does NOT test is rehearsal-preventability (healthy never learns B).
    #   * "mixed"   (rehearsal; P0.3 default): phase B = `replay_a_mult`×A ∪ B — the encoder learns B
    #     while replaying A, so the crater is the task-A loss replay would prevent (matched task-B
    #     exposure). `replay_a_mult` sweeps the rehearsal dose (1 = the P0.3 1:1 A∪B) and is the default
    #     so the MAE forgetting runs reproduce bit-for-bit.
    if not replay:  # PC: forgets on B-only
        b_batches = split["B"]["batches"]
    elif str(config.get("healthy_replay_mode", "mixed")) == "a_only":  # healthy: no-shift oracle
        b_batches = split["A"]["batches"]
    else:  # healthy: rehearsal (mixed A∪B at the given dose)
        b_batches = int(config.get("replay_a_mult", 1)) * split["A"]["batches"] + split["B"]["batches"]
    # Phase B: encoder always trainable; probe task A after each epoch (when logging) to trace warming vs forgetting.
    # Read the P0.4 grad-norm instrument off phase B (the crater) for the cross-mode specificity check.
    grad_stats_b: dict[str, Any] = {"max_grad_norm": 0.0, "all_finite": True}
    loss_b, traj_b = _train_with_schedule(
        method,
        b_batches,
        int(config.epochs_b),
        optimizer,
        device,
        warmup_epochs=0,
        probe_fn=a_probe,
        grad_stats=grad_stats_b,
    )
    r10 = _probe(config, method, split["A"])  # R[1][0]: task A after phase B (craters if forgotten)
    r11 = _probe(config, method, split["B"])  # R[1][1]: task B right after learning it
    recon_a_after_b = _recon_loss(config, method, split["A"]["query"].images, device) if is_mae else float("nan")
    recon_b_after_b = _recon_loss(config, method, split["B"]["query"].images, device) if is_mae else float("nan")
    health_b = monitor.measure(method, 1)  # post-B canary geometry + phase-B drift
    drift = health_b  # the record + log below read cka_drift / cosine_drift off this
    # P0.6.1: the past-data-free drift, referenced to the post-A current-stream (phase-B) snapshot.
    cs_drift = cs_monitor.measure(method, 1) if cs_monitor is not None else None

    matrix = {0: {0: r00}, 1: {0: r10, 1: r11}}
    record = {
        "run": run_name,
        "replay": replay,
        "task_a_init": a_init,
        "matrix": {str(i): {str(j): v for j, v in row.items()} for i, row in matrix.items()},
        "task_a_learned": r00 - learn_floor,  # headroom above the learning floor (chance if pretrained)
        "backward_transfer": backward_transfer(matrix),
        "forgetting_measure": forgetting_measure(matrix),
        "cka_drift": drift["cka_drift"],
        "cosine_drift": drift["cosine_drift"],
        "loss_a": loss_a,
        "loss_b": loss_b,
        "recon_a_after_a": recon_a_after_a,
        "recon_a_after_b": recon_a_after_b,
        "recon_b_after_b": recon_b_after_b,
        "recon_a_rise": recon_a_after_b - recon_a_after_a,  # >0 ⇒ forgot how to reconstruct task A
        "task_a_traj_a": traj_a,  # per-epoch task-A probe (phase A) — empty unless log_task_a_trajectory
        "task_a_traj_b": traj_b,  # per-epoch task-A probe (phase B) — empty unless log_task_a_trajectory
        "phase_b_max_grad_norm": grad_stats_b["max_grad_norm"],  # P0.4 instrument off phase B (cross-mode specificity)
        "phase_b_grad_finite": grad_stats_b["all_finite"],
    }
    if not is_mae:  # P0.6.0/A1: the JE projector-surface label-free readers, read on the task-A canary
        record["proj"] = _projector_readers(health_a, health_b)
    if cs_drift is not None:  # P0.6.1/C3: past-data-free current-stream drift (backbone + projector)
        record["cs_drift"] = {
            "cka_drift": cs_drift["cka_drift"],
            "cosine_drift": cs_drift["cosine_drift"],
            "cka_drift_proj": cs_drift.get("cka_drift_proj"),
            "cosine_drift_proj": cs_drift.get("cosine_drift_proj"),
        }
    logger.info(
        f"arm '{run_name}' (replay={replay}): taskA init={a_init:.3f} -> afterA={r00:.3f} -> afterB={r10:.3f} "
        f"| taskB afterB={r11:.3f} | BWT={record['backward_transfer']:+.4f} FM={record['forgetting_measure']:+.4f} "
        f"| cka_drift={drift['cka_drift']:.4f} | recon_A {recon_a_after_a:.4f}->{recon_a_after_b:.4f} "
        f"(rise {record['recon_a_rise']:+.4f})"
    )
    return record, method


def _run_arm_supervised(
    config: DictConfig, split: dict[str, Any], *, replay: bool, run_name: str
) -> tuple[dict[str, Any], SupervisedMethod]:
    """Supervised analogue of :func:`_run_arm` — induce forgetting via a labelled objective.

    Identical two-phase / replay-toggle structure and *identical* frozen-backbone probe + drift
    reads as :func:`_run_arm`; only the training signal differs (per-task cross-entropy instead of
    MAE reconstruction), supplying the respecialisation pressure the self-supervised objective does
    not (P0.3.4). The seed is reset so phase A is bit-identical across arms. The MAE-native recon
    readouts are undefined here and reported as ``NaN``.

    Args:
        config: The composed config.
        split: The task split from :func:`_task_split`.
        replay: Phase-B replay toggle — ``False`` = PC (B-only, forgets), ``True`` = healthy (A∪B).
        run_name: Label recorded on the returned record.

    Returns:
        A dict with the accuracy matrix, BWT / FM, phase-B drift, and per-phase task accuracies.
    """
    torch.manual_seed(config.seed)  # bit-identical init + identical phase A across arms
    device = torch.device(config.device)
    # skip_phase_a: the encoder is ALREADY a well-learned phase-A representation (an ImageNet-
    # pretrained ViT-B/16 — P0.3.4). Keep its loaded weights (no from_scratch re-init) and do not
    # train phase A; the pretrained backbone IS phase A, and phase B alone fine-tunes it. When
    # false (P0.3.4), a from-scratch TinyViT is trained on task A first.
    skip_phase_a = bool(config.get("skip_phase_a", False))
    _guard_from_scratch_encoder(config, skip_phase_a)
    encoder = instantiate(config.encoder)
    method = SupervisedMethod(encoder)
    if not skip_phase_a:
        apply_encoder_init(method.encoder, "from_scratch")
    method.to(device)

    a_classes, b_classes = list(config.task_a_classes), list(config.task_b_classes)
    map_a = {c: i for i, c in enumerate(a_classes)}  # task-A |A|-way head
    map_b = {c: i for i, c in enumerate(b_classes)}  # task-B |B|-way head (PC phase B)
    map_ab = {c: i for i, c in enumerate(a_classes + b_classes)}  # |A∪B|-way head (replay phase B)

    drift_eval = EvalSets(probe_support=split["A"]["support"], probe_query=split["A"]["query"])
    monitor = HealthMonitor(
        eval_sets=drift_eval, run_knn=False, run_linear=False, run_alignment=False, knn_k=config.knn_k
    )

    if skip_phase_a:
        # Pretrained phase A: no random-init floor to measure (the backbone is not random); use the
        # chance baseline as the headroom reference so task_a_learned reads "above chance".
        a_init = 1.0 / len(a_classes)
        loss_a = float("nan")
    else:
        a_init = _probe(config, method, split["A"])  # random-init task-A floor (headroom reference)
        loss_a = _train_supervised(
            method, split["A"]["batches"], split["A"]["label_batches"], map_a, config.epochs_a, config, device
        )
    r00 = _probe(config, method, split["A"])  # R[0][0]: task A right after learning it (pretrained if skipped)
    monitor.measure(method, 0)  # pin the drift reference to the post-A representation

    if replay:  # healthy: phase B replays A∪B (task A revisited -> holds)
        img_b = split["A"]["batches"] + split["B"]["batches"]
        lbl_b = split["A"]["label_batches"] + split["B"]["label_batches"]
        map_b_phase = map_ab
    else:  # PC: phase B trains task B only (respecializes -> forgets A)
        img_b, lbl_b, map_b_phase = split["B"]["batches"], split["B"]["label_batches"], map_b
    loss_b = _train_supervised(method, img_b, lbl_b, map_b_phase, config.epochs_b, config, device)
    r10 = _probe(config, method, split["A"])  # R[1][0]: task A after phase B (craters if forgotten)
    r11 = _probe(config, method, split["B"])  # R[1][1]: task B right after learning it
    drift = monitor.measure(method, 1)

    # P0.3.5 recoverability guard (opt-in): re-learn task A from the post-B backbone and log how fast
    # it returns. MUST run last — it mutates the backbone (after r10/r11/drift are already read).
    recovery_steps = int(config.get("recovery_steps", 0))
    recovery_curve = _recovery_curve(method, split, map_a, recovery_steps, config, device) if recovery_steps else []

    matrix = {0: {0: r00}, 1: {0: r10, 1: r11}}
    nan = float("nan")  # MAE-native recon readouts do not apply to the supervised vehicle
    record = {
        "run": run_name,
        "replay": replay,
        "task_a_init": a_init,
        "matrix": {str(i): {str(j): v for j, v in row.items()} for i, row in matrix.items()},
        "task_a_learned": r00 - a_init,
        "backward_transfer": backward_transfer(matrix),
        "forgetting_measure": forgetting_measure(matrix),
        "cka_drift": drift["cka_drift"],
        "cosine_drift": drift["cosine_drift"],
        "loss_a": loss_a,
        "loss_b": loss_b,
        "recon_a_after_a": nan,
        "recon_a_after_b": nan,
        "recon_b_after_b": nan,
        "recon_a_rise": nan,
        "recovery_curve": recovery_curve,  # P0.3.5: task-A recovery trajectory (empty unless recovery_steps>0)
    }
    logger.info(
        f"arm '{run_name}' (supervised, replay={replay}): taskA init={a_init:.3f} -> afterA={r00:.3f} "
        f"-> afterB={r10:.3f} | taskB afterB={r11:.3f} | BWT={record['backward_transfer']:+.4f} "
        f"FM={record['forgetting_measure']:+.4f} | cka_drift={drift['cka_drift']:.4f}"
    )
    return record, method


def _ssl_relearn_curve(
    config: DictConfig, method: MAE, split: dict[str, Any], steps: int, device: torch.device
) -> list[dict[str, float]]:
    """Relearn task A via MAE (label-free) for ``steps`` updates — the P0.3.7 savings per-init curve.

    Probes task A at geometric checkpoints while relearning.
    ``method`` wraps the encoder state under test with a **fresh** decoder, so the decoder-warming
    nuisance is identical across inits and any difference in probe-recovery *rate* is attributable to
    the encoder's retained task-A structure. Both the frozen-backbone probe and the MAE-native recon
    loss are logged (the recon curve stays readable even if the indirect probe moves slowly).

    Returns ``{step, acc, recon}`` from step 0 (pre-relearn) through ``steps`` at ``0,1,2,4,…``.
    """
    method.to(device)
    savings_lr = config.get("savings_lr", None)  # null → the phase-B lr (the optimiser that dug it)
    optimizer = (
        instantiate(config.optim, params=method.parameters(), lr=savings_lr)
        if savings_lr is not None
        else instantiate(config.optim, params=method.parameters())
    )
    batches = split["A"]["batches"]
    checkpoints = {0, steps}
    k = 1
    while k < steps:  # geometric so early (fast) relearning is densely sampled
        checkpoints.add(k)
        k *= 2

    def _snapshot(done: int) -> dict[str, float]:
        return {
            "step": float(done),
            "acc": _probe(config, method, split["A"]),
            "recon": _recon_loss(config, method, split["A"]["query"].images, device),
        }

    curve = [_snapshot(0)]
    done = 0
    while done < steps:
        for imgs in batches:
            if done >= steps:
                break
            method.train()
            optimizer.zero_grad()
            loss = method.training_step(imgs.to(device))
            loss.backward()
            torch.nn.utils.clip_grad_norm_(method.parameters(), 1.0)
            optimizer.step()
            done += 1
            if done in checkpoints:
                curve.append(_snapshot(done))
    return curve


def _savings_probe(
    config: DictConfig,
    split: dict[str, Any],
    pc_method: MAE,
    healthy_method: MAE,
    r00: float,
    steps: int,
    device: torch.device,
) -> dict[str, Any]:
    """P0.3.7 savings paradigm: is the label-free MAE fire *genuine* loss or *apparent* (recoverable)?

    Relearn task A via MAE from three encoder inits — **scratch** (random-init MAE, the
    inherited-vs-relearned floor: task-A competence reachable without the pretrained structure),
    **pc_forgot** (the post-phase-B forgotten encoder), and **healthy_held** (the replay arm that
    held = the floor) — each wrapped in a **fresh** decoder, and compare how fast the task-A probe
    returns. Relearn ≈ scratch ⇒ genuine catastrophic loss; relearn ≈ healthy ⇒ the structure
    survived and only the readout moved (apparent, recoverable interference).

    Note the pre-forget well ``r00`` is ImageNet-*pretraining*-sourced, so under a matched budget a
    from-scratch MAE stays far below it on task-A data alone — the reading is *budget-matched*
    (retained structure, not task-A relearning, carries the competence), not a claim of absolute
    unreachability. And ``scratch`` is not a clean savings-0 (truly-forgotten) pole: it lost *all*
    structure, not task A specifically, so this floor reads inherited-vs-relearned, not a genuine pole.
    """
    scratch_enc = instantiate(config.encoder)
    scratch_enc.reset_parameters_from_scratch()  # wipe the loaded checkpoint → a genuine naive learner
    inits = {"scratch": scratch_enc, "pc_forgot": pc_method.encoder, "healthy_held": healthy_method.encoder}
    curves: dict[str, list[dict[str, float]]] = {}
    for name, encoder in inits.items():
        relearn = instantiate(config.ssl, encoder=encoder)  # fresh decoder auto-sized to the encoder
        logger.info(f"savings relearn '{name}': {steps} task-A MAE updates (fresh decoder)")
        curves[name] = _ssl_relearn_curve(config, relearn, split, steps, device)

    def _reach(curve: list[dict[str, float]], tau: float) -> float | None:
        return next((pt["step"] for pt in curve if pt["acc"] >= tau), None)

    summary: dict[str, Any] = {}
    for name, curve in curves.items():
        accs = [pt["acc"] for pt in curve]
        summary[name] = {
            "start_acc": curve[0]["acc"],
            "max_acc": max(accs),
            "final_acc": curve[-1]["acc"],
            "start_recon": curve[0]["recon"],
            "final_recon": curve[-1]["recon"],
            "steps_to_r00": _reach(curve, r00),
        }
    # Savings fraction on the [scratch, healthy] poles, read at a target scratch can actually reach
    # (its own ceiling): 0 ⇒ relearns like scratch (genuine loss), 1 ⇒ like healthy (fully retained).
    # NB degenerate when both pretrained arms start above that ceiling (→ 1.0 regardless of relearning):
    # then it reads *retention* (crater bottom above the naive ceiling), not relearning *speed*.
    tau = summary["scratch"]["max_acc"]
    u = {n: _reach(c, tau) for n, c in curves.items()}
    savings = None
    if u["scratch"] is not None and u["healthy_held"] is not None and u["pc_forgot"] is not None:
        span = u["scratch"] - u["healthy_held"]
        savings = (u["scratch"] - u["pc_forgot"]) / span if span else 1.0
    s_sc, s_pc, s_h = summary["scratch"], summary["pc_forgot"], summary["healthy_held"]
    frac = f"{savings:.2f}" if savings is not None else "n/a"
    logger.info(
        f"SAVINGS [P0.3.7] (task-A relearn, R00={r00:.3f}): "
        f"scratch {s_sc['start_acc']:.3f}→{s_sc['max_acc']:.3f} | "
        f"pc_forgot {s_pc['start_acc']:.3f}→{s_pc['max_acc']:.3f} | "
        f"healthy {s_h['start_acc']:.3f}→{s_h['max_acc']:.3f} | savings_fraction={frac}"
    )
    return {"r00": r00, "steps": steps, "savings_fraction": savings, "summary": summary, "curves": curves}


def _projector_gate(config: DictConfig, pc: dict[str, Any], healthy: dict[str, Any]) -> dict[str, Any]:
    """P0.6.0/A1 reporting: is a JE projector label-free reader two-sided against the BWT crater?

    The A1 deliverable, **reported not hard-gated** (thresholds settle after the runs, mirroring how
    drift / the recon gap are reported). ``alignment_proj`` rises when task-A positive pairs drift apart
    at the projector, so the forgetting signal is the PC arm's rise *exceeding* the replay-protected
    healthy arm's — two-sided on the replay toggle. ``rankme_proj`` is the collapse-specificity
    **precondition**: it must *hold* (not crater) on the PC arm, else the projector move is collapse (the
    P0.2 mode), not forgetting, and the run is out of scope for a forgetting verdict.

    Args:
        config: The composed config (``gate.rankme_proj_hold_frac`` sets the not-collapsed bar).
        pc: The PC arm's ``proj`` reader block.
        healthy: The healthy (replay) arm's ``proj`` reader block.

    Returns:
        A flat dict of the two-sided projector reads + the ``rankme_proj`` precondition boolean.
    """
    frac = float(config.gate.get("rankme_proj_hold_frac", 0.5))
    pc_r0, pc_r1 = pc.get("rankme_after_a"), pc.get("rankme_after_b")
    rankme_holds = pc_r0 is not None and pc_r1 is not None and pc_r0 > 0 and pc_r1 >= frac * pc_r0

    def separates(reader: str) -> bool | None:
        p, h = pc.get(f"{reader}_rise"), healthy.get(f"{reader}_rise")
        return None if p is None or h is None else p > h

    return {
        "proj_align_pc_rise": pc.get("align_rise"),
        "proj_align_healthy_rise": healthy.get("align_rise"),
        "proj_align_separates": separates("align"),  # PC task-A pairs drift apart more than replay?
        "proj_uniformity_pc_rise": pc.get("uniformity_rise"),
        "proj_uniformity_healthy_rise": healthy.get("uniformity_rise"),
        "proj_uniformity_separates": separates("uniformity"),
        "proj_pc_rankme_after_a": pc_r0,
        "proj_pc_rankme_after_b": pc_r1,
        "proj_pc_rankme_holds": rankme_holds,  # collapse-specificity precondition (must be True)
    }


def _current_stream_drift_report(pc: dict[str, Any], healthy: dict[str, Any]) -> dict[str, Any]:
    """P0.6.1/C3 reporting: is the *past-data-free* current-stream drift two-sided against the crater?

    The deliverable of P0.6.1, **reported not hard-gated** (thresholds settle after the runs, like the
    other label-free reads). Both arms are pinned to the SAME post-A phase-B snapshot, so a forgetting-
    specific signal is the PC arm (free to respecialise onto phase B) drifting *materially more* than
    the healthy arm (which never trains phase B) on that shared reference. Reported at both surfaces —
    the backbone and the quality-tied projector (``_proj``). A separation says drift carries a
    forgetting-specific signal *past-data-free*; drift alike across arms is the pre-registered null
    (drift reads plasticity indiscriminately). ``rankme_proj`` collapse-specificity is the projector
    gate's job (a JE arm always carries it), so it is not re-checked here.

    Args:
        pc: The PC (B-only) arm's ``cs_drift`` block.
        healthy: The healthy arm's ``cs_drift`` block.

    Returns:
        A flat dict of the two-sided current-stream drift reads at both surfaces.
    """
    p, h = pc["cs_drift"], healthy["cs_drift"]

    def separates(key: str) -> bool | None:
        pv, hv = p.get(key), h.get(key)
        return None if pv is None or hv is None else pv > hv

    out: dict[str, Any] = {}
    for metric in ("cka_drift", "cosine_drift"):
        for surf in ("", "_proj"):
            key = f"{metric}{surf}"
            out[f"cs_{key}_pc"] = p.get(key)
            out[f"cs_{key}_healthy"] = h.get(key)
            out[f"cs_{key}_separates"] = separates(key)  # PC drifts more than the replay arm?
    return out


def _evaluate_forgetting_gate(config: DictConfig, pc: dict[str, Any], healthy: dict[str, Any]) -> dict[str, Any]:
    """Two-sided forgetting gate: the PC must forget *for the right reason*, healthy must hold.

    The gate rests on the **transfer-probe** forgetting metrics, which P0.3.0 found to be the
    reliable signal on a distinct-task split. It **passes** iff phase A genuinely learned task A
    (so there is something to forget: task-A gain ≥ ``learn_min``), task A then craters in the PC
    arm (BWT ≤ −``bwt_fire``, FM ≥ ``fm_fire``), the healthy arm **holds** — does not forget, i.e.
    BWT ≥ −``bwt_quiet`` (one-sided: improving task A is not a failure) —
    and the PC forgets at least ``min_forget_ratio``× the healthy arm's forgetting (``contrast`` —
    so the verdict rests on the *difference* the replay toggle makes, not an absolute bar alone).

    Two label-free signals are **reported but not gated**, per P0.3.0's calibration:

    * **backbone drift** — ``cka_drift`` separates only weakly and near-threshold (a coin-flip at
      toy scale), and ``cosine_drift`` *inverts* (larger in the healthy arm), so drift is not a
      trustworthy standalone forgetting detector here. Reported for the record, not gated.
    * **MAE-native recon gap** — ``pc_recon_a_rise − healthy_recon_a_rise``: how much *less* the PC
      improved held-out task-A reconstruction than the replay arm. It separates cleanly, but as
      *relative retention* (both arms' recon improves; the PC just improves less), and its absolute
      magnitude is dominated by task difficulty — so it corroborates rather than gates.

    Args:
        config: The composed config (``gate`` block holds the thresholds).
        pc: The PC (B-only) arm record.
        healthy: The healthy (A-replay) arm record.

    Returns:
        A dict of the measured numbers, the hard-``checks`` booleans, the non-gating ``reported``
        observations, and the overall ``passed`` (all hard checks).
    """
    g = config.gate
    pc_bwt, pc_fm = pc["backward_transfer"], pc["forgetting_measure"]
    h_bwt, h_fm = healthy["backward_transfer"], healthy["forgetting_measure"]
    forget_ratio = (pc_fm / h_fm) if h_fm and h_fm > 0 else float("inf")
    # MAE-native recon contrast as a DIFFERENCE, not a ratio: the healthy rise is usually negative
    # (replay improves task-A recon), which makes a ratio degenerate. The gap = how much less the PC
    # retained. Positive ⇒ the PC held task-A reconstruction worse than the replay arm.
    recon_gap = pc["recon_a_rise"] - healthy["recon_a_rise"]

    checks = {
        "pc_learned_A": pc["task_a_learned"] >= g.learn_min,
        "pc_bwt_fires": pc_bwt is not None and pc_bwt <= -g.bwt_fire,
        "pc_fm_fires": pc_fm is not None and pc_fm >= g.fm_fire,
        # One-sided: "holds" means the healthy arm did not FORGET (BWT ≥ −bwt_quiet). A POSITIVE BWT is
        # not a failure — it is task A *improving*, the opposite of forgetting, and the A-only healthy arm
        # (P0.6.0: phase B = task A only) routinely over-improves off a shallow well (e.g. +0.05). A
        # symmetric |BWT| ≤ bwt_quiet would spuriously fail those seeds; it also contradicts P0.3.9's own
        # "over-improvement still holds" reading (its shallow-well control reports a +0.043 hold).
        "healthy_holds": h_bwt is not None and h_bwt >= -g.bwt_quiet,
        "contrast": forget_ratio >= g.min_forget_ratio,
    }
    reported: dict[str, Any] = {
        "pc_cka_drift": pc["cka_drift"],
        "healthy_cka_drift": healthy["cka_drift"],
        "pc_cosine_drift": pc["cosine_drift"],
        "healthy_cosine_drift": healthy["cosine_drift"],
        "drift_separates": pc["cka_drift"] >= healthy["cka_drift"],  # weak/near-threshold, not gated
        "pc_recon_a_rise": pc["recon_a_rise"],
        "healthy_recon_a_rise": healthy["recon_a_rise"],
        "recon_forget_gap": recon_gap,
        "recon_corroborates": recon_gap >= g.recon_gap_min,  # relative retention; corroborates, not gated
    }
    if "proj" in pc and "proj" in healthy:  # P0.6.0/A1: the JE projector label-free readers, two-sided
        reported.update(_projector_gate(config, pc["proj"], healthy["proj"]))
    if "cs_drift" in pc and "cs_drift" in healthy:  # P0.6.1/C3: past-data-free current-stream drift
        reported.update(_current_stream_drift_report(pc, healthy))
    return {
        "mode": "forgetting",
        "pc_task_a_learned": pc["task_a_learned"],
        "pc_backward_transfer": pc_bwt,
        "pc_forgetting_measure": pc_fm,
        "healthy_backward_transfer": h_bwt,
        "healthy_forgetting_measure": h_fm,
        "forget_ratio": forget_ratio,
        "thresholds": OmegaConf.to_container(g),
        "checks": checks,
        "reported": reported,
        "passed": all(checks.values()),
    }


def _render_summary(gate: dict[str, Any]) -> str:
    """Render the forgetting-gate verdict block (hard gate + reported label-free observations)."""
    c, t, r = gate["checks"], gate["thresholds"], gate["reported"]
    verdict = "PASS ✅" if gate["passed"] else "FAIL ❌"
    base = (
        f"POSITIVE-CONTROL GATE [forgetting]: {verdict}\n"
        f"  PC learned task A       = {gate['pc_task_a_learned']:+.3f} "
        f"(>= {t['learn_min']} -> something to forget?  {c['pc_learned_A']})\n"
        f"  PC BackwardTransfer     = {gate['pc_backward_transfer']:+.4f} "
        f"(<= -{t['bwt_fire']} -> task A craters?  {c['pc_bwt_fires']})\n"
        f"  PC ForgettingMeasure    = {gate['pc_forgetting_measure']:+.4f} "
        f"(>= {t['fm_fire']} -> fires?  {c['pc_fm_fires']})\n"
        f"  healthy BackwardTransfer= {gate['healthy_backward_transfer']:+.4f} "
        f"(>= -{t['bwt_quiet']} -> task A holds (did not forget)?  {c['healthy_holds']})\n"
        f"  forget ratio (PC/healthy FM) = {gate['forget_ratio']:.2f}x "
        f"(>= {t['min_forget_ratio']}x -> contrast?  {c['contrast']})\n"
        f"  [reported, not gated] cka_drift PC {r['pc_cka_drift']:.4f} vs healthy "
        f"{r['healthy_cka_drift']:.4f}; cosine_drift PC {r['pc_cosine_drift']:.4f} vs healthy "
        f"{r['healthy_cosine_drift']:.4f} (weak/inverted -> not a standalone forgetting detector)\n"
        f"  [reported, not gated] MAE recon gap = {r['recon_forget_gap']:+.4f} "
        f"(PC held task-A recon {r['recon_forget_gap']:+.4f} worse than replay; corroborates?  "
        f"{r['recon_corroborates']})"
    )
    if "proj_align_separates" in r:  # P0.6.0/A1: the JE projector-surface label-free readers
        base += (
            f"\n  [A1, reported] projector canary (task-A): alignment_proj rise PC "
            f"{r['proj_align_pc_rise']} vs healthy {r['proj_align_healthy_rise']} "
            f"(PC pairs drift apart more?  {r['proj_align_separates']}); rankme_proj PC "
            f"{r['proj_pc_rankme_after_a']} -> {r['proj_pc_rankme_after_b']} "
            f"(holds, not collapse?  {r['proj_pc_rankme_holds']})"
        )
    if "cs_cka_drift_separates" in r:  # P0.6.1/C3: the past-data-free current-stream drift reader
        base += (
            f"\n  [C3, reported] current-stream drift (past-data-free, phase-B ref): cka backbone PC "
            f"{r['cs_cka_drift_pc']} vs healthy {r['cs_cka_drift_healthy']} "
            f"(PC drifts more?  {r['cs_cka_drift_separates']}); cka proj PC "
            f"{r['cs_cka_drift_proj_pc']} vs healthy {r['cs_cka_drift_proj_healthy']} "
            f"(PC drifts more?  {r['cs_cka_drift_proj_separates']})"
        )
    return base


@hydra.main(version_base=None, config_path="../cafl4ds/configs", config_name="positive_control_forgetting")  # type: ignore[misc]
def main(config: DictConfig) -> None:
    """Run both arms, apply the forgetting gate, and write the comparison."""
    out_dir = Path(HydraConfig.get().runtime.output_dir)
    split = _task_split(config)

    # training_mode selects the forgetting *vehicle*: 'ssl' (MAE reconstruction, the faithful
    # system objective) or 'supervised' (cross-entropy, a guaranteed respecialisation pressure for
    # instrument calibration when SSL is too forgetting-resistant to fire — see P0.3.4). Both read
    # forgetting through the identical frozen-backbone probe + drift instruments.
    supervised = config.get("training_mode", "ssl") == "supervised"
    # Short vehicle name for the run tag + log, from the SSL target (mae / simsiam / barlow — the P0.6 JE
    # vehicle), so a SimSiam run is not mislabelled "mae". Supervised keeps its own cross-entropy tag.
    ssl_name = str(config.ssl.get("_target_", "ssl")).rsplit(".", 1)[-1].removeprefix("build_").lower()
    run_arm = _run_arm_supervised if supervised else _run_arm
    tag = "sup" if supervised else ssl_name
    logger.info(f"forgetting vehicle: {'supervised cross-entropy' if supervised else f'self-supervised ({ssl_name})'}")

    # Label the healthy arm by its phase-B composition: the A-only no-shift oracle vs a rehearsal dose.
    healthy_mode = str(config.get("healthy_replay_mode", "mixed"))
    healthy_label = (
        f"{tag}_healthy_aonly"
        if healthy_mode == "a_only"
        else f"{tag}_healthy_replay_r{int(config.get('replay_a_mult', 1))}"
    )
    healthy, healthy_method = run_arm(config, split, replay=True, run_name=healthy_label)
    pc, pc_method = run_arm(config, split, replay=False, run_name=f"{tag}_pc_forget")

    gate = _evaluate_forgetting_gate(config, pc, healthy)
    logger.info(_render_summary(gate))

    comparison: dict[str, Any] = {"gate": gate, "pc": pc, "healthy": healthy}

    # P0.3.7 savings guard (SSL/MAE only): once a fire exists, is it *genuine* loss or *apparent*
    # (recoverable) interference? Relearn task A from scratch / PC-post-forget / healthy-post-B and
    # compare rates. Runs last (mutates the arm encoders). See docs/experiments/phase0/P0.3.7.md.
    savings_steps = int(config.get("savings_steps", 0))
    if savings_steps and not supervised:
        device = torch.device(config.device)
        r00 = float(pc["matrix"]["0"]["0"])
        comparison["savings"] = _savings_probe(config, split, pc_method, healthy_method, r00, savings_steps, device)

    (out_dir / "comparison.json").write_text(dumps_valid(comparison), encoding="utf-8")
    logger.info(f"wrote comparison + gate to {out_dir / 'comparison.json'}")

    if not gate["passed"]:
        logger.error(
            "Gate did NOT pass. Either the PC did not leave a clean forgetting fingerprint (task A "
            "never learned, or did not crater) or the healthy control did not hold — investigate the "
            "regime before trusting the forgetting instruments downstream."
        )
        sys.exit(1)


if __name__ == "__main__":
    main()  # pylint: disable=no-value-for-parameter
