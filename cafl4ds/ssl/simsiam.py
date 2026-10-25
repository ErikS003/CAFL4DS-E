"""SimSiam SSL method — the joint-embedding ``C`` backbone (Chen & He 2021).

SimSiam is the simplest of the joint-embedding family: a projector + predictor with a
**stop-gradient**, and *no* negatives and *no* momentum/EMA target (that is BYOL, a later
drop-in). Its learning signal is pulling together two augmented views, which makes it
susceptible to representational **collapse** — the degradation mode the Phase-1 positive
control will elicit and the health instruments (rank, alignment/uniformity) will catch. The
health monitor reads the *encoder* embedding, never the projector/predictor outputs.
"""

from __future__ import annotations

import torch
from torchvision.transforms import v2

from cafl4ds.models.heads import MLPHead
from cafl4ds.models.vit import TinyViTEncoder
from cafl4ds.ssl.augment import TwoView, make_ssl_augment
from cafl4ds.ssl.base import SSLMethod


def _neg_cosine(p: torch.Tensor, z: torch.Tensor, stop_grad: bool = True) -> torch.Tensor:
    """Negative cosine similarity, with the target branch optionally stop-gradient'd.

    Args:
        p: Predictor output of one branch ``[B, d]`` (gradient flows).
        z: Projector output of the other branch ``[B, d]``.
        stop_grad: When ``True`` (SimSiam as published) the target ``z`` is detached — the
            stop-gradient that prevents collapse. When ``False`` the gradient flows through
            both branches, removing that anti-collapse mechanism (the positive-control
            ablation, see :class:`SimSiam`).

    Returns:
        The mean negative cosine similarity (scalar); minimized when ``p`` aligns with ``z``.
    """
    return _neg_cosine_per_sample(p, z, stop_grad=stop_grad).mean()


def _neg_cosine_per_sample(p: torch.Tensor, z: torch.Tensor, stop_grad: bool = True) -> torch.Tensor:
    """Per-sample negative cosine similarity ``[B]`` (the un-reduced :func:`_neg_cosine`).

    Args:
        p: Predictor output of one branch ``[B, d]``.
        z: Projector output of the other branch ``[B, d]``.
        stop_grad: Whether to detach the target ``z`` (see :func:`_neg_cosine`).

    Returns:
        The per-sample negative cosine similarity ``[B]`` (lower is better).
    """
    p = torch.nn.functional.normalize(p, dim=1)
    z = torch.nn.functional.normalize(z.detach() if stop_grad else z, dim=1)
    return -(p * z).sum(dim=1)


class SimSiam(SSLMethod):
    """SimSiam over the shared :class:`~cafl4ds.models.vit.TinyViTEncoder`.

    SimSiam avoids collapse with two coupled mechanisms: a **predictor** on one branch and a
    **stop-gradient** on the target branch (Chen & He 2021). The ``anti_collapse`` flag toggles
    *both* off together, giving the documented collapse ablation used as the Phase-0 positive
    control (P0.2): with the predictor bypassed (``p = z``) and no stop-gradient, the objective
    reduces to ``-cosine(z1, z2)`` with gradients flowing through both branches, whose trivial
    global optimum maps every input to one constant vector (cosine → 1, loss → −1). Collapse is
    then mathematically forced and scale-independent — which is exactly why the toy CPU/HPU
    regime suffices to calibrate the collapse instruments (RankMe, alignment/uniformity).
    """

    def __init__(
        self,
        encoder: TinyViTEncoder,
        projector: MLPHead,
        predictor: MLPHead,
        augment: v2.Transform | None = None,
        anti_collapse: bool = True,
        collapse_alpha: float = 0.0,
        stopgrad_beta: float = 1.0,
    ) -> None:
        """Build the SimSiam method.

        Args:
            encoder: The shared backbone encoder.
            projector: The projection MLP (3-layer, last BatchNorm) mapping the pooled
                embedding to the latent space.
            predictor: The prediction MLP (2-layer, no last BatchNorm) applied to one branch.
            augment: Per-view augmentation; defaults to
                :func:`~cafl4ds.ssl.augment.make_ssl_augment` sized to the encoder.
            anti_collapse: When ``True`` (default) run SimSiam as published — predictor + stop-
                gradient. When ``False`` disable **both** anti-collapse mechanisms (predictor
                bypassed, stop-gradient off): the forced-collapse positive control (P0.2).
            collapse_alpha: **Partial-collapse loss-blend knob** (P0.2.5), active only while
                ``anti_collapse=True``. The training loss becomes
                ``(1 - alpha) * L_healthy + alpha * L_collapse``, where ``L_collapse`` is the
                forced-collapse objective (predictor bypassed, stop-gradient off). ``0.0``
                (default) is pure healthy SimSiam; ``1.0`` reproduces the P0.2 collapse objective.
                Interpolates the *whole objective* between the two calibrated poles.
            stopgrad_beta: **Partial-collapse soft-stop-gradient knob** (P0.2.5), active only while
                ``anti_collapse=True``. Fraction of the target branch that is stop-gradient'd:
                the target becomes ``beta * z.detach() + (1 - beta) * z``. ``1.0`` (default) is
                the published full stop-gradient (byte-identical to the pre-knob path); ``0.0``
                lets gradient flow fully through the target (predictor kept on). Weakens only the
                single essential anti-collapse mechanism — a mechanistically distinct path to
                collapse from ``collapse_alpha``. The two knobs are independent; P0.2.5 varies one
                at a time.

        Raises:
            ValueError: If either knob is outside ``[0, 1]``.
        """
        if not 0.0 <= collapse_alpha <= 1.0:
            raise ValueError(f"collapse_alpha must be in [0, 1]; got {collapse_alpha}.")
        if not 0.0 <= stopgrad_beta <= 1.0:
            raise ValueError(f"stopgrad_beta must be in [0, 1]; got {stopgrad_beta}.")
        super().__init__(encoder)
        self.projector = projector
        self.predictor = predictor
        self.anti_collapse = anti_collapse
        self.collapse_alpha = collapse_alpha
        self.stopgrad_beta = stopgrad_beta
        img_size = int(round(encoder.num_patches**0.5)) * encoder.patch_size
        base_augment = augment if augment is not None else make_ssl_augment(img_size)
        self.two_view = TwoView(base_augment)

    @property
    def name(self) -> str:
        """Return the method identifier (``"simsiam_collapse"`` for the PC ablation)."""
        return "simsiam" if self.anti_collapse else "simsiam_collapse"

    def embedding_surfaces(self, imgs: torch.Tensor) -> dict[str, torch.Tensor]:
        """Backbone (pooled encoder) **and** projector-output surfaces (no gradient).

        Adds the ``"proj"`` surface to the base ``"backbone"`` one: the projector output
        ``projector(encoder.embed(x))``, where SimSiam's collapse fingerprint (the terminal
        ``BatchNorm(affine=False)``) lives and where the VICReg/alignment collapse terms are
        canonically defined. The monitor reads its geometry instruments at both (P0.2.2).

        Args:
            imgs: A batch of images ``[B, C, H, W]``.

        Returns:
            ``{"backbone": [B, embed_dim], "proj": [B, proj_dim]}``.
        """
        backbone = self.encode(imgs)  # pooled encoder embedding (no grad, device-coerced)
        with torch.no_grad():
            proj = self.projector(backbone)
        return {"backbone": backbone, "proj": proj}

    def make_views(self, imgs: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        """Return a SimSiam positive pair (two independently augmented views) for alignment."""
        view_1, view_2 = self.two_view(imgs)  # unpack: `two_view.__call__` is Any without torch stubs
        return view_1, view_2

    def _branch(self, view: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        """Encode → project (→ predict, unless collapsing) one augmented view.

        Args:
            view: An augmented image batch ``[B, C, H, W]``.

        Returns:
            A ``(prediction, projection)`` pair, each ``[B, latent_dim]``. With
            ``anti_collapse=False`` the predictor is removed from the path, so ``p = z``.
        """
        z = self.projector(self.encoder.embed(view))
        p = self.predictor(z) if self.anti_collapse else z
        return p, z

    def training_step(self, imgs: torch.Tensor) -> torch.Tensor:
        """Compute the symmetric negative-cosine loss on two views.

        With ``anti_collapse=True`` and both knobs at their defaults this is the published
        stop-gradient objective; with ``anti_collapse=False`` the stop-gradient is dropped and
        the predictor bypassed (``p = z``), yielding the forced-collapse positive control. The
        P0.2.5 knobs (``collapse_alpha`` / ``stopgrad_beta``) interpolate *between* those two
        poles while ``anti_collapse=True`` — see :meth:`_healthy_loss`.

        Args:
            imgs: A batch of raw images ``[B, C, H, W]``. Labels never enter this path.

        Returns:
            The symmetrized SimSiam loss (scalar), in ``[-1, 1]`` (lower is better).
        """
        view_1, view_2 = self.two_view(imgs)
        z1 = self.projector(self.encoder.embed(view_1))
        z2 = self.projector(self.encoder.embed(view_2))
        if not self.anti_collapse:
            # Forced-collapse PC (P0.2): predictor bypassed (p = z), stop-gradient off.
            return 0.5 * (_neg_cosine(z1, z2, stop_grad=False) + _neg_cosine(z2, z1, stop_grad=False))
        loss_healthy = self._healthy_loss(z1, z2)
        if self.collapse_alpha <= 0.0:
            return loss_healthy
        # Loss-blend knob: mix in the forced-collapse objective (bypass + no stop-gradient).
        loss_collapse = 0.5 * (_neg_cosine(z1, z2, stop_grad=False) + _neg_cosine(z2, z1, stop_grad=False))
        return (1.0 - self.collapse_alpha) * loss_healthy + self.collapse_alpha * loss_collapse

    def _healthy_loss(self, z1: torch.Tensor, z2: torch.Tensor) -> torch.Tensor:
        """The healthy SimSiam loss on two projector outputs, with the soft-stop-gradient knob.

        At ``stopgrad_beta = 1.0`` (default) this is the published objective — predictor on each
        branch, target fully stop-gradient'd — byte-identical to the pre-knob path. Below ``1.0``
        the target is blended ``beta * z.detach() + (1 - beta) * z`` so a controlled fraction of
        the gradient flows through it, weakening the essential anti-collapse mechanism (P0.2.5).

        Args:
            z1: Projector output of view 1 ``[B, d]``.
            z2: Projector output of view 2 ``[B, d]``.

        Returns:
            The symmetrized healthy loss (scalar).
        """
        p1, p2 = self.predictor(z1), self.predictor(z2)
        if self.stopgrad_beta >= 1.0:
            return 0.5 * (_neg_cosine(p1, z2, stop_grad=True) + _neg_cosine(p2, z1, stop_grad=True))
        z1_t = self.stopgrad_beta * z1.detach() + (1.0 - self.stopgrad_beta) * z1
        z2_t = self.stopgrad_beta * z2.detach() + (1.0 - self.stopgrad_beta) * z2
        return 0.5 * (_neg_cosine(p1, z2_t, stop_grad=False) + _neg_cosine(p2, z1_t, stop_grad=False))

    def per_sample_loss(self, imgs: torch.Tensor) -> torch.Tensor:
        """Per-image symmetric negative-cosine loss ``[B]`` (no gradient).

        The per-frame version of :meth:`training_step`, used as an informativeness signal by
        the loss-gate knob. Two fresh views are drawn (as in a training step) and the symmetric
        negative cosine is kept per image instead of averaged over the batch.

        Args:
            imgs: A batch of raw images ``[B, C, H, W]``.

        Returns:
            A detached ``[B]`` tensor of per-image losses in ``[-1, 1]`` (lower is better).
        """
        with torch.no_grad():
            view_1, view_2 = self.two_view(imgs)
            p1, z1 = self._branch(view_1)
            p2, z2 = self._branch(view_2)
            sg = self.anti_collapse
            per_sample = 0.5 * (
                _neg_cosine_per_sample(p1, z2, stop_grad=sg) + _neg_cosine_per_sample(p2, z1, stop_grad=sg)
            )
        return per_sample
