"""Barlow Twins SSL method — the redundancy-collapse ``C`` vehicle (Zbontar et al. 2021).

Barlow Twins is a joint-embedding method with no predictor, no stop-gradient, and no
negatives: it pulls two augmented views together by driving their **cross-correlation matrix**
toward the identity. The objective has two terms on that matrix ``C`` (computed on
batch-normalized projector outputs):

* the **invariance** term ``(C_ii - 1)^2`` — make each feature agree across the two views;
* the **redundancy-reduction** term ``lambda * C_ij^2`` (``i != j``) — decorrelate distinct
  features so they carry non-redundant information.

That second term is Barlow's *anti-collapse* mechanism, and it is the analogue of SimSiam's
predictor + stop-gradient. Toggling it off (``anti_collapse=False``) is the one-toggle ablation
that makes this the Phase-0 **redundancy-collapse** positive control (P0.2.3): the projector's
terminal ``BatchNorm(affine=False)`` still pins every feature to unit variance (so per-dimension
variance *cannot* vanish — this is not point collapse), but with no decorrelation pressure the
features are free to become copies of one another. The trivial optimum is therefore a
**redundancy / dimensional** collapse: the off-diagonal covariance climbs and the effective rank
falls, while `mean_feature_var` stays healthy — the sub-mode `offdiag_cov` is built to catch and
that SimSiam's point-collapse vehicle (P0.2/P0.2.1/P0.2.2) never produces. As with SimSiam, the
health monitor reads the *encoder* embedding and the projector surface, never a training head.
"""

from __future__ import annotations

import torch
from torchvision.transforms import v2

from cafl4ds.models.heads import MLPHead
from cafl4ds.models.vit import TinyViTEncoder
from cafl4ds.ssl.augment import TwoView, make_ssl_augment
from cafl4ds.ssl.base import SSLMethod


def _off_diagonal(x: torch.Tensor) -> torch.Tensor:
    """Return a flat view of the off-diagonal elements of a square matrix.

    Args:
        x: A square matrix ``[D, D]``.

    Returns:
        A 1-D tensor of its ``D * (D - 1)`` off-diagonal entries (the standard Barlow Twins
        indexing trick: drop the last element, reshape to ``[D - 1, D + 1]``, keep columns
        ``1:``, flatten). The reshape assumes — and fails loudly otherwise — that ``x`` is
        square.
    """
    n = x.shape[0]
    return x.flatten()[:-1].view(n - 1, n + 1)[:, 1:].flatten()


class BarlowTwins(SSLMethod):
    """Barlow Twins over the shared :class:`~cafl4ds.models.vit.TinyViTEncoder`.

    Barlow Twins avoids collapse with a single mechanism — the **redundancy-reduction**
    (off-diagonal cross-correlation) term (Zbontar et al. 2021). The ``anti_collapse`` flag
    toggles it: with it ``True`` (default) the method runs as published (invariance +
    ``lambd`` * redundancy-reduction); with it ``False`` the redundancy-reduction term is
    dropped, giving the documented **redundancy-collapse** positive control (P0.2.3). Because
    the projector's terminal ``BatchNorm(affine=False)`` holds every feature at unit variance,
    the ablated optimum is *not* point collapse (per-dimension variance is preserved) but
    dimensional collapse — the features become mutually redundant, so off-diagonal covariance
    climbs while `mean_feature_var` stays quiet.
    """

    def __init__(
        self,
        encoder: TinyViTEncoder,
        projector: MLPHead,
        augment: v2.Transform | None = None,
        lambd: float = 5e-3,
        anti_collapse: bool = True,
    ) -> None:
        """Build the Barlow Twins method.

        Args:
            encoder: The shared backbone encoder.
            projector: The projection MLP (last layer a bias-free ``BatchNorm(affine=False)``,
                i.e. ``MLPHead(..., last_bn=True)``) mapping the pooled embedding to the space
                the cross-correlation is computed in. The terminal BatchNorm *is* Barlow's
                per-feature standardization, so the loss needs no separate normalization.
            augment: Per-view augmentation; defaults to
                :func:`~cafl4ds.ssl.augment.make_ssl_augment` sized to the encoder.
            lambd: Weight of the redundancy-reduction (off-diagonal) term. Ignored when
                ``anti_collapse=False`` (the term is dropped entirely).
            anti_collapse: When ``True`` (default) run Barlow Twins as published — invariance +
                redundancy reduction. When ``False`` drop the redundancy-reduction term: the
                forced redundancy-collapse positive control (P0.2.3).
        """
        super().__init__(encoder)
        self.projector = projector
        self.lambd = lambd
        self.anti_collapse = anti_collapse
        img_size = int(round(encoder.num_patches**0.5)) * encoder.patch_size
        base_augment = augment if augment is not None else make_ssl_augment(img_size)
        self.two_view = TwoView(base_augment)

    @property
    def name(self) -> str:
        """Return the method identifier (``"barlow_collapse"`` for the PC ablation)."""
        return "barlow" if self.anti_collapse else "barlow_collapse"

    def embedding_surfaces(self, imgs: torch.Tensor) -> dict[str, torch.Tensor]:
        """Backbone (pooled encoder) **and** projector-output surfaces (no gradient).

        Mirrors SimSiam: adds the ``"proj"`` surface (``projector(encoder.embed(x))``) to the
        base ``"backbone"`` one, so the monitor reads the geometry instruments at both surfaces
        (P0.2.2 / P0.2.3). The projector's terminal ``BatchNorm(affine=False)`` is where
        Barlow's redundancy-collapse fingerprint (correlated features at unit variance) lives.

        Args:
            imgs: A batch of images ``[B, C, H, W]``.

        Returns:
            ``{"backbone": [B, embed_dim], "proj": [B, proj_dim]}`` (no gradient tracking).
        """
        backbone = self.encode(imgs)  # pooled encoder embedding (no grad, device-coerced)
        with torch.no_grad():
            proj = self.projector(backbone)
        return {"backbone": backbone, "proj": proj}

    def make_views(self, imgs: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        """Return a Barlow Twins positive pair (two independently augmented views)."""
        view_1, view_2 = self.two_view(imgs)  # unpack: `two_view.__call__` is Any without torch stubs
        return view_1, view_2

    def training_step(self, imgs: torch.Tensor) -> torch.Tensor:
        """Compute the Barlow Twins cross-correlation loss on two views.

        The projector outputs are already per-feature standardized by its terminal
        ``BatchNorm(affine=False)``, so ``C = z1^T z2 / B`` is directly the cross-correlation
        matrix. With ``anti_collapse=True`` the loss is the published invariance +
        redundancy-reduction objective; with it ``False`` only the invariance term is kept,
        yielding the forced redundancy-collapse positive control.

        Args:
            imgs: A batch of raw images ``[B, C, H, W]``. Labels never enter this path.

        Returns:
            The Barlow Twins loss (scalar), ready for ``.backward()``.
        """
        view_1, view_2 = self.two_view(imgs)
        z1 = self.projector(self.encoder.embed(view_1))
        z2 = self.projector(self.encoder.embed(view_2))
        batch_size = z1.shape[0]
        cross_corr = (z1.T @ z2) / batch_size  # [D, D]; BN'd inputs => cross-correlation
        invariance = (torch.diagonal(cross_corr) - 1).pow(2).sum()
        if not self.anti_collapse:
            return invariance  # redundancy-reduction dropped => redundancy-collapse PC
        redundancy = _off_diagonal(cross_corr).pow(2).sum()
        return invariance + self.lambd * redundancy
