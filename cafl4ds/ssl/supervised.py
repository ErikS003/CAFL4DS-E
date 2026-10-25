"""Supervised classifier wrapper — a training-signal vehicle for forgetting calibration.

Not an SSL method in spirit, but it satisfies the :class:`~cafl4ds.ssl.base.SSLMethod`
interface (``encode`` / ``embedding_surfaces``) so the forgetting harness reads it through the
*same* frozen-backbone probe and drift instruments as MAE. Its role is to supply the
**guaranteed respecialisation pressure** the self-supervised MAE objective does not: SSL
reconstruction learns broadly transferable features and barely forgets (P0.3.0–P0.3.3), so a
supervised cross-entropy signal is used instead to *induce* a clean catastrophic-forgetting
event on which to calibrate BWT/FM. A linear classification head is trained per task on top of
the backbone (the head is created and owned by the harness, not this class, since its width
changes per task); training task B overwrites the task-A representation and the transfer probe
on A craters. Only the backbone persists across tasks — exactly the representation the
instruments read.
"""

from __future__ import annotations

import torch

from cafl4ds.ssl.base import SSLMethod


class SupervisedMethod(SSLMethod):
    """The shared encoder exposed through the SSLMethod probe/measurement surface.

    Trained supervised by the forgetting harness (cross-entropy through a per-task linear head
    the harness manages externally); this class only provides the frozen-backbone read the
    probe and drift instruments use. :meth:`training_step` is intentionally unsupported —
    supervised training needs labels, which never flow through the label-free SSL path.
    """

    @property
    def name(self) -> str:
        """Short method identifier for logging."""
        return "supervised"

    def training_step(self, imgs: torch.Tensor) -> torch.Tensor:
        """Not supported — supervised training uses labels, applied by the harness.

        Args:
            imgs: Unused.

        Raises:
            NotImplementedError: Always; the harness trains this via a labelled cross-entropy
                loop over :meth:`~cafl4ds.models.vit.TinyViTEncoder.embed`, not the label-free
                ``training_step`` path.
        """
        raise NotImplementedError(
            "SupervisedMethod is trained with labels by the forgetting harness, not via the label-free training_step."
        )
