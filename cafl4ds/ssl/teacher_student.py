"""Fixed-teacher continual adaptation for a residual student adapter."""

from __future__ import annotations

import copy

import torch
from torch.nn import functional as F

from cafl4ds.models.adapters import ResidualAdapterEncoder
from cafl4ds.ssl.base import SSLMethod
from cafl4ds.ssl.simsiam import SimSiam


class TeacherStudentAdapter(SSLMethod):
    """Adapt a small residual while a competent SimSiam model supplies fixed targets.

    The student learns augmentation invariance on incoming images by matching its two augmented
    predictions to the fixed teacher's clean-view projection. A separate clean-view feature loss
    penalizes movement away from the teacher and supplies the continual-retention constraint.
    """

    def __init__(self, base: SimSiam, bottleneck_dim: int = 16, retention_weight: float = 1.0) -> None:
        """Build an identity-matched student from a fully pretrained SimSiam method."""
        if retention_weight < 0.0:
            raise ValueError(f"retention_weight must be non-negative; got {retention_weight}.")
        teacher = copy.deepcopy(base.encoder)
        super().__init__(ResidualAdapterEncoder(base.encoder, bottleneck_dim))
        self.teacher = teacher
        self.projector = base.projector
        self.predictor = base.predictor
        self.two_view = base.two_view
        self.retention_weight = retention_weight
        for module in (self.teacher, self.projector, self.predictor):
            module.requires_grad_(False)
            module.eval()
        self._frozen_online_modules = (self.teacher, self.projector, self.predictor)

    @property
    def name(self) -> str:
        """Keep the joint-embedding trust-map family used by the pretrained method."""
        return "simsiam"

    def training_step(self, imgs: torch.Tensor) -> torch.Tensor:
        """Return fixed-teacher plasticity plus clean-view retention loss."""
        view_1, view_2 = self.two_view(imgs)
        with torch.no_grad():
            teacher_features = self.teacher.embed(imgs)
            teacher_projection = self.projector(teacher_features)
        prediction_1 = self.predictor(self.projector(self.encoder.embed(view_1)))
        prediction_2 = self.predictor(self.projector(self.encoder.embed(view_2)))
        plasticity = -0.5 * (
            F.cosine_similarity(prediction_1, teacher_projection, dim=1).mean()
            + F.cosine_similarity(prediction_2, teacher_projection, dim=1).mean()
        )
        student_clean = self.encoder.embed(imgs)
        retention = (1.0 - F.cosine_similarity(student_clean, teacher_features, dim=1)).mean()
        return plasticity + self.retention_weight * retention

    def embedding_surfaces(self, imgs: torch.Tensor) -> dict[str, torch.Tensor]:
        """Expose the adapted backbone and its fixed-projector surface to the monitor."""
        backbone = self.encode(imgs)
        with torch.no_grad():
            projection = self.projector(backbone)
        return {"backbone": backbone, "proj": projection}

    def make_views(self, imgs: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        """Return the same pretrained augmentation pair used by the online objective."""
        view_1, view_2 = self.two_view(imgs)
        return view_1, view_2
