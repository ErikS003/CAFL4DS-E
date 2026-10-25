"""Tests for the fixed-teacher continual-adaptation objective."""

from __future__ import annotations

import pytest
import torch

from cafl4ds.models.vit import TinyViTEncoder
from cafl4ds.ssl.factory import build_simsiam
from cafl4ds.ssl.teacher_student import TeacherStudentAdapter


def _method(weight: float = 1.0) -> TeacherStudentAdapter:
    encoder = TinyViTEncoder(img_size=16, patch_size=8, embed_dim=12, depth=2, num_heads=3)
    return TeacherStudentAdapter(
        build_simsiam(encoder, proj_hidden=16, proj_dim=8, pred_hidden=4),
        bottleneck_dim=4,
        retention_weight=weight,
    )


def test_teacher_student_starts_identity_matched_and_only_adapter_trains() -> None:
    """Teacher and student begin equal while gradients are confined to the residual adapter."""
    method = _method()
    images = torch.rand(3, 3, 16, 16)
    with torch.no_grad():
        assert torch.equal(method.encoder.embed(images), method.teacher.embed(images))
    loss = method.training_step(images)
    loss.backward()
    assert method.encoder.adapter[-1].weight.grad is not None
    assert all(parameter.grad is None for parameter in method.teacher.parameters())
    assert all(parameter.grad is None for parameter in method.projector.parameters())
    assert all(parameter.grad is None for parameter in method.predictor.parameters())


def test_teacher_modules_remain_eval_after_online_train_restore() -> None:
    """The streaming-loop hook can keep every fixed target module out of train mode."""
    method = _method()
    method.train()
    for module in method._frozen_online_modules:
        module.eval()
    assert all(not module.training for module in method._frozen_online_modules)


def test_teacher_student_rejects_negative_retention_weight() -> None:
    """A negative retention penalty would reward forgetting and is invalid."""
    with pytest.raises(ValueError, match="non-negative"):
        _method(-1.0)
