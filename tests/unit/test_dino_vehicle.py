"""Check the SSL objective's gradient and effective-batch contracts."""

import torch

from cafl4ds.dino_vehicle import BoundedRGB, ScaledDirection, distillation, views


def test_cross_view_loss_detaches_teacher_and_omits_same_view() -> None:
    """Compare explicit six cross-view CE terms and require teacher gradients absent."""
    generator = torch.Generator().manual_seed(71)
    student = [torch.randn(8, 7, generator=generator, requires_grad=True) for _ in range(4)]
    teacher = [torch.randn(8, 7, generator=generator, requires_grad=True) for _ in range(2)]
    center = torch.randn(1, 7, generator=generator)
    loss = distillation(student, teacher, center)
    manual = (
        sum(
            -((t.detach() - center).div(0.07).softmax(-1) * s.div(0.1).log_softmax(-1)).sum(-1).mean()
            for i, t in enumerate(teacher)
            for j, s in enumerate(student)
            if i != j
        )
        / 6
    )
    torch.testing.assert_close(loss, manual)
    loss.backward()
    assert all(t.grad is None for t in teacher)
    assert all(s.grad is not None and torch.isfinite(s.grad).all() for s in student)


def test_microbatch_reduction_preserves_loss_and_gradients() -> None:
    """Fixed-center separable DINO loss must match batch32 under eight microbatches."""
    generator = torch.Generator().manual_seed(83)
    student = [torch.randn(32, 9, generator=generator, requires_grad=True) for _ in range(4)]
    teacher = [torch.randn(32, 9, generator=generator) for _ in range(2)]
    center = torch.randn(1, 9, generator=generator)
    full = distillation(student, teacher, center)
    micro = sum(
        distillation([s[i : i + 4] for s in student], [t[i : i + 4] for t in teacher], center) / 8
        for i in range(0, 32, 4)
    )
    torch.testing.assert_close(full, micro)
    full_grad = torch.autograd.grad(full, student, retain_graph=True)
    micro_grad = torch.autograd.grad(micro, student)
    for left, right in zip(full_grad, micro_grad, strict=True):
        torch.testing.assert_close(left, right)


def test_prototype_projection_matches_row_weight_normalization() -> None:
    """Explicit gains/directions reproduce normalized prototype rows and backpropagate."""
    layer = ScaledDirection(7)
    with torch.no_grad():
        layer.weight_v.normal_()
        layer.weight_g.uniform_(0.5, 2)
    x = torch.randn(5, 256, requires_grad=True)
    reference = x @ (layer.weight_v / layer.weight_v.norm(dim=1, keepdim=True) * layer.weight_g).T
    torch.testing.assert_close(layer(x), reference)
    layer(x).sum().backward()
    assert layer.weight_g.grad is not None and layer.weight_v.grad is not None


def test_float_bicubic_crops_keep_valid_rgb_before_hue_conversion() -> None:
    """High-contrast float crops must not create invalid negative HSV inputs."""
    image = torch.zeros(3, 224, 224)
    image[0, :, 112:] = 1
    image[1, 112:, :] = 1
    image[2, ::2, ::2] = 1
    resized = torch.nn.functional.interpolate(image[None], size=(96, 96), mode="bicubic", align_corners=False)[0]
    assert resized.min() < 0 or resized.max() > 1
    bounded = BoundedRGB()(resized)
    assert bounded.min() >= 0 and bounded.max() <= 1
    with torch.random.fork_rng():
        torch.manual_seed(323031)
        for crop in views(image[None].repeat(4, 1, 1, 1)):
            assert torch.isfinite(crop).all() and crop.min() >= 0 and crop.max() <= 1
