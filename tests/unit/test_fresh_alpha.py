"""Check allocation leakage and that hypothetical backprop preserves actual learning."""

import copy

import pytest
import torch

from cafl4ds.filters.steerable import SteerableReplay, fit_anchors
from cafl4ds.fresh_alpha import action_observations, allocate_roles, support_folds
from cafl4ds.image_utility import embed
from cafl4ds.models.vit import TinyViTEncoder
from cafl4ds.ssl.factory import build_mae
from scripts.audit_fresh_alpha_data import image_paths
from scripts.run_image_utility_experiment import identical
from scripts.run_slow_edge_pilot import optimizer_for, update


def test_historical_image_hash_keys_are_recorded() -> None:
    """Hash-map keys count as exposed paths even if no explicit role list accompanies them."""
    recorded = image_paths({"image_hashes": {"first.jpg": "abc"}, "query_paths": ["second.jpg"]})
    assert len(recorded) == 2
    assert any(path.endswith("first.jpg") for path in recorded)
    assert any(path.endswith("second.jpg") for path in recorded)


def test_roles_are_balanced_disjoint_private_and_capacity_checked() -> None:
    """Private role draws cannot overlap or silently shrink an underfilled cell."""
    cells = {f"{d}_{c}": {"unused_paths": [f"{d}_{c}_{i}.jpg" for i in range(20)]} for d in range(2) for c in range(3)}
    rng = torch.get_rng_state().clone()
    roles = allocate_roles(cells, {"support": 8, "query": 6}, 3)
    assert len(roles["support"]) == 48 and len(roles["query"]) == 36
    assert not set(roles["support"]) & set(roles["query"])
    assert torch.equal(rng, torch.get_rng_state())
    assert roles == allocate_roles(cells, {"support": 8, "query": 6}, 3)
    with pytest.raises(ValueError, match="require"):
        allocate_roles(cells, {"support": 16, "query": 6}, 3)


def test_cv_folds_never_use_query() -> None:
    """Every support image is withheld exactly once, with no fold overlap."""
    labels = torch.arange(3).repeat_interleave(16).repeat(2)
    domains = torch.arange(2).repeat_interleave(48)
    support = torch.arange(96)
    folds = list(support_folds(labels, domains, support, 4))
    assert len(folds) == 4
    assert torch.equal(torch.cat([v for _, v in folds]).sort().values, support)
    for fitting, validation in folds:
        assert len(fitting) == 72 and len(validation) == 24
        assert not torch.isin(fitting, validation).any()


def test_backprop_observations_do_not_change_next_update() -> None:
    """A gradient observation leaves the following optimizer update bitwise unchanged."""
    torch.set_num_threads(2)
    torch.manual_seed(45)
    method = build_mae(TinyViTEncoder(img_size=16, embed_dim=24, depth=1, num_heads=3), decoder_dim=24)
    images = torch.rand(96, 3, 16, 16)
    centers, target = fit_anchors(embed(method, images), count=2, seed=7)
    selector = SteerableReplay(method.encoder, centers, target, seed=7)
    optimizer = optimizer_for(method)
    update(method, optimizer, images[:32], 0, selector)
    snapshot = copy.deepcopy(method.state_dict())
    gradients = [None if p.grad is None else p.grad.clone() for p in method.parameters()]
    actuator = selector.state_dict()
    rng = torch.get_rng_state().clone()
    trial, private = copy.deepcopy(method), copy.deepcopy(selector)
    opt = optimizer_for(trial)
    opt.load_state_dict(copy.deepcopy(optimizer.state_dict()))
    _, actions = action_observations(method, selector, images[32:64], 1, images[64:], embed(method, images[64:]))
    assert set(actions) == {"0.0", "0.5", "1.0"}
    assert identical(method.state_dict(), snapshot) and identical(selector.state_dict(), actuator)
    assert torch.equal(rng, torch.get_rng_state())
    assert identical(gradients, [p.grad for p in method.parameters()])
    update(method, optimizer, images[32:64], 1, selector)
    torch.set_rng_state(rng)
    update(trial, opt, images[32:64], 1, private)
    assert identical(method.state_dict(), trial.state_dict())
    assert identical(optimizer.state_dict(), opt.state_dict())
    assert identical(selector.state_dict(), private.state_dict())
