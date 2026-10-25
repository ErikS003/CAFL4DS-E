"""Test private masks, skipped optimizer/RNG state and the complete causal arrival clock."""

from __future__ import annotations

import copy
import json
from pathlib import Path
from types import SimpleNamespace

import pytest
import torch
from omegaconf import OmegaConf
from torch import nn

from cafl4ds.artifacts import atomic_json
from cafl4ds.update_filtering import optimize_current, update_mask
from scripts import report_update_filtering as audit
from scripts import run_update_filtering as runner


class ToySSL(nn.Module):
    """Small stochastic objective whose RNG and Adam state are easy to inspect."""

    def __init__(self) -> None:
        """Create a real trainable encoder."""
        super().__init__()
        self.encoder = nn.Linear(2, 2)

    def training_step(self, images: torch.Tensor) -> torch.Tensor:
        """Consume RNG only when optimization is requested."""
        return (self.encoder(images) - torch.rand_like(images)).square().mean()


def test_masks_are_image_independent_exact_and_private() -> None:
    """Random masks do not change SSL RNG and have exact per-block accepted budgets."""
    state = torch.get_rng_state().clone()
    for seed in (5003, 5009, 5011, 5021, 5023):
        for policy in ("always", "fixed_half", "random_half"):
            mask = update_mask(policy, seed)
            assert mask == audit.independent_mask(policy, seed)
            assert [sum(mask[i : i + 144]) for i in range(0, 576, 144)] == [144 if policy == "always" else 72] * 4
    assert torch.equal(state, torch.get_rng_state())
    assert update_mask("random_half", 5003) != update_mask("random_half", 5009)
    assert update_mask("fixed_half", 5003)[:4] == [False, True, False, True]
    with pytest.raises(ValueError):
        update_mask("unknown", 5003)


def test_skips_preserve_model_optimizer_and_rng() -> None:
    """Skipping between updates is equivalent to omitting that call entirely."""
    model = ToySSL()
    optimizer = torch.optim.AdamW(model.parameters(), lr=0.001)
    current = torch.ones(32, 2)
    optimize_current(model, optimizer, current, True)
    before = copy.deepcopy(model.state_dict())
    state = copy.deepcopy(optimizer.state_dict())
    rng = torch.get_rng_state().clone()
    model.eval()
    assert optimize_current(model, optimizer, current, False) is None
    assert not model.training
    assert torch.equal(rng, torch.get_rng_state())
    assert all(torch.equal(value, model.state_dict()[key]) for key, value in before.items())
    for key, values in state["state"].items():
        assert all(torch.equal(value, optimizer.state_dict()["state"][key][name]) for name, value in values.items())
    control = copy.deepcopy(model)
    control_optimizer = torch.optim.AdamW(control.parameters(), lr=0.001)
    control_optimizer.load_state_dict(state)
    optimize_current(model, optimizer, current, True)
    torch.set_rng_state(rng)
    optimize_current(control, control_optimizer, current, True)
    assert all(torch.equal(value, control.state_dict()[key]) for key, value in model.state_dict().items())


@pytest.mark.parametrize("policy", ["always", "fixed_half", "random_half"])
def test_full_arrival_clock_and_corrupt_ledger_rejection(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, policy: str
) -> None:
    """Real optimization, skipped batches and complete saved states survive independent audit."""
    model = ToySSL()
    config = OmegaConf.create(
        {"optim": {"_target_": "torch.optim.AdamW", "lr": 0.001}, "recurrence": {"fine_tune_steps": 100}}
    )
    data = SimpleNamespace(images=torch.ones(18432, 2))

    def evaluate(*args: object, **kwargs: object) -> tuple[dict[str, object], torch.Tensor]:
        return {"step": args[3]}, torch.zeros(1)

    def save(method: ToySSL, path: Path) -> None:
        torch.save(method.state_dict(), path)

    monkeypatch.setattr(runner, "evaluate", evaluate)
    monkeypatch.setattr(runner.warmup, "save_well", save)
    result = runner.collect_arm(
        config,
        model,
        data,
        None,
        torch.arange(18432),
        5003,
        "recurrence",
        policy,
        {"step": 0},
        torch.zeros(1),
        tmp_path,
        lambda **f: None,
    )
    directory = tmp_path / f"recurrence_{policy}/seed_5003"
    assert result["budget"]["updates"] == (576 if policy == "always" else 288)
    audit.audit_ledger(directory, result, list(range(18432)), policy, 5003)
    assert [p["step"] for p in result["trajectory"]] == list(range(0, 577, 36))
    ledger = json.loads((directory / "ledger.json").read_text())
    ledger["steps"][0]["current"][0] = 32
    atomic_json(directory / "ledger.json", ledger)
    with pytest.raises(ValueError, match="Causal"):
        audit.audit_ledger(directory, result, list(range(18432)), policy, 5003)
