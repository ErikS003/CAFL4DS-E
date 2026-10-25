"""Private fixed-bank reconstruction measurements and a non-actuating loss alarm."""

from __future__ import annotations

import copy

import numpy as np
import torch


def bank_losses(model, images, seeds=(84211, 84223)):
    """Read per-image losses under common private views without changing learner RNG/modes."""
    if len(images) == 0 or len(images) % 32:
        raise ValueError("Complete 32-image measurement batches required")
    device = next(model.parameters()).device
    devices = [device.index or 0] if device.type == "cuda" else []
    modes = {name: module.training for name, module in model.named_modules()}
    readings = []
    try:
        for seed in seeds:
            with torch.random.fork_rng(devices=devices):
                torch.random.default_generator.manual_seed(seed)
                if devices:
                    torch.cuda.manual_seed(seed)
                model.train()
                with torch.no_grad():
                    readings.append(
                        torch.cat([model.per_sample_loss(batch.to(device)).cpu() for batch in images.split(32)]).numpy()
                    )
    finally:
        for name, module in model.named_modules():
            module.training = modes[name]
    values = np.stack(readings)
    if values.shape != (len(seeds), len(images)) or not np.isfinite(values).all() or np.any(values < 0):
        raise ValueError("Invalid fixed-bank loss measurement")
    return values


class ShadowLossAlarm:
    """Record T3 hysteresis on the declared observation clock without choosing any action."""

    def __init__(self):
        """Start before the fixed arrival-132 reference."""
        self.reference = None
        self.latched = False
        self.high_streak = self.low_streak = 0
        self.last_observation = self.last_alarm = None
        self.events = []

    @staticmethod
    def scheduled(step):
        """Use the same eight-arrival monitor cadence as the geometry experiments."""
        return step >= 132 and (step - 132) % 8 == 0

    def observe(self, step, loss):
        """Update shadow state on scheduled observations, with no allocation method."""
        if not self.scheduled(step):
            return False
        if not np.isfinite(loss) or loss <= 0:
            raise ValueError("Positive finite reference-bank loss required")
        if self.last_observation is not None and step <= self.last_observation:
            raise ValueError("Shadow observations must advance strictly")
        if step == 132:
            if self.reference is not None:
                raise ValueError("The fixed loss reference cannot be replaced")
            self.reference, self.last_observation = float(loss), step
            return False
        if self.reference is None:
            raise ValueError("Missing arrival-132 loss reference")
        risk = loss / self.reference - 1
        high, low = risk >= 0.02 - 1e-12, risk <= 0.01 + 1e-12
        fired = rearmed = False
        if self.latched:
            self.high_streak = 0
            self.low_streak = self.low_streak + 1 if low else 0
            if self.low_streak == 2:
                self.latched, self.low_streak, rearmed = False, 0, True
        else:
            self.low_streak = 0
            self.high_streak = self.high_streak + 1 if high else 0
            if self.high_streak == 2:
                self.latched, self.high_streak, fired = True, 0, True
                self.last_alarm = step
        self.last_observation = step
        self.events.append(
            {
                "step": step,
                "loss": float(loss),
                "risk": float(risk),
                "high": high,
                "low": low,
                "latched": self.latched,
                "high_streak": self.high_streak,
                "low_streak": self.low_streak,
                "alarm": fired,
                "rearmed": rearmed,
            }
        )
        return fired

    def state_dict(self):
        """Preserve all shadow state in parent snapshots."""
        return copy.deepcopy(vars(self))


def align_continuation(memory, landmark):
    """Align only the private sampler after the different first intervention blocks."""
    state = memory.state_dict()
    generator = torch.Generator().manual_seed(89201 + landmark)
    state["rng"] = generator.get_state()
    memory.load_state_dict(state)


def action_effects(current_scores, replay_scores):
    """Return domain-wise paired effects and each reader draw's conservative utility."""
    current, replay = np.asarray(current_scores), np.asarray(replay_scores)
    if current.shape != (3, 3) or replay.shape != (3, 3):
        raise ValueError("Three draws with balanced/day/night scores required")
    effects = replay - current
    return {
        "per_draw_effects": effects,
        "pooled_effects": effects.mean(0),
        "per_draw_utility": effects[:, 1:].min(1),
        "pooled_utility": effects.mean(0)[1:].min(),
    }
