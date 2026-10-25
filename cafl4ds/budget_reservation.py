"""Matched-budget spend-now versus reserve actions on chronological arrivals."""

from cafl4ds.short_buffer import capture
from cafl4ds.update_filtering import optimize_current

ARMS = ("spend", "reserve", "hold")
WINDOW = 1024


def checkpoints(pattern):
    transitions = [1024 * k for k in range(1, len(pattern)) if pattern[k] != pattern[k - 1]]
    first, second = transitions[:2]
    return {"before_shift": first - 512, "stable_second": first, "before_return": second - 512}


def accepted(offset, arm):
    if arm not in ARMS or offset < 0 or offset % 32:
        raise ValueError("Known action and aligned nonnegative offset required")
    if offset >= WINDOW:
        return (offset // 32) % 2 == 0
    return (arm == "spend" and offset < 512) or (arm == "reserve" and offset >= 512)


def run(model, optimizer, arrivals, arm, observe=None):
    if len(arrivals) < WINDOW or len(arrivals) % 64:
        raise ValueError("Require a complete intervention and continuation")
    records = []
    device = next(model.parameters()).device
    for offset in range(0, len(arrivals), 32):
        train = accepted(offset, arm)
        outcome = optimize_current(model, optimizer, arrivals[offset : offset + 32].to(device), train)
        records.append({"offset": offset, "accepted": train, "loss_gradient": outcome})
        clock = offset + 32
        if clock in (WINDOW, len(arrivals)) and observe:
            observe(clock, capture(model, optimizer))
    return records
