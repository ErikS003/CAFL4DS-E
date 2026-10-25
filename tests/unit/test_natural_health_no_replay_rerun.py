"""Check independently reconstructed admission over reset and recovery episodes."""

import numpy as np

from cafl4ds.natural_health_rate import NaturalHealthRate
from scripts.verify_natural_health_no_replay_rerun import reconstruct


def test_independent_credit_reconstruction_matches_live_monitor_over_long_clock():
    """Multiple warning episodes retain causal stage and admission-credit order."""
    rng = np.random.default_rng(629)
    reference = rng.normal(size=(48, 12))
    banks = {"132": reference}
    live = NaturalHealthRate()
    live.observe(132, reference)
    accepted, rates = [True] * 132, [1.0] * 132
    for step in range(133, 1025):
        take, rate = live.admit(step)
        accepted.append(take)
        rates.append(rate)
        if NaturalHealthRate.scheduled(step):
            banks[str(step)] = np.zeros_like(reference) if 140 <= step <= 180 or 508 <= step <= 596 else reference
            live.observe(step, banks[str(step)])
    a, r, events = reconstruct(banks, 1024)
    assert a == accepted and r == rates
    assert [e["step"] for e in events if e["alarm"]] == [e["step"] for e in live.events if e["alarm"]]
    assert set(r[200:500]) == {0.25, 0.5, 0.75, 1.0}


def test_auxiliary_future_bank_does_not_influence_preceding_admission():
    """Off-clock readouts cannot advance persistence or change past decisions."""
    rng = np.random.default_rng(630)
    reference = rng.normal(size=(40, 8))
    banks = {str(step): reference for step in range(132, 1025, 8)}
    before, _, _ = reconstruct(banks, 1024)
    banks["1024"] = np.zeros_like(reference)  # Off-clock auxiliary reading.
    after, _, _ = reconstruct(banks, 1024)
    assert before == after == [True] * 1024
