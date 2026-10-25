"""Prespecified point-budget and acquired-only stopping interpretations."""

from __future__ import annotations

import itertools

import numpy as np


def diagnostic(episodes, mode):
    if mode not in ("dual_point", "acquired_conditional"):
        raise ValueError("Known exploratory reference interpretation required")
    refs = ("warm", "acquired") if mode == "dual_point" else ("acquired",)

    def feasible(points, k):
        for p in points[: k + 1]:
            for ref in refs:
                values = p[ref]["mean_gain_pp"] if mode == "dual_point" else p[ref]["crossed"]["lower"]
                if min(values[1:]) < -1:
                    return False
        return True

    chosen, eligible = [], []
    for e in episodes:
        p = e["points"]
        eligible.append(feasible(p, 0))
        rewards = [
            q["acquired"]["mean_gain_pp"][0] if mode == "dual_point" else q["acquired"]["crossed"]["lower"][0]
            for q in p
        ]
        allowed = [k for k in range(5) if feasible(p, k) and (k == 0 or rewards[k] >= 0.25)]
        chosen.append(max(allowed, key=lambda k: (np.tanh(rewards[k] / 0.5), -k)) if allowed else 0)

    def outcomes(assignment):
        values = np.array(
            [e["points"][k]["acquired"]["mean_gain_pp"] for e, k in zip(episodes, assignment, strict=True)]
        )
        return {
            "blocks": list(map(int, assignment)),
            "ssl_updates": int(8 * sum(assignment)),
            "mean_gain_pp": values.mean(0),
            "source_gain_pp": values,
            "feasible_histories": [
                e["source"] for e, k in zip(episodes, assignment, strict=True) if feasible(e["points"], k)
            ],
        }

    indices = np.flatnonzero(eligible)
    permutations = list(itertools.permutations([chosen[i] for i in indices])) or [()]
    random = []
    for permutation in permutations:
        assigned = chosen.copy()
        for i, k in zip(indices, permutation, strict=True):
            assigned[i] = k
        random.append(outcomes(assigned))
    informed = outcomes(chosen)
    assert all(r["ssl_updates"] == informed["ssl_updates"] for r in random)
    return {
        "mode": mode,
        "informed_stop": informed,
        "always_train": outcomes([4] * len(episodes)),
        "always_hold": outcomes([0] * len(episodes)),
        "initially_feasible_histories": [e["source"] for e, ok in zip(episodes, eligible, strict=True) if ok],
        "matched_random_stop": {
            "assignment_permutations": len(random),
            "ssl_updates": informed["ssl_updates"],
            "mean_gain_pp": np.mean([r["mean_gain_pp"] for r in random], axis=0),
            "expected_feasible_histories": float(np.mean([len(r["feasible_histories"]) for r in random])),
        },
        "policy_qualified": False,
        "scope": "optimistic_reused_outcome_diagnostic_not_a_prospective_policy",
    }
