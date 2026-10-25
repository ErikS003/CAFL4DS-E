"""Fixed-reference cumulative capability budgets and retrospective stopping headroom."""

from __future__ import annotations

import itertools

import numpy as np

from cafl4ds.block_anchor import BlockAnchorStudy

BLOCKS = (0, 1, 2, 3, 4)


def status(measurement, budget=1.0):
    lo, hi = (np.asarray(measurement["crossed"][k])[1:] for k in ("lower", "upper"))
    return [
        "within" if a >= -budget else "violation" if b < -budget else "unknown" for a, b in zip(lo, hi, strict=True)
    ]


def path_status(points, blocks):
    """Every visited boundary must protect both original and fixed acquired references."""
    states = [s for p in points[: blocks + 1] for ref in ("warm", "acquired") for s in status(p[ref])]
    return "violation" if "violation" in states else "unknown" if "unknown" in states else "within"


def summarize_assignment(episodes, assignment):
    outcomes = np.array([e["points"][k]["acquired"]["mean_gain_pp"] for e, k in zip(episodes, assignment, strict=True)])
    states = [path_status(e["points"], k) for e, k in zip(episodes, assignment, strict=True)]
    return {
        "blocks": list(map(int, assignment)),
        "ssl_updates": int(8 * sum(assignment)),
        "mean_gain_pp": outcomes.mean(0),
        "source_gain_pp": outcomes,
        "within_histories": int(states.count("within")),
        "violation_histories": int(states.count("violation")),
        "unknown_histories": int(states.count("unknown")),
        "path_status": states,
    }


class CumulativeBudgetStudy:
    def __init__(self, minimum_gain_pp=0.25, budget_pp=1.0, gain_scale_pp=0.5):
        if budget_pp != 1.0 or minimum_gain_pp != 0.25 or gain_scale_pp <= 0:
            raise ValueError("The inherited benefit margin and domain budgets stay fixed")
        self.minimum_gain_pp = minimum_gain_pp
        self.budget_pp = budget_pp
        self.gain_scale_pp = gain_scale_pp
        self.precision_reader = BlockAnchorStudy()

    def summarize(self, episodes, panels, final_panels):
        if len(episodes) != 6 or any(len(e["points"]) != 5 for e in episodes):
            raise ValueError("Six entire histories and five fixed stopping opportunities required")
        precision = {
            f"acquired_{8 * b}": self.precision_reader.precision([e["points"][b]["acquired"] for e in episodes])
            for b in BLOCKS[1:]
        }
        precision["warm"] = self.precision_reader.precision([p["warm"] for e in episodes for p in e["points"]])
        precision["local"] = self.precision_reader.precision([p["local"] for e in episodes for p in e["points"][1:]])
        interim_query_pass = all(v["query"]["passed"] for v in precision.values())
        precision_pass = panels == final_panels and all(
            v[k]["passed"] for v in precision.values() for k in ("support", "crossed")
        )
        choices, eligible, useful, preventable, accumulation = [], [], [], [], []
        for e in episodes:
            points = e["points"]
            initial_ok = path_status(points, 0) == "within"
            eligible.append(initial_ok)
            allowed = [
                b
                for b in BLOCKS
                if path_status(points, b) == "within"
                and (b == 0 or points[b]["acquired"]["crossed"]["lower"][0] >= self.minimum_gain_pp)
            ]
            # Reward is monotone in lower balanced gain; fewer updates break exact ties.
            choice = (
                max(
                    allowed,
                    key=lambda b: (np.tanh(points[b]["acquired"]["crossed"]["lower"][0] / self.gain_scale_pp), -b),
                )
                if allowed
                else 0
            )
            choices.append(choice)
            if choice > 0:
                useful.append(e["source"])
            if initial_ok and path_status(points, 4) == "violation" and path_status(points, choice) == "within":
                preventable.append(e["source"])
            for b in BLOCKS[1:]:
                # A cumulative violation can occur while the final local block is smaller than the budget.
                if "violation" in status(points[b]["acquired"]) and "violation" not in status(points[b]["local"]):
                    accumulation.append(
                        {
                            "source": e["source"],
                            "blocks": b,
                            "local_status": status(points[b]["local"]),
                            "cumulative_status": status(points[b]["acquired"]),
                        }
                    )
        policies = {
            "always_train": summarize_assignment(episodes, [4] * 6),
            "always_hold": summarize_assignment(episodes, [0] * 6),
            "informed_stop": summarize_assignment(episodes, choices),
        }
        eligible_i = np.flatnonzero(eligible)
        permutations = list(itertools.permutations([choices[i] for i in eligible_i])) or [()]
        random_records = []
        for permutation in permutations:
            assignment = choices.copy()
            for i, k in zip(eligible_i, permutation, strict=True):
                assignment[i] = k
            random_records.append(summarize_assignment(episodes, assignment))
        assert all(r["ssl_updates"] == policies["informed_stop"]["ssl_updates"] for r in random_records)
        random_gains = np.array([r["mean_gain_pp"] for r in random_records])
        random_violation = np.array([r["violation_histories"] for r in random_records])
        policies["matched_random_stop"] = {
            "assignment_permutations": len(random_records),
            "ssl_updates": policies["informed_stop"]["ssl_updates"],
            "mean_gain_pp": random_gains.mean(0),
            "assignment_range_pp": np.quantile(random_gains, [0, 0.5, 1], axis=0),
            "expected_violation_histories": float(random_violation.mean()),
            "expected_unknown_histories": float(np.mean([r["unknown_histories"] for r in random_records])),
            "scope": "exact_work_matched_random_stopping_over_initially_resolved_histories",
        }
        advantage = (
            np.asarray(policies["informed_stop"]["mean_gain_pp"]) - policies["matched_random_stop"]["mean_gain_pp"]
        )
        opportunity = len(useful) >= 2 and len(preventable) >= 2 and advantage[0] > 0
        return {
            "panels": panels,
            "final_panels": final_panels,
            "precision": precision,
            "interim_query_pass": bool(interim_query_pass),
            "precision_pass": bool(precision_pass),
            "initial_reference_resolved_histories": [
                e["source"] for e, ok in zip(episodes, eligible, strict=True) if ok
            ],
            "initial_reference_unresolved_histories": [
                e["source"] for e, ok in zip(episodes, eligible, strict=True) if not ok
            ],
            "useful_informed_histories": useful,
            "preventable_violation_histories": preventable,
            "cumulative_without_resolved_local_violation": accumulation,
            "policies": policies,
            "informed_advantage_over_random_pp": advantage,
            "retrospective_opportunity": bool(opportunity),
            "measurement_and_opportunity_ready": bool(precision_pass and opportunity),
            "policy_qualified": False,
            "predictor_fitted": False,
            "controller_activated": False,
            "decision": "prepare_separate_preaction_prediction_protocol"
            if precision_pass and opportunity
            else "stop_before_predictor_or_active_admission",
            "scope": "optimistic_retrospective_stopping_diagnostic_on_six_reused_histories",
            "unresolved_fallback": "hold_for_accounting_is_not_a_claim_of_original_reference_safety",
        }
