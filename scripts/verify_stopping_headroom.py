"""Independent stopping headroom reconstruction with enumerated LP vertices."""

import argparse
import hashlib
import itertools
import json
from pathlib import Path

import numpy as np
from scipy.stats import t

from cafl4ds.artifacts import atomic_json
from scripts.run_short_buffer_experiment import digest


def read(path):
    return json.loads(path.read_text(encoding="utf-8"))


def vertex_optimum(scores, budget):
    average = scores.mean(0)
    doses = np.array([0, 4, 8, 16, 32])
    constraint = np.vstack((-np.eye(5), doses, -average[:, 1], -average[:, 2]))
    bound = np.concatenate((np.zeros(5), [budget, 1 - average[0, 1], 1 - average[0, 2]]))
    optimum = -np.inf
    for active in itertools.combinations(range(8), 4):
        matrix = np.vstack((np.ones(5), constraint[list(active)]))
        rhs = np.concatenate(([1], bound[list(active)]))
        try:
            q = np.linalg.solve(matrix, rhs)
        except np.linalg.LinAlgError:
            continue
        if np.all(constraint @ q <= bound + 1e-7):
            optimum = max(optimum, float(q @ average[:, 0]))
    assert np.isfinite(optimum)
    return optimum, constraint, bound


def choices(scores, guard):
    result = []
    for case in scores:
        candidates = [j for j in range(5) if not guard or np.min(case[j, 1:] - case[0, 1:]) >= -1 - 1e-10]
        best = max(case[j, 0] for j in candidates)
        result.append(next(j for j in candidates if case[j, 0] >= best - 1e-10))
    return np.array(result)


def check_evaluation(scores, picked, saved):
    selected = scores[np.arange(18), picked]
    delta = selected - scores[:, 0]
    doses = np.array([0, 4, 8, 16, 32])
    assert np.allclose(saved["scores"], selected, atol=1e-10, rtol=0)
    assert np.allclose(saved["mean_accuracy"], selected.mean(0), atol=1e-10, rtol=0)
    assert np.allclose(saved["mean_effect_vs_hold"], delta.mean(0), atol=1e-10, rtol=0)
    assert abs(saved["mean_updates"] - doses[picked].mean()) < 1e-10
    assert saved["dose_counts"] == {str(int(d)): int(sum(doses[picked] == d)) for d in doses}
    assert saved["domain_loss_gt_1_cases"] == np.any(delta[:, 1:] < -1 - 1e-10, axis=1).sum()
    assert abs(saved["worst_domain_effect_pp"] - delta[:, 1:].min()) < 1e-10
    assert abs(saved["mean_worst_domain_effect_pp"] - delta[:, 1:].min(1).mean()) < 1e-10


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, required=True)
    out = parser.parse_args().output.resolve()
    lock, summary = read(out / "input_lock.json"), read(out / "summary.json")
    assert all(digest(Path(p)) == h for p, h in lock["hashes"].items())
    assert read(out / "verification.json")["passed"]
    prior = Path(lock["prior"])
    rows = [r for r in read(prior / "collection.json")["records"] if r["role"] == "six_stream"]
    with np.load(out / "query_split.npz", allow_pickle=False) as a:
        query, labels, domains, half_a, half_b = [a[k].copy() for k in ("query", "labels", "domains", "A", "B")]
    root = Path(__file__).resolve().parents[1]
    with np.load(root / "outputs/regime-windows/20261013-gpu/evaluation_arrays.npz", allow_pickle=False) as original:
        assert np.array_equal(query, original["query"])
        assert np.array_equal(labels, original["labels"][query])
        assert np.array_equal(domains, original["domains"][query])
    expected_a = np.zeros(400, bool)
    for domain in (0, 1):
        for label in (0, 1):
            members = [i for i in range(400) if domains[i] == domain and labels[i] == label]
            assert len(members) == 100
            ordered = sorted(
                members,
                key=lambda i: hashlib.sha256(f"stopping-headroom:20261014:{int(query[i])}".encode()).hexdigest(),
            )
            expected_a[ordered[:50]] = True
    assert np.array_equal(expected_a, half_a) and np.array_equal(~expected_a, half_b)
    identities = [tuple(v) for v in lock["case_identities"]]
    assert len(identities) == 18 and len({v[0] for v in identities}) == 6
    tables = {r: {p: np.zeros((18, 5, 3)) for p in ("all", "A", "B")} for r in ("finetune", "linear", "knn")}
    for c, identity in enumerate(identities):
        selected = [r for r in rows if (r["stream"], r["checkpoint"]) == identity]
        assert len(selected) == 3 and len({r["seed"] for r in selected}) == 3
        for k, dose in enumerate((0, 4, 8, 16, 32)):
            for row in selected:
                directory = Path(row["landmarks"][str(dose)]["readout"])
                with np.load(directory / "predictions.npz", allow_pickle=False) as p:
                    for reader in tables:
                        for draw in (51011, 51017, 51023):
                            correct = p[f"{draw}_{reader}"] == labels
                            for part, mask in (("all", np.ones(400, bool)), ("A", half_a), ("B", half_b)):
                                for m, domain in enumerate((None, 0, 1)):
                                    membership = mask if domain is None else mask & (domains == domain)
                                    tables[reader][part][c, k, m] += (
                                        100 * np.sum(correct & membership) / sum(membership) / 9
                                    )
    with np.load(out / "score_tables.npz", allow_pickle=False) as saved:
        for reader, parts in tables.items():
            for part, values in parts.items():
                assert np.allclose(saved[f"{reader}_{part}"], values, atol=1e-10, rtol=0)
                # Selection uses the recorded rational aggregation order after equality.
                parts[part] = saved[f"{reader}_{part}"].copy()
    doses = np.array([0, 4, 8, 16, 32])
    checked_lp = 0
    for reader, parts in tables.items():
        result = summary["primary"] if reader == "finetune" else summary["secondary"][reader]
        full = parts["all"]
        common = int(np.flatnonzero(full[:, :, 0].mean(0) >= full[:, :, 0].mean(0).max() - 1e-10)[0])
        assert result["best_pooled_fixed_dose"] == doses[common]
        for k, saved in enumerate(result["fixed_curve"]):
            assert saved["dose"] == doses[k]
            check_evaluation(full, np.full(18, k), saved)
        for name, guard in (("unconstrained", False), ("retention_aware", True)):
            picked = choices(full, guard)
            saved = result["hindsight"][name]
            assert saved["choices"] == picked.tolist()
            check_evaluation(full, picked, saved["evaluation"])
            selected = full[np.arange(18), picked]
            assert np.allclose(saved["evaluation"]["mean_accuracy"], selected.mean(0), atol=1e-10, rtol=0)
            assert abs(saved["evaluation"]["mean_updates"] - doses[picked].mean()) < 1e-10
            delta = (selected - full[:, common]).reshape(6, 3, 3).mean(1)
            for i, m in enumerate(("balanced", "day", "night")):
                values = delta[:, i]
                width = t.ppf(0.975, 5) * values.std(ddof=1) / np.sqrt(6)
                assert np.allclose(
                    saved["vs_best_fixed"][m]["ci95_pp"],
                    [values.mean() - width, values.mean() + width],
                    atol=1e-10,
                    rtol=0,
                )
        for name, select, test in (("A_to_B", parts["A"], parts["B"]), ("B_to_A", parts["B"], parts["A"])):
            fold = result["query_half_folds"][name]
            picked = choices(select, True)
            assert fold["retention_choices"] == picked.tolist()
            assert fold["unconstrained_choices"] == choices(select, False).tolist()
            check_evaluation(test, picked, fold["oracle"])
            check_evaluation(test, choices(select, False), fold["unconstrained_oracle"])
            fixed = int(np.flatnonzero(select[:, :, 0].mean(0) >= select[:, :, 0].mean(0).max() - 1e-10)[0])
            assert fold["best_fixed_dose"] == doses[fixed]
            check_evaluation(test, np.full(18, fixed), fold["best_fixed"])
            q = np.asarray(fold["mixture_probabilities"])
            budget = doses[picked].mean()
            optimum, constraint, bound = vertex_optimum(select, budget)
            assert abs(q.sum() - 1) < 1e-8 and np.all(constraint @ q <= bound + 1e-7)
            assert abs(q @ select.mean(0)[:, 0] - optimum) < 1e-7
            checked_lp += 1
            selected = test[np.arange(18), picked]
            mixed = sum(q[k] * test[:, k] for k in range(5))
            assert np.allclose(selected.mean(0), fold["oracle"]["mean_accuracy"], atol=1e-10, rtol=0)
            assert np.allclose(mixed, fold["mixture_expected_case_scores"], atol=1e-10, rtol=0)
            assert abs(fold["mixture_mean_updates"] - q @ doses) < 1e-10
            assert (
                fold["mixture_domain_loss_gt_1_cases"]
                == np.any(mixed[:, 1:] - test[:, 0, 1:] < -1 - 1e-10, axis=1).sum()
            )
            gaps = (selected - mixed).reshape(6, 3, 3).mean(1)
            hold = (selected - test[:, 0]).reshape(6, 3, 3).mean(1)
            contrasts = {f"vs_mixture_{m}": gaps[:, i] for i, m in enumerate(("balanced", "day", "night"))}
            contrasts.update({f"vs_hold_{m}": hold[:, i + 1] for i, m in enumerate(("day", "night"))})
            for key, values in contrasts.items():
                for kind, family in (("ordinary", 1), ("adjusted", 10)):
                    width = t.ppf(1 - 0.05 / (2 * family), 5) * values.std(ddof=1) / np.sqrt(6)
                    saved = fold["contrasts"][key][kind]
                    assert abs(saved["mean_pp"] - values.mean()) < 1e-10
                    assert np.allclose(
                        saved["ci95_pp"], [values.mean() - width, values.mean() + width], atol=1e-10, rtol=0
                    )
            screen = {
                "material_gain": gaps[:, 0].mean() >= 0.5,
                "positive_adjusted_gain": fold["contrasts"]["vs_mixture_balanced"]["adjusted"]["ci95_pp"][0] > 0,
                "retention_bounds": all(
                    fold["contrasts"][k]["adjusted"]["ci95_pp"][0] >= -1
                    for k in contrasts
                    if k != "vs_mixture_balanced"
                ),
                "positive_sources": sum(gaps[:, 0] > 0) >= 4,
            }
            assert screen == fold["screen"] and fold["investigation_screen_passed"] == all(screen.values())
        assert result["investigation_screen_passed"] == all(
            f["investigation_screen_passed"] for f in result["query_half_folds"].values()
        )
    verification = {
        "passed": True,
        "prediction_sets": 2430,
        "score_scalars": 2430,
        "case_groups": 18,
        "history_groups": 6,
        "enumerated_LP_checks": checked_lp,
        "query_halves_verified": True,
        "new_SSL_updates": 0,
        "new_finetune_fits": 0,
        "hashes": len(lock["hashes"]),
    }
    atomic_json(out / "independent-verification.json", verification)
    print(json.dumps(verification))


if __name__ == "__main__":
    main()
