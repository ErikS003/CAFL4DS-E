"""Independent CPU verification of sustained replay decisions and readouts."""

from __future__ import annotations

import argparse
import hashlib
from pathlib import Path

import numpy as np
import torch
from scipy.stats import sem, t

from cafl4ds.artifacts import atomic_json
from scripts.run_short_buffer_experiment import digest
from scripts.verify_prediction_pilots import torch_geometry
from scripts.verify_temporal_batch_grouping import check, read, verify_scores

ROOT = Path(__file__).resolve().parents[1]
ARMS = ("current", "uniform25", "uniform50", "recent25", "recent50")
SEEDS = (5003, 5009, 5011)
PRIMARY = (("uniform25", "current"), ("uniform50", "uniform25"), ("recent25", "uniform25"))


def independent_plan(stream, seed, arm):
    """Reconstruct reservoir and selection from a chronological arrival loop."""
    base = seed + int(hashlib.sha256(stream.encode()).hexdigest()[:8], 16)
    rng = np.random.default_rng(base + 74111)
    memory, result, seen = list(range(512)), [], 512
    for step in range(192):
        if step < 8:
            result.append(
                {
                    "step": step,
                    "available": (step + 1) * 64,
                    "old": [],
                    "current": list(range(step * 64, step * 64 + 32)),
                }
            )
            continue
        block, position = divmod(step - 8, 8)
        origin = 512 + block * 512
        if position == 0:
            while seen < origin:
                slot = int(rng.integers(0, seen + 1))
                if slot <= 511:
                    memory[slot] = seen
                seen += 1
            pool = sorted(memory, reverse=True)[:128][::-1] if arm.startswith("recent") else memory.copy()
            sampler = np.random.default_rng(base + 94103 + block)
            permutation = sampler.permutation(pool).tolist()[:128]
        n = 0 if arm == "current" else {"25": 8, "50": 16}[arm[-2:]]
        old = [permutation[16 * position + k] for k in range(n)]
        # Integer linspace truncates each equally spaced nonnegative coordinate.
        current = [origin + position * 64 + int(k * 31 / (31 - n)) for k in range(32 - n)]
        result.append({"step": step, "available": origin + 512, "old": old, "current": current})
    return result


def verify_interval(values, saved, family):
    width = float(t.isf(0.05 / (2 * family), len(values) - 1) * sem(values))
    mean = float(np.mean(values))
    check(mean, saved["mean_pp"])
    check([mean - width, mean + width], saved["ci95_pp"])
    check(values, saved["source_values_pp"])


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, required=True)
    out = parser.parse_args().output.resolve()
    torch.set_num_threads(4)
    lock = read(out / "input_lock.json")
    if any(digest(Path(p)) != h for p, h in lock["hashes"].items()):
        raise ValueError("Frozen inputs changed")
    if not read(out / "verification.json")["passed"]:
        raise ValueError("Replay must pass first")
    rows = [read(Path(p)) for p in read(out / "collection.json")["records"]]
    lookup = {(r["stream"], r["seed"], r["arm"]): r for r in rows}
    groups = sorted(s["name"] for s in lock["streams"])
    if len(rows) != 90 or set(lookup) != {(g, s, a) for g in groups for s in SEEDS for a in ARMS}:
        raise ValueError("Incomplete or duplicate grid")
    warm = np.load(out / "warm_bank.npz")["embeddings"]
    with np.load(ROOT / "outputs/regime-windows/20261013-gpu/evaluation_arrays.npz") as a:
        labels, domains = a["labels"][a["query"]], a["domains"][a["query"]]
    warm_scores = read(out / "warm_scores.json")
    for seed in SEEDS:
        verify_scores(
            ROOT / f"outputs/coverage-admission/20261013-gpu/warm/seed_{seed}", labels, domains, warm_scores[str(seed)]
        )
    decisions, health_count, intervals = 0, 0, 0
    for row in rows:
        if not row["complete"] or not row["exact_replay"] or len(row["trace"]) != 192:
            raise ValueError("Incomplete trajectory")
        if any(digest(Path(p)) != h for p, h in row["artifact_hashes"].items()):
            raise ValueError("Changed evidence")
        expected = independent_plan(row["stream"], row["seed"], row["arm"])
        for target, actual in zip(expected, row["trace"], strict=True):
            if any(actual[k] != v for k, v in target.items()):
                raise ValueError("Independent causal selection differs")
            if (
                len(set(actual["old"] + actual["current"])) != 32
                or max(actual["old"] + actual["current"]) >= actual["available"]
            ):
                raise ValueError("Invalid available batch")
            if actual["old"] and max(actual["old"]) >= (actual["step"] // 8) * 512:
                raise ValueError("History is not strictly before block")
            m = actual["manipulation"]
            check(len(actual["old"]), m["old_count"])
            check(len(actual["current"]), m["current_count"])
            check(
                np.mean([(actual["step"] // 8) * 512 - i for i in actual["old"]]) if actual["old"] else 0,
                m["mean_old_age"],
            )
            check(np.mean([actual["available"] - 1 - i for i in actual["current"]]), m["mean_current_age_at_release"])
            decisions += 1
        for endpoint in row["health"].values():
            z = np.load(endpoint["path"])["embeddings"]
            values = torch_geometry(torch.from_numpy(z), torch.from_numpy(warm))
            for metric, value in values.items():
                check(value, endpoint["values"][metric])
                health_count += 1
        for endpoint in row["readouts"].values():
            directory = Path(endpoint["directory"])
            verify_scores(directory, labels, domains, endpoint["scores"])
            with np.load(directory / "predictions.npz") as a:
                for draw in (51011, 51017, 51023):
                    check(100 * np.mean(a[f"{draw}_finetune"] == labels), endpoint["panel_balanced_pp"][str(draw)])
        if row["arm"] == "current":
            legacy = read(
                ROOT
                / f"outputs/adaptive-mixture/20261014-gpu/runs/{row['stream']}/current/seed_{row['seed']}/trajectory.json"
            )
            if [r["loss_gradient"] for r in row["trace"]] != [r["loss_gradient"] for r in legacy["trace"]]:
                raise ValueError("Cached control trace differs")
            for name in row["readouts"]:
                if row["readouts"][name]["fingerprint"] != legacy["readouts"][name]["fingerprint"]:
                    raise ValueError("Cached control endpoint differs")
    for g in groups:
        for seed in SEEDS:
            if len({lookup[g, seed, a]["calibration_fingerprint"] for a in ARMS}) != 1:
                raise ValueError("Common calibration differs")
            for prefix in ("uniform", "recent"):
                small, large = lookup[g, seed, prefix + "25"], lookup[g, seed, prefix + "50"]
                for left, right in zip(small["trace"], large["trace"], strict=True):
                    if left["old"] != right["old"][:8]:
                        raise ValueError("Doses are not paired nested historical subsets")
            for dose in (25, 50):
                if [r["current"] for r in lookup[g, seed, f"uniform{dose}"]["trace"]] != [
                    r["current"] for r in lookup[g, seed, f"recent{dose}"]["trace"]
                ]:
                    raise ValueError("Current identities differ within dose")
    summary = read(out / "summary.json")
    for horizon, result in summary["horizons"].items():

        def quality(g, s, a, r, d):
            return warm_scores[str(s)][r][d] if a == "warm" else lookup[g, s, a]["readouts"][horizon]["scores"][r][d]

        for arm, readers in result["means"].items():
            for reader, domains_ in readers.items():
                for domain, value in domains_.items():
                    check(np.mean([quality(g, s, arm, reader, domain) for g in groups for s in SEEDS]), value)
        for key, readers in result["contrasts"].items():
            arm, comparator = key.split("_minus_")
            for reader, domains_ in readers.items():
                for domain, families in domains_.items():
                    values = [
                        float(
                            np.mean(
                                [
                                    quality(g, s, arm, reader, domain) - quality(g, s, comparator, reader, domain)
                                    for s in SEEDS
                                ]
                            )
                        )
                        for g in groups
                    ]
                    for name, family in (("ordinary", 1), ("primary_family3", 3), ("qualification_family18", 18)):
                        verify_interval(values, families[name], family)
                        intervals += 1
        interaction = [
            float(
                np.mean(
                    [
                        quality(g, s, "recent50", "finetune", "balanced")
                        - quality(g, s, "uniform50", "finetune", "balanced")
                        - quality(g, s, "recent25", "finetune", "balanced")
                        + quality(g, s, "uniform25", "finetune", "balanced")
                        for s in SEEDS
                    ]
                )
            )
            for g in groups
        ]
        verify_interval(interaction, result["secondary_interaction"], 1)
        if horizon == "terminal":
            leads = []
            for arm, comparator in PRIMARY:
                gate = summary["gates"][arm]

                def stat(a, c, d):
                    return result["contrasts"][f"{a}_minus_{c}"]["finetune"][d]["qualification_family18"]

                primary = stat(arm, comparator, "balanced")
                useful = stat(arm, "warm", "balanced")
                seed_values = [
                    np.mean(
                        [
                            quality(g, s, arm, "finetune", "balanced")
                            - quality(g, s, comparator, "finetune", "balanced")
                            for g in groups
                        ]
                    )
                    for s in SEEDS
                ]
                panel_values = [
                    np.mean(
                        [
                            lookup[g, s, arm]["readouts"][horizon]["panel_balanced_pp"][str(d)]
                            - lookup[g, s, comparator]["readouts"][horizon]["panel_balanced_pp"][str(d)]
                            for g in groups
                            for s in SEEDS
                        ]
                    )
                    for d in (51011, 51017, 51023)
                ]
                check(seed_values, gate["seed_gains_pp"])
                check(panel_values, gate["panel_gains_pp"])
                expected = {
                    "primary_family3_signal": result["contrasts"][f"{arm}_minus_{comparator}"]["finetune"]["balanced"][
                        "primary_family3"
                    ]["ci95_pp"][0]
                    > 0,
                    "practical_primary": primary["mean_pp"] >= 0.25 and primary["ci95_pp"][0] > 0,
                    "source_seed_support_stability": min(primary["source_values_pp"]) >= 0
                    and min(seed_values) > 0
                    and min(panel_values) > 0,
                    "useful_ssl": useful["mean_pp"] >= 0.1 and useful["ci95_pp"][0] > 0,
                    "domain_retention": all(
                        stat(arm, c, d)["ci95_pp"][0] >= -1 for c in ("current", "warm") for d in ("A", "B")
                    ),
                }
                if any(bool(v) != gate[k] for k, v in expected.items()):
                    raise ValueError("Gate differs")
                lead = all(expected[k] for k in expected if k != "primary_family3_signal")
                if bool(lead) != gate["development_lead"]:
                    raise ValueError("Lead differs")
                leads.append(lead)
            if bool(any(leads)) != summary["development_lead"]:
                raise ValueError("Overall lead differs")
    atomic_json(
        out / "independent-verification.json",
        {
            "passed": True,
            "decisions": decisions,
            "health_scalars": health_count,
            "readouts": 183,
            "intervals": intervals + 2,
            "raw_predictions_reconstructed": True,
            "source_seed_support_gates": True,
            "independent_gpu_retraining": False,
        },
    )
    print({"passed": True, "decisions": decisions, "health_scalars": health_count, "intervals": intervals + 2})


if __name__ == "__main__":
    main()
