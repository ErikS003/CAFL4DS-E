"""Fixed earlier-age screens, paired age contrasts and descriptive health."""

from __future__ import annotations

import numpy as np
from omegaconf import OmegaConf
from threadpoolctl import threadpool_limits

from cafl4ds.early_state_adaptation import ARMS, FIELDS, screen
from cafl4ds.functional_retention import plain
from cafl4ds.target_alignment_measurement import metric_means, resampling_counts
from scripts.prepare_early_state_adaptation import CONFIG, ROOT, read, write


def build_summary(correct, labels, domains, study):
    """Correctness: reader, age, algorithm, source, future, support, query.

    Algorithm order is parent/current/balanced. Baseline parent predictions are
    deliberately repeated over futures to maintain the matched contrasts.
    """
    expected = (3, 2, 3, 3, 4, 12, len(labels))
    assert correct.shape == expected and np.isin(correct, [0, 1]).all(), correct.shape
    panels, queries = resampling_counts(labels, domains, 12, study.bootstrap_draws, study.bootstrap_seed)
    output = {
        "readers": {},
        "prioritizes_health_mechanism_study": False,
        "controller_qualified": False,
        "causal_health_meaning_established": False,
        "population_confirmation": False,
        "training_RNG_replication": False,
        "one_pp_precision_claim": False,
        "conditional_sources": 3,
        "conditional_futures": 4,
    }
    for ri, kind in enumerate(("finetune", "knn", "linear")):
        effects = 100.0 * (correct[ri, :, 1:].astype(float) - correct[ri, :, :1])
        result = {
            "earlier_screens": {},
            "parent_accuracy_pp": metric_means(100.0 * correct[ri, :, 0], labels, domains),
            "late_effects_pp": {},
            "early_minus_late_pp": {},
        }
        for ai, arm in enumerate(ARMS):
            result["earlier_screens"][arm] = screen(effects[0, ai], labels, domains, panels, queries)
            result["late_effects_pp"][arm] = metric_means(effects[1, ai], labels, domains)
            result["early_minus_late_pp"][arm] = metric_means(effects[0, ai] - effects[1, ai], labels, domains)
        output["readers"][kind] = result
    output["prioritizes_health_mechanism_study"] = any(
        v["passes"] for v in output["readers"]["finetune"]["earlier_screens"].values()
    )
    return plain(output), panels, queries


def analysis(study):
    out = ROOT / study.output
    assert not (out / "summary.json").exists(), "Preserve first analysis"
    with np.load(ROOT / study.evaluation / "data.npz") as z:
        labels, domains = z["labels"][z["query"]], z["domains"][z["query"]]
    plan = read(out / "plan.json")
    seeds = read(ROOT / study.evaluation / "data-role-manifest.json")["support_seeds"]
    correct = np.zeros((3, 2, 3, 3, 4, 12, len(labels)), dtype=np.uint8)
    intermediate = np.zeros_like(correct)
    for p in plan["parents"]:
        age, source, future = study.ages.index(p["age"]), p["block"] - 1, p["context"] - 1
        for ai, arm in enumerate(("parent", *ARMS)):
            for step, destination in ((128, intermediate), (study.steps, correct)):
                directory = out / "readouts" / (p["baseline"] if arm == "parent" else f"{p['id']}/{arm}/{step}")
                for si, seed in enumerate(seeds):
                    with np.load(directory / f"panel-{seed}.npz") as z:
                        for ri, kind in enumerate(("finetune", "knn", "linear")):
                            pred = z["probabilities"][: len(labels)].argmax(-1) if kind == "finetune" else z[kind][0]
                            destination[ri, age, ai, source, future, si] = pred == labels
    summary, panels, queries = build_summary(correct, labels, domains, study)
    summary["intermediate_effects_pp"] = metric_means(
        100.0 * (intermediate[:, :, 1:].astype(float) - intermediate[:, :, :1]), labels, domains
    ).tolist()
    health = []
    for p in plan["parents"]:
        parent_scores = np.asarray(read(out / "readouts" / p["baseline"] / "complete.json")["scores"]["finetune"])
        for arm in ARMS:
            record = read(out / "branches" / p["id"] / arm / "record.json")
            for step in study.capability_points:
                scores = np.asarray(
                    read(out / "readouts" / p["id"] / arm / str(step) / "complete.json")["scores"]["finetune"]
                )
                for bank in ("monitor", "audit"):
                    health.append(
                        {
                            "parent": p["id"],
                            "source": p["block"],
                            "age": p["age"],
                            "future": p["context"],
                            "arm": arm,
                            "step": step,
                            "bank": bank,
                            "health": [record["readings"][str(step)][bank]["change"][f] for f in FIELDS],
                            "mean_encoder_step_l2": float(
                                np.mean([r["encoder_step_l2"] for r in record["trace"][:step]])
                            ),
                            "capability_change_pp": (scores - parent_scores).mean(0).tolist(),
                        }
                    )
    summary["health_descriptive"] = health
    np.savez_compressed(
        out / "analysis-arrays.npz",
        correct=correct,
        intermediate=intermediate,
        labels=labels,
        domains=domains,
        panels=panels,
        queries=queries,
    )
    write(out / "summary.json", summary)
    print(
        {
            "health_mechanism_priority": summary["prioritizes_health_mechanism_study"],
            "earlier_screens": summary["readers"]["finetune"]["earlier_screens"],
        },
        flush=True,
    )


if __name__ == "__main__":
    with threadpool_limits(limits=4):
        analysis(OmegaConf.load(CONFIG))
