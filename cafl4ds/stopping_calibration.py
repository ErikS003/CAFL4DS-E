"""Absolute SSL stopping allowances and prospective dwell-scale transfer."""

import statistics

from scipy.stats import t

BUDGETS = (0, 48, 96, 192, 288, 384)
SEEDS = (5003, 5009, 5011)
DRAWS = (51011, 51017, 51023)
PATTERNS = ("DNDNDNDNDNDN", "NDNDNDNDNDND", "DDDNNNDDDNNN", "NNNDDDNNNDDD", "DDDDDDNNNDDD", "NNNNNNDDDNNN")
READERS = ("finetune", "linear", "knn")
METRICS = ("balanced", "A", "B")


def allocation(clips):
    groups = {
        (light, scene): [
            c
            for c in clips
            if c["keyframe_attributes"]["timeofday"] == light and c["keyframe_attributes"]["scene"] == scene
        ]
        for light in ("daytime", "night")
        for scene in ("city street", "highway")
    }
    if any(len(g) != 36 for g in groups.values()):
        raise ValueError("Expected36 fresh clips per cell")
    offsets = dict.fromkeys(groups, 0)
    streams = []
    for k, pattern in enumerate(PATTERNS):
        periods = []
        for light in pattern:
            pair = []
            for scene in ("city street", "highway"):
                key = ("daytime" if light == "D" else "night", scene)
                pair.append(groups[key][offsets[key]]["clip_id"])
                offsets[key] += 1
            periods.append(pair)
        streams.append(
            {
                "name": f"fresh_{k:02d}",
                "role": "heldout",
                "dwell": ("short", "medium", "long")[k // 2],
                "pattern": pattern,
                "periods": periods,
            }
        )
    return streams


def choose(means):
    if set(means) != set(BUDGETS):
        raise ValueError("All six allowance outcomes required")
    return min(BUDGETS, key=lambda k: (-round(means[k], 12), k))


def interval(values, family=1):
    mean = statistics.fmean(values)
    width = float(t.ppf(1 - 0.05 / family / 2, len(values) - 1)) * statistics.stdev(values) / len(values) ** 0.5
    return {"mean_pp": mean, "ci95_pp": [mean - width, mean + width], "source_stream_values_pp": values}


def gates(budget, contrasts, savings):
    positive = contrasts[0]["balanced"]["mean_pp"] >= 0.5 and contrasts[0]["balanced"]["ci95_pp"][0] > 0
    full_ni = all(contrasts[384][m]["ci95_pp"][0] > -1 for m in METRICS)
    prior = (
        all(contrasts[192][m]["ci95_pp"][0] > -1 for m in METRICS)
        if budget <= 192
        else contrasts[192]["balanced"]["mean_pp"] >= 0.5 and contrasts[192]["balanced"]["ci95_pp"][0] > 0
    )
    return {
        "positive_ssl": positive,
        "full_noninferiority": full_ni,
        "prior_budget_comparison": prior,
        "work_saving": budget < 384 and savings >= 0.25,
        "combined": bool(positive and full_ni and prior and budget < 384 and savings >= 0.25),
    }
