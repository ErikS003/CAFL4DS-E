"""Bounded whole-stream admission schedules and source-level inference."""

from itertools import combinations
import statistics

from scipy.stats import t

SEEDS = (5003, 5009, 5011)
DRAWS = (51011, 51017, 51023)
HALF = tuple(sorted("".join("1" if k in pair else "0" for k in range(4)) for pair in combinations(range(4), 2)))
SCHEDULES = (*HALF, "1111", "0000")
PATTERNS = ("DNDN", "NDND", "DDND", "NNDN", "DNDN", "NDND", "DDND", "DDND", "NNDN", "NNDN")
PERIOD = 3072
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
    if any(len(g) != 60 for g in groups.values()):
        raise ValueError("Expected60 clips per lighting/scene cell")
    offsets = dict.fromkeys(groups, 0)
    streams = []
    for index, pattern in enumerate(PATTERNS):
        periods = []
        for light in pattern:
            ids = []
            for k in range(6):
                key = ("daytime" if light == "D" else "night", "city street" if k % 2 == 0 else "highway")
                ids.append(groups[key][offsets[key]]["clip_id"])
                offsets[key] += 1
            periods.append(ids)
        streams.append(
            {
                "name": f"stream_{index:02d}",
                "role": "development" if index < 4 else "heldout",
                "pattern": pattern,
                "periods": periods,
            }
        )
    return streams


def choose(means):
    if set(means) != set(HALF):
        raise ValueError("All six half schedules required")
    return min(HALF, key=lambda schedule: (-round(means[schedule], 12), schedule))


def interval(values, family=1):
    mean = statistics.fmean(values)
    width = float(t.ppf(1 - 0.05 / family / 2, len(values) - 1)) * statistics.stdev(values) / len(values) ** 0.5
    return {"mean_pp": mean, "ci95_pp": [mean - width, mean + width], "stream_values_pp": values}


def qualifies(contrasts):
    return (
        contrasts["balanced"]["mean_pp"] >= 0.5
        and contrasts["balanced"]["ci95_pp"][0] > 0
        and all(contrasts[m]["ci95_pp"][0] > -1 for m in ("A", "B"))
    )
