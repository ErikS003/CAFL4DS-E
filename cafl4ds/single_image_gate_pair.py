"""Deterministic cached temporal membership for one gate/full comparison."""

from collections import deque


def select_clips(records, history, pattern, start, offered, frames=96):
    """Preserve the history prefix, then fill its light pattern without repeats."""
    if start % frames or offered % frames or offered <= 0:
        raise ValueError("The stream must contain whole decoded clips")
    if not pattern or set(pattern) - {"D", "N"}:
        raise ValueError("Expected a day/night pattern")
    if len({row["clip_id"] for row in records}) != len(records):
        raise ValueError("Cached clip IDs must be unique")
    count = offered // frames
    prefix = sorted((r for r in records if r["history"] == history), key=lambda r: r["position"])
    if not prefix:
        raise ValueError("No inherited history prefix")
    selected = prefix[:count]
    used = {row["clip_id"] for row in selected}
    remaining = sorted(records, key=lambda r: (r["history"], r["position"], r["clip_id"]))
    pools = {
        light: deque(
            r for r in remaining if r["clip_id"] not in used and r["keyframe_attributes"]["timeofday"] == light
        )
        for light in ("daytime", "night")
    }
    for position in range(count):
        symbol = pattern[(start // frames + position) % len(pattern)]
        light = "daytime" if symbol == "D" else "night"
        if position < len(selected):
            if selected[position]["keyframe_attributes"]["timeofday"] != light:
                raise ValueError("Inherited prefix violates the day/night pattern")
        else:
            if not pools[light]:
                raise ValueError("Insufficient cached clips for the requested pattern")
            selected.append(pools[light].popleft())
    return selected
