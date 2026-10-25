"""Apply explicitly recorded execution changes without rewriting scientific locks."""

import json
from pathlib import Path

from cafl4ds.single_image_long_gate import GATES
from scripts.run_short_buffer_experiment import digest

RETIRABLE = frozenset(
    {"independent_10", "independent_50", "hold"} | {f"{kind}_{gate}" for gate in GATES for kind in ("global", "clip")}
)


def active_specs(root, specs):
    """Retire explicitly named controls while protecting the gates, full and 25%."""
    path = root / "execution-amendment.json"
    if not path.exists():
        return specs
    amendment = json.loads(path.read_text())
    retired = set(amendment["retired_policies"])
    if not retired or not retired <= RETIRABLE:
        raise ValueError("Unsupported execution-policy amendment")
    if any(specs[name]["kind"] not in {"independent", "global", "clip", "hold"} for name in retired):
        raise ValueError("An execution amendment cannot retire a protected policy")
    return {name: spec for name, spec in specs.items() if name not in retired}


def validate_hashes(root, hashes):
    """Validate original data plus recorded source-only execution amendments."""
    path = root / "execution-amendment.json"
    amendment = json.loads(path.read_text()) if path.exists() else {}
    overrides = amendment.get("source_overrides", {})
    for name, expected in hashes.items():
        override = overrides.get(name)
        if override is not None:
            if Path(name).suffix != ".py" or override["original_sha256"] != expected:
                raise ValueError("Source amendment does not match the original frozen file")
            expected = override["amended_sha256"]
        assert digest(Path(name)) == expected, name
    for name, expected in amendment.get("additional_source_hashes", {}).items():
        assert digest(Path(name)) == expected, name


def capped_histories(root, histories, policy_count, seed_count):
    """Apply a user cap only at complete history/seed/policy comparison boundaries."""
    path = root / "execution-amendment.json"
    amendment = json.loads(path.read_text()) if path.exists() else {}
    cap = amendment.get("max_completed_trajectories")
    if cap is None:
        return histories
    per_history = policy_count * seed_count
    if type(cap) is not int or cap <= 0 or cap % per_history or cap > len(histories) * per_history:
        raise ValueError("Trajectory cap must retain complete policies and both seeds per history")
    return dict(list(histories.items())[: cap // per_history])
