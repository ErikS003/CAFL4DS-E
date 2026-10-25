"""Reconstruct the no-memory admission rerun independently of live decisions."""

from __future__ import annotations

import base64
import hashlib
import re
import shutil

import numpy as np
from omegaconf import OmegaConf
from threadpoolctl import threadpool_limits

from cafl4ds.health_calibration import geometry as absolute_geometry
from cafl4ds.measurements import cosine_drift, linear_cka
from scripts.natural_health_view import SIGNALS, write
from scripts.natural_health_view import verify as verify_probes
from scripts.report_natural_health_no_replay_rerun import FIGURES
from scripts.run_preaction_stopping import ROOT, check, digest, read
from scripts.verify_dnd_signal_diagnostic import geometry


def reconstruct(banks, horizon):
    """Rebuild credit before measurement and alarm persistence after the update."""
    credit, last, streak = 0, None, 0
    accepted, rates, events = [], [], []
    for step in range(1, horizon + 1):
        units = 4 if last is None else min(4, 1 + (step - last - 1) // 32)
        credit += units
        take = credit >= 4
        credit -= 4 if take else 0
        accepted.append(bool(take))
        rates.append(units / 4)
        if step <= 132 or (step - 132) % 8:
            continue
        values = geometry(banks["132"], banks[str(step)])
        unhealthy = bool((values >= 0.1 - 1e-12).any())
        streak = streak + 1 if unhealthy else 0
        fired = streak >= 2
        events.append(
            {"step": step, "values": values.tolist(), "unhealthy": unhealthy, "streak": streak, "alarm": fired}
        )
        if fired:
            last, streak, credit = step, 0, 0
    return accepted, rates, events


def audit(config, out, collected, arrays):
    """Check exact rerun, every frozen prediction, causal admission and current pixels."""
    assert read(out / "training-rerun-verification.json")["passed"]
    check(read(out / "collection-complete.json")["hashes"])
    assert not (out / "verification.json").exists()
    probe_result = verify_probes(OmegaConf.merge(config, {"source": config.output}), out, collected, arrays)
    assert all(
        probe_result["verification_work"][f"{k}_completed"] <= config.max_fits_per_kind for k in ("knn", "linear")
    )
    intake = read(ROOT / config.frames / "intake.json")
    pixels = [np.load(c["frames_path"], mmap_mode="r") for c in intake["clips"]]
    edges = np.r_[0, np.cumsum([len(x) for x in pixels])]
    input_hashes = []
    for step in range(4096):
        lo, hi = step * 32, (step + 1) * 32
        fragments = [
            a[max(0, lo - start) : min(len(a), hi - start)]
            for a, start, end in zip(pixels, edges[:-1], edges[1:], strict=True)
            if lo < end and hi > start
        ]
        rgb = np.concatenate(fragments)
        assert rgb.shape == (32, 64, 64, 3)
        input_hashes.append(hashlib.sha256(rgb.tobytes()).hexdigest())
    for arm in ("full", "adaptive"):
        row = collected["records"][arm]
        check(read(out / arm / "complete.json")["hashes"])
        with np.load(row["embedding_path"]) as banks:
            with threadpool_limits(limits=4):
                accepted, rates, events = reconstruct(banks, 4096)
            if arm == "adaptive":
                assert accepted == row["accepted"] and rates == row["rates"]
            else:
                assert all(row["accepted"]) and set(row["rates"]) == {1.0}
            assert len(events) == len(row["events"])
            for a, b in zip(events, row["events"], strict=True):
                assert all(a[k] == b[k] for k in ("step", "unhealthy", "streak", "alarm"))
                np.testing.assert_allclose(a["values"], b["values"], atol=1e-12, rtol=0)
        for trace, expected in zip(row["trace"], input_hashes, strict=True):
            assert trace["input_sha256"] == (expected if trace["accepted"] else None)
            assert trace["response"] is None if not trace["accepted"] else len(trace["response"]) == 2
        health = collected["audit_health"][arm]
        with np.load(out / f"audit-banks-{arm}.npz") as banks, threadpool_limits(limits=4):
            for i, point in enumerate(health["points"]):
                z, reference = banks[str(point)], banks["132"]
                expected = absolute_geometry(z)
                expected.update(cka=max(0, 1 - linear_cka(reference, z)), cosine=cosine_drift(reference, z))
                for name in SIGNALS:
                    value = health["values"][name][i]
                    if value is not None:
                        np.testing.assert_allclose(value, expected[name], atol=2e-6, rtol=2e-5)
    write(
        out / "independent-verification.json",
        {
            "passed": True,
            "all_current_input_pixels_verified": True,
            "no_historical_training_inputs": True,
            "live_alarm_and_integer_admission_independently_reconstructed": True,
            "original_and_audit_geometry_crosschecked": True,
            "all_frozen_readers_refitted": True,
            "exact_SSL_and_two_FT_refits": read(out / "training-rerun-verification.json"),
            "probe_verification": probe_result,
        },
    )
    return probe_result


def delivery(config, out):
    """Check immutable inputs, all six notebook images, links and actual resources."""
    assert read(out / "independent-verification.json")["passed"]
    for name in ("input_lock.json", "collection-complete.json", "complete.json"):
        check(read(out / name)["hashes"])
    notebook = read(ROOT / config.notebook)
    images = [v["data"]["image/png"] for c in notebook["cells"] for v in c.get("outputs", [])]
    assert len(images) == len(FIGURES) == 6
    for name, value in zip(FIGURES, images, strict=True):
        assert base64.b64decode(value) == (out / "figures" / f"{name}.png").read_bytes()
    for cell in notebook["cells"]:
        if cell["cell_type"] == "code":
            compile("".join(cell["source"]), str(config.notebook), "exec")
    for target in re.findall(r"\]\(([^)]+)\)", (out / "report.md").read_text(encoding="utf-8")):
        assert (out / target).resolve().exists(), target
    work = [read(p) for p in (out / "attempts").glob("*.json")]
    total = sum(r.get("ssl_started", 0) for r in work)
    assert total == sum(r.get("ssl_completed", 0) for r in work) <= 16384
    actual = sum(r["updates"] for r in read(out / "collection.json")["records"].values())
    assert total == 2 * actual
    assert read(out / "attempts/rerun.json")["replay_fits"] == 2
    logical_bytes = sum(p.stat().st_size for p in out.rglob("*") if p.is_file())
    assert logical_bytes < config.max_output_mib * 2**20
    assert shutil.disk_usage(out).free >= config.minimum_free_mib * 2**20
    write(
        out / "delivery-verification.json",
        {
            "passed": True,
            "physical_SSL_updates": total,
            "collection_SSL_updates": actual,
            "notebook_embedded_figures": 6,
            "replay_memory": False,
            "logical_output_bytes": logical_bytes,
            "free_bytes": shutil.disk_usage(out).free,
            "controller_qualified": False,
            "notebook_sha256": digest(ROOT / config.notebook),
        },
    )
    print({"delivery": "passed", "physical_SSL_updates": total, "figures": 6}, flush=True)
