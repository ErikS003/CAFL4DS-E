"""Independently reconstruct paired statistics and check physical state/fit replay."""

from __future__ import annotations

import os

os.environ.setdefault("CUBLAS_WORKSPACE_CONFIG", ":4096:8")

import copy
from pathlib import Path

import numpy as np
import torch
from hydra.utils import instantiate
from omegaconf import OmegaConf
from PIL import Image
from threadpoolctl import threadpool_limits

from cafl4ds.artifacts import atomic_json
from cafl4ds.short_buffer import capture
from cafl4ds.update_filtering import optimize_current
from scripts.run_adaptation_experiment import _method
from scripts.run_collapse_risk_regression import digest, read
from scripts.run_paired_update_response import fingerprint
from scripts.run_regime_windows_experiment import fine_tune_gpu
from scripts.run_retention_anchor_signal import gather, setup
from scripts.run_short_buffer_experiment import encode

ROOT = Path(__file__).resolve().parents[1]
CONFIG = ROOT / "cafl4ds/configs/target_alignment_measurement.yaml"


def independent_statistics(before, after, y, d, panel_counts, query_counts):
    difference = np.where(after.argmax(-1) == y, 100.0, 0.0) - np.where(before.argmax(-1) == y, 100.0, 0.0)
    pw = panel_counts / len(before)
    domain_panels, queries, crosses = [], [], []
    for domain in (0, 1):
        cells, qsamples, xsamples = [], [], []
        for label in (0, 1):
            ids = np.flatnonzero((d == domain) & (y == label))
            cell = difference[:, ids]
            counts = query_counts[:, ids] / len(ids)
            cells.append(cell.mean(1))
            qsamples.append(np.einsum("rq,q->r", counts, cell.mean(0)))
            xsamples.append(np.einsum("rp,pq,rq->r", pw, cell, counts, optimize=True))
        domain_panels.append(np.mean(cells, axis=0))
        queries.append(np.mean(qsamples, axis=0))
        crosses.append(np.mean(xsamples, axis=0))
    panel = np.stack([(domain_panels[0] + domain_panels[1]) / 2, *domain_panels], axis=1)

    def combine(a):
        return np.stack([(a[0] + a[1]) / 2, *a], axis=1)

    result = {
        "mean_gain_pp": panel.mean(0),
        "panel_gain_pp": panel,
        "positive_panel_fraction": (panel > 1e-8).mean(0),
        "negative_panel_fraction": (panel < -1e-8).mean(0),
    }
    for key, samples in (("support", pw @ panel), ("query", combine(queries)), ("crossed", combine(crosses))):
        bounds = np.percentile(samples, [2.5, 97.5], axis=0)
        result[key] = {"lower": bounds[0], "upper": bounds[1], "half_width": (bounds[1] - bounds[0]) / 2}
    return result


def assert_nested(expected, actual):
    if isinstance(expected, dict):
        assert set(expected) == set(actual)
        for key in expected:
            assert_nested(expected[key], actual[key])
    else:
        np.testing.assert_allclose(expected, actual, rtol=1e-10, atol=1e-10)


def main():
    config = OmegaConf.load(CONFIG)
    out, source = ROOT / config.output, ROOT / config.source
    assert not (out / "independent-verification.json").exists()
    lock, collection, summary = (read(out / n) for n in ("input_lock.json", "collection.json", "summary.json"))
    assert collection["complete"] and len(collection["units"]) == 6 and len(collection["records"]) == 18
    for path, expected in lock["hashes"].items():
        assert digest(path) == expected, path
    for unit in collection["units"]:
        assert all(digest(p) == h for p, h in unit["hashes"].items())
    manifest = read(out / "data-role-manifest.json")
    records = manifest["records"]
    ids = {r["clip_id"] for r in records}
    assert len(ids) == 1224 and not ids.intersection(manifest["excluded_ids"])
    images = torch.load(out / "images.pt", weights_only=True)
    with np.load(out / "data.npz") as data:
        arrays = {k: data[k] for k in data.files}
    with np.load(out / "resamples.npz") as data:
        resamples = {k: data[k] for k in data.files}
    assert len(arrays["query"]) == 1024 and len(arrays["legacy_query"]) == 400
    for index, row in enumerate(records):
        original = read(row["label_path"])
        assert int(original["attributes"]["scene"] == "highway") == arrays["labels"][index] == row["label"]
        assert int(original["attributes"]["timeofday"] == "night") == arrays["domains"][index] == row["domain"]
        assert original["attributes"]["weather"] == "clear"
        if index % 25 == 0:
            with Image.open(row["path"]) as opened:
                rgb = np.asarray(opened.convert("RGB").resize((64, 64), Image.Resampling.BILINEAR)).copy()
            expected = torch.from_numpy(rgb).permute(2, 0, 1).float() / 255
            assert torch.equal(expected, images[index])
    # Independent support reconstruction with the canonical cell order and local RNG.
    for position, seed in enumerate(config.support_seeds):
        g = torch.Generator().manual_seed(seed)
        wanted = []
        for label in (0, 1):
            for domain in (0, 1):
                cell = arrays["support_pool"][
                    (arrays["labels"][arrays["support_pool"]] == label)
                    & (arrays["domains"][arrays["support_pool"]] == domain)
                ]
                take = 12 + int(domain == (position + 1) % 2)
                wanted.extend(cell[torch.randperm(len(cell), generator=g)[:take].numpy()])
        assert np.array_equal(wanted, arrays[f"support_{seed}"])
        assert not set(wanted).intersection(arrays["query"])
    for role, prefix, seed in (
        ("query", "", config.bootstrap_seed),
        ("legacy_query", "legacy_", config.bootstrap_seed + 1),
    ):
        ix = arrays[role]
        rng = np.random.default_rng(seed)
        assert np.array_equal(
            resamples[prefix + "panel"], rng.multinomial(12, np.full(12, 1 / 12), size=config.bootstrap_repetitions)
        )
        for domain in (0, 1):
            for label in (0, 1):
                positions = np.flatnonzero((arrays["labels"][ix] == label) & (arrays["domains"][ix] == domain))
                expected = rng.multinomial(
                    len(positions), np.full(len(positions), 1 / len(positions)), size=config.bootstrap_repetitions
                )
                assert np.array_equal(resamples[prefix + "query"][:, positions], expected)
    for row in collection["records"]:
        before, after = [
            np.load(out / "runs" / row["source"] / f"at_{step}/predictions.npz")["probabilities"]
            for step in (row["age"], row["age"] + 32)
        ]
        for role, ix, sl, prefix in (
            ("fresh", arrays["query"], slice(0, 1024), ""),
            ("legacy", arrays["legacy_query"], slice(1024, None), "legacy_"),
        ):
            actual = independent_statistics(
                before[:, sl],
                after[:, sl],
                arrays["labels"][ix],
                arrays["domains"][ix],
                resamples[prefix + "panel"],
                resamples[prefix + "query"],
            )
            assert_nested(actual, row[role])
    sf = np.mean([np.asarray(r["fresh"]["support"]["half_width"])[1:] <= 0.5 + 1e-8 for r in collection["records"]])
    cf = np.mean([np.asarray(r["fresh"]["crossed"]["half_width"])[1:] <= 1 + 1e-8 for r in collection["records"]])
    useful = {
        r["source"]
        for r in collection["records"]
        if r["fresh"]["mean_gain_pp"][0] >= 0.25 and r["fresh"]["crossed"]["lower"][0] > 0
    }
    harmful = {r["source"] for r in collection["records"] if min(r["fresh"]["crossed"]["upper"][1:]) < -1}
    gates = {
        "support_precision": bool(sf >= 0.8),
        "crossed_precision": bool(cf >= 0.8),
        "useful_sources": len(useful) >= 2,
        "harmful_sources": len(harmful) >= 2,
    }
    assert gates == summary["gates"] and all(gates.values()) == summary["measurement_qualified"]
    assert sorted(useful) == summary["useful_sources"] and sorted(harmful) == summary["harmful_sources"]
    assert sum(u["classifier_fits"] for u in collection["units"]) == summary["downstream_fits"] == 432
    assert sum(u["reconstruction_ssl_updates"] for u in collection["units"]) == 1776
    print({"stage": "statistics_and_roles_verified", "gates": gates}, flush=True)
    setup()
    model_config = OmegaConf.create(lock["model_config"])
    torch.manual_seed(11)
    initial = _method(model_config).cuda()
    intake = read(ROOT / config.frames / "intake.json")
    stream = intake["streams"][0]
    arrivals = gather(stream, {c["clip_id"]: c for c in intake["clips"]}, list(range(296 * 32)))
    torch.manual_seed(config.ssl_seed)
    model = copy.deepcopy(initial)
    optimizer = instantiate(model_config.optim, params=model.parameters())
    old = read(source / "runs" / stream["name"] / f"seed_{config.ssl_seed}" / "parent-path.json")
    points = {age + k for age in config.parents.values() for k in (0, 32)}
    for step in range(297):
        if str(step) in old["checkpoints"]:
            assert fingerprint(capture(model, optimizer)) == old["checkpoints"][str(step)]["fingerprint"]
        if step in points:
            saved = torch.load(out / "runs" / stream["name"] / f"at_{step}/encoder.pt", weights_only=True)
            assert all(torch.equal(saved[k], v.cpu()) for k, v in model.encoder.state_dict().items())
        if step < 296:
            assert (
                list(optimize_current(model, optimizer, arrivals[step * 32 : (step + 1) * 32].cuda(), True))
                == old["trace"][step]
            )
    query = np.concatenate([arrays["query"], arrays["legacy_query"]])
    for name, step, seed in (("fresh_00", 128, config.support_seeds[0]), ("fresh_05", 296, config.support_seeds[-1])):
        encoder = instantiate(model_config.encoder).cuda()
        encoder.load_state_dict(torch.load(out / "runs" / name / f"at_{step}/encoder.pt", weights_only=True))
        selected = arrays[f"support_{seed}"]
        with torch.random.fork_rng(devices=[0]):
            tuned, head, audit = fine_tune_gpu(
                encoder, images[selected], torch.from_numpy(arrays["labels"][selected]), seed + 25
            )
            z = encode(tuned, images[query])
            with torch.no_grad():
                prediction = head(z.cuda()).softmax(-1).cpu().numpy()
        with np.load(out / "runs" / name / f"at_{step}/panel_{seed}.npz") as p:
            assert np.array_equal(prediction, p["probabilities"])
        old_audit = read(out / "runs" / name / f"at_{step}/panel_{seed}.json")
        for key, value in audit.items():
            assert old_audit[key] == value
    output_paths = [
        p
        for p in out.rglob("*")
        if p.is_file() and "source_snapshot" not in p.parts and p.name not in ("status.json", "report.md")
    ]
    assert sum(p.stat().st_size for p in output_paths) < 512 * 2**20
    atomic_json(
        out / "independent-verification.json",
        {
            "passed": True,
            "paired_contrasts": 18,
            "domain_contrasts": 36,
            "support_draws": 12,
            "fresh_queries": 1024,
            "new_support_pool": 200,
            "physical_checkpoint_replay_updates": 296,
            "exact_fit_replays": 2,
            "replayed_supervised_steps": 200,
            "hashes": {str(p): digest(p) for p in output_paths},
            "verifier_sha256": digest(__file__),
        },
    )
    print({"passed": True, "checkpoint_replay_updates": 296, "exact_fit_replays": 2}, flush=True)


if __name__ == "__main__":
    with threadpool_limits(limits=4):
        main()
