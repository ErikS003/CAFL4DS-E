"""Audit the expanded budget, all paired statistics and canonical fit equivalence."""

from __future__ import annotations

import os

os.environ.setdefault("CUBLAS_WORKSPACE_CONFIG", ":4096:8")

from pathlib import Path

import numpy as np
import torch
from hydra.utils import instantiate
from omegaconf import OmegaConf
from threadpoolctl import threadpool_limits

from cafl4ds.artifacts import atomic_json
from cafl4ds.target_precision_budget import planned_seeds
from scripts.run_collapse_risk_regression import digest, read
from scripts.run_regime_windows_experiment import fine_tune_gpu
from scripts.run_retention_anchor_signal import setup
from scripts.run_short_buffer_experiment import encode
from scripts.verify_target_alignment_measurement import assert_nested, independent_statistics

ROOT = Path(__file__).resolve().parents[1]
CONFIG = ROOT / "cafl4ds/configs/target_precision_budget.yaml"


def main():
    config = OmegaConf.load(CONFIG)
    source, out = ROOT / config.source, ROOT / config.output
    assert not (out / "independent-verification.json").exists()
    lock, final, summary, manifest = (
        read(out / n) for n in ("input_lock.json", "collection.json", "summary.json", "data-role-manifest.json")
    )
    assert final["complete"] and final["panels"] in (64, 256)
    for path, expected in lock["hashes"].items():
        assert digest(path) == expected, path
    with np.load(out / "data.npz") as data:
        arrays = {k: data[k] for k in data.files}
    seeds = planned_seeds(read(source / "input_lock.json")["config"]["support_seeds"], 256, 62001)
    assert seeds == manifest["support_seeds"]
    pool, query = arrays["support_pool"], arrays["query"]
    assert len(pool) == 200 and len(query) == 3663
    ids = [r["clip_id"] for r in manifest["records"]]
    assert len(set(ids)) == len(ids) == 3863 and not set(ids).intersection(manifest["excluded_ids"])
    for i, row in enumerate(manifest["records"]):
        label = read(row["label_path"])["attributes"]
        assert label["weather"] == "clear"
        assert int(label["scene"] == "highway") == row["label"] == arrays["labels"][i]
        assert int(label["timeofday"] == "night") == row["domain"] == arrays["domains"][i]
    for position, seed in enumerate(seeds):
        generator = torch.Generator().manual_seed(seed)
        expected = []
        for c in (0, 1):
            for d in (0, 1):
                cell = pool[(arrays["labels"][pool] == c) & (arrays["domains"][pool] == d)]
                take = 12 + int(d == (position + 1) % 2)
                expected.extend(cell[torch.randperm(len(cell), generator=generator)[:take].numpy()])
        assert np.array_equal(expected, arrays[f"support_{seed}"])
        assert len(set(expected)) == 50 and not set(expected).intersection(query)
    checkpoints = [(u["source"], int(k)) for u in read(source / "collection.json")["units"] for k in u["bindings"]]
    pilot_replays = 0
    for name, step in checkpoints:
        directory = out / "runs" / name / f"at_{step}"
        for seed in seeds[: final["panels"]]:
            path = directory / f"panel_{seed}.npz"
            audit = read(directory / f"panel_{seed}.json")
            assert digest(path) == audit["sha256"] and audit["steps"] == 100
            assert audit["encoder_sha256"] == digest(source / "runs" / name / f"at_{step}/encoder.pt")
            if seed in seeds[:12]:
                with np.load(path) as p, np.load(source / "runs" / name / f"at_{step}/panel_{seed}.npz") as old:
                    value = np.concatenate([p["probabilities"][:1024], p["probabilities"][-400:]])
                    assert np.array_equal(value, old["probabilities"])
                pilot_replays += 1
    for stage in (64, 256):
        path = out / f"stage_{stage}.json"
        if not path.exists():
            continue
        stage_result = read(path)
        with np.load(out / f"resamples_{stage}.npz") as data:
            pc, qc = data["panel"], data["query"]
        rng = np.random.default_rng(config.bootstrap_seed + stage)
        assert np.array_equal(pc, rng.multinomial(stage, np.full(stage, 1 / stage), size=2000))
        y, d = arrays["labels"][query], arrays["domains"][query]
        for domain in (0, 1):
            for label in (0, 1):
                ix = np.flatnonzero((d == domain) & (y == label))
                assert np.array_equal(qc[:, ix], rng.multinomial(len(ix), np.full(len(ix), 1 / len(ix)), size=2000))
        for row in stage_result["records"]:
            a, b = [
                np.load(out / "runs" / row["source"] / f"at_{k}/predictions_{stage}.npz")["probabilities"]
                for k in (row["age"], row["age"] + 32)
            ]
            for step, aggregate in ((row["age"], a), (row["age"] + 32, b)):
                for index, seed in enumerate(seeds[:stage]):
                    with np.load(out / "runs" / row["source"] / f"at_{step}/panel_{seed}.npz") as saved:
                        assert np.array_equal(saved["probabilities"], aggregate[index])
            independently = independent_statistics(a[:, :3663], b[:, :3663], y, d, pc, qc)
            assert_nested(independently, row["fresh"])
        records = stage_result["records"]
        sf = np.mean([np.asarray(r["fresh"]["support"]["half_width"])[1:] <= 0.5 + 1e-8 for r in records])
        cf = np.mean([np.asarray(r["fresh"]["crossed"]["half_width"])[1:] <= 1 + 1e-8 for r in records])
        useful = {
            r["source"]
            for r in records
            if r["fresh"]["mean_gain_pp"][0] >= 0.25 and r["fresh"]["crossed"]["lower"][0] > 0
        }
        harmful = {r["source"] for r in records if min(r["fresh"]["crossed"]["upper"][1:]) < -1}
        gates = {
            "support_precision": bool(sf >= 0.8),
            "crossed_precision": bool(cf >= 0.8),
            "useful_sources": len(useful) >= 2,
            "harmful_sources": len(harmful) >= 2,
        }
        assert gates == stage_result["summary"]["gates"]
        qf = np.mean([np.asarray(r["fresh"]["query"]["half_width"])[1:] <= 1 + 1e-8 for r in records])
        assert np.isclose(qf, stage_result["interim_query_gate"]["query_precision_fraction"])
        print({"verified_stage": stage, "gates": gates, "query_fraction": qf}, flush=True)
    interim = read(out / "stage_64.json")
    assert (final["panels"] == 256) == interim["interim_query_gate"]["continue_to_final_budget"]
    assert final["records"] == read(out / f"stage_{final['panels']}.json")["records"]
    assert summary["measurement_qualified"] == (final["panels"] == 256 and all(gates.values()))
    attempts = [read(p) for p in (out / "attempts").glob("*.json")]
    assert all(a["complete"] for a in attempts)
    assert sum(a["completed_fits"] for a in attempts) == summary["unique_classifier_fits"] == final["panels"] * 36
    assert sum(a["started_fits"] for a in attempts) == sum(a["completed_fits"] for a in attempts)
    setup()
    images = torch.load(out / "images.pt", weights_only=True)
    for name, step, seed in (("fresh_00", 128, seeds[12]), ("fresh_05", 296, seeds[final["panels"] - 1])):
        encoder = instantiate(OmegaConf.create(lock["model_config"]).encoder).cuda()
        encoder.load_state_dict(torch.load(source / "runs" / name / f"at_{step}/encoder.pt", weights_only=True))
        selected = arrays[f"support_{seed}"]
        tuned, head, _ = fine_tune_gpu(
            encoder, images[selected], torch.from_numpy(arrays["labels"][selected]), seed + 25
        )
        prediction = []
        for ix in (np.concatenate([arrays["pilot_query"], arrays["legacy_query"]]), arrays["additional_query"]):
            z = encode(tuned, images[ix])
            with torch.no_grad():
                prediction.append(head(z.cuda()).softmax(-1).cpu().numpy())
        expected = np.concatenate([prediction[0][:1024], prediction[1], prediction[0][1024:]])
        with np.load(out / "runs" / name / f"at_{step}/panel_{seed}.npz") as saved:
            assert np.array_equal(expected, saved["probabilities"])
    paths = [
        p
        for p in out.rglob("*")
        if p.is_file()
        and "source_snapshot" not in p.parts
        and p.suffix in (".json", ".npz", ".pt")
        and p.name != "status.json"
    ]
    assert sum(p.stat().st_size for p in paths) < 2 * 2**30
    atomic_json(
        out / "independent-verification.json",
        {
            "passed": True,
            "panels": final["panels"],
            "exact_inherited_fit_replays": pilot_replays,
            "new_fit_replays": 2,
            "replay_supervised_steps": 200,
            "ssl_updates": 0,
            "preflight_replay_fits": 40,
            "helper_replay_fits": read(ROOT / config.preflight / "helper-replay.json")["fits"],
            "hashes": {str(p): digest(p) for p in paths},
            "verifier_sha256": digest(__file__),
        },
    )
    print({"passed": True, "pilot_replays": pilot_replays, "new_fit_replays": 2}, flush=True)


if __name__ == "__main__":
    with threadpool_limits(limits=4):
        main()
