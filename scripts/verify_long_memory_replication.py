"""Independently audit source prefixes, primitive diets and downstream readouts."""

from __future__ import annotations

import os

os.environ.setdefault("CUBLAS_WORKSPACE_CONFIG", ":4096:8")

import argparse
import hashlib
from pathlib import Path

import numpy as np
import torch
from hydra.utils import instantiate
from omegaconf import OmegaConf

from cafl4ds.health_calibration import changes, embeddings, geometry
from cafl4ds.natural_health_rate import OrderedFrames
from cafl4ds.reservoir_action_replication import restore
from cafl4ds.target_alignment_measurement import metric_means
from scripts.evaluate_checkpoint_robustness import probe_predictions
from scripts.natural_health_view import write
from scripts.prepare_long_memory_replication import CONFIG, block_config
from scripts.run_adaptation_experiment import _method
from scripts.run_health_measurement import reference_images
from scripts.run_paired_update_response import fingerprint
from scripts.run_preaction_stopping import ROOT, check, digest, read
from scripts.run_retention_anchor_signal import setup
from scripts.run_target_precision_budget import fit_and_score


def source_check(study, block):
    config = block_config(study, block)
    intake = read(ROOT / config.frames / "intake.json")
    selection = read(ROOT / config.intake / "selection.json")
    exclusion = read(ROOT / config.intake / "prior-exclusions.json")
    check(exclusion["manifests"])
    excluded = set(exclusion["ids"])
    old = read(ROOT / study.selection_source)
    required = set(old["warm_clip_ids"]) | set(old["monitor_clip_ids"]) | set(old["evaluation_clip_ids"])
    prior = read(ROOT / study.prior / "input_lock.json")
    required.update(Path(row["path"]).stem for row in prior["monitor_reference"]["images"])
    required.update(row["clip_id"] for row in prior["audit_reference"]["images"])
    prior_positions = []
    for name in exclusion["manifests"]:
        rows = read(name)["clips"]
        if isinstance(rows, dict):
            required.update(rows)
            rows = list(rows.values())
        for row in rows:
            if isinstance(row, str):
                required.add(Path(row).stem)
            elif isinstance(row, dict):
                identity = row.get("clip_id") or row.get("id") or Path(row.get("path", "")).stem
                if identity:
                    required.add(identity)
                if "archive_position" in row:
                    prior_positions.append(int(row["archive_position"]))
    for earlier in study.blocks:
        if earlier >= block:
            break
        required.update(
            c["clip_id"] for c in read(ROOT / block_config(study, earlier).intake / "selection.json")["clips"]
        )
    assert required <= excluded, "An earlier reservation or protected population was omitted"
    assert max(prior_positions) == exclusion["start_after_archive_position"]
    expected = []
    for n, entry in enumerate(read(ROOT / study.archive_index)["entries"]):
        identity = Path(entry["path"]).stem
        if n > exclusion["start_after_archive_position"] and "/train/" in entry["path"] and identity not in excluded:
            expected.append((n, identity))
            if len(expected) == 384:
                break
    assert expected == [(c["archive_position"], c["clip_id"]) for c in selection["clips"]]
    assert [c["clip_id"] for c in intake["clips"]] == [c["clip_id"] for c in selection["clips"][: len(intake["clips"])]]
    assert not excluded.intersection(c["clip_id"] for c in intake["clips"])
    total = 0
    for clip in intake["clips"]:
        assert digest(clip["frames_path"]) == clip["frames_sha256"]
        data = np.load(clip["frames_path"], mmap_mode="r", allow_pickle=False)
        assert data.dtype == np.uint8 and data.shape[1:] == (64, 64, 3)
        assert len(data) == len(clip["frames"])
        previous = -1
        for k, (image, event) in enumerate(zip(data, clip["frames"], strict=True)):
            assert event["pts_seconds"] > previous and event["pts_seconds"] >= k / 16
            assert hashlib.sha256(image.tobytes()).hexdigest() == event["rgb64_sha256"]
            previous = event["pts_seconds"]
        total += len(data)
    assert total == intake["frames_used"] == 131072 and intake["batches"] == 4096
    return {
        "block": block,
        "frames": total,
        "used_clips": len(intake["clips"]),
        "reserved_clips": 384,
        "passed": True,
        "new_candidate_selection_uses_no_attributes_or_outcomes": True,
        "all_prior_reservations_and_protected_populations_independently_reconstructed": True,
    }


def primitive_diet(record, stream, final):
    sample = torch.Generator().manual_seed(73919)
    admit = torch.Generator().manual_seed(90101)
    identities, pixels = [], []
    arm = record["arm"]
    current_subsets, seen = [], 0
    for step, row in enumerate(record["trace"]):
        images = stream.batch(step)
        assert row["step"] == step + 1
        assert hashlib.sha256(images.tobytes()).hexdigest() == row["input_sha256"]
        if arm == "current":
            now, old = list(range(32)), []
        else:
            count = min(16, len(identities))
            old = torch.randperm(len(identities), generator=sample)[:count].tolist() if count else []
            now = sorted(torch.randperm(32, generator=sample)[: 32 - count].tolist()) if count else list(range(32))
        current_ids = [seen + i for i in now]
        past_ids = [identities[i] for i in old]
        assert row["diet"]["current_ids"] == current_ids and row["diet"]["replay_ids"] == past_ids
        assert all(i < seen for i in past_ids)
        mixed = np.concatenate([images[now], np.stack([pixels[i] for i in old])]) if old else images[now]
        assert hashlib.sha256(mixed.tobytes()).hexdigest() == row["train_sha256"]
        current_subsets.append(current_ids)
        for image in images:
            if arm == "fifo":
                identities.append(seen)
                pixels.append(image.copy())
                if len(identities) > 256:
                    identities.pop(0)
                    pixels.pop(0)
            elif arm == "reservoir":
                slot = len(identities) if seen < 256 else int(torch.randint(seen + 1, (1,), generator=admit))
                if slot < 256:
                    if seen < 256:
                        identities.append(seen)
                        pixels.append(image.copy())
                    else:
                        identities[slot], pixels[slot] = seen, image.copy()
            seen += 1
        if arm != "current":
            assert row["diet"]["oldest_resident_id"] == min(identities)
            assert row["diet"]["resident_frames"] == len(identities)
            assert row["diet"]["admitted_frames_seen"] == seen
            assert row["diet"]["replay_arrival_ages"] == [step - i // 32 for i in past_ids]
    assert seen == 131072
    if arm == "current":
        assert final["memory"] is None
    else:
        memory = final["memory"]
        assert memory["ids"] == identities
        np.testing.assert_array_equal(memory["images"].numpy(), np.stack(pixels))
        torch.testing.assert_close(
            memory["sample_rng" if arm == "reservoir" else "rng"], sample.get_state(), rtol=0, atol=0
        )
        if arm == "reservoir":
            torch.testing.assert_close(memory["admission_rng"], admit.get_state(), rtol=0, atol=0)
    return current_subsets


def verify(study):
    out = ROOT / study.study_root
    lock = read(out / "input_lock.json")
    check(lock["hashes"])
    source_audits = [source_check(study, b) for b in study.blocks]
    all_hashes, updates = {}, 0
    method = OmegaConf.create(lock["model_config"])
    template = _method(method).cuda()
    banks = {"monitor": reference_images(lock["monitor_reference"]), "audit": reference_images(lock["audit_reference"])}
    for block in study.blocks:
        stream = OrderedFrames(read(ROOT / block_config(study, block).frames / "intake.json")["clips"])
        for seed in study.ssl_seeds:
            paired = {}
            for arm in study.arms:
                directory = out / f"block{block}-seed{seed}-{arm}"
                record = read(directory / "record.json")
                assert read(directory / "training-verification.json")["passed"]
                final = torch.load(directory / "final_state.pt", weights_only=True, map_location="cpu")
                assert fingerprint(final) == record["final_fingerprint"]
                paired[arm] = primitive_diet(record, stream, final)
                model, _ = restore(template, method.optim, final["learner"])
                with np.load(directory / "bank-features.npz") as saved:
                    for bank, images in banks.items():
                        value = embeddings(model, images)
                        np.testing.assert_array_equal(value, saved[f"4096_{bank}"])
                        metrics = record["readings"]["4096"]["metrics"][bank]
                        for k, v in geometry(value).items():
                            assert v == metrics[k]
                        assert changes(saved[f"132_{bank}"], value) == metrics["change"]
                for stage in ("collection", "reproduction"):
                    work = read(directory / f"{stage}-work.json")
                    assert work["complete"] and work["ssl_started"] == work["ssl_completed"] == 4096
                    updates += work["ssl_started"]
                for name in ("record.json", "final_state.pt", "training-verification.json", "bank-features.npz"):
                    all_hashes[str(directory / name)] = digest(directory / name)
            assert paired["fifo"] == paired["reservoir"]
    assert updates == 221184
    images = torch.load(ROOT / study.evaluation / "images.pt", weights_only=True)
    with np.load(ROOT / study.evaluation / "data.npz") as saved:
        arrays = {k: saved[k] for k in saved.files}
    refits = {"knn": 0, "linear": 0, "finetune": 0}
    encoder = instantiate(method.encoder).cuda()
    audit_dir = out / "independent-reader-audits"
    audit_dir.mkdir(exist_ok=True)
    for job in read(out / "reader-jobs.json"):
        directory = out / "readouts" / job["id"]
        assert digest(job["path"]) == job["encoder_sha256"]
        completed = read(directory / "complete.json")
        check(completed["hashes"])
        audit_path = audit_dir / f"{job['id']}.json"
        if audit_path.exists():
            audit = read(audit_path)
            assert audit["completed_sha256"] == digest(directory / "complete.json")
            for k in refits:
                refits[k] += audit["refits"][k]
            all_hashes[str(audit_path)] = digest(audit_path)
            continue
        encoder.load_state_dict(torch.load(job["path"], weights_only=True, map_location="cpu"))
        wrapper = torch.nn.Module()
        wrapper.add_module("encoder", encoder)
        actual_features = embeddings(wrapper, images)
        assert hashlib.sha256(actual_features.tobytes()).hexdigest() == completed["features_sha256"]
        assert list(actual_features.shape) == completed["features_shape"]
        scores = {k: [] for k in refits}
        job_refits = dict.fromkeys(refits, 0)
        for draw in job["draws"]:
            selected = arrays[f"support_{draw}"]
            with np.load(directory / f"panel-{draw}.npz") as saved:
                for kind in ("knn", "linear"):
                    prediction = probe_predictions(
                        kind,
                        torch.from_numpy(actual_features[selected]),
                        torch.from_numpy(arrays["labels"][selected]),
                        torch.from_numpy(actual_features[arrays["query"]]),
                    ).numpy()
                    np.testing.assert_array_equal(prediction, saved[kind][0])
                    job_refits[kind] += 1
                    scores[kind].append(
                        metric_means(
                            100 * (prediction == arrays["labels"][arrays["query"]]),
                            arrays["labels"][arrays["query"]],
                            arrays["domains"][arrays["query"]],
                        )
                    )
                probabilities = saved["probabilities"]
                scores["finetune"].append(
                    metric_means(
                        100 * (probabilities[:3663].argmax(-1) == arrays["labels"][arrays["query"]]),
                        arrays["labels"][arrays["query"]],
                        arrays["domains"][arrays["query"]],
                    )
                )
                if draw == 61001 and (job["id"] == "warm" or job["id"].endswith("-4096")):
                    actual, _ = fit_and_score(encoder, images, arrays, draw)
                    np.testing.assert_array_equal(actual, probabilities)
                    job_refits["finetune"] += 1
        for k, values in scores.items():
            np.testing.assert_array_equal(values, completed["scores"][k])
            refits[k] += job_refits[k]
        write(
            audit_path,
            {
                "passed": True,
                "physical_encoder_embeddings_exact": True,
                "refits": job_refits,
                "completed_sha256": digest(directory / "complete.json"),
            },
        )
        all_hashes[str(audit_path)] = digest(audit_path)
        print({"independent_reader_verified": job["id"]}, flush=True)
    assert refits == {"knn": 9679, "linear": 9679, "finetune": 28}
    write(
        out / "independent-verification.json",
        {
            "passed": True,
            "sources": source_audits,
            "physical_scientific_SSL_updates": updates,
            "reader_refits": refits,
            "all_primitive_diets_pixels_and_RNG_verified": True,
            "hashes": all_hashes,
        },
    )


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--sources-only", action="store_true")
    args = parser.parse_args()
    setup()
    study = OmegaConf.load(CONFIG)
    if args.sources_only:
        for block in study.blocks:
            value = source_check(study, block)
            write(ROOT / block_config(study, block).intake / "source-preflight.json", value)
            print(value, flush=True)
    else:
        verify(study)


if __name__ == "__main__":
    main()
