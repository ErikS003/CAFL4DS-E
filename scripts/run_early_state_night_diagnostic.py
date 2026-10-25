"""CPU-only cached night-loss decomposition and association diagnostics."""

from __future__ import annotations

import argparse
import csv
import hashlib
import importlib.metadata
import json
import shutil
import sys
import time
from pathlib import Path

import numpy as np
import yaml
from threadpoolctl import threadpool_limits

from cafl4ds.early_state_night_diagnostic import (
    ASSAYS,
    COMPONENTS,
    FIELDS,
    SIGNALS,
    association_rows,
    components,
    query_summary,
    signal_values,
    transitions,
)

ROOT = Path(__file__).resolve().parents[1]
CONFIG = ROOT / "cafl4ds/configs/early_state_night_diagnostic.yaml"
DECLARATION = ROOT / "docs/project-plan/early-state-night-diagnostic-declaration.md"


def read(path):
    return json.loads(Path(path).read_text(encoding="utf-8"))


def plain(value):
    if isinstance(value, np.ndarray):
        return value.tolist()
    if isinstance(value, np.generic):
        return value.item()
    if isinstance(value, dict):
        return {str(k): plain(v) for k, v in value.items()}
    if isinstance(value, (list, tuple)):
        return [plain(v) for v in value]
    return value


def write(path, value):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(json.dumps(plain(value), indent=2, allow_nan=False) + "\n", encoding="utf-8")
    temporary.replace(path)


def digest(path):
    h = hashlib.sha256()
    with Path(path).open("rb") as handle:
        for block in iter(lambda: handle.read(2**20), b""):
            h.update(block)
    return h.hexdigest()


def csv_write(path, rows):
    with Path(path).open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


def reserve(study):
    out = ROOT / study["output"]
    assert shutil.disk_usage(out).free >= study["minimum_free_mib"] * 2**20, "Disk reserve reached"
    assert sum(p.stat().st_size for p in out.rglob("*") if p.is_file()) <= study["max_output_mib"] * 2**20


def check_lock(study):
    lock = read(ROOT / study["output"] / "input_lock.json")
    for path, h in lock["hashes"].items():
        assert digest(path) == h, path
    return lock


def freeze(study):
    out, source = ROOT / study["output"], ROOT / study["source"]
    out.mkdir(parents=True, exist_ok=True)
    assert not (out / "input_lock.json").exists(), "Preserve the exploratory analysis lock"
    verification = read(source / "verification.json")
    assert verification["passed"] and verification["summary_sha256"] == digest(source / "summary.json")
    assert read(source / "final-delivery-verification.json")["passed"]
    paths = [
        CONFIG,
        DECLARATION,
        Path(__file__),
        ROOT / "cafl4ds/early_state_night_diagnostic.py",
        ROOT / "scripts/report_early_state_night_diagnostic.py",
        ROOT / "tests/unit/test_early_state_night_diagnostic.py",
        out / "tests.log",
        out / "quality.log",
    ]
    paths += [
        source / n
        for n in (
            "verification.json",
            "final-delivery-verification.json",
            "summary.json",
            "plan.json",
            "analysis-arrays.npz",
            "input_lock.json",
            "evaluation/data.npz",
            "evaluation/data-role-manifest.json",
        )
    ]
    for complete in sorted((source / "readouts").rglob("complete.json")):
        data = read(complete)
        vpath = complete.parent / "verification.json"
        v = read(vpath)
        assert v["passed"] and v["complete_sha256"] == digest(complete)
        paths += [complete, vpath]
        for p, h in data["hashes"].items():
            assert digest(p) == h
            paths.append(Path(p))
    assert len(list((source / "readouts").rglob("complete.json"))) == 102
    for record in sorted((source / "branches").glob("*/*/record.json")):
        vpath = record.parent / "reproduction.json"
        assert read(vpath)["passed"] and read(vpath)["record_sha256"] == digest(record)
        paths += [record, vpath, record.parent / "bank-features.npz"]
    paths = sorted(set(p.resolve() for p in paths))
    for p in paths:
        if p.suffix in (".py", ".md", ".yaml"):
            target = out / "source_snapshot" / p.relative_to(ROOT.resolve())
            target.parent.mkdir(parents=True, exist_ok=True)
            shutil.copyfile(p, target)
    write(
        out / "input_lock.json",
        {
            "hashes": {str(p): digest(p) for p in paths},
            "config": study,
            "exploratory_after_known_39_outcomes": True,
            "before_this_diagnostics_derived_outputs": True,
            "frozen_unix": time.time(),
            "packages": {n: importlib.metadata.version(n) for n in ("numpy", "scipy", "matplotlib", "PyYAML")},
            "new_SSL_updates": 0,
            "new_readout_fits": 0,
            "download_bytes": 0,
        },
    )
    reserve(study)
    print({"frozen": True, "inputs": len(paths), "new_model_work": 0}, flush=True)


def analyze(study):
    out, source = ROOT / study["output"], ROOT / study["source"]
    assert not (out / "summary.json").exists(), "Preserve first diagnostic results"
    check_lock(study)
    reserve(study)
    with np.load(source / "analysis-arrays.npz") as z:
        correct = [z["intermediate"], z["correct"]]
        labels, domains = z["labels"], z["domains"]
    manifest = read(source / "evaluation/data-role-manifest.json")
    seeds = manifest["support_seeds"]
    query_records = [r for r in manifest["records"] if r["role"] == "fresh_query"]
    assert len(query_records) == len(labels)
    write(
        out / "query-index.json",
        {
            "support_seeds": seeds,
            "queries": [
                {"column": i, "clip_id": r["clip_id"], "label": int(labels[i]), "domain": int(domains[i])}
                for i, r in enumerate(query_records)
            ],
        },
    )
    shape = (3, 2, 2, 3, 4, 2, 12, 7)
    gains, losses, improvements, posts = [np.zeros(shape) for _ in range(4)]
    parent = np.zeros((3, 2, 3, 4, 12, 7))
    count_shape = (3, 2, 2, 3, 4, 2, len(labels))
    lost_counts, gained_counts = np.zeros(count_shape, np.uint8), np.zeros(count_shape, np.uint8)
    support_rows, case_rows, query_rows = [], [], []
    for ri, reader in enumerate(study["readers"]):
        for ai, age in enumerate(study["ages"]):
            before = correct[0][ri, ai, 0]
            np.testing.assert_array_equal(before, correct[1][ri, ai, 0])
            parent[ri, ai] = components(100.0 * before, labels, domains)
            for armi, arm in enumerate(study["arms"]):
                for ti, step in enumerate(study["checkpoints"]):
                    after = correct[ti][ri, ai, armi + 1]
                    lost, improved = transitions(before, after)
                    key = ri, ai, armi, slice(None), slice(None), ti
                    gains[key] = components(100.0 * (after.astype(float) - before), labels, domains)
                    losses[key] = components(100.0 * lost, labels, domains)
                    improvements[key] = components(100.0 * improved, labels, domains)
                    posts[key] = components(100.0 * after, labels, domains)
                    lost_counts[key] = lost.sum(-2)
                    gained_counts[key] = improved.sum(-2)
                    for si in range(3):
                        for fi in range(4):
                            base = {
                                "reader": reader,
                                "age": age,
                                "arm": arm,
                                "source": si + 1,
                                "future": fi + 1,
                                "checkpoint": step,
                            }
                            for ci, component in enumerate(COMPONENTS):
                                values = gains[ri, ai, armi, si, fi, ti, :, ci]
                                for di, seed in enumerate(seeds):
                                    support_rows.append(
                                        {
                                            **base,
                                            "support_seed": seed,
                                            "component": component,
                                            "parent_accuracy_pp": parent[ri, ai, si, fi, di, ci],
                                            "post_accuracy_pp": posts[ri, ai, armi, si, fi, ti, di, ci],
                                            "gain_pp": values[di],
                                            "newly_wrong_pp": losses[ri, ai, armi, si, fi, ti, di, ci],
                                            "newly_correct_pp": improvements[ri, ai, armi, si, fi, ti, di, ci],
                                        }
                                    )
                                case_rows.append(
                                    {
                                        **base,
                                        "component": component,
                                        "gain_pp": float(values.mean()),
                                        "parent_accuracy_pp": float(parent[ri, ai, si, fi, :, ci].mean()),
                                        "post_accuracy_pp": float(posts[ri, ai, armi, si, fi, ti, :, ci].mean()),
                                        "newly_wrong_pp": float(losses[ri, ai, armi, si, fi, ti, :, ci].mean()),
                                        "newly_correct_pp": float(improvements[ri, ai, armi, si, fi, ti, :, ci].mean()),
                                        "negative_support_draws": int((values < 0).sum()),
                                        "support_draws_below_minus_one": int((values < -1).sum()),
                                    }
                                )
                            cells = [(domains == d) & (labels == c) for d in (0, 1) for c in (0, 1)]
                            selected = [np.ones(len(labels), bool), domains == 0, domains == 1, *cells]
                            for component, mask in zip(COMPONENTS, selected, strict=True):
                                query_rows.append(
                                    {
                                        **base,
                                        "component": component,
                                        **query_summary(
                                            lost_counts[ri, ai, armi, si, fi, ti, mask],
                                            gained_counts[ri, ai, armi, si, fi, ti, mask],
                                            12,
                                        ),
                                    }
                                )
    signals = []
    for ai, age in enumerate(study["ages"]):
        for armi, arm in enumerate(study["arms"]):
            for si in range(3):
                for fi in range(4):
                    record = read(source / "branches" / f"b{si + 1}-a{age}-f{fi + 1}" / arm / "record.json")
                    for origin in study["origins"]:
                        signals.append(
                            {
                                "age": age,
                                "arm": arm,
                                "source": si + 1,
                                "future": fi + 1,
                                "origin": origin,
                                **signal_values(record, origin, study["trailing_updates"]),
                            }
                        )
    correlations, banks = [], []
    for ai, age in enumerate(study["ages"]):
        for armi, arm in enumerate(study["arms"]):
            for origin, endpoint, assay in ASSAYS:
                rows = [r for r in signals if r["age"] == age and r["arm"] == arm and r["origin"] == origin]
                sources, futures = [r["source"] for r in rows], [r["future"] for r in rows]
                for ci in (2, 5, 6):
                    risk = -gains[0, ai, armi, :, :, study["checkpoints"].index(endpoint), :, ci].mean(-1)
                    if assay == "remaining":
                        risk += gains[0, ai, armi, :, :, 0, :, ci].mean(-1)
                    for field in SIGNALS:
                        details = association_rows([r[field] for r in rows], risk.ravel(), sources, futures)
                        correlations += [
                            {
                                "age": age,
                                "arm": arm,
                                "origin": origin,
                                "endpoint": endpoint,
                                "assay": assay,
                                "component": COMPONENTS[ci],
                                "signal": field,
                                **r,
                            }
                            for r in details
                        ]
            for origin in study["origins"]:
                rows = [r for r in signals if r["age"] == age and r["arm"] == arm and r["origin"] == origin]
                for field in FIELDS:
                    banks += [
                        {"age": age, "arm": arm, "origin": origin, "signal": field, **r}
                        for r in association_rows(
                            [r[f"monitor_{field}"] for r in rows],
                            [r[f"audit_{field}"] for r in rows],
                            [r["source"] for r in rows],
                            [r["future"] for r in rows],
                        )
                    ]
    np.savez_compressed(
        out / "decomposition.npz",
        gains=gains,
        losses=losses,
        improvements=improvements,
        posts=posts,
        parent=parent,
        lost_counts=lost_counts,
        gained_counts=gained_counts,
        labels=labels,
        domains=domains,
    )
    for filename, rows in (
        ("support-effects.csv", support_rows),
        ("case-effects.csv", case_rows),
        ("query-transition-summary.csv", query_rows),
        ("signals.csv", signals),
        ("correlations.csv", correlations),
        ("bank-agreement.csv", banks),
    ):
        csv_write(out / filename, rows)
    focus = gains[0, 0, 1, :, 3]
    focus_summary = {
        "means_pp": focus.mean((0, 2)),
        "source_means_pp": focus.mean(2),
        "support_means_pp": focus.mean(0),
        "leave_source_out_means_pp": [np.delete(focus, i, 0).mean((0, 2)) for i in range(3)],
        "negative_night_support_draws": (focus.mean(0)[..., 2] < 0).sum(-1),
        "night_support_draws_below_minus_one": (focus.mean(0)[..., 2] < -1).sum(-1),
    }
    summary = {
        "exploratory": True,
        "new_SSL_updates": 0,
        "new_readout_fits": 0,
        "download_bytes": 0,
        "original_gate_unchanged": True,
        "causal_health_interpretation": False,
        "alarm_validated": False,
        "mean_gains_pp": gains.mean((3, 4, 6)),
        "future_means_pp": gains.mean((3, 6)),
        "source_means_pp": gains.mean((4, 6)),
        "future4_earlier_half_replay": focus_summary,
        "components": COMPONENTS,
        "signals": SIGNALS,
        "assays": ASSAYS,
        "rows": {
            "support": len(support_rows),
            "cases": len(case_rows),
            "query": len(query_rows),
            "signals": len(signals),
            "correlations": len(correlations),
            "bank_agreement": len(banks),
        },
    }
    write(out / "summary.json", summary)
    reserve(study)
    assert "torch" not in sys.modules, "This diagnostic must never import a learner runtime"
    print({"analyzed": True, "rows": summary["rows"], "new_model_work": 0}, flush=True)


def verify(study):
    import warnings
    from scipy.stats import ConstantInputWarning, pearsonr, spearmanr

    out, source = ROOT / study["output"], ROOT / study["source"]
    check_lock(study)
    summary = read(out / "summary.json")
    plan = read(source / "plan.json")
    seeds = read(source / "evaluation/data-role-manifest.json")["support_seeds"]
    with np.load(source / "evaluation/data.npz") as z:
        labels, domains = z["labels"][z["query"]], z["domains"][z["query"]]
    with np.load(out / "decomposition.npz") as z:
        arrays = {k: z[k] for k in z.files}
    with np.load(source / "analysis-arrays.npz") as z:
        original = [z["intermediate"], z["correct"]]
    seen, panels = set(), 0
    for p in plan["parents"]:
        ai, si, fi = study["ages"].index(p["age"]), p["block"] - 1, p["context"] - 1
        for armi, arm in enumerate(study["arms"]):
            for ti, step in enumerate(study["checkpoints"]):
                accumulated_lost, accumulated_gained = np.zeros(len(labels), int), np.zeros(len(labels), int)
                for di, seed in enumerate(seeds):
                    before_path = source / "readouts" / p["baseline"] / f"panel-{seed}.npz"
                    after_path = source / "readouts" / p["id"] / arm / str(step) / f"panel-{seed}.npz"
                    with np.load(before_path) as a, np.load(after_path) as b:
                        for ri, reader in enumerate(study["readers"]):
                            before = a["probabilities"][: len(labels)].argmax(-1) if ri == 0 else a[reader][0]
                            after = b["probabilities"][: len(labels)].argmax(-1) if ri == 0 else b[reader][0]
                            bc, ac = before == labels, after == labels
                            np.testing.assert_array_equal(bc, original[ti][ri, ai, 0, si, fi, di])
                            np.testing.assert_array_equal(ac, original[ti][ri, ai, armi + 1, si, fi, di])
                            cells = []
                            for d in (0, 1):
                                for c in (0, 1):
                                    m = (domains == d) & (labels == c)
                                    counts = np.array(
                                        [np.sum(bc[m]), np.sum(ac[m]), np.sum(bc[m] & ~ac[m]), np.sum(~bc[m] & ac[m])]
                                    )
                                    cells.append(100.0 * counts / int(m.sum()))
                            x = np.stack(cells)
                            expected = np.vstack([x.mean(0), x[:2].mean(0), x[2:].mean(0), x])
                            key = ri, ai, armi, si, fi, ti, di
                            for name, column in (("posts", 1), ("losses", 2), ("improvements", 3)):
                                np.testing.assert_allclose(arrays[name][key], expected[:, column], atol=1e-12, rtol=0)
                            np.testing.assert_allclose(
                                arrays["gains"][key], expected[:, 1] - expected[:, 0], atol=1e-12, rtol=0
                            )
                            np.testing.assert_array_equal(arrays["parent"][ri, ai, si, fi, di], expected[:, 0])
                    seen.update((before_path, after_path))
                for ri, reader in enumerate(study["readers"]):
                    a, b = original[ti][ri, ai, 0, si, fi], original[ti][ri, ai, armi + 1, si, fi]
                    accumulated_lost = np.sum((a == 1) & (b == 0), axis=0)
                    accumulated_gained = np.sum((a == 0) & (b == 1), axis=0)
                    key = ri, ai, armi, si, fi, ti
                    np.testing.assert_array_equal(arrays["lost_counts"][key], accumulated_lost)
                    np.testing.assert_array_equal(arrays["gained_counts"][key], accumulated_gained)
    panels = len(seen)
    assert panels == 1224
    np.testing.assert_allclose(arrays["improvements"] - arrays["losses"], arrays["gains"], atol=1e-12, rtol=0)
    for filename in ("support-effects.csv", "case-effects.csv", "query-transition-summary.csv"):
        with (out / filename).open(encoding="utf-8") as handle:
            for row in csv.DictReader(handle):
                ri = study["readers"].index(row["reader"])
                ai = study["ages"].index(int(row["age"]))
                armi = study["arms"].index(row["arm"])
                si, fi = int(row["source"]) - 1, int(row["future"]) - 1
                ti = study["checkpoints"].index(int(row["checkpoint"]))
                ci = COMPONENTS.index(row["component"])
                if filename == "query-transition-summary.csv":
                    mask = (
                        np.ones(len(labels), bool)
                        if ci == 0
                        else domains == ci - 1
                        if ci < 3
                        else (domains == (ci - 3) // 2) & (labels == (ci - 3) % 2)
                    )
                    a = arrays["lost_counts"][ri, ai, armi, si, fi, ti, mask].astype(float)
                    b = arrays["gained_counts"][ri, ai, armi, si, fi, ti, mask].astype(float)
                    assert int(row["queries"]) == len(a)
                    assert int(row["persistent_lost_queries"]) == int((a >= 9).sum())
                    assert int(row["persistent_gained_queries"]) == int((b >= 9).sum())
                    assert float(row["net_loss_query_fraction"]) == float(np.count_nonzero(a > b) / len(a))
                    positive = np.maximum(a - b, 0)
                    if positive.sum() == 0:
                        assert row["top_decile_positive_net_loss_share"] == ""
                    else:
                        take = (len(a) + 9) // 10
                        expected = float(sum(sorted(positive, reverse=True)[:take]) / positive.sum())
                        assert float(row["top_decile_positive_net_loss_share"]) == expected
                    continue
                key = ri, ai, armi, si, fi, ti, slice(None), ci
                expected = {
                    "parent_accuracy_pp": arrays["parent"][ri, ai, si, fi, :, ci],
                    "post_accuracy_pp": arrays["posts"][key],
                    "gain_pp": arrays["gains"][key],
                    "newly_wrong_pp": arrays["losses"][key],
                    "newly_correct_pp": arrays["improvements"][key],
                }
                if filename == "support-effects.csv":
                    di = seeds.index(int(row["support_seed"]))
                    for field, values in expected.items():
                        assert float(row[field]) == float(values[di])
                else:
                    for field, values in expected.items():
                        assert float(row[field]) == float(values.mean())
                    assert int(row["negative_support_draws"]) == int((expected["gain_pp"] < 0).sum())
                    assert int(row["support_draws_below_minus_one"]) == int((expected["gain_pp"] < -1).sum())
    old_summary = read(source / "summary.json")
    for ri, reader in enumerate(study["readers"]):
        for armi, arm in enumerate(study["arms"]):
            actual = arrays["gains"][ri, 0, armi, :, :, 1].mean((0, 1, 2))[:3]
            np.testing.assert_allclose(
                actual, old_summary["readers"][reader]["earlier_screens"][arm]["mean_pp"], atol=1e-12, rtol=0
            )
    signal_rows = list(csv.DictReader((out / "signals.csv").open(encoding="utf-8")))
    for row in signal_rows:
        origin = int(row["origin"])
        record = read(
            source / "branches" / f"b{row['source']}-a{row['age']}-f{row['future']}" / row["arm"] / "record.json"
        )
        trace = record["trace"][:origin]
        for bank in ("monitor", "audit"):
            for field in FIELDS:
                assert float(row[f"{bank}_{field}"]) == record["readings"][str(origin)][bank]["change"][field]
        for field, col in (("loss", 0), ("gradient", 1), ("step_l2", 2)):
            values = [v["response"][col] if col < 2 else v["encoder_step_l2"] for v in trace]
            assert float(row[f"cumulative_{field}"]) == float(np.asarray(values).mean())
            assert float(row[f"trailing_{field}"]) == float(np.asarray(values[-16:]).mean())
        assert float(row["encoder_path_length"]) == sum(v["encoder_step_l2"] for v in trace)
    verified_correlations = 0
    for filename in ("correlations.csv", "bank-agreement.csv"):
        with (out / filename).open(encoding="utf-8") as handle:
            for row in csv.DictReader(handle):
                selected = [r for r in signal_rows if all(r[k] == row[k] for k in ("age", "arm", "origin"))]
                if row["deletion"] != "all":
                    key = "source" if row["deletion"] == "leave_source" else "future"
                    selected = [r for r in selected if r[key] != row["group"]]
                assert len(selected) == int(row["n_cases"])
                if filename == "bank-agreement.csv":
                    x = [float(r[f"monitor_{row['signal']}"]) for r in selected]
                    y = [float(r[f"audit_{row['signal']}"]) for r in selected]
                else:
                    x = [float(r[row["signal"]]) for r in selected]
                    ai, armi = study["ages"].index(int(row["age"])), study["arms"].index(row["arm"])
                    ci = COMPONENTS.index(row["component"])
                    ti = study["checkpoints"].index(int(row["endpoint"]))
                    y = []
                    for r in selected:
                        si, fi = int(r["source"]) - 1, int(r["future"]) - 1
                        risk = -float(arrays["gains"][0, ai, armi, si, fi, ti, :, ci].mean())
                        if row["assay"] == "remaining":
                            risk += float(arrays["gains"][0, ai, armi, si, fi, 0, :, ci].mean())
                        y.append(risk)
                with warnings.catch_warnings():
                    warnings.simplefilter("ignore", ConstantInputWarning)
                    expected = (pearsonr(x, y).statistic, spearmanr(x, y).statistic)
                for field, value in zip(("pearson", "spearman"), expected, strict=True):
                    if np.isnan(value):
                        assert row[field] == ""
                    else:
                        np.testing.assert_allclose(float(row[field]), value, atol=1e-12, rtol=0)
                verified_correlations += 1
    assert verified_correlations == 8160
    np.testing.assert_allclose(arrays["gains"].mean((3, 4, 6)), summary["mean_gains_pp"], atol=1e-12, rtol=0)
    np.testing.assert_array_equal(arrays["gains"].mean((3, 6)), summary["future_means_pp"])
    np.testing.assert_array_equal(arrays["gains"].mean((4, 6)), summary["source_means_pp"])
    focus = arrays["gains"][0, 0, 1, :, 3]
    np.testing.assert_allclose(
        focus.mean((0, 2)), summary["future4_earlier_half_replay"]["means_pp"], atol=1e-12, rtol=0
    )
    for i in range(3):
        np.testing.assert_allclose(
            np.delete(focus, i, 0).mean((0, 2)),
            summary["future4_earlier_half_replay"]["leave_source_out_means_pp"][i],
            atol=1e-12,
            rtol=0,
        )
    assert not summary["causal_health_interpretation"] and not summary["alarm_validated"]
    assert all(summary[k] == 0 for k in ("new_SSL_updates", "new_readout_fits", "download_bytes"))
    check_lock(study)
    write(
        out / "verification.json",
        {
            "passed": True,
            "original_panels_reconstructed": panels,
            "independent_correlation_rows": verified_correlations,
            "summary_sha256": digest(out / "summary.json"),
            "no_new_model_work": True,
            "all_locked_inputs_unchanged": True,
        },
    )
    assert "torch" not in sys.modules
    print({"verified": True, "panels": panels, "correlation_rows": verified_correlations}, flush=True)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--stage", choices=("freeze", "analyze", "verify", "report", "all"), default="all")
    args = parser.parse_args()
    study = yaml.safe_load(CONFIG.read_text())
    with threadpool_limits(limits=4):
        for stage in ("freeze", "analyze", "verify", "report"):
            if args.stage not in (stage, "all"):
                continue
            if stage == "report":
                from scripts.report_early_state_night_diagnostic import report

                report(study)
            else:
                {"freeze": freeze, "analyze": analyze, "verify": verify}[stage](study)


if __name__ == "__main__":
    main()
