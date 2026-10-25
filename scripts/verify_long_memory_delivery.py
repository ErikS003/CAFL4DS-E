"""Check the completed notebook, artifact hashes, work caps and manual visual audit."""

from __future__ import annotations

import base64
import re
import shutil

from omegaconf import OmegaConf

from scripts.natural_health_view import write
from scripts.prepare_long_memory_replication import CONFIG
from scripts.run_preaction_stopping import ROOT, check, digest, read


def main():
    study = OmegaConf.load(CONFIG)
    out = ROOT / study.study_root
    check(read(out / "input_lock.json")["hashes"])
    independent = read(out / "independent-verification.json")
    assert independent["passed"] and independent["physical_scientific_SSL_updates"] == 221184
    statistics = read(out / "statistics-verification.json")
    assert statistics["passed"] and statistics["summary_sha256"] == digest(out / "summary.json")
    presentation = read(out / "presentation-verification.json")
    assert presentation["passed"]
    check(presentation["hashes"])
    visual = read(out / "visual-qa.json")
    assert visual["passed"] and len(visual["hashes"]) == 6
    check(visual["hashes"])
    presentation["visual_inspection_pending"] = False
    write(out / "presentation-verification.json", presentation)
    notebook = read(ROOT / study.notebook)
    outputs = [o for c in notebook["cells"] for o in c.get("outputs", []) if "image/png" in o.get("data", {})]
    figures = sorted((out / "figures").glob("*.png"))
    assert len(figures) == len(outputs) == 6
    for figure, output in zip(figures, outputs, strict=True):
        assert figure.read_bytes() == base64.b64decode(output["data"]["image/png"])
    links = 0
    for document in (
        out / "report.md",
        ROOT / "docs/experiments/phase1/long-memory-replication.md",
        ROOT / "docs/experiments/latest-results/long-memory-replication.md",
    ):
        for link in re.findall(r"\]\(([^)]+)\)", document.read_text(encoding="utf-8")):
            if "://" not in link and not link.startswith("#"):
                assert (document.parent / link.split("#")[0]).exists(), (document, link)
                links += 1
    engineering = read(out / "engineering-reader-verification.json")
    assert engineering["passed"] and engineering["discarded_SSL_updates"] == 64
    assert engineering["actual_engineering_finetune_fits"] == 6
    engineering_counts = [
        read(out / "preflight.json"),
        read(out / "engineering-timing.json"),
        read(ROOT / "outputs/long-memory-engineering/20261018-reader/reader-worker-0.json"),
        read(ROOT / "outputs/long-memory-engineering/20261018-reader-compact/reader-worker-0.json"),
    ]
    assert sum(c.get("ssl_started", 0) for c in engineering_counts) == 64
    assert sum(c.get("finetune_started", 0) for c in engineering_counts) == 6
    source_bytes = sum(p.stat().st_size for p in out.rglob("*") if p.is_file() and any("-frames" in x for x in p.parts))
    output_bytes = sum(
        p.stat().st_size for p in out.rglob("*") if p.is_file() and not any("-frames" in x for x in p.parts)
    )
    assert output_bytes <= study.max_output_mib * 2**20
    assert shutil.disk_usage(out).free >= study.minimum_free_mib * 2**20
    downloads = sum(read(out / f"block{b}-intake/progress.json")["download_bytes"] for b in study.blocks)
    assert downloads <= 36 * 2**30
    write(
        out / "delivery-verification.json",
        {
            "passed": True,
            "all_embedded_images_exact": True,
            "visual_inspection_passed": True,
            "links_checked": links,
            "scientific_SSL": 221184,
            "engineering_SSL": 64,
            "scientific_output_logical_bytes": output_bytes,
            "retained_source_logical_bytes": source_bytes,
            "network_bytes": downloads,
            "summary_sha256": digest(out / "summary.json"),
            "notebook_sha256": digest(ROOT / study.notebook),
        },
    )
    print(read(out / "delivery-verification.json"), flush=True)


if __name__ == "__main__":
    main()
