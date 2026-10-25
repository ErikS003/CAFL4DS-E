"""Decode bounded subsets AFTER complete BDD partition and rare-scene filtering."""

from __future__ import annotations

from pathlib import Path

import torch
from PIL import Image
from torchvision import transforms

from cafl4ds.data.attributes import (
    AttributedImages,
    BDD100KSource,
    _count_categories,
    _driving_attributes,
    _filter_valid_records,
    _image_filename,
    _index_scenes,
    _iter_bdd_records,
    _rank_regimes,
)

Record = tuple[Path, tuple[str, str], str, dict[str, int]]


class BDDSubsetIndex:
    """Share full train metadata across warm/stream subsets within one experiment."""

    def __init__(self) -> None:
        """Initialize a per-run metadata cache, without decoded images."""
        self.records: dict[tuple[Path, Path], list[Record]] = {}

    def load(  # noqa: C901 - explicit full-partition and bounded-decoding stages
        self,
        source: BDD100KSource,
        seed: int,
        total: int | None = None,
        per_cell: int | None = None,
        scenes: list[str] | None = None,
        domains: list[str] | None = None,
        excluded_paths: set[str] | None = None,
    ) -> tuple[AttributedImages, list[str]]:
        """Apply original partition rules before subsetting; return image paths for provenance."""
        if source.max_images is not None or (total is None) == (per_cell is None):
            raise ValueError("Require uncapped full metadata and exactly one subset budget")
        if (total is not None and total < 1) or (per_cell is not None and per_cell < 1):
            raise ValueError("Subset budgets must be positive")
        images_dir, labels_file = source._paths()
        key = (images_dir, labels_file)
        if key not in self.records:
            valid = []
            for record in _iter_bdd_records(labels_file):
                attributes = _driving_attributes(record)
                if attributes is None:
                    continue
                timeofday, weather, scene = attributes
                path = images_dir / _image_filename(record)
                if path.is_file():
                    valid.append((path, (timeofday, weather), scene, _count_categories(record)))
            self.records[key] = valid
        full = _filter_valid_records(
            self.records[key],
            min_canary_count=source.min_canary_count,
            partition_role=source.partition_role,
            warm_fraction=source.warm_fraction,
            partition_seed=source.partition_seed,
        )
        filtered = [
            r
            for r in full
            if (scenes is None or r[2] in scenes)
            and (domains is None or f"{r[1][0]}·{r[1][1]}" in domains)
            and (excluded_paths is None or str(r[0]) not in excluded_paths)
        ]
        generator = torch.Generator().manual_seed(seed)
        selected: list[Record] = []
        if total is not None:
            if len(filtered) < total:
                raise ValueError("Insufficient warm reference images")
            selected = [filtered[i] for i in torch.randperm(len(filtered), generator=generator)[:total].tolist()]
        else:
            cells: dict[tuple[tuple[str, str], str], list[Record]] = {}
            for r in filtered:
                cells.setdefault((r[1], r[2]), []).append(r)
            if scenes is not None and domains is not None and len(cells) != len(scenes) * len(domains):
                raise ValueError("Missing declared BDD cell")
            for cell in sorted(cells):
                candidates = cells[cell]
                if len(candidates) < int(per_cell or 0):
                    raise ValueError(f"Insufficient BDD images in cell {cell}")
                selected.extend(
                    candidates[i] for i in torch.randperm(len(candidates), generator=generator)[:per_cell].tolist()
                )
        if not selected:
            raise ValueError("Empty BDD subset")
        order, regime_ids, regime_names = _rank_regimes({r[1] for r in selected})
        scene_ids, scene_names = _index_scenes({r[2] for r in selected})
        resize = transforms.Compose([transforms.Resize((source.img_size, source.img_size)), transforms.ToTensor()])
        images = []
        for r in selected:
            with Image.open(r[0]) as image:
                images.append(resize(image.convert("RGB")))
        return AttributedImages(
            torch.stack(images),
            torch.tensor([regime_ids[r[1]] for r in selected]),
            torch.tensor([scene_ids[r[2]] for r in selected]),
            order,
            regime_names,
            scene_names,
            object_categories=[r[3] for r in selected],
        ), [str(r[0]) for r in selected]
