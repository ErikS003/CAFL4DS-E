"""Unit tests for the P1.0.2 regime stream (``cafl4ds.data.regime``)."""

from __future__ import annotations

import pytest
import torch

from cafl4ds.data.attributes import AttributedImages, AttributeSource, SyntheticAttributeSource
from cafl4ds.data.regime import RegimeStream, count_ordered_batches, iter_ordered_batches
from cafl4ds.data.streams import StreamBatch


def _source() -> SyntheticAttributeSource:
    """A modest network-free attributed source with enough images to reserve probes from."""
    return SyntheticAttributeSource(num_regimes=3, num_canary_classes=3, per_cell=20, img_size=12, long_tail=False)


class _CategorySource(AttributeSource):
    """A tiny attributed source carrying per-image object-category counts (a BDD label stand-in)."""

    def __init__(self, per_cell: int = 8) -> None:
        self.per_cell = per_cell

    @property
    def num_canary_classes(self) -> int:
        return 2

    def load(self) -> AttributedImages:
        images, era_key, canary, cats = [], [], [], []
        for regime in (0, 1):
            for scene in (0, 1):
                for _ in range(self.per_cell):
                    images.append(torch.rand(3, 8, 8))
                    era_key.append(regime)
                    canary.append(scene)
                    cats.append({"car": 1} if scene == 0 else {"person": 2})
        return AttributedImages(
            images=torch.stack(images),
            era_key=torch.tensor(era_key, dtype=torch.long),
            canary=torch.tensor(canary, dtype=torch.long),
            regime_order=[0, 1],
            regime_names={0: "daytime·clear", 1: "night·rainy"},
            canary_names={0: "highway", 1: "city street"},
            object_categories=cats,
        )


def test_eval_sets_are_balanced_on_the_canary_axis_and_disjoint() -> None:
    """The probe support/query are canary-labelled, balanced per class, and share no image."""
    stream = RegimeStream(_source(), batch_size=8, support_per_canary=5, query_per_canary=5)
    ev = stream.eval_sets
    assert set(ev.probe_support.labels.tolist()) == {0, 1, 2}
    for cls in (0, 1, 2):
        assert int((ev.probe_support.labels == cls).sum()) == 5
        assert int((ev.probe_query.labels == cls).sum()) == 5
    # support and query images are disjoint (no shared row)
    sup = {tuple(img.flatten().tolist()) for img in ev.probe_support.images}
    qry = {tuple(img.flatten().tolist()) for img in ev.probe_query.images}
    assert sup.isdisjoint(qry)


def test_condition_queries_are_balanced_and_disjoint_from_training() -> None:
    """Optional per-regime queries cover every canary and never enter the training stream."""
    stream = RegimeStream(
        _source(),
        batch_size=8,
        support_per_canary=5,
        query_per_canary=5,
        era_query_per_cell=2,
    )
    assert set(stream.eval_sets.per_era) == {0, 1, 2}
    held_out = set()
    for query in stream.eval_sets.per_era.values():
        assert {int(label) for label in query.labels} == {0, 1, 2}
        assert all(int((query.labels == label).sum()) == 2 for label in (0, 1, 2))
        held_out |= {tuple(image.flatten().tolist()) for image in query.images}
    training = {tuple(image.flatten().tolist()) for batch in stream for image in batch.images}
    assert held_out.isdisjoint(training)


def test_independent_evaluation_source_keeps_all_training_images_and_eval_disjoint() -> None:
    """An external eval split supplies probes without reserving or leaking into the train stream."""
    train = SyntheticAttributeSource(
        num_regimes=3, num_canary_classes=3, per_cell=20, img_size=12, long_tail=False, seed=1
    )
    evaluation = SyntheticAttributeSource(
        num_regimes=3, num_canary_classes=3, per_cell=12, img_size=12, long_tail=False, seed=2
    )
    stream = RegimeStream(
        train,
        evaluation_source=evaluation,
        batch_size=16,
        support_per_canary=3,
        query_per_canary=3,
        era_query_per_cell=2,
    )

    assert sum(batch.images.shape[0] for batch in stream) == train.load().images.shape[0]
    assert stream.eval_num_canary_classes == 3
    training = {tuple(image.flatten().tolist()) for batch in stream for image in batch.images}
    held_out = {
        tuple(image.flatten().tolist())
        for eval_set in (
            stream.eval_sets.probe_support,
            stream.eval_sets.probe_query,
            *stream.eval_sets.per_era.values(),
        )
        for image in eval_set.images
    }
    assert training.isdisjoint(held_out)


def test_default_walk_delivers_eras_contiguously_in_order() -> None:
    """With no block_size, batches walk the regimes in order, each era a contiguous run."""
    stream = RegimeStream(_source(), batch_size=8, support_per_canary=5, query_per_canary=5)
    eras = [b.era for b in stream]
    assert eras == sorted(eras)  # non-decreasing → contiguous walk
    assert set(eras) == {0, 1, 2}
    assert eras[0] == 0 and eras[-1] == 2


def test_every_batch_is_label_free_and_single_era() -> None:
    """Each delivered batch is images-only and carries exactly one era tag."""
    stream = RegimeStream(_source(), batch_size=8)
    for b in stream:
        assert isinstance(b, StreamBatch)
        assert b.images.ndim == 4
        assert isinstance(b.era, int)


def test_len_matches_iteration() -> None:
    """The advertised length equals the number of batches actually yielded."""
    stream = RegimeStream(_source(), batch_size=8)
    assert len(stream) == len(list(stream))


def test_block_size_interleaves_regimes() -> None:
    """block_size round-robins the regimes, so eras recur instead of forming one run each."""
    stream = RegimeStream(_source(), batch_size=4, block_size=4)
    eras = [b.era for b in stream]
    # contiguous would be sorted; interleaving breaks that (an era recurs after a later era)
    assert eras != sorted(eras)
    # every era still appears
    assert set(eras) == {0, 1, 2}


def test_block_size_rejects_non_positive() -> None:
    """A non-positive correlation knob is a clear error."""
    with pytest.raises(ValueError, match="block_size must be a positive"):
        RegimeStream(_source(), block_size=0)


def test_max_train_per_regime_caps_the_stream() -> None:
    """Capping training-per-regime shortens the stream vs. the uncapped default."""
    full = RegimeStream(_source(), batch_size=8)
    capped = RegimeStream(_source(), batch_size=8, max_train_per_regime=3)
    assert sum(b.images.shape[0] for b in capped) < sum(b.images.shape[0] for b in full)
    assert sum(b.images.shape[0] for b in capped) <= 3 * capped.num_eras


def test_regime_order_override_changes_the_walk() -> None:
    """Passing an explicit regime order re-sequences the eras."""
    stream = RegimeStream(_source(), batch_size=8, regime_order=[2, 0, 1])
    eras = [b.era for b in stream]
    # era ids are walk positions, so a contiguous walk still reads 0,1,2 — but the *regimes* behind
    # them are reordered; the first era now holds regime 2's images. We assert the walk stays
    # contiguous and complete under the override.
    assert eras == sorted(eras)
    assert set(eras) == {0, 1, 2}


def test_is_deterministic_in_seed() -> None:
    """Same seed reproduces the exact batch sequence (images + era tags)."""
    a = list(RegimeStream(_source(), batch_size=8, seed=3))
    b = list(RegimeStream(_source(), batch_size=8, seed=3))
    assert len(a) == len(b)
    assert all(torch.equal(x.images, y.images) and x.era == y.era for x, y in zip(a, b, strict=True))


def _probe_images(stream: RegimeStream) -> tuple[set[tuple[float, ...]], set[tuple[float, ...]]]:
    """The stream's held-out (support, query) probe sets as hashable image contents."""
    ev = stream.eval_sets
    sup = {tuple(img.flatten().tolist()) for img in ev.probe_support.images}
    qry = {tuple(img.flatten().tolist()) for img in ev.probe_query.images}
    return sup, qry


def test_reservation_is_fixed_across_drive_seeds() -> None:
    """Different drive seeds reserve the *identical* probe set — the reservation is decoupled from seed.

    The A1 fix: the held-out canary probe set is drawn from ``canary_seed`` (fixed), not the drive
    ``seed``, so every drive in a seed ensemble (and the warm well) holds out the same images and no
    drive ever trains on another drive's probe queries.
    """
    a = RegimeStream(_source(), batch_size=8, support_per_canary=5, query_per_canary=5, seed=0)
    b = RegimeStream(_source(), batch_size=8, support_per_canary=5, query_per_canary=5, seed=7)
    assert _probe_images(a) == _probe_images(b)  # identical held-out probe set across drive seeds
    # ... while the within-regime training order genuinely varies with the drive seed
    walk_a = [tuple(img.flatten().tolist()) for batch in a for img in batch.images]
    walk_b = [tuple(img.flatten().tolist()) for batch in b for img in batch.images]
    assert set(walk_a) == set(walk_b)  # same training pool (identical reservation removed the same rows)
    assert walk_a != walk_b  # but a different frame order


def test_canary_seed_redraws_the_reservation() -> None:
    """Changing ``canary_seed`` (independently of the drive seed) re-draws the held-out probe set."""
    a = RegimeStream(_source(), batch_size=8, support_per_canary=5, query_per_canary=5, canary_seed=0)
    b = RegimeStream(_source(), batch_size=8, support_per_canary=5, query_per_canary=5, canary_seed=1)
    assert _probe_images(a) != _probe_images(b)


def test_num_eras_counts_regimes_with_training_data() -> None:
    """num_eras reflects the regimes that actually contribute training batches."""
    stream = RegimeStream(_source(), batch_size=8)
    assert stream.num_eras == 3


def test_raises_when_a_canary_class_is_too_small_for_reservations() -> None:
    """Asking to reserve more probes than a canary class has is a clear error."""
    tiny = SyntheticAttributeSource(num_regimes=2, num_canary_classes=2, per_cell=3, long_tail=False)
    with pytest.raises(ValueError, match="reserved for probes"):
        RegimeStream(tiny, support_per_canary=20, query_per_canary=20)


def test_era_names_map_walk_positions_to_regime_names() -> None:
    """era_names resolves each delivered era (walk position) to its source regime name."""
    stream = RegimeStream(_source(), batch_size=8)
    assert stream.era_names == {0: "regime0", 1: "regime1", 2: "regime2"}
    # the map only covers eras that actually deliver batches
    assert set(stream.era_names) == {b.era for b in stream}


def test_era_composition_scene_histogram_sums_to_image_count() -> None:
    """Each era's scene histogram totals its delivered image count; no category labels → empty hist."""
    stream = RegimeStream(_source(), batch_size=8, support_per_canary=5, query_per_canary=5)
    comp = stream.era_composition()
    assert set(comp) == {b.era for b in stream}
    for entry in comp.values():
        assert sum(entry["scene_hist"].values()) == entry["n_images"]
        assert entry["category_hist"] == {}  # synthetic source carries no detection labels


def test_era_composition_aggregates_object_categories() -> None:
    """With detection labels, each era's category histogram sums the per-image box counts."""
    stream = RegimeStream(_CategorySource(per_cell=8), batch_size=4, support_per_canary=2, query_per_canary=2)
    comp = stream.era_composition()
    # each era has both scenes; per training image: highway→{car:1}, city street→{person:2}
    for era, entry in comp.items():
        highway = entry["scene_hist"].get("highway", 0)
        city = entry["scene_hist"].get("city street", 0)
        assert entry["category_hist"] == {"car": highway, "person": 2 * city}, era


def test_stationary_batches_cover_the_training_pool_without_probes() -> None:
    """The stationary diet reshuffles the exact training pool — same images, no era ordering."""
    stream = RegimeStream(_source(), batch_size=8, support_per_canary=5, query_per_canary=5)
    walk = list(stream)
    stationary = list(stream.stationary_batches())
    # every stationary batch is tagged era 0 (no regime ordering) ...
    assert {b.era for b in stationary} == {0}
    # ... and the image content matches the walk's training pool exactly (a permutation of it)
    walk_imgs = {tuple(img.flatten().tolist()) for b in walk for img in b.images}
    stat_imgs = {tuple(img.flatten().tolist()) for b in stationary for img in b.images}
    assert walk_imgs == stat_imgs


def test_iter_and_count_helpers_agree_and_never_split_eras() -> None:
    """The low-level batch helpers agree on count and never merge two eras into one batch."""
    order = [(0, 0), (0, 1), (0, 2), (1, 3), (1, 4)]  # 3 of era 0, 2 of era 1
    images = torch.arange(5 * 3 * 4 * 4, dtype=torch.float32).reshape(5, 3, 4, 4)
    batches = list(iter_ordered_batches(order, images, batch_size=2, drop_last=False))
    assert count_ordered_batches(order, batch_size=2, drop_last=False) == len(batches)
    # era 0 -> [2,1] (boundary flush), era 1 -> [2]  => eras 0,0,1
    assert [b.era for b in batches] == [0, 0, 1]
    assert [b.images.shape[0] for b in batches] == [2, 1, 2]
