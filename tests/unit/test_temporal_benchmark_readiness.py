"""Protect against false freshness from nested manifests and clip reuse."""

from scripts.audit_temporal_benchmark_readiness import index_fields, recorded_ids


def test_hash_map_keys_and_windows_paths_are_exposure() -> None:
    """Media hash-map keys must exclude their clip, regardless of path spelling."""
    manifest = {"hashes": {r"C:\\dataset\\train\\abc-123.JPG": "digest"}}
    assert recorded_ids(manifest) == {"abc-123"}
    assert recorded_ids({"video": "different/abc-123.mov"}) == {"abc-123"}


def test_explicit_clip_ids_are_exposure_but_unrelated_strings_are_not() -> None:
    """Clip reuse remains reuse when a lock omits original image paths."""
    assert recorded_ids({"clip_ids": ["abc-123", "def-456"], "sha256": "aabb"}) == {
        "abc-123",
        "def-456",
    }
    assert recorded_ids({"clips": [{"clip_id": "abc-123", "session_id": None}]}) == {"abc-123"}


def test_nested_source_indices_require_reconstruction() -> None:
    """An index-only lock cannot silently become evidence of cleared exposure."""
    manifest = {"evaluation": {"original_indices": [1, 2]}, "train_source_indices": [3]}
    assert index_fields(manifest) == {"original_indices", "train_source_indices"}
