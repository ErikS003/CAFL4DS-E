"""Bounded frame loading and transactional checkpoints for long temporal runs."""

from collections import OrderedDict
from pathlib import Path

import numpy as np
import torch

from cafl4ds.artifacts import atomic_json
from scripts.run_short_buffer_experiment import digest


class FrameStream:
    """Expose dense-equivalent CPU slices while caching at most two uint8 clips."""

    def __init__(self, paths, frames=96):
        self.paths = [Path(p) for p in paths]
        self.frames = frames
        self.cache = OrderedDict()

    def __len__(self):
        return len(self.paths) * self.frames

    def clip(self, index):
        if index not in self.cache:
            pixels = np.load(self.paths[index], mmap_mode="r", allow_pickle=False)
            if pixels.shape != (self.frames, 64, 64, 3) or pixels.dtype != np.uint8:
                raise ValueError("Unexpected decoded RGB clip format")
            self.cache[index] = pixels
        self.cache.move_to_end(index)
        while len(self.cache) > 2:
            self.cache.popitem(last=False)
        return self.cache[index]

    def __getitem__(self, item):
        scalar = isinstance(item, int)
        if scalar:
            index = item if item >= 0 else len(self) + item
            if not 0 <= index < len(self):
                raise IndexError(item)
            item = slice(index, index + 1)
        start, stop, step = item.indices(len(self))
        if step != 1:
            raise ValueError("Only contiguous temporal slices are supported")
        pieces = []
        position = start
        while position < stop:
            clip, offset = divmod(position, self.frames)
            count = min(stop - position, self.frames - offset)
            pieces.append(self.clip(clip)[offset : offset + count].copy())
            position += count
        pixels = np.concatenate(pieces) if pieces else np.empty((0, 64, 64, 3), dtype=np.uint8)
        result = torch.from_numpy(pixels).permute(0, 3, 1, 2).float() / 255
        return result[0] if scalar else result


def remove_generation(path):
    """Remove only the bounded files belonging to one checkpoint generation."""
    if path.parent.name != "resume_states" or not path.name.startswith("step_"):
        raise ValueError("Checkpoint cleanup escaped its intended directory")
    for child in path.iterdir():
        known = child.name in {"model.pt", "trace.npz", "commit.json"}
        temporary = child.name.startswith(".commit.json.") and child.name.endswith(".tmp")
        if not (known or temporary) or not child.is_file():
            raise ValueError("Unexpected file in a checkpoint generation")
    for child in path.iterdir():
        child.unlink()
    path.rmdir()


def load_checkpoint(branch):
    """Return the newest intact commit; a corrupt latest commit cannot mask its predecessor."""
    folder = branch / "resume_states"
    if not folder.exists():
        return None
    invalid = []
    for directory in sorted(folder.glob("step_*"), reverse=True):
        if directory.name.endswith(".pending") or not (directory / "commit.json").exists():
            continue
        try:
            import json

            record = json.loads((directory / "commit.json").read_text())
            assert all(digest(directory / name) == value for name, value in record["file_hashes"].items())
            with np.load(directory / "trace.npz", allow_pickle=False) as trace:
                assert all(np.isfinite(trace[k]).all() for k in trace.files)
                assert not trace["accepted"][record["done"] :].any()
            if invalid:
                atomic_json(
                    branch / "recovery-fallback.json", dict(invalid_generations=invalid, restored=str(directory))
                )
            return directory, record
        except (OSError, ValueError, AssertionError, KeyError) as error:
            invalid.append(dict(path=str(directory), reason=repr(error)))
    if invalid:
        raise ValueError("No valid checkpoint generation remains")
    return None


def commit_checkpoint(branch, state, arrays, record):
    """Publish a complete generation atomically, keeping the preceding generation."""
    folder = branch / "resume_states"
    folder.mkdir(exist_ok=True)
    destination = folder / f"step_{record['done']:08d}"
    pending = destination.with_name(destination.name + ".pending")
    if pending.exists():
        remove_generation(pending)
    pending.mkdir()
    torch.save(state, pending / "model.pt")
    np.savez_compressed(pending / "trace.npz", **arrays)
    record = dict(record, file_hashes={name: digest(pending / name) for name in ("model.pt", "trace.npz")})
    atomic_json(pending / "commit.json", record)
    if destination.exists():
        # A prior incomplete/corrupt attempt at this exact step is kept for inspection.
        destination.rename(branch / (destination.name + ".quarantined"))
    pending.rename(destination)
    committed = [p for p in sorted(folder.glob("step_*")) if not p.name.endswith(".pending")]
    for previous in committed[:-2]:
        remove_generation(previous)
    return destination
