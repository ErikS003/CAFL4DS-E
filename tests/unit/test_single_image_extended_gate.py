"""Long-run slices and crash recovery must preserve the scientific trajectory."""

import numpy as np
import torch

from cafl4ds.single_image_extended_gate import FrameStream, commit_checkpoint, load_checkpoint


def test_lazy_frame_slices_match_dense_across_clip_boundaries(tmp_path):
    rng = np.random.default_rng(66)
    paths, pixels = [], []
    for i in range(4):
        value = rng.integers(0, 256, (96, 64, 64, 3), dtype=np.uint8)
        path = tmp_path / f"{i}.npy"
        np.save(path, value)
        pixels.append(value)
        paths.append(path)
    dense = torch.from_numpy(np.concatenate(pixels)).permute(0, 3, 1, 2).float() / 255
    stream = FrameStream(paths)
    for start, stop in [(0, 1), (90, 103), (191, 197), (288, 384), (384, 384), (0, 384)]:
        torch.testing.assert_close(stream[start:stop], dense[start:stop], rtol=0, atol=0)
        assert len(stream.cache) <= 2
    torch.testing.assert_close(stream[-1], dense[-1], rtol=0, atol=0)


def test_corrupt_latest_generation_falls_back_without_mixing_traces(tmp_path):
    branch = tmp_path / "gate"
    branch.mkdir()
    for done in (96, 192):
        accepted = np.arange(384) < done
        commit_checkpoint(
            branch, {"value": torch.tensor([done])}, {"accepted": accepted, "values": np.arange(384)}, {"done": done}
        )
    latest, record = load_checkpoint(branch)
    assert record["done"] == 192
    (latest / "trace.npz").write_bytes(b"interrupted disk write")
    previous, record = load_checkpoint(branch)
    assert record["done"] == 96
    assert torch.load(previous / "model.pt", weights_only=True)["value"].item() == 96
    assert (branch / "recovery-fallback.json").exists()


def test_only_two_complete_checkpoint_generations_are_retained(tmp_path):
    branch = tmp_path / "gate"
    branch.mkdir()
    for done in (96, 192, 288):
        commit_checkpoint(branch, {"value": torch.tensor(done)}, {"accepted": np.arange(384) < done}, {"done": done})
    assert len(list((branch / "resume_states").iterdir())) == 2
    assert load_checkpoint(branch)[1]["done"] == 288
