"""Long intervention boundaries and recurrent admission preserve paired exposure."""

import torch

from cafl4ds.data.streams import StreamBatch
from cafl4ds.filters.base import FilterContext
from cafl4ds.filters.replay_ratio import ReplayRatio
from cafl4ds.long_replay import ReplayIntervention, continuation


def test_continuation_preserves_tail_and_repeats_only_second_domain() -> None:
    """Extension never leaks the first domain into a post-transition continuation."""
    order = list(range(512))
    expanded = continuation(order, 10)
    assert len(expanded) == 1024
    assert expanded[:192] == order[320:]
    assert expanded[192:384] == order[320:]
    assert min(expanded) >= 320
    assert all(len(set(expanded[i : i + 32])) == 32 for i in range(0, 1024, 32))


def test_intervention_preserves_memory_and_stops_at_eight_with_recurrence() -> None:
    """Repeated image content still has correct distinct occurrence ages and RNG."""
    original = ReplayRatio(16, seed=5)
    original.reservoir._buffer = [torch.full((1, 2, 2), float(i)) for i in range(256)]
    original.reservoir._seen = 320
    saved = {"buffer": original.reservoir._buffer, "seen": 320, "rng": original.reservoir._gen.get_state()}
    calls = []

    def observe(_method: object, images: torch.Tensor) -> torch.Tensor:
        calls.append(len(images))
        return torch.cat((images.flatten(1), torch.ones(len(images), 1)), dim=1)

    changed = ReplayIntervention("recent", 10, 5, observe)
    changed.restore_memory(saved)
    changed.restore_arrivals(list(range(256)))
    for offset in range(32):
        incoming = torch.arange(128).float().reshape(32, 1, 2, 2)
        batch = StreamBatch(incoming, era=0, step=10 + offset)
        context = FilterContext(None, 10 + offset)  # type: ignore[arg-type]
        base = original.select(batch, context)
        other = changed.select(batch, context)
        assert torch.equal(torch.stack(original.reservoir._buffer), torch.stack(changed.reservoir._buffer))
        assert torch.equal(original.reservoir._gen.get_state(), changed.reservoir._gen.get_state())
        assert torch.equal(base[:16], other[:16])
        assert len(set(changed.arrivals)) == 256
        if offset >= 8:
            assert torch.equal(base, other)
            assert not changed.last["intervention_active"]
        else:
            obs = changed.last["observation"]
            assert {obs["candidate_arrivals"][i] for i in obs["selected_positions"]} == set(
                sorted(obs["candidate_arrivals"])[-16:]
            )
    assert calls == [64] * 8
    assert max(changed.arrivals) >= 320
