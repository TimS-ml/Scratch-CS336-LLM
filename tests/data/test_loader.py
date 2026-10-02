from __future__ import annotations

import numpy as np
import pytest
import torch

from scratch_cs336.data.loader import PretrainLoader


class IdSource:
    """Window ``i`` is ``[i] * (seq_len + 1)``, so a batch row reveals which window it came from."""

    def __init__(self, n: int):
        self.n = n

    def num_windows(self, seq_len: int) -> int:
        return self.n

    def window(self, i: int, seq_len: int) -> np.ndarray:
        return np.full(seq_len + 1, i, dtype=np.uint16)


class StreamSource:
    """Windows over the stream 0, 1, 2, ... (stride seq_len), like a token cache."""

    def num_windows(self, seq_len: int) -> int:
        return 50

    def window(self, i: int, seq_len: int) -> np.ndarray:
        return np.arange(i * seq_len, i * seq_len + seq_len + 1, dtype=np.uint16)


def _loader(source, dp_rank=0, dp_size=1, global_batch_size=8, seq_len=4, seed=7) -> PretrainLoader:
    return PretrainLoader(source, seq_len, global_batch_size, dp_rank, dp_size, seed)


@pytest.mark.parametrize("dp_size", [1, 2, 4])
def test_ranks_partition_the_dp1_batch(dp_size: int):
    reference = _loader(IdSource(37))
    ranks = [_loader(IdSource(37), r, dp_size) for r in range(dp_size)]
    for step in (0, 1, 5, 11, 100):
        x_ref, y_ref = reference.batch(step)
        parts = [rank.batch(step) for rank in ranks]
        assert all(x.shape == (8 // dp_size, 4) and x.dtype == torch.int64 for x, _ in parts)
        assert torch.equal(torch.cat([x for x, _ in parts]), x_ref)
        assert torch.equal(torch.cat([y for _, y in parts]), y_ref)


def test_targets_are_inputs_shifted_by_one():
    x, y = _loader(StreamSource()).batch(3)
    assert torch.equal(y[:, :-1], x[:, 1:])
    assert torch.equal(y - x, torch.ones_like(x))


def test_batch_is_a_pure_function_of_step():
    loader = _loader(IdSource(37))
    first = loader.batch(9)
    loader.batch(2)
    assert torch.equal(first[0], loader.batch(9)[0])
    assert torch.equal(first[0], _loader(IdSource(37)).batch(9)[0])


def test_resume_from_state_dict_reproduces_the_sequence():
    loader = _loader(IdSource(37), dp_rank=1, dp_size=2)
    it = iter(loader)
    seen = [next(it) for _ in range(5)]
    state = loader.state_dict()
    assert state == {"step": 5}
    expected = [next(it) for _ in range(4)]

    resumed = _loader(IdSource(37), dp_rank=1, dp_size=2)
    resumed.load_state_dict(state)
    got = [b for _, b in zip(range(4), resumed, strict=False)]
    assert all(torch.equal(a[0], b[0]) and torch.equal(a[1], b[1]) for a, b in zip(expected, got, strict=True))
    assert not torch.equal(seen[0][0], got[0][0])


@pytest.mark.parametrize("n_windows,batch", [(12, 4), (10, 4), (7, 8)])
def test_every_window_appears_once_per_epoch(n_windows: int, batch: int):
    loader = _loader(IdSource(n_windows), global_batch_size=batch)
    epochs = 3
    ids = [int(i) for step in range(-(-epochs * n_windows // batch)) for i in loader.batch(step)[0][:, 0]]
    ids = ids[: epochs * n_windows]
    for e in range(epochs):
        assert sorted(ids[e * n_windows : (e + 1) * n_windows]) == list(range(n_windows))
    assert ids[:n_windows] != ids[n_windows : 2 * n_windows]  # a fresh order every epoch


def test_invalid_layouts_are_rejected():
    with pytest.raises(ValueError):
        _loader(IdSource(10), global_batch_size=6, dp_size=4)
    with pytest.raises(ValueError):
        _loader(IdSource(10), dp_rank=2, dp_size=2)
    with pytest.raises(ValueError):
        _loader(IdSource(0))
