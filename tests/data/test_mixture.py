from __future__ import annotations

from collections import Counter

import numpy as np
import pytest

from scratch_cs336.data.mixture import MixtureSource


class TaggedSource:
    """Window ``i`` is ``[tag, i, i, ...]`` so we can see where a mixture window came from."""

    def __init__(self, tag: int, n: int):
        self.tag, self.n = tag, n

    def num_windows(self, seq_len: int) -> int:
        return self.n

    def window(self, i: int, seq_len: int) -> np.ndarray:
        out = np.full(seq_len + 1, i, dtype=np.int64)
        out[0] = self.tag
        return out


def _mixture(seed: int = 0, block_size: int = 20) -> MixtureSource:
    sources = {"a": TaggedSource(0, 1000), "b": TaggedSource(1, 1000), "c": TaggedSource(2, 1000)}
    return MixtureSource(sources, {"a": 0.5, "b": 0.3, "c": 0.2}, seed=seed, block_size=block_size)


def test_every_block_has_exact_proportions():
    mixture = _mixture()
    for block in range(10):
        names = Counter(mixture.locate(block * 20 + p, 4)[0] for p in range(20))
        assert names == {"a": 10, "b": 6, "c": 4}


def test_windows_come_from_the_named_source_without_repeats():
    mixture = _mixture()
    n = mixture.num_windows(4)
    seen = set()
    for i in range(n):
        name, index = mixture.locate(i, 4)
        window = mixture.window(i, 4)
        assert window[0] == "abc".index(name) and window[1] == index
        seen.add((name, index))
    assert len(seen) == n  # a bijection onto the first n/block * count windows of each source


def test_length_is_set_by_the_scarcest_source():
    sources = {"a": TaggedSource(0, 1000), "b": TaggedSource(1, 30)}
    mixture = MixtureSource(sources, {"a": 0.5, "b": 0.5}, seed=0, block_size=10)
    assert mixture.num_windows(4) == 60  # b supplies 5 per block: 6 blocks of 10


def test_order_is_deterministic_in_seed_and_shuffled_within_blocks():
    a, b = _mixture(seed=0), _mixture(seed=1)
    order = lambda m: [m.locate(i, 4)[0] for i in range(200)]  # noqa: E731
    assert order(a) == order(_mixture(seed=0))
    assert order(a) != order(b)
    assert order(a)[:10] != ["a"] * 10  # not grouped by source


def test_zero_weight_source_is_never_used_and_tiny_blocks_are_rejected():
    sources = {"a": TaggedSource(0, 100), "b": TaggedSource(1, 100)}
    mixture = MixtureSource(sources, {"a": 1.0, "b": 0.0}, seed=0, block_size=8)
    assert {mixture.locate(i, 4)[0] for i in range(mixture.num_windows(4))} == {"a"}
    with pytest.raises(ValueError):
        MixtureSource(sources, {"a": 0.99, "b": 0.01}, seed=0, block_size=8)
