"""Weighted mixture of window sources with exact per-block proportions and O(1) random access."""

from __future__ import annotations

from bisect import bisect_right

import numpy as np

from scratch_cs336.data.loader import WindowSource
from scratch_cs336.data.permutation import FeistelPermutation, derive_seed


def block_counts(weights: dict[str, float], block_size: int) -> dict[str, int]:
    """Integer slots per source summing to ``block_size`` (largest-remainder rounding of ``weights``)."""
    if any(w < 0 for w in weights.values()) or sum(weights.values()) <= 0:
        raise ValueError(f"weights must be non-negative with a positive sum, got {weights}")
    total = sum(weights.values())
    ideal = {name: w / total * block_size for name, w in weights.items()}
    counts = {name: int(x) for name, x in ideal.items()}
    leftover = block_size - sum(counts.values())
    by_remainder = sorted(ideal, key=lambda name: (ideal[name] - counts[name], weights[name]), reverse=True)
    for name in by_remainder[:leftover]:
        counts[name] += 1
    for name, w in weights.items():
        if w > 0 and counts[name] == 0:
            raise ValueError(f"block_size={block_size} is too small to give source {name!r} (weight {w}) a slot")
    return counts


class MixtureSource:
    """Every block of ``block_size`` consecutive mixture windows holds exactly ``counts[name]`` windows of each
    source, in an order shuffled per block by ``seed``. The mixture ends when the scarcest source runs out, so
    the proportions hold for every block, not just on average.
    """

    def __init__(
        self,
        sources: dict[str, WindowSource],
        weights: dict[str, float],
        seed: int,
        block_size: int = 1024,
    ):
        if sources.keys() != weights.keys():
            raise ValueError(f"sources {sorted(sources)} and weights {sorted(weights)} must have the same keys")
        self.block_size = block_size
        self.seed = seed
        self._counts = {name: c for name, c in block_counts(weights, block_size).items() if c > 0}
        self._sources = {name: sources[name] for name in self._counts}
        self._names = list(self._counts)
        self._offsets = np.concatenate([[0], np.cumsum(list(self._counts.values()))]).tolist()

    @property
    def counts(self) -> dict[str, int]:
        return dict(self._counts)

    def num_windows(self, seq_len: int) -> int:
        blocks = min(self._sources[n].num_windows(seq_len) // c for n, c in self._counts.items())
        return blocks * self.block_size

    def locate(self, i: int, seq_len: int) -> tuple[str, int]:
        """Mixture position ``i`` -> ``(source name, window index within that source)``."""
        if not 0 <= i < self.num_windows(seq_len):
            raise IndexError(f"window {i} outside [0, {self.num_windows(seq_len)})")
        block, position = divmod(i, self.block_size)
        slot = FeistelPermutation(self.block_size, derive_seed(self.seed, block))(position)
        k = bisect_right(self._offsets, slot) - 1
        name = self._names[k]
        return name, block * self._counts[name] + slot - self._offsets[k]

    def window(self, i: int, seq_len: int) -> np.ndarray:
        name, index = self.locate(i, seq_len)
        return self._sources[name].window(index, seq_len)
