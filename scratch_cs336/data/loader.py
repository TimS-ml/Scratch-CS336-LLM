"""Deterministic distributed pretraining loader: every rank derives its rows as a pure function of the step."""

from __future__ import annotations

from collections.abc import Iterator
from typing import Protocol

import numpy as np
import torch

from scratch_cs336.data.permutation import FeistelPermutation, derive_seed


class WindowSource(Protocol):
    """Fixed-length token windows: ``window(i, seq_len)`` has ``seq_len + 1`` tokens (inputs + shifted targets)."""

    def num_windows(self, seq_len: int) -> int: ...

    def window(self, i: int, seq_len: int) -> np.ndarray: ...


class PretrainLoader:
    """Global sample ``g = step * G + j`` is window ``perm_{seed, g // N}(g % N)``; rank ``r`` owns ``j`` in
    ``[r * G / dp, (r + 1) * G / dp)``. Each epoch visits every window exactly once; resume = restore ``step``.
    """

    def __init__(
        self,
        source: WindowSource,
        seq_len: int,
        global_batch_size: int,
        dp_rank: int,
        dp_size: int,
        seed: int,
    ):
        if global_batch_size % dp_size:
            raise ValueError(f"global_batch_size {global_batch_size} is not divisible by dp_size {dp_size}")
        if not 0 <= dp_rank < dp_size:
            raise ValueError(f"dp_rank {dp_rank} outside [0, {dp_size})")
        self.num_windows = source.num_windows(seq_len)
        if self.num_windows < 1:
            raise ValueError(f"source has no windows of seq_len={seq_len}")
        self.source = source
        self.seq_len = seq_len
        self.global_batch_size = global_batch_size
        self.local_batch_size = global_batch_size // dp_size
        self.dp_rank = dp_rank
        self.dp_size = dp_size
        self.seed = seed
        self.step = 0
        self._perms: dict[int, FeistelPermutation] = {}

    def _permutation(self, epoch: int) -> FeistelPermutation:
        if epoch not in self._perms:
            if len(self._perms) > 2:
                self._perms.clear()
            self._perms[epoch] = FeistelPermutation(self.num_windows, derive_seed(self.seed, epoch))
        return self._perms[epoch]

    def window_index(self, g: int) -> int:
        """Source window used by global sample ``g``."""
        epoch, position = divmod(g, self.num_windows)
        return self._permutation(epoch)(position)

    def batch(self, step: int) -> tuple[torch.Tensor, torch.Tensor]:
        """This rank's ``(x, y)`` LongTensors of shape ``[G / dp, seq_len]`` for ``step``."""
        first = step * self.global_batch_size + self.dp_rank * self.local_batch_size
        rows = np.stack(
            [
                self.source.window(self.window_index(g), self.seq_len)
                for g in range(first, first + self.local_batch_size)
            ]
        ).astype(np.int64)
        tokens = torch.from_numpy(rows)
        return tokens[:, :-1].contiguous(), tokens[:, 1:].contiguous()

    def __iter__(self) -> Iterator[tuple[torch.Tensor, torch.Tensor]]:
        while True:
            batch = self.batch(self.step)
            self.step += 1
            yield batch

    def state_dict(self) -> dict[str, int]:
        return {"step": self.step}

    def load_state_dict(self, state: dict[str, int]) -> None:
        self.step = int(state["step"])
