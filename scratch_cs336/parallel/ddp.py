"""Scratch DistributedDataParallel (CS336 hw2 contract).

* Wrapping broadcasts parameters and buffers from rank 0 of ``group``.
* Communication overlaps backward: a ``post_accumulate_grad`` hook per parameter marks it ready and launches
  an async all-reduce as soon as its bucket (one parameter, or ~``bucket_size_mb`` of them) is complete.
  Buckets are launched strictly in index order so every rank issues collectives in the same sequence even if
  hook order differed.
* ``finish_grad_sync()`` launches anything not launched yet (parameters that received no gradient count as
  zero), waits, and averages over the group.
* Frozen parameters are broadcast but never reduced; tied parameters appear once in ``parameters()`` and their
  hook fires once per backward after all uses have accumulated.
"""

from __future__ import annotations

from collections.abc import Iterator
from contextlib import contextmanager

import torch
import torch.distributed as dist
from torch import Tensor, nn


def _buckets(params: list[nn.Parameter], bucket_size_mb: float | None) -> list[list[nn.Parameter]]:
    if bucket_size_mb is None:
        return [[p] for p in params]
    cap = bucket_size_mb * 2**20
    buckets: list[list[nn.Parameter]] = [[]]
    size = 0
    for p in params:
        nbytes = p.numel() * p.element_size()
        if buckets[-1] and (size + nbytes > cap or p.dtype != buckets[-1][0].dtype):
            buckets.append([])
            size = 0
        buckets[-1].append(p)
        size += nbytes
    return [b for b in buckets if b]


class DDP(nn.Module):
    def __init__(
        self, module: nn.Module, group: dist.ProcessGroup | None = None, bucket_size_mb: float | None = None
    ) -> None:
        super().__init__()
        self.module = module
        self.group = group if group is not None else dist.group.WORLD
        self._world = dist.get_world_size(self.group)
        with torch.no_grad():
            for t in [*module.parameters(), *module.buffers()]:
                dist.broadcast(t, group=self.group, group_src=0)
        # Reverse registration order approximates the order gradients become ready in backward.
        params = [p for p in module.parameters() if p.requires_grad][::-1]
        self._buckets = _buckets(params, bucket_size_mb)
        self._bucket_of = {p: i for i, b in enumerate(self._buckets) for p in b}
        self._missing = [len(b) for b in self._buckets]
        self._next = 0
        self._inflight: list[tuple[dist.Work, int, Tensor]] = []
        self._sync = True
        for p in params:
            p.register_post_accumulate_grad_hook(self._on_grad_ready)

    def forward(self, *args, **kwargs):
        return self.module(*args, **kwargs)

    @contextmanager
    def no_sync(self) -> Iterator[None]:
        """Accumulate gradients locally; the first backward outside the context reduces the sum."""
        self._sync = False
        try:
            yield
        finally:
            self._sync = True

    def _on_grad_ready(self, p: nn.Parameter) -> None:
        if not self._sync:
            return
        self._missing[self._bucket_of[p]] -= 1
        while self._next < len(self._buckets) and self._missing[self._next] == 0:
            self._launch(self._next)

    def _launch(self, i: int) -> None:
        bucket = self._buckets[i]
        for p in bucket:
            if p.grad is None:
                p.grad = torch.zeros_like(p)
        flat = bucket[0].grad if len(bucket) == 1 else torch.cat([p.grad.reshape(-1) for p in bucket])
        self._inflight.append((dist.all_reduce(flat, group=self.group, async_op=True), i, flat))
        self._next = i + 1

    def finish_grad_sync(self) -> None:
        while self._next < len(self._buckets):
            self._launch(self._next)
        for work, i, flat in self._inflight:
            work.wait()
            flat.div_(self._world)
            bucket = self._buckets[i]
            if len(bucket) > 1:
                for p, g in zip(bucket, flat.split([p.numel() for p in bucket]), strict=True):
                    p.grad.copy_(g.view_as(p))
        self._inflight.clear()
        self._missing = [len(b) for b in self._buckets]
        self._next = 0
