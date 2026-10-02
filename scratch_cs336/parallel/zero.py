"""Scratch ZeRO-1 optimizer-state sharding (CS336 hw2 contract).

Every rank of ``group`` holds the full parameters and (already averaged) gradients. Trainable parameters are
assigned to owners greedily by size, largest first, so ranks own balanced element counts. Each rank runs the
inner optimizer only over the parameters it owns (so only it allocates their state), then every owner
broadcasts its updated parameters, flattened per (owner, dtype).
"""

from __future__ import annotations

from collections.abc import Callable, Iterable
from typing import Any

import torch
import torch.distributed as dist
from torch import Tensor, nn


class ShardedOptimizer(torch.optim.Optimizer):
    def __init__(
        self,
        params: Iterable[nn.Parameter] | Iterable[dict[str, Any]],
        optimizer_cls: type[torch.optim.Optimizer],
        group: dist.ProcessGroup | None = None,
        **kwargs: Any,
    ) -> None:
        self.group = group if group is not None else dist.group.WORLD
        self._rank = dist.get_rank(self.group)
        self._world = dist.get_world_size(self.group)
        self._loads = [0] * self._world
        self._owned: list[list[Tensor]] = [[] for _ in range(self._world)]
        self._local_groups: list[dict[str, Any]] = []
        self.inner: torch.optim.Optimizer | None = None
        super().__init__(params, kwargs)
        self.inner = optimizer_cls(self._local_groups, **kwargs)

    def add_param_group(self, param_group: dict[str, Any]) -> None:
        super().add_param_group(param_group)
        group = self.param_groups[-1]
        trainable = [p for p in group["params"] if p.requires_grad]
        local = []
        for p in sorted(trainable, key=lambda p: -p.numel()):
            owner = min(range(self._world), key=lambda r: self._loads[r])
            self._loads[owner] += p.numel()
            self._owned[owner].append(p)
            if owner == self._rank:
                local.append(p)
        local_group = {k: v for k, v in group.items() if k != "params"} | {"params": local}
        if self.inner is None:
            self._local_groups.append(local_group)
        else:
            self.inner.add_param_group(local_group)

    @torch.no_grad()
    def step(self, closure: Callable[[], float] | None = None) -> float | None:
        loss = None
        if closure is not None:
            with torch.enable_grad():
                loss = closure()
        for outer, inner in zip(self.param_groups, self.inner.param_groups, strict=True):
            inner.update({k: v for k, v in outer.items() if k != "params"})
        self.inner.step()
        self._broadcast_owned()
        return loss

    def _broadcast_owned(self) -> None:
        inflight = []
        for owner, params in enumerate(self._owned):
            for dtype in dict.fromkeys(p.dtype for p in params):
                same = [p for p in params if p.dtype == dtype]
                flat = torch.cat([p.reshape(-1) for p in same])
                work = dist.broadcast(flat, group=self.group, group_src=owner, async_op=True)
                inflight.append((work, same, flat))
        for work, same, flat in inflight:
            work.wait()
            for p, chunk in zip(same, flat.split([p.numel() for p in same]), strict=True):
                p.copy_(chunk.view_as(p))

    def state_dict(self) -> dict[str, Any]:
        """This rank's shard of the optimizer state (load it back on the same rank and layout)."""
        return self.inner.state_dict()

    def load_state_dict(self, state_dict: dict[str, Any]) -> None:
        self.inner.load_state_dict(state_dict)
