"""Strategy-agnostic parallelism API.

``parallelize(model, mesh, cfg)`` turns a full, identically-initialized model into a :class:`ParallelModel`:

1. broadcast rank 0's parameters and buffers to every rank (all strategies start from identical weights);
2. activation checkpointing of each ``model.blocks[i]`` if requested;
3. tensor parallelism over ``tp`` from ``model.tp_plan()`` (only when ``tp > 1``);
4. the data-parallel strategy over the dp dims on the local (TP-sharded) module.

Data-parallel layout: DDP all-reduces over the flattened dp group; ZeRO-1 does the same and shards optimizer state
over ``dp_shard`` (replicas update redundantly); FSDP shards over ``dp_shard`` and, when ``dp_replicate > 1``,
all-reduces across replicas (HSDP). Gradients are always averaged over dp.

Mixed precision is owned by the parallel model: FSDP all-gathers parameters cast to ``compute_dtype``; DDP/ZeRO
run the forward under ``torch.autocast``. Master parameters, gradients and optimizer state keep the parameters'
dtype (fp32); ``compute_dtype="float32"`` means no mixed precision.
"""

from __future__ import annotations

import functools
from abc import ABC, abstractmethod
from collections.abc import Callable, Iterable, Iterator
from contextlib import AbstractContextManager, contextmanager, nullcontext
from dataclasses import dataclass
from enum import StrEnum
from typing import Any, Protocol

import torch
import torch.distributed as dist
from torch import Tensor, nn
from torch.distributed.tensor import DTensor
from torch.utils.checkpoint import checkpoint

from scratch_cs336.distributed.mesh import Mesh, MeshConfig
from scratch_cs336.parallel.ddp import DDP
from scratch_cs336.parallel.fsdp import FSDP, is_block
from scratch_cs336.parallel.native import (
    LocalStateZeroRedundancyOptimizer,
    apply_native_tp,
    fully_shard_model,
    local_state_optimizer,
)
from scratch_cs336.parallel.tp import apply_tp, gather_tp, resolve_plan, tp_shard_dims
from scratch_cs336.parallel.zero import ShardedOptimizer

DDP_BUCKET_MB = 25.0
COMPUTE_DTYPES = {"float32": None, "bfloat16": torch.bfloat16}  # float32: no mixed precision


class Strategy(StrEnum):
    DDP = "ddp"
    ZERO1 = "zero1"
    FSDP = "fsdp"


class Backend(StrEnum):
    SCRATCH = "scratch"
    NATIVE = "native"


@dataclass(frozen=True)
class ParallelConfig:
    mesh: MeshConfig
    strategy: Strategy
    backend: Backend
    compute_dtype: str = "float32"
    activation_checkpointing: bool = False

    def __post_init__(self) -> None:
        if self.compute_dtype not in COMPUTE_DTYPES:
            raise ValueError(f"compute_dtype must be one of {sorted(COMPUTE_DTYPES)}, got {self.compute_dtype!r}")

    @property
    def mixed_precision_dtype(self) -> torch.dtype | None:
        """Forward/backward dtype, or None to compute in the parameters' dtype."""
        return COMPUTE_DTYPES[self.compute_dtype]


class ParallelModel(Protocol):
    module: nn.Module

    def __call__(self, *args: Any, **kwargs: Any) -> Any: ...
    def parameters(self) -> list[nn.Parameter]: ...
    def no_sync(self) -> AbstractContextManager[None]: ...
    def finish_grad_sync(self) -> None: ...
    def clip_grad_norm_(self, max_norm: float) -> float: ...
    def sharded_state_dict(self) -> dict[str, Tensor]: ...
    def load_sharded_state_dict(self, state_dict: dict[str, Tensor]) -> None: ...
    def full_state_dict(self) -> dict[str, Tensor]: ...


# (local gradient, sharded over tp, sharded over dp_shard)
GradEntry = tuple[Tensor, bool, bool]


class _Parallelized(ABC):
    """Shared behavior of the concrete parallel models; subclasses provide the strategy specifics."""

    module: nn.Module

    def __init__(self, module: nn.Module, mesh: Mesh, tp_dims: dict[str, int], state_keys: list[str]) -> None:
        self.module = module
        self.mesh = mesh
        self.tp_dims = tp_dims
        self._state_keys = state_keys

    def parameters(self) -> list[nn.Parameter]:
        """Trainable parameters the optimizer must own (local shards)."""
        return [p for p in self.module.parameters() if p.requires_grad]

    @abstractmethod
    def __call__(self, *args: Any, **kwargs: Any) -> Any: ...

    @abstractmethod
    def no_sync(self) -> AbstractContextManager[None]: ...

    @abstractmethod
    def finish_grad_sync(self) -> None: ...

    @abstractmethod
    def sharded_state_dict(self) -> dict[str, Tensor]: ...

    @abstractmethod
    def load_sharded_state_dict(self, state_dict: dict[str, Tensor]) -> None: ...

    @abstractmethod
    def full_state_dict(self) -> dict[str, Tensor]:
        """Unsharded state dict on every rank, on CPU (collective)."""

    @abstractmethod
    def _grad_entries(self) -> list[GradEntry]: ...

    @torch.no_grad()
    def clip_grad_norm_(self, max_norm: float) -> float:
        """Clip gradients by the global L2 norm over every shard (dp and tp); returns the pre-clip norm.

        Each element is counted once: shards are summed over the groups they are split across, replicas are not.
        """
        entries = self._grad_entries()
        device = self.mesh.env.device
        sq = torch.zeros(4, dtype=torch.float64, device=device)  # replicated, tp-only, dp-only, tp+dp
        for g, tp_sharded, dp_sharded in entries:
            sq[int(tp_sharded) + 2 * int(dp_sharded)] += torch.linalg.vector_norm(g).double() ** 2
        if self.mesh.shard_size > 1:
            dist.all_reduce(sq[2:], group=self.mesh.shard_group)
        if self.mesh.tp_size > 1:
            tp_part = sq[1::2].clone()
            dist.all_reduce(tp_part, group=self.mesh.tp_group)
            sq[1::2] = tp_part
        total = sq.sum().sqrt()
        coef = torch.clamp(max_norm / (total + 1e-6), max=1.0)
        for g, _, _ in entries:
            g.mul_(coef.to(g.dtype))
        return total.item()

    def _unshard_tp(self, tensors: dict[str, Tensor]) -> dict[str, Tensor]:
        """CPU copies of ``tensors`` (already unsharded over dp) with tp shards gathered, in state-dict order."""
        out = {}
        for key in self._state_keys:
            t = tensors[key]
            if key in self.tp_dims:
                t = gather_tp(t, self.tp_dims[key], self.mesh.tp_group)
            out[key] = t.detach().to("cpu", copy=True)
        return out


class _Replicated(_Parallelized):
    """DDP and ZeRO-1 (scratch or torch DDP): full local parameters on every dp rank."""

    def __init__(
        self,
        wrapper: nn.Module,
        module: nn.Module,
        mesh: Mesh,
        tp_dims: dict[str, int],
        state_keys: list[str],
        compute_dtype: torch.dtype | None,
        finish: Callable[[], None],
    ) -> None:
        super().__init__(module, mesh, tp_dims, state_keys)
        self._wrapper = wrapper
        self._compute_dtype = compute_dtype
        self._finish = finish

    def __call__(self, *args: Any, **kwargs: Any) -> Any:
        ctx = (
            nullcontext()
            if self._compute_dtype is None
            else torch.autocast(self.mesh.env.device.type, dtype=self._compute_dtype)
        )
        with ctx:
            return self._wrapper(*args, **kwargs)

    def no_sync(self) -> AbstractContextManager[None]:
        return self._wrapper.no_sync()

    def finish_grad_sync(self) -> None:
        self._finish()

    def _grad_entries(self) -> list[GradEntry]:
        return [(p.grad, n in self.tp_dims, False) for n, p in self.module.named_parameters() if p.grad is not None]

    def sharded_state_dict(self) -> dict[str, Tensor]:
        return self.module.state_dict()

    def load_sharded_state_dict(self, state_dict: dict[str, Tensor]) -> None:
        self.module.load_state_dict(state_dict)

    @torch.no_grad()
    def full_state_dict(self) -> dict[str, Tensor]:
        return self._unshard_tp(self.module.state_dict())


class _ScratchFSDP(_Parallelized):
    def __init__(self, fsdp: FSDP, mesh: Mesh, tp_dims: dict[str, int], state_keys: list[str]) -> None:
        super().__init__(fsdp.module, mesh, tp_dims, state_keys)
        self.fsdp = fsdp

    def __call__(self, *args: Any, **kwargs: Any) -> Any:
        return self.fsdp(*args, **kwargs)

    def parameters(self) -> list[nn.Parameter]:
        return [f.shard for f in self.fsdp.flat_params if f.requires_grad]

    def no_sync(self) -> AbstractContextManager[None]:
        return self.fsdp.no_sync()

    def finish_grad_sync(self) -> None:
        self.fsdp.finish_grad_sync()

    def _grad_entries(self) -> list[GradEntry]:
        return [(f.shard.grad, f.tp_sharded, True) for f in self.fsdp.flat_params if f.shard.grad is not None]

    def sharded_state_dict(self) -> dict[str, Tensor]:
        return self.fsdp.sharded_state_dict()

    def load_sharded_state_dict(self, state_dict: dict[str, Tensor]) -> None:
        self.fsdp.load_sharded_state_dict(state_dict)

    @torch.no_grad()
    def full_state_dict(self) -> dict[str, Tensor]:
        return self._unshard_tp(self.fsdp.gather_full_params() | self.module.state_dict())


class _NativeFSDP(_Parallelized):
    """FSDP2 (``fully_shard``) over the local TP shards; parameters are 1-D DTensors over the dp mesh."""

    def __call__(self, *args: Any, **kwargs: Any) -> Any:
        return self.module(*args, **kwargs)

    @contextmanager
    def no_sync(self) -> Iterator[None]:
        self.module.set_requires_gradient_sync(False)
        try:
            yield
        finally:
            self.module.set_requires_gradient_sync(True)

    def finish_grad_sync(self) -> None:
        pass  # FSDP2 finishes reduce-scatter / all-reduce in its post-backward callback.

    def _grad_entries(self) -> list[GradEntry]:
        return [
            (p.grad.to_local(), n in self.tp_dims, True)
            for n, p in self.module.named_parameters()
            if p.grad is not None
        ]

    def sharded_state_dict(self) -> dict[str, Tensor]:
        return {k: v.to_local() if isinstance(v, DTensor) else v for k, v in self.module.state_dict().items()}

    @torch.no_grad()
    def load_sharded_state_dict(self, state_dict: dict[str, Tensor]) -> None:
        for k, v in self.module.state_dict().items():
            (v.to_local() if isinstance(v, DTensor) else v).copy_(state_dict[k])

    @torch.no_grad()
    def full_state_dict(self) -> dict[str, Tensor]:
        sd = self.module.state_dict()
        return self._unshard_tp({k: v.full_tensor() if isinstance(v, DTensor) else v for k, v in sd.items()})


def _checkpoint_blocks(model: nn.Module) -> None:
    blocks = getattr(model, "blocks", None)
    if not isinstance(blocks, nn.ModuleList):
        raise ValueError("activation checkpointing needs `model.blocks: nn.ModuleList`")
    for block in blocks:
        block.forward = functools.partial(checkpoint, block.forward, use_reentrant=False)


def parallelize(model: nn.Module, mesh: Mesh, cfg: ParallelConfig) -> ParallelModel:
    """Shard ``model`` in place according to ``cfg`` over ``mesh`` (collective on every rank)."""
    shape = (mesh.replicate_size, mesh.shard_size, mesh.tp_size)
    if cfg.mesh.resolve(mesh.env.world_size) != shape:
        raise ValueError(f"cfg.mesh {cfg.mesh} does not describe the built mesh {shape}")
    dtype = cfg.mixed_precision_dtype
    with torch.no_grad():
        for t in [*model.parameters(), *model.buffers()]:
            dist.broadcast(t, src=0)
    state_keys = list(model.state_dict().keys())
    styles = {}
    if mesh.tp_size > 1:
        if not hasattr(model, "tp_plan"):
            raise ValueError(f"{type(model).__name__} has no tp_plan(); cannot use tensor parallelism")
        styles = resolve_plan(model, model.tp_plan())
    tp_dims = tp_shard_dims(model, styles)
    if cfg.activation_checkpointing:
        _checkpoint_blocks(model)
    replicated = cfg.strategy in (Strategy.DDP, Strategy.ZERO1)

    if cfg.backend == Backend.SCRATCH:
        if styles:
            apply_tp(model, styles, mesh.tp_group)
        if replicated:
            ddp = DDP(model, mesh.dp_group, bucket_size_mb=DDP_BUCKET_MB)
            return _Replicated(ddp, model, mesh, tp_dims, state_keys, dtype, ddp.finish_grad_sync)
        fsdp = FSDP(
            model,
            shard_group=mesh.shard_group,
            replicate_group=mesh.replicate_group if mesh.replicate_size > 1 else None,
            compute_dtype=dtype,
            tp_sharded=set(tp_dims),
        )
        return _ScratchFSDP(fsdp, mesh, tp_dims, state_keys)

    if styles:
        apply_native_tp(model, styles, mesh.device_mesh["tp"])
    if replicated:
        ddp = nn.parallel.DistributedDataParallel(model, process_group=mesh.dp_group, bucket_cap_mb=DDP_BUCKET_MB)
        return _Replicated(ddp, model, mesh, tp_dims, state_keys, dtype, lambda: None)
    units = [m for n, m in model.named_modules() if n and is_block(n, m)]
    fully_shard_model(model, units, mesh.device_mesh["dp_replicate", "dp_shard"], dtype)
    return _NativeFSDP(model, mesh, tp_dims, state_keys)


def build_optimizer(
    pmodel: ParallelModel,
    cfg: ParallelConfig,
    optimizer_cls: type[torch.optim.Optimizer],
    params: Iterable[nn.Parameter] | Iterable[dict[str, Any]] | None = None,
    **kwargs: Any,
) -> torch.optim.Optimizer:
    """Optimizer over ``pmodel``'s local trainable parameters; ``state_dict()`` is per rank and ``torch.save``-able.

    ``params`` (e.g. weight-decay groups) must cover exactly ``pmodel.parameters()``; default: all of them.
    """
    if not isinstance(pmodel, _Parallelized):
        raise TypeError("build_optimizer expects a model returned by parallelize()")
    own = pmodel.parameters()
    if params is None:
        params = own
    else:
        params = list(params)
        given = [p for g in params for p in (g["params"] if isinstance(g, dict) else [g])]
        if len(given) != len(own) or {id(p) for p in given} != {id(p) for p in own}:
            raise ValueError("params must hold every parameter of pmodel.parameters() exactly once")
    if cfg.strategy == Strategy.ZERO1:
        if cfg.backend == Backend.SCRATCH:
            return ShardedOptimizer(params, optimizer_cls, group=pmodel.mesh.shard_group, **kwargs)
        return LocalStateZeroRedundancyOptimizer(
            params, optimizer_class=optimizer_cls, process_group=pmodel.mesh.shard_group, **kwargs
        )
    if cfg.strategy == Strategy.FSDP and cfg.backend == Backend.NATIVE:
        return local_state_optimizer(optimizer_cls)(params, **kwargs)
    return optimizer_cls(params, **kwargs)
