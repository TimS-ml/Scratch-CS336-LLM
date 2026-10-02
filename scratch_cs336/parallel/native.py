"""Torch-native counterparts of the scratch strategies (``backend=native``).

* TP: DTensor ``parallelize_module`` with ``ColwiseParallel`` / ``RowwiseParallel``. The stock styles only accept
  ``nn.Linear``/``nn.Embedding``; the subclasses below apply the same linear partitioning to any module holding
  ``weight [out, in]`` (+ ``bias``) used as ``F.linear`` (our models' ``Linear``).
  Parameters are then *stored* as plain local shards: the registered parameter becomes ``DTensor.to_local()`` and
  the module rebuilds the DTensor view (``from_local``) only for the duration of its own forward. Model code that
  derives head counts from ``weight.shape`` therefore sees local shapes (the TP rule), and DDP / ZeRO / FSDP2 see
  ordinary tensors, exactly like the scratch backend.
  HEADWISE: ``distribute_tensor(p, [Shard(0)])``, stored as its local shard; no communication.
  REPLICATE: during forward each parameter reads as
  ``DTensor.from_local(p, [Replicate()]).to_local(grad_placements=[Partial()])`` — identity forward, and its
  gradient is declared partial so the ``from_local`` backward all-reduces it over ``tp``.
* DDP / ZeRO-1: ``DistributedDataParallel`` (+ ``ZeroRedundancyOptimizer``).
* FSDP / HSDP: FSDP2 ``fully_shard`` per unit and on the root over the 2-D ``(dp_replicate, dp_shard)`` mesh,
  ``MixedPrecisionPolicy(param_dtype=compute_dtype, reduce_dtype=float32)``.
"""

from __future__ import annotations

from collections.abc import Mapping
from functools import partial
from typing import Any

import torch
from torch import Tensor, nn
from torch.distributed.device_mesh import DeviceMesh
from torch.distributed.fsdp import MixedPrecisionPolicy, fully_shard
from torch.distributed.optim import ZeroRedundancyOptimizer
from torch.distributed.tensor import DTensor, Partial, Replicate, Shard, distribute_module, distribute_tensor
from torch.distributed.tensor.parallel import ColwiseParallel, RowwiseParallel, parallelize_module

from scratch_cs336.parallel.plan import TPStyle
from scratch_cs336.parallel.tp import check_shardable, swap_params_in_forward


class _LinearColwise(ColwiseParallel):
    def _apply(self, module: nn.Module, device_mesh: DeviceMesh) -> nn.Module:
        return distribute_module(
            module,
            device_mesh,
            self._partition_linear_fn,
            partial(self._prepare_input_fn, self.input_layouts, self.desired_input_layouts),
            partial(self._prepare_output_fn, self.output_layouts, self.use_local_output),
        )


class _LinearRowwise(RowwiseParallel):
    def _apply(self, module: nn.Module, device_mesh: DeviceMesh) -> nn.Module:
        self.desired_input_layouts = (Shard(-1),)
        return distribute_module(
            module,
            device_mesh,
            self._partition_linear_fn,
            partial(self._prepare_input_fn, self.input_layouts, self.desired_input_layouts),
            partial(self._prepare_output_fn, self.output_layouts, self.use_local_output),
        )


def _enter_tp_region(tp_mesh: DeviceMesh, name: str, t: Tensor) -> Tensor:
    replicated = DTensor.from_local(t, tp_mesh, [Replicate()], run_check=False)
    return replicated.to_local(grad_placements=[Partial()])


def _store_local(module: nn.Module) -> dict[str, tuple]:
    """Replace the module's direct DTensor parameters by their local shards; returns their DTensor specs."""
    specs = {}
    for pname, p in list(module.named_parameters(recurse=False)):
        if isinstance(p, DTensor):
            specs[pname] = (p.device_mesh, p.placements, p.shape, p.stride())
            module.register_parameter(pname, nn.Parameter(p.to_local().detach(), requires_grad=p.requires_grad))
    return specs


def apply_native_tp(model: nn.Module, styles: Mapping[str, TPStyle], tp_mesh: DeviceMesh) -> None:
    check_shardable(model, styles, tp_mesh.size())
    linear = {TPStyle.COLWISE: _LinearColwise, TPStyle.ROWWISE: _LinearRowwise}
    parallelize_module(model, tp_mesh, {n: linear[s]() for n, s in styles.items() if s in linear})
    for name, style in styles.items():
        for sub in model.get_submodule(name).modules():
            match style:
                case TPStyle.COLWISE | TPStyle.ROWWISE:
                    specs = _store_local(sub)

                    def as_dtensor(pname: str, t: Tensor, specs: dict[str, tuple] = specs) -> Tensor:
                        mesh, placements, shape, stride = specs[pname]
                        return DTensor.from_local(t, mesh, placements, run_check=False, shape=shape, stride=stride)

                    if specs:
                        swap_params_in_forward(sub, as_dtensor)
                case TPStyle.HEADWISE:
                    for pname, p in list(sub.named_parameters(recurse=False)):
                        local = distribute_tensor(p.detach(), tp_mesh, [Shard(0)]).to_local()
                        sub.register_parameter(pname, nn.Parameter(local, requires_grad=p.requires_grad))
                case TPStyle.REPLICATE:
                    swap_params_in_forward(sub, partial(_enter_tp_region, tp_mesh))


def fully_shard_model(
    model: nn.Module, units: list[nn.Module], dp_mesh: DeviceMesh, compute_dtype: torch.dtype | None
) -> None:
    """FSDP2 every unit, then the root; ``compute_dtype`` (None: compute in the parameters' dtype)."""
    mp = MixedPrecisionPolicy() if compute_dtype is None else MixedPrecisionPolicy(compute_dtype, torch.float32)
    for unit in units:
        fully_shard(unit, mesh=dp_mesh, mp_policy=mp)
    fully_shard(model, mesh=dp_mesh, mp_policy=mp)


class LocalStateZeroRedundancyOptimizer(ZeroRedundancyOptimizer):
    """ZeRO-1 whose ``state_dict`` is this rank's shard (no consolidation onto one rank)."""

    def state_dict(self) -> dict[str, Any]:
        return self.optim.state_dict()

    def load_state_dict(self, state_dict: dict[str, Any]) -> None:
        self.optim.load_state_dict(state_dict)


def local_state_optimizer(optimizer_cls: type[torch.optim.Optimizer]) -> type[torch.optim.Optimizer]:
    """Subclass of ``optimizer_cls`` whose ``state_dict`` holds local shards of DTensor state (``torch.save``-able
    per rank) and whose ``load_state_dict`` re-wraps them with the matching parameter's placements."""

    class LocalState(optimizer_cls):
        def state_dict(self) -> dict[str, Any]:
            sd = super().state_dict()
            sd["state"] = {
                k: {n: v.to_local() if isinstance(v, DTensor) else v for n, v in s.items()}
                for k, s in sd["state"].items()
            }
            return sd

        def load_state_dict(self, state_dict: dict[str, Any]) -> None:
            params = [p for g in self.param_groups for p in g["params"]]

            def like(p: Tensor, v: Any) -> Any:
                if isinstance(p, DTensor) and isinstance(v, Tensor) and v.dim() > 0 and v.shape == p.to_local().shape:
                    return DTensor.from_local(v, p.device_mesh, p.placements, shape=p.shape, stride=p.stride())
                return v

            state = {k: {n: like(params[int(k)], v) for n, v in s.items()} for k, s in state_dict["state"].items()}
            super().load_state_dict({**state_dict, "state": state})

    LocalState.__name__ = LocalState.__qualname__ = f"LocalState{optimizer_cls.__name__}"
    return LocalState
