"""Scratch FSDP / HSDP (ZeRO-3, CS336 hw2 contract).

Units: every module selected by ``is_unit`` (default: the children of ``model.blocks``) plus the root, which owns
all remaining parameters. Within a unit, parameters are grouped by (requires_grad, tp-sharded, dtype, ndim >= 2) and
each group is flattened into one *master* buffer in the parameters' dtype (fp32 in training), zero-padded to a
multiple of the shard-group size; each rank keeps only its contiguous slice (an ``nn.Parameter`` the optimizer
updates).

Per micro-step and unit:

* pre-forward: all-gather the shards (cast to ``compute_dtype``) into a reusable buffer, alias it as a fresh
  autograd leaf, and install views of that leaf as the modules' parameter attributes.
* post-forward: free the gathered storage (the root keeps it: its backward starts right away) and hook the
  outputs so the first gradient reaching them re-gathers before the unit's backward runs.
* the leaf's ``post_accumulate_grad`` hook fires once all of the unit's gradients exist: cast to the master dtype,
  free the storage, and launch an async reduce-scatter (or, under ``no_sync``, accumulate the unsharded gradient).
* ``finish_grad_sync``: wait, average over the shard group, all-reduce and average over the replicate group
  (HSDP), and accumulate into the shard's ``.grad``.

Freeing/regathering resizes the storage in place while autograd holds views of the leaf; the gathered buffer
and the leaf share a storage but not a version counter, so refilling the buffer does not invalidate tensors
saved for backward.
"""

from __future__ import annotations

from collections.abc import Callable, Collection, Iterator
from contextlib import contextmanager
from dataclasses import dataclass

import torch
import torch.distributed as dist
from torch import Tensor, nn

UnitFilter = Callable[[str, nn.Module], bool]


def is_block(name: str, module: nn.Module) -> bool:
    """Default unit filter: each ``blocks[i]`` of the root module."""
    return name.startswith("blocks.") and name.count(".") == 1


@dataclass
class _Slot:
    fqns: list[str]
    bindings: list[tuple[nn.Module, str]]
    shape: torch.Size
    offset: int

    @property
    def numel(self) -> int:
        return self.shape.numel()


class FlatParam:
    """A flattened, padded, sharded group of parameters belonging to one unit."""

    def __init__(
        self,
        slots: list[_Slot],
        tensors: list[Tensor],
        requires_grad: bool,
        tp_sharded: bool,
        group: dist.ProcessGroup,
        compute_dtype: torch.dtype | None,
    ) -> None:
        self.slots = slots
        self.requires_grad = requires_grad
        self.tp_sharded = tp_sharded
        self.group = group
        self.compute_dtype = compute_dtype or tensors[0].dtype
        self.world = dist.get_world_size(group)
        total = sum(t.numel() for t in tensors)
        self.padded = -(-total // self.world) * self.world
        n = self.padded // self.world
        rank = dist.get_rank(group)
        device = tensors[0].device
        flat = torch.zeros(self.padded, dtype=tensors[0].dtype, device=device)
        flat[:total].copy_(torch.cat([t.detach().reshape(-1) for t in tensors]))
        self.shard = nn.Parameter(flat[rank * n : (rank + 1) * n].clone(), requires_grad=requires_grad)
        self._buf = torch.empty(self.padded, dtype=self.compute_dtype, device=device)
        self._buf.untyped_storage().resize_(0)
        self.gathered = False
        self.sync = True
        self._unsynced: Tensor | None = None
        self._inflight: list[tuple[dist.Work, Tensor, Tensor]] = []

    def gather(self) -> None:
        if self.gathered:
            return
        self._buf.untyped_storage().resize_(self.padded * self._buf.element_size())
        dist.all_gather_single(self._buf, self.shard.detach().to(self.compute_dtype), group=self.group)
        self.gathered = True

    def free(self) -> None:
        self._buf.untyped_storage().resize_(0)
        self.gathered = False

    def bind(self) -> None:
        """Install views of a fresh leaf aliasing the gathered buffer as the module attributes."""
        with torch.no_grad():
            leaf = torch.empty(0, dtype=self.compute_dtype, device=self._buf.device)
            leaf.set_(self._buf.untyped_storage(), 0, (self.padded,), (1,))
        if self.requires_grad and torch.is_grad_enabled():
            leaf.requires_grad_(True)
            leaf.register_post_accumulate_grad_hook(self._on_grad)
        for slot in self.slots:
            view = leaf[slot.offset : slot.offset + slot.numel].view(slot.shape)
            for module, name in slot.bindings:
                setattr(module, name, view)

    def _on_grad(self, leaf: Tensor) -> None:
        grad = leaf.grad.to(self.shard.dtype)
        leaf.grad = None
        self.free()
        if self._unsynced is not None:
            grad = grad + self._unsynced
            self._unsynced = None
        if not self.sync:
            self._unsynced = grad
            return
        out = torch.empty_like(self.shard)
        work = dist.reduce_scatter_single(out, grad, group=self.group, async_op=True)
        self._inflight.append((work, out, grad))

    def wait_reduced(self) -> Tensor | None:
        """Sum of this rank's reduce-scattered gradient chunks since the last call (shard-group averaged)."""
        total = None
        for work, out, _ in self._inflight:
            work.wait()
            total = out if total is None else total.add_(out)
        self._inflight.clear()
        return None if total is None else total.div_(self.world)

    def full(self) -> Tensor:
        """Unsharded master values (collective over the shard group)."""
        out = torch.empty(self.padded, dtype=self.shard.dtype, device=self.shard.device)
        dist.all_gather_single(out, self.shard.detach(), group=self.group)
        return out


class _Unit:
    def __init__(self, flats: list[FlatParam], reshard_after_forward: bool, compute_dtype: torch.dtype | None) -> None:
        self.flats = flats
        self.reshard_after_forward = reshard_after_forward
        self.compute_dtype = compute_dtype

    def pre_forward(self, module: nn.Module, args: tuple, kwargs: dict) -> tuple[tuple, dict]:
        for f in self.flats:
            f.gather()
            f.bind()
        if self.compute_dtype is not None:
            args = tuple(_cast(a, self.compute_dtype) for a in args)
            kwargs = {k: _cast(v, self.compute_dtype) for k, v in kwargs.items()}
        return args, kwargs

    def post_forward(self, module: nn.Module, args: tuple, kwargs: dict, output: object) -> None:
        tensors = [t for t in _tensors(output) if t.requires_grad]
        if not tensors:
            self.free()
            return
        if self.reshard_after_forward:
            self.free()
        torch.autograd.graph.register_multi_grad_hook(tensors, self.pre_backward, mode="any")

    def pre_backward(self, grad: Tensor) -> None:
        for f in self.flats:
            f.gather()

    def free(self) -> None:
        for f in self.flats:
            f.free()


def _cast(x: object, dtype: torch.dtype) -> object:
    return x.to(dtype) if isinstance(x, Tensor) and x.is_floating_point() else x


def _tensors(x: object) -> Iterator[Tensor]:
    if isinstance(x, Tensor):
        yield x
    elif isinstance(x, tuple | list):
        for v in x:
            yield from _tensors(v)
    elif isinstance(x, dict):
        for v in x.values():
            yield from _tensors(v)


class FSDP(nn.Module):
    """Fully-sharded data parallel over ``shard_group``; replicas over ``replicate_group`` make it HSDP.

    All ranks must hold identical copies of ``module`` when wrapping. ``tp_sharded`` names (FQNs) of parameters
    that are tensor-parallel shards, kept in separate flat groups so norms/gathers can treat them apart.
    Parameters only exist (as views) inside forward/backward; use :meth:`gather_full_params` to read them.
    """

    def __init__(
        self,
        module: nn.Module,
        shard_group: dist.ProcessGroup | None = None,
        replicate_group: dist.ProcessGroup | None = None,
        compute_dtype: torch.dtype | None = None,
        is_unit: UnitFilter = is_block,
        tp_sharded: Collection[str] = (),
    ) -> None:
        super().__init__()
        self.module = module
        self.shard_group = shard_group if shard_group is not None else dist.group.WORLD
        self.replicate_group = replicate_group
        self.replicate_size = 1 if replicate_group is None else dist.get_world_size(replicate_group)
        self.param_fqns = [n for n, _ in module.named_parameters(remove_duplicate=False)]
        unit_names = [n for n, m in module.named_modules() if n and is_unit(n, m)]
        for a in unit_names:
            for b in unit_names:
                if b.startswith(a + "."):
                    raise ValueError(f"FSDP units must not nest: {a!r} contains {b!r}")

        def unit_of(mod_name: str) -> str:
            return next((u for u in unit_names if mod_name == u or mod_name.startswith(u + ".")), "")

        # Parameter identity -> (unit, fqns, bindings), in registration order.
        found: dict[int, tuple[Tensor, str, list[str], list[tuple[nn.Module, str]]]] = {}
        for mod_name, mod in module.named_modules(remove_duplicate=False):
            for pname, p in mod.named_parameters(recurse=False):
                fqn = f"{mod_name}.{pname}" if mod_name else pname
                unit = unit_of(mod_name)
                if id(p) not in found:
                    found[id(p)] = (p, unit, [], [])
                elif found[id(p)][1] != unit:
                    raise ValueError(f"tied parameter {fqn!r} spans FSDP units {found[id(p)][1]!r} and {unit!r}")
                found[id(p)][2].append(fqn)
                if not any(m is mod and n == pname for m, n in found[id(p)][3]):
                    found[id(p)][3].append((mod, pname))

        self._flats: list[FlatParam] = []
        for unit_name in [*unit_names, ""]:
            groups: dict[tuple, list[tuple[Tensor, list[str], list[tuple[nn.Module, str]]]]] = {}
            for p, unit, fqns, bindings in found.values():
                if unit == unit_name:
                    key = (p.requires_grad, any(f in tp_sharded for f in fqns), p.dtype, p.ndim >= 2)
                    groups.setdefault(key, []).append((p, fqns, bindings))
            flats = []
            for (requires_grad, is_tp, _, is_matrix), members in groups.items():
                slots, offset = [], 0
                for p, fqns, bindings in members:
                    slots.append(_Slot(fqns, bindings, p.shape, offset))
                    offset += p.numel()
                flat = FlatParam(
                    slots, [p for p, _, _ in members], requires_grad, is_tp, self.shard_group, compute_dtype
                )
                # Flat shards are 1-D; tag them so weight-decay grouping can still tell matrices from vectors.
                flat.shard.is_matrix = is_matrix
                flats.append(flat)
                for _, _, bindings in members:
                    for mod, pname in bindings:
                        del mod._parameters[pname]
            if not flats:
                continue
            self._flats.extend(flats)
            unit_module = module.get_submodule(unit_name)
            unit = _Unit(flats, reshard_after_forward=bool(unit_name), compute_dtype=compute_dtype)
            unit_module.register_forward_pre_hook(unit.pre_forward, prepend=True, with_kwargs=True)
            unit_module.register_forward_hook(unit.post_forward, with_kwargs=True)
        self.shards = nn.ParameterList([f.shard for f in self._flats])

    @property
    def flat_params(self) -> list[FlatParam]:
        return self._flats

    def forward(self, *args, **kwargs):
        return self.module(*args, **kwargs)

    @contextmanager
    def no_sync(self) -> Iterator[None]:
        """Accumulate unsharded fp32 gradients locally; the first backward outside the context reduces them."""
        for f in self._flats:
            f.sync = False
        try:
            yield
        finally:
            for f in self._flats:
                f.sync = True

    def finish_grad_sync(self) -> None:
        reduced = [(f, f.wait_reduced()) for f in self._flats]
        if self.replicate_size > 1:
            works = [dist.all_reduce(g, group=self.replicate_group, async_op=True) for _, g in reduced if g is not None]
            for w in works:
                w.wait()
        for f, g in reduced:
            if g is not None:
                g.div_(self.replicate_size)
                f.shard.grad = g if f.shard.grad is None else f.shard.grad.add_(g)
            if not f.requires_grad:
                f.free()

    @torch.no_grad()
    def gather_full_params(self) -> dict[str, Tensor]:
        """FQN -> unsharded fp32 parameter (every alias of a tied parameter), collective over the shard group."""
        out: dict[str, Tensor] = {}
        for f in self._flats:
            full = f.full()
            for slot in f.slots:
                value = full[slot.offset : slot.offset + slot.numel].view(slot.shape).clone()
                for fqn in slot.fqns:
                    out[fqn] = value
        return {n: out[n] for n in self.param_fqns}

    def sharded_state_dict(self) -> dict[str, Tensor]:
        """This rank's flat master shards plus the module's buffers (live tensors, not copies)."""
        return {f"_flat.{i}": f.shard for i, f in enumerate(self._flats)} | self.module.state_dict()

    @torch.no_grad()
    def load_sharded_state_dict(self, state_dict: dict[str, Tensor]) -> None:
        for i, f in enumerate(self._flats):
            f.shard.copy_(state_dict[f"_flat.{i}"])
        self.module.load_state_dict({k: v for k, v in state_dict.items() if not k.startswith("_flat.")})
