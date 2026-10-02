"""Scratch tensor parallelism (Megatron-style) driven by a model's ``tp_plan()``.

A plan maps fnmatch patterns over module FQNs to a :class:`TPStyle`:

* ``COLWISE``: ``weight [out, in]`` (and ``bias``) sharded on dim 0. The module input *enters* the TP region:
  identity forward, all-reduce of its gradient over ``tp`` backward.
* ``ROWWISE``: ``weight`` sharded on dim 1 (no bias). The partial output is all-reduced over ``tp`` forward,
  identity backward.
* ``HEADWISE``: every parameter of the module sharded on dim 0, no communication.
* ``REPLICATE``: parameters stay whole on every tp rank but are used inside the TP region (e.g. a per-head
  q/k norm), so each tp rank only sees a partial gradient; the parameter itself enters the region, i.e. its
  gradient is summed over ``tp``.

Models derive head counts from weight shapes, so sharding weights is all TP needs to do to a module.
"""

from __future__ import annotations

from collections.abc import Callable, Iterator, Mapping
from fnmatch import fnmatchcase

import torch
import torch.distributed as dist
from torch import Tensor, nn

from scratch_cs336.parallel.plan import TPStyle

ParamFn = Callable[[str, Tensor], Tensor]


class _CopyToRegion(torch.autograd.Function):
    @staticmethod
    def forward(ctx, x: Tensor, group: dist.ProcessGroup) -> Tensor:
        ctx.group = group
        return x.view_as(x)

    @staticmethod
    def backward(ctx, grad: Tensor) -> tuple[Tensor, None]:
        grad = grad.contiguous().clone()
        dist.all_reduce(grad, group=ctx.group)
        return grad, None


class _ReduceFromRegion(torch.autograd.Function):
    @staticmethod
    def forward(ctx, x: Tensor, group: dist.ProcessGroup) -> Tensor:
        out = x.clone(memory_format=torch.contiguous_format)
        dist.all_reduce(out, group=group)
        return out

    @staticmethod
    def backward(ctx, grad: Tensor) -> tuple[Tensor, None]:
        return grad, None


def copy_to_tp_region(x: Tensor, group: dist.ProcessGroup) -> Tensor:
    return _CopyToRegion.apply(x, group)


def reduce_from_tp_region(x: Tensor, group: dist.ProcessGroup) -> Tensor:
    return _ReduceFromRegion.apply(x, group)


def resolve_plan(model: nn.Module, plan: Mapping[str, TPStyle]) -> dict[str, TPStyle]:
    """Module FQN -> style for every module matched by a plan pattern; patterns matching nothing are ignored."""
    styles: dict[str, TPStyle] = {}
    for name, _ in model.named_modules():
        matched = {TPStyle(style) for pattern, style in plan.items() if fnmatchcase(name, pattern)}
        if len(matched) > 1:
            raise ValueError(f"module {name!r} matches conflicting tp styles {sorted(matched)}")
        if matched:
            styles[name] = matched.pop()
    return styles


def _join(prefix: str, name: str) -> str:
    return f"{prefix}.{name}" if prefix else name


def _sharded_params(module: nn.Module, style: TPStyle) -> Iterator[tuple[str, int]]:
    """(param name relative to ``module``, shard dim) for every parameter the style shards."""
    match style:
        case TPStyle.COLWISE:
            names = {n for n, _ in module.named_parameters(recurse=False)}
            if "weight" not in names or names - {"weight", "bias"}:
                raise ValueError(f"colwise module must hold exactly `weight` (+ `bias`), got {sorted(names)}")
            yield from ((n, 0) for n in sorted(names))
        case TPStyle.ROWWISE:
            names = {n for n, _ in module.named_parameters(recurse=False)}
            if names != {"weight"}:
                raise ValueError(f"rowwise module must hold exactly `weight` (no bias), got {sorted(names)}")
            yield "weight", 1
        case TPStyle.HEADWISE:
            yield from ((n, 0) for n, _ in module.named_parameters())
        case TPStyle.REPLICATE:
            return


def tp_shard_dims(model: nn.Module, styles: Mapping[str, TPStyle]) -> dict[str, int]:
    """Param FQN (every alias of a tied parameter) -> dim it is sharded on over ``tp``.

    Computed on the unsharded model; parameters absent from the result are replicated across ``tp``.
    """
    by_id: dict[int, int] = {}
    for mod_name, style in styles.items():
        module = model.get_submodule(mod_name)
        for pname, dim in _sharded_params(module, style):
            param = module.get_parameter(pname)
            if by_id.get(id(param), dim) != dim:
                raise ValueError(f"parameter {_join(mod_name, pname)!r} is sharded twice on different dims")
            by_id[id(param)] = dim
    return {n: by_id[id(p)] for n, p in model.named_parameters(remove_duplicate=False) if id(p) in by_id}


def check_shardable(model: nn.Module, styles: Mapping[str, TPStyle], tp_size: int) -> None:
    aliases: dict[int, int] = {}
    for _, p in model.named_parameters(remove_duplicate=False):
        aliases[id(p)] = aliases.get(id(p), 0) + 1
    for mod_name, style in styles.items():
        module = model.get_submodule(mod_name)
        for pname, dim in _sharded_params(module, style):
            param = module.get_parameter(pname)
            if param.shape[dim] % tp_size:
                raise ValueError(
                    f"{_join(mod_name, pname)} dim {dim} ({param.shape[dim]}) not divisible by tp={tp_size}"
                )
            if aliases[id(param)] > 1:
                raise ValueError(f"tied parameter {_join(mod_name, pname)!r} cannot be tensor-parallel sharded")


def swap_params_in_forward(module: nn.Module, fn: ParamFn) -> None:
    """While ``module.forward`` runs, each of its direct parameters reads as ``fn(name, param)``.

    The attribute is swapped, not the registered parameter object, so optimizers, DDP hooks and FSDP views
    are untouched; gradients reach the parameter through ``fn``'s autograd graph. Works whether the attribute
    is a registered ``nn.Parameter`` or a plain tensor installed by an FSDP unit.
    """
    names = [n for n, _ in module.named_parameters(recurse=False)]
    if not names:
        return
    stack: list[list[tuple[str, Tensor, bool]]] = []

    def pre_hook(mod: nn.Module, args: tuple) -> None:
        saved = []
        for n in names:
            t = getattr(mod, n)
            registered = n in mod._parameters
            if registered:
                del mod._parameters[n]
            mod.__dict__[n] = fn(n, t)
            saved.append((n, t, registered))
        stack.append(saved)

    def post_hook(mod: nn.Module, args: tuple, output: object) -> None:
        for n, t, registered in stack.pop():
            del mod.__dict__[n]
            if registered:
                mod._parameters[n] = t
            else:
                mod.__dict__[n] = t

    module.register_forward_pre_hook(pre_hook)
    module.register_forward_hook(post_hook, always_call=True)


def _shard_(module: nn.Module, name: str, dim: int, rank: int, size: int) -> None:
    p = module.get_parameter(name)
    owner = module.get_submodule(name.rpartition(".")[0])
    local = p.detach().chunk(size, dim)[rank].clone()
    owner.register_parameter(name.rpartition(".")[2], nn.Parameter(local, requires_grad=p.requires_grad))


def apply_tp(model: nn.Module, styles: Mapping[str, TPStyle], group: dist.ProcessGroup) -> None:
    """Shard ``model`` in place over ``group`` according to resolved ``styles`` (see :func:`resolve_plan`)."""
    rank, size = dist.get_rank(group), dist.get_world_size(group)
    check_shardable(model, styles, size)
    for mod_name, style in styles.items():
        module = model.get_submodule(mod_name)
        for pname, dim in list(_sharded_params(module, style)):
            _shard_(module, pname, dim, rank, size)
        match style:
            case TPStyle.COLWISE:
                module.register_forward_pre_hook(lambda m, args: (copy_to_tp_region(args[0], group), *args[1:]))
            case TPStyle.ROWWISE:
                module.register_forward_hook(lambda m, args, out: reduce_from_tp_region(out, group))
            case TPStyle.REPLICATE:
                for sub in module.modules():
                    swap_params_in_forward(sub, lambda n, t: copy_to_tp_region(t, group))


def gather_tp(t: Tensor, dim: int, group: dist.ProcessGroup) -> Tensor:
    """Concatenate the tp shards of ``t`` along ``dim`` (collective over ``group``)."""
    parts = [torch.empty_like(t) for _ in range(dist.get_world_size(group))]
    dist.all_gather(parts, t.contiguous(), group=group)
    return torch.cat(parts, dim=dim)
