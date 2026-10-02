"""AdamW from scratch, parameter grouping, and learning-rate schedules."""

from __future__ import annotations

import math
from collections.abc import Callable, Iterable
from dataclasses import dataclass
from enum import StrEnum
from typing import Any

import torch
from torch import nn

from scratch_cs336.parallel.api import ParallelConfig, ParallelModel, build_optimizer


class AdamW(torch.optim.Optimizer):
    """Decoupled-weight-decay Adam with bias correction.

    Uses only elementwise tensor ops, so params may be plain tensors, flat shards, or DTensors.
    Matches ``torch.optim.AdamW(foreach=False, fused=False)``.
    """

    def __init__(
        self,
        params: Iterable[torch.Tensor] | Iterable[dict[str, Any]],
        lr: float = 1e-3,
        betas: tuple[float, float] = (0.9, 0.999),
        eps: float = 1e-8,
        weight_decay: float = 1e-2,
    ) -> None:
        if lr < 0 or eps < 0 or weight_decay < 0 or not all(0 <= b < 1 for b in betas):
            raise ValueError(f"invalid AdamW hyperparameters: {lr=} {betas=} {eps=} {weight_decay=}")
        super().__init__(params, dict(lr=lr, betas=betas, eps=eps, weight_decay=weight_decay))

    @torch.no_grad()
    def step(self, closure: Callable[[], float] | None = None) -> float | None:
        loss = None
        if closure is not None:
            with torch.enable_grad():
                loss = closure()
        for group in self.param_groups:
            lr, (beta1, beta2), eps, wd = group["lr"], group["betas"], group["eps"], group["weight_decay"]
            for p in group["params"]:
                if p.grad is None:
                    continue
                state = self.state[p]
                if not state:
                    state["step"] = 0
                    state["exp_avg"] = torch.zeros_like(p)
                    state["exp_avg_sq"] = torch.zeros_like(p)
                state["step"] += 1
                t = state["step"]
                m, v, g = state["exp_avg"], state["exp_avg_sq"], p.grad
                p.mul_(1 - lr * wd)
                m.lerp_(g, 1 - beta1)
                v.mul_(beta2).addcmul_(g, g, value=1 - beta2)
                bias1 = 1 - beta1**t
                bias2_sqrt = math.sqrt(1 - beta2**t)
                denom = (v.sqrt() / bias2_sqrt).add_(eps)
                p.addcdiv_(m, denom, value=-lr / bias1)
        return loss


def param_groups(model: nn.Module | ParallelModel, weight_decay: float) -> list[dict[str, Any]]:
    """Decay matrices (ndim >= 2); no decay for norms, biases, and other 1-D params. Tied params appear once.

    Works on a plain module or a ``ParallelModel`` (its local shards); flat FSDP shards carry ``is_matrix``.
    """
    decay, no_decay = [], []
    for p in model.parameters():  # parameters() already de-duplicates tied tensors
        if p.requires_grad:
            (decay if getattr(p, "is_matrix", p.ndim >= 2) else no_decay).append(p)
    groups = [{"params": decay, "weight_decay": weight_decay}, {"params": no_decay, "weight_decay": 0.0}]
    return [g for g in groups if g["params"]]


class OptimizerName(StrEnum):
    ADAMW = "adamw"
    TORCH_ADAMW = "torch_adamw"


@dataclass(frozen=True)
class OptimizerConfig:
    name: OptimizerName = OptimizerName.ADAMW
    lr: float = 3e-4
    betas: tuple[float, float] = (0.9, 0.95)
    eps: float = 1e-8
    weight_decay: float = 0.1


def optimizer_spec(cfg: OptimizerConfig) -> tuple[type[torch.optim.Optimizer], dict[str, Any]]:
    """Optimizer class and constructor kwargs (lr, betas, eps, weight_decay) for ``cfg``."""
    cls = {OptimizerName.ADAMW: AdamW, OptimizerName.TORCH_ADAMW: torch.optim.AdamW}[OptimizerName(cfg.name)]
    kwargs: dict[str, Any] = dict(lr=cfg.lr, betas=cfg.betas, eps=cfg.eps, weight_decay=cfg.weight_decay)
    if cls is torch.optim.AdamW:
        kwargs.update(foreach=False, fused=False)
    return cls, kwargs


def build_inner_optimizer(
    params_or_groups: Iterable[torch.Tensor] | Iterable[dict[str, Any]], cfg: OptimizerConfig
) -> torch.optim.Optimizer:
    cls, kwargs = optimizer_spec(cfg)
    return cls(params_or_groups, **kwargs)


def build_parallel_optimizer(
    pmodel: ParallelModel, parallel: ParallelConfig, cfg: OptimizerConfig
) -> torch.optim.Optimizer:
    """``cfg``'s optimizer over ``pmodel``'s local shards with :func:`param_groups` weight decay."""
    cls, kwargs = optimizer_spec(cfg)
    return build_optimizer(pmodel, parallel, cls, param_groups(pmodel, cfg.weight_decay), **kwargs)


class ScheduleName(StrEnum):
    COSINE = "cosine"
    WSD = "wsd"
    CONSTANT = "constant"


@dataclass(frozen=True)
class ScheduleConfig:
    name: ScheduleName = ScheduleName.COSINE
    warmup_steps: int = 0
    total_steps: int = 1
    min_lr_ratio: float = 0.1
    decay_steps: int | None = None  # WSD only; None = the last fifth of total_steps


def lr_at(step: int, max_lr: float, cfg: ScheduleConfig) -> float:
    """Learning rate for 0-indexed optimizer ``step``."""
    warmup = cfg.warmup_steps
    if step < warmup:
        return step / warmup * max_lr
    min_lr = max_lr * cfg.min_lr_ratio
    match ScheduleName(cfg.name):
        case ScheduleName.CONSTANT:
            return max_lr
        case ScheduleName.COSINE:
            if step > cfg.total_steps:
                return min_lr
            progress = (step - warmup) / max(cfg.total_steps - warmup, 1)
            return min_lr + 0.5 * (1 + math.cos(math.pi * progress)) * (max_lr - min_lr)
        case ScheduleName.WSD:
            decay = cfg.decay_steps if cfg.decay_steps is not None else int(0.2 * cfg.total_steps)
            decay_start = cfg.total_steps - decay
            if step < decay_start:
                return max_lr
            if step >= cfg.total_steps:
                return min_lr
            return max_lr + (step - decay_start) / decay * (min_lr - max_lr)
