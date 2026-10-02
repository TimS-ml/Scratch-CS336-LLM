"""CS336 hw2 contracts for the scratch building blocks used directly (no ``parallelize``)."""

from __future__ import annotations

import copy
import functools

import pytest
import torch
import torch.distributed as dist
import torch.nn.functional as F
from torch import nn
from toy_lm import VOCAB, make_model

from scratch_cs336.distributed import DistEnv
from scratch_cs336.distributed.spawn import run_distributed
from scratch_cs336.parallel.ddp import DDP
from scratch_cs336.parallel.fsdp import FSDP
from scratch_cs336.parallel.zero import ShardedOptimizer

WORLD = 2
BUCKETS = [None, 0.004]  # one parameter per bucket / ~4 KiB buckets holding several parameters


class _FC2(nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self.fc = nn.Linear(10, 50, bias=True)
        self.fc.bias.requires_grad = False

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.fc(x)


class ToyModel(nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self.fc1 = nn.Linear(10, 10, bias=False)
        self.fc2 = _FC2()
        self.fc3 = nn.Linear(50, 10, bias=False)
        self.no_grad_fixed_param = nn.Parameter(torch.tensor([2.0, 2.0]), requires_grad=False)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.fc3(F.relu(self.fc2(F.relu(self.fc1(x)))))


class ToyModelWithTiedWeights(nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self.fc1 = nn.Linear(10, 10, bias=False)
        self.fc2 = nn.Linear(10, 50, bias=False)
        self.fc3 = nn.Linear(50, 10, bias=False)
        self.fc4 = nn.Linear(10, 50, bias=False)
        self.fc5 = nn.Linear(50, 10, bias=False)
        self.fc4.weight = self.fc2.weight

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        for fc in (self.fc1, self.fc2, self.fc3, self.fc4):
            x = F.relu(fc(x))
        return self.fc5(x)


MODELS = {"toy": ToyModel, "tied": ToyModelWithTiedWeights}


def _max_diff(a: nn.Module | dict, b: nn.Module) -> float:
    ours = dict(a.named_parameters()) if isinstance(a, nn.Module) else a
    return max((ours[n] - p).abs().max().item() for n, p in b.named_parameters())


def _ddp_case(env: DistEnv, model_cls: type[nn.Module], bucket_size_mb: float | None) -> dict[str, float]:
    torch.manual_seed(0)
    baseline = model_cls()
    torch.manual_seed(env.rank)  # ranks start from different weights; wrapping must broadcast rank 0's
    local_init = model_cls()
    out = {"before_wrap": _max_diff(local_init, baseline)}
    ddp = DDP(local_init, bucket_size_mb=bucket_size_mb)
    out["after_wrap"] = _max_diff(ddp.module, baseline)
    gen = torch.Generator().manual_seed(42)
    x, y = torch.randn(20, 10, generator=gen), torch.randn(20, 10, generator=gen)
    opt_ddp = torch.optim.SGD(ddp.parameters(), lr=0.1)
    opt_base = torch.optim.SGD(baseline.parameters(), lr=0.1)
    local = slice(env.rank * 10, (env.rank + 1) * 10)
    for _ in range(5):
        F.mse_loss(ddp(x[local]), y[local]).backward()
        ddp.finish_grad_sync()
        opt_ddp.step()
        opt_ddp.zero_grad()
        F.mse_loss(baseline(x), y).backward()
        opt_base.step()
        opt_base.zero_grad()
    out["after_training"] = _max_diff(ddp.module, baseline)
    return out


def _zero_case(env: DistEnv, model_cls: type[nn.Module]) -> dict:
    torch.manual_seed(42)
    model = model_cls()
    sharded_model = copy.deepcopy(model)
    kw = {"lr": 0.1, "weight_decay": 0.1, "betas": (0.9, 0.999), "eps": 1e-8}
    opt = torch.optim.AdamW(model.parameters(), **kw)
    sharded = ShardedOptimizer(sharded_model.parameters(), torch.optim.AdamW, **kw)
    for _ in range(10):
        x, y = torch.rand(32, 10), torch.rand(32, 10)  # same stream on every rank
        for m, o in ((model, opt), (sharded_model, sharded)):
            o.zero_grad()
            ((y - m(x)) ** 2).sum().backward()
            o.step()
    index = {id(p): i for i, p in enumerate(sharded_model.parameters())}
    trainable = {index[id(p)] for p in sharded_model.parameters() if p.requires_grad}
    return {
        "max_diff": _max_diff(sharded_model, model),
        "owned": {index[id(p)] for p in sharded.inner.state},
        "trainable": trainable,
    }


def _fsdp_case(env: DistEnv, compute_dtype: torch.dtype) -> dict:
    baseline = make_model()
    fsdp = FSDP(make_model(), compute_dtype=compute_dtype)
    opt_fsdp = torch.optim.SGD(fsdp.parameters(), lr=0.01)
    opt_base = torch.optim.SGD(baseline.parameters(), lr=0.01)
    gen = torch.Generator().manual_seed(123)
    ids = torch.randint(0, VOCAB, (8, 9), generator=gen)
    local = ids[env.rank * 4 : (env.rank + 1) * 4]
    out = {"diffs": [], "grad_ok": [], "logits_dtype": None}
    for _ in range(3):
        logits = fsdp(local[:, :-1])
        out["logits_dtype"] = logits.dtype
        F.cross_entropy(logits.float().reshape(-1, VOCAB), local[:, 1:].reshape(-1)).backward()
        fsdp.finish_grad_sync()
        out["grad_ok"].append(
            all(p.grad.dtype == torch.float32 and p.grad.shape == p.shape for p in fsdp.parameters() if p.requires_grad)
        )
        opt_fsdp.step()
        opt_fsdp.zero_grad()

        cast = {n: p.to(compute_dtype) for n, p in baseline.named_parameters()}
        base_logits = torch.func.functional_call(baseline, cast, (ids[:, :-1],))
        F.cross_entropy(base_logits.float().reshape(-1, VOCAB), ids[:, 1:].reshape(-1)).backward()
        opt_base.step()
        opt_base.zero_grad()
        out["diffs"].append(_max_diff(fsdp.gather_full_params(), baseline))
    return out


def _all_cases(env: DistEnv) -> dict:
    out = {}
    for name, cls in MODELS.items():
        for bucket in BUCKETS:
            out["ddp", name, bucket] = _ddp_case(env, cls, bucket)
        out["zero", name] = _zero_case(env, cls)
    for dtype in (torch.float32, torch.bfloat16):
        out["fsdp", dtype] = _fsdp_case(env, dtype)
    dist.barrier()
    return out


@functools.cache
def _results() -> list[dict]:
    return run_distributed(_all_cases, WORLD)


@pytest.mark.parametrize("bucket_size_mb", BUCKETS, ids=["per-param", "bucketed"])
@pytest.mark.parametrize("model", list(MODELS))
def test_ddp_broadcasts_and_matches_single_process(model: str, bucket_size_mb: float | None) -> None:
    for rank, results in enumerate(_results()):
        r = results["ddp", model, bucket_size_mb]
        assert (r["before_wrap"] > 0) == (rank > 0)
        assert r["after_wrap"] == 0.0
        assert r["after_training"] < 1e-6


@pytest.mark.parametrize("model", list(MODELS))
def test_sharded_optimizer_matches_adamw_and_partitions_state(model: str) -> None:
    ranks = [rank["zero", model] for rank in _results()]
    for r in ranks:
        assert r["max_diff"] < 1e-6
        assert r["owned"] < r["trainable"]  # each rank holds state for a strict subset
    assert set.union(*(r["owned"] for r in ranks)) == ranks[0]["trainable"]
    assert sum(len(r["owned"]) for r in ranks) == len(ranks[0]["trainable"])


@pytest.mark.parametrize(("dtype", "atol"), [(torch.float32, 1e-6), (torch.bfloat16, 1e-4)], ids=["fp32", "bf16"])
def test_fsdp_matches_single_process(dtype: torch.dtype, atol: float) -> None:
    for rank in _results():
        r = rank["fsdp", dtype]
        assert r["logits_dtype"] == dtype
        assert all(r["grad_ok"])
        assert max(r["diffs"]) < atol, r["diffs"]
