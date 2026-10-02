import json
import math

import pytest
import torch
from torch import nn

from scratch_cs336.distributed import DistEnv
from scratch_cs336.distributed.spawn import run_distributed
from scratch_cs336.tracking import JsonlTracker, TrackerConfig, build_tracker
from scratch_cs336.train.optim import AdamW, ScheduleConfig, ScheduleName, lr_at, param_groups


class Net(nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self.fc1 = nn.Linear(8, 16)
        self.norm = nn.LayerNorm(16)
        self.fc2 = nn.Linear(16, 4)

    def forward(self, x):
        return self.fc2(torch.relu(self.norm(self.fc1(x))))


def test_adamw_matches_torch_with_groups():
    torch.manual_seed(0)
    a, b = Net(), Net()
    b.load_state_dict(a.state_dict())
    kw = dict(lr=1e-2, betas=(0.9, 0.95), eps=1e-8)
    oa = AdamW(param_groups(a, 0.1), **kw)
    ob = torch.optim.AdamW(param_groups(b, 0.1), foreach=False, **kw)
    x, y = torch.randn(32, 8), torch.randn(32, 4)
    for _ in range(20):
        for m, o in ((a, oa), (b, ob)):
            o.zero_grad()
            ((m(x) - y) ** 2).mean().backward()
            o.step()
    for pa, pb in zip(a.parameters(), b.parameters(), strict=True):
        torch.testing.assert_close(pa, pb, rtol=1e-5, atol=1e-6)


def test_param_groups_decay_split_and_tied_once():
    m = nn.Sequential(nn.Linear(4, 4), nn.LayerNorm(4))
    m.tied = m[0]
    g = param_groups(m, 0.1)
    assert [x["weight_decay"] for x in g] == [0.1, 0.0]
    assert [p.ndim for p in g[0]["params"]] == [2]
    assert sum(len(x["params"]) for x in g) == 4


def _fsdp_parity(env: DistEnv):
    from torch.distributed.device_mesh import init_device_mesh
    from torch.distributed.fsdp import fully_shard
    from torch.distributed.tensor import DTensor

    torch.manual_seed(0)
    ref, sharded = Net(), Net()
    sharded.load_state_dict(ref.state_dict())
    mesh = init_device_mesh("cpu", (env.world_size,))
    fully_shard(sharded, mesh=mesh)
    kw = dict(lr=1e-2, betas=(0.9, 0.95), eps=1e-8)
    o_ref = torch.optim.AdamW(param_groups(ref, 0.1), foreach=False, **kw)
    o_sh = AdamW(param_groups(sharded, 0.1), **kw)
    assert all(isinstance(p, DTensor) for p in sharded.parameters())
    g = torch.Generator().manual_seed(1)
    for _ in range(10):
        x, y = torch.randn(16, 8, generator=g), torch.randn(16, 4, generator=g)
        # Same global batch on every rank: FSDP averages identical grads, equal to the full-batch ref grad.
        for m, o in ((ref, o_ref), (sharded, o_sh)):
            o.zero_grad()
            ((m(x) - y) ** 2).mean().backward()
            o.step()
    return {k: v.full_tensor() for k, v in sharded.state_dict().items()}, ref.state_dict()


def test_adamw_fsdp2_dtensor_parity():
    for got, want in run_distributed(_fsdp_parity, 2):
        for k in want:
            torch.testing.assert_close(got[k], want[k], rtol=1e-4, atol=1e-5)


def cs336_cosine(it, max_lr, min_lr, warmup, cycle):
    if it < warmup:
        return it / warmup * max_lr
    if it > cycle:
        return min_lr
    return min_lr + 0.5 * (1 + math.cos((it - warmup) / (cycle - warmup) * math.pi)) * (max_lr - min_lr)


@pytest.mark.parametrize("it", [0, 3, 7, 8, 9, 15, 20, 21, 25])
def test_cosine_matches_cs336(it):
    cfg = ScheduleConfig(ScheduleName.COSINE, warmup_steps=7, total_steps=21, min_lr_ratio=0.1)
    assert lr_at(it, 1.0, cfg) == pytest.approx(cs336_cosine(it, 1.0, 0.1, 7, 21))


def test_cosine_boundaries():
    cfg = ScheduleConfig(ScheduleName.COSINE, warmup_steps=10, total_steps=110, min_lr_ratio=0.1)
    assert lr_at(10, 2.0, cfg) == pytest.approx(2.0)
    assert lr_at(60, 2.0, cfg) == pytest.approx(1.1)
    assert lr_at(110, 2.0, cfg) == pytest.approx(0.2)
    assert lr_at(500, 2.0, cfg) == pytest.approx(0.2)


def test_wsd():
    cfg = ScheduleConfig(ScheduleName.WSD, warmup_steps=10, total_steps=100, min_lr_ratio=0.1, decay_steps=20)
    assert lr_at(5, 1.0, cfg) == pytest.approx(0.5)
    assert lr_at(79, 1.0, cfg) == pytest.approx(1.0)
    assert lr_at(80, 1.0, cfg) == pytest.approx(1.0)
    assert lr_at(90, 1.0, cfg) == pytest.approx(0.55)
    assert lr_at(100, 1.0, cfg) == pytest.approx(0.1)


def test_jsonl_round_trip_and_non_main_is_null(tmp_path):
    t = JsonlTracker(tmp_path / "m.jsonl")
    t.log({"loss": 1.5}, step=3)
    rows = [json.loads(line) for line in (tmp_path / "m.jsonl").read_text().splitlines()]  # flushed before finish
    assert rows[0]["loss"] == 1.5 and rows[0]["step"] == 3 and "time" in rows[0]
    t.finish()
    build_tracker(TrackerConfig(), tmp_path / "x", is_main=False).log({"a": 1.0}, 0)
    assert not (tmp_path / "x").exists()
