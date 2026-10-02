import pytest
import torch
import torch.distributed as dist

from scratch_cs336.distributed import DistEnv, MeshConfig, build_mesh
from scratch_cs336.distributed.spawn import run_distributed


@pytest.mark.parametrize(
    ("cfg", "world", "expected"),
    [
        (MeshConfig(), 8, (1, 8, 1)),
        (MeshConfig(replicate=2, shard=-1, tensor=2), 8, (2, 2, 2)),
        (MeshConfig(replicate=-1, shard=2, tensor=1), 6, (3, 2, 1)),
    ],
)
def test_resolve(cfg, world, expected):
    assert cfg.resolve(world) == expected


@pytest.mark.parametrize(
    "cfg",
    [MeshConfig(replicate=-1, shard=-1), MeshConfig(shard=3), MeshConfig(shard=-1, tensor=3), MeshConfig(tensor=0)],
)
def test_resolve_rejects(cfg):
    with pytest.raises(ValueError):
        cfg.resolve(8)


def _groups(env: DistEnv):
    mesh = build_mesh(MeshConfig(replicate=1, shard=-1, tensor=2), env)
    t = torch.tensor([float(env.rank)])
    dp, tp = t.clone(), t.clone()
    dist.all_reduce(dp, group=mesh.dp_group)
    dist.all_reduce(tp, group=mesh.tp_group)
    return mesh.dp_rank, mesh.tp_rank, dp.item(), tp.item()


def test_groups_partition_ranks():
    # world 4, mesh (1, 2, 2): tp pairs {0,1},{2,3}; dp pairs {0,2},{1,3}.
    out = run_distributed(_groups, 4)
    assert [o[:2] for o in out] == [(0, 0), (0, 1), (1, 0), (1, 1)]
    assert [o[2] for o in out] == [2.0, 4.0, 2.0, 4.0]
    assert [o[3] for o in out] == [1.0, 1.0, 5.0, 5.0]
