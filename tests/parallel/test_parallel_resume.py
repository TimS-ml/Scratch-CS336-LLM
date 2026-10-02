"""Resuming from a sharded checkpoint in fresh processes continues training exactly.

Run A trains 3 steps, saves (async) model + optimizer shards, and trains 2 more. Run B, in new processes, restores
step 3 and trains the same 2 steps. Both must end with the same parameters and report the same losses.
"""

from __future__ import annotations

import functools
from pathlib import Path

import pytest
import torch
from toy_lm import OPTIM, make_model, train

from scratch_cs336.checkpoint import Checkpointer
from scratch_cs336.distributed import DistEnv, MeshConfig, build_mesh
from scratch_cs336.distributed.spawn import run_distributed
from scratch_cs336.parallel import Backend, ParallelConfig, Strategy, build_optimizer, parallelize

SAVE_AT, STOP = 3, 5
LAYOUTS = {
    "dp2": (2, MeshConfig(replicate=1, shard=2, tensor=1), [(s, b) for s in Strategy for b in Backend]),
    "tp2xdp2": (4, MeshConfig(replicate=1, shard=2, tensor=2), [(Strategy.FSDP, b) for b in Backend]),
}


def _setup(env: DistEnv, mesh_cfg: MeshConfig, strategy: Strategy, backend: Backend):
    mesh = build_mesh(mesh_cfg, env)
    cfg = ParallelConfig(mesh_cfg, strategy, backend)
    pmodel = parallelize(make_model(tied=True), mesh, cfg)
    return mesh, pmodel, build_optimizer(pmodel, cfg, torch.optim.AdamW, **OPTIM)


def _uninterrupted(env: DistEnv, mesh_cfg: MeshConfig, cases: list, root: Path) -> dict:
    out = {}
    for strategy, backend in cases:
        mesh, pmodel, opt = _setup(env, mesh_cfg, strategy, backend)
        ck = Checkpointer(root / f"{strategy}-{backend}", mesh, async_save=True)
        train(pmodel, opt, mesh, 0, SAVE_AT)
        ck.save(SAVE_AT, {"model": pmodel.sharded_state_dict(), "optim": opt.state_dict(), "step": SAVE_AT})
        losses, _ = train(pmodel, opt, mesh, SAVE_AT, STOP)
        ck.wait()
        out[strategy, backend] = (losses, pmodel.full_state_dict())
    return out


def _resumed(env: DistEnv, mesh_cfg: MeshConfig, cases: list, root: Path) -> dict:
    out = {}
    for strategy, backend in cases:
        mesh, pmodel, opt = _setup(env, mesh_cfg, strategy, backend)
        ck = Checkpointer(root / f"{strategy}-{backend}", mesh)
        state = ck.load(ck.latest_step())
        pmodel.load_sharded_state_dict(state["model"])
        opt.load_state_dict(state["optim"])
        losses, _ = train(pmodel, opt, mesh, state["step"], STOP)
        out[strategy, backend] = (losses, pmodel.full_state_dict())
    return out


@functools.cache
def _run(layout: str, root: Path) -> tuple[list[dict], list[dict]]:
    world, mesh_cfg, cases = LAYOUTS[layout]
    first = run_distributed(_uninterrupted, world, mesh_cfg, cases, root)
    return first, run_distributed(_resumed, world, mesh_cfg, cases, root)


@pytest.fixture(scope="module")
def ckpt_root(tmp_path_factory: pytest.TempPathFactory) -> Path:
    return tmp_path_factory.mktemp("resume")


@pytest.mark.parametrize(
    ("layout", "strategy", "backend"),
    [pytest.param(layout, s, b, id=f"{layout}-{s}-{b}") for layout, (_, _, cases) in LAYOUTS.items() for s, b in cases],
)
def test_resume_matches_uninterrupted_run(layout: str, strategy: Strategy, backend: Backend, ckpt_root: Path) -> None:
    first, resumed = _run(layout, ckpt_root / layout)
    for a, b in zip(first, resumed, strict=True):
        losses_a, state_a = a[strategy, backend]
        losses_b, state_b = b[strategy, backend]
        assert losses_a == losses_b
        assert state_a.keys() == state_b.keys()
        assert all(torch.equal(state_b[name], state_a[name]) for name in state_a)  # bitwise
