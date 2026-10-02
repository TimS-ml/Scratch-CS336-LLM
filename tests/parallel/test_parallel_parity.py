"""Every strategy x backend (and the 2-D layouts) trains exactly like a single process.

Each layout runs all its cases in one spawned world; every case trains the toy LM for ``STEPS`` AdamW steps with
gradient accumulation and clipping, and returns per-rank losses, pre-clip norms and ``full_state_dict()``.
"""

from __future__ import annotations

import functools
from dataclasses import dataclass

import pytest
import torch
from toy_lm import MAX_NORM, OPTIM, make_model, reference_run, train

from scratch_cs336.distributed import DistEnv, MeshConfig, build_mesh
from scratch_cs336.distributed.spawn import run_distributed
from scratch_cs336.parallel import Backend, ParallelConfig, Strategy, build_optimizer, parallelize

STEPS = 5
LAYOUTS = {
    "dp2": (2, MeshConfig(replicate=1, shard=2, tensor=1)),
    "tp2xdp2": (4, MeshConfig(replicate=1, shard=2, tensor=2)),
    "hsdp2x2": (4, MeshConfig(replicate=2, shard=2, tensor=1)),
}


@dataclass(frozen=True)
class Case:
    strategy: Strategy
    backend: Backend
    tied: bool = False
    compute_dtype: str = "float32"
    activation_checkpointing: bool = False

    def __str__(self) -> str:
        flags = ["tied"] * self.tied + ["bf16"] * (self.compute_dtype == "bfloat16")
        flags += ["ac"] * self.activation_checkpointing
        return "-".join([self.strategy, self.backend, *flags])


def _cases(layout: str) -> list[Case]:
    grid = [(s, b) for s in Strategy for b in Backend]
    cases = [Case(s, b, tied) for s, b in grid for tied in (False, True)]
    if layout == "dp2":
        cases += [Case(s, b, compute_dtype="bfloat16") for s, b in grid]
        cases += [Case(s, b, activation_checkpointing=True) for s, b in grid]
    return cases


def _run_cases(env: DistEnv, mesh_cfg: MeshConfig, cases: list[Case]) -> dict[Case, tuple]:
    mesh = build_mesh(mesh_cfg, env)
    out = {}
    for case in cases:
        cfg = ParallelConfig(mesh_cfg, case.strategy, case.backend, case.compute_dtype, case.activation_checkpointing)
        pmodel = parallelize(make_model(case.tied), mesh, cfg)
        opt = build_optimizer(pmodel, cfg, torch.optim.AdamW, **OPTIM)
        losses, norms = train(pmodel, opt, mesh, 0, STEPS)
        out[case] = (losses, norms, pmodel.full_state_dict())
    return out


@functools.cache
def _spawn(layout: str) -> list[dict[Case, tuple]] | Exception:
    world, mesh_cfg = LAYOUTS[layout]
    try:
        return run_distributed(_run_cases, world, mesh_cfg, _cases(layout))
    except Exception as e:  # cached so one broken layout costs one spawn, not one per test
        return e


def _layout_results(layout: str) -> list[dict[Case, tuple]]:
    result = _spawn(layout)
    if isinstance(result, Exception):
        raise result
    return result


@functools.cache
def _reference(tied: bool) -> tuple:
    return reference_run(make_model(tied), STEPS)


PARAMS = [pytest.param(layout, case, id=f"{layout}-{case}") for layout in LAYOUTS for case in _cases(layout)]


@pytest.mark.parametrize(("layout", "case"), [p for p in PARAMS if p.values[1].compute_dtype == "float32"])
def test_matches_single_process(layout: str, case: Case) -> None:
    ref_losses, ref_norms, ref_state = _reference(case.tied)
    assert max(ref_norms) > MAX_NORM  # clipping is exercised
    for losses, norms, state in (rank[case] for rank in _layout_results(layout)):
        torch.testing.assert_close(losses, ref_losses, atol=1e-5, rtol=0)
        torch.testing.assert_close(norms, ref_norms, atol=1e-5, rtol=0)
        assert list(state) == list(ref_state)
        for name, ref in ref_state.items():
            torch.testing.assert_close(state[name], ref, atol=1e-5, rtol=0, msg=lambda m, n=name: f"{n}: {m}")


@pytest.mark.parametrize(("layout", "case"), [p for p in PARAMS if p.values[1].compute_dtype == "bfloat16"])
def test_bf16_compute_stays_close(layout: str, case: Case) -> None:
    ref_losses, _, ref_state = _reference(case.tied)
    for losses, _, state in (rank[case] for rank in _layout_results(layout)):
        assert all(s.dtype == torch.float32 for s in state.values())
        torch.testing.assert_close(losses, ref_losses, atol=2e-2, rtol=0)
        for name, ref in ref_state.items():
            torch.testing.assert_close(state[name], ref, atol=2e-2, rtol=0, msg=lambda m, n=name: f"{n}: {m}")
