from __future__ import annotations

import functools
import time
from pathlib import Path

import pytest
import torch
import torch.distributed as dist

from scratch_cs336.checkpoint import Checkpointer
from scratch_cs336.distributed import DistEnv, MeshConfig, build_mesh
from scratch_cs336.distributed.spawn import run_distributed


def _state(step: int, rank: int) -> dict:
    return {
        "w": torch.full((3,), float(step + 100 * rank)),
        "step": step,
        "nested": {"ids": [torch.arange(4) * step], "betas": (0.9, 0.95), "none": None},
    }


def _scenario(env: DistEnv, root: Path) -> dict:
    mesh = build_mesh(MeshConfig(replicate=1, shard=2, tensor=1), env)
    out = {}

    ck = Checkpointer(root / "async", mesh, keep_last=2, permanent_every=3, async_save=True)
    for step in range(1, 7):
        ck.save(step, _state(step, env.rank))
    ck.wait()
    if env.rank == 0:  # a save that crashed after one rank wrote its shard
        (root / "async" / "step_000007").mkdir()
        torch.save(_state(7, 0), root / "async" / "step_000007" / "rank_00000.pt")
    dist.barrier()
    out["committed"] = ck.committed_steps()
    out["latest"] = ck.latest_step()
    out["loaded"] = ck.load(6)
    with pytest.raises(FileNotFoundError):
        ck.load(7)
    ck.save(8, _state(8, env.rank))
    ck.wait()
    out["dirs_after_next_save"] = sorted(p.name for p in (root / "async").iterdir())

    sync = Checkpointer(root / "sync", mesh, keep_last=1, async_save=False)
    sync.save(5, _state(5, env.rank))
    out["sync_latest"] = sync.latest_step()
    out["sync_loaded"] = sync.load(5)

    other_layout = Checkpointer(root / "async", build_mesh(MeshConfig(replicate=2, shard=1, tensor=1), env))
    with pytest.raises(ValueError, match="mesh"):
        other_layout.load(6)
    return out


def _load_with_world_one(env: DistEnv, root: Path) -> str:
    ck = Checkpointer(root / "async", build_mesh(MeshConfig(), env))
    with pytest.raises(ValueError, match="world_size") as e:
        ck.load(6)
    return str(e.value)


@functools.cache
def _results(root: Path) -> list[dict]:
    return run_distributed(_scenario, 2, root)


@pytest.fixture(scope="module")
def results(tmp_path_factory: pytest.TempPathFactory) -> tuple[Path, list[dict]]:
    root = tmp_path_factory.mktemp("ckpt")
    return root, _results(root)


def _assert_state_equal(a: object, b: object) -> None:
    if isinstance(a, torch.Tensor):
        assert torch.equal(a, b)
    elif isinstance(a, dict):
        assert a.keys() == b.keys()
        for k in a:
            _assert_state_equal(a[k], b[k])
    elif isinstance(a, list | tuple):
        assert type(a) is type(b) and len(a) == len(b)
        for x, y in zip(a, b, strict=True):
            _assert_state_equal(x, y)
    else:
        assert a == b


def test_round_trip_restores_each_ranks_own_state(results) -> None:
    _, ranks = results
    for rank, r in enumerate(ranks):
        _assert_state_equal(r["loaded"], _state(6, rank))
        _assert_state_equal(r["sync_loaded"], _state(5, rank))
        assert r["sync_latest"] == 5


def test_snapshot_ignores_mutation_after_save(tmp_path: Path) -> None:
    out = run_distributed(_mutate_after_save, 2, tmp_path)
    assert all(torch.equal(t, torch.zeros(3)) for t in out)


def _mutate_after_save(env: DistEnv, root: Path) -> torch.Tensor:
    ck = Checkpointer(root, build_mesh(MeshConfig(), env), async_save=True)
    live = torch.zeros(3)
    ck.save(1, {"w": live})
    live.add_(7.0)
    ck.wait()
    return ck.load(1)["w"]


def test_shard_write_failure_on_one_rank_fails_every_rank_without_committing(tmp_path: Path) -> None:
    out = run_distributed(_fail_on_rank_one, 2, tmp_path)
    for r in out:
        assert r["elapsed"] < 30.0  # a bare barrier would wait for the 30-minute process-group timeout
        assert r["latest_after_failure"] == 1  # step 2 never got metadata.json
        assert r["latest_after_retry"] == 3
    assert "unpicklable" in out[1]["error"] and "other rank(s) failed" in out[0]["error"]


class _Unpicklable:
    def __reduce__(self):
        raise TypeError("unpicklable")


def _fail_on_rank_one(env: DistEnv, root: Path) -> dict:
    ck = Checkpointer(root, build_mesh(MeshConfig(replicate=1, shard=2, tensor=1), env), async_save=True)
    ck.save(1, _state(1, env.rank))
    ck.wait()
    start = time.monotonic()
    ck.save(2, {**_state(2, env.rank), "bad": _Unpicklable() if env.rank == 1 else None})
    with pytest.raises(RuntimeError) as e:
        ck.wait()
    out = {"elapsed": time.monotonic() - start, "error": f"{e.value} / {e.value.__cause__}"}
    out["latest_after_failure"] = ck.latest_step()
    ck.save(3, _state(3, env.rank))  # the group is still in step: the next save commits normally
    ck.wait()
    out["latest_after_retry"] = ck.latest_step()
    return out


def test_retention_keeps_last_and_permanent_and_ignores_incomplete(results) -> None:
    _, ranks = results
    for r in ranks:
        assert r["committed"] == [3, 5, 6]  # keep_last=2 -> {5, 6}; permanent_every=3 -> {3}
        assert r["latest"] == 6  # step 7 has no metadata.json
        assert r["dirs_after_next_save"] == ["step_000003", "step_000006", "step_000008"]


def test_load_rejects_other_world_size(results) -> None:
    root, _ = results
    [message] = run_distributed(_load_with_world_one, 1, root)
    assert "world_size=2" in message
