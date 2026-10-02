"""Run a function on N local processes with a real process group.

Used by tests and CPU smoke runs (gloo). Production launches go through ``torchrun`` / Slurm
(see ``scratch_cs336.launch``); this helper only reproduces what torchrun sets up.

``fn`` must be a module-level callable (it is pickled into spawned processes) with signature
``fn(env: DistEnv, *args) -> T``. Per-rank return values are collected and returned in rank order.
"""

from __future__ import annotations

import os
import tempfile
from collections.abc import Callable
from pathlib import Path
from typing import Any

import torch
import torch.multiprocessing as mp

from scratch_cs336.distributed.env import destroy_distributed, find_free_port, init_distributed


def _worker(
    rank: int, fn: Callable[..., Any], world_size: int, port: int, backend: str, out_dir: str, args: tuple
) -> None:
    os.environ.update(
        MASTER_ADDR="127.0.0.1",
        MASTER_PORT=str(port),
        RANK=str(rank),
        WORLD_SIZE=str(world_size),
        LOCAL_RANK=str(rank),
        LOCAL_WORLD_SIZE=str(world_size),
    )
    torch.set_num_threads(1)
    env = init_distributed(backend)
    try:
        result = fn(env, *args)
        torch.save(result, Path(out_dir) / f"rank{rank}.pt")
    finally:
        destroy_distributed()


def run_distributed(fn: Callable[..., Any], world_size: int, *args: Any, backend: str = "gloo") -> list[Any]:
    port = find_free_port()
    with tempfile.TemporaryDirectory() as out_dir:
        mp.start_processes(
            _worker,
            args=(fn, world_size, port, backend, out_dir, args),
            nprocs=world_size,
            join=True,
            start_method="spawn",
        )
        return [torch.load(Path(out_dir) / f"rank{r}.pt", weights_only=False) for r in range(world_size)]
