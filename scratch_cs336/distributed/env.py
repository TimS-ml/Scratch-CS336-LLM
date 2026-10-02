"""Process-group bootstrap.

Every entry point calls :func:`init_distributed` exactly once. A plain ``python -m ...`` run becomes a
world of size 1 with a real process group, so single-process and multi-process runs share one code path.

Rank discovery order:
1. torchrun / torchelastic (``RANK``, ``WORLD_SIZE``, ``LOCAL_RANK``, ``LOCAL_WORLD_SIZE``).
2. Slurm ``srun`` without torchrun (``SLURM_PROCID``, ``SLURM_NTASKS``, ``SLURM_LOCALID``); the
   launcher must export ``MASTER_ADDR``/``MASTER_PORT``.
3. Nothing set: world of one on localhost with a free port.
"""

from __future__ import annotations

import os
import socket
from dataclasses import dataclass
from datetime import timedelta

import torch
import torch.distributed as dist


@dataclass(frozen=True)
class DistEnv:
    rank: int
    world_size: int
    local_rank: int
    local_world_size: int
    device: torch.device
    backend: str

    @property
    def is_main(self) -> bool:
        return self.rank == 0

    @property
    def num_nodes(self) -> int:
        return self.world_size // self.local_world_size


def find_free_port() -> int:
    with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as s:
        s.bind(("127.0.0.1", 0))
        return s.getsockname()[1]


def _discover() -> tuple[int, int, int, int]:
    env = os.environ
    if "RANK" in env and "WORLD_SIZE" in env:
        rank, world = int(env["RANK"]), int(env["WORLD_SIZE"])
        local_rank = int(env.get("LOCAL_RANK", rank))
        local_world = int(env.get("LOCAL_WORLD_SIZE", world))
        return rank, world, local_rank, local_world
    if "SLURM_PROCID" in env and "SLURM_NTASKS" in env:
        rank, world = int(env["SLURM_PROCID"]), int(env["SLURM_NTASKS"])
        local_rank = int(env.get("SLURM_LOCALID", 0))
        local_world = int(env.get("SLURM_NTASKS_PER_NODE", str(world)).split("(")[0])
        if "MASTER_ADDR" not in env or "MASTER_PORT" not in env:
            raise RuntimeError("Slurm launch without torchrun requires MASTER_ADDR and MASTER_PORT")
        env["RANK"], env["WORLD_SIZE"], env["LOCAL_RANK"] = str(rank), str(world), str(local_rank)
        return rank, world, local_rank, local_world
    env.setdefault("MASTER_ADDR", "127.0.0.1")
    env.setdefault("MASTER_PORT", str(find_free_port()))
    env["RANK"], env["WORLD_SIZE"], env["LOCAL_RANK"], env["LOCAL_WORLD_SIZE"] = "0", "1", "0", "1"
    return 0, 1, 0, 1


def init_distributed(backend: str | None = None, timeout_minutes: int = 30) -> DistEnv:
    """Initialize the default process group (idempotent) and pin this process to its device.

    ``backend`` defaults to ``nccl`` when CUDA is available, otherwise ``gloo`` on CPU.
    """
    rank, world, local_rank, local_world = _discover()
    if backend is None:
        backend = "nccl" if torch.cuda.is_available() else "gloo"
    if backend == "nccl":
        torch.cuda.set_device(local_rank)
        device = torch.device("cuda", local_rank)
    else:
        device = torch.device("cpu")
    if not dist.is_initialized():
        dist.init_process_group(
            backend=backend,
            rank=rank,
            world_size=world,
            timeout=timedelta(minutes=timeout_minutes),
            device_id=device if backend == "nccl" else None,
        )
    return DistEnv(rank, world, local_rank, local_world, device, backend)


def destroy_distributed() -> None:
    if dist.is_initialized():
        dist.barrier()
        dist.destroy_process_group()
