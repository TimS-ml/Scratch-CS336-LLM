from __future__ import annotations

from dataclasses import dataclass


@dataclass(frozen=True)
class Resources:
    nnodes: int = 1
    nproc_per_node: int = 1
    gpus_per_node: int = 0
    cpus_per_task: int | None = None
    mem: str | None = None  # Slurm syntax, e.g. "64G"
    time: str = "24:00:00"

    def __post_init__(self) -> None:
        if self.nnodes < 1 or self.nproc_per_node < 1:
            raise ValueError(f"nnodes and nproc_per_node must be >= 1, got {self}")
