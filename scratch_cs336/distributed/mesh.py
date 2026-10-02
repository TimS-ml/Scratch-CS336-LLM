"""Declarative device mesh.

The mesh has three named dims, outermost to innermost::

    ("dp_replicate", "dp_shard", "tp")

* ``tp`` is innermost so tensor-parallel peers share the fastest links (same node, NVLink/PCIe).
* ``dp_shard`` holds FSDP / ZeRO shards of parameters, grads and optimizer state.
* ``dp_replicate`` is outermost: pure replicas (DDP, or the "H" in HSDP across nodes).

The data-parallel group used by the loader and loss averaging is the flattened ``dp`` dim
(``dp_replicate x dp_shard``). This mirrors Marin/Levanter's ``replica`` / ``data`` / ``model`` axes,
with one dim allowed to be ``-1`` to absorb the remaining devices.
"""

from __future__ import annotations

from dataclasses import dataclass
from functools import cached_property

import torch.distributed as dist
from torch.distributed.device_mesh import DeviceMesh, init_device_mesh

from scratch_cs336.distributed.env import DistEnv

MESH_DIMS = ("dp_replicate", "dp_shard", "tp")


@dataclass(frozen=True)
class MeshConfig:
    replicate: int = 1
    shard: int = -1
    tensor: int = 1

    def resolve(self, world_size: int) -> tuple[int, int, int]:
        sizes = [self.replicate, self.shard, self.tensor]
        unknown = [i for i, s in enumerate(sizes) if s == -1]
        if len(unknown) > 1:
            raise ValueError(f"at most one mesh dim may be -1, got {self}")
        known = 1
        for s in sizes:
            if s != -1:
                if s < 1:
                    raise ValueError(f"mesh dims must be positive or -1, got {self}")
                known *= s
        if unknown:
            if world_size % known:
                raise ValueError(f"world size {world_size} not divisible by fixed mesh dims {self}")
            sizes[unknown[0]] = world_size // known
        if sizes[0] * sizes[1] * sizes[2] != world_size:
            raise ValueError(f"mesh {sizes} does not cover world size {world_size}")
        return sizes[0], sizes[1], sizes[2]


@dataclass(frozen=True)
class Mesh:
    """Resolved mesh plus the process groups the rest of the code needs."""

    device_mesh: DeviceMesh
    env: DistEnv

    @property
    def replicate_size(self) -> int:
        return self.device_mesh["dp_replicate"].size()

    @property
    def shard_size(self) -> int:
        return self.device_mesh["dp_shard"].size()

    @property
    def tp_size(self) -> int:
        return self.device_mesh["tp"].size()

    @property
    def dp_size(self) -> int:
        return self.replicate_size * self.shard_size

    @cached_property
    def dp_mesh(self) -> DeviceMesh:
        """Flattened ``dp_replicate x dp_shard`` 1-D mesh."""
        return self.device_mesh["dp_replicate", "dp_shard"]._flatten("dp")

    @property
    def dp_group(self) -> dist.ProcessGroup:
        return self.dp_mesh.get_group()

    @property
    def dp_rank(self) -> int:
        return self.dp_mesh.get_local_rank()

    @property
    def shard_group(self) -> dist.ProcessGroup:
        return self.device_mesh.get_group("dp_shard")

    @property
    def replicate_group(self) -> dist.ProcessGroup:
        return self.device_mesh.get_group("dp_replicate")

    @property
    def tp_group(self) -> dist.ProcessGroup:
        return self.device_mesh.get_group("tp")

    @property
    def tp_rank(self) -> int:
        return self.device_mesh.get_local_rank("tp")


def build_mesh(cfg: MeshConfig, env: DistEnv) -> Mesh:
    shape = cfg.resolve(env.world_size)
    device_mesh = init_device_mesh(env.device.type, shape, mesh_dim_names=MESH_DIMS)
    mesh = Mesh(device_mesh, env)
    _ = mesh.dp_mesh  # flatten collectively on every rank, up front
    return mesh
