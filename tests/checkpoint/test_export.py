from __future__ import annotations

from pathlib import Path

import pytest
import torch
from safetensors.torch import load_file
from torch import nn

from scratch_cs336.checkpoint import export_full
from scratch_cs336.distributed import DistEnv, MeshConfig, build_mesh
from scratch_cs336.distributed.spawn import run_distributed
from scratch_cs336.parallel import Backend, ParallelConfig, Strategy, parallelize


class TiedLM(nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self.embed = nn.Embedding(11, 6)
        self.blocks = nn.ModuleList(nn.Linear(6, 6) for _ in range(2))
        self.lm_head = nn.Linear(6, 11, bias=False)
        self.lm_head.weight = self.embed.weight

    def forward(self, ids: torch.Tensor) -> torch.Tensor:
        x = self.embed(ids)
        for block in self.blocks:
            x = x + block(x)
        return self.lm_head(x)


def _model() -> TiedLM:
    torch.manual_seed(0)
    return TiedLM()


def _export(env: DistEnv, backend: Backend, root: Path) -> bool:
    mesh_cfg = MeshConfig(replicate=1, shard=2, tensor=1)
    pmodel = parallelize(_model(), build_mesh(mesh_cfg, env), ParallelConfig(mesh_cfg, Strategy.FSDP, backend))
    path = export_full(pmodel, root)
    return path.is_file()  # visible on every rank once export_full returns


@pytest.mark.parametrize("backend", list(Backend))
def test_export_loads_strictly_into_unsharded_model(backend: Backend, tmp_path: Path) -> None:
    assert all(run_distributed(_export, 2, backend, tmp_path))
    exported = load_file(tmp_path / "model.safetensors")
    fresh = TiedLM()
    fresh.load_state_dict(exported, strict=True)
    for name, ref in _model().state_dict().items():
        assert torch.equal(fresh.state_dict()[name], ref), name
