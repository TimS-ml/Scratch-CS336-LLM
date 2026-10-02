"""Full (unsharded) export for evaluation / HF / vLLM; the only path across mesh layouts."""

from __future__ import annotations

import os
from pathlib import Path

import torch.distributed as dist
from safetensors.torch import save_file

from scratch_cs336.parallel.api import ParallelModel

MODEL_FILE = "model.safetensors"


def export_full(pmodel: ParallelModel, path: str | Path) -> Path:
    """Write ``<path>/model.safetensors`` from rank 0 (collective; returns once the file exists).

    Tied parameters are written under every name so the file loads strictly into the unsharded model.
    """
    state = pmodel.full_state_dict()
    out = Path(path) / MODEL_FILE
    if dist.get_rank() == 0:
        out.parent.mkdir(parents=True, exist_ok=True)
        seen: set[int] = set()
        tensors = {}
        for k, v in state.items():
            v = v.contiguous()
            ptr = v.untyped_storage().data_ptr()
            tensors[k] = v.clone() if ptr in seen else v
            seen.add(ptr)
        tmp = out.with_name(out.name + ".tmp")
        save_file(tensors, tmp)
        os.replace(tmp, out)
    dist.barrier()
    return out
