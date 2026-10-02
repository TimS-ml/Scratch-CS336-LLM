"""Held-out loss and perplexity over a window source; ``python -m scratch_cs336.eval.perplexity`` evaluates an
exported ``final/`` directory (``model.safetensors`` + ``config.json``) on a token cache."""

from __future__ import annotations

import json
import math
from dataclasses import dataclass, field
from pathlib import Path

import draccus
import numpy as np
import torch
import torch.distributed as dist
import torch.nn.functional as F
from safetensors.torch import load_file

from scratch_cs336.checkpoint.export import MODEL_FILE
from scratch_cs336.data.cache import TokenCache
from scratch_cs336.data.loader import WindowSource
from scratch_cs336.distributed import Mesh, MeshConfig, build_mesh, destroy_distributed, init_distributed
from scratch_cs336.models import ModelConfig, TransformerLM
from scratch_cs336.parallel import Backend, ParallelConfig, ParallelModel, Strategy, parallelize
from scratch_cs336.train.trainer import MODEL_CONFIG_FILE

RESULT_FILE = "eval.json"


@torch.no_grad()
def evaluate_loss(
    pmodel: ParallelModel,
    source: WindowSource,
    seq_len: int,
    batch_size: int,
    mesh: Mesh,
    max_batches: int | None = None,
) -> dict[str, float]:
    """Mean next-token loss over the first ``min(num_windows, max_batches * batch_size)`` windows of ``source``.

    ``batch_size`` is global: dp rank ``r`` evaluates the contiguous windows ``[r N / dp, (r + 1) N / dp)`` in
    chunks of ``ceil(batch_size / dp)``. Collective over the whole mesh.
    """
    available = source.num_windows(seq_len)
    n = available if max_batches is None else min(available, max_batches * batch_size)
    if n < 1:
        raise ValueError(f"source has no windows of seq_len={seq_len} to evaluate")
    dp, rank = mesh.dp_size, mesh.dp_rank
    lo, hi = rank * n // dp, (rank + 1) * n // dp
    chunk = math.ceil(batch_size / dp)
    # Same number of forwards on every rank (FSDP forwards are collective); a rank out of windows runs a dummy one.
    forwards = math.ceil(math.ceil(n / dp) / chunk)
    device = mesh.env.device
    sums = torch.zeros(2, dtype=torch.float64, device=device)  # loss sum, tokens
    was_training = pmodel.module.training
    pmodel.module.eval()
    try:
        for f in range(forwards):
            start = lo + f * chunk
            stop = min(hi, start + chunk)
            indices = range(start, stop) if stop > start else range(1)
            rows = np.stack([source.window(i, seq_len) for i in indices]).astype(np.int64)
            tokens = torch.from_numpy(rows).to(device)
            logits = pmodel(tokens[:, :-1])
            if stop > start:
                targets = tokens[:, 1:]
                sums[0] += F.cross_entropy(logits.float().flatten(0, 1), targets.flatten(), reduction="sum").double()
                sums[1] += targets.numel()
    finally:
        pmodel.module.train(was_training)
    dist.all_reduce(sums, group=mesh.dp_group)
    loss = (sums[0] / sums[1]).item()
    return {"loss": loss, "perplexity": math.exp(loss), "tokens": sums[1].item()}


def exported_config(path: str | Path) -> ModelConfig:
    """The :class:`ModelConfig` of a ``final/`` directory written by the trainer, without loading weights."""
    return ModelConfig(**json.loads((Path(path) / MODEL_CONFIG_FILE).read_text()))


def load_exported(path: str | Path) -> TransformerLM:
    """Full-precision model from a ``final/`` directory written by the trainer (CPU, unsharded)."""
    model = TransformerLM(exported_config(path))
    model.load_state_dict(load_file(Path(path) / MODEL_FILE))
    return model


@dataclass(frozen=True)
class PerplexityConfig:
    model_dir: str = ""  # exported final/ directory
    cache: str = ""  # token cache directory
    seq_len: int = 128
    batch_size: int = 16
    max_batches: int | None = None
    parallel: ParallelConfig = field(
        default_factory=lambda: ParallelConfig(MeshConfig(), Strategy.DDP, Backend.SCRATCH)
    )
    output_dir: str = ""  # writes eval.json here when set


def main() -> None:
    cfg = draccus.parse(config_class=PerplexityConfig)
    env = init_distributed()
    mesh = build_mesh(cfg.parallel.mesh, env)
    exported_config(cfg.model_dir).check_tensor_parallel(mesh.tp_size)
    pmodel = parallelize(load_exported(cfg.model_dir).to(env.device), mesh, cfg.parallel)
    result = evaluate_loss(pmodel, TokenCache.open(cfg.cache), cfg.seq_len, cfg.batch_size, mesh, cfg.max_batches)
    if env.is_main:
        print(json.dumps(result), flush=True)
        if cfg.output_dir:
            Path(cfg.output_dir).mkdir(parents=True, exist_ok=True)
            (Path(cfg.output_dir) / RESULT_FILE).write_text(json.dumps(result, indent=2))
    destroy_distributed()


if __name__ == "__main__":
    main()
