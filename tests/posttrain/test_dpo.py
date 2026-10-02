import copy
import json
import math
from functools import partial
from pathlib import Path

import pytest

from scratch_cs336.distributed import DistEnv, MeshConfig, build_mesh
from scratch_cs336.distributed.spawn import run_distributed
from scratch_cs336.models.presets import PRESETS
from scratch_cs336.models.transformer import TransformerLM
from scratch_cs336.parallel import Backend, ParallelConfig, Strategy, parallelize
from scratch_cs336.posttrain.data import PreferenceSource
from scratch_cs336.posttrain.dpo import ReferenceScored, dpo_batch_loss
from scratch_cs336.train.config import TrainerConfig
from scratch_cs336.train.optim import OptimizerConfig, ScheduleConfig, build_parallel_optimizer
from scratch_cs336.train.trainer import Trainer
from tests.posttrain.char_tokenizer import CharTokenizer

ROWS = [{"prompt": f"pick {i}", "chosen": f"yes {i}", "rejected": f"no way {i}"} for i in range(8)]
CFG = TrainerConfig(
    num_steps=5, global_batch_size=8, micro_batch_size=2, log_every=1, async_checkpoint=False,
    optimizer=OptimizerConfig(lr=3e-3, weight_decay=0.0), schedule=ScheduleConfig(name="constant"),
)  # fmt: skip


def _fit(env: DistEnv, out: Path) -> None:
    parallel = ParallelConfig(MeshConfig(shard=2), Strategy.FSDP, Backend.SCRATCH)
    mesh = build_mesh(parallel.mesh, env)
    model = TransformerLM(PRESETS["cs336-tiny"])
    model.init_weights(0)
    reference = parallelize(copy.deepcopy(model).requires_grad_(False), mesh, parallel)
    pmodel = parallelize(model, mesh, parallel)
    pairs = PreferenceSource.from_rows(ROWS, CharTokenizer(), 64, CFG.global_batch_size, mesh.dp_rank, mesh.dp_size, 0)
    source = ReferenceScored(pairs, reference, env.device, CFG.micro_batch_size)
    optimizer = build_parallel_optimizer(pmodel, parallel, CFG.optimizer)
    Trainer(CFG, pmodel, optimizer, source, partial(dpo_batch_loss, beta=0.5), mesh, out).fit()


def test_dpo_starts_at_log2_and_separates_chosen_from_rejected(tmp_path: Path):
    run_distributed(_fit, 2, tmp_path)
    records = [r for r in map(json.loads, (tmp_path / "metrics.jsonl").read_text().splitlines()) if "loss" in r]
    first, last = records[0], records[-1]
    # The policy starts equal to the frozen reference: zero implicit rewards, loss log(2).
    assert first["loss"] == pytest.approx(math.log(2), abs=1e-6)
    assert first["reward_margin"] == pytest.approx(0.0, abs=1e-6)
    assert last["reward_margin"] > 1.0 and last["reward_accuracy"] == 1.0
    assert last["loss"] < first["loss"] / 2
