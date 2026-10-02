"""Pretraining entry point: ``torchrun ... -m scratch_cs336.train.pretrain --config_path <yaml> [--a.b value ...]``."""

from __future__ import annotations

import dataclasses
from dataclasses import dataclass, field
from functools import partial
from pathlib import Path
from typing import Any

import draccus
import torch
import torch.nn.functional as F

from scratch_cs336.data.cache import TokenCache
from scratch_cs336.data.loader import PretrainLoader, WindowSource
from scratch_cs336.data.mixture import MixtureSource
from scratch_cs336.distributed import DistEnv, MeshConfig, build_mesh, destroy_distributed, init_distributed
from scratch_cs336.eval.perplexity import evaluate_loss
from scratch_cs336.models import PRESETS, ModelConfig, TransformerLM
from scratch_cs336.models.hf import load_hf_config, load_hf_pretrained
from scratch_cs336.parallel import Backend, ParallelConfig, ParallelModel, Strategy, parallelize
from scratch_cs336.tokenizer import check_tokenizer_vocab, load_tokenizer
from scratch_cs336.train.config import TrainerConfig
from scratch_cs336.train.optim import build_parallel_optimizer
from scratch_cs336.train.trainer import Batch, LossOutput, Trainer

IGNORE_INDEX = -100


@dataclass(frozen=True)
class MixtureComponent:
    cache: str = ""
    weight: float = 1.0


@dataclass(frozen=True)
class PretrainConfig:
    output_dir: str = ""
    model: str = "cs336-tiny"  # PRESETS key
    vocab_size: int | None = None  # overrides the preset's vocabulary, e.g. to match a freshly trained BPE
    init_from_hf: str | None = None  # HF repo or local snapshot; replaces `model` (config and weights)
    tokenizer: str | None = None  # load_tokenizer spec; checked by check_tokenizer_vocab (exact match for HF init)
    train_cache: str = ""
    mixture: tuple[MixtureComponent, ...] = ()  # weighted token caches; replaces train_cache when non-empty
    val_cache: str | None = None
    seq_len: int = 128
    parallel: ParallelConfig = field(
        default_factory=lambda: ParallelConfig(MeshConfig(), Strategy.FSDP, Backend.SCRATCH)
    )
    trainer: TrainerConfig = field(
        default_factory=lambda: TrainerConfig(num_steps=1000, global_batch_size=32, micro_batch_size=8)
    )
    eval_batches: int = 8  # validation batches of trainer.global_batch_size windows per evaluation


class PretrainBatches:
    """``BatchSource`` view of a :class:`PretrainLoader`: ``{"input_ids", "labels"}`` for this rank's rows.

    The loader is a pure function of the step, so the state only pins the data identity: resuming on a different
    cache, seed or batch size is an error rather than a silent change of the token stream.
    """

    def __init__(self, loader: PretrainLoader) -> None:
        self.loader = loader

    def batch(self, step: int) -> Batch:
        input_ids, labels = self.loader.batch(step)
        return {"input_ids": input_ids, "labels": labels}

    def state_dict(self) -> dict[str, Any]:
        loader = self.loader
        return {"seed": loader.seed, "num_windows": loader.num_windows, "global_batch_size": loader.global_batch_size}

    def load_state_dict(self, state: dict[str, Any]) -> None:
        if state != self.state_dict():
            raise ValueError(f"checkpoint was trained on data {state}, this run reads {self.state_dict()}")


def lm_loss(pmodel: ParallelModel, batch: Batch) -> LossOutput:
    """Token-sum next-token cross entropy; labels equal to ``IGNORE_INDEX`` do not count."""
    logits = pmodel(batch["input_ids"])
    labels = batch["labels"]
    loss_sum = F.cross_entropy(
        logits.float().flatten(0, 1), labels.flatten(), ignore_index=IGNORE_INDEX, reduction="sum"
    )
    return LossOutput(loss_sum, (labels != IGNORE_INDEX).sum())


def model_config(cfg: PretrainConfig) -> ModelConfig:
    """The model's :class:`ModelConfig` without loading (or downloading) weights; validate it before ``build_model``."""
    if cfg.init_from_hf is not None:
        return load_hf_config(cfg.init_from_hf)
    if cfg.model not in PRESETS:
        raise ValueError(f"unknown model preset {cfg.model!r}; choose from {sorted(PRESETS)}")
    model_cfg = PRESETS[cfg.model]
    if cfg.vocab_size is not None:
        model_cfg = dataclasses.replace(model_cfg, vocab_size=cfg.vocab_size)
    return model_cfg


def build_model(cfg: PretrainConfig) -> TransformerLM:
    """Full fp32 model on CPU, from HF weights or a preset with deterministic init."""
    if cfg.init_from_hf is not None:
        model_cfg, state = load_hf_pretrained(cfg.init_from_hf, dtype=torch.float32)
        model = TransformerLM(model_cfg)
        model.load_state_dict(state)
    else:
        model = TransformerLM(model_config(cfg))
        model.init_weights(cfg.trainer.seed)
    if cfg.seq_len > model.cfg.max_seq_len:
        raise ValueError(f"seq_len={cfg.seq_len} exceeds the model's max_seq_len={model.cfg.max_seq_len}")
    if cfg.tokenizer is not None:
        check_tokenizer_vocab(cfg.tokenizer, load_tokenizer(cfg.tokenizer), model.cfg.vocab_size, cfg.init_from_hf)
    return model


def train_source(cfg: PretrainConfig) -> WindowSource:
    if not cfg.mixture:
        return TokenCache.open(cfg.train_cache)
    caches = {c.cache: TokenCache.open(c.cache) for c in cfg.mixture}
    return MixtureSource(caches, {c.cache: c.weight for c in cfg.mixture}, seed=cfg.trainer.seed)


def run(cfg: PretrainConfig, env: DistEnv) -> None:
    if not cfg.output_dir:
        raise ValueError("output_dir is required")
    if bool(cfg.train_cache) == bool(cfg.mixture):
        raise ValueError("set exactly one of train_cache / mixture")
    mesh = build_mesh(cfg.parallel.mesh, env)
    model_config(cfg).check_tensor_parallel(mesh.tp_size)
    model = build_model(cfg)
    flops_per_token = model.flops_per_token(cfg.seq_len)
    pmodel = parallelize(model.to(env.device), mesh, cfg.parallel)
    optimizer = build_parallel_optimizer(pmodel, cfg.parallel, cfg.trainer.optimizer)
    loader = PretrainLoader(
        train_source(cfg), cfg.seq_len, cfg.trainer.global_batch_size, mesh.dp_rank, mesh.dp_size, cfg.trainer.seed
    )
    evaluators = {}
    if cfg.val_cache is not None:
        evaluators["val"] = partial(
            evaluate_loss,
            source=TokenCache.open(cfg.val_cache),
            seq_len=cfg.seq_len,
            batch_size=cfg.trainer.global_batch_size,
            mesh=mesh,
            max_batches=cfg.eval_batches,
        )
    trainer = Trainer(
        cfg.trainer,
        pmodel,
        optimizer,
        PretrainBatches(loader),
        lm_loss,
        mesh,
        Path(cfg.output_dir),
        evaluators=evaluators,
        flops_per_token=flops_per_token,
    )
    trainer.fit()


def main() -> None:
    cfg = draccus.parse(config_class=PretrainConfig)
    env = init_distributed()
    run(cfg, env)
    destroy_distributed()


if __name__ == "__main__":
    main()
