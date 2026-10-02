"""TinyStories, CS336 hw1 setup: TinyStoriesV2-GPT4 -> BPE (10k) -> token caches -> cs336-17m (ctx 256) -> eval.

python -m experiments.tinystories                                    # print the plan (cpu smoke)
python -m experiments.tinystories --device cpu --run --root /tmp/exp
python -m experiments.tinystories --device 4090x4 --run --root runs
"""

from __future__ import annotations

from dataclasses import dataclass

from scratch_cs336.data.tokenize import DOCUMENT_SEPARATOR
from scratch_cs336.distributed import MeshConfig
from scratch_cs336.launch import Resources
from scratch_cs336.parallel import Backend, ParallelConfig, Strategy
from scratch_cs336.pipeline import InputPath, Step, experiment_main
from scratch_cs336.pipeline.builders import evaluate, final_model, hf_download, pretrain, tokenize, train_tokenizer
from scratch_cs336.train import TrainerConfig
from scratch_cs336.train.optim import OptimizerConfig, ScheduleConfig

TRAIN_FILE = "TinyStoriesV2-GPT4-train.txt"
VALID_FILE = "TinyStoriesV2-GPT4-valid.txt"
SEQ_LEN = 256


@dataclass(frozen=True)
class Device:
    resources: Resources
    parallel: ParallelConfig
    global_batch_size: int
    micro_batch_size: int  # rows per forward on one rank
    num_steps: int
    max_bytes: int | None = None  # per downloaded file; None = the full dataset
    eval_batches: int | None = None  # validation batches per evaluation; None = the whole validation set


def _parallel(strategy: Strategy, backend: Backend, compute_dtype: str) -> ParallelConfig:
    return ParallelConfig(MeshConfig(), strategy, backend, compute_dtype)


# hw1 budget on GPUs: 327.68M tokens = 128 x 256 x 10k steps.
DEVICES: dict[str, Device] = {
    "cpu": Device(
        Resources(nproc_per_node=2),
        _parallel(Strategy.FSDP, Backend.SCRATCH, "float32"),
        global_batch_size=8,
        micro_batch_size=2,
        num_steps=100,
        max_bytes=10_000_000,
        eval_batches=4,
    ),
    "4090x1": Device(
        Resources(nproc_per_node=1, gpus_per_node=1),
        _parallel(Strategy.DDP, Backend.SCRATCH, "bfloat16"),
        global_batch_size=128,
        micro_batch_size=64,
        num_steps=10_000,
    ),
    "4090x4": Device(
        Resources(nproc_per_node=4, gpus_per_node=4),
        _parallel(Strategy.FSDP, Backend.SCRATCH, "bfloat16"),
        global_batch_size=128,
        micro_batch_size=32,
        num_steps=10_000,
    ),
    "h100x8": Device(
        Resources(nproc_per_node=8, gpus_per_node=8),
        _parallel(Strategy.FSDP, Backend.NATIVE, "bfloat16"),
        global_batch_size=256,
        micro_batch_size=32,
        num_steps=5_000,
    ),
}


@dataclass(frozen=True)
class Options:
    device: str = "cpu"
    lr: float = 3e-3
    steps: int | None = None  # overrides the device's step budget


def build(opts: Options) -> list[Step]:
    if opts.device not in DEVICES:
        raise SystemExit(f"--device must be one of {sorted(DEVICES)}, got {opts.device!r}")
    device = DEVICES[opts.device]
    steps = opts.steps or device.num_steps

    raw = hf_download("tinystories-raw", "roneneldan/TinyStories", (TRAIN_FILE, VALID_FILE), max_bytes=device.max_bytes)
    bpe = train_tokenizer("tinystories-bpe-10k", InputPath(raw, TRAIN_FILE), 10_000, (DOCUMENT_SEPARATOR,))
    train_cache = tokenize("tinystories-train-tokens", InputPath(raw, TRAIN_FILE), InputPath(bpe))
    valid_cache = tokenize("tinystories-valid-tokens", InputPath(raw, VALID_FILE), InputPath(bpe))

    trainer = TrainerConfig(
        num_steps=steps,
        global_batch_size=device.global_batch_size,
        micro_batch_size=device.micro_batch_size,
        optimizer=OptimizerConfig(lr=opts.lr, betas=(0.9, 0.95), weight_decay=0.1),
        schedule=ScheduleConfig(warmup_steps=max(1, steps // 50), total_steps=steps, min_lr_ratio=0.1),
        max_grad_norm=1.0,
        log_every=max(1, min(100, steps // 20)),
        eval_every=max(1, steps // 5),
        checkpoint_every=max(1, steps // 5),
    )
    lm = pretrain(
        "tinystories-cs336-17m",
        model="cs336-17m",
        train=InputPath(train_cache),
        val=InputPath(valid_cache),
        tokenizer=InputPath(bpe),
        seq_len=SEQ_LEN,
        parallel=device.parallel,
        trainer=trainer,
        resources=device.resources,
        eval_batches=device.eval_batches or 50,
    )
    final_eval = evaluate(
        "tinystories-cs336-17m-eval",
        final_model(lm),
        InputPath(valid_cache),
        seq_len=SEQ_LEN,
        batch_size=device.global_batch_size,
        max_batches=device.eval_batches,
        parallel=device.parallel,
        resources=device.resources,
    )
    return [final_eval]


if __name__ == "__main__":
    experiment_main(build, options=Options)
