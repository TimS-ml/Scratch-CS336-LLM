"""Configuration of the generic training loop."""

from __future__ import annotations

from dataclasses import dataclass, field

from scratch_cs336.tracking import TrackerConfig
from scratch_cs336.train.optim import OptimizerConfig, ScheduleConfig


@dataclass(frozen=True)
class TrainerConfig:
    num_steps: int
    global_batch_size: int  # rows per optimizer step, summed over the dp group
    micro_batch_size: int  # rows per forward/backward on one rank
    optimizer: OptimizerConfig = field(default_factory=OptimizerConfig)
    schedule: ScheduleConfig = field(default_factory=ScheduleConfig)
    max_grad_norm: float | None = 1.0
    log_every: int = 10
    eval_every: int | None = None
    checkpoint_every: int | None = None
    keep_last: int = 2
    permanent_every: int | None = None
    seed: int = 0
    tracker: TrackerConfig = field(default_factory=TrackerConfig)
    async_checkpoint: bool = True

    def __post_init__(self) -> None:
        if min(self.num_steps, self.global_batch_size, self.micro_batch_size, self.log_every) < 1:
            raise ValueError(f"num_steps, batch sizes and log_every must be >= 1: {self}")
        for name in ("eval_every", "checkpoint_every", "permanent_every"):
            value = getattr(self, name)
            if value is not None and value < 1:
                raise ValueError(f"{name} must be >= 1 or None, got {value}")

    def grad_accum(self, dp_size: int) -> int:
        """Micro-batches per optimizer step on each rank."""
        rows = self.micro_batch_size * dp_size
        if self.global_batch_size % rows:
            raise ValueError(
                f"global_batch_size={self.global_batch_size} is not divisible by "
                f"micro_batch_size * dp_size = {self.micro_batch_size} * {dp_size}"
            )
        return self.global_batch_size // rows
