"""Experiment tracking: jsonl + wandb, active on rank 0 only."""

from __future__ import annotations

import json
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Protocol


class Tracker(Protocol):
    def log(self, metrics: dict[str, float], step: int) -> None: ...
    def log_config(self, cfg: dict[str, Any]) -> None: ...
    def finish(self) -> None: ...


class NullTracker:
    def log(self, metrics: dict[str, float], step: int) -> None:
        pass

    def log_config(self, cfg: dict[str, Any]) -> None:
        pass

    def finish(self) -> None:
        pass


class JsonlTracker:
    """Appends one JSON object per line: metrics plus ``step`` and wall-clock ``time``."""

    def __init__(self, path: Path) -> None:
        self.path = Path(path)
        self.path.parent.mkdir(parents=True, exist_ok=True)
        self._f = self.path.open("a")

    def _write(self, record: dict[str, Any]) -> None:
        self._f.write(json.dumps(record, default=str) + "\n")
        self._f.flush()

    def log(self, metrics: dict[str, float], step: int) -> None:
        self._write({**metrics, "step": step, "time": time.time()})

    def log_config(self, cfg: dict[str, Any]) -> None:
        self._write({"config": cfg, "time": time.time()})

    def finish(self) -> None:
        if not self._f.closed:
            self._f.close()


class WandbTracker:
    def __init__(self, project: str, run_name: str | None, config: dict[str, Any] | None = None) -> None:
        try:
            import wandb
        except ImportError as e:
            raise ImportError("WandbTracker requires the `wandb` package (pip install wandb)") from e
        self._run = wandb.init(project=project, name=run_name, config=config)

    def log(self, metrics: dict[str, float], step: int) -> None:
        self._run.log(metrics, step=step)

    def log_config(self, cfg: dict[str, Any]) -> None:
        self._run.config.update(cfg, allow_val_change=True)

    def finish(self) -> None:
        self._run.finish()


class MultiTracker:
    def __init__(self, trackers: list[Tracker]) -> None:
        self.trackers = trackers

    def log(self, metrics: dict[str, float], step: int) -> None:
        for t in self.trackers:
            t.log(metrics, step)

    def log_config(self, cfg: dict[str, Any]) -> None:
        for t in self.trackers:
            t.log_config(cfg)

    def finish(self) -> None:
        for t in self.trackers:
            t.finish()


@dataclass(frozen=True)
class TrackerConfig:
    kinds: tuple[str, ...] = ("jsonl",)
    project: str = "scratch-cs336"
    run_name: str | None = None


def build_tracker(cfg: TrackerConfig, output_dir: Path, is_main: bool) -> Tracker:
    if not is_main:
        return NullTracker()
    trackers: list[Tracker] = []
    for kind in cfg.kinds:
        match kind:
            case "jsonl":
                trackers.append(JsonlTracker(Path(output_dir) / "metrics.jsonl"))
            case "wandb":
                trackers.append(WandbTracker(cfg.project, cfg.run_name))
            case _:
                raise ValueError(f"unknown tracker kind {kind!r}")
    return trackers[0] if len(trackers) == 1 else MultiTracker(trackers)
