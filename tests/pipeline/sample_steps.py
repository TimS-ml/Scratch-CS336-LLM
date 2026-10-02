from __future__ import annotations

from collections.abc import Callable
from dataclasses import dataclass
from enum import StrEnum
from pathlib import Path

from scratch_cs336.pipeline import InputPath, Step


class Mode(StrEnum):
    FAST = "fast"
    SLOW = "slow"


@dataclass(frozen=True)
class Inner:
    lr: float = 1e-3
    mode: Mode = Mode.FAST


@dataclass(frozen=True)
class Cfg:
    inner: Inner = Inner()
    upstream: tuple[InputPath, ...] = ()
    extra: dict[str, int | str | Path] | None = None


def noop(cfg: Cfg, out: Path) -> None:
    (out / "done.txt").write_text("ok")


def diamond(
    lr: float = 1e-3,
    run: Callable[[Cfg, Path], None] = noop,
    run_for: dict[str, Callable[[Cfg, Path], None]] | None = None,
) -> list[Step]:
    """a -> (b, c) -> d; returns the sink only (dependencies are discovered through InputPath)."""
    run_for = run_for or {}

    def make(name: str, config: Cfg) -> Step:
        return Step(name, config, run=run_for.get(name, run))

    a = make("a", Cfg(inner=Inner(lr=lr)))
    b = make("b", Cfg(upstream=(InputPath(a, "x"),)))
    c = make("c", Cfg(upstream=(InputPath(a),)))
    d = make("d", Cfg(upstream=(InputPath(b), InputPath(c, "y"))))
    return [d]


if __name__ == "__main__":
    print(",".join(f"{s.name}:{s.hash_id}" for s in diamond()[0].deps), diamond()[0].hash_id)
