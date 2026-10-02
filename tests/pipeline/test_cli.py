from dataclasses import dataclass
from pathlib import Path

import pytest

from scratch_cs336.pipeline import experiment_main
from tests.pipeline.sample_steps import Cfg, Mode, diamond


@dataclass(frozen=True)
class Opts:
    mode: Mode = Mode.FAST
    lr: float = 1e-3
    max_bytes: int | None = None
    smoke: bool = False


def test_plan_only_by_default_then_run_then_force(tmp_path, capsys):
    log: list[str] = []

    def record(name: str):
        def run(cfg: Cfg, out: Path) -> None:
            log.append(name)

        return run

    def build(opts: Opts):
        return diamond(run_for={n: record(n) for n in "abcd"})

    root = ["--root", str(tmp_path)]
    experiment_main(build, Opts, root)
    assert log == [] and capsys.readouterr().out.count("[run") == 4

    experiment_main(build, Opts, [*root, "--run"])
    assert sorted(log) == ["a", "b", "c", "d"]

    log.clear()
    experiment_main(build, Opts, [*root, "--run", "--force", "a", "--force", "c"])
    assert sorted(log) == ["a", "c"]

    with pytest.raises(ValueError, match="zzz"):
        experiment_main(build, Opts, [*root, "--force", "zzz"])


def test_options_dataclass_becomes_flags_passed_to_build(tmp_path):
    seen: list[Opts] = []

    def build(opts: Opts):
        seen.append(opts)
        return diamond(lr=opts.lr)

    experiment_main(build, Opts, ["--root", str(tmp_path)])
    experiment_main(
        build, Opts, ["--root", str(tmp_path), "--mode", "slow", "--lr", "0.5", "--max-bytes", "7", "--smoke"]
    )
    assert seen == [Opts(), Opts(Mode.SLOW, 0.5, 7, True)]
