import json
from dataclasses import dataclass
from enum import StrEnum
from pathlib import Path

import draccus
import pytest

from scratch_cs336.launch import Resources
from scratch_cs336.pipeline import Step, run_steps
from scratch_cs336.pipeline.runner import LOCK_FILE, STATUS_FILE, Status, StepLockedError
from tests.pipeline.sample_steps import Cfg, diamond, noop


class NoLauncher:
    def run(self, module, config_path, resources: Resources, log_dir) -> None:
        raise AssertionError("run steps must not touch the launcher")


def recorder(log: list[str], name: str):
    def run(cfg: Cfg, out: Path) -> None:
        log.append(name)
        (out / "done.txt").write_text("ok")

    return run


def recording_diamond(log: list[str], **kw) -> list[Step]:
    return diamond(run_for={n: recorder(log, n) for n in "abcd"}, **kw)


def status_of(path: Path) -> dict:
    return json.loads((path / STATUS_FILE).read_text())


def test_diamond_runs_each_step_once_in_dependency_order_then_fully_caches(tmp_path):
    log: list[str] = []
    paths = run_steps(recording_diamond(log), tmp_path, NoLauncher(), dry_run=False)
    assert log.index("a") == 0 and log[-1] == "d" and sorted(log) == ["a", "b", "c", "d"]
    assert set(paths) == {"a", "b", "c", "d"}
    assert all(status_of(p)["status"] == "SUCCESS" for p in paths.values())

    log.clear()
    assert run_steps(recording_diamond(log), tmp_path, NoLauncher(), dry_run=False) == paths
    assert log == []


def test_run_steps_receive_resolved_dependency_paths(tmp_path):
    seen: dict[str, Cfg] = {}

    def spy(cfg: Cfg, out: Path) -> None:
        seen["d"] = cfg

    paths = run_steps(diamond(run_for={"d": spy}), tmp_path, NoLauncher(), dry_run=False)
    assert seen["d"].upstream == (str(paths["b"]), str(paths["c"] / "y"))
    provenance = json.loads((paths["d"] / ".step.json").read_text())
    assert provenance["name"] == "d" and sorted(provenance["deps"]) == sorted(p.name for p in (paths["b"], paths["c"]))


def test_dry_run_prints_plan_and_touches_nothing(tmp_path, capsys):
    log: list[str] = []
    run_steps(recording_diamond(log), tmp_path, NoLauncher())
    assert log == [] and not any(tmp_path.iterdir())
    plan = capsys.readouterr().out
    assert plan.count("[run") == 4 and plan.index(" a ") < plan.index(" d ")

    run_steps(recording_diamond(log), tmp_path, NoLauncher(), dry_run=False)
    capsys.readouterr()
    run_steps(recording_diamond(log), tmp_path, NoLauncher())
    assert capsys.readouterr().out.count("[cached") == 4


def test_changed_hyperparameter_forks_new_paths_and_reuses_nothing_downstream(tmp_path):
    log: list[str] = []
    first = run_steps(recording_diamond(log), tmp_path, NoLauncher(), dry_run=False)
    second = run_steps(recording_diamond(log, lr=5e-4), tmp_path, NoLauncher(), dry_run=False)
    assert all(first[n] != second[n] for n in "abcd")
    assert len(log) == 8


def test_failure_marks_failed_releases_lock_and_rerun_retries(tmp_path):
    log: list[str] = []
    calls = {"b": 0}

    def flaky(cfg: Cfg, out: Path) -> None:
        calls["b"] += 1
        if calls["b"] == 1:
            raise RuntimeError("boom")
        log.append("b")

    steps = diamond(run_for={"a": recorder(log, "a"), "b": flaky, "c": recorder(log, "c"), "d": recorder(log, "d")})
    with pytest.raises(RuntimeError, match="boom"):
        run_steps(steps, tmp_path, NoLauncher(), dry_run=False)
    b_dir = next(tmp_path.glob("b-*"))
    assert status_of(b_dir)["status"] == Status.FAILED and "boom" in status_of(b_dir)["error"]
    assert not (b_dir / LOCK_FILE).exists()
    assert not list(tmp_path.glob("d-*"))  # never reached

    log.clear()
    run_steps(steps, tmp_path, NoLauncher(), dry_run=False)
    assert "a" not in log  # cached from the first attempt
    assert {"b", "d"} <= set(log)
    assert status_of(b_dir)["status"] == Status.SUCCESS


def test_existing_lock_raises_with_owner_and_is_never_stolen(tmp_path):
    (step,) = [Step("s", Cfg(), run=noop)]
    out = step.output_path(tmp_path)
    out.mkdir(parents=True)
    (out / LOCK_FILE).write_text(json.dumps({"host": "other-box", "pid": 4242}))
    with pytest.raises(StepLockedError, match="other-box"):
        run_steps([step], tmp_path, NoLauncher(), dry_run=False)
    assert (out / LOCK_FILE).exists()
    assert not (out / STATUS_FILE).exists()


def test_lock_is_exclusive_while_a_step_runs(tmp_path):
    inner = Step("s", Cfg(), run=noop)
    outcomes: list[str] = []

    def reenter(cfg: Cfg, out: Path) -> None:
        try:
            run_steps([inner], tmp_path, NoLauncher(), dry_run=False)
        except StepLockedError:
            outcomes.append("locked")

    # Same name and config: the nested runner contends with the outer one for the same output dir.
    outer = Step("s", Cfg(), run=reenter)
    run_steps([outer], tmp_path, NoLauncher(), dry_run=False)
    assert outcomes == ["locked"]


def test_force_reruns_only_named_steps(tmp_path):
    log: list[str] = []
    run_steps(recording_diamond(log), tmp_path, NoLauncher(), dry_run=False)
    log.clear()
    run_steps(recording_diamond(log), tmp_path, NoLauncher(), dry_run=False, force={"b"})
    assert log == ["b"]


def test_force_of_unknown_step_is_an_error(tmp_path):
    with pytest.raises(ValueError, match="nope"):
        run_steps(diamond(), tmp_path, NoLauncher(), force={"nope"})


def test_two_different_steps_with_one_name_are_rejected(tmp_path):
    from scratch_cs336.pipeline import InputPath

    one = Step("dup", Cfg(), run=noop)
    two = Step("dup", Cfg(extra={"k": 1}), run=noop)
    top = Step("top", Cfg(upstream=(InputPath(one), InputPath(two))), run=noop)
    with pytest.raises(ValueError, match="dup"):
        run_steps([top], tmp_path, NoLauncher())


def test_keyboard_interrupt_marks_failed(tmp_path):
    def interrupted(cfg: Cfg, out: Path) -> None:
        raise KeyboardInterrupt

    step = Step("s", Cfg(), run=interrupted)
    with pytest.raises(KeyboardInterrupt):
        run_steps([step], tmp_path, NoLauncher(), dry_run=False)
    assert status_of(step.output_path(tmp_path))["status"] == Status.FAILED
    assert not (step.output_path(tmp_path) / LOCK_FILE).exists()


def test_lock_of_dead_local_pid_is_recovered_but_live_pid_still_raises(tmp_path):
    import socket
    import subprocess
    import sys

    step = Step("s", Cfg(), run=noop)
    out = step.output_path(tmp_path)
    out.mkdir(parents=True)
    host = socket.gethostname()

    child = subprocess.Popen([sys.executable, "-c", "import time; time.sleep(60)"])
    try:
        (out / LOCK_FILE).write_text(json.dumps({"host": host, "pid": child.pid, "started_at": "x"}))
        with pytest.raises(StepLockedError):
            run_steps([step], tmp_path, NoLauncher(), dry_run=False)
    finally:
        child.kill()
        child.wait()

    (out / LOCK_FILE).write_text(json.dumps({"host": host, "pid": child.pid, "started_at": "x"}))
    with pytest.warns(UserWarning, match=str(child.pid)):
        run_steps([step], tmp_path, NoLauncher(), dry_run=False)
    assert status_of(out)["status"] == Status.SUCCESS
    assert not (out / LOCK_FILE).exists()


class Mode(StrEnum):
    FAST = "fast"
    SLOW = "slow"


@dataclass(frozen=True)
class EntryCfg:
    output_dir: str = ""
    mode: Mode = Mode.FAST
    modes: tuple[Mode, ...] = ()


def test_entrypoint_config_yaml_round_trips_through_draccus(tmp_path):
    parsed: list[EntryCfg] = []

    class ParsingLauncher:
        def run(self, module, config_path, resources, log_dir) -> None:
            parsed.append(draccus.parse(EntryCfg, args=["--config_path", str(config_path)]))

    step = Step("e", EntryCfg(mode=Mode.SLOW, modes=(Mode.SLOW, Mode.FAST)), entrypoint="unused.module")
    run_steps([step], tmp_path, ParsingLauncher(), dry_run=False)
    assert parsed == [EntryCfg(str(step.output_path(tmp_path)), Mode.SLOW, (Mode.SLOW, Mode.FAST))]
