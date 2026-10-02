from __future__ import annotations

import dataclasses
import json
import os
import socket
import warnings
from collections.abc import Iterable
from datetime import UTC, datetime
from enum import StrEnum
from pathlib import Path
from typing import Any

import draccus

from scratch_cs336.launch.base import Launcher
from scratch_cs336.launch.resources import Resources
from scratch_cs336.pipeline.step import Step, canonical_json

STATUS_FILE = ".status.json"
LOCK_FILE = ".lock"
PROVENANCE_FILE = ".step.json"
CONFIG_FILE = "config.yaml"

# draccus resolves StrEnum members to its ``str`` encoder (``str`` precedes ``Enum`` in the MRO), which hands the
# enum object to yaml as a python-tagged object the entrypoint cannot load back; write the value instead.
draccus.encode.register(StrEnum, lambda x, _=None: x.value, include_subclasses=True)


class Status(StrEnum):
    RUNNING = "RUNNING"
    SUCCESS = "SUCCESS"
    FAILED = "FAILED"


class StepLockedError(RuntimeError):
    """Another runner holds (or crashed while holding) the step's lock."""


def _now() -> str:
    return datetime.now(UTC).isoformat(timespec="seconds")


def _write_json(path: Path, payload: dict[str, Any]) -> None:
    tmp = path.with_name(path.name + ".tmp")
    tmp.write_text(json.dumps(payload, indent=2))
    os.replace(tmp, path)


def read_status(out: Path) -> Status | None:
    path = out / STATUS_FILE
    if not path.exists():
        return None
    return Status(json.loads(path.read_text())["status"])


def _write_status(out: Path, status: Status, started_at: str, error: str | None = None) -> None:
    payload = {
        "status": status.value,
        "started_at": started_at,
        "updated_at": _now(),
        "host": socket.gethostname(),
        "pid": os.getpid(),
    }
    if error is not None:
        payload["error"] = error
    _write_json(out / STATUS_FILE, payload)


def _lock_payload() -> dict[str, Any]:
    return {"host": socket.gethostname(), "pid": os.getpid(), "started_at": _now()}


def _is_dead_local_owner(owner: dict[str, Any]) -> bool:
    if owner.get("host") != socket.gethostname() or not isinstance(owner.get("pid"), int):
        return False
    try:
        os.kill(owner["pid"], 0)
    except ProcessLookupError:
        return True
    except PermissionError:  # exists, owned by another user
        return False
    return False


def _acquire_lock(out: Path) -> Path:
    """Exclusive creation. A lock left by a dead process on this host is taken over; any other is an error."""
    lock = out / LOCK_FILE
    payload = _lock_payload()
    try:
        fd = os.open(lock, os.O_CREAT | os.O_EXCL | os.O_WRONLY)
    except FileExistsError:
        text = lock.read_text() if lock.exists() else ""
        try:
            owner = json.loads(text)
        except json.JSONDecodeError:
            owner = {}
        if not _is_dead_local_owner(owner):
            raise StepLockedError(
                f"{out} is locked by {text.strip() or 'unknown owner'}. If that process is gone, delete {lock} and re-run."
            ) from None
        warnings.warn(f"taking over stale lock {lock} left by dead pid {owner['pid']}", stacklevel=2)
        tmp = lock.with_name(f"{LOCK_FILE}.{os.getpid()}.tmp")
        tmp.write_text(json.dumps(payload))
        os.replace(tmp, lock)
        if json.loads(lock.read_text()) != payload:  # another runner took over at the same moment
            raise StepLockedError(f"{out}: lost the race for stale lock {lock}") from None
        return lock
    with os.fdopen(fd, "w") as f:
        json.dump(payload, f)
    return lock


def _topological(steps: Iterable[Step]) -> list[Step]:
    ordered: dict[str, Step] = {}  # output_name -> step, dependencies first
    by_name: dict[str, Step] = {}

    def visit(step: Step) -> None:
        if step.output_name in ordered:
            return
        other = by_name.setdefault(step.name, step)
        if other.output_name != step.output_name:
            raise ValueError(
                f"two different steps are both named {step.name!r}: {other.output_name}, {step.output_name}"
            )
        for dep in step.deps:
            visit(dep)
        ordered[step.output_name] = step

    for step in steps:
        visit(step)
    return list(ordered.values())


def _execute(step: Step, root: Path, out: Path, launcher: Launcher) -> None:
    resolved = step.resolve(root)
    if step.entrypoint is not None:
        resolved = dataclasses.replace(resolved, output_dir=str(out))
    _write_json(
        out / PROVENANCE_FILE,
        {
            "name": step.name,
            "hash": step.hash_id,
            "config": json.loads(canonical_json(resolved)),
            "deps": [d.output_name for d in step.deps],
        },
    )
    if step.run is not None:
        step.run(resolved, out)
        return
    assert step.entrypoint is not None
    config_path = out / CONFIG_FILE
    with config_path.open("w") as f:
        draccus.dump(resolved, f)
    launcher.run(step.entrypoint, config_path, step.resources or Resources(), out)


def _run_step(step: Step, root: Path, launcher: Launcher, force: bool) -> None:
    out = step.output_path(root)
    out.mkdir(parents=True, exist_ok=True)
    lock = _acquire_lock(out)
    try:
        if not force and read_status(out) == Status.SUCCESS:  # finished by another runner while we waited
            return
        started_at = _now()
        _write_status(out, Status.RUNNING, started_at)
        try:
            _execute(step, root, out, launcher)
        except BaseException as e:
            _write_status(out, Status.FAILED, started_at, error=f"{type(e).__name__}: {e}")
            raise
        _write_status(out, Status.SUCCESS, started_at)
    finally:
        lock.unlink(missing_ok=True)


def run_steps(
    steps: Iterable[Step],
    root: Path,
    launcher: Launcher,
    dry_run: bool = True,
    force: set[str] | frozenset[str] = frozenset(),
) -> dict[str, Path]:
    """Run ``steps`` and their transitive dependencies in topological order; return ``{step.name: output_dir}``.

    Steps whose ``.status.json`` says SUCCESS are skipped unless their name is in ``force`` (only the named
    steps rerun; dependents keep their cache since their hashes do not change). With ``dry_run`` the plan is
    printed and nothing is touched.
    """
    root = Path(root)
    ordered = _topological(steps)
    unknown = set(force) - {s.name for s in ordered}
    if unknown:
        raise ValueError(f"--force names not in the plan: {sorted(unknown)}")

    todo = {s.output_name: s.name in force or read_status(s.output_path(root)) != Status.SUCCESS for s in ordered}
    for s in ordered:
        label = "run" if todo[s.output_name] else "cached"
        print(f"[{label:6}] {s.name:<30} {s.output_path(root)}")
    if not dry_run:
        for s in ordered:
            if todo[s.output_name]:
                print(f"=== {s.name} ===", flush=True)
                _run_step(s, root, launcher, force=s.name in force)
    return {s.name: s.output_path(root) for s in ordered}
