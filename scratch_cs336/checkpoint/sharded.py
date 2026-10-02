"""Sharded checkpoints: state is stored on disk the way it is sharded in memory.

Layout::

    <root>/step_000100/rank_00000.pt ... rank_<W-1>.pt   # torch.save of each rank's own state
    <root>/step_000100/metadata.json                    # {step, world_size, mesh}; written last by rank 0

A step directory without ``metadata.json`` is incomplete (crash mid-save) and is ignored. Saving snapshots every
tensor to CPU synchronously, so training may mutate its state right after ``save`` returns; with ``async_save`` the
file writes, the cross-rank success check, the commit and retention run on a background thread, coordinated over a
dedicated gloo group so they never interleave with the training collectives. If any rank fails to write its shard,
no rank commits and every rank raises (from ``save`` when synchronous, else from the next ``wait``/``save``).
``wait()`` blocks until the last save is committed on every rank.

Resume requires the same mesh; changing layouts goes through the full export.
"""

from __future__ import annotations

import json
import os
import re
import shutil
import threading
from collections.abc import Callable
from pathlib import Path
from typing import Any

import torch
import torch.distributed as dist
from torch import Tensor
from torch.distributed.tensor import DTensor

from scratch_cs336.distributed.mesh import Mesh

METADATA = "metadata.json"
_STEP_DIR = re.compile(r"step_(\d{6,})")


def _snapshot(x: Any) -> Any:
    if isinstance(x, DTensor):
        x = x.to_local()
    if isinstance(x, Tensor):
        return x.detach().to("cpu", copy=True)
    if isinstance(x, dict):
        return {k: _snapshot(v) for k, v in x.items()}
    if type(x) in (list, tuple):
        return type(x)(_snapshot(v) for v in x)
    return x


def _atomic_write(path: Path, write: Any) -> None:
    tmp = path.with_name(path.name + ".tmp")
    write(tmp)
    os.replace(tmp, path)


class Checkpointer:
    """Collective: construct, ``save`` and ``load`` on every rank."""

    def __init__(
        self,
        root: str | Path,
        mesh: Mesh,
        keep_last: int = 2,
        permanent_every: int | None = None,
        async_save: bool = True,
    ) -> None:
        if keep_last < 1:
            raise ValueError(f"keep_last must be >= 1, got {keep_last}")
        self.root = Path(root)
        self.mesh = mesh
        self.keep_last = keep_last
        self.permanent_every = permanent_every
        self.async_save = async_save
        self.rank = dist.get_rank()
        self.world_size = dist.get_world_size()
        self._group = dist.new_group(backend="gloo")
        self._thread: threading.Thread | None = None
        self._error: BaseException | None = None

    @property
    def mesh_shape(self) -> list[int]:
        return [self.mesh.replicate_size, self.mesh.shard_size, self.mesh.tp_size]

    def step_dir(self, step: int) -> Path:
        return self.root / f"step_{step:06d}"

    def save(self, step: int, state: dict[str, Any]) -> None:
        self.wait()
        snapshot = _snapshot(state)
        if self.async_save:
            self._thread = threading.Thread(target=self._write_guarded, args=(step, snapshot), daemon=True)
            self._thread.start()
        else:
            self._write(step, snapshot)

    def wait(self) -> None:
        if self._thread is not None:
            self._thread.join()
            self._thread = None
        if self._error is not None:
            error, self._error = self._error, None
            raise RuntimeError("asynchronous checkpoint save failed") from error

    def _write_guarded(self, step: int, snapshot: dict[str, Any]) -> None:
        try:
            self._write(step, snapshot)
        except BaseException as e:  # surfaced by wait()
            self._error = e

    def _write(self, step: int, snapshot: dict[str, Any]) -> None:
        d = self.step_dir(step)
        self._agree(step, "write their shard", lambda: self._write_shard(d, snapshot))
        self._agree(step, "commit", lambda: self._commit(d, step) if self.rank == 0 else None)

    def _write_shard(self, d: Path, snapshot: dict[str, Any]) -> None:
        d.mkdir(parents=True, exist_ok=True)
        _atomic_write(d / f"rank_{self.rank:05d}.pt", lambda p: torch.save(snapshot, p))

    def _commit(self, d: Path, step: int) -> None:
        meta = {"step": step, "world_size": self.world_size, "mesh": self.mesh_shape}
        _atomic_write(d / METADATA, lambda p: p.write_text(json.dumps(meta)))
        self._apply_retention(step)

    def _agree(self, step: int, phase: str, action: Callable[[], None]) -> None:
        """Run ``action`` here, then count failures across ranks: every rank raises if any rank failed."""
        error: Exception | None = None
        try:
            action()
        except Exception as e:
            error = e
        failures = torch.tensor([int(error is not None)])
        dist.all_reduce(failures, group=self._group)
        if error is not None:
            raise error
        if failures.item():
            raise RuntimeError(f"checkpoint step {step}: {failures.item()} other rank(s) failed to {phase}")

    def _step_dirs(self) -> dict[int, Path]:
        if not self.root.is_dir():
            return {}
        return {int(m.group(1)): p for p in self.root.iterdir() if p.is_dir() and (m := _STEP_DIR.fullmatch(p.name))}

    def committed_steps(self) -> list[int]:
        return sorted(s for s, p in self._step_dirs().items() if (p / METADATA).is_file())

    def latest_step(self) -> int | None:
        steps = self.committed_steps()
        return steps[-1] if steps else None

    def _apply_retention(self, current: int) -> None:
        committed = self.committed_steps()
        keep = set(committed[-self.keep_last :])
        if self.permanent_every:
            keep |= {s for s in committed if s % self.permanent_every == 0}
        for s, path in self._step_dirs().items():
            stale_incomplete = s < current and s not in committed
            if (s in committed and s not in keep) or stale_incomplete:
                shutil.rmtree(path, ignore_errors=True)

    def load(self, step: int) -> dict[str, Any]:
        d = self.step_dir(step)
        if not (d / METADATA).is_file():
            raise FileNotFoundError(f"no committed checkpoint at {d}")
        meta = json.loads((d / METADATA).read_text())
        if meta["world_size"] != self.world_size or meta["mesh"] != self.mesh_shape:
            raise ValueError(
                f"checkpoint {d} was saved with world_size={meta['world_size']} mesh={meta['mesh']}, "
                f"current run has world_size={self.world_size} mesh={self.mesh_shape}; use the full export to reshard"
            )
        return torch.load(d / f"rank_{self.rank:05d}.pt", map_location="cpu", weights_only=True)
