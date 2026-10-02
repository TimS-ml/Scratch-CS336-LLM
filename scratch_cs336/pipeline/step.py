"""Experiment DAG nodes.

A :class:`Step`'s output directory is ``<root>/<name>-<hash8>``. The hash covers the step name and its config
with every :class:`InputPath` replaced by the dependency's own ``<name>-<hash8>``, so changing any upstream
config forks every downstream path, while the ``root`` location never matters.
"""

from __future__ import annotations

import dataclasses
import hashlib
import json
import re
from collections.abc import Callable
from dataclasses import dataclass
from enum import Enum
from functools import cached_property
from pathlib import Path
from typing import Any

from scratch_cs336.launch.resources import Resources

_NAME = re.compile(r"[A-Za-z0-9_.-]+")


@dataclass(frozen=True)
class InputPath:
    """Reference to (a sub-path of) another step's output directory."""

    step: Step
    subpath: str = ""


def _map_inputs(value: Any, fn: Callable[[InputPath], Any]) -> Any:
    """Rebuild ``value`` with every InputPath replaced by ``fn(input_path)``."""
    if isinstance(value, InputPath):
        return fn(value)
    if dataclasses.is_dataclass(value) and not isinstance(value, type):
        changes = {f.name: _map_inputs(getattr(value, f.name), fn) for f in dataclasses.fields(value) if f.init}
        return dataclasses.replace(value, **changes)
    if isinstance(value, list):
        return [_map_inputs(v, fn) for v in value]
    if isinstance(value, tuple):
        return tuple(_map_inputs(v, fn) for v in value)
    if isinstance(value, dict):
        return {k: _map_inputs(v, fn) for k, v in value.items()}
    return value


def _canonical(value: Any) -> Any:
    """JSON-able, order-independent form of a config value."""
    if isinstance(value, InputPath):
        return {"__input__": f"{value.step.output_name}/{value.subpath}"}
    if dataclasses.is_dataclass(value) and not isinstance(value, type):
        fields = {f.name: _canonical(getattr(value, f.name)) for f in dataclasses.fields(value)}
        return {"__type__": type(value).__qualname__, **fields}
    if isinstance(value, Enum):
        return _canonical(value.value)
    if isinstance(value, Path):
        return str(value)
    if value is None or isinstance(value, bool | int | float | str):
        return value
    if isinstance(value, list | tuple):
        return [_canonical(v) for v in value]
    if isinstance(value, dict):
        if not all(isinstance(k, str) for k in value):
            raise TypeError(f"config dict keys must be str, got {sorted(map(repr, value))}")
        return {k: _canonical(v) for k, v in value.items()}
    raise TypeError(f"unsupported config value of type {type(value).__name__}: {value!r}")


def _dumps(value: Any) -> str:
    return json.dumps(value, sort_keys=True, separators=(",", ":"))


@dataclass(frozen=True, eq=False)  # identity hash/eq: a DAG node is the object itself
class Step[C]:
    name: str
    config: C
    run: Callable[[C, Path], None] | None = None
    entrypoint: str | None = None
    resources: Resources | None = None

    def __post_init__(self) -> None:
        if not _NAME.fullmatch(self.name):
            raise ValueError(f"step name must match {_NAME.pattern}, got {self.name!r}")
        if (self.run is None) == (self.entrypoint is None):
            raise ValueError(f"step {self.name!r}: exactly one of run / entrypoint is required")
        if self.entrypoint is not None:
            fields = {f.name for f in dataclasses.fields(self.config)} if dataclasses.is_dataclass(self.config) else ()
            if "output_dir" not in fields:
                raise ValueError(f"step {self.name!r}: entrypoint config must be a dataclass with an output_dir field")
        _canonical(self.config)  # fail at construction on unsupported values

    @cached_property
    def deps(self) -> tuple[Step, ...]:
        found: dict[str, Step] = {}

        def collect(ref: InputPath) -> None:
            found.setdefault(ref.step.output_name, ref.step)

        _map_inputs(self.config, collect)
        return tuple(found.values())

    @cached_property
    def hash_id(self) -> str:
        payload = _dumps({"name": self.name, "config": _canonical(self.config)})
        return hashlib.sha256(payload.encode()).hexdigest()[:8]

    @property
    def output_name(self) -> str:
        return f"{self.name}-{self.hash_id}"

    def output_path(self, root: Path) -> Path:
        return root / self.output_name

    def resolve(self, root: Path) -> C:
        """Config copy with every InputPath replaced by the concrete path string under ``root``."""

        def concrete(ref: InputPath) -> str:
            path = ref.step.output_path(root)
            return str(path / ref.subpath if ref.subpath else path)

        return _map_inputs(self.config, concrete)


def canonical_json(config: Any) -> str:
    """Canonical JSON of a (resolved or unresolved) config, for provenance files."""
    return _dumps(_canonical(config))
