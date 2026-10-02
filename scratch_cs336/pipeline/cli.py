from __future__ import annotations

import argparse
import dataclasses
import types
import typing
from collections.abc import Callable, Sequence
from enum import Enum, StrEnum
from pathlib import Path
from typing import Any

from scratch_cs336.launch.base import Launcher
from scratch_cs336.launch.local import LocalLauncher
from scratch_cs336.launch.slurm import SlurmConfig, SlurmLauncher
from scratch_cs336.pipeline.runner import run_steps
from scratch_cs336.pipeline.step import Step


class LauncherKind(StrEnum):
    LOCAL = "local"
    SLURM = "slurm"


def _add_options(parser: Any, options: type) -> None:
    """One ``--field-name`` flag per field of the ``options`` dataclass (bool fields are on/off switches)."""
    hints = typing.get_type_hints(options)
    for f in dataclasses.fields(options):
        if f.default is not dataclasses.MISSING:
            default = f.default
        elif f.default_factory is not dataclasses.MISSING:
            default = f.default_factory()
        else:
            raise ValueError(f"experiment option {f.name!r} needs a default")
        flag = "--" + f.name.replace("_", "-")
        kind = hints[f.name]
        if isinstance(kind, types.UnionType):  # `X | None`: parse as X
            kind = next(a for a in typing.get_args(kind) if a is not types.NoneType)
        if kind is bool:
            parser.add_argument(flag, dest=f.name, action=argparse.BooleanOptionalAction, default=default)
        elif isinstance(kind, type) and issubclass(kind, Enum):
            parser.add_argument(flag, dest=f.name, type=kind, choices=list(kind), default=default)
        else:
            parser.add_argument(flag, dest=f.name, type=kind, default=default, help=f"default: {default}")


def experiment_main(build: Callable[[Any], list[Step]], options: type, argv: Sequence[str] | None = None) -> None:
    """CLI for an experiment file: prints the plan by default, executes it with ``--run``.

    ``options`` is a dataclass whose fields all have defaults: each field becomes a ``--field-name`` flag and
    ``build`` receives the parsed instance.
    """
    parser = argparse.ArgumentParser(description="Experiment DAG: prints the plan, executes it with --run.")
    parser.add_argument("--root", type=Path, default=Path("runs"), help="directory holding <name>-<hash> outputs")
    parser.add_argument("--run", action="store_true", help="execute the plan (default: print it only)")
    parser.add_argument("--launcher", type=LauncherKind, choices=list(LauncherKind), default=LauncherKind.LOCAL)
    parser.add_argument("--partition")
    parser.add_argument("--account")
    parser.add_argument("--qos")
    parser.add_argument("--force", action="append", default=[], metavar="NAME", help="rerun this step (repeatable)")
    _add_options(parser.add_argument_group("experiment options"), options)
    args = parser.parse_args(argv)

    launcher: Launcher
    if args.launcher == LauncherKind.SLURM:
        launcher = SlurmLauncher(SlurmConfig(partition=args.partition, account=args.account, qos=args.qos))
    else:
        launcher = LocalLauncher()
    values: dict[str, Any] = {f.name: getattr(args, f.name) for f in dataclasses.fields(options)}
    steps = build(options(**values))
    run_steps(steps, args.root, launcher, dry_run=not args.run, force=set(args.force))
