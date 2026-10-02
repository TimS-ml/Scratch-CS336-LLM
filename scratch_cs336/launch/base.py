from __future__ import annotations

from pathlib import Path
from typing import Protocol

from scratch_cs336.launch.resources import Resources


class LaunchError(RuntimeError):
    """A launched job exited unsuccessfully; ``log_path`` is where to look."""

    def __init__(self, message: str, log_path: Path | None = None):
        super().__init__(message if log_path is None else f"{message} (log: {log_path})")
        self.log_path = log_path


class Launcher(Protocol):
    def run(self, module: str, config_path: Path, resources: Resources, log_dir: Path) -> None:
        """Run ``python -m module --config_path config_path`` across ``resources``; block until done.

        Raises :class:`LaunchError` if the job does not finish successfully.
        """
        ...
