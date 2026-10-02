from __future__ import annotations

import subprocess
import sys
from pathlib import Path

from scratch_cs336.launch.base import LaunchError
from scratch_cs336.launch.resources import Resources


class LocalLauncher:
    """Single-machine launch via ``torchrun --standalone`` (multi-GPU box, RunPod/vast.ai pod)."""

    def __init__(self, torchrun: str | None = None):
        self.torchrun = torchrun or str(Path(sys.executable).parent / "torchrun")

    def command(self, module: str, config_path: Path, resources: Resources) -> list[str]:
        if resources.nnodes != 1:
            raise ValueError(f"LocalLauncher runs on one machine, got nnodes={resources.nnodes}")
        return [
            self.torchrun,
            "--standalone",
            "--nnodes",
            "1",
            "--local-addr",  # all ranks are on this host; skip hostname resolution, which can hang on odd DNS setups
            "127.0.0.1",
            "--nproc-per-node",
            str(resources.nproc_per_node),
            "-m",
            module,
            "--config_path",
            str(config_path),
        ]

    def run(self, module: str, config_path: Path, resources: Resources, log_dir: Path) -> None:
        cmd = self.command(module, config_path, resources)
        log_dir.mkdir(parents=True, exist_ok=True)
        log_path = log_dir / "launch.log"
        with log_path.open("wb") as log:
            proc = subprocess.run(cmd, stdout=log, stderr=subprocess.STDOUT)
        if proc.returncode != 0:
            raise LaunchError(f"{module} exited with code {proc.returncode}", log_path)
