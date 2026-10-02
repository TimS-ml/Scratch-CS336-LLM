from __future__ import annotations

import re
import shlex
import subprocess
import time
from collections.abc import Callable, Sequence
from dataclasses import dataclass
from pathlib import Path

from scratch_cs336.launch.base import LaunchError
from scratch_cs336.launch.resources import Resources

RDZV_PORT = 29500

# sacct states after which the job will never change again.
TERMINAL_STATES = frozenset(
    {
        "COMPLETED",
        "FAILED",
        "CANCELLED",
        "TIMEOUT",
        "OUT_OF_MEMORY",
        "NODE_FAIL",
        "PREEMPTED",
        "BOOT_FAIL",
        "DEADLINE",
    }
)

# Runs an external command and returns its stdout; raises LaunchError on non-zero exit.
CommandRunner = Callable[[Sequence[str]], str]


@dataclass(frozen=True)
class SlurmConfig:
    partition: str | None = None
    account: str | None = None
    qos: str | None = None
    extra_sbatch: tuple[str, ...] = ()  # raw option strings, e.g. "--constraint=a100"
    setup: tuple[str, ...] = ()  # shell lines run before srun (module load, venv activate, ...)
    poll_seconds: float = 30


def run_command(args: Sequence[str]) -> str:
    proc = subprocess.run(list(args), capture_output=True, text=True)
    if proc.returncode != 0:
        raise LaunchError(f"`{shlex.join(args)}` failed: {proc.stderr.strip()}")
    return proc.stdout


class SlurmLauncher:
    """Submit one sbatch job per step; ``srun`` starts one torchrun agent per node (c10d rendezvous)."""

    def __init__(
        self,
        cfg: SlurmConfig,
        dry_run: bool = False,
        runner: CommandRunner = run_command,
        sleep: Callable[[float], None] = time.sleep,
    ):
        self.cfg = cfg
        self.dry_run = dry_run
        self.runner = runner
        self.sleep = sleep

    def render(self, module: str, config_path: Path, resources: Resources, log_dir: Path) -> str:
        cfg = self.cfg
        job_name = re.sub(r"[^A-Za-z0-9_.-]", "_", log_dir.resolve().name)
        directives = [
            f"--job-name={job_name}",
            f"--nodes={resources.nnodes}",
            "--ntasks-per-node=1",
            f"--time={resources.time}",
            f"--output={log_dir.resolve() / 'slurm-%j.out'}",
        ]
        if resources.gpus_per_node:
            directives.append(f"--gpus-per-node={resources.gpus_per_node}")
        if resources.cpus_per_task is not None:
            directives.append(f"--cpus-per-task={resources.cpus_per_task}")
        if resources.mem is not None:
            directives.append(f"--mem={resources.mem}")
        for flag, value in (("partition", cfg.partition), ("account", cfg.account), ("qos", cfg.qos)):
            if value is not None:
                directives.append(f"--{flag}={value}")
        directives.extend(cfg.extra_sbatch)

        torchrun = (
            "srun torchrun"
            ' --nnodes "$SLURM_NNODES"'
            f" --nproc-per-node {resources.nproc_per_node}"
            ' --rdzv-id "$SLURM_JOB_ID"'
            " --rdzv-backend c10d"
            f' --rdzv-endpoint "$HEAD_NODE:{RDZV_PORT}"'
            f" -m {shlex.quote(module)}"
            f" --config_path {shlex.quote(str(config_path.resolve()))}"
        )
        lines = [
            "#!/bin/bash",
            *(f"#SBATCH {d}" for d in directives),
            "set -euo pipefail",
            *cfg.setup,
            'HEAD_NODE=$(scontrol show hostnames "$SLURM_JOB_NODELIST" | head -n1)',
            torchrun,
        ]
        return "\n".join(lines) + "\n"

    def run(self, module: str, config_path: Path, resources: Resources, log_dir: Path) -> None:
        log_dir.mkdir(parents=True, exist_ok=True)
        script_path = log_dir / "job.sbatch"
        script_path.write_text(self.render(module, config_path, resources, log_dir))
        if self.dry_run:
            return
        job_id = self.runner(["sbatch", "--parsable", str(script_path)]).strip().split(";")[0]
        log_path = log_dir / f"slurm-{job_id}.out"
        try:
            state = self._wait(job_id)
        except KeyboardInterrupt:
            self.runner(["scancel", job_id])
            raise
        if state != "COMPLETED":
            raise LaunchError(f"slurm job {job_id} ended in state {state}", log_path)

    def _wait(self, job_id: str) -> str:
        while True:
            out = self.runner(["sacct", "-j", job_id, "-n", "-X", "-o", "State"]).split()
            # sacct prints nothing until the job reaches the accounting database, and
            # "CANCELLED by <uid>" has a suffix.
            state = out[0].rstrip("+") if out else ""
            if state in TERMINAL_STATES:
                return state
            self.sleep(self.cfg.poll_seconds)
