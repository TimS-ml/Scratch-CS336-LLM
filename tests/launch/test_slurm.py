import subprocess
from pathlib import Path

import pytest

from scratch_cs336.launch import LaunchError, Resources, SlurmConfig, SlurmLauncher

RESOURCES = Resources(nnodes=2, nproc_per_node=4, gpus_per_node=4, cpus_per_task=8, mem="64G", time="01:00:00")


def test_dry_run_writes_valid_bash_without_submitting(tmp_path):
    def forbidden(args):
        raise AssertionError(f"dry run executed {args}")

    launcher = SlurmLauncher(
        SlurmConfig(
            partition="gpu", account="lab", qos="high", extra_sbatch=("--constraint=a100",), setup=("module load cuda",)
        ),
        dry_run=True,
        runner=forbidden,
    )
    launcher.run("some.module", tmp_path / "config.yaml", RESOURCES, tmp_path / "step dir")
    script = tmp_path / "step dir" / "job.sbatch"
    assert script.exists()
    subprocess.run(["bash", "-n", str(script)], check=True)


def test_script_is_executable_bash_that_hands_srun_the_cluster_geometry(tmp_path):
    """Run the rendered script with fake scontrol/srun binaries and inspect what srun received."""
    bin_dir = tmp_path / "bin"
    bin_dir.mkdir()
    (bin_dir / "scontrol").write_text("#!/bin/bash\nprintf 'node-a\\nnode-b\\n'\n")
    (bin_dir / "srun").write_text('#!/bin/bash\necho "$@" > "$SRUN_OUT"\n')
    for f in bin_dir.iterdir():
        f.chmod(0o755)

    launcher = SlurmLauncher(SlurmConfig(setup=("export FOO=1",)), dry_run=True)
    launcher.run("some.module", tmp_path / "config.yaml", RESOURCES, tmp_path)
    srun_out = tmp_path / "srun.out"
    env = {
        "PATH": f"{bin_dir}:/usr/bin:/bin",
        "SLURM_JOB_NODELIST": "node-[a-b]",
        "SLURM_NNODES": "2",
        "SLURM_JOB_ID": "77",
        "SRUN_OUT": str(srun_out),
    }
    subprocess.run(["bash", str(tmp_path / "job.sbatch")], check=True, env=env)
    args = srun_out.read_text().split()
    assert args[:1] == ["torchrun"]
    assert args[args.index("--nnodes") + 1] == "2"
    assert args[args.index("--nproc-per-node") + 1] == "4"
    assert args[args.index("--rdzv-id") + 1] == "77"
    assert args[args.index("--rdzv-endpoint") + 1] == "node-a:29500"
    assert args[args.index("-m") + 1] == "some.module"


class FakeSlurm:
    def __init__(self, states: list[str]):
        self.states = list(states)
        self.calls: list[list[str]] = []
        self.sleeps: list[float] = []

    def __call__(self, args):
        self.calls.append(list(args))
        if args[0] == "sbatch":
            return "123;cluster\n"
        if args[0] == "sacct":
            return self.states.pop(0) + "\n"
        return ""

    def sleep(self, seconds: float) -> None:
        self.sleeps.append(seconds)


def launcher_for(fake: FakeSlurm) -> SlurmLauncher:
    return SlurmLauncher(SlurmConfig(poll_seconds=5), runner=fake, sleep=fake.sleep)


def test_completed_job_returns_after_polling_through_non_terminal_states(tmp_path):
    fake = FakeSlurm(["", "PENDING", "RUNNING", "COMPLETED"])
    launcher_for(fake).run("m", tmp_path / "c.yaml", RESOURCES, tmp_path)
    assert fake.calls[0][:2] == ["sbatch", "--parsable"]
    assert all(c[:3] == ["sacct", "-j", "123"] for c in fake.calls[1:])
    assert fake.sleeps == [5, 5, 5]


@pytest.mark.parametrize("state", ["FAILED", "TIMEOUT", "CANCELLED by 1001", "OUT_OF_MEMORY"])
def test_unsuccessful_job_raises_with_log_path(tmp_path, state):
    fake = FakeSlurm(["RUNNING", state])
    with pytest.raises(LaunchError) as exc:
        launcher_for(fake).run("m", tmp_path / "c.yaml", RESOURCES, tmp_path)
    assert exc.value.log_path == Path(tmp_path / "slurm-123.out")
    assert state.split()[0] in str(exc.value)


def test_interrupt_while_waiting_cancels_the_job(tmp_path):
    fake = FakeSlurm(["RUNNING"])

    def interrupt(seconds: float) -> None:
        raise KeyboardInterrupt

    launcher = SlurmLauncher(SlurmConfig(), runner=fake, sleep=interrupt)
    with pytest.raises(KeyboardInterrupt):
        launcher.run("m", tmp_path / "c.yaml", RESOURCES, tmp_path)
    assert fake.calls[-1] == ["scancel", "123"]
