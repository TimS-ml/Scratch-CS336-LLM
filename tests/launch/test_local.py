import json
from pathlib import Path

import pytest

from scratch_cs336.launch import LaunchError, LocalLauncher, Resources
from scratch_cs336.pipeline import InputPath, Step, run_steps
from scratch_cs336.pipeline.runner import STATUS_FILE
from tests.launch.dummy_entry import DummyConfig

ENTRY = "tests.launch.dummy_entry"
REPO_ROOT = Path(__file__).resolve().parents[2]


@pytest.fixture(autouse=True)
def _cwd_repo_root(monkeypatch):
    monkeypatch.chdir(REPO_ROOT)  # torchrun -m resolves the entry module from the cwd


def write_data(cfg, out: Path) -> None:
    (out / "payload.txt").write_text("hello")


def test_entrypoint_step_runs_two_ranks_and_receives_resolved_upstream_path(tmp_path):
    upstream = Step("upstream", {"seed": 1}, run=lambda cfg, out: write_data(cfg, out))
    train = Step(
        "train",
        DummyConfig(data=InputPath(upstream, "payload.txt")),  # type: ignore[arg-type]
        entrypoint=ENTRY,
        resources=Resources(nproc_per_node=2),
    )
    paths = run_steps([train], tmp_path, LocalLauncher(), dry_run=False)

    out = paths["train"]
    ranks = [json.loads((out / f"rank_{r}.json").read_text()) for r in range(2)]
    assert [r["rank"] for r in ranks] == [0, 1]
    assert {r["world_size"] for r in ranks} == {2}
    assert {r["data"] for r in ranks} == {"hello"}
    assert (out / "launch.log").exists() and (out / "config.yaml").exists()
    assert json.loads((out / STATUS_FILE).read_text())["status"] == "SUCCESS"

    # Second invocation is a cache hit: rank files are not rewritten.
    stamp = (out / "rank_0.json").stat().st_mtime_ns
    run_steps([train], tmp_path, LocalLauncher(), dry_run=False)
    assert (out / "rank_0.json").stat().st_mtime_ns == stamp


def test_failing_job_raises_launch_error_with_log_and_marks_step_failed(tmp_path):
    step = Step("bad", DummyConfig(fail=True), entrypoint=ENTRY, resources=Resources(nproc_per_node=2))
    with pytest.raises(LaunchError) as exc:
        run_steps([step], tmp_path, LocalLauncher(), dry_run=False)
    assert exc.value.log_path == step.output_path(tmp_path) / "launch.log"
    assert exc.value.log_path.exists()
    assert json.loads((step.output_path(tmp_path) / STATUS_FILE).read_text())["status"] == "FAILED"


def test_local_launcher_rejects_multi_node(tmp_path):
    with pytest.raises(ValueError, match="nnodes"):
        LocalLauncher().run(ENTRY, tmp_path / "c.yaml", Resources(nnodes=2), tmp_path)
