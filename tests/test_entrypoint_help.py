"""Every draccus entry point renders ``--help`` (field comments become argparse help, where a bare ``%`` crashes)."""

import subprocess
import sys

import pytest

ENTRY_POINTS = [
    "scratch_cs336.train.pretrain",
    "scratch_cs336.eval.perplexity",
    "scratch_cs336.posttrain.sft",
    "scratch_cs336.posttrain.dpo",
    "scratch_cs336.posttrain.grpo",
    "scratch_cs336.posttrain.reward_eval",
]


@pytest.mark.parametrize("module", ENTRY_POINTS)
def test_help_renders(module: str) -> None:
    proc = subprocess.run([sys.executable, "-m", module, "--help"], capture_output=True, text=True, timeout=60)
    assert proc.returncode == 0, proc.stderr[-2000:]
    assert "--config_path" in proc.stdout
