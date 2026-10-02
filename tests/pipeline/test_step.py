import os
import subprocess
import sys
from pathlib import Path

import pytest

from scratch_cs336.pipeline import InputPath, Step
from tests.pipeline.sample_steps import Cfg, Inner, Mode, diamond, noop

REPO_ROOT = Path(__file__).resolve().parents[2]


def sink(**kw) -> Step:
    return diamond(**kw)[0]


def test_hash_is_stable_across_processes():
    def in_subprocess(seed: str) -> str:
        env = {**os.environ, "PYTHONHASHSEED": seed}
        res = subprocess.run(
            [sys.executable, "-m", "tests.pipeline.sample_steps"],
            cwd=REPO_ROOT,
            env=env,
            capture_output=True,
            text=True,
        )
        assert res.returncode == 0, res.stderr
        return res.stdout.strip()

    s = sink()
    expected = ",".join(f"{d.name}:{d.hash_id}" for d in s.deps) + " " + s.hash_id
    assert in_subprocess("1") == in_subprocess("2") == expected


def test_hash_changes_with_own_config_and_name():
    base = Step("s", Cfg(), run=noop)
    assert Step("s", Cfg(), run=noop).hash_id == base.hash_id
    assert Step("s", Cfg(inner=Inner(lr=2e-3)), run=noop).hash_id != base.hash_id
    assert Step("s", Cfg(inner=Inner(mode=Mode.SLOW)), run=noop).hash_id != base.hash_id
    assert Step("t", Cfg(), run=noop).hash_id != base.hash_id


def test_hash_changes_with_any_transitive_dependency():
    base = sink()
    assert sink(lr=1e-3).hash_id == base.hash_id
    changed = sink(lr=5e-4)
    # The root of the diamond only appears indirectly in the sink's config.
    assert changed.hash_id != base.hash_id
    assert {d.name for d in changed.deps} == {"b", "c"}


def test_hash_depends_on_referenced_subpath():
    a = Step("a", Cfg(), run=noop)
    assert (
        Step("b", Cfg(upstream=(InputPath(a, "x"),)), run=noop).hash_id
        != Step("b", Cfg(upstream=(InputPath(a, "y"),)), run=noop).hash_id
    )


def test_hash_and_output_name_do_not_depend_on_root(tmp_path):
    s = sink()
    assert s.output_path(tmp_path / "one").name == s.output_path(tmp_path / "two").name == s.output_name
    # Resolved configs differ only by root.
    assert s.resolve(tmp_path / "one") != s.resolve(tmp_path / "two")


def test_dict_key_order_does_not_change_hash():
    one = Step("s", Cfg(extra={"a": 1, "b": "x"}), run=noop)
    two = Step("s", Cfg(extra={"b": "x", "a": 1}), run=noop)
    assert one.hash_id == two.hash_id


def test_resolve_replaces_inputs_with_concrete_paths(tmp_path):
    a = Step("a", Cfg(), run=noop)
    b = Step("b", Cfg(upstream=(InputPath(a, "x/y"), InputPath(a))), run=noop)
    resolved = b.resolve(tmp_path)
    assert resolved.upstream == (str(tmp_path / a.output_name / "x/y"), str(tmp_path / a.output_name))


def test_deps_are_unique_and_found_inside_nested_containers():
    a = Step("a", Cfg(), run=noop)
    b = Step("b", Cfg(extra={"k": 1}), run=noop)
    s = Step("s", Cfg(upstream=(InputPath(a), InputPath(a, "z"), InputPath(b))), run=noop)
    assert s.deps == (a, b)


@pytest.mark.parametrize(
    "kwargs",
    [
        dict(run=None, entrypoint=None),
        dict(run=noop, entrypoint="some.module"),
    ],
)
def test_exactly_one_of_run_and_entrypoint(kwargs):
    with pytest.raises(ValueError, match="exactly one"):
        Step("s", Cfg(), **kwargs)


def test_entrypoint_config_requires_output_dir():
    with pytest.raises(ValueError, match="output_dir"):
        Step("s", Cfg(), entrypoint="some.module")


def test_unsupported_config_values_fail_at_construction():
    with pytest.raises(TypeError, match="unsupported"):
        Step("s", Cfg(extra={"f": object()}), run=noop)  # type: ignore[dict-item]


def test_name_cannot_escape_the_root():
    with pytest.raises(ValueError, match="name"):
        Step("../evil", Cfg(), run=noop)
