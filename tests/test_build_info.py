import json
import subprocess
import sys
from pathlib import Path
from types import SimpleNamespace

import pytest

from src import __version__, build_info

COMMIT = "a" * 40


@pytest.fixture(autouse=True)
def reset_build_info(monkeypatch):
    monkeypatch.setattr(build_info, "_build_info", None)
    monkeypatch.setattr(build_info, "_capture_started", False)


@pytest.fixture
def source_package(tmp_path, monkeypatch):
    root = tmp_path / "checkout"
    package = root / "src"
    package.mkdir(parents=True)
    (root / ".git").write_text("gitdir: elsewhere", encoding="utf-8")
    (root / "pyproject.toml").write_text('[project]\nname = "murmur"', "utf-8")
    monkeypatch.setattr(build_info, "__file__", str(package / "build_info.py"))
    return package


@pytest.mark.parametrize("status", ["", " M src/main.py", "?? new.py"])
def test_source_metadata_uses_package_root_and_bounded_git(
    source_package, tmp_path, monkeypatch, status
):
    calls = []
    monkeypatch.chdir(tmp_path)

    def run(command, **kwargs):
        calls.append((command, kwargs))
        assert command[:3] == ["git", "-C", str(source_package.parent)]
        assert kwargs["capture_output"] is True
        assert kwargs["timeout"] == 2
        assert kwargs["check"] is True
        outputs = {
            ("rev-parse", "--show-toplevel"): str(source_package.parent),
            ("rev-parse", "--verify", "HEAD"): COMMIT,
            ("status", "--porcelain=v1"): status,
        }
        return SimpleNamespace(stdout=outputs[tuple(command[3:])] + "\n")

    monkeypatch.setattr(build_info.subprocess, "run", run)
    assert build_info.get_build_info() == {
        "app_version": __version__,
        "git_commit": COMMIT,
        "git_dirty": bool(status),
    }
    assert len(calls) == 3


@pytest.mark.parametrize(
    "error",
    [
        FileNotFoundError("git"),
        subprocess.TimeoutExpired("git", 2),
        subprocess.CalledProcessError(128, "git"),
    ],
)
def test_unavailable_git_stays_unknown_for_process(source_package, monkeypatch, error):
    calls = []

    def fail(*args, **kwargs):
        calls.append(args)
        raise error

    monkeypatch.setattr(build_info.subprocess, "run", fail)
    expected = {"app_version": __version__, "git_commit": None, "git_dirty": None}
    assert build_info.get_build_info() == expected
    assert build_info.get_build_info() == expected
    assert len(calls) == 1


def test_identity_is_cached_and_callers_cannot_mutate_it(source_package, monkeypatch):
    calls = []

    def read(root):
        calls.append(root)
        return {"git_commit": COMMIT, "git_dirty": False}

    monkeypatch.setattr(build_info, "_read_git_info", read)
    first = build_info.get_build_info()
    first["git_commit"] = "changed"
    # A later checkout change cannot rewrite an earlier run's identity.
    monkeypatch.setattr(
        build_info, "_source_root", lambda _: pytest.fail("re-read package location")
    )
    assert build_info.get_build_info()["git_commit"] == COMMIT
    assert calls == [source_package.parent]


def test_background_capture_freezes_launch_identity_without_blocking(
    source_package, monkeypatch
):
    import threading

    release = threading.Event()
    probes = []

    def slow_read(root):
        probes.append(root)
        release.wait(5)
        return {"git_commit": COMMIT, "git_dirty": False}

    monkeypatch.setattr(build_info, "_read_git_info", slow_read)
    build_info.capture_build_info_in_background()
    # A second entry point (run.py, then main) returns at once mid-probe.
    finished = threading.Event()
    threading.Thread(
        target=lambda: (build_info.capture_build_info_in_background(), finished.set())
    ).start()
    assert finished.wait(1)
    # Startup continues while Git is still running.
    assert build_info._build_info is None
    release.set()
    # Later callers wait on the same capture instead of probing again.
    assert build_info.get_build_info()["git_commit"] == COMMIT
    assert len(probes) == 1


@pytest.mark.parametrize("frozen", [False, True])
@pytest.mark.parametrize("embedded", [False, True])
def test_installed_and_frozen_never_query_checkout(
    tmp_path, monkeypatch, frozen, embedded
):
    root = tmp_path / ("frozen" if frozen else "site-packages")
    package = root / "src"
    package.mkdir(parents=True)
    # Even repository markers next to the package and a checkout as cwd are ignored.
    (root / ".git").mkdir()
    (root / "pyproject.toml").touch()
    monkeypatch.chdir(root)
    monkeypatch.setattr(build_info, "__file__", str(package / "build_info.py"))
    monkeypatch.setattr(sys, "frozen", frozen, raising=False)
    monkeypatch.setattr(
        build_info, "_git", lambda *args: pytest.fail("installed build invoked Git")
    )
    expected = {"app_version": __version__, "git_commit": None, "git_dirty": None}
    if embedded:
        expected = {"app_version": "1.2.3", "git_commit": COMMIT, "git_dirty": True}
        (package / "_build_info.json").write_text(json.dumps(expected), "utf-8")
    assert build_info.get_build_info() == expected


@pytest.mark.parametrize(
    "content",
    ["not json", "[]", '{"app_version": null, "git_commit": "short", "git_dirty": 1}'],
)
def test_invalid_embedded_metadata_is_unknown(tmp_path, monkeypatch, content):
    package = tmp_path / "src"
    package.mkdir()
    monkeypatch.setattr(build_info, "__file__", str(package / "build_info.py"))
    monkeypatch.setattr(
        build_info, "_git", lambda *args: pytest.fail("archive install invoked Git")
    )
    (package / "_build_info.json").write_text(content, "utf-8")
    assert build_info.get_build_info() == {
        "app_version": __version__,
        "git_commit": None,
        "git_dirty": None,
    }


@pytest.mark.parametrize("commit", ["short", "", "a" * 64])
def test_git_requires_full_revision(source_package, monkeypatch, commit):
    values = iter([str(source_package.parent), commit, ""])
    monkeypatch.setattr(build_info, "_git", lambda *args: next(values))
    result = build_info.get_build_info()
    assert result["git_commit"] == (commit if len(commit) == 64 else None)
    assert result["git_dirty"] is (False if len(commit) == 64 else None)


def test_git_must_identify_exact_source_root(source_package, monkeypatch):
    calls = []

    def git(*args):
        calls.append(args)
        return str(source_package.parent.parent)

    monkeypatch.setattr(build_info, "_git", git)
    assert build_info.get_build_info()["git_commit"] is None
    assert len(calls) == 1


def test_status_failure_keeps_revision_with_unknown_dirty(source_package, monkeypatch):
    def git(root: Path, *args):
        if args == ("status", "--porcelain=v1"):
            raise subprocess.TimeoutExpired("git", 2)
        return str(root) if args[-1] == "--show-toplevel" else COMMIT

    monkeypatch.setattr(build_info, "_git", git)
    assert build_info.get_build_info() == {
        "app_version": __version__,
        "git_commit": COMMIT,
        "git_dirty": None,
    }
