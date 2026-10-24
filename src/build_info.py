"""Resolve checkout identity once at startup and reuse it for session statistics.

Source identity describes the checkout at launch. Restart after editing or
changing commits to keep comparisons meaningful. Installed builds only trust
embedded metadata, never a checkout found through the working directory.
"""

import json
import re
import subprocess
import sys
import threading
from pathlib import Path

from src import __version__

_BUILD_INFO_FILENAME = "_build_info.json"
_GIT_TIMEOUT_SECONDS = 2
_build_info: dict | None = None
_build_info_lock = threading.Lock()
_capture_started = False
# Separate from _build_info_lock, which is held while Git runs.
_capture_lock = threading.Lock()


def _git(repo_root: Path, *args: str) -> str:
    result = subprocess.run(
        ["git", "-C", str(repo_root), *args],
        capture_output=True,
        text=True,
        encoding="utf-8",
        errors="replace",
        timeout=_GIT_TIMEOUT_SECONDS,
        check=True,
        creationflags=getattr(subprocess, "CREATE_NO_WINDOW", 0),
    )
    return result.stdout.strip()


def _source_root(package_dir: Path) -> Path | None:
    root = package_dir.parent
    if getattr(sys, "frozen", False) or any(
        part.lower() in {"site-packages", "dist-packages"} for part in package_dir.parts
    ):
        return None
    if (root / ".git").exists() and (root / "pyproject.toml").is_file():
        return root
    return None


def _read_git_info(repo_root: Path) -> dict:
    """Return identity only; discard status output without retaining paths."""
    unknown = {"git_commit": None, "git_dirty": None}
    try:
        if Path(_git(repo_root, "rev-parse", "--show-toplevel")).resolve() != (
            repo_root.resolve()
        ):
            return unknown
        commit = _git(repo_root, "rev-parse", "--verify", "HEAD")
        if re.fullmatch(r"[0-9a-f]{40}|[0-9a-f]{64}", commit) is None:
            return unknown
        try:
            dirty = bool(_git(repo_root, "status", "--porcelain=v1"))
        except (OSError, subprocess.SubprocessError):
            dirty = None
        return {"git_commit": commit, "git_dirty": dirty}
    except (OSError, subprocess.SubprocessError):
        return unknown


def _read_embedded_info(package_dir: Path) -> dict:
    try:
        data = json.loads((package_dir / _BUILD_INFO_FILENAME).read_text("utf-8"))
    except (OSError, ValueError):
        return {}
    if not isinstance(data, dict):
        return {}
    commit = data.get("git_commit")
    return {
        "app_version": data.get("app_version")
        if isinstance(data.get("app_version"), str) and data["app_version"]
        else __version__,
        "git_commit": commit
        if isinstance(commit, str)
        and re.fullmatch(r"[0-9a-f]{40}|[0-9a-f]{64}", commit)
        else None,
        "git_dirty": data.get("git_dirty")
        if isinstance(data.get("git_dirty"), bool)
        else None,
    }


def capture_build_info_in_background() -> None:
    """Freeze launch identity without making startup wait on Git probes.

    Callers that need identity later block on the same lock until capture ends.
    Repeat calls are no-ops, so each entry point can request capture safely.
    """
    global _capture_started
    with _capture_lock:
        if _capture_started:
            return
        _capture_started = True
    threading.Thread(target=get_build_info, name="build-info", daemon=True).start()


def get_build_info() -> dict:
    """Freeze identity on first call; callers receive an independent snapshot.

    Startup triggers this through capture_build_info_in_background(). Subsequent
    calls perform no file or Git I/O, even when initial identity was unknown.
    """
    global _build_info
    with _build_info_lock:
        if _build_info is None:
            package_dir = Path(__file__).resolve().parent
            info = {"app_version": __version__, "git_commit": None, "git_dirty": None}
            if (root := _source_root(package_dir)) is not None:
                info.update(_read_git_info(root))
            else:
                info.update(_read_embedded_info(package_dir))
            _build_info = info
        return _build_info.copy()
