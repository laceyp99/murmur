import importlib.util
import json
from pathlib import Path

import pytest

MODULE_PATH = (
    Path(__file__).resolve().parent.parent / "tools" / "prepare_windows_build.py"
)
SPEC = importlib.util.spec_from_file_location("prepare_windows_build", MODULE_PATH)
prepare_windows_build = importlib.util.module_from_spec(SPEC)
assert SPEC.loader is not None
SPEC.loader.exec_module(prepare_windows_build)


@pytest.mark.parametrize(
    ("version", "expected"),
    [
        ("0.1.0", (0, 1, 0, 0)),
        ("1.2.3.4", (1, 2, 3, 4)),
    ],
)
def test_windows_version_accepts_supported_versions(version, expected):
    assert prepare_windows_build._windows_version(version) == expected


@pytest.mark.parametrize("version", ["1.2", "1.2.3.dev1", "1.2.3.4.5", "1.2.70000"])
def test_windows_version_rejects_unsupported_versions(version):
    with pytest.raises(ValueError, match=r"version|components"):
        prepare_windows_build._windows_version(version)


@pytest.mark.parametrize("git_available", [False, True])
def test_prepare_build_generates_icon_version_and_metadata(
    tmp_path, monkeypatch, git_available
):
    from src import build_info

    repo_root = tmp_path / "repo"
    output_dir = repo_root / "build" / "windows"
    repo_root.mkdir()
    (repo_root / "pyproject.toml").write_text(
        '[project]\nversion = "1.2.3"\n', encoding="utf-8"
    )

    from PIL import Image

    (repo_root / "src" / "resources").mkdir(parents=True)
    Image.new("RGBA", (256, 256), color=(73, 109, 137, 255)).save(
        repo_root / "src" / "resources" / "icon.png"
    )

    def git(root, *args):
        assert root == repo_root
        if not git_available:
            raise FileNotFoundError("git")
        if args[-1] == "--show-toplevel":
            return str(repo_root)
        if args[-1] == "HEAD":
            return "a" * 40
        return " M private-filename"

    monkeypatch.setattr(build_info, "_git", git)
    monkeypatch.chdir(tmp_path)

    prepare_windows_build.prepare_build(repo_root, output_dir)

    assert (output_dir / "murmur.ico").is_file()
    version_info = (output_dir / "version_info.txt").read_text(encoding="utf-8")
    assert "filevers=(1, 2, 3, 0)" in version_info
    assert "StringStruct(u'ProductName', u'murmur')" in version_info
    assert json.loads((output_dir / "_build_info.json").read_text("utf-8")) == {
        "app_version": "1.2.3",
        "git_commit": "a" * 40 if git_available else None,
        "git_dirty": True if git_available else None,
    }
