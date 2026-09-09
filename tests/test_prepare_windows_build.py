import importlib.util
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


def test_prepare_build_generates_icon_and_version_info(tmp_path):
    repo_root = tmp_path / "repo"
    output_dir = repo_root / "build" / "windows"
    repo_root.mkdir()
    (repo_root / "pyproject.toml").write_text(
        '[project]\nversion = "1.2.3"\n', encoding="utf-8"
    )

    from PIL import Image

    Image.new("RGBA", (256, 256), color=(73, 109, 137, 255)).save(
        repo_root / "murmur tray logo.png"
    )

    prepare_windows_build.prepare_build(repo_root, output_dir)

    assert (output_dir / "murmur.ico").is_file()
    version_info = (output_dir / "version_info.txt").read_text(encoding="utf-8")
    assert "filevers=(1, 2, 3, 0)" in version_info
    assert "StringStruct(u'ProductName', u'Murmur')" in version_info
