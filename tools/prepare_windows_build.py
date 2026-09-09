"""Generate deterministic icon and version inputs for the Windows build."""

import argparse
import re
import tomllib
from pathlib import Path

from PIL import Image

ICON_SIZES = (16, 24, 32, 48, 64, 128, 256)


def _read_project_version(pyproject_path: Path) -> str:
    with pyproject_path.open("rb") as handle:
        project = tomllib.load(handle)["project"]
    return str(project["version"])


def _windows_version(version: str) -> tuple[int, int, int, int]:
    match = re.fullmatch(r"(\d+)\.(\d+)\.(\d+)(?:\.(\d+))?", version)
    if match is None:
        raise ValueError(
            f"Project version {version!r} must contain three or four integers"
        )
    parts = [int(part) if part is not None else 0 for part in match.groups()]
    if any(part > 65535 for part in parts):
        raise ValueError("Windows version components must not exceed 65535")
    return tuple(parts)


def _write_icon(source: Path, destination: Path) -> None:
    with Image.open(source) as image:
        image.save(
            destination,
            format="ICO",
            sizes=[(size, size) for size in ICON_SIZES],
        )


def _write_version_info(destination: Path, version: str) -> None:
    version_tuple = _windows_version(version)
    destination.write_text(
        f"""VSVersionInfo(
  ffi=FixedFileInfo(
    filevers={version_tuple},
    prodvers={version_tuple},
    mask=0x3f,
    flags=0x0,
    OS=0x40004,
    fileType=0x1,
    subtype=0x0,
    date=(0, 0)
  ),
  kids=[
    StringFileInfo([
      StringTable(
        u'040904B0',
        [
          StringStruct(u'CompanyName', u'laceyp99'),
          StringStruct(u'FileDescription', u'Murmur - Local Speech-to-Text Hotkey App'),
          StringStruct(u'FileVersion', u'{version}'),
          StringStruct(u'InternalName', u'murmur'),
          StringStruct(u'LegalCopyright', u'Copyright (c) laceyp99'),
          StringStruct(u'OriginalFilename', u'murmur.exe'),
          StringStruct(u'ProductName', u'Murmur'),
          StringStruct(u'ProductVersion', u'{version}')
        ]
      )
    ]),
    VarFileInfo([VarStruct(u'Translation', [1033, 1200])])
  ]
)
""",
        encoding="utf-8",
    )


def prepare_build(repo_root: Path, output_dir: Path) -> None:
    """Create the generated resources consumed by the PyInstaller spec."""
    logo_path = repo_root / "murmur tray logo.png"
    pyproject_path = repo_root / "pyproject.toml"
    for required_path in (logo_path, pyproject_path):
        if not required_path.is_file():
            raise FileNotFoundError(f"Required build input is missing: {required_path}")

    output_dir.mkdir(parents=True, exist_ok=True)
    version = _read_project_version(pyproject_path)
    _write_icon(logo_path, output_dir / "murmur.ico")
    _write_version_info(output_dir / "version_info.txt", version)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--repo-root", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    args = parser.parse_args()
    prepare_build(args.repo_root.resolve(), args.output_dir.resolve())


if __name__ == "__main__":
    main()
