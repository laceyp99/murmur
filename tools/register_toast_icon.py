"""Register or remove murmur's Windows toast notification icon on demand.

Windows labels murmur's toasts by its AppUserModelID and reads the display
name and icon from the current user's registry. Run this after changing
app/toast-icon.png; murmur itself never writes this registry key.

The toast icon is separate from app/icon.png because Windows inverts mostly
dark toast icons in dark mode, turning the black app tile white. It is also a
small, thicker-stroked waveform because Windows downscales large icons to about
20px with a filter that drops thin lines.
"""

import argparse
import contextlib
import hashlib
import os
import winreg
from pathlib import Path

# Must match _WINDOWS_APP_USER_MODEL_ID in src/windows_identity.py.
APP_USER_MODEL_ID = "murmur"
DISPLAY_NAME = "murmur"
TOAST_ICON_PREFIX = "toast-icon"
REGISTRY_KEY = rf"Software\Classes\AppUserModelId\{APP_USER_MODEL_ID}"


def _app_data_dir() -> Path:
    return Path(os.environ["APPDATA"]) / "murmur"


def register(repo_root: Path, app_data_dir: Path) -> Path:
    """Copy the toast icon to AppData and point murmur's toast identity at it."""
    source_path = repo_root / "app" / "toast-icon.png"
    if not source_path.is_file():
        raise FileNotFoundError(f"Toast icon is missing: {source_path}")

    # Windows caches toast icons by path, so a changed icon needs a new name.
    icon_bytes = source_path.read_bytes()
    digest = hashlib.sha256(icon_bytes).hexdigest()[:12]
    icon_path = app_data_dir / f"{TOAST_ICON_PREFIX}-{digest}.png"
    app_data_dir.mkdir(parents=True, exist_ok=True)
    icon_path.write_bytes(icon_bytes)
    _remove_icon_copies(app_data_dir, keep=icon_path)

    with winreg.CreateKey(winreg.HKEY_CURRENT_USER, REGISTRY_KEY) as key:
        winreg.SetValueEx(key, "DisplayName", 0, winreg.REG_SZ, DISPLAY_NAME)
        winreg.SetValueEx(key, "IconUri", 0, winreg.REG_SZ, str(icon_path))
    return icon_path


def remove(app_data_dir: Path) -> None:
    """Delete murmur's toast identity key and copied icons if present."""
    with contextlib.suppress(FileNotFoundError):
        winreg.DeleteKey(winreg.HKEY_CURRENT_USER, REGISTRY_KEY)
    _remove_icon_copies(app_data_dir)


def _remove_icon_copies(app_data_dir: Path, keep: Path | None = None) -> None:
    for path in app_data_dir.glob(f"{TOAST_ICON_PREFIX}*.png"):
        if path != keep:
            path.unlink(missing_ok=True)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument(
        "--remove",
        action="store_true",
        help="delete the registry key and copied icon instead of registering",
    )
    args = parser.parse_args()

    app_data_dir = _app_data_dir()
    if args.remove:
        remove(app_data_dir)
        print(f"Removed HKCU\\{REGISTRY_KEY}")
        return

    icon_path = register(Path(__file__).resolve().parent.parent, app_data_dir)
    print(f"Registered HKCU\\{REGISTRY_KEY} with icon {icon_path}")


if __name__ == "__main__":
    main()
