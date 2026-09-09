"""Shared asset path helpers."""

import sys
from pathlib import Path

from PIL import Image

from .config import get_app_data_dir

PROJECT_ROOT = Path(__file__).resolve().parent.parent
APP_ICON_FILENAME = "murmur.ico"


def _get_resource_root() -> Path:
    """Return the source or frozen-bundle directory containing app assets."""
    bundle_root = getattr(sys, "_MEIPASS", None)
    if bundle_root is not None:
        return Path(bundle_root)
    return PROJECT_ROOT


def get_logo_path() -> Path | None:
    """Return the preferred app logo path when it exists."""
    for filename in ("murmur tray logo.png", "murmur.png"):
        path = _get_resource_root() / filename
        if path.exists():
            return path
    return None


def get_app_icon_path() -> Path | None:
    """Return a Windows .ico path generated from the app logo."""
    logo_path = get_logo_path()
    if logo_path is None:
        return None

    icon_dir = get_app_data_dir()
    icon_path = icon_dir / APP_ICON_FILENAME
    try:
        if (
            icon_path.exists()
            and icon_path.stat().st_mtime >= logo_path.stat().st_mtime
        ):
            return icon_path

        icon_dir.mkdir(parents=True, exist_ok=True)
        with Image.open(logo_path) as image:
            image.save(
                icon_path,
                format="ICO",
                sizes=[(16, 16), (24, 24), (32, 32), (48, 48), (64, 64), (256, 256)],
            )
        return icon_path
    except Exception:
        return None
