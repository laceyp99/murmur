"""Shared asset path helpers."""

import sys
from importlib.resources import files
from pathlib import Path

from PIL import Image

from .config import get_app_data_dir

APP_ICON_FILENAME = "murmur.ico"


def get_logo_path() -> Path | None:
    """Return the logo from an unpacked Python package or frozen bundle."""
    bundle_root = getattr(sys, "_MEIPASS", None)
    if bundle_root is not None:
        path = Path(bundle_root) / "app" / "icon.png"
    else:
        path = Path(str(files("src").joinpath("resources", "icon.png")))
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
