"""
Auto-start management for murmur on Windows.
"""

import contextlib
import sys
from pathlib import Path

try:
    import winreg
except ImportError:
    winreg = None


def _get_launch_command() -> str:
    """Return the current murmur launch command for Windows startup."""
    executable = Path(sys.executable)
    if getattr(sys, "frozen", False):
        return f'"{executable}"'

    if executable.name.lower() == "python.exe":
        executable = executable.with_name("pythonw.exe")

    return f'"{executable}" -m src'


def set_autostart(enabled: bool) -> None:
    """
    Enable or disable auto-start with Windows.
    Uses the packaged executable when frozen, otherwise the current Python
    environment's windowless interpreter and installed ``src`` module.
    Registry failures propagate so callers can report an unsuccessful change.
    """
    if winreg is None:
        return

    app_name = "murmur"

    cmd = _get_launch_command()

    key_path = r"Software\Microsoft\Windows\CurrentVersion\Run"

    try:
        reg_key = winreg.OpenKey(
            winreg.HKEY_CURRENT_USER, key_path, 0, winreg.KEY_SET_VALUE
        )
    except FileNotFoundError:
        if not enabled:
            return
        raise

    with reg_key:
        if enabled:
            winreg.SetValueEx(reg_key, app_name, 0, winreg.REG_SZ, cmd)
        else:
            with contextlib.suppress(FileNotFoundError):
                winreg.DeleteValue(reg_key, app_name)


def is_autostart_enabled() -> bool:
    """Check auto-start, propagating registry failures other than missing entries."""
    if winreg is None:
        return False

    key_path = r"Software\Microsoft\Windows\CurrentVersion\Run"
    app_name = "murmur"

    try:
        with winreg.OpenKey(
            winreg.HKEY_CURRENT_USER, key_path, 0, winreg.KEY_READ
        ) as key:
            winreg.QueryValueEx(key, app_name)
        return True
    except FileNotFoundError:
        return False
