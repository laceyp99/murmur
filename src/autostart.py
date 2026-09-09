"""
Auto-start management for Murmur on Windows.
"""

import contextlib
import sys
from pathlib import Path

try:
    import winreg
except ImportError:
    winreg = None


def _get_launch_command() -> str:
    """Return the current Murmur launch command for Windows startup."""
    executable = Path(sys.executable)
    if getattr(sys, "frozen", False):
        return f'"{executable}"'

    if executable.name.lower() == "python.exe":
        executable = executable.with_name("pythonw.exe")

    script_path = Path(__file__).resolve().parent.parent / "run.py"
    return f'"{executable}" "{script_path}"'


def set_autostart(enabled: bool):
    """
    Enable or disable auto-start with Windows.
    Uses the packaged executable when frozen, otherwise the current Python
    environment's windowless interpreter and ``run.py``.
    """
    if winreg is None:
        return

    app_name = "Murmur"

    cmd = _get_launch_command()

    key_path = r"Software\Microsoft\Windows\CurrentVersion\Run"

    try:
        reg_key = winreg.OpenKey(
            winreg.HKEY_CURRENT_USER, key_path, 0, winreg.KEY_SET_VALUE
        )
        if enabled:
            winreg.SetValueEx(reg_key, app_name, 0, winreg.REG_SZ, cmd)
        else:
            with contextlib.suppress(FileNotFoundError):
                winreg.DeleteValue(reg_key, app_name)
        winreg.CloseKey(reg_key)
    except Exception as e:
        print(f"Failed to update autostart registry: {e}")


def is_autostart_enabled() -> bool:
    """Check if auto-start is currently enabled in the registry."""
    if winreg is None:
        return False

    key_path = r"Software\Microsoft\Windows\CurrentVersion\Run"
    app_name = "Murmur"

    try:
        key = winreg.OpenKey(winreg.HKEY_CURRENT_USER, key_path, 0, winreg.KEY_READ)
        try:
            winreg.QueryValueEx(key, app_name)
            enabled = True
        except FileNotFoundError:
            enabled = False
        winreg.CloseKey(key)
        return enabled
    except Exception:
        return False
