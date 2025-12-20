"""
Auto-start management for Murmur on Windows.
"""

import os
import sys
import winreg
from pathlib import Path

def set_autostart(enabled: bool):
    """
    Enable or disable auto-start with Windows.
    Uses sys.executable to ensure it uses the same environment that launched the app.
    """
    app_name = "Murmur"
    
    # sys.executable points to the current python.exe or pythonw.exe
    current_python = sys.executable
    
    # Ensure we use the windowless version for background startup
    if current_python.lower().endswith("python.exe"):
        python_exe = current_python.lower().replace("python.exe", "pythonw.exe")
    else:
        python_exe = current_python

    script_path = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "run.py"))
    cmd = f'"{python_exe}" "{script_path}"'
    
    key_path = r"Software\Microsoft\Windows\CurrentVersion\Run"
    
    try:
        reg_key = winreg.OpenKey(winreg.HKEY_CURRENT_USER, key_path, 0, winreg.KEY_SET_VALUE)
        if enabled:
            winreg.SetValueEx(reg_key, app_name, 0, winreg.REG_SZ, cmd)
        else:
            try:
                winreg.DeleteValue(reg_key, app_name)
            except FileNotFoundError:
                pass
        winreg.CloseKey(reg_key)
    except Exception as e:
        print(f"Failed to update autostart registry: {e}")

def is_autostart_enabled() -> bool:
    """Check if auto-start is currently enabled in the registry."""
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
