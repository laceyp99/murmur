"""
Auto-start management for Murmur on Windows.
"""

import os
import sys
import winreg
from pathlib import Path

def set_autostart(enabled: bool) -> bool:
    """
    Enable or disable auto-start for Murmur in the Windows Registry.
    
    Args:
        enabled: Whether to enable auto-start.
        
    Returns:
        True if successful, False otherwise.
    """
    app_name = "Murmur"
    # Get the path to the pythonw.exe inside the venv
    # If running from venv, sys.executable will be ...\venv\Scripts\python.exe
    # We want to ensure we use pythonw.exe for background running
    venv_python = sys.executable.lower().replace("python.exe", "pythonw.exe")
    script_path = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "run.py"))
    
    cmd = f'"{venv_python}" "{script_path}"'
    
    key = winreg.HKEY_CURRENT_USER
    key_path = r"Software\Microsoft\Windows\CurrentVersion\Run"
    
    try:
        reg_key = winreg.OpenKey(key, key_path, 0, winreg.KEY_SET_VALUE)
        if enabled:
            winreg.SetValueEx(reg_key, app_name, 0, winreg.REG_SZ, cmd)
        else:
            try:
                winreg.DeleteValue(reg_key, app_name)
            except FileNotFoundError:
                pass
        winreg.CloseKey(reg_key)
        return True
    except Exception as e:
        print(f"Failed to update autostart registry: {e}")
        return False

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
