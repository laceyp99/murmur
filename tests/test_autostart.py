import sys
from types import SimpleNamespace

import pytest

from src import autostart as autostart_module


def test_launch_command_uses_pythonw_for_source_mode(tmp_path, monkeypatch):
    source_root = tmp_path / "murmur project"
    module_path = source_root / "src" / "autostart.py"
    python_path = tmp_path / "venv" / "Scripts" / "python.exe"
    monkeypatch.setattr(autostart_module.sys, "executable", str(python_path))
    monkeypatch.setattr(autostart_module.sys, "frozen", False, raising=False)
    monkeypatch.setattr(autostart_module, "__file__", str(module_path))

    assert autostart_module._get_launch_command() == (
        f'"{python_path.with_name("pythonw.exe")}" "{source_root / "run.py"}"'
    )


def test_launch_command_uses_packaged_executable(monkeypatch):
    monkeypatch.setattr(
        autostart_module.sys,
        "executable",
        r"C:\Program Files\Murmur\murmur.exe",
    )
    monkeypatch.setattr(autostart_module.sys, "frozen", True, raising=False)

    assert autostart_module._get_launch_command() == (
        '"C:\\Program Files\\Murmur\\murmur.exe"'
    )


def test_set_autostart_noops_when_winreg_is_unavailable(monkeypatch):
    monkeypatch.setattr(autostart_module, "winreg", None)

    autostart_module.set_autostart(True)

    assert autostart_module.is_autostart_enabled() is False


@pytest.mark.skipif(sys.platform != "win32", reason="Windows registry only")
def test_set_autostart_uses_pythonw_and_registry(monkeypatch):
    captured = {}

    class FakeRegistryKey:
        pass

    registry_key = FakeRegistryKey()

    def open_key(root, key_path, reserved, access):
        captured["open_key"] = (root, key_path, reserved, access)
        return registry_key

    def set_value_ex(reg_key, app_name, reserved, reg_type, cmd):
        captured["set_value_ex"] = (reg_key, app_name, reserved, reg_type, cmd)

    def delete_value(reg_key, app_name):
        captured["delete_value"] = (reg_key, app_name)

    def close_key(reg_key):
        captured["close_key"] = reg_key

    fake_winreg = SimpleNamespace(
        HKEY_CURRENT_USER=object(),
        KEY_SET_VALUE=object(),
        REG_SZ=1,
        OpenKey=open_key,
        SetValueEx=set_value_ex,
        DeleteValue=delete_value,
        CloseKey=close_key,
    )

    monkeypatch.setattr(autostart_module, "winreg", fake_winreg)
    monkeypatch.setattr(autostart_module, "_get_launch_command", lambda: "command")

    autostart_module.set_autostart(True)

    assert captured["open_key"] == (
        fake_winreg.HKEY_CURRENT_USER,
        r"Software\Microsoft\Windows\CurrentVersion\Run",
        0,
        fake_winreg.KEY_SET_VALUE,
    )
    assert captured["set_value_ex"] == (
        registry_key,
        "Murmur",
        0,
        fake_winreg.REG_SZ,
        "command",
    )
    assert captured["close_key"] is not None
    assert "delete_value" not in captured
