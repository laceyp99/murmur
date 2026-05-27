from types import SimpleNamespace
import sys

import pytest

from src import autostart as autostart_module


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
    monkeypatch.setattr(autostart_module.sys, "executable", r"C:\Python312\python.exe")
    monkeypatch.setattr(
        autostart_module.os.path,
        "dirname",
        lambda _: r"C:\Users\Patrick\Desktop\PROJECTS\murmur\src",
    )
    monkeypatch.setattr(
        autostart_module.os.path,
        "abspath",
        lambda _: r"C:\Users\Patrick\Desktop\PROJECTS\murmur\run.py",
    )

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
        '"c:\\python312\\pythonw.exe" "C:\\Users\\Patrick\\Desktop\\PROJECTS\\murmur\\run.py"',
    )
    assert captured["close_key"] is not None
    assert "delete_value" not in captured
