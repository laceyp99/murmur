import builtins
import importlib
import sys
from types import SimpleNamespace

from src.tray import TrayManager


def test_src_main_imports_without_pystray_backend(monkeypatch):
    original_import = builtins.__import__

    def fake_import(name, global_vars=None, local_vars=None, fromlist=(), level=0):
        if name == "pystray" or name.startswith("pystray."):
            raise OSError('Bad display name ""')
        return original_import(name, global_vars, local_vars, fromlist, level)

    for module_name in list(sys.modules):
        if module_name.startswith(("src.main", "src.tray", "pystray", "Xlib")):
            sys.modules.pop(module_name, None)

    monkeypatch.setattr(builtins, "__import__", fake_import)

    module = importlib.import_module("src.main")

    assert module.MurmurApp is not None


def test_tray_run_noops_when_pystray_import_fails(monkeypatch):
    original_import = builtins.__import__

    def fake_import(name, global_vars=None, local_vars=None, fromlist=(), level=0):
        if name == "pystray" or name.startswith("pystray."):
            raise OSError('Bad display name ""')
        return original_import(name, global_vars, local_vars, fromlist, level)

    monkeypatch.setattr(builtins, "__import__", fake_import)

    tray = TrayManager()
    tray.run()

    assert tray.icon is None


def test_on_settings_delegates_to_settings_gui(monkeypatch):
    show_calls = []
    monkeypatch.setitem(
        sys.modules,
        "src.settings_gui",
        SimpleNamespace(show_settings=lambda: show_calls.append(True)),
    )

    tray = TrayManager()
    tray._on_settings()
    tray._on_settings()

    assert show_calls == [True, True]
