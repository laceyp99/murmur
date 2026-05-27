import builtins
import importlib
import sys

from src.tray import TrayManager


def test_src_main_imports_without_pystray_backend(monkeypatch):
    original_import = builtins.__import__

    def fake_import(name, globals=None, locals=None, fromlist=(), level=0):
        if name == "pystray" or name.startswith("pystray."):
            raise OSError('Bad display name ""')
        return original_import(name, globals, locals, fromlist, level)

    for module_name in list(sys.modules):
        if (
            module_name == "src.main"
            or module_name.startswith("src.main")
            or module_name == "src.tray"
            or module_name.startswith("src.tray")
            or module_name == "pystray"
            or module_name.startswith("pystray")
            or module_name == "Xlib"
            or module_name.startswith("Xlib")
        ):
            sys.modules.pop(module_name, None)

    monkeypatch.setattr(builtins, "__import__", fake_import)

    module = importlib.import_module("src.main")

    assert module.MurmurApp is not None


def test_tray_run_noops_when_pystray_import_fails(monkeypatch):
    original_import = builtins.__import__

    def fake_import(name, globals=None, locals=None, fromlist=(), level=0):
        if name == "pystray" or name.startswith("pystray."):
            raise OSError('Bad display name ""')
        return original_import(name, globals, locals, fromlist, level)

    monkeypatch.setattr(builtins, "__import__", fake_import)

    tray = TrayManager()
    tray.run()

    assert tray.icon is None
