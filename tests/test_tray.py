import builtins
import importlib
import sys
from types import SimpleNamespace

import src.tray as tray_module
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


def test_on_settings_ignores_repeat_clicks_while_window_thread_is_alive(monkeypatch):
    created_threads = []

    class FakeThread:
        def __init__(self, target, daemon):
            self.target = target
            self.daemon = daemon
            self._alive = False
            created_threads.append(self)

        def start(self):
            self._alive = True

        def is_alive(self):
            return self._alive

    monkeypatch.setitem(
        sys.modules, "src.settings_gui", SimpleNamespace(show_settings=lambda: None)
    )
    monkeypatch.setattr(tray_module.threading, "Thread", FakeThread)

    tray = TrayManager()
    tray._on_settings()
    tray._on_settings()

    assert len(created_threads) == 1
    assert created_threads[0].daemon is True


def test_on_settings_allows_reopening_after_previous_window_exits(monkeypatch):
    created_threads = []

    class FakeThread:
        def __init__(self, target, daemon):
            self.target = target
            self.daemon = daemon
            self._alive = False
            created_threads.append(self)

        def start(self):
            self._alive = True

        def is_alive(self):
            return self._alive

    monkeypatch.setitem(
        sys.modules, "src.settings_gui", SimpleNamespace(show_settings=lambda: None)
    )
    monkeypatch.setattr(tray_module.threading, "Thread", FakeThread)

    tray = TrayManager()
    tray._on_settings()
    created_threads[0]._alive = False
    tray._on_settings()

    assert len(created_threads) == 2
