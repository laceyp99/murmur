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


def test_cancel_runs_on_background_thread_only_when_enabled(monkeypatch):
    from src import tray as tray_module

    calls = []
    enabled = [False]
    threads = []
    monkeypatch.setattr(
        tray_module.threading,
        "Thread",
        lambda **kwargs: SimpleNamespace(start=lambda: threads.append(kwargs)),
    )
    tray = TrayManager(
        on_cancel_callback=lambda session: calls.append(session),
        get_cancel_session_callback=lambda: 7 if enabled[0] else None,
    )
    tray._on_cancel()
    assert threads == []
    enabled[0] = True
    assert tray._can_cancel() is True
    tray._on_cancel()
    assert calls == []
    assert threads[0]["daemon"] is True
    threads[0]["target"](*threads[0]["args"])
    assert calls == [7]


def test_tray_background_cancel_failure_omits_exception_contents(capsys):
    def fail(_session):
        raise ValueError("private transcript")

    TrayManager(on_cancel_callback=fail)._run_cancel(7)
    error = capsys.readouterr().err
    assert "ValueError" in error
    assert "private transcript" not in error


def test_tray_cancel_captures_session_before_background_dispatch(monkeypatch):
    from src import tray as tray_module

    session = [7]
    threads = []
    calls = []
    monkeypatch.setattr(
        tray_module.threading,
        "Thread",
        lambda **kw: SimpleNamespace(start=lambda: threads.append(kw)),
    )
    tray = TrayManager(
        on_cancel_callback=calls.append,
        get_cancel_session_callback=lambda: session[0],
    )
    tray._on_cancel()
    session[0] = 8
    threads[0]["target"](*threads[0]["args"])
    assert calls == [7]
    session[0] = None
    tray._on_cancel()
    assert len(threads) == 1
