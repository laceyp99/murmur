from types import SimpleNamespace

from src import windows_identity


def test_configure_windows_app_identity_sets_id_once(monkeypatch):
    calls = []
    shell32 = SimpleNamespace(
        SetCurrentProcessExplicitAppUserModelID=lambda app_id: calls.append(app_id)
    )
    monkeypatch.setattr(
        windows_identity.ctypes,
        "windll",
        SimpleNamespace(shell32=shell32),
        raising=False,
    )
    monkeypatch.setattr(windows_identity, "_WINDOWS_APP_ID_SET", False)

    windows_identity.configure_windows_app_identity()
    windows_identity.configure_windows_app_identity()

    assert calls == ["murmur"]
