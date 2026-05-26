from types import SimpleNamespace

import pytest


settings_module = pytest.importorskip("src.settings_gui")


class FakeValue:
    def __init__(self, value):
        self._value = value

    def get(self):
        return self._value

    def set(self, value):
        self._value = value


class FakeConfig:
    def __init__(self):
        self.enable_logging = False
        self.start_with_windows = False
        self.set_calls = []

    def set(self, key, value):
        self.set_calls.append((key, value))
        setattr(self, key, value)


class FakeLogger:
    def __init__(self):
        self.enabled_calls = []

    def set_enabled(self, enabled):
        self.enabled_calls.append(enabled)


def test_save_stamps_logging_consent_and_enables_logger(monkeypatch):
    config = FakeConfig()
    logger = FakeLogger()
    set_autostart_calls = []
    info_calls = []

    monkeypatch.setattr(settings_module, "get_config", lambda: config)
    monkeypatch.setattr(settings_module, "get_logger", lambda: logger)
    monkeypatch.setattr(
        settings_module,
        "set_autostart",
        lambda enabled: set_autostart_calls.append(enabled),
    )
    monkeypatch.setattr(
        settings_module,
        "messagebox",
        SimpleNamespace(
            askyesno=lambda *args, **kwargs: True,
            showinfo=lambda *args, **kwargs: info_calls.append((args, kwargs)),
        ),
    )
    monkeypatch.setattr(
        settings_module,
        "datetime",
        SimpleNamespace(now=lambda: SimpleNamespace(isoformat=lambda: "2026-05-25T12:00:00")),
    )

    window = settings_module.SettingsWindow.__new__(settings_module.SettingsWindow)
    window.config = config
    window.logger = logger
    window.hotkey_var = FakeValue("ctrl+alt+space")
    window.model_var = FakeValue("small")
    window.device_var = FakeValue("cpu")
    window.lang_var = FakeValue("")
    window.notify_var = FakeValue(True)
    window.logging_var = FakeValue(True)
    window.pause_media_var = FakeValue(False)
    window.autostart_var = FakeValue(True)
    window.root = SimpleNamespace(destroy=lambda: None)

    window._save()

    assert config.set_calls == [
        ("hotkey", "ctrl+alt+space"),
        ("model", "small"),
        ("device", "cpu"),
        ("language", None),
        ("enable_notifications", True),
        ("enable_logging", True),
        ("pause_media_while_recording", False),
        ("logging_consent_updated_at", "2026-05-25T12:00:00"),
        ("logging_consent_source", "settings"),
        ("start_with_windows", True),
    ]
    assert logger.enabled_calls == [True]
    assert set_autostart_calls == [True]
    assert len(info_calls) == 1