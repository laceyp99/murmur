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

    def update(self, values):
        for key, value in values.items():
            self.set(key, value)


def make_numeric_vars(**overrides):
    values = {
        "vad_aggressiveness": "1",
        "vad_padding_ms": "220",
        "vad_silence_duration_ms": "400",
        "max_recording_duration": "300",
        "ollama_timeout_seconds": "60",
    }
    values.update(overrides)
    return {key: FakeValue(value) for key, value in values.items()}


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
    monkeypatch.setattr(settings_module, "is_hotkey_valid", lambda value: True)
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
            showerror=lambda *args, **kwargs: None,
        ),
    )
    monkeypatch.setattr(
        settings_module,
        "datetime",
        SimpleNamespace(
            now=lambda: SimpleNamespace(isoformat=lambda: "2026-05-25T12:00:00")
        ),
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
    window.ollama_enabled_var = FakeValue(True)
    window.ollama_endpoint_var = FakeValue(" http://localhost:11434 ")
    window.ollama_model_name_var = FakeValue("granite4.1:3b")
    window.ollama_preload_model_var = FakeValue(False)
    window.numeric_vars = make_numeric_vars(
        vad_aggressiveness="99",
        ollama_timeout_seconds="30",
    )
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
        ("ollama_enabled", True),
        ("ollama_endpoint", "http://localhost:11434"),
        ("ollama_model_name", "granite4.1:3b"),
        ("ollama_preload_model", False),
        ("vad_aggressiveness", 3),
        ("vad_padding_ms", 220),
        ("vad_silence_duration_ms", 400),
        ("max_recording_duration", 300),
        ("ollama_timeout_seconds", 60),
        ("logging_consent_updated_at", "2026-05-25T12:00:00"),
        ("logging_consent_source", "settings"),
        ("start_with_windows", True),
    ]
    assert logger.enabled_calls == [True]
    assert set_autostart_calls == [True]
    assert len(info_calls) == 1


def test_save_rejects_invalid_hotkey_without_persisting_changes(monkeypatch):
    config = FakeConfig()
    logger = FakeLogger()
    error_calls = []
    info_calls = []
    set_autostart_calls = []
    destroy_calls = []

    monkeypatch.setattr(settings_module, "get_config", lambda: config)
    monkeypatch.setattr(settings_module, "get_logger", lambda: logger)
    monkeypatch.setattr(settings_module, "is_hotkey_valid", lambda value: False)
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
            showerror=lambda *args, **kwargs: error_calls.append((args, kwargs)),
        ),
    )

    window = settings_module.SettingsWindow.__new__(settings_module.SettingsWindow)
    window.config = config
    window.logger = logger
    window.hotkey_var = FakeValue("not-a-real-key")
    window.model_var = FakeValue("small")
    window.device_var = FakeValue("cpu")
    window.lang_var = FakeValue("")
    window.notify_var = FakeValue(True)
    window.logging_var = FakeValue(False)
    window.pause_media_var = FakeValue(True)
    window.autostart_var = FakeValue(False)
    window.ollama_enabled_var = FakeValue(True)
    window.ollama_endpoint_var = FakeValue("http://localhost:11434")
    window.ollama_model_name_var = FakeValue("granite4.1:3b")
    window.ollama_preload_model_var = FakeValue(True)
    window.numeric_vars = make_numeric_vars()
    window.root = SimpleNamespace(destroy=lambda: destroy_calls.append(True))

    window._save()

    assert config.set_calls == []
    assert logger.enabled_calls == []
    assert set_autostart_calls == []
    assert destroy_calls == []
    assert len(info_calls) == 0
    assert len(error_calls) == 1


def test_save_rejects_non_numeric_setting_and_resets_field(monkeypatch):
    config = FakeConfig()
    logger = FakeLogger()
    error_calls = []
    destroy_calls = []

    monkeypatch.setattr(settings_module, "is_hotkey_valid", lambda value: True)
    monkeypatch.setattr(
        settings_module,
        "messagebox",
        SimpleNamespace(
            askyesno=lambda *args, **kwargs: True,
            showinfo=lambda *args, **kwargs: None,
            showerror=lambda *args, **kwargs: error_calls.append((args, kwargs)),
        ),
    )

    window = settings_module.SettingsWindow.__new__(settings_module.SettingsWindow)
    window.config = config
    window.logger = logger
    window.hotkey_var = FakeValue("ctrl+alt+space")
    window.model_var = FakeValue("small")
    window.device_var = FakeValue("cpu")
    window.lang_var = FakeValue("")
    window.notify_var = FakeValue(True)
    window.logging_var = FakeValue(False)
    window.pause_media_var = FakeValue(True)
    window.autostart_var = FakeValue(False)
    window.ollama_enabled_var = FakeValue(True)
    window.ollama_endpoint_var = FakeValue("http://localhost:11434")
    window.ollama_model_name_var = FakeValue("granite4.1:3b")
    window.ollama_preload_model_var = FakeValue(True)
    window.numeric_vars = make_numeric_vars(max_recording_duration="abc")
    window.root = SimpleNamespace(destroy=lambda: destroy_calls.append(True))

    window._save()

    assert config.set_calls == []
    assert logger.enabled_calls == []
    assert destroy_calls == []
    assert window.numeric_vars["max_recording_duration"].get() == "300"
    assert len(error_calls) == 1


def test_ollama_connection_test_uses_unsaved_values(monkeypatch):
    checks = []
    info_calls = []

    monkeypatch.setattr(
        settings_module,
        "check_ollama_connection",
        lambda **kwargs: checks.append(kwargs)
        or SimpleNamespace(ok=True, message="connected"),
    )
    monkeypatch.setattr(
        settings_module,
        "messagebox",
        SimpleNamespace(
            askyesno=lambda *args, **kwargs: True,
            showinfo=lambda *args, **kwargs: info_calls.append((args, kwargs)),
            showerror=lambda *args, **kwargs: None,
        ),
    )

    window = settings_module.SettingsWindow.__new__(settings_module.SettingsWindow)
    window.ollama_endpoint_var = FakeValue("http://127.0.0.1:11434")
    window.ollama_model_name_var = FakeValue("qwen:latest")
    window.numeric_vars = make_numeric_vars(ollama_timeout_seconds="300")

    window._test_ollama_connection()

    assert checks == [
        {
            "endpoint": "http://127.0.0.1:11434",
            "model_name": "qwen:latest",
            "timeout": 5,
        }
    ]
    assert len(info_calls) == 1
