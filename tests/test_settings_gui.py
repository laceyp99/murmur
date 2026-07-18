from types import SimpleNamespace
import queue

import pytest

from src import settings_gui as settings_module
from src.config import DEFAULT_CONFIG


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

    def get(self, key, default=None):
        return getattr(self, key, DEFAULT_CONFIG.get(key, default))


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
        self.summary = SimpleNamespace(file_count=0, total_bytes=0)
        self.purge_calls = 0

    def set_enabled(self, enabled):
        self.enabled_calls.append(enabled)

    def get_storage_summary(self):
        return self.summary

    def purge_all(self):
        self.purge_calls += 1
        return self.summary.file_count


def test_settings_window_service_focuses_existing_window_and_recreates_closed(
    monkeypatch,
):
    created_windows = []

    class FakeWindow:
        def __init__(self, master, on_close):
            self.master = master
            self.on_close = on_close
            self.open = True
            self.focus_calls = 0
            created_windows.append(self)

        def is_open(self):
            return self.open

        def focus(self):
            self.focus_calls += 1

        def close(self):
            self.open = False
            self.on_close(self)

    monkeypatch.setattr(settings_module, "SettingsWindow", FakeWindow)

    master = object()
    service = settings_module._SettingsWindowService(master)

    service.show()
    service.show()
    created_windows[0].close()
    service.show()

    assert len(created_windows) == 2
    assert created_windows[0].master is master
    assert created_windows[0].focus_calls == 2
    assert created_windows[1].focus_calls == 1


def test_apply_window_icon_reuses_loaded_resources_across_retries(monkeypatch):
    icon_path = "C:/fake/icon.ico"
    logo_path = "C:/fake/logo.png"
    after_callbacks = []
    iconbitmap_calls = []
    iconphoto_calls = []
    load_image_calls = []
    photoimage_calls = []

    class FakeCFunc:
        def __init__(self, func):
            self._func = func

        def __call__(self, *args, **kwargs):
            return self._func(*args, **kwargs)

    class FakeUser32:
        def __init__(self):
            self.GetParent = FakeCFunc(lambda hwnd: 0)
            self.LoadImageW = FakeCFunc(self._load_image)
            self.SendMessageW = FakeCFunc(lambda *args: 0)
            self.SetClassLongPtrW = FakeCFunc(lambda *args: 0)
            self.SetClassLongW = FakeCFunc(lambda *args: 0)

        def _load_image(self, _hinstance, _path, _image_type, width, height, _flags):
            load_image_calls.append((width, height))
            return 1000 + len(load_image_calls)

    class FakeWindow:
        def winfo_id(self):
            return 123

        def iconbitmap(self, default):
            iconbitmap_calls.append(default)

        def iconphoto(self, *_args):
            iconphoto_calls.append(_args)

        def after(self, _delay, callback):
            after_callbacks.append(callback)

    monkeypatch.setattr(settings_module, "get_app_icon_path", lambda: icon_path)
    monkeypatch.setattr(settings_module, "get_logo_path", lambda: logo_path)
    monkeypatch.setattr(
        settings_module.tk,
        "PhotoImage",
        lambda file: photoimage_calls.append(file) or SimpleNamespace(file=file),
    )
    monkeypatch.setattr(
        settings_module.ctypes,
        "windll",
        SimpleNamespace(user32=FakeUser32()),
        raising=False,
    )

    window = FakeWindow()
    result = settings_module._apply_window_icon(window)

    for callback in after_callbacks:
        callback()

    assert load_image_calls == [(16, 16), (32, 32)]
    assert photoimage_calls == [logo_path]
    assert len(iconbitmap_calls) == 5
    assert len(iconphoto_calls) == 5
    assert result["native"] == [1001, 1002]
    assert len(result["tk"]) == 1


def test_save_stamps_logging_consent_and_enables_logger(monkeypatch):
    config = FakeConfig()
    config.device = "cuda"
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
    assert "Device" in info_calls[0][0][1]


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


def test_save_rejects_non_numeric_setting_and_restores_persisted_value(monkeypatch):
    config = FakeConfig()
    config.max_recording_duration = 600
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
    assert window.numeric_vars["max_recording_duration"].get() == "600"
    assert len(error_calls) == 1


def test_save_rejects_non_finite_numeric_setting(monkeypatch):
    config = FakeConfig()
    config.max_recording_duration = 450
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
    window.numeric_vars = make_numeric_vars(max_recording_duration="inf")
    window.root = SimpleNamespace(destroy=lambda: destroy_calls.append(True))

    window._save()

    assert config.set_calls == []
    assert logger.enabled_calls == []
    assert destroy_calls == []
    assert window.numeric_vars["max_recording_duration"].get() == "450"
    assert len(error_calls) == 1


@pytest.mark.parametrize(
    ("endpoint", "model_name"),
    [
        ("", "granite4.1:3b"),
        ("http://localhost:11434", ""),
    ],
)
def test_save_rejects_blank_ollama_settings_when_cleanup_enabled(
    monkeypatch, endpoint, model_name
):
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
    window.ollama_endpoint_var = FakeValue(endpoint)
    window.ollama_model_name_var = FakeValue(model_name)
    window.ollama_preload_model_var = FakeValue(True)
    window.numeric_vars = make_numeric_vars()
    window.root = SimpleNamespace(destroy=lambda: destroy_calls.append(True))

    window._save()

    assert config.set_calls == []
    assert logger.enabled_calls == []
    assert destroy_calls == []
    assert len(error_calls) == 1
    assert "Ollama endpoint and model" in error_calls[0][0][1]


def test_changed_restart_settings_warns_when_save_clamps_existing_config():
    config = FakeConfig()
    config.max_recording_duration = 5000
    window = settings_module.SettingsWindow.__new__(settings_module.SettingsWindow)
    window.config = config

    changed_settings = window._changed_restart_settings(
        {"max_recording_duration": 1800}
    )

    assert changed_settings == ["Maximum recording duration"]


def test_changed_restart_settings_warns_when_existing_numeric_config_is_invalid():
    config = FakeConfig()
    config.vad_aggressiveness = "not-a-number"
    window = settings_module.SettingsWindow.__new__(settings_module.SettingsWindow)
    window.config = config

    changed_settings = window._changed_restart_settings({"vad_aggressiveness": 1})

    assert changed_settings == ["VAD aggressiveness"]


def test_slider_number_of_steps_uses_numeric_setting_step():
    vad_padding = settings_module.SETTINGS_BY_KEY["vad_padding_ms"]

    assert settings_module._slider_number_of_steps(vad_padding) == 75


def test_reset_tab_to_defaults_preserves_other_tab_edits():
    window = settings_module.SettingsWindow.__new__(settings_module.SettingsWindow)
    hotkey_var = FakeValue("ctrl+alt+p")
    vad_var = FakeValue("800")
    model_var = FakeValue("large")
    window.setting_vars = {
        "hotkey": hotkey_var,
        "vad_padding_ms": vad_var,
        "model": model_var,
    }

    window._reset_tab_to_defaults("VAD")

    assert hotkey_var.get() == "ctrl+alt+p"
    assert vad_var.get() == "220"
    assert model_var.get() == "large"


def test_ollama_connection_test_uses_unsaved_values(monkeypatch):
    checks = []
    info_calls = []
    after_callbacks = []
    thread_starts = []

    class FakeButton:
        def __init__(self):
            self.configure_calls = []

        def configure(self, **kwargs):
            self.configure_calls.append(kwargs)

    class FakeRoot:
        def after(self, _delay, callback):
            after_callbacks.append(callback)

    class FakeThread:
        def __init__(self, target, daemon):
            self.target = target
            self.daemon = daemon

        def start(self):
            thread_starts.append(self)

    monkeypatch.setattr(
        settings_module,
        "threading",
        SimpleNamespace(
            Thread=FakeThread,
            current_thread=settings_module.threading.current_thread,
        ),
    )
    monkeypatch.setattr(
        settings_module,
        "_ollama_connection_test_results",
        queue.Queue(),
    )
    monkeypatch.setattr(
        settings_module,
        "check_ollama_connection",
        lambda **kwargs: (
            checks.append(kwargs) or SimpleNamespace(ok=True, message="connected")
        ),
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
    window.root = FakeRoot()
    window._ollama_test_button = FakeButton()
    window.ollama_endpoint_var = FakeValue("http://127.0.0.1:11434")
    window.ollama_model_name_var = FakeValue("qwen:latest")
    window.numeric_vars = make_numeric_vars(ollama_timeout_seconds="300")

    window._test_ollama_connection()

    assert len(thread_starts) == 1
    assert checks == []
    assert window._ollama_test_button.configure_calls[0] == {
        "state": "disabled",
        "text": "Testing…",
    }

    thread_starts[0].target()

    assert checks == [
        {
            "endpoint": "http://127.0.0.1:11434",
            "model_name": "qwen:latest",
            "timeout": 300,
        }
    ]
    assert len(after_callbacks) == 1
    assert info_calls == []

    after_callbacks[0]()

    assert len(info_calls) == 1
    assert window._ollama_test_button.configure_calls[-1] == {
        "state": "normal",
        "text": "Test Ollama Connection",
    }


def test_purge_training_data_confirms_count_and_size(monkeypatch):
    logger = FakeLogger()
    logger.summary = SimpleNamespace(file_count=2, total_bytes=1536)
    confirm_messages = []
    info_calls = []

    monkeypatch.setattr(
        settings_module,
        "messagebox",
        SimpleNamespace(
            askyesno=lambda title, message: confirm_messages.append(message) or True,
            showinfo=lambda *args, **kwargs: info_calls.append((args, kwargs)),
            showerror=lambda *args, **kwargs: None,
        ),
    )

    window = settings_module.SettingsWindow.__new__(settings_module.SettingsWindow)
    window.logger = logger

    window._purge_training_data()

    assert logger.purge_calls == 1
    assert "Files: 2" in confirm_messages[0]
    assert "Approximate size: 1.5 KB" in confirm_messages[0]
    assert len(info_calls) == 1
