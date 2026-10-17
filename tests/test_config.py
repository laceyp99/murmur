import json
import math
import threading

import pytest

import src.config as config_module
import src.logger as logger_module
from src.audio import AudioRecorder
from src.config import (
    DEFAULT_CONFIG,
    DEFAULT_OLLAMA_MODEL_NAME,
    DEFAULT_OLLAMA_TIMEOUT_SECONDS,
    Config,
)
from src.logger import DataLogger


def test_config_defaults(tmp_path, monkeypatch):
    monkeypatch.setenv("APPDATA", str(tmp_path))
    cfg = Config()
    assert DEFAULT_OLLAMA_MODEL_NAME == "qwen3.5:4b"
    assert cfg.ollama_enabled is True
    assert cfg.ollama_endpoint == "http://localhost:11434"
    assert cfg.ollama_model_name == DEFAULT_OLLAMA_MODEL_NAME
    assert cfg.ollama_timeout_seconds == DEFAULT_OLLAMA_TIMEOUT_SECONDS
    assert cfg.ollama_preload_model is True
    assert cfg.show_recording_overlay is True
    assert cfg.enable_notifications is False


def test_overlay_and_toast_settings_are_independent(tmp_path, monkeypatch):
    monkeypatch.setenv("APPDATA", str(tmp_path))
    cfg = Config()
    cfg.update({"show_recording_overlay": False, "enable_notifications": True})

    reloaded = Config()

    assert reloaded.show_recording_overlay is False
    assert reloaded.enable_notifications is True


def test_saved_notification_choice_is_kept_without_migration(tmp_path, monkeypatch):
    monkeypatch.setenv("APPDATA", str(tmp_path))
    config_dir = tmp_path / config_module.APP_DIR_NAME
    config_dir.mkdir()
    (config_dir / "config.json").write_text(
        json.dumps({"enable_notifications": True}), encoding="utf-8"
    )

    cfg = Config()

    assert cfg.enable_notifications is True
    assert cfg.show_recording_overlay is True


def test_config_persistence(tmp_path, monkeypatch):
    monkeypatch.setenv("APPDATA", str(tmp_path))
    cfg = Config()
    cfg.set("ollama_enabled", False)
    cfg.set("ollama_endpoint", "http://127.0.0.1:11434")

    # Reload config from disk
    cfg2 = Config()
    assert cfg2.ollama_enabled is False
    assert cfg2.ollama_endpoint == "http://127.0.0.1:11434"


def test_config_recovers_from_malformed_json_without_overwriting_backup(
    tmp_path, monkeypatch
):
    monkeypatch.setenv("APPDATA", str(tmp_path))
    config_dir = tmp_path / "murmur"
    config_dir.mkdir()
    config_file = config_dir / "config.json"
    config_file.write_text('{"hotkey": ', encoding="utf-8")

    cfg = Config()

    assert cfg.hotkey == DEFAULT_CONFIG["hotkey"]
    backup_files = list(config_dir.glob("config.corrupt*.json"))
    assert len(backup_files) == 1
    assert backup_files[0].read_text(encoding="utf-8") == '{"hotkey": '
    assert json.loads(config_file.read_text(encoding="utf-8")) == DEFAULT_CONFIG
    notice = cfg.consume_startup_notice()
    assert backup_files[0].name in notice
    assert cfg.consume_startup_notice() is None


def test_config_recovers_from_wrong_shaped_json_payload(tmp_path, monkeypatch):
    monkeypatch.setenv("APPDATA", str(tmp_path))
    config_dir = tmp_path / "murmur"
    config_dir.mkdir()
    config_file = config_dir / "config.json"
    config_file.write_text(json.dumps(["ctrl+alt+space"]), encoding="utf-8")

    cfg = Config()

    assert cfg.get_all() == DEFAULT_CONFIG
    backup_files = list(config_dir.glob("config.corrupt*.json"))
    assert len(backup_files) == 1
    assert json.loads(backup_files[0].read_text(encoding="utf-8")) == ["ctrl+alt+space"]
    assert json.loads(config_file.read_text(encoding="utf-8")) == DEFAULT_CONFIG
    assert backup_files[0].name in cfg.consume_startup_notice()


def test_config_recovers_from_invalid_utf8_bytes(tmp_path, monkeypatch):
    monkeypatch.setenv("APPDATA", str(tmp_path))
    config_dir = tmp_path / "murmur"
    config_dir.mkdir()
    config_file = config_dir / "config.json"
    invalid_bytes = b'{"hotkey": "\x80"}'
    config_file.write_bytes(invalid_bytes)

    cfg = Config()

    assert cfg.get_all() == DEFAULT_CONFIG
    backup_files = list(config_dir.glob("config.corrupt*.json"))
    assert len(backup_files) == 1
    assert backup_files[0].read_bytes() == invalid_bytes
    assert json.loads(config_file.read_text(encoding="utf-8")) == DEFAULT_CONFIG
    assert backup_files[0].name in cfg.consume_startup_notice()


@pytest.mark.parametrize(
    ("values", "invalid_key"),
    [
        ({"enable_logging": "false"}, "enable_logging"),
        ({"enable_logging": 0}, "enable_logging"),
        ({"enable_logging": 1}, "enable_logging"),
        ({"enable_logging": None}, "enable_logging"),
        ({"ollama_enabled": 1}, "ollama_enabled"),
        ({"sample_rate": None}, "sample_rate"),
        ({"sample_rate": 0}, "sample_rate"),
        ({"sample_rate": -16000}, "sample_rate"),
        ({"sample_rate": 16000.5}, "sample_rate"),
        ({"sample_rate": True}, "sample_rate"),
        ({"max_recording_duration": math.inf}, "max_recording_duration"),
        ({"max_recording_duration": -math.inf}, "max_recording_duration"),
        ({"max_recording_duration": math.nan}, "max_recording_duration"),
        ({"max_recording_duration": 10**1000}, "max_recording_duration"),
        ({"max_recording_duration": -1}, "max_recording_duration"),
        ({"max_recording_duration": 1801}, "max_recording_duration"),
        ({"vad_aggressiveness": "2"}, "vad_aggressiveness"),
        ({"vad_aggressiveness": 4}, "vad_aggressiveness"),
        ({"vad_padding_ms": 0}, "vad_padding_ms"),
        ({"vad_silence_duration_ms": None}, "vad_silence_duration_ms"),
        ({"ollama_timeout_seconds": math.inf}, "ollama_timeout_seconds"),
        ({"ollama_timeout_seconds": math.nan}, "ollama_timeout_seconds"),
        ({"hotkey": 12}, "hotkey"),
        ({"language": 12}, "language"),
        ({"microphone": {"name": "USB Mic"}}, "microphone"),
    ],
)
def test_invalid_known_values_are_backed_up_and_repaired(
    tmp_path, monkeypatch, values, invalid_key
):
    monkeypatch.setenv("APPDATA", str(tmp_path))
    config_dir = tmp_path / "murmur"
    config_dir.mkdir()
    original = {"hotkey": "ctrl+alt+space", "custom_option": {"kept": True}, **values}
    config_file = config_dir / "config.json"
    config_file.write_text(json.dumps(original), encoding="utf-8")

    cfg = Config()

    assert cfg.get(invalid_key) == DEFAULT_CONFIG[invalid_key]
    if invalid_key != "hotkey":
        assert cfg.hotkey == "ctrl+alt+space"
    assert cfg.get("custom_option") == {"kept": True}
    notice = cfg.consume_startup_notice()
    assert invalid_key in notice
    assert "Settings or your config file" in notice
    backups = list(config_dir.glob("config.corrupt*.json"))
    assert len(backups) == 1
    assert backups[0].read_text(encoding="utf-8") == json.dumps(original)
    repaired = json.loads(config_file.read_text(encoding="utf-8"))
    assert repaired["custom_option"] == {"kept": True}
    assert repaired["hotkey"] == cfg.hotkey
    assert repaired[invalid_key] == DEFAULT_CONFIG[invalid_key]
    assert cfg.consume_startup_notice() is None
    assert Config().consume_startup_notice() is None
    assert len(list(config_dir.glob("config.corrupt*.json"))) == 1


def test_valid_custom_config_values_are_preserved(tmp_path, monkeypatch):
    monkeypatch.setenv("APPDATA", str(tmp_path))
    config_dir = tmp_path / "murmur"
    config_dir.mkdir()
    config_file = config_dir / "config.json"
    values = {
        "sample_rate": 44100,
        "model": "custom-whisper-model",
        "ollama_model_name": "custom:latest",
        "max_recording_duration": 45.5,
        "microphone": {"name": "USB Mic", "hostapi": "Windows WASAPI"},
        "future_setting": "preserved",
    }
    config_file.write_text(json.dumps(values), encoding="utf-8")

    cfg = Config()

    assert cfg.get_all()["sample_rate"] == 44100
    assert cfg.model_name == "custom-whisper-model"
    assert cfg.ollama_model_name == "custom:latest"
    assert cfg.get("max_recording_duration") == 45.5
    assert cfg.microphone == values["microphone"]
    assert cfg.get("future_setting") == "preserved"
    assert cfg.consume_startup_notice() is None


def test_logging_getter_requires_literal_true(tmp_path, monkeypatch):
    monkeypatch.setenv("APPDATA", str(tmp_path))
    cfg = Config()
    cfg._config["enable_logging"] = "false"
    assert cfg.enable_logging is False


@pytest.mark.parametrize(
    ("payload", "expected_rate", "expected_logging"),
    [
        ({"sample_rate": None}, 16000, False),
        ({"enable_logging": "false"}, 16000, False),
    ],
)
def test_invalid_loaded_config_is_safe_for_runtime_construction(
    tmp_path, monkeypatch, payload, expected_rate, expected_logging
):
    monkeypatch.setenv("APPDATA", str(tmp_path))
    config_dir = tmp_path / "murmur"
    config_dir.mkdir()
    (config_dir / "config.json").write_text(json.dumps(payload), encoding="utf-8")
    cfg = Config()
    monkeypatch.setattr(logger_module, "get_config", lambda: cfg)

    recorder = AudioRecorder(cfg)
    logger = DataLogger(tmp_path / "training-data")

    assert recorder.sample_rate == expected_rate
    assert recorder._max_recording_samples == expected_rate * 300
    assert logger._enabled is expected_logging


def test_valid_literal_logging_true_is_preserved(tmp_path, monkeypatch):
    monkeypatch.setenv("APPDATA", str(tmp_path))
    config_dir = tmp_path / "murmur"
    config_dir.mkdir()
    (config_dir / "config.json").write_text(
        json.dumps({"enable_logging": True}), encoding="utf-8"
    )

    assert Config().enable_logging is True


@pytest.mark.parametrize(
    "original",
    [
        '{"hotkey":',
        '["wrong shape"]',
        '{"sample_rate": null, "hotkey": "ctrl+alt+space"}',
    ],
)
def test_failed_repair_keeps_active_config_for_retry(tmp_path, monkeypatch, original):
    monkeypatch.setenv("APPDATA", str(tmp_path))
    config_dir = tmp_path / "murmur"
    config_dir.mkdir()
    config_file = config_dir / "config.json"
    config_file.write_text(original, encoding="utf-8")

    def fail_replace(source, destination):
        raise OSError("disk full")

    with monkeypatch.context() as patch:
        patch.setattr(config_module.os, "replace", fail_replace)
        with pytest.raises(config_module.ConfigError, match="Failed to write config"):
            Config()

    assert config_file.read_text(encoding="utf-8") == original
    backups = list(config_dir.glob("config.corrupt-*.json"))
    assert len(backups) == 1
    assert backups[0].read_text(encoding="utf-8") == original
    assert not list(config_dir.glob("*.tmp"))

    repaired = Config()
    assert repaired.consume_startup_notice() is not None
    if "ctrl+alt+space" in original:
        assert repaired.hotkey == "ctrl+alt+space"


@pytest.mark.parametrize("enabled", [False, True, "false"])
def test_saved_blank_ollama_fields_require_disabled_cleanup(
    tmp_path, monkeypatch, enabled
):
    monkeypatch.setenv("APPDATA", str(tmp_path))
    cfg = Config()
    cfg.update(
        {"ollama_enabled": enabled, "ollama_endpoint": "", "ollama_model_name": ""}
    )

    reloaded = Config()

    if enabled is False:
        assert reloaded.ollama_endpoint == ""
        assert reloaded.ollama_model_name == ""
        assert reloaded.consume_startup_notice() is None
        assert not list(cfg.config_dir.glob("config.corrupt-*.json"))
    else:
        assert reloaded.ollama_endpoint == DEFAULT_CONFIG["ollama_endpoint"]
        assert reloaded.ollama_model_name == DEFAULT_CONFIG["ollama_model_name"]
        assert reloaded.consume_startup_notice() is not None


def test_config_set_raises_when_atomic_save_fails(tmp_path, monkeypatch):
    monkeypatch.setenv("APPDATA", str(tmp_path))
    cfg = Config()
    config_file = tmp_path / "murmur" / "config.json"
    original_contents = config_file.read_text(encoding="utf-8")
    expected_error = getattr(config_module, "ConfigError", Exception)

    def fail_replace(source, destination):
        raise OSError("disk full")

    monkeypatch.setattr(config_module.os, "replace", fail_replace)

    with pytest.raises(expected_error, match="Failed to write config file"):
        cfg.set("model", "medium")

    assert cfg.model_name == "small"
    assert config_file.read_text(encoding="utf-8") == original_contents


def test_concurrent_updates_preserve_disjoint_keys(tmp_path, monkeypatch):
    monkeypatch.setenv("APPDATA", str(tmp_path))
    cfg = Config()
    original_save = cfg._save_config
    first_save_started = threading.Event()
    second_save_finished = threading.Event()
    save_count = 0
    save_count_lock = threading.Lock()

    def coordinated_save(config_data):
        nonlocal save_count
        with save_count_lock:
            save_count += 1
            current_save = save_count

        if current_save == 1:
            first_save_started.set()
            second_save_finished.wait(timeout=0.2)

        original_save(config_data)

        if current_save == 2:
            second_save_finished.set()

    monkeypatch.setattr(cfg, "_save_config", coordinated_save)
    first_thread = threading.Thread(target=cfg.update, args=({"thread_a": True},))
    second_thread = threading.Thread(target=cfg.update, args=({"thread_b": True},))

    first_thread.start()
    assert first_save_started.wait(timeout=1)
    second_thread.start()
    first_thread.join(timeout=1)
    second_thread.join(timeout=1)

    assert not first_thread.is_alive()
    assert not second_thread.is_alive()
    assert cfg.get("thread_a") is True
    assert cfg.get("thread_b") is True
    persisted_config = json.loads(cfg.config_file.read_text(encoding="utf-8"))
    assert persisted_config["thread_a"] is True
    assert persisted_config["thread_b"] is True
