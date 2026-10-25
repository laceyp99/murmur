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
    assert cfg.keep_microphone_ready is False


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
    cfg.set("keep_microphone_ready", True)

    # Reload config from disk
    cfg2 = Config()
    assert cfg2.ollama_enabled is False
    assert cfg2.ollama_endpoint == "http://127.0.0.1:11434"
    assert cfg2.keep_microphone_ready is True


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
    assert "settings were restored to their defaults" in notice
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
    assert "Your original file was backed up" in cfg.consume_startup_notice()


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
    assert "Your original file was backed up" in cfg.consume_startup_notice()


@pytest.mark.parametrize(
    ("invalid_key", "value"),
    [
        ("enable_logging", "false"),
        ("enable_logging", 0),
        ("enable_logging", 1),
        ("enable_logging", None),
        ("keep_microphone_ready", "false"),
        ("sample_rate", None),
        ("sample_rate", 0),
        ("sample_rate", -16000),
        ("sample_rate", 16000.5),
        ("sample_rate", True),
        ("sample_rate", 96001),
        ("sample_rate", 176400),
        ("sample_rate", 192000),
        ("max_recording_duration", math.inf),
        ("max_recording_duration", -math.inf),
        ("max_recording_duration", math.nan),
        ("max_recording_duration", 10**1000),
        ("max_recording_duration", -1),
        ("max_recording_duration", 1801),
        ("vad_aggressiveness", "2"),
        ("vad_aggressiveness", 4),
        ("vad_padding_ms", 0),
        ("vad_silence_duration_ms", None),
        ("ollama_timeout_seconds", math.inf),
        ("ollama_timeout_seconds", math.nan),
        ("hotkey", 12),
        ("language", 12),
        ("microphone", {"name": "USB Mic"}),
        ("speaker_names", "Pat"),
        ("speaker_names", ["Pat", 3]),
    ],
)
def test_invalid_known_values_are_backed_up_and_repaired(
    tmp_path, monkeypatch, invalid_key, value
):
    monkeypatch.setenv("APPDATA", str(tmp_path))
    config_dir = tmp_path / "murmur"
    config_dir.mkdir()
    original = {"hotkey": "ctrl+alt+space", "custom_option": {"kept": True}}
    original[invalid_key] = value
    config_file = config_dir / "config.json"
    config_file.write_text(json.dumps(original), encoding="utf-8")

    cfg = Config()

    assert cfg.get(invalid_key) == DEFAULT_CONFIG[invalid_key]
    if invalid_key != "hotkey":
        assert cfg.hotkey == "ctrl+alt+space"
    assert cfg.get("custom_option") == {"kept": True}
    notice = cfg.consume_startup_notice()
    assert notice
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


@pytest.mark.parametrize(
    "sample_rate", [8000, 16000, 22050, 24000, 44100, 48000, 96000]
)
def test_valid_custom_config_values_are_preserved(tmp_path, monkeypatch, sample_rate):
    monkeypatch.setenv("APPDATA", str(tmp_path))
    config_dir = tmp_path / "murmur"
    config_dir.mkdir()
    config_file = config_dir / "config.json"
    values = {
        "sample_rate": sample_rate,
        "model": "custom-whisper-model",
        "ollama_model_name": "custom:latest",
        "max_recording_duration": 45.5,
        "microphone": {"name": "USB Mic", "hostapi": "Windows WASAPI"},
        "future_setting": "preserved",
        "enable_logging": True,
    }
    config_file.write_text(json.dumps(values), encoding="utf-8")

    cfg = Config()

    for key, value in values.items():
        assert cfg.get(key) == value
    assert cfg.enable_logging is True
    assert cfg.consume_startup_notice() is None


def test_logging_getter_requires_literal_true(tmp_path, monkeypatch):
    monkeypatch.setenv("APPDATA", str(tmp_path))
    cfg = Config()
    cfg.set("enable_logging", "false")
    assert cfg.enable_logging is False


def test_invalid_loaded_config_is_safe_for_runtime_construction(tmp_path, monkeypatch):
    monkeypatch.setenv("APPDATA", str(tmp_path))
    config_dir = tmp_path / "murmur"
    config_dir.mkdir()
    (config_dir / "config.json").write_text(
        json.dumps({"sample_rate": None, "enable_logging": "false"}), encoding="utf-8"
    )
    cfg = Config()
    monkeypatch.setattr(logger_module, "get_config", lambda: cfg)

    recorder = AudioRecorder(cfg)
    logger = DataLogger(tmp_path / "training-data")

    assert recorder.sample_rate == 16000
    assert recorder._max_recording_samples == 16000 * 300
    assert logger.is_enabled() is False


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

    if enabled is not True:
        assert reloaded.ollama_endpoint == ""
        assert reloaded.ollama_model_name == ""
        assert reloaded.ollama_enabled is False
    else:
        assert reloaded.ollama_endpoint == DEFAULT_CONFIG["ollama_endpoint"]
        assert reloaded.ollama_model_name == DEFAULT_CONFIG["ollama_model_name"]
    if enabled is False:
        assert reloaded.consume_startup_notice() is None
        assert not list(cfg.config_dir.glob("config.corrupt-*.json"))
    else:
        assert reloaded.consume_startup_notice() is not None


@pytest.mark.parametrize("invalid_enabled", [0, 1, None, "false", "true", [], {}])
def test_invalid_ollama_enablement_disables_cleanup(
    tmp_path, monkeypatch, invalid_enabled
):
    monkeypatch.setenv("APPDATA", str(tmp_path))
    cfg = Config()
    saved = {
        "ollama_enabled": invalid_enabled,
        "ollama_endpoint": "http://remote:11434",
    }
    cfg.update(saved)

    repaired = Config()

    assert repaired.ollama_enabled is False
    assert repaired.ollama_endpoint == saved["ollama_endpoint"]
    assert "Ollama cleanup: off" in repaired.consume_startup_notice()
    assert (
        json.loads(cfg.config_file.read_text(encoding="utf-8"))["ollama_enabled"]
        is False
    )
    backups = list(cfg.config_dir.glob("config.corrupt-*.json"))
    assert len(backups) == 1
    assert (
        json.loads(backups[0].read_text(encoding="utf-8"))["ollama_enabled"]
        == invalid_enabled
    )
    assert Config().ollama_enabled is False


def test_recovery_notice_lists_repaired_speaker_names_plainly(tmp_path, monkeypatch):
    monkeypatch.setenv("APPDATA", str(tmp_path))
    Config().update({"speaker_names": "Pat"})

    assert "Speaker names: Pat." in Config().consume_startup_notice()


def test_recovery_notice_stays_available_until_dismissed_without_saving(
    tmp_path, monkeypatch
):
    monkeypatch.setenv("APPDATA", str(tmp_path))
    cfg = Config()
    cfg.update({"sample_rate": None, "ollama_enabled": "false"})
    repaired = Config()
    saved = repaired.config_file.read_bytes()
    settings = repaired.get_all()

    notice = repaired.get_startup_notice()
    assert "Sample rate: 16,000 Hz" in notice
    assert "Ollama cleanup: off" in notice
    assert repaired.get_startup_notice() == notice
    assert repaired.consume_startup_notice() == notice
    assert repaired.get_startup_notice() is None
    assert repaired.config_file.read_bytes() == saved
    assert repaired.get_all() == settings


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


@pytest.mark.parametrize(
    ("saved", "expected"),
    [(True, True), (False, False), ("false", False), (None, False), (1, False)],
)
def test_statistics_saved_collection_flag(tmp_path, monkeypatch, saved, expected):
    monkeypatch.setenv("APPDATA", str(tmp_path))
    config_dir = tmp_path / config_module.APP_DIR_NAME
    config_dir.mkdir()
    (config_dir / "config.json").write_text(
        json.dumps({"enable_statistics": saved}), encoding="utf-8"
    )
    cfg = Config()
    assert cfg.enable_statistics is expected
    assert Config().enable_statistics is expected


def test_statistics_missing_flag_defaults_on_and_uses_account_path(
    tmp_path, monkeypatch
):
    monkeypatch.setenv("APPDATA", str(tmp_path))
    config_dir = tmp_path / config_module.APP_DIR_NAME
    config_dir.mkdir()
    (config_dir / "config.json").write_text("{}", encoding="utf-8")
    assert Config().enable_statistics is True
    assert config_module.get_statistics_db_path() == config_dir / "statistics.sqlite3"
    assert (
        config_module.get_statistics_db_path().parent
        == config_module.get_training_data_dir().parent
    )
