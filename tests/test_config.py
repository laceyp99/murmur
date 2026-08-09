import json
import threading

import pytest

import src.config as config_module
from src.config import (
    DEFAULT_CONFIG,
    DEFAULT_OLLAMA_MODEL_NAME,
    DEFAULT_OLLAMA_TIMEOUT_SECONDS,
    Config,
)


def test_config_defaults(tmp_path, monkeypatch):
    monkeypatch.setenv("APPDATA", str(tmp_path))
    cfg = Config()
    assert cfg.ollama_enabled is True
    assert cfg.ollama_endpoint == "http://localhost:11434"
    assert cfg.ollama_model_name == DEFAULT_OLLAMA_MODEL_NAME
    assert cfg.ollama_timeout_seconds == DEFAULT_OLLAMA_TIMEOUT_SECONDS
    assert cfg.ollama_preload_model is True


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
