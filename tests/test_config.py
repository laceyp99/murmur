import json

import pytest

import src.config as config_module
from src.config import (
    Config,
    DEFAULT_CONFIG,
    DEFAULT_OLLAMA_MODEL_NAME,
    DEFAULT_OLLAMA_TIMEOUT_SECONDS,
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
