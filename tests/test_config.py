import pytest

from src.config import Config


def test_config_defaults(tmp_path, monkeypatch):
    monkeypatch.setenv("APPDATA", str(tmp_path))
    cfg = Config()
    assert cfg.ollama_enabled is False
    assert cfg.ollama_endpoint == "http://localhost:11434"
    assert cfg.ollama_model_name == "qwen-3.5:2b"
    assert cfg.ollama_timeout_seconds == 5
    assert cfg.ollama_preload_model is True


def test_config_persistence(tmp_path, monkeypatch):
    monkeypatch.setenv("APPDATA", str(tmp_path))
    cfg = Config()
    cfg.set("ollama_enabled", True)
    cfg.set("ollama_endpoint", "http://127.0.0.1:11434")

    # Reload config from disk
    cfg2 = Config()
    assert cfg2.ollama_enabled is True
    assert cfg2.ollama_endpoint == "http://127.0.0.1:11434"
