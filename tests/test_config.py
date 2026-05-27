from src.config import (
    Config,
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
