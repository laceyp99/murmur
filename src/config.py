"""
Configuration management for Murmur.
"""

import json
import os
from pathlib import Path
from typing import Any, Dict, Optional


APP_DIR_NAME = "murmur"

DEFAULT_OLLAMA_MODEL_NAME = "granite4.1:3b"
DEFAULT_OLLAMA_TIMEOUT_SECONDS = 60


DEFAULT_CONFIG = {
    "hotkey": "ctrl+shift+space",
    "model": "small",
    "device": "cuda",
    "language": None,
    "sample_rate": 16000,
    "vad_aggressiveness": 1,
    "vad_padding_ms": 220,
    "vad_silence_duration_ms": 400,
    "max_recording_duration": 300,
    "enable_logging": False,
    "enable_notifications": True,
    "start_with_windows": True,
    "pause_media_while_recording": True,
    "logging_consent_updated_at": None,
    "logging_consent_source": None,
    # Ollama LLM post-processor configuration
    "ollama_enabled": True,
    "ollama_endpoint": "http://localhost:11434",
    "ollama_model_name": DEFAULT_OLLAMA_MODEL_NAME,
    "ollama_timeout_seconds": DEFAULT_OLLAMA_TIMEOUT_SECONDS,
    "ollama_preload_model": True,
}

# Global config instance
_config_instance = None


def get_app_data_dir() -> Path:
    """Get the canonical Murmur AppData directory."""
    return Path(os.environ.get("APPDATA", ".")) / APP_DIR_NAME


def get_training_data_dir() -> Path:
    """Get the canonical training data directory."""
    return get_app_data_dir() / "training_data"


class Config:
    """Manages application configuration."""

    def __init__(self):
        self.config_dir = get_app_data_dir()
        self.config_file = self.config_dir / "config.json"
        self._config: Dict[str, Any] = {}
        self._load()

    def _load(self):
        """Load configuration from file or create default."""
        self.config_dir.mkdir(parents=True, exist_ok=True)

        if self.config_file.exists():
            try:
                with open(self.config_file, "r") as f:
                    self._config = {**DEFAULT_CONFIG, **json.load(f)}
            except (json.JSONDecodeError, IOError):
                self._config = DEFAULT_CONFIG.copy()
                self._save()
        else:
            self._config = DEFAULT_CONFIG.copy()
            self._save()

    def _save(self):
        """Save configuration to file."""
        try:
            with open(self.config_file, "w") as f:
                json.dump(self._config, f, indent=2)
        except IOError:
            pass

    def get(self, key: str, default: Any = None) -> Any:
        """Get a configuration value."""
        return self._config.get(key, default)

    def set(self, key: str, value: Any):
        """Set a configuration value and save."""
        self._config[key] = value
        self._save()

    def get_all(self) -> Dict[str, Any]:
        """Get all configuration values."""
        return self._config.copy()

    @property
    def hotkey(self) -> str:
        """Get the hotkey setting."""
        return self.get("hotkey", "ctrl+shift+space")

    @property
    def model_name(self) -> str:
        """Get the model name setting."""
        return self.get("model", "small")

    @property
    def language(self) -> Optional[str]:
        """Get the language setting."""
        return self.get("language", None)

    @property
    def device(self) -> str:
        """Get the device setting (cuda or cpu)."""
        return self.get("device", "cuda")

    @property
    def sample_rate(self) -> int:
        """Get the sample rate setting."""
        return self.get("sample_rate", 16000)

    @property
    def vad_aggressiveness(self) -> int:
        """Get the WebRTC VAD aggressiveness setting."""
        return self.get("vad_aggressiveness", 1)

    @property
    def vad_padding_ms(self) -> int:
        """Get the user-facing VAD end-padding anchor in milliseconds."""
        return self.get("vad_padding_ms", 220)

    @property
    def vad_silence_duration_ms(self) -> int:
        """Get the silence duration required to close a VAD segment."""
        return self.get("vad_silence_duration_ms", 400)

    @property
    def enable_notifications(self) -> bool:
        """Get the notifications setting."""
        return self.get("enable_notifications", True)

    @property
    def start_with_windows(self) -> bool:
        """Get the start with windows setting."""
        return self.get("start_with_windows", True)

    @property
    def pause_media_while_recording(self) -> bool:
        """Get the pause media while recording setting."""
        return self.get("pause_media_while_recording", True)

    @property
    def enable_logging(self) -> bool:
        """Get the training-data logging setting."""
        return self.get("enable_logging", False)

    @property
    def ollama_enabled(self) -> bool:
        """Whether the Ollama LLM post-processor is enabled."""
        return self.get("ollama_enabled", True)

    @property
    def ollama_endpoint(self) -> str:
        """URL of the Ollama endpoint to use for LLM requests."""
        return self.get("ollama_endpoint", "http://localhost:11434")

    @property
    def ollama_model_name(self) -> str:
        """Default Ollama model name to use for post-processing."""
        return self.get("ollama_model_name", DEFAULT_OLLAMA_MODEL_NAME)

    @property
    def ollama_timeout_seconds(self) -> int:
        """Timeout in seconds for Ollama requests; values below 60 are raised to 60."""
        raw_timeout = self.get(
            "ollama_timeout_seconds", DEFAULT_OLLAMA_TIMEOUT_SECONDS
        )
        try:
            timeout_seconds = float(raw_timeout)
        except (TypeError, ValueError):
            return DEFAULT_OLLAMA_TIMEOUT_SECONDS
    
        if timeout_seconds <= 0:
            return DEFAULT_OLLAMA_TIMEOUT_SECONDS
        
        return int(max(timeout_seconds, DEFAULT_OLLAMA_TIMEOUT_SECONDS))

    @property
    def ollama_preload_model(self) -> bool:
        """Whether to attempt to preload/warm the Ollama model on startup."""
        return self.get("ollama_preload_model", True)


def get_config() -> Config:
    """Get the global config instance."""
    global _config_instance
    if _config_instance is None:
        _config_instance = Config()
    return _config_instance
