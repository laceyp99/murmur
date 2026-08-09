"""Metadata and value helpers for the Murmur settings panel."""

from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Any, Literal

from .config import DEFAULT_CONFIG

TAB_ORDER = (
    "General",
    "VAD",
    "Transcription",
    "LLM Cleanup",
    "Data Privacy",
)

ControlType = Literal["bool", "number", "select", "text"]
ValueType = Literal["bool", "int", "str", "optional_str"]


@dataclass(frozen=True)
class SettingMetadata:
    """Describe how one config key is edited in the settings panel."""

    key: str
    label: str
    tab: str
    control: ControlType
    value_type: ValueType
    default: Any
    help_text: str = ""
    min_value: float | None = None
    max_value: float | None = None
    step: float | None = None
    choices: tuple[str, ...] = ()
    restart_required: bool = False

    @property
    def bounded(self) -> bool:
        return self.control == "number"


SETTINGS: tuple[SettingMetadata, ...] = (
    SettingMetadata(
        key="hotkey",
        label="Hotkey",
        tab="General",
        control="text",
        value_type="str",
        default=DEFAULT_CONFIG["hotkey"],
        help_text="Keyboard shortcut that starts and stops recording.",
        restart_required=True,
    ),
    SettingMetadata(
        key="enable_notifications",
        label="Enable notifications",
        tab="General",
        control="bool",
        value_type="bool",
        default=DEFAULT_CONFIG["enable_notifications"],
    ),
    SettingMetadata(
        key="start_with_windows",
        label="Start with Windows",
        tab="General",
        control="bool",
        value_type="bool",
        default=DEFAULT_CONFIG["start_with_windows"],
    ),
    SettingMetadata(
        key="pause_media_while_recording",
        label="Pause media while recording",
        tab="General",
        control="bool",
        value_type="bool",
        default=DEFAULT_CONFIG["pause_media_while_recording"],
    ),
    SettingMetadata(
        key="vad_aggressiveness",
        label="VAD aggressiveness",
        tab="VAD",
        control="number",
        value_type="int",
        default=DEFAULT_CONFIG["vad_aggressiveness"],
        min_value=0,
        max_value=3,
        step=1,
        help_text="Higher values reject more background noise but may cut quiet speech.",
        restart_required=True,
    ),
    SettingMetadata(
        key="vad_padding_ms",
        label="Speech padding",
        tab="VAD",
        control="number",
        value_type="int",
        default=DEFAULT_CONFIG["vad_padding_ms"],
        min_value=50,
        max_value=800,
        step=10,
        help_text="Extra audio kept around detected speech starts and stops.",
        restart_required=True,
    ),
    SettingMetadata(
        key="vad_silence_duration_ms",
        label="Silence to stop",
        tab="VAD",
        control="number",
        value_type="int",
        default=DEFAULT_CONFIG["vad_silence_duration_ms"],
        min_value=100,
        max_value=1500,
        step=25,
        help_text="How long silence must last before murmur finalizes a speech segment.",
        restart_required=True,
    ),
    SettingMetadata(
        key="model",
        label="Whisper model",
        tab="Transcription",
        control="select",
        value_type="str",
        default=DEFAULT_CONFIG["model"],
        choices=("tiny", "base", "small", "medium", "large"),
        restart_required=True,
    ),
    SettingMetadata(
        key="device",
        label="Device",
        tab="Transcription",
        control="select",
        value_type="str",
        default=DEFAULT_CONFIG["device"],
        choices=("cuda", "cpu"),
        restart_required=True,
    ),
    SettingMetadata(
        key="language",
        label="Language",
        tab="Transcription",
        control="text",
        value_type="optional_str",
        default=DEFAULT_CONFIG["language"],
        help_text="Leave blank or enter none for automatic language detection.",
        restart_required=True,
    ),
    SettingMetadata(
        key="max_recording_duration",
        label="Maximum recording duration",
        tab="Transcription",
        control="number",
        value_type="int",
        default=DEFAULT_CONFIG["max_recording_duration"],
        min_value=30,
        max_value=1800,
        step=30,
        help_text="Hard stop for one recording, in seconds.",
        restart_required=True,
    ),
    SettingMetadata(
        key="ollama_enabled",
        label="Enable Ollama cleanup",
        tab="LLM Cleanup",
        control="bool",
        value_type="bool",
        default=DEFAULT_CONFIG["ollama_enabled"],
    ),
    SettingMetadata(
        key="ollama_endpoint",
        label="Ollama endpoint",
        tab="LLM Cleanup",
        control="text",
        value_type="str",
        default=DEFAULT_CONFIG["ollama_endpoint"],
        restart_required=True,
    ),
    SettingMetadata(
        key="ollama_model_name",
        label="Ollama model",
        tab="LLM Cleanup",
        control="text",
        value_type="str",
        default=DEFAULT_CONFIG["ollama_model_name"],
        restart_required=True,
    ),
    SettingMetadata(
        key="ollama_timeout_seconds",
        label="Ollama timeout",
        tab="LLM Cleanup",
        control="number",
        value_type="int",
        default=DEFAULT_CONFIG["ollama_timeout_seconds"],
        min_value=60,
        max_value=300,
        step=15,
        help_text="Maximum seconds to wait for one final cleanup request.",
        restart_required=True,
    ),
    SettingMetadata(
        key="ollama_preload_model",
        label="Preload Ollama model",
        tab="LLM Cleanup",
        control="bool",
        value_type="bool",
        default=DEFAULT_CONFIG["ollama_preload_model"],
        restart_required=True,
    ),
    SettingMetadata(
        key="enable_logging",
        label="Enable training data logging",
        tab="Data Privacy",
        control="bool",
        value_type="bool",
        default=DEFAULT_CONFIG["enable_logging"],
        help_text="Stores raw WAV audio and transcript text locally after confirmation.",
    ),
)

SETTINGS_BY_KEY = {setting.key: setting for setting in SETTINGS}


class NumericSettingError(ValueError):
    """Raised when numeric settings text cannot be parsed."""


def clamp_number(value: float, setting: SettingMetadata) -> int:
    """Clamp a numeric setting to its configured bounds."""
    if not math.isfinite(value):
        raise ValueError(setting.key)
    if setting.min_value is not None:
        value = max(setting.min_value, value)
    if setting.max_value is not None:
        value = min(setting.max_value, value)
    return round(value)


def parse_numeric_text(text: str, setting: SettingMetadata) -> int:
    """Parse and clamp user-entered numeric text."""
    try:
        return clamp_number(float(text.strip()), setting)
    except (TypeError, ValueError, OverflowError) as exc:
        raise NumericSettingError(setting.key) from exc


def normalize_value(value: Any, setting: SettingMetadata) -> Any:
    """Normalize a raw config or UI value for saving/display."""
    if setting.control == "number":
        try:
            return clamp_number(float(value), setting)
        except (TypeError, ValueError, OverflowError):
            try:
                return clamp_number(float(setting.default), setting)
            except (TypeError, ValueError, OverflowError) as default_exc:
                raise ValueError(
                    f"Setting '{setting.key}' has an unparseable default: {setting.default!r}"
                ) from default_exc

    if setting.control == "select":
        text = "" if value is None else str(value).strip()
        return text if text in setting.choices else setting.default

    if setting.value_type == "bool":
        return bool(value)

    if setting.value_type == "optional_str":
        text = "" if value is None else str(value).strip()
        return None if not text or text.lower() == "none" else text

    return "" if value is None else str(value).strip()


def settings_for_tab(tab: str) -> tuple[SettingMetadata, ...]:
    """Return settings displayed in one tab, preserving metadata order."""
    return tuple(setting for setting in SETTINGS if setting.tab == tab)
