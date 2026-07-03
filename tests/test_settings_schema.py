from src.config import DEFAULT_CONFIG
from src.settings_schema import (
    SETTINGS,
    SETTINGS_BY_KEY,
    TAB_ORDER,
    NumericSettingError,
    SettingMetadata,
    normalize_value,
    parse_numeric_text,
    settings_for_tab,
)


def test_settings_metadata_covers_gui_managed_config_without_sample_rate():
    expected_keys = set(DEFAULT_CONFIG) - {
        "sample_rate",
        "logging_consent_updated_at",
        "logging_consent_source",
    }

    assert set(SETTINGS_BY_KEY) == expected_keys
    setting_tabs = {setting.tab for setting in SETTINGS}
    assert setting_tabs <= set(TAB_ORDER)
    assert set(TAB_ORDER) <= setting_tabs


def test_tabs_have_expected_order_and_settings():
    assert TAB_ORDER == (
        "General",
        "VAD",
        "Transcription",
        "LLM Cleanup",
        "Data Privacy",
    )
    assert [setting.key for setting in settings_for_tab("VAD")] == [
        "vad_aggressiveness",
        "vad_padding_ms",
        "vad_silence_duration_ms",
    ]


def test_bounded_numeric_metadata_has_defaults_within_range():
    for setting in SETTINGS:
        if not setting.bounded:
            continue

        assert setting.min_value is not None
        assert setting.max_value is not None
        assert setting.step is not None
        assert setting.min_value <= setting.default <= setting.max_value


def test_runtime_cached_settings_are_marked_restart_required():
    assert SETTINGS_BY_KEY["hotkey"].restart_required is True
    assert SETTINGS_BY_KEY["ollama_timeout_seconds"].restart_required is True


def test_numeric_values_are_clamped_for_display_and_save():
    timeout = SETTINGS_BY_KEY["ollama_timeout_seconds"]

    assert normalize_value(10, timeout) == 60
    assert normalize_value(999, timeout) == 300
    assert parse_numeric_text("120", timeout) == 120


def test_numeric_text_entry_allows_values_between_slider_steps():
    vad_padding = SETTINGS_BY_KEY["vad_padding_ms"]

    assert parse_numeric_text("237", vad_padding) == 237


def test_non_numeric_text_raises_for_save_parsing():
    timeout = SETTINGS_BY_KEY["ollama_timeout_seconds"]

    try:
        parse_numeric_text("abc", timeout)
    except NumericSettingError as exc:
        assert str(exc) == "ollama_timeout_seconds"
    else:
        raise AssertionError("expected NumericSettingError")


def test_non_finite_numeric_text_raises_for_save_parsing():
    timeout = SETTINGS_BY_KEY["ollama_timeout_seconds"]

    try:
        parse_numeric_text("inf", timeout)
    except NumericSettingError as exc:
        assert str(exc) == "ollama_timeout_seconds"
    else:
        raise AssertionError("expected NumericSettingError")


def test_non_finite_numeric_config_normalizes_to_default():
    timeout = SETTINGS_BY_KEY["ollama_timeout_seconds"]

    assert normalize_value("inf", timeout) == 60


def test_numeric_config_raises_clear_error_when_default_is_invalid():
    setting = SettingMetadata(
        key="demo_timeout",
        label="Demo timeout",
        tab="General",
        control="number",
        value_type="int",
        default="auto",
        min_value=0,
        max_value=10,
        step=1,
    )

    try:
        normalize_value("bad", setting)
    except ValueError as exc:
        assert str(exc) == "Setting 'demo_timeout' has an unparseable default: 'auto'"
    else:
        raise AssertionError("expected ValueError")
