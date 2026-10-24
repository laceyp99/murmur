import csv
import json
import queue
import threading
from concurrent.futures import Future
from types import SimpleNamespace

import pytest

from src import settings_gui as gui
from src.config import DEFAULT_CONFIG, ConfigError


class Value:
    def __init__(self, value):
        self.value = value

    def get(self):
        return self.value

    def set(self, value):
        self.value = value


@pytest.fixture
def window(monkeypatch):
    window = gui.SettingsWindow.__new__(gui.SettingsWindow)
    window._closed = False
    window._statistics_token = None
    window._statistics_results = queue.Queue()
    window._statistics_buttons = []
    window.messages = []
    window.scheduled = []
    window._statistics_status = SimpleNamespace(
        configure=lambda **kw: window.messages.append(kw["text"])
    )
    window.root = SimpleNamespace(after=lambda *args: window.scheduled.append(args))
    window.dialogs = []
    monkeypatch.setattr(
        gui.messagebox, "showinfo", lambda *a, **k: window.dialogs.append(a)
    )
    monkeypatch.setattr(
        gui.messagebox, "showerror", lambda *a, **k: window.dialogs.append(a)
    )
    return window


def test_view_preserves_zero_unknown_and_separate_contexts():
    view = gui._statistics_view(
        {
            "sessions": 2,
            "enabled": False,
            "incomplete": True,
            "storage_available": False,
            "captured_minutes": 0,
            "success_rate": None,
            "delivered_words": None,
            "fallback_count": 0,
            "fallback_denominator": 1,
            "latency": {
                "count": 1,
                "median_seconds": 0,
                "p95_seconds": None,
                "small_sample": True,
            },
            "stages": {
                "clipboard": {"attempts": 2, "success_count": 1, "failed_count": 1},
                "ollama_cleanup": {"attempts": 2, "skipped_count": 2},
                "preparation": {"attempts": 0},
            },
            "model_groups": [
                {
                    "whisper_model": "small",
                    "whisper_device": "cpu",
                    "cleanup_enabled": True,
                    "cleanup_model": "model-a",
                    "sessions": 1,
                }
            ],
            "cleanup_groups": [
                {"model": "model-a", "attempts": 2, "failed_count": 1},
                {"model": "model-b", "attempts": 1, "success_count": 1},
                {"model": None, "attempts": 3, "skipped_count": 3},
            ],
        }
    )
    assert view.empty is None
    assert any("Storage unavailable" in notice for notice in view.notices)
    assert any("missing" in notice for notice in view.notices)
    assert any("Collection is off" in notice for notice in view.notices)
    assert view.tiles == (
        ("2", "Sessions"),
        ("n/a", "Success rate"),
        ("0 ms", "Median wait"),
        ("0", "Words delivered"),
    )
    rows = {label: value for _, section in view.sections for label, value in section}
    assert rows["Needed transcription fallback"] == "0 of 1"
    assert rows["Stop to clipboard (median / p95)"] == "0 ms / n/a"
    assert rows["Total recorded"] == "0.0 min"
    assert rows["Clipboard"] == "1 ok, 1 failed"
    assert rows["Ollama cleanup"] == "2 skipped"
    assert "Preparation" not in rows
    assert "unknown cleanup" not in rows
    assert rows["small · CPU + model-a cleanup"] == "1 session · n/a median"
    assert rows["model-a cleanup"] == "1 failed"
    assert rows["model-b cleanup"] == "1 ok"
    assert "Fewer than 20 successful sessions so far (1)" in view.footnotes[0]


def test_single_cleanup_model_is_not_repeated_under_models():
    view = gui._statistics_view(
        {"sessions": 1, "cleanup_groups": [{"model": "model-a", "success_count": 1}]}
    )
    labels = [label for _, section in view.sections for label, _ in section]
    assert "model-a cleanup" not in labels


@pytest.mark.parametrize(
    ("seconds", "text"),
    [
        (None, "n/a"),
        (0.0002, "<1 ms"),
        (0.0035, "4 ms"),
        (0.4, "0.40 s"),
        (21.6, "21.6 s"),
    ],
)
def test_durations_use_milliseconds_for_quick_steps(seconds, text):
    assert gui._seconds(seconds) == text


@pytest.mark.parametrize(
    ("enabled", "title"), [(True, "No sessions yet"), (False, "Collection is off")]
)
def test_empty_history_shows_one_message_instead_of_zero_rows(enabled, title):
    view = gui._statistics_view({"sessions": 0, "enabled": enabled})
    assert view.empty[0] == title
    assert view.tiles == view.sections == view.footnotes == ()


def test_csv_has_stable_allowlist_and_formula_safe_values(tmp_path):
    path = tmp_path / "stats.csv"
    gui._write_statistics_csv(
        path,
        [
            {
                "session_id": "one",
                "capture_seconds": 12,
                "whisper_model": "  =HYPERLINK(1)",
                "cleanup_model": "@model",
                "app_version": "café",
                "git_commit": "abc",
                "git_dirty": None,
                "transcript": "private",
                "endpoint": "private",
                "stage_capture_seconds": 0,
                "stage_capture_outcome": "success",
                "issues": json.dumps([{"code": "capture_failed", "count": 1}]),
            }
        ],
    )
    with path.open(encoding="utf-8", newline="") as stream:
        rows = list(csv.DictReader(stream))
    row = rows[0]
    assert row["whisper_model"].startswith("'  =")
    assert row["cleanup_model"] == "'@model"
    assert row["app_version"] == "café"
    assert row["capture_seconds"] == "12"
    assert row["stage_capture_seconds"] == "0"
    assert row["git_dirty"] == ""
    assert "transcript" not in row
    assert "endpoint" not in row
    assert "raw_error" not in json.loads(row["issues"])[0]
    assert "private" not in path.read_text(encoding="utf-8")
    second = tmp_path / "empty.csv"
    gui._write_statistics_csv(second, [])
    assert (
        path.read_text(encoding="utf-8").splitlines()[0]
        == second.read_text(encoding="utf-8").splitlines()[0]
    )


def test_clear_waits_for_commit_then_refreshes_on_ui_thread(window, monkeypatch):
    future = Future()
    started = threading.Event()

    def clear():
        started.set()
        return future

    monkeypatch.setattr(gui, "get_usage_stats", lambda: SimpleNamespace(clear=clear))
    refreshed = []
    window._refresh_statistics = lambda: refreshed.append(True)
    window._statistics_request("clear")
    assert started.wait(1)
    window._poll_statistics_result()
    assert not window.dialogs
    assert not refreshed
    future.set_result(None)
    # Waiting for the worker's queue entry does not execute any Tk callbacks.
    entry = window._statistics_results.get(timeout=1)
    assert not window.dialogs
    window._statistics_results.put(entry)
    window._poll_statistics_result()
    assert "deleted" in window.dialogs[0][1]
    assert refreshed == [True]


def test_closed_and_stale_results_do_not_update_widgets(window):
    current = threading.Event()
    window._statistics_token = current
    window._statistics_results.put((threading.Event(), "summary", {}, None))
    window._poll_statistics_result()
    assert not window.messages
    window._closed = True
    window._statistics_results.put((current, "clear", None, None))
    window._poll_statistics_result()
    assert not window.dialogs


def test_busy_and_failed_delete_do_not_report_success(window):
    for error, expected in (
        (gui.StatisticsBusyError(), "busy"),
        (OSError("private"), "unavailable"),
    ):
        token = threading.Event()
        window._statistics_token = token
        window._statistics_results.put((token, "clear", None, error))
        window._poll_statistics_result()
        assert expected in window.messages[-1]
        assert "private" not in window.messages[-1]
        assert "deleted" not in window.dialogs[-1][1]


def test_export_cancel_and_delete_decline_make_no_request(window, monkeypatch):
    monkeypatch.setattr(gui.filedialog, "asksaveasfilename", lambda **kw: "")
    monkeypatch.setattr(gui.messagebox, "askyesno", lambda *a, **kw: False)
    window._statistics_request = lambda *a: pytest.fail("unexpected request")
    window._export_statistics()
    window._delete_statistics()


@pytest.mark.parametrize("fail_save", [False, True])
def test_collection_applies_only_after_successful_save(window, monkeypatch, fail_save):
    events = []
    config = dict(DEFAULT_CONFIG)

    def update(values):
        events.append("save")
        if fail_save:
            raise ConfigError("failed")
        config.update(values)

    window.config = SimpleNamespace(**config, get=config.get, update=update)
    window.logger = SimpleNamespace(
        set_enabled=lambda value: events.append(("training", value))
    )
    variables = {
        "hotkey": "hotkey",
        "model": "model",
        "device": "device",
        "lang": "language",
        "notify": "enable_notifications",
        "overlay": "show_recording_overlay",
        "logging": "enable_logging",
        "pause_media": "pause_media_while_recording",
        "keep_microphone_ready": "keep_microphone_ready",
        "autostart": "start_with_windows",
        "ollama_enabled": "ollama_enabled",
        "ollama_endpoint": "ollama_endpoint",
        "ollama_model_name": "ollama_model_name",
        "ollama_preload_model": "ollama_preload_model",
    }
    for attr, key in variables.items():
        setattr(
            window, attr + "_var", Value(config[key] if config[key] is not None else "")
        )
    window.statistics_var = Value(False)
    window.numeric_vars = {}
    window._close = lambda: events.append("close")
    monkeypatch.setattr(gui, "is_hotkey_valid", lambda value: True)
    monkeypatch.setattr(
        gui,
        "get_usage_stats",
        lambda: SimpleNamespace(
            set_enabled=lambda value: events.append(("statistics", value))
        ),
    )
    window._save()
    assert events[0] == "save"
    if fail_save:
        assert events == ["save"]
    else:
        assert events.index(("statistics", False)) > events.index("save")
        assert ("training", DEFAULT_CONFIG["enable_logging"]) in events
        assert config["enable_statistics"] is False
