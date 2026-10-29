import _thread
import importlib.metadata
import io
import json
import runpy
import sys
import threading
from types import SimpleNamespace

import numpy as np
import pytest

from src.config import DEFAULT_CONFIG
from src.hotkey import HotkeyManager, HotkeyState, RecordingRequest

main_module = pytest.importorskip("src.main")
AudioData = main_module.AudioData
FinalTranscript = main_module.FinalTranscript
MurmurApp = main_module.MurmurApp
VADSettings = main_module.VADSettings


@pytest.fixture(autouse=True)
def overlay_calls(monkeypatch):
    """Capture overlay requests so tests never start the real Tk UI thread."""
    settings_gui = pytest.importorskip("src.settings_gui")
    monkeypatch.setattr(main_module, "get_usage_stats", lambda: None)
    monkeypatch.setattr(main_module, "capture_build_info_in_background", lambda: None)
    calls = []
    monkeypatch.setattr(
        settings_gui,
        "request_overlay_state",
        lambda state, session=None, message="": calls.append((state, session, message)),
    )
    monkeypatch.setattr(settings_gui, "close_overlay", lambda: calls.append("close"))
    return calls


class FakeSegmenter:
    def __init__(self, segments=None, error=None, sample_rate=16000):
        self.sample_rate = sample_rate
        self._segments = list(segments or [])
        self._error = error
        self.calls = []

    def segment_audio(self, audio):
        self.calls.append(audio.copy())
        if self._error is not None:
            raise self._error
        return list(self._segments)


class FakeTranscriber:
    def __init__(self):
        self.segment_calls = []
        self.audio_calls = []
        self.live_segment_calls = []
        self.finalize_calls = []
        self.finalize_segment_calls = []
        self.load_model_calls = 0
        self.warm_llm_post_processor_calls = 0

    def transcribe_segments(self, segments, debug=False, *, cancel_event=None):
        self.segment_calls.append({"segments": list(segments), "debug": debug})
        return SimpleNamespace(final=FinalTranscript("Segmented text."))

    def transcribe(self, audio_data, *, cancel_event=None):
        self.audio_calls.append(audio_data)
        return FinalTranscript("Fallback text.")

    def transcribe_segment(self, segment):
        self.live_segment_calls.append(segment)
        return f"live-{segment.segment_id}"

    def finalize_text(self, text):
        self.finalize_calls.append(text)
        if not text:
            return FinalTranscript("")
        return FinalTranscript(f"FINAL:{text}", text, "success")

    def finalize_segment_texts(self, segment_texts, *, cancel_event=None):
        segment_texts = list(segment_texts)
        self.finalize_segment_calls.append(segment_texts)
        return self.finalize_text(" ".join(segment_texts))

    def load_model(self):
        self.load_model_calls += 1

    def warm_llm_post_processor(self):
        self.warm_llm_post_processor_calls += 1
        return True

    def get_device_info(self):
        return {"device": "cpu"}


class FakeConfig:
    vad_aggressiveness = 1
    vad_padding_ms = 500
    vad_silence_duration_ms = 400
    ollama_enabled = False
    ollama_preload_model = True
    start_with_windows = False
    show_recording_overlay = True


def make_app(segmenter, transcriber, config=None):
    app = MurmurApp.__new__(MurmurApp)
    app.segmenter = segmenter
    app.live_segmenter = None
    app.live_transcription_worker = None
    app.live_transcript_accumulator = None
    app.transcriber = transcriber
    app.config = config or FakeConfig()
    app._vad_disabled_reason = None
    app._live_vad_disabled_reason = None
    app._live_pipeline_degraded = False
    app._live_pipeline_degraded_reason = None
    app._recording_stop_lock = threading.RLock()
    app._recording_stop_started = False
    app._lifecycle_lock = threading.RLock()
    app._cancel_event = threading.Event()
    app._session_active = True
    app._delivery_started = False
    app._shutdown_event = threading.Event()
    app._media_pause_thread = None
    app._current_request = committed_request()
    return app


def committed_request() -> RecordingRequest:
    """Stand in for a hotkey attempt whose capture already started."""
    request = RecordingRequest()
    request.commit_capture()
    return request


def stop_current(app) -> None:
    """Stop the current session the way a post-commit hotkey press does."""
    app._on_recording_stop(app._current_request, True)


class FakeRecorder:
    def __init__(self, audio_data):
        self.audio_data = audio_data
        self.stop_calls = 0
        self.sample_rate = 16000
        self.block_callback = None
        self.block_callback_error_handler = None
        self.recording_limit_callback = None
        self.close_calls = 0

    def prepare(self):
        return False

    def close(self):
        self.close_calls += 1

    def stop_recording(self):
        self.stop_calls += 1
        return self.audio_data

    def set_block_callback(self, callback, *, replay_buffered=False):
        self.block_callback = callback

    def set_block_callback_error_handler(self, handler):
        self.block_callback_error_handler = handler

    def set_recording_limit_callback(self, callback):
        self.recording_limit_callback = callback


class FakeTray:
    def __init__(self):
        self.statuses = []

    def set_status(self, status):
        self.statuses.append(status)


class FakeNotifications:
    def __init__(self):
        self.completed = []
        self.copied_to_clipboard = 0
        self.messages = []
        self.errors = []
        self.recording_limits = []

    def notify_recording_started(self):
        pass

    def notify_transcription_complete(self, text):
        self.completed.append(text)

    def notify_transcription_copied(self):
        self.copied_to_clipboard += 1

    def notify_clipboard_failure_retry(self):
        self.notify_error("Clipboard copy failed. Please retry the recording.")

    def notify_clipboard_failure_retry_with_training_data(self):
        self.notify_error(
            "Clipboard copy failed. Please retry the recording. "
            "The transcript was saved to training data."
        )

    def notify(self, title, message):
        self.messages.append((title, message))

    def notify_error(self, message):
        self.errors.append(message)

    def notify_recording_limit_reached(self, duration_seconds):
        self.recording_limits.append(duration_seconds)


class FakeLogger:
    def __init__(self, enabled=True):
        self.enabled = enabled
        self.entries = []
        self.cleanup_details = []

    def log(self, audio_data, text, **cleanup):
        if not self.enabled:
            return None
        self.entries.append((audio_data, text))
        self.cleanup_details.append(cleanup)
        return self.entries[-1]

    def is_enabled(self):
        return self.enabled

    def get_log_directory(self):
        return "C:/murmur/training_data"

    def get_entry_count(self):
        return 0

    def get_total_duration(self):
        return 0.0


class FakeHotkeyManager:
    def __init__(self):
        self.processing_calls = 0
        self.idle_calls = 0
        self.request = None

    def set_processing(self):
        self.processing_calls += 1

    def set_idle(self):
        self.idle_calls += 1

    def stop_requests(self):
        if self.request is not None:
            self.request.request_stop()
        return self.request


class FakeStartupConfig:
    def __init__(self, hotkey, startup_notice=None):
        self.hotkey = hotkey
        self.model_name = "small"
        self.start_with_windows = False
        self.set_calls = []
        self._startup_notice = startup_notice

    def set(self, key, value):
        self.set_calls.append((key, value))
        setattr(self, key, value)

    def get_startup_notice(self):
        return self._startup_notice


class FakeStartupHotkeyManager:
    def __init__(self, config, should_succeed, error_by_hotkey=None):
        self.config = config
        self.should_succeed = should_succeed
        self.register_calls = []
        self.error_by_hotkey = error_by_hotkey or {}
        self.last_registration_error = None

    def register(self, on_start=None, on_stop=None, on_state_change=None):
        self.register_calls.append(self.config.hotkey)
        success = self.should_succeed(self.config.hotkey)
        if success:
            self.last_registration_error = None
        else:
            self.last_registration_error = self.error_by_hotkey.get(
                self.config.hotkey, "keyboard backend unavailable"
            )
        return success

    def unregister(self):
        return None

    def get_last_registration_error(self):
        return self.last_registration_error


class FakeMediaController:
    def __init__(self):
        self.play_calls = 0

    def play(self, source_id):
        self.play_calls += 1
        return True


def make_audio_data():
    audio = np.array([0.1, -0.3, 0.2], dtype=np.float32)
    return AudioData(audio=audio, sample_rate=16000, duration=len(audio) / 16000)


def make_start_app(config, hotkey_manager):
    app = MurmurApp.__new__(MurmurApp)
    app.config = config
    app.hotkey_manager = hotkey_manager
    app.transcriber = FakeTranscriber()
    app.logger = FakeLogger()
    app.notifications = FakeNotifications()
    app._running = False
    app.recorder = FakeRecorder(None)
    app._shutdown_event = threading.Event()
    app._on_recording_start = lambda: None
    app._on_recording_stop = lambda: None
    app._on_state_change = lambda state: None
    return app


def test_transcribe_audio_uses_vad_segments_when_available():
    segment = SimpleNamespace(duration=0.35)
    segmenter = FakeSegmenter(segments=[segment])
    transcriber = FakeTranscriber()
    app = make_app(segmenter, transcriber)

    transcript = app._transcribe_audio(make_audio_data())

    assert transcript.text == "Segmented text."
    assert len(segmenter.calls) == 1
    assert len(transcriber.segment_calls) == 1
    assert transcriber.segment_calls[0]["segments"] == [segment]
    assert transcriber.segment_calls[0]["debug"] is True
    assert transcriber.audio_calls == []


def test_transcribe_audio_falls_back_when_vad_returns_no_segments():
    segmenter = FakeSegmenter(segments=[])
    transcriber = FakeTranscriber()
    app = make_app(segmenter, transcriber)
    audio_data = make_audio_data()

    transcript = app._transcribe_audio(audio_data)

    assert transcript.text == "Fallback text."
    assert len(transcriber.segment_calls) == 0
    assert transcriber.audio_calls == [audio_data]


def test_transcribe_audio_falls_back_when_vad_raises():
    segmenter = FakeSegmenter(error=RuntimeError("vad unavailable"))
    transcriber = FakeTranscriber()
    app = make_app(segmenter, transcriber)
    audio_data = make_audio_data()

    transcript = app._transcribe_audio(audio_data)

    assert transcript.text == "Fallback text."
    assert len(transcriber.segment_calls) == 0
    assert transcriber.audio_calls == [audio_data]


def test_transcribe_audio_disables_vad_when_segmenter_init_fails(monkeypatch):
    init_attempts = []

    def raising_segmenter(*args, **kwargs):
        init_attempts.append(kwargs)
        raise RuntimeError("missing webrtcvad")

    monkeypatch.setattr(main_module, "WebRTCVADSegmenter", raising_segmenter)

    transcriber = FakeTranscriber()
    app = make_app(segmenter=None, transcriber=transcriber)
    audio_data = make_audio_data()

    first_text = app._transcribe_audio(audio_data)
    second_text = app._transcribe_audio(audio_data)

    assert first_text.text == "Fallback text."
    assert second_text.text == "Fallback text."
    assert len(init_attempts) == 1
    assert len(transcriber.segment_calls) == 0
    assert transcriber.audio_calls == [audio_data, audio_data]
    assert app._vad_disabled_reason == "VAD unavailable: missing webrtcvad"


def test_get_segmenter_builds_from_vad_config(monkeypatch):
    segmenter = FakeSegmenter(sample_rate=44100)
    captured_settings = None

    def build_segmenter(*, settings):
        nonlocal captured_settings
        captured_settings = settings
        return segmenter

    config = SimpleNamespace(
        vad_aggressiveness=2,
        vad_padding_ms=550,
        vad_silence_duration_ms=650,
    )
    app = make_app(segmenter=None, transcriber=FakeTranscriber(), config=config)

    monkeypatch.setattr(main_module, "WebRTCVADSegmenter", build_segmenter)

    built_segmenter = app._get_segmenter(44100)

    assert built_segmenter is segmenter
    assert captured_settings == VADSettings(
        sample_rate=44100,
        aggressiveness=2,
        start_padding_ms=330,
        end_padding_ms=550,
        silence_duration_ms=650,
    )


def test_on_live_segment_ready_queues_segment_for_background_transcription():
    transcriber = FakeTranscriber()
    app = make_app(segmenter=None, transcriber=transcriber)
    app._start_live_transcription()
    segment = SimpleNamespace(
        segment_id=3,
        start_sample=1600,
        end_sample=3200,
        duration=0.1,
        audio=np.ones(1600, dtype=np.float32),
        sample_rate=16000,
    )

    app._on_live_segment_ready(segment)
    app._stop_live_transcription()

    assert [
        queued_segment.segment_id for queued_segment in transcriber.live_segment_calls
    ] == [3]
    assert app.live_transcript_accumulator.get_text() == "live-3"


def test_init_preloads_whisper_and_ollama_when_enabled(monkeypatch):
    fake_transcriber = FakeTranscriber()
    fake_recorder = FakeRecorder(audio_data=None)
    fake_config = SimpleNamespace(
        ollama_enabled=True,
        ollama_preload_model=True,
        start_with_windows=False,
    )
    identity_calls = []

    monkeypatch.setattr(
        main_module,
        "configure_windows_app_identity",
        lambda: identity_calls.append(True),
    )
    monkeypatch.setattr(main_module, "get_config", lambda: fake_config)
    monkeypatch.setattr(main_module, "AudioRecorder", lambda: fake_recorder)
    monkeypatch.setattr(main_module, "Transcriber", lambda: fake_transcriber)
    monkeypatch.setattr(main_module, "HotkeyManager", lambda: SimpleNamespace())
    monkeypatch.setattr(
        main_module, "get_notification_manager", lambda: SimpleNamespace()
    )
    monkeypatch.setattr(main_module, "get_logger", lambda: SimpleNamespace())
    monkeypatch.setattr(main_module, "TrayManager", lambda **kwargs: SimpleNamespace())
    monkeypatch.setattr(main_module, "get_media_controller", lambda: SimpleNamespace())

    app = main_module.MurmurApp(preload_model=True)

    assert fake_transcriber.load_model_calls == 1
    assert fake_transcriber.warm_llm_post_processor_calls == 1
    assert identity_calls == [True]
    assert fake_recorder.recording_limit_callback.__self__ is app
    assert (
        fake_recorder.recording_limit_callback.__func__
        is main_module.MurmurApp._on_recording_limit_reached
    )


def test_init_skips_ollama_warmup_when_disabled(monkeypatch):
    fake_transcriber = FakeTranscriber()
    fake_config = SimpleNamespace(
        ollama_enabled=False,
        ollama_preload_model=True,
        start_with_windows=False,
    )

    monkeypatch.setattr(main_module, "configure_windows_app_identity", lambda: None)
    monkeypatch.setattr(main_module, "get_config", lambda: fake_config)
    monkeypatch.setattr(main_module, "AudioRecorder", lambda: SimpleNamespace())
    monkeypatch.setattr(main_module, "Transcriber", lambda: fake_transcriber)
    monkeypatch.setattr(main_module, "HotkeyManager", lambda: SimpleNamespace())
    monkeypatch.setattr(
        main_module, "get_notification_manager", lambda: SimpleNamespace()
    )
    monkeypatch.setattr(main_module, "get_logger", lambda: SimpleNamespace())
    monkeypatch.setattr(main_module, "TrayManager", lambda **kwargs: SimpleNamespace())
    monkeypatch.setattr(main_module, "get_media_controller", lambda: SimpleNamespace())

    main_module.MurmurApp(preload_model=True)

    assert fake_transcriber.load_model_calls == 1
    assert fake_transcriber.warm_llm_post_processor_calls == 0


def test_start_live_transcription_replaces_existing_worker():
    transcriber = FakeTranscriber()
    app = make_app(segmenter=None, transcriber=transcriber)

    app._start_live_transcription()
    first_worker = app.live_transcription_worker
    first_accumulator = app.live_transcript_accumulator
    app._start_live_transcription()

    assert first_worker is not app.live_transcription_worker
    assert first_accumulator is not app.live_transcript_accumulator
    app._stop_live_transcription()


def test_live_transcription_callback_stdout_omits_transcript_text(capsys):
    transcriber = FakeTranscriber()
    app = make_app(segmenter=None, transcriber=transcriber)
    chunk = main_module.TranscriptChunk(
        segment_id=7,
        text="private live chunk",
        latency_seconds=0.42,
    )

    app._on_live_segment_transcribed(chunk)
    app._on_live_chunk_appended(chunk, "private live chunk accumulated")
    app._on_live_segment_failed(
        SimpleNamespace(segment_id=7),
        RuntimeError("private live chunk leaked in exception"),
        2,
    )

    stdout = capsys.readouterr().out
    assert "Live segment transcribed: id=7 latency=0.42s text_length=18" in stdout
    assert "Live transcript appended: id=7 current_text_length=30" in stdout
    assert "Live segment failed: id=7 attempts=2" in stdout
    assert "private live chunk" not in stdout
    assert "leaked in exception" not in stdout


@pytest.mark.parametrize("warmup_fails", [False, True])
def test_optional_microphone_warmup_precedes_hotkey_registration(warmup_fails):
    config = FakeStartupConfig(DEFAULT_CONFIG["hotkey"])
    manager = FakeStartupHotkeyManager(config, lambda hotkey: True)
    app = make_start_app(config, manager)
    prepared = []

    def prepare():
        assert manager.register_calls == []
        prepared.append(True)
        if warmup_fails:
            raise OSError("device not ready")
        return True

    app.recorder.prepare = prepare
    app.start()

    assert prepared == [True]
    assert manager.register_calls == [DEFAULT_CONFIG["hotkey"]]
    assert app.notifications.messages[-1] == (
        "murmur",
        "Ready! Press hotkey to start recording.",
    )


def test_start_recovers_from_invalid_configured_hotkey(monkeypatch):
    config = FakeStartupConfig("not-a-real-key")
    hotkey_manager = FakeStartupHotkeyManager(
        config, lambda hotkey: hotkey == DEFAULT_CONFIG["hotkey"]
    )
    app = make_start_app(config, hotkey_manager)

    monkeypatch.setattr(
        main_module,
        "is_hotkey_valid",
        lambda hotkey: hotkey == DEFAULT_CONFIG["hotkey"],
    )

    app.start()

    assert hotkey_manager.register_calls == [
        "not-a-real-key",
        DEFAULT_CONFIG["hotkey"],
    ]
    assert config.set_calls == [("hotkey", DEFAULT_CONFIG["hotkey"])]
    assert app.notifications.messages == [
        (
            "murmur",
            "Configured hotkey 'not-a-real-key' was invalid and has been reset to 'ctrl+shift+space'.",
        ),
        ("murmur", "Ready! Press hotkey to start recording."),
    ]


def test_start_exits_when_valid_hotkey_registration_still_fails(monkeypatch):
    config = FakeStartupConfig(DEFAULT_CONFIG["hotkey"])
    hotkey_manager = FakeStartupHotkeyManager(
        config,
        lambda hotkey: False,
        {DEFAULT_CONFIG["hotkey"]: "keyboard hook unavailable"},
    )
    app = make_start_app(config, hotkey_manager)

    monkeypatch.setattr(main_module, "is_hotkey_valid", lambda hotkey: True)
    monkeypatch.setattr(
        main_module.sys,
        "exit",
        lambda code: (_ for _ in ()).throw(SystemExit(code)),
    )

    with pytest.raises(SystemExit) as exc_info:
        app.start()

    assert exc_info.value.code == 1
    assert hotkey_manager.register_calls == [DEFAULT_CONFIG["hotkey"]]
    assert config.set_calls == []
    assert app.notifications.messages == [
        (
            "murmur",
            "Default hotkey 'ctrl+shift+space' could not be registered (keyboard hook unavailable). Check permissions or whether another application is using it.",
        )
    ]


def test_start_preserves_config_recovery_notice_for_settings():
    config = FakeStartupConfig(
        DEFAULT_CONFIG["hotkey"],
        startup_notice=(
            "Config file was unreadable and has been reset to defaults. "
            "The original file was backed up to 'config.corrupt-20260601010101000000.json'."
        ),
    )
    hotkey_manager = FakeStartupHotkeyManager(config, lambda hotkey: True)
    app = make_start_app(config, hotkey_manager)

    app.start()

    assert hotkey_manager.register_calls == [DEFAULT_CONFIG["hotkey"]]
    assert app.notifications.messages == [
        ("murmur", "Ready! Press hotkey to start recording."),
    ]
    assert config.get_startup_notice() is not None


def test_hotkey_pre_dispatch_cancel_does_not_poison_next_attempt(
    monkeypatch, overlay_calls
):
    app = make_stop_app(make_audio_data())
    app._session_active = False
    app.config.pause_media_while_recording = False
    app.notifications.notify_recording_started = lambda: None
    app.recorder.set_capture_error_callback = lambda callback: None
    app.recorder.set_recording_limit_callback = lambda callback: None
    opened, copied, queued = [], [], []
    app.recorder.start_recording = lambda: opened.append(True)
    monkeypatch.setattr(main_module, "copy_to_clipboard", copied.append)
    monkeypatch.setattr(app, "_start_live_transcription", lambda: None)
    monkeypatch.setattr(app, "_start_live_segmentation", lambda: None)
    monkeypatch.setattr(app, "_stop_live_transcription", lambda: None)
    monkeypatch.setattr(app, "_stop_live_segmentation", lambda: None)

    class HeldThread:
        def __init__(self, *, target, args):
            self.target = target
            self.args = args

        def start(self):
            queued.append(self)

    monkeypatch.setattr("src.hotkey.threading.Thread", HeldThread)
    manager = HotkeyManager.__new__(HotkeyManager)
    manager.config = SimpleNamespace(hotkey="ctrl+shift+space")
    manager._state = HotkeyState.IDLE
    manager._on_start = app._on_recording_start
    manager._on_stop = app._on_recording_stop
    manager._on_state_change = app._on_state_change
    manager._hotkey_registered = False
    manager._last_registration_error = None
    manager._removers = []
    manager._guard = None
    manager._lock = threading.Lock()
    manager._request = None
    manager._stopped = False
    app.hotkey_manager = manager

    # Complete one ordinary session so later pending callbacks cannot inherit it.
    manager._on_hotkey_pressed()
    completed_start = queued.pop(0)
    completed_start.target(*completed_start.args)
    manager._on_hotkey_pressed()
    completed_stop = queued.pop(0)
    completed_stop.target(*completed_stop.args)
    assert opened == [True]
    assert copied == ["Fallback text."]
    assert len(app.logger.entries) == 1

    # Hold attempt A's start callback while the real dispatcher records its stop.
    manager._on_hotkey_pressed()
    first_start = queued.pop(0)
    manager._on_hotkey_pressed()
    first_stop = queued.pop(0)
    first_request = first_start.args[0]
    assert first_request.cancelled.is_set()
    first_stop.target(*first_stop.args)
    first_start.target(*first_start.args)

    assert opened == [True]
    assert copied == ["Fallback text."]
    assert len(app.logger.entries) == 1

    # A second delayed attempt is cancelled before either start callback runs.
    manager._on_hotkey_pressed()
    second_start = queued.pop(0)
    manager._on_hotkey_pressed()
    second_stop = queued.pop(0)
    second_stop.target(*second_stop.args)
    assert manager.get_state() == HotkeyState.IDLE
    first_start.target(*first_start.args)
    second_start.target(*second_start.args)
    assert opened == [True]

    # A fresh dispatch gets a new request and completes the normal path.
    manager._on_hotkey_pressed()
    third_start = queued.pop(0)
    third_start.target(*third_start.args)
    assert opened == [True, True]
    assert any(call[0] == main_module.RECORDING for call in overlay_calls)
    manager._on_hotkey_pressed()
    third_stop = queued.pop(0)
    third_stop.target(*third_stop.args)

    assert copied == ["Fallback text.", "Fallback text."]
    assert len(app.logger.entries) == 2
    assert not queued


def test_start_recovers_from_runtime_hotkey_registration_failure(monkeypatch):
    config = FakeStartupConfig("ctrl+alt+space")
    hotkey_manager = FakeStartupHotkeyManager(
        config,
        lambda hotkey: hotkey == DEFAULT_CONFIG["hotkey"],
        {"ctrl+alt+space": "keyboard hook unavailable"},
    )
    app = make_start_app(config, hotkey_manager)

    monkeypatch.setattr(main_module, "is_hotkey_valid", lambda hotkey: True)

    app.start()

    assert hotkey_manager.register_calls == [
        "ctrl+alt+space",
        DEFAULT_CONFIG["hotkey"],
    ]
    assert config.set_calls == [("hotkey", DEFAULT_CONFIG["hotkey"])]
    assert app.notifications.messages == [
        (
            "murmur",
            "Configured hotkey 'ctrl+alt+space' could not be registered (keyboard hook unavailable) and has been reset to 'ctrl+shift+space'.",
        ),
        ("murmur", "Ready! Press hotkey to start recording."),
    ]


def test_finalize_recording_uses_live_transcript_before_offline_fallback(
    monkeypatch, capsys
):
    transcriber = FakeTranscriber()
    app = make_app(segmenter=None, transcriber=transcriber)
    app.live_transcript_accumulator = main_module.TranscriptAccumulator()
    app.live_transcript_accumulator.add_chunk(
        SimpleNamespace(segment_id=0, text="hello world", latency_seconds=0.1)
    )
    app.live_transcript_accumulator.add_chunk(
        SimpleNamespace(segment_id=1, text="again", latency_seconds=99.0)
    )
    app.tray = FakeTray()
    app.notifications = FakeNotifications()
    app.logger = FakeLogger()
    app.hotkey_manager = FakeHotkeyManager()

    copied_text = []
    monkeypatch.setattr(
        main_module, "copy_to_clipboard", lambda text: copied_text.append(text) or True
    )
    monkeypatch.setattr(main_module.time, "perf_counter", lambda: 11.25)

    audio_data = make_audio_data()
    app._finalize_recording(audio_data, finalization_started_at=10.0)

    assert copied_text == ["FINAL:hello world again"]
    assert transcriber.audio_calls == []
    assert transcriber.segment_calls == []
    assert transcriber.finalize_segment_calls == [["hello world", "again"]]
    assert transcriber.finalize_calls == ["hello world again"]
    assert app.notifications.copied_to_clipboard == 1
    assert app.notifications.completed == []
    [(logged_audio, logged_text)] = app.logger.entries
    assert logged_audio is audio_data
    assert logged_text == "FINAL:hello world again"
    assert app.logger.cleanup_details == [
        {
            "cleanup_input": "hello world again",
            "cleanup_status": "success",
            "cleanup_model": None,
            "session_id": None,
        }
    ]
    stdout = capsys.readouterr().out
    assert "Live segments: count=2 avg_latency=49.55s max_latency=99.00s" in stdout
    # Readiness belongs to the stop owner after finalization and worker teardown.
    assert app.tray.statuses == []
    assert app.hotkey_manager.processing_calls == 1
    assert app.hotkey_manager.idle_calls == 0


def test_complete_transcription_stdout_omits_transcript_on_success(
    monkeypatch,
    capsys,
):
    transcriber = FakeTranscriber()
    app = make_app(segmenter=None, transcriber=transcriber)
    app.notifications = FakeNotifications()
    app.logger = FakeLogger()

    copied_text = []
    monkeypatch.setattr(
        main_module, "copy_to_clipboard", lambda text: copied_text.append(text) or True
    )
    monkeypatch.setattr(main_module.time, "perf_counter", lambda: 2.2)

    app._complete_transcription(
        make_audio_data(),
        FinalTranscript("private dictated text"),
        finalization_started_at=1.0,
        live_segment_metrics=main_module.LiveSegmentMetrics.empty(),
    )

    stdout = capsys.readouterr().out
    assert "Finalized in 1.2s; copied to clipboard." in stdout
    assert "Live segments: count=0 avg_latency=n/a max_latency=n/a" in stdout
    assert "private dictated text" not in stdout
    assert copied_text == ["private dictated text"]
    assert app.notifications.copied_to_clipboard == 1
    assert app.notifications.completed == []


def test_complete_transcription_copies_before_training_data_logging(monkeypatch):
    transcriber = FakeTranscriber()
    app = make_app(segmenter=None, transcriber=transcriber)
    app.notifications = FakeNotifications()
    call_order = []

    class OrderedLogger(FakeLogger):
        def log(self, audio_data, text, **cleanup):
            call_order.append("log")
            return super().log(audio_data, text, **cleanup)

    app.logger = OrderedLogger()

    def copy(text):
        call_order.append("copy")
        return True

    monkeypatch.setattr(main_module, "copy_to_clipboard", copy)
    monkeypatch.setattr(main_module.time, "perf_counter", lambda: 2.2)

    app._complete_transcription(
        make_audio_data(),
        FinalTranscript("private dictated text"),
        finalization_started_at=1.0,
        live_segment_metrics=main_module.LiveSegmentMetrics.empty(),
    )

    assert call_order == ["copy", "log"]
    assert len(app.logger.entries) == 1


def test_process_audio_logs_empty_live_segment_metrics_for_offline_fallback(
    monkeypatch,
    capsys,
):
    transcriber = FakeTranscriber()
    app = make_app(segmenter=None, transcriber=transcriber)
    app.notifications = FakeNotifications()
    app.logger = FakeLogger()
    app.tray = FakeTray()
    app.hotkey_manager = FakeHotkeyManager()

    monkeypatch.setattr(main_module, "copy_to_clipboard", lambda text: True)
    monkeypatch.setattr(main_module.time, "perf_counter", lambda: 11.5)

    audio_data = make_audio_data()
    app._process_audio(audio_data, finalization_started_at=10.0)

    stdout = capsys.readouterr().out
    assert "Live segments: count=0 avg_latency=n/a max_latency=n/a" in stdout
    [(logged_audio, logged_text)] = app.logger.entries
    assert logged_audio is audio_data
    assert logged_text == "Fallback text."


def test_complete_transcription_stdout_omits_transcript_on_clipboard_failure(
    monkeypatch,
    capsys,
):
    transcriber = FakeTranscriber()
    app = make_app(segmenter=None, transcriber=transcriber)
    app.notifications = FakeNotifications()
    app.logger = FakeLogger()
    live_segment_metrics = main_module.LiveSegmentMetrics(
        segment_count=2,
        latency_avg_seconds=0.25,
        latency_max_seconds=0.4,
    )

    monkeypatch.setattr(main_module, "copy_to_clipboard", lambda text: False)
    monkeypatch.setattr(main_module.time, "perf_counter", lambda: 2.5)

    audio_data = make_audio_data()
    app._complete_transcription(
        audio_data,
        FinalTranscript("private dictated text"),
        finalization_started_at=1.0,
        live_segment_metrics=live_segment_metrics,
    )

    stdout = capsys.readouterr().out
    assert "Finalized in 1.5s; failed to copy to clipboard." in stdout
    assert "Live segments: count=2 avg_latency=0.25s max_latency=0.40s" in stdout
    assert "private dictated text" not in stdout
    assert app.notifications.errors == [
        "Clipboard copy failed. Please retry the recording. The transcript was saved to training data."
    ]
    [(logged_audio, logged_text)] = app.logger.entries
    assert logged_audio is audio_data
    assert logged_text == "private dictated text"


def test_complete_transcription_clipboard_failure_retry_message_when_logging_disabled(
    monkeypatch,
):
    transcriber = FakeTranscriber()
    app = make_app(segmenter=None, transcriber=transcriber)
    app.notifications = FakeNotifications()
    app.logger = FakeLogger(enabled=False)

    monkeypatch.setattr(main_module, "copy_to_clipboard", lambda text: False)
    monkeypatch.setattr(main_module.time, "perf_counter", lambda: 2.5)

    app._complete_transcription(
        make_audio_data(),
        FinalTranscript("private dictated text"),
        finalization_started_at=1.0,
        live_segment_metrics=main_module.LiveSegmentMetrics.empty(),
    )

    assert app.notifications.errors == [
        "Clipboard copy failed. Please retry the recording."
    ]
    assert app.logger.entries == []


def test_process_audio_error_stdout_omits_exception_text(monkeypatch, capsys):
    transcriber = FakeTranscriber()
    app = make_app(segmenter=None, transcriber=transcriber)
    app.notifications = FakeNotifications()
    app.tray = FakeTray()
    app.hotkey_manager = FakeHotkeyManager()

    def fail_transcription(audio_data):
        raise RuntimeError("private dictated text leaked in exception")

    monkeypatch.setattr(app, "_transcribe_audio", fail_transcription)

    app._process_audio(make_audio_data(), finalization_started_at=10.0)

    stdout = capsys.readouterr().out
    assert "Transcription failed." in stdout
    assert "private dictated text" not in stdout
    assert "leaked in exception" not in stdout
    assert app.notifications.errors == ["Transcription failed."]


def test_finalize_recording_falls_back_to_offline_processing_when_live_text_missing(
    monkeypatch,
):
    transcriber = FakeTranscriber()
    app = make_app(segmenter=None, transcriber=transcriber)
    called = []
    audio_data = make_audio_data()

    monkeypatch.setattr(
        app,
        "_process_audio",
        lambda provided_audio, *, finalization_started_at: called.append(
            (provided_audio, finalization_started_at)
        ),
    )

    app._finalize_recording(audio_data, finalization_started_at=10.0)

    assert called == [(audio_data, 10.0)]


def test_finalize_recording_falls_back_to_offline_processing_when_live_pipeline_degraded(
    monkeypatch,
):
    transcriber = FakeTranscriber()
    app = make_app(segmenter=None, transcriber=transcriber)
    app.live_transcript_accumulator = main_module.TranscriptAccumulator()
    app.live_transcript_accumulator.add_chunk(
        SimpleNamespace(segment_id=0, text="partial live", latency_seconds=0.1)
    )
    app._live_pipeline_degraded = True
    app._live_pipeline_degraded_reason = "live segment failed"

    called = []
    audio_data = make_audio_data()
    monkeypatch.setattr(
        app,
        "_process_audio",
        lambda provided_audio, *, finalization_started_at: called.append(
            (provided_audio, finalization_started_at)
        ),
    )

    app._finalize_recording(audio_data, finalization_started_at=10.0)

    assert called == [(audio_data, 10.0)]
    assert transcriber.finalize_calls == []


def test_on_live_pipeline_degraded_marks_app_once_without_user_notification():
    transcriber = FakeTranscriber()
    app = make_app(segmenter=None, transcriber=transcriber)
    app.notifications = FakeNotifications()

    app._on_live_pipeline_degraded("worker failed")
    app._on_live_pipeline_degraded("later failure")

    assert app._live_pipeline_degraded is True
    assert app._live_pipeline_degraded_reason == "worker failed"
    assert app.notifications.messages == []


def test_build_live_segmenter_wires_degradation_callback(monkeypatch):
    transcriber = FakeTranscriber()
    app = make_app(segmenter=None, transcriber=transcriber)
    captured = {}

    class FakeLiveVADSegmentationWorker:
        def __init__(self, **kwargs):
            captured.update(kwargs)

    monkeypatch.setattr(
        main_module,
        "LiveVADSegmentationWorker",
        FakeLiveVADSegmentationWorker,
    )

    worker = app._build_live_segmenter(16000)

    assert isinstance(worker, FakeLiveVADSegmentationWorker)
    assert captured["settings"] == VADSettings.from_app_config(
        app.config,
        sample_rate=16000,
    )
    assert captured["on_segment"] == app._on_live_segment_ready

    captured["on_worker_degraded"]("live VAD failed")

    assert app._live_pipeline_degraded is True
    assert app._live_pipeline_degraded_reason == "live VAD failed"


def test_start_live_segmentation_registers_recorder_callbacks():
    transcriber = FakeTranscriber()
    recorder = FakeRecorder(audio_data=None)
    worker = SimpleNamespace(start_calls=0, submit_audio_block=lambda block: block)

    def start_worker():
        worker.start_calls += 1

    worker.start = start_worker

    app = make_app(segmenter=None, transcriber=transcriber)
    app.recorder = recorder
    app.live_segmenter = None
    app.notifications = FakeNotifications()
    app._build_live_segmenter = lambda sample_rate: worker

    app._start_live_segmentation()

    assert worker.start_calls == 1
    assert app.live_segmenter is worker
    assert recorder.block_callback == worker.submit_audio_block
    assert recorder.block_callback_error_handler == app._on_live_block_callback_error


def test_stop_live_segmentation_clears_recorder_callbacks_and_stops_worker():
    transcriber = FakeTranscriber()
    recorder = FakeRecorder(audio_data=None)
    stopped = []
    worker = SimpleNamespace(stop=lambda: stopped.append(True))

    app = make_app(segmenter=None, transcriber=transcriber)
    app.recorder = recorder
    app.live_segmenter = worker

    app._stop_live_segmentation()

    assert recorder.block_callback is None
    assert recorder.block_callback_error_handler is None
    assert stopped == [True]
    assert app.live_segmenter is None


def test_on_live_block_callback_error_marks_pipeline_degraded_once():
    transcriber = FakeTranscriber()
    app = make_app(segmenter=None, transcriber=transcriber)
    app.notifications = FakeNotifications()

    app._on_live_block_callback_error(RuntimeError("queue submit failed"))
    app._on_live_block_callback_error(RuntimeError("later failure"))

    assert app._live_pipeline_degraded is True
    assert app._live_pipeline_degraded_reason == "Live audio callback failed."
    assert app.notifications.messages == []


def test_recording_limit_handler_notifies_and_uses_stop_flow():
    transcriber = FakeTranscriber()
    app = make_app(segmenter=None, transcriber=transcriber)
    app.notifications = FakeNotifications()
    app.hotkey_manager = FakeHotkeyManager()
    stop_calls = []
    app._finish_recording_stop = lambda: stop_calls.append(True)

    app._handle_recording_limit_reached(300)

    assert app.notifications.recording_limits == [300]
    assert app.hotkey_manager.processing_calls == 1
    assert stop_calls == [True]


def test_capture_error_reports_early_end_and_uses_stop_flow_once():
    app = make_app(segmenter=None, transcriber=FakeTranscriber())
    app.notifications = FakeNotifications()
    app.hotkey_manager = FakeHotkeyManager()
    app.tray = FakeTray()
    app._recording_limit_stop_lock = threading.Lock()
    app._recording_limit_stop_started = False
    app._recording_stop_lock = threading.Lock()
    app._recording_stop_started = False
    stop_calls = []
    app._finish_recording_stop = lambda: stop_calls.append(True)

    app._on_capture_error("Microphone disconnected or stopped")
    app._on_capture_error("Microphone disconnected or stopped")

    assert stop_calls == [True]
    assert app.hotkey_manager.processing_calls == 1
    assert app.tray.statuses == ["Microphone error"]
    assert "finalizing captured audio" in app.notifications.errors[0]


def test_late_capture_error_does_not_strand_hotkey_after_manual_stop():
    app = make_app(segmenter=None, transcriber=FakeTranscriber())
    app.notifications = FakeNotifications()
    app.hotkey_manager = FakeHotkeyManager()
    app.tray = FakeTray()
    app._recording_limit_stop_lock = threading.Lock()
    app._recording_limit_stop_started = False
    app._recording_stop_lock = threading.Lock()
    app._recording_stop_started = False
    stop_calls = []
    app._finish_recording_stop = lambda: stop_calls.append(True)

    stop_current(app)
    app._on_capture_error("Microphone disconnected or stopped")

    assert stop_calls == [True]
    assert app.hotkey_manager.processing_calls == 0
    assert app.notifications.errors == []


@pytest.mark.parametrize(
    ("enabled", "playing", "pause_succeeded", "recording_failed", "expected_plays"),
    [
        (True, True, True, False, 1),
        (True, True, False, False, 0),
        (True, False, True, False, 0),
        (False, True, True, False, 0),
        (True, True, True, True, 0),
        (True, True, False, True, 0),
    ],
)
def test_media_resumes_only_after_successful_pause(
    monkeypatch, enabled, playing, pause_succeeded, recording_failed, expected_plays
):
    app = make_app(segmenter=None, transcriber=FakeTranscriber())
    app.config.pause_media_while_recording = enabled
    app._paused_media_source = "media.app"
    app.notifications = FakeNotifications()
    app.notifications.notify_recording_started = lambda: None
    app.hotkey_manager = FakeHotkeyManager()
    app.tray = FakeTray()
    app.recorder = FakeRecorder(None)
    app.recorder.set_capture_error_callback = lambda callback: None
    plays = []
    pauses = []
    app.media_controller = SimpleNamespace(
        is_media_playing=lambda: playing,
        pause=lambda: pauses.append(True) or ("media.app" if pause_succeeded else None),
        play=lambda source_id: plays.append(source_id) or True,
    )

    def start_recording():
        if recording_failed:
            raise RuntimeError("microphone unavailable")

    app.recorder.start_recording = start_recording
    for name in (
        "_start_live_transcription",
        "_start_live_segmentation",
        "_stop_live_transcription",
        "_stop_live_segmentation",
    ):
        monkeypatch.setattr(app, name, lambda: None)
    app._on_recording_start(RecordingRequest())
    if not recording_failed:
        if app._media_pause_thread is not None:
            app._media_pause_thread.join(2)
            assert not app._media_pause_thread.is_alive()
        stop_current(app)
    assert len(pauses) == int(enabled and playing and not recording_failed)
    assert plays == ["media.app"] * expected_plays
    assert app._paused_media_source is None


def test_accepted_capture_error_cannot_stop_next_session(monkeypatch):
    app = make_app(segmenter=None, transcriber=FakeTranscriber())
    app.config.pause_media_while_recording = False
    app._paused_media_source = None
    app.notifications = FakeNotifications()
    app.notifications.notify_recording_started = lambda: None
    app.hotkey_manager = FakeHotkeyManager()
    app.tray = FakeTray()
    app.recorder = FakeRecorder(None)
    callbacks = []
    app.recorder.set_capture_error_callback = callbacks.append
    app.recorder.start_recording = lambda: None
    for name in (
        "_start_live_transcription",
        "_start_live_segmentation",
        "_stop_live_transcription",
        "_stop_live_segmentation",
    ):
        monkeypatch.setattr(app, name, lambda: None)

    app._on_recording_start(RecordingRequest())
    accepted = threading.Event()
    release = threading.Event()
    old_callback = callbacks[-1]

    def delayed_error():
        accepted.set()
        release.wait(2)
        old_callback("old stream stopped")

    worker = threading.Thread(target=delayed_error)
    worker.start()
    try:
        assert accepted.wait(2)
        stop_current(app)
        app._on_recording_start(RecordingRequest())
    finally:
        release.set()
        worker.join(2)

    assert not worker.is_alive()
    assert app.recorder.stop_calls == 1
    assert app._recording_stop_started is False
    assert app.hotkey_manager.processing_calls == 0
    assert app.notifications.errors == []
    # The current session's error still claims and completes its stop.
    callbacks[-1]("current stream stopped")
    assert app.recorder.stop_calls == 2
    assert app.hotkey_manager.processing_calls == 1


def test_on_recording_start_leaves_recorder_reusable_after_failed_stream_start(
    monkeypatch,
    overlay_calls,
):
    fake_sd = SimpleNamespace(streams=[], fail_next_start=True)

    class FakeStream:
        def __init__(self, **kwargs):
            self.started = False
            self.closed = False
            fake_sd.streams.append(self)

        def start(self):
            if fake_sd.fail_next_start:
                fake_sd.fail_next_start = False
                raise OSError("device busy")
            self.started = True
            overlay_calls.append("stream-started")

        @property
        def active(self):
            return self.started and not self.closed

        def close(self):
            self.closed = True

    fake_sd.InputStream = FakeStream
    fake_sd.default = SimpleNamespace(device=(0, -1))
    fake_sd.query_hostapis = lambda: [{"name": "Windows WASAPI"}]
    fake_sd.query_devices = lambda: [
        {"name": "Default microphone", "hostapi": 0, "max_input_channels": 1}
    ]
    monkeypatch.setattr("src.audio.sd", fake_sd)

    app = make_app(segmenter=None, transcriber=FakeTranscriber())
    app.config.pause_media_while_recording = False
    app.recorder = main_module.AudioRecorder(
        config=SimpleNamespace(get=lambda key, default=None: default)
    )
    app.tray = FakeTray()
    app.notifications = FakeNotifications()
    app.notifications.notify_recording_started = lambda: None
    app.hotkey_manager = FakeHotkeyManager()
    app._paused_media_source = None
    live_calls = []
    monkeypatch.setattr(app, "_start_live_transcription", lambda: None)
    monkeypatch.setattr(app, "_start_live_segmentation", lambda: None)
    monkeypatch.setattr(
        app, "_stop_live_transcription", lambda: live_calls.append("transcription")
    )
    monkeypatch.setattr(
        app, "_stop_live_segmentation", lambda: live_calls.append("segmentation")
    )

    app._on_recording_start(RecordingRequest())

    [failed_stream] = fake_sd.streams
    assert failed_stream.closed is True
    assert app.recorder.is_recording() is False
    assert app.tray.statuses[-1] == "Error"
    assert app.hotkey_manager.idle_calls == 1
    assert live_calls == ["transcription", "segmentation"]
    # A failed start shows the capture error and never the recording pulse.
    assert overlay_calls == [("error", 1, "Couldn't start recording")]

    app._on_recording_start(RecordingRequest())

    retry_stream = fake_sd.streams[-1]
    assert retry_stream.started is True
    assert app.recorder.is_recording() is True
    assert app.tray.statuses[-1] == "Recording..."
    assert app.hotkey_manager.idle_calls == 1
    # The recording pulse is requested only after the input stream started.
    assert overlay_calls[1:] == ["stream-started", ("recording", 2, "")]


def test_on_recording_stop_finalizes_live_pipeline_and_resumes_media(
    monkeypatch, capsys
):
    transcriber = FakeTranscriber()
    audio_data = make_audio_data()
    app = make_app(segmenter=None, transcriber=transcriber)
    app.recorder = FakeRecorder(audio_data)
    app.tray = FakeTray()
    app.notifications = FakeNotifications()
    app.logger = FakeLogger()
    app.hotkey_manager = FakeHotkeyManager()
    app.media_controller = FakeMediaController()
    app._paused_media_source = "media.app"
    app.live_transcript_accumulator = main_module.TranscriptAccumulator()
    app.live_transcript_accumulator.add_chunk(
        SimpleNamespace(segment_id=0, text="tail kept", latency_seconds=0.1)
    )

    stop_order = []
    segmentation_stops = []
    transcription_stops = []
    copied_text = []
    play_media = app.media_controller.play
    monkeypatch.setattr(
        app.recorder,
        "stop_recording",
        lambda: stop_order.append("recorder") or audio_data,
    )
    monkeypatch.setattr(
        app.media_controller,
        "play",
        lambda source_id: stop_order.append("media") or play_media(source_id),
    )
    monkeypatch.setattr(
        app,
        "_stop_live_segmentation",
        lambda: stop_order.append("segmentation") or segmentation_stops.append(True),
    )
    monkeypatch.setattr(
        app,
        "_stop_live_transcription",
        lambda: stop_order.append("transcription") or transcription_stops.append(True),
    )
    monkeypatch.setattr(
        main_module, "copy_to_clipboard", lambda text: copied_text.append(text) or True
    )
    perf_counter_values = iter([20.0, 20.5, 20.75, 20.75])
    monkeypatch.setattr(
        main_module.time, "perf_counter", lambda: next(perf_counter_values)
    )

    stop_current(app)

    assert app.tray.statuses[0] == "Finalizing..."
    assert stop_order == ["recorder", "media", "segmentation", "transcription"]
    assert segmentation_stops == [True]
    assert transcription_stops == [True]
    assert copied_text == ["FINAL:tail kept"]
    [(logged_audio, logged_text)] = app.logger.entries
    assert logged_audio is audio_data
    assert logged_text == "FINAL:tail kept"
    stdout = capsys.readouterr().out
    assert "Live segments: count=1 avg_latency=0.10s max_latency=0.10s" in stdout
    assert app.media_controller.play_calls == 1
    assert app.hotkey_manager.processing_calls == 1
    assert app.hotkey_manager.idle_calls == 1


def make_stop_app(audio_data):
    app = make_app(segmenter=None, transcriber=FakeTranscriber())
    app.recorder = FakeRecorder(audio_data)
    app.tray = FakeTray()
    app.notifications = FakeNotifications()
    app.logger = FakeLogger()
    app.hotkey_manager = FakeHotkeyManager()
    app.media_controller = FakeMediaController()
    app._paused_media_source = None
    app._overlay_session = 7
    app._stop_live_segmentation = lambda: None
    app._stop_live_transcription = lambda: None
    return app


@pytest.mark.parametrize(
    ("failure_stage", "error", "exit_code"),
    [
        (None, None, None),
        ("tray", KeyboardInterrupt(), None),
        ("tray", RuntimeError("tray failed"), 1),
        ("start", RuntimeError("startup failed"), 1),
        ("start", SystemExit(7), 7),
        ("construct", RuntimeError("construction failed"), 1),
    ],
    ids=[
        "normal-return",
        "tray-interrupt",
        "tray-error",
        "startup-error",
        "startup-exit",
        "construction-error",
    ],
)
@pytest.mark.parametrize("platform", ["win32", "linux"])
def test_main_stops_constructed_app_on_exit(
    monkeypatch, failure_stage, error, exit_code, platform
):
    calls = []
    tray_threads = []
    monkeypatch.setattr(main_module.sys, "platform", platform)

    def start():
        calls.append("start")
        if failure_stage == "start":
            raise error

    def run_tray():
        calls.append("tray")
        tray_threads.append(threading.current_thread())
        if failure_stage == "tray":
            raise error

    app = SimpleNamespace(
        start=start,
        tray=SimpleNamespace(run=run_tray),
        stop=lambda: calls.append("stop"),
    )

    def construct_app(*, preload_model):
        assert preload_model is True
        calls.append("construct")
        if failure_stage == "construct":
            raise error
        return app

    monkeypatch.setattr(main_module, "MurmurApp", construct_app)

    if exit_code is None:
        main_module.main()
    else:
        with pytest.raises(SystemExit) as exc_info:
            main_module.main()
        assert exc_info.value.code == exit_code

    expected = ["construct"]
    if failure_stage != "construct":
        expected.append("start")
        if failure_stage != "start":
            expected.append("tray")
        expected.append("stop")
    assert calls == expected
    if tray_threads:
        assert (tray_threads[0] is threading.main_thread()) == (platform != "win32")
        if platform == "win32":
            assert not tray_threads[0].is_alive()


def test_windows_main_interrupt_stops_and_joins_blocked_tray(monkeypatch, capsys):
    monkeypatch.setattr(main_module.sys, "platform", "win32")
    released = threading.Event()
    tray_threads = []
    calls = []

    def run_tray():
        tray_threads.append(threading.current_thread())
        assert threading.current_thread() is not threading.main_thread()
        # Simulate a console interrupt while the tray cannot return by itself.
        _thread.interrupt_main()
        assert released.wait(5), "shutdown did not release the tray worker"
        calls.append("tray-finished")

    def stop():
        calls.append("stop")
        released.set()

    app = SimpleNamespace(
        start=lambda: None,
        tray=SimpleNamespace(run=run_tray),
        stop=stop,
    )
    monkeypatch.setattr(main_module, "MurmurApp", lambda **kwargs: app)

    main_module.main()

    assert calls == ["stop", "tray-finished"]
    assert not tray_threads[0].is_alive()
    assert "Interrupted by user." in capsys.readouterr().out


@pytest.mark.parametrize(
    ("entrypoint", "encoding"),
    [
        ("script", "cp1252"),
        ("script", "ascii"),
        ("script", None),
        ("console", "cp1252"),
        ("console", "ascii"),
        ("module", "cp1252"),
        ("module", "ascii"),
    ],
)
@pytest.mark.parametrize("outcome", ["copied", "no_audio", "start_failure"])
def test_entrypoints_record_with_restricted_output(
    monkeypatch, entrypoint, encoding, outcome
):
    """Exercise real start/stop and offline finalization without desktop hardware."""
    import run

    app = make_stop_app(None if outcome == "no_audio" else make_audio_data())
    app.segmenter = FakeSegmenter()
    app.config.pause_media_while_recording = True
    app.notifications.notify_recording_started = lambda: None
    app.recorder.set_capture_error_callback = lambda callback: None
    starts = []
    copies = []

    def start_recording():
        starts.append(True)
        if outcome == "start_failure":
            raise RuntimeError("microphone unavailable")

    app.recorder.start_recording = start_recording
    # Exercise background media warning output and the offline VAD fallback.
    app.media_controller.is_media_playing = lambda: True
    app.media_controller.pause = lambda: None
    app._start_live_transcription = lambda: None
    app._start_live_segmentation = lambda: None

    def start_app():
        app._on_recording_start(RecordingRequest())
        if app._media_pause_thread is not None:
            app._media_pause_thread.join(2)
            assert not app._media_pause_thread.is_alive()

    app.start = start_app
    app.tray.run = (
        (lambda: stop_current(app)) if outcome != "start_failure" else lambda: None
    )
    app.stop = lambda: None
    monkeypatch.setattr(main_module, "MurmurApp", lambda **kwargs: app)
    monkeypatch.setattr(
        main_module, "copy_to_clipboard", lambda text: copies.append(text) or True
    )
    monkeypatch.setattr(sys, "argv", ["run.py"])

    with (
        io.BytesIO() as output,
        io.TextIOWrapper(
            output, encoding=encoding or "ascii", errors="strict"
        ) as stream,
        monkeypatch.context() as streams,
    ):
        streams.setattr(sys, "stdout", stream if encoding else None)
        streams.setattr(sys, "stderr", stream if encoding else None)
        try:
            if entrypoint == "script":
                # run.py is also used by pythonw, the VBS launcher and packaging.
                runpy.run_path(run.__file__, run_name="__main__")
            elif entrypoint == "module":
                runpy.run_module("src", run_name="__main__")
            else:
                console = next(
                    entry
                    for entry in importlib.metadata.distribution("murmur").entry_points
                    if entry.name == "murmur" and entry.group == "console_scripts"
                )
                console.load()()
            if encoding:
                assert sys.stdout is stream
                stream.flush()
                messages = output.getvalue().decode(encoding)
                assert ("Recording started..." in messages) == (
                    outcome != "start_failure"
                )
                assert ("Warning: Failed to pause media playback" in messages) == (
                    outcome != "start_failure"
                )
                if outcome != "start_failure":
                    assert "Recording stopped." in messages
                if outcome == "copied":
                    assert "copied to clipboard." in messages
        finally:
            if encoding is None:
                sys.stdout.close()
                sys.stderr.close()

    assert starts == [True]
    assert app.recorder.stop_calls == int(outcome != "start_failure")
    assert copies == (["Fallback text."] if outcome == "copied" else [])
    assert app.hotkey_manager.idle_calls == 1
    assert app._session_active is False
    assert not app.notifications.errors or outcome == "start_failure"


def test_stop_shows_processing_then_copied(monkeypatch, overlay_calls):
    app = make_stop_app(make_audio_data())
    monkeypatch.setattr(main_module, "copy_to_clipboard", lambda text: True)

    stop_current(app)

    assert overlay_calls == [("processing", 7, ""), ("success", 7, "")]


@pytest.mark.parametrize("live", [False, True])
def test_overflow_warns_once_at_final_delivery_without_changing_text(
    monkeypatch, overlay_calls, capsys, live
):
    audio = make_audio_data()
    audio.input_overflowed = True
    app = make_stop_app(audio)
    if live:
        app.live_transcript_accumulator = main_module.TranscriptAccumulator()
        app.live_transcript_accumulator.add_chunk(
            SimpleNamespace(segment_id=0, text="private text", latency_seconds=0.1)
        )
    else:
        monkeypatch.setattr(
            app, "_transcribe_audio", lambda audio: FinalTranscript("private text")
        )
    copied = []
    monkeypatch.setattr(
        main_module, "copy_to_clipboard", lambda text: copied.append(text) or True
    )

    stop_current(app)

    expected = "FINAL:private text" if live else "private text"
    assert copied == [expected]
    assert app.logger.entries[0][0] is audio
    assert app.logger.entries[0][1] == expected
    assert app.notifications.copied_to_clipboard == 0
    assert app.notifications.messages == [
        ("murmur", "Some audio was lost. Check the copied text.")
    ]
    assert overlay_calls == [
        ("processing", 7, ""),
        ("error", 7, "Copied; some audio lost"),
    ]
    stdout = capsys.readouterr().out
    assert stdout.count("Warning: Microphone input overflow;") == 1
    assert "private text" not in stdout
    assert "private text" not in repr(overlay_calls)


@pytest.mark.parametrize("input_overflowed", [False, True])
@pytest.mark.parametrize(
    ("setup", "message"),
    [
        ("no_audio", "No audio captured"),
        ("no_speech", "No speech detected"),
        ("transcription_error", "Transcription failed"),
        ("clipboard_error", "Couldn't copy to clipboard"),
    ],
)
def test_stop_failures_show_fixed_error_without_transcript(
    monkeypatch, overlay_calls, setup, message, input_overflowed
):
    audio = make_audio_data()
    audio.input_overflowed = input_overflowed
    app = make_stop_app(None if setup == "no_audio" else audio)
    monkeypatch.setattr(main_module, "copy_to_clipboard", lambda text: False)
    if setup == "no_speech":
        monkeypatch.setattr(
            app, "_transcribe_audio", lambda audio_data: FinalTranscript("")
        )
    elif setup == "transcription_error":

        def fail(audio_data):
            raise RuntimeError("private dictated text")

        monkeypatch.setattr(app, "_transcribe_audio", fail)

    stop_current(app)

    assert overlay_calls[-1] == ("error", 7, message)
    expected_processing = [] if setup == "no_audio" else [("processing", 7, "")]
    assert overlay_calls[:-1] == expected_processing
    assert "Fallback text" not in repr(overlay_calls)
    assert app.hotkey_manager.idle_calls == 1
    assert app.notifications.copied_to_clipboard == 0
    assert not any(
        "Some audio was lost" in msg for _, msg in app.notifications.messages
    )


def test_live_finalization_error_reports_before_returning_to_idle(
    monkeypatch, overlay_calls
):
    app = make_stop_app(make_audio_data())
    app.live_transcript_accumulator = main_module.TranscriptAccumulator()
    app.live_transcript_accumulator.add_chunk(
        SimpleNamespace(segment_id=0, text="live text", latency_seconds=0.1)
    )
    events = overlay_calls

    def fail_copy(text):
        raise RuntimeError("clipboard exploded")

    monkeypatch.setattr(main_module, "copy_to_clipboard", fail_copy)
    app.hotkey_manager.set_idle = lambda: events.append("idle")

    with pytest.raises(RuntimeError):
        stop_current(app)

    assert events[-2:] == [("error", 7, "Transcription failed"), "idle"]


def test_disabled_overlay_sends_nothing_but_dictation_still_copies(
    monkeypatch, overlay_calls
):
    app = make_stop_app(make_audio_data())
    app.config.show_recording_overlay = False
    copied = []
    monkeypatch.setattr(
        main_module, "copy_to_clipboard", lambda text: copied.append(text) or True
    )

    app._show_overlay(main_module.HIDDEN)
    stop_current(app)

    assert overlay_calls == []
    assert copied == ["Fallback text."]
    assert app.notifications.copied_to_clipboard == 1
    assert app.tray.statuses[-1] == "Ready"


def test_disabling_overlay_mid_session_clears_shown_overlay(overlay_calls):
    app = make_stop_app(make_audio_data())
    app._show_overlay(main_module.RECORDING)
    app.config.show_recording_overlay = False

    app._show_overlay(main_module.PROCESSING)
    app._show_overlay(main_module.SUCCESS)

    assert overlay_calls == [("recording", 7, ""), "close"]


def test_overlay_failure_does_not_interrupt_dictation(monkeypatch, capsys):
    settings_gui = pytest.importorskip("src.settings_gui")

    def broken_overlay(*args, **kwargs):
        raise RuntimeError("overlay broke")

    monkeypatch.setattr(settings_gui, "request_overlay_state", broken_overlay)
    app = make_stop_app(make_audio_data())
    copied = []
    monkeypatch.setattr(
        main_module, "copy_to_clipboard", lambda text: copied.append(text) or True
    )

    stop_current(app)

    assert copied == ["Fallback text."]
    assert app.notifications.copied_to_clipboard == 1
    assert app.hotkey_manager.idle_calls == 1
    assert "Recording overlay request failed." in capsys.readouterr().out


@pytest.mark.parametrize("input_overflowed", [False, True])
def test_cancel_capture_discards_session_and_resumes_media(
    monkeypatch, overlay_calls, input_overflowed
):
    audio = make_audio_data()
    audio.input_overflowed = input_overflowed
    app = make_stop_app(audio)
    app._paused_media_source = "media.app"
    copied = []
    monkeypatch.setattr(main_module, "copy_to_clipboard", copied.append)

    app._on_recording_cancel(7)
    app._on_recording_cancel(7)

    assert app.recorder.stop_calls == 1
    assert app.media_controller.play_calls == 1
    assert app.transcriber.audio_calls == []
    assert app.transcriber.finalize_calls == []
    assert copied == []
    assert app.logger.entries == []
    assert app.hotkey_manager.idle_calls == 1
    assert not app._can_cancel()
    assert overlay_calls == [("cancelled", 7, ""), ("cancelled_done", 7, "")]


def test_cancel_during_processing_waits_for_call_and_suppresses_delivery(
    monkeypatch, overlay_calls
):
    app = make_stop_app(make_audio_data())
    entered, release = threading.Event(), threading.Event()
    copied = []
    errors = []

    def transcribe(audio_data):
        entered.set()
        assert release.wait(2)
        return FinalTranscript("late result")

    def stop():
        try:
            stop_current(app)
        except Exception as exc:
            errors.append(exc)

    monkeypatch.setattr(app, "_transcribe_audio", transcribe)
    monkeypatch.setattr(main_module, "copy_to_clipboard", copied.append)
    thread = threading.Thread(target=stop)
    thread.start()
    try:
        assert entered.wait(2)
        app._on_recording_cancel(7)
        assert app.hotkey_manager.idle_calls == 0
        assert app._session_active
        assert overlay_calls[-1] == ("cancelled", 7, "")
        app._on_recording_cancel(7)
        assert app.recorder.stop_calls == 1
    finally:
        release.set()
        thread.join(2)
    assert not thread.is_alive()
    assert not errors
    assert copied == []
    assert app.logger.entries == []
    assert app.notifications.errors == []
    assert app.live_transcript_accumulator is None
    assert app.hotkey_manager.idle_calls == 1
    assert overlay_calls[-1] == ("cancelled_done", 7, "")


def test_exit_cancels_normal_stop_while_final_transcription_is_blocked(
    monkeypatch, overlay_calls
):
    app = make_stop_app(make_audio_data())
    entered, release = threading.Event(), threading.Event()
    copied, errors = [], []
    app.hotkey_manager.unregister = lambda: None
    app.tray.stop = lambda: None
    app.recorder.is_recording = lambda: False
    closed = threading.Event()
    close_overlay = app._close_overlay

    def close():
        close_overlay()
        closed.set()

    monkeypatch.setattr(app, "_close_overlay", close)

    def transcribe(audio_data):
        entered.set()
        assert release.wait(2)
        return FinalTranscript("ignored after exit")

    def normal_stop():
        try:
            stop_current(app)
        except Exception as exc:
            errors.append(exc)

    def exit_app():
        try:
            app.stop()
        except Exception as exc:
            errors.append(exc)

    monkeypatch.setattr(app, "_transcribe_audio", transcribe)
    monkeypatch.setattr(main_module, "copy_to_clipboard", copied.append)
    stop_thread = threading.Thread(target=normal_stop)
    stop_thread.start()
    exit_thread = None
    try:
        assert entered.wait(2)
        exit_thread = threading.Thread(target=exit_app)
        exit_thread.start()
        assert app._shutdown_event.wait(2)
        assert closed.wait(2)
        assert stop_thread.is_alive()
        assert overlay_calls == [(main_module.PROCESSING, 7, ""), "close"]
        assert app._cancel_event.is_set()
        assert app.recorder.stop_calls == 1
    finally:
        release.set()
        stop_thread.join(2)
        if exit_thread is not None:
            exit_thread.join(2)

    assert not stop_thread.is_alive()
    assert exit_thread is None or not exit_thread.is_alive()
    assert errors == []
    assert copied == []
    assert app.logger.entries == []
    assert app.hotkey_manager.idle_calls == 0
    assert app._session_active is False
    assert overlay_calls == [(main_module.PROCESSING, 7, ""), "close"]


@pytest.mark.parametrize("play_result", [False, "raises"])
def test_exit_capture_discards_queued_workers_and_restores_media_once(
    monkeypatch, overlay_calls, play_result
):
    app = make_stop_app(make_audio_data())
    events = []
    queued_segments = [object()]
    app._paused_media_source = "media.app"
    app.hotkey_manager.unregister = lambda: None
    app.tray.stop = lambda: None
    app.recorder.stop_recording = lambda: (
        events.append("recorder.stop") or make_audio_data()
    )

    class QueuedWorker:
        def __init__(self, label):
            self.label = label

        def cancel(self):
            events.append(f"{self.label}.cancel")
            queued_segments.clear()

        def stop(self):
            assert not queued_segments
            events.append(f"{self.label}.stop")

        def is_degraded(self):
            return False

    app.live_segmenter = QueuedWorker("vad")
    app.live_transcription_worker = QueuedWorker("transcription")
    monkeypatch.setattr(
        app, "_stop_live_segmentation", MurmurApp._stop_live_segmentation.__get__(app)
    )
    monkeypatch.setattr(
        app, "_stop_live_transcription", MurmurApp._stop_live_transcription.__get__(app)
    )

    def play(source_id):
        events.append("media.play")
        if play_result == "raises":
            raise RuntimeError("media provider failed")
        return play_result

    app.media_controller.play = play
    monkeypatch.setattr(
        main_module, "copy_to_clipboard", lambda text: pytest.fail("copied")
    )

    app.stop()
    app.stop()

    assert events.count("media.play") == 1
    assert events.index("vad.cancel") < events.index("recorder.stop")
    assert events.index("transcription.cancel") < events.index("recorder.stop")
    assert events.index("recorder.stop") < events.index("media.play")
    assert events.index("media.play") < events.index("vad.stop")
    assert events.index("media.play") < events.index("transcription.stop")
    assert queued_segments == []
    assert app._paused_media_source is None
    assert app.transcriber.audio_calls == []
    assert app.transcriber.segment_calls == []
    assert app.logger.entries == []
    assert app.recorder.stop_calls == 0


def test_delivery_claim_prevents_cancel_before_copy_returns(monkeypatch, overlay_calls):
    app = make_stop_app(make_audio_data())
    entered, release = threading.Event(), threading.Event()
    copied = []
    errors = []

    def copy(text):
        entered.set()
        assert release.wait(2)
        copied.append(text)
        return True

    def stop():
        try:
            stop_current(app)
        except Exception as exc:
            errors.append(exc)

    monkeypatch.setattr(main_module, "copy_to_clipboard", copy)
    thread = threading.Thread(target=stop)
    thread.start()
    try:
        assert entered.wait(2)
        app._on_recording_cancel(7)
        assert not app._cancel_event.is_set()
        assert not app._can_cancel()
    finally:
        release.set()
        thread.join(2)
    assert not thread.is_alive()
    assert not errors
    assert copied == ["Fallback text."]
    assert len(app.logger.entries) == 1
    assert overlay_calls[-1] == ("success", 7, "")


def test_exit_after_delivery_claim_finishes_copy_and_log_without_late_ui(
    monkeypatch, overlay_calls
):
    app = make_stop_app(make_audio_data())
    entered, release = threading.Event(), threading.Event()
    copied, errors = [], []
    app.hotkey_manager.unregister = lambda: None
    app.tray.stop = lambda: None
    app.recorder.is_recording = lambda: False
    closed = threading.Event()
    close_overlay = app._close_overlay

    def close():
        close_overlay()
        closed.set()

    monkeypatch.setattr(app, "_close_overlay", close)

    def copy(text):
        entered.set()
        assert release.wait(2)
        copied.append(text)
        return True

    def stop():
        try:
            stop_current(app)
        except Exception as exc:
            errors.append(exc)

    def exit_app():
        try:
            app.stop()
        except Exception as exc:
            errors.append(exc)

    monkeypatch.setattr(main_module, "copy_to_clipboard", copy)
    stop_thread = threading.Thread(target=stop)
    stop_thread.start()
    exit_thread = None
    try:
        assert entered.wait(2)
        assert app._delivery_started
        exit_thread = threading.Thread(target=exit_app)
        exit_thread.start()
        assert app._shutdown_event.wait(2)
        assert closed.wait(2)
        assert stop_thread.is_alive()
        assert overlay_calls == [(main_module.PROCESSING, 7, ""), "close"]
        assert not app._cancel_event.is_set()
    finally:
        release.set()
        stop_thread.join(2)
        if exit_thread is not None:
            exit_thread.join(2)

    assert not stop_thread.is_alive()
    assert exit_thread is None or not exit_thread.is_alive()
    assert errors == []
    assert copied == ["Fallback text."]
    assert len(app.logger.entries) == 1
    assert overlay_calls == [(main_module.PROCESSING, 7, ""), "close"]


def test_stale_cancel_and_limit_do_not_claim_newer_session():
    app = make_stop_app(make_audio_data())
    app._on_recording_cancel(6)
    app._handle_recording_limit_reached(60, session=6)
    assert not app._cancel_event.is_set()
    assert not app._recording_stop_started
    assert app.recorder.stop_calls == 0
    assert app._can_cancel()


@pytest.mark.parametrize("exit_app", [False, True])
def test_cancel_or_exit_while_normal_stop_drains_live_worker(
    monkeypatch, overlay_calls, exit_app
):
    app = make_stop_app(make_audio_data())
    entered, release = threading.Event(), threading.Event()
    errors = []
    monkeypatch.setattr(
        app, "_stop_live_transcription", MurmurApp._stop_live_transcription.__get__(app)
    )

    def transcribe(segment):
        entered.set()
        assert release.wait(2)
        return "discarded live text"

    monkeypatch.setattr(app.transcriber, "transcribe_segment", transcribe)
    app._start_live_transcription()
    worker = app.live_transcription_worker
    worker.submit_segment(SimpleNamespace(segment_id=0))
    if exit_app:
        app.hotkey_manager.unregister = lambda: None
        app.tray.stop = lambda: None
        app.recorder.is_recording = lambda: False

    def stop():
        try:
            stop_current(app)
        except Exception as exc:
            errors.append(exc)

    thread = threading.Thread(target=stop)
    thread.start()
    exit_thread = None
    try:
        assert entered.wait(2)
        if exit_app:
            exit_thread = threading.Thread(target=app.stop)
            exit_thread.start()
            assert app._shutdown_event.wait(2)
            assert app._cancel_event.is_set()
        else:
            app._on_recording_cancel(7)
        assert app.hotkey_manager.idle_calls == 0
        if not exit_app:
            assert overlay_calls[-1] == ("cancelled", 7, "")
    finally:
        release.set()
        thread.join(2)
        if exit_thread is not None:
            exit_thread.join(2)
        worker.stop()
    assert not thread.is_alive()
    assert exit_thread is None or not exit_thread.is_alive()
    assert not errors
    assert app.live_transcription_worker is None
    assert app.live_transcript_accumulator is None
    assert app.transcriber.audio_calls == []
    assert app.transcriber.finalize_calls == []
    assert app.logger.entries == []
    assert app.hotkey_manager.idle_calls == (0 if exit_app else 1)


@pytest.mark.parametrize("exit_app", [False, True])
def test_cancel_or_exit_during_stream_open_closes_without_commit(
    monkeypatch, overlay_calls, exit_app
):
    app = make_stop_app(make_audio_data())
    entered, release = threading.Event(), threading.Event()
    joining, finish_join = threading.Event(), threading.Event()
    errors = []
    app.config.pause_media_while_recording = True
    app.media_controller.is_media_playing = lambda: True
    app.media_controller.pause = lambda: "media.app"
    app.notifications.notify_recording_started = lambda: None
    app.recorder.set_capture_error_callback = lambda callback: None
    app.hotkey_manager.unregister = lambda: None
    app.tray.stop = lambda: None
    monkeypatch.setattr(app, "_start_live_transcription", lambda: None)
    monkeypatch.setattr(app, "_start_live_segmentation", lambda: None)

    def wait_for_inference():
        # Playback must recover before an already-running call can delay teardown.
        assert app.media_controller.play_calls == 0
        joining.set()
        assert finish_join.wait(2)

    monkeypatch.setattr(app, "_stop_live_transcription", wait_for_inference)

    def start_stream():
        entered.set()
        assert release.wait(2)

    def start():
        try:
            app._on_recording_start(RecordingRequest())
        except Exception as exc:
            errors.append(exc)

    def cancel():
        try:
            if exit_app:
                app.stop()
            else:
                app._on_recording_cancel(app._overlay_session)
        except Exception as exc:
            errors.append(exc)

    def show(state, message=""):
        overlay_calls.append((state, app._overlay_session, message))

    monkeypatch.setattr(app.recorder, "start_recording", start_stream, raising=False)
    monkeypatch.setattr(app, "_show_overlay", show)
    starter = threading.Thread(target=start)
    canceller = threading.Thread(target=cancel)
    starter.start()
    try:
        assert entered.wait(2)
        canceller.start()
        assert app._cancel_event.wait(2)
        assert app.hotkey_manager.idle_calls == 0
        assert app.recorder.stop_calls == 0
        release.set()
        assert joining.wait(2)
        assert app.media_controller.play_calls == 0
        assert app.recorder.stop_calls == 1
    finally:
        release.set()
        finish_join.set()
        starter.join(2)
        if canceller.ident is not None:
            canceller.join(2)
    assert not starter.is_alive()
    assert not canceller.is_alive()
    assert not errors
    assert app.transcriber.audio_calls == []
    assert app.hotkey_manager.idle_calls == (0 if exit_app else 1)
    assert main_module.CANCELLED not in [call[0] for call in overlay_calls]
    assert main_module.CANCELLED_DONE not in [call[0] for call in overlay_calls]


@pytest.mark.parametrize("committed", [False, True])
def test_retired_hotkey_stop_does_not_cancel_or_finalize_next_attempt(
    monkeypatch, committed
):
    app = make_stop_app(make_audio_data())
    old, new = RecordingRequest(), RecordingRequest()
    if committed:
        assert old.commit_capture()
    old.request_stop()
    app._current_request = old
    app._cancel_event = old.cancelled
    app.hotkey_manager.request = old

    def replace_attempt():
        old.invalidate()
        app._current_request = new
        app._cancel_event = new.cancelled
        app.hotkey_manager.request = new
        app._session_active = True
        app._recording_stop_started = False

    if committed:
        claim = app._claim_recording_stop

        def delayed_claim(**kwargs):
            replace_attempt()
            return claim(**kwargs)

        monkeypatch.setattr(app, "_claim_recording_stop", delayed_claim)
    else:
        cancel = app._on_recording_cancel

        def delayed_cancel(**kwargs):
            replace_attempt()
            return cancel(**kwargs)

        monkeypatch.setattr(app, "_on_recording_cancel", delayed_cancel)

    app._on_recording_stop(old, committed)

    assert app._current_request is new
    assert not new.cancelled.is_set()
    assert app.recorder.stop_calls == 0
    assert app.hotkey_manager.idle_calls == 0
    assert app.media_controller.play_calls == 0


@pytest.mark.parametrize("exit_app", [False, True, "hotkey"])
@pytest.mark.parametrize("cancel_during_pause", [False, True])
def test_media_wait_cancel_or_exit_closes_capture_and_restores_late_pause(
    monkeypatch, overlay_calls, exit_app, cancel_during_pause
):
    app = make_stop_app(None)
    app.config.pause_media_while_recording = True
    request = RecordingRequest()
    entered, release, closed = (threading.Event(), threading.Event(), threading.Event())
    errors = []
    pauses, plays, opens = [], [], []

    def detect_playing():
        if not cancel_during_pause:
            entered.set()
            assert release.wait(2)
        return True

    def pause():
        if cancel_during_pause:
            entered.set()
            assert release.wait(2)
        pauses.append(True)
        return "media.app"

    app.media_controller.is_media_playing = detect_playing
    app.media_controller.pause = pause
    app.media_controller.play = lambda source_id: plays.append(True) or True
    app.recorder.start_recording = lambda: opens.append(True)
    original_stop = app.recorder.stop_recording

    def close_capture():
        result = original_stop()
        closed.set()
        return result

    app.recorder.stop_recording = close_capture
    app.hotkey_manager.request = request
    app.recorder.set_capture_error_callback = lambda callback: None
    app.recorder.set_recording_limit_callback = lambda callback: None
    app.hotkey_manager.stop_requests = lambda: (request.request_stop(), request)[1]
    app.hotkey_manager.unregister = lambda: None
    app.tray.stop = lambda: None
    monkeypatch.setattr(app, "_start_live_transcription", lambda: None)
    monkeypatch.setattr(app, "_start_live_segmentation", lambda: None)
    monkeypatch.setattr(app, "_stop_live_transcription", lambda: None)
    monkeypatch.setattr(app, "_stop_live_segmentation", lambda: None)

    def start():
        try:
            app._on_recording_start(request)
        except Exception as exc:
            errors.append(exc)

    starter = threading.Thread(target=start)

    def stop_session():
        if exit_app == "hotkey":
            assert request.request_stop()
            app._on_recording_stop(request, True)
        elif exit_app:
            app.stop()
        else:
            app._on_recording_cancel(app._overlay_session)

    action = threading.Thread(target=stop_session)
    starter.start()
    try:
        entered_startup = entered.wait(2)
        if not entered_startup:
            release.set()
            starter.join(2)
            assert errors == []
            pytest.fail("startup did not enter media detection")
        action.start()
        if exit_app != "hotkey":
            assert request.cancelled.wait(2)
        assert closed.wait(2)
        assert action.is_alive()
    finally:
        release.set()
        starter.join(2)
        if action.ident is not None:
            action.join(2)

    assert not starter.is_alive()
    assert not action.is_alive()
    assert errors == []
    assert opens == [True]
    assert pauses == ([True] if cancel_during_pause else [])
    assert plays == ([True] if cancel_during_pause else [])
    assert main_module.RECORDING in [
        call[0] for call in overlay_calls if isinstance(call, tuple)
    ]
    if not exit_app:
        assert main_module.CANCELLED in [
            call[0] for call in overlay_calls if isinstance(call, tuple)
        ]
    assert app.hotkey_manager.idle_calls == (0 if exit_app is True else 1)


def test_capture_and_cue_precede_workers_and_blocked_media(overlay_calls):
    app = make_stop_app(None)
    app.config.pause_media_while_recording = True
    request = RecordingRequest()
    app.hotkey_manager.request = request
    entered, release = threading.Event(), threading.Event()
    errors, events = [], []

    def detect_playing():
        events.append("media")
        entered.set()
        assert release.wait(2)
        return False

    def start():
        try:
            app._on_recording_start(request)
        except Exception as exc:
            errors.append(exc)

    app.media_controller.is_media_playing = detect_playing
    app.recorder.set_capture_error_callback = lambda callback: None

    def open_capture():
        assert not request.capture_committed
        events.append("capture")

    def cue():
        assert request.capture_committed
        events.append("cue")

    app.recorder.start_recording = open_capture
    app.notifications.notify_recording_started = cue
    app._start_live_transcription = lambda: events.append("transcription")
    app._start_live_segmentation = lambda: events.append("vad")
    starter = threading.Thread(target=start)
    starter.start()
    try:
        assert entered.wait(2)
        starter.join(2)
        assert not starter.is_alive()
        assert events == ["capture", "cue", "transcription", "vad", "media"]
        assert app.tray.statuses[-1] == "Recording..."
        assert request.request_stop()
    finally:
        release.set()
        starter.join(2)

    assert not starter.is_alive()
    assert not errors
    assert app._session_active
    app._on_recording_stop(request, True)

    assert app.tray.statuses[-1] == "Ready"
    assert app.hotkey_manager.idle_calls == 1
    assert app.recorder.stop_calls == 1
    assert main_module.RECORDING in [
        call[0] for call in overlay_calls if isinstance(call, tuple)
    ]


@pytest.mark.parametrize("failed_worker", ["transcription", "vad"])
def test_worker_setup_failure_keeps_capture_for_offline_processing(
    monkeypatch, failed_worker
):
    audio = make_audio_data()
    app = make_stop_app(audio)
    app.config.pause_media_while_recording = False
    app.recorder.set_capture_error_callback = lambda callback: None
    app.recorder.start_recording = lambda: None
    app._start_live_transcription = lambda: None
    app._start_live_segmentation = lambda: None

    def fail_worker():
        raise RuntimeError("worker unavailable")

    name = (
        "_start_live_transcription"
        if failed_worker == "transcription"
        else "_start_live_segmentation"
    )
    monkeypatch.setattr(app, name, fail_worker)
    copies = []
    monkeypatch.setattr(
        main_module, "copy_to_clipboard", lambda text: copies.append(text) or True
    )
    request = RecordingRequest()
    app._on_recording_start(request)

    assert request.capture_committed
    assert app._session_active
    assert app.recorder.stop_calls == 0
    assert app._live_pipeline_degraded
    assert app.notifications.errors == []
    stop_current(app)

    assert app.recorder.stop_calls == 1
    assert app.transcriber.audio_calls == [audio]
    assert copies == ["Fallback text."]


def test_failed_start_does_not_release_session_claimed_by_cancellation(monkeypatch):
    app = make_stop_app(None)
    app.config.pause_media_while_recording = False
    app.notifications.notify_recording_started = lambda: None
    app.recorder.set_capture_error_callback = lambda callback: None
    armed = threading.Event()
    release_boundary = threading.Event()
    cancellation_claimed = threading.Event()
    release_cleanup = threading.Event()
    errors = []
    lock = threading.RLock()

    class ReleaseBoundaryLock:
        """Let cancellation claim the failed session before release takes its lock."""

        def __enter__(self):
            if threading.current_thread() is starter and armed.is_set():
                armed.clear()
                release_boundary.set()
                assert cancellation_claimed.wait(2)
            lock.acquire()
            return self

        def __exit__(self, *args):
            lock.release()

    def fail_start():
        armed.set()
        raise RuntimeError("microphone unavailable")

    finish_stop = app._finish_recording_stop

    def delayed_cleanup():
        cancellation_claimed.set()
        assert release_cleanup.wait(2)
        finish_stop()

    def start():
        try:
            app._on_recording_start(RecordingRequest())
        except Exception as exc:
            errors.append(exc)

    def cancel():
        try:
            assert release_boundary.wait(2)
            app._on_recording_cancel(app._overlay_session)
        except Exception as exc:
            errors.append(exc)

    monkeypatch.setattr(app, "_recording_stop_lock", ReleaseBoundaryLock())
    monkeypatch.setattr(app, "_start_live_transcription", lambda: None)
    monkeypatch.setattr(app, "_start_live_segmentation", lambda: None)
    monkeypatch.setattr(app.recorder, "start_recording", fail_start, raising=False)
    monkeypatch.setattr(app, "_finish_recording_stop", delayed_cleanup)
    starter = threading.Thread(target=start)
    canceller = threading.Thread(target=cancel)
    starter.start()
    canceller.start()
    try:
        assert cancellation_claimed.wait(2)
        starter.join(2)
        assert not starter.is_alive()
        assert app._cancel_event.is_set()
        assert app._session_active
        assert app.hotkey_manager.idle_calls == 0
    finally:
        release_cleanup.set()
        starter.join(2)
        canceller.join(2)
    assert not canceller.is_alive()
    assert not errors
    assert app.hotkey_manager.idle_calls == 1
    assert not app._session_active
    assert app.recorder.stop_calls == 1


def test_exit_closes_microphone_before_waiting_for_session_teardown(monkeypatch):
    app = make_stop_app(None)
    app.hotkey_manager.unregister = lambda: None
    app.tray.stop = lambda: None
    waiting, release = threading.Event(), threading.Event()

    def wait_for_teardown():
        assert app.recorder.close_calls == 1
        waiting.set()
        assert release.wait(2)

    monkeypatch.setattr(app, "_finish_recording_stop", wait_for_teardown)
    shutdown = threading.Thread(target=app.stop)
    shutdown.start()
    try:
        assert waiting.wait(2)
        assert shutdown.is_alive()
        assert app.recorder.close_calls == 1
    finally:
        release.set()
        shutdown.join(2)
    assert not shutdown.is_alive()


def test_stop_app_closes_overlay(overlay_calls):
    app = make_stop_app(None)
    app._running = True
    app.hotkey_manager.unregister = lambda: None
    app.tray.stop = lambda: None
    app.recorder.is_recording = lambda: False

    app.stop()

    assert overlay_calls == ["close"]
    assert app.recorder.close_calls == 1


@pytest.fixture
def usage_service(tmp_path):
    from src.usage_stats import UsageStatsService

    service = UsageStatsService(
        tmp_path / "statistics.sqlite3", enabled=True, provenance={}
    )
    service.summary().result(timeout=3)
    yield service
    assert service.close()


@pytest.mark.parametrize("copied", [False, True])
def test_statistics_clipboard_outcome_has_counts_without_text(
    usage_service, monkeypatch, copied
):
    app = make_app(None, FakeTranscriber())
    app._stats_session = usage_service.start_session()
    app.logger = FakeLogger(enabled=False)
    app.notifications = FakeNotifications()
    monkeypatch.setattr(main_module, "copy_to_clipboard", lambda _: copied)
    app._complete_transcription(
        make_audio_data(),
        FinalTranscript("synthetic private marker"),
        finalization_started_at=main_module.time.perf_counter(),
        live_segment_metrics=main_module.LiveSegmentMetrics.empty(),
    )
    rows = usage_service.export_rows().result(timeout=3)
    assert len(rows) == 1
    row = rows[0]
    assert row["outcome"] == ("success" if copied else "failed")
    assert row["final_word_count"] == 3
    assert (row["stop_to_clipboard_seconds"] is not None) is copied
    assert "synthetic private marker" not in str(rows)
    assert row["stage_clipboard_outcome"] == ("success" if copied else "failed")


def test_training_row_shares_statistics_session_id(usage_service, monkeypatch):
    app = make_app(None, FakeTranscriber())
    app._stats_session = usage_service.start_session()
    app.logger = FakeLogger()
    app.notifications = FakeNotifications()
    monkeypatch.setattr(main_module, "copy_to_clipboard", lambda _: True)
    app._complete_transcription(
        make_audio_data(),
        FinalTranscript("Hello."),
        finalization_started_at=main_module.time.perf_counter(),
        live_segment_metrics=main_module.LiveSegmentMetrics.empty(),
    )
    [row] = usage_service.export_rows().result(timeout=3)
    [cleanup] = app.logger.cleanup_details
    assert cleanup["session_id"] == row["session_id"]


@pytest.mark.parametrize("segmentation_error", [False, True])
def test_statistics_separates_normal_routing_from_error_fallback(
    usage_service, monkeypatch, segmentation_error
):
    segmenter = FakeSegmenter(
        error=RuntimeError("private raw exception") if segmentation_error else None
    )
    app = make_app(segmenter, FakeTranscriber())
    app._stats_session = usage_service.start_session()
    app.logger = FakeLogger(enabled=False)
    app.notifications = FakeNotifications()
    app.hotkey_manager = FakeHotkeyManager()
    monkeypatch.setattr(main_module, "copy_to_clipboard", lambda _: True)
    app._process_audio(
        make_audio_data(), finalization_started_at=main_module.time.perf_counter()
    )
    summary = usage_service.summary().result(timeout=3)
    assert summary["counts"]["success"] == 1
    assert summary["fallback_denominator"] == 1
    assert summary["fallback_count"] == int(segmentation_error)
    rows = usage_service.export_rows().result(timeout=3)
    assert "private raw exception" not in str(rows)
    assert rows[0]["transcription_route"] == "full_clip"


def test_statistics_files_unexpected_finalize_error_under_finalization(
    usage_service, monkeypatch
):
    app = make_app(None, FakeTranscriber())
    app.tray = FakeTray()
    app.hotkey_manager = FakeHotkeyManager()
    app._stats_session = usage_service.start_session()

    def fail_finalize():
        raise RuntimeError("cleanup bug")

    monkeypatch.setattr(app, "_stop_and_finalize_session", fail_finalize)
    with pytest.raises(RuntimeError):
        app._finish_recording_stop()
    row = usage_service.export_rows().result(timeout=3)[0]
    assert row["outcome"] == "failed"
    issues = json.loads(row["issues"])
    assert [(i["stage"], i["code"]) for i in issues] == [
        ("finalization", "unknown_error")
    ]


@pytest.mark.parametrize("capture_error", [False, True])
def test_statistics_capture_stage_fails_after_mid_recording_mic_error(
    usage_service, monkeypatch, capture_error
):
    app = make_app(None, FakeTranscriber())
    app.recorder = FakeRecorder(make_audio_data())
    app.tray = FakeTray()
    app._stats_session = usage_service.start_session()
    app._stats_capture_error = capture_error
    for name in (
        "_show_overlay",
        "_resume_owned_media",
        "_stop_live_segmentation",
        "_stop_live_transcription",
        "_finalize_recording",
    ):
        monkeypatch.setattr(app, name, lambda *a, **k: None)
    app._stop_and_finalize_session()
    app._stats_session.finish("success")
    row = usage_service.export_rows().result(timeout=3)[0]
    assert row["stage_capture_outcome"] == ("failed" if capture_error else "success")


@pytest.mark.parametrize(
    ("observed", "expected"),
    [
        (None, {"whisper_failed": 1}),
        (("offline_whisper", "whisper_failed"), {"whisper_failed": 1}),
        (("local_cleanup", "local_cleanup_failed"), {"local_cleanup_failed": 1}),
    ],
)
def test_statistics_counts_one_transcription_failure_once(
    usage_service, monkeypatch, observed, expected
):
    from src.observations import emit_observation

    app = make_app(None, FakeTranscriber())
    app._stats_session = usage_service.start_session()
    app.notifications = FakeNotifications()
    app.hotkey_manager = FakeHotkeyManager()

    def fail_transcription(audio_data):
        if observed:
            stage, code = observed
            emit_observation(
                stage=None if stage == "offline_whisper" else stage,
                outcome="failed",
                seconds=0.1,
                code=code,
                category="error",
            )
        raise RuntimeError("model unavailable")

    monkeypatch.setattr(app, "_transcribe_audio", fail_transcription)
    app._process_audio(make_audio_data(), finalization_started_at=10.0)
    summary = usage_service.summary().result(timeout=3)
    assert summary["counts"]["failed"] == 1
    assert summary["issues"] == expected


def test_statistics_disabled_cleanup_is_off_without_an_issue(usage_service):
    app = make_app(None, FakeTranscriber())
    collector = usage_service.start_session()
    app._stats_session = collector
    app._stats_observer()(
        {
            "stage": "llm_cleanup",
            "outcome": "off",
            "seconds": 0.0,
            "code": "cleanup_disabled",
            "category": "skip",
            "metadata": {"cleanup_model": None, "cleanup_enabled": False},
        }
    )
    collector.finish("success")
    summary = usage_service.summary().result(timeout=3)
    stage = summary["stages"]["ollama_cleanup"]
    assert (stage["off_count"], stage["skipped_count"]) == (1, 0)
    assert summary["issues"] == {}


def test_statistics_observer_classifies_recovered_cleanup_and_uses_fixed_collector(
    usage_service,
):
    app = make_app(None, FakeTranscriber())
    first = usage_service.start_session()
    app._stats_session = first
    observer = app._stats_observer()
    observer(
        {
            "stage": "llm_cleanup",
            "outcome": "failed",
            "seconds": 0.3,
            "code": "cleanup_timeout",
            "category": "timeout",
            "metadata": {"cleanup_model": "actual-model", "cleanup_enabled": True},
        }
    )
    first.finish("success")
    app._stats_session = usage_service.start_session()
    observer(
        {
            "stage": "whisper_live",
            "outcome": "success",
            "seconds": 9.0,
            "code": "whisper_completed",
            "category": "success",
            "metadata": {"whisper_model": "old-model", "device": "cpu"},
        }
    )
    summary = usage_service.summary().result(timeout=3)
    assert summary["recovered_failures"] == 1
    assert summary["stages"]["live_whisper"]["count"] == 0
    rows = usage_service.export_rows().result(timeout=3)
    completed = next(row for row in rows if row["outcome"] == "success")
    assert completed["cleanup_model"] == "actual-model"
    assert completed["stage_ollama_cleanup_failed_count"] == 1


def test_statistics_failure_does_not_prevent_clipboard_delivery(monkeypatch):
    class BrokenCollector:
        def stage(self, *args, **kwargs):
            raise OSError("synthetic private marker")

        issue = finish = stage

    app = make_app(None, FakeTranscriber())
    app._stats_session = BrokenCollector()
    app.logger = FakeLogger(enabled=False)
    app.notifications = FakeNotifications()
    deliveries = []
    monkeypatch.setattr(
        main_module, "copy_to_clipboard", lambda text: deliveries.append(text) or True
    )
    app._complete_transcription(
        make_audio_data(),
        FinalTranscript("delivered"),
        finalization_started_at=main_module.time.perf_counter(),
        live_segment_metrics=main_module.LiveSegmentMetrics.empty(),
    )
    assert deliveries == ["delivered"]
    assert app.notifications.copied_to_clipboard == 1
