from types import SimpleNamespace

import numpy as np
import pytest

from src.config import DEFAULT_CONFIG


main_module = pytest.importorskip("src.main")
AudioData = main_module.AudioData
MurmurApp = main_module.MurmurApp
VADSettings = main_module.VADSettings


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
        self.load_model_calls = 0
        self.warm_llm_post_processor_calls = 0

    def transcribe_segments(self, segments, debug=False):
        self.segment_calls.append({"segments": list(segments), "debug": debug})
        return SimpleNamespace(text="Segmented text.")

    def transcribe(self, audio_data):
        self.audio_calls.append(audio_data)
        return "Fallback text."

    def transcribe_segment(self, segment):
        self.live_segment_calls.append(segment)
        return f"live-{segment.segment_id}"

    def finalize_text(self, text):
        self.finalize_calls.append(text)
        return f"FINAL:{text}" if text else ""

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
    return app


class FakeRecorder:
    def __init__(self, audio_data):
        self.audio_data = audio_data
        self.stop_calls = 0
        self.sample_rate = 16000
        self.block_callback = None
        self.block_callback_error_handler = None
        self.recording_limit_callback = None

    def stop_recording(self):
        self.stop_calls += 1
        return self.audio_data

    def set_block_callback(self, callback):
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
        self.messages = []
        self.errors = []
        self.recording_limits = []

    def notify_transcription_complete(self, text):
        self.completed.append(text)

    def notify(self, title, message):
        self.messages.append((title, message))

    def notify_error(self, message):
        self.errors.append(message)

    def notify_recording_limit_reached(self, duration_seconds):
        self.recording_limits.append(duration_seconds)


class FakeLogger:
    def __init__(self):
        self.entries = []

    def log(self, audio_data, text, elapsed):
        self.entries.append((audio_data, text, elapsed))

    def is_enabled(self):
        return False

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

    def set_processing(self):
        self.processing_calls += 1

    def set_idle(self):
        self.idle_calls += 1


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

    def consume_startup_notice(self):
        notice = self._startup_notice
        self._startup_notice = None
        return notice


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

    def play(self):
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
    app._on_recording_start = lambda: None
    app._on_recording_stop = lambda: None
    app._on_state_change = lambda state: None
    return app


def test_transcribe_audio_uses_vad_segments_when_available():
    segment = SimpleNamespace(duration=0.35)
    segmenter = FakeSegmenter(segments=[segment])
    transcriber = FakeTranscriber()
    app = make_app(segmenter, transcriber)

    text = app._transcribe_audio(make_audio_data())

    assert text == "Segmented text."
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

    text = app._transcribe_audio(audio_data)

    assert text == "Fallback text."
    assert len(transcriber.segment_calls) == 0
    assert transcriber.audio_calls == [audio_data]


def test_transcribe_audio_falls_back_when_vad_raises():
    segmenter = FakeSegmenter(error=RuntimeError("vad unavailable"))
    transcriber = FakeTranscriber()
    app = make_app(segmenter, transcriber)
    audio_data = make_audio_data()

    text = app._transcribe_audio(audio_data)

    assert text == "Fallback text."
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

    assert first_text == "Fallback text."
    assert second_text == "Fallback text."
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

    monkeypatch.setattr(main_module, "get_config", lambda: fake_config)
    monkeypatch.setattr(main_module, "AudioRecorder", lambda: fake_recorder)
    monkeypatch.setattr(main_module, "Transcriber", lambda: fake_transcriber)
    monkeypatch.setattr(main_module, "HotkeyManager", lambda: SimpleNamespace())
    monkeypatch.setattr(
        main_module, "get_notification_manager", lambda: SimpleNamespace()
    )
    monkeypatch.setattr(main_module, "get_logger", lambda: SimpleNamespace())
    monkeypatch.setattr(
        main_module, "TrayManager", lambda on_exit_callback: SimpleNamespace()
    )
    monkeypatch.setattr(main_module, "get_media_controller", lambda: SimpleNamespace())

    app = main_module.MurmurApp(preload_model=True)

    assert fake_transcriber.load_model_calls == 1
    assert fake_transcriber.warm_llm_post_processor_calls == 1
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

    monkeypatch.setattr(main_module, "get_config", lambda: fake_config)
    monkeypatch.setattr(main_module, "AudioRecorder", lambda: SimpleNamespace())
    monkeypatch.setattr(main_module, "Transcriber", lambda: fake_transcriber)
    monkeypatch.setattr(main_module, "HotkeyManager", lambda: SimpleNamespace())
    monkeypatch.setattr(
        main_module, "get_notification_manager", lambda: SimpleNamespace()
    )
    monkeypatch.setattr(main_module, "get_logger", lambda: SimpleNamespace())
    monkeypatch.setattr(
        main_module, "TrayManager", lambda on_exit_callback: SimpleNamespace()
    )
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


def test_start_notifies_when_config_recovery_happened():
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
        (
            "murmur",
            "Config file was unreadable and has been reset to defaults. The original file was backed up to 'config.corrupt-20260601010101000000.json'.",
        ),
        ("murmur", "Ready! Press hotkey to start recording."),
    ]


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


def test_finalize_recording_uses_live_transcript_before_offline_fallback(monkeypatch):
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
    assert transcriber.finalize_calls == ["hello world again"]
    assert app.notifications.completed == ["FINAL:hello world again"]
    assert app.logger.entries == [(audio_data, "FINAL:hello world again", 1.25)]
    assert app.tray.statuses == ["Ready"]
    assert app.hotkey_manager.processing_calls == 1
    assert app.hotkey_manager.idle_calls == 1


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
        "private dictated text",
        finalization_started_at=1.0,
    )

    stdout = capsys.readouterr().out
    assert "Finalized in 1.2s; copied to clipboard." in stdout
    assert "private dictated text" not in stdout
    assert copied_text == ["private dictated text"]
    assert app.notifications.completed == ["private dictated text"]


def test_complete_transcription_stdout_omits_transcript_on_clipboard_failure(
    monkeypatch,
    capsys,
):
    transcriber = FakeTranscriber()
    app = make_app(segmenter=None, transcriber=transcriber)
    app.notifications = FakeNotifications()
    app.logger = FakeLogger()

    monkeypatch.setattr(main_module, "copy_to_clipboard", lambda text: False)

    app._complete_transcription(
        make_audio_data(),
        "private dictated text",
        finalization_started_at=1.0,
    )

    stdout = capsys.readouterr().out
    assert "Transcribed successfully, but failed to copy to clipboard." in stdout
    assert "private dictated text" not in stdout
    assert app.notifications.errors == ["Failed to copy to clipboard"]
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
    app._on_recording_stop = lambda: stop_calls.append(True)

    app._handle_recording_limit_reached(300)

    assert app.notifications.recording_limits == [300]
    assert app.hotkey_manager.processing_calls == 1
    assert stop_calls == [True]


def test_on_recording_stop_finalizes_live_pipeline_and_resumes_media(monkeypatch):
    transcriber = FakeTranscriber()
    audio_data = make_audio_data()
    app = make_app(segmenter=None, transcriber=transcriber)
    app.recorder = FakeRecorder(audio_data)
    app.tray = FakeTray()
    app.notifications = FakeNotifications()
    app.logger = FakeLogger()
    app.hotkey_manager = FakeHotkeyManager()
    app.media_controller = FakeMediaController()
    app._was_media_playing = True
    app.live_transcript_accumulator = main_module.TranscriptAccumulator()
    app.live_transcript_accumulator.add_chunk(
        SimpleNamespace(segment_id=0, text="tail kept", latency_seconds=0.1)
    )

    segmentation_stops = []
    transcription_stops = []
    copied_text = []
    monkeypatch.setattr(
        app, "_stop_live_segmentation", lambda: segmentation_stops.append(True)
    )
    monkeypatch.setattr(
        app, "_stop_live_transcription", lambda: transcription_stops.append(True)
    )
    monkeypatch.setattr(
        main_module, "copy_to_clipboard", lambda text: copied_text.append(text) or True
    )
    perf_counter_values = iter([20.0, 20.75])
    monkeypatch.setattr(
        main_module.time, "perf_counter", lambda: next(perf_counter_values)
    )

    app._on_recording_stop()

    assert app.tray.statuses[0] == "Finalizing..."
    assert segmentation_stops == [True]
    assert transcription_stops == [True]
    assert copied_text == ["FINAL:tail kept"]
    assert app.logger.entries == [(audio_data, "FINAL:tail kept", 0.75)]
    assert app.media_controller.play_calls == 1
    assert app.hotkey_manager.processing_calls == 1
    assert app.hotkey_manager.idle_calls == 1
