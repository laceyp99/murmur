from types import SimpleNamespace

import numpy as np
import pytest


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


class FakeConfig:
    vad_aggressiveness = 1
    vad_padding_ms = 500
    vad_silence_duration_ms = 400


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

    def stop_recording(self):
        self.stop_calls += 1
        return self.audio_data

    def set_block_callback(self, callback):
        self.block_callback = callback

    def set_block_callback_error_handler(self, handler):
        self.block_callback_error_handler = handler


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

    def notify_transcription_complete(self, text):
        self.completed.append(text)

    def notify(self, title, message):
        self.messages.append((title, message))

    def notify_error(self, message):
        self.errors.append(message)


class FakeLogger:
    def __init__(self):
        self.entries = []

    def log(self, audio_data, text, elapsed):
        self.entries.append((audio_data, text, elapsed))


class FakeHotkeyManager:
    def __init__(self):
        self.processing_calls = 0
        self.idle_calls = 0

    def set_processing(self):
        self.processing_calls += 1

    def set_idle(self):
        self.idle_calls += 1


class FakeMediaController:
    def __init__(self):
        self.play_calls = 0

    def play(self):
        self.play_calls += 1
        return True


def make_audio_data():
    audio = np.array([0.1, -0.3, 0.2], dtype=np.float32)
    return AudioData(audio=audio, sample_rate=16000, duration=len(audio) / 16000)


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

    assert [queued_segment.segment_id for queued_segment in transcriber.live_segment_calls] == [3]
    assert app.live_transcript_accumulator.get_text() == "live-3"


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


def test_finalize_recording_uses_live_transcript_before_offline_fallback(monkeypatch):
    transcriber = FakeTranscriber()
    app = make_app(segmenter=None, transcriber=transcriber)
    app.live_transcript_accumulator = main_module.TranscriptAccumulator()
    app.live_transcript_accumulator.add_chunk(
        SimpleNamespace(segment_id=0, text="hello world", latency_seconds=0.1)
    )
    app.tray = FakeTray()
    app.notifications = FakeNotifications()
    app.logger = FakeLogger()
    app.hotkey_manager = FakeHotkeyManager()

    copied_text = []
    monkeypatch.setattr(main_module, "copy_to_clipboard", lambda text: copied_text.append(text) or True)

    audio_data = make_audio_data()
    app._finalize_recording(audio_data)

    assert copied_text == ["FINAL:hello world"]
    assert transcriber.audio_calls == []
    assert transcriber.segment_calls == []
    assert transcriber.finalize_calls == ["hello world"]
    assert app.notifications.completed == ["FINAL:hello world"]
    assert app.logger.entries == [(audio_data, "FINAL:hello world", 0.0)]
    assert app.tray.statuses == ["Ready"]
    assert app.hotkey_manager.processing_calls == 1
    assert app.hotkey_manager.idle_calls == 1


def test_finalize_recording_falls_back_to_offline_processing_when_live_text_missing(monkeypatch):
    transcriber = FakeTranscriber()
    app = make_app(segmenter=None, transcriber=transcriber)
    called = []
    audio_data = make_audio_data()

    monkeypatch.setattr(app, "_process_audio", lambda provided_audio: called.append(provided_audio))

    app._finalize_recording(audio_data)

    assert called == [audio_data]


def test_finalize_recording_falls_back_to_offline_processing_when_live_pipeline_degraded(monkeypatch):
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
    monkeypatch.setattr(app, "_process_audio", lambda provided_audio: called.append(provided_audio))

    app._finalize_recording(audio_data)

    assert called == [audio_data]
    assert transcriber.finalize_calls == []


def test_on_live_pipeline_degraded_marks_app_once_and_notifies_once():
    transcriber = FakeTranscriber()
    app = make_app(segmenter=None, transcriber=transcriber)
    app.notifications = FakeNotifications()

    app._on_live_pipeline_degraded("worker failed")
    app._on_live_pipeline_degraded("later failure")

    assert app._live_pipeline_degraded is True
    assert app._live_pipeline_degraded_reason == "worker failed"
    assert app.notifications.messages == [
        (
            "murmur",
            "Live transcription paused. Final transcript will finish after recording.",
        )
    ]


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
    assert app._live_pipeline_degraded_reason == "Live audio callback failed: queue submit failed"
    assert app.notifications.messages == [
        (
            "murmur",
            "Live transcription paused. Final transcript will finish after recording.",
        )
    ]


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
    monkeypatch.setattr(app, "_stop_live_segmentation", lambda: segmentation_stops.append(True))
    monkeypatch.setattr(app, "_stop_live_transcription", lambda: transcription_stops.append(True))
    monkeypatch.setattr(main_module, "copy_to_clipboard", lambda text: copied_text.append(text) or True)

    app._on_recording_stop()

    assert app.tray.statuses[0] == "Finalizing..."
    assert segmentation_stops == [True]
    assert transcription_stops == [True]
    assert copied_text == ["FINAL:tail kept"]
    assert app.media_controller.play_calls == 1
    assert app.hotkey_manager.processing_calls == 1
    assert app.hotkey_manager.idle_calls == 1