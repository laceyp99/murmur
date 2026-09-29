import numpy as np
import pytest

from src.audio import AudioRecorder
from src.input_devices import InputBackendState, input_status


class FakeConfig:
    def __init__(self, values=None):
        self._values = dict(values or {})

    def get(self, key, default=None):
        return self._values.get(key, default)


class FakeInputStream:
    def __init__(self, start_error=None, on_start=None, **kwargs):
        self.kwargs = kwargs
        self.start_error = start_error
        self.on_start = on_start
        self.started = False
        self.stopped = False
        self.closed = False

    def start(self):
        if self.on_start is not None:
            self.on_start(self)
        if self.start_error is not None:
            raise self.start_error
        self.started = True

    def stop(self):
        self.stopped = True

    def close(self):
        self.closed = True


class FakeSoundDevice:
    def __init__(self):
        self.streams = []
        self.constructor_errors = []
        self.start_errors = []
        self.on_start = None
        self.default = type("Default", (), {"device": (0, -1)})()
        self.devices = [
            {"name": "Default microphone", "hostapi": 0, "max_input_channels": 1}
        ]

    def query_hostapis(self):
        return [{"name": "Windows WASAPI"}]

    def query_devices(self):
        return self.devices

    def InputStream(self, **kwargs):  # noqa: N802 - mirror sounddevice's API
        if self.constructor_errors:
            raise self.constructor_errors.pop(0)
        start_error = self.start_errors.pop(0) if self.start_errors else None
        stream = FakeInputStream(
            start_error=start_error, on_start=self.on_start, **kwargs
        )
        self.streams.append(stream)
        return stream


@pytest.fixture(autouse=True)
def isolated_input_backend(monkeypatch):
    backend = FakeSoundDevice()
    backend._terminate = lambda: None
    backend._initialize = lambda: None
    state = InputBackendState(backend)
    monkeypatch.setattr("src.audio.input_backend", state)
    monkeypatch.setattr("src.input_devices.input_backend", state)
    return state


def make_fake_recorder(monkeypatch, *, sample_rate=10):
    fake_sd = FakeSoundDevice()
    monkeypatch.setattr("src.audio.sd", fake_sd)
    recorder = AudioRecorder(config=FakeConfig({"sample_rate": sample_rate}))
    return recorder, fake_sd


def start_fake_recorder(monkeypatch, *, sample_rate=10, max_recording_duration=0.5):
    fake_sd = FakeSoundDevice()
    monkeypatch.setattr("src.audio.sd", fake_sd)
    recorder = AudioRecorder(
        config=FakeConfig(
            {
                "sample_rate": sample_rate,
                "max_recording_duration": max_recording_duration,
            }
        )
    )
    recorder.start_recording()
    return recorder


def test_audio_callback_hands_block_to_registered_callback():
    recorder = AudioRecorder.__new__(AudioRecorder)
    recorder._lock = __import__("threading").RLock()
    recorder._recording = True
    recorder._audio_data = []
    recorder._block_callback = None
    recorder._on_block_callback_error = None
    recorder._block_callback_failed = False

    callback_blocks = []
    recorder.set_block_callback(callback_blocks.append)

    input_block = np.array([[0.1], [-0.2], [0.3]], dtype=np.float32)
    recorder._audio_callback(input_block, frames=3, time_info=None, status=None)

    assert len(recorder._audio_data) == 1
    np.testing.assert_array_equal(recorder._audio_data[0], input_block)
    assert len(callback_blocks) == 1
    np.testing.assert_array_equal(callback_blocks[0], input_block)


def test_audio_callback_logs_error_once_disables_callback_and_keeps_recording(capsys):
    recorder = AudioRecorder.__new__(AudioRecorder)
    recorder._lock = __import__("threading").RLock()
    recorder._recording = True
    recorder._audio_data = []
    recorder._block_callback = None
    recorder._on_block_callback_error = None
    recorder._block_callback_failed = False

    callback_errors = []
    callback_calls = []

    def failing_callback(block):
        callback_calls.append(block.copy())
        raise RuntimeError("vad callback broke")

    recorder.set_block_callback(failing_callback)
    recorder.set_block_callback_error_handler(callback_errors.append)

    first_block = np.array([[0.1], [-0.2], [0.3]], dtype=np.float32)
    second_block = np.array([[0.4], [0.5], [-0.6]], dtype=np.float32)

    recorder._audio_callback(first_block, frames=3, time_info=None, status=None)
    recorder._audio_callback(second_block, frames=3, time_info=None, status=None)

    assert len(recorder._audio_data) == 2
    np.testing.assert_array_equal(recorder._audio_data[0], first_block)
    np.testing.assert_array_equal(recorder._audio_data[1], second_block)
    assert len(callback_calls) == 1
    assert [str(error) for error in callback_errors] == ["vad callback broke"]
    assert recorder._block_callback is None
    assert recorder._block_callback_failed is True
    stdout = capsys.readouterr().out
    assert "Audio block callback failed; disabling live callback." in stdout
    assert "vad callback broke" not in stdout


def test_start_recording_requires_sounddevice(monkeypatch):
    class FakeConfig:
        def get(self, key, default=None):
            return default

    recorder = AudioRecorder(config=FakeConfig())
    monkeypatch.setattr("src.audio.sd", None)

    with pytest.raises(RuntimeError, match="PortAudio"):
        recorder.start_recording()


def test_start_recording_rolls_back_when_stream_construction_fails(monkeypatch):
    recorder, fake_sd = make_fake_recorder(monkeypatch)
    fake_sd.constructor_errors.append(OSError("no input device"))

    with pytest.raises(OSError, match="no input device"):
        recorder.start_recording()

    assert recorder.is_recording() is False
    assert recorder._stream is None
    assert recorder._recording_start is None
    assert recorder.stop_recording() is None


def test_start_recording_closes_stream_when_start_fails(monkeypatch):
    recorder, fake_sd = make_fake_recorder(monkeypatch)
    fake_sd.start_errors.append(OSError("device busy"))

    with pytest.raises(OSError, match="device busy"):
        recorder.start_recording()

    [stream] = fake_sd.streams
    assert stream.closed is True
    assert stream.started is False
    assert recorder.is_recording() is False
    assert recorder._stream is None
    assert recorder.stop_recording() is None


def test_start_recording_keeps_original_error_when_close_fails(monkeypatch):
    recorder, fake_sd = make_fake_recorder(monkeypatch)
    fake_sd.start_errors.append(OSError("device busy"))

    def failing_close():
        raise RuntimeError("close failed")

    fake_sd.on_start = lambda stream: setattr(stream, "close", failing_close)

    with pytest.raises(OSError, match="device busy"):
        recorder.start_recording()

    assert recorder.is_recording() is False
    assert recorder._stream is None


def test_start_recording_accepts_blocks_while_stream_starts(monkeypatch):
    recorder, fake_sd = make_fake_recorder(monkeypatch)
    early_block = np.array([[0.1], [0.2]], dtype=np.float32)
    fake_sd.on_start = lambda stream: stream.kwargs["callback"](
        early_block, frames=2, time_info=None, status=None
    )

    recorder.start_recording()
    audio_data = recorder.stop_recording()

    assert audio_data is not None
    np.testing.assert_allclose(audio_data.audio, early_block.flatten())


def test_start_recording_can_retry_after_failed_start(monkeypatch):
    recorder, fake_sd = make_fake_recorder(monkeypatch)
    fake_sd.start_errors.append(OSError("device busy"))

    with pytest.raises(OSError, match="device busy"):
        recorder.start_recording()

    recorder.start_recording()

    failed_stream, retry_stream = fake_sd.streams
    assert failed_stream.closed is True
    assert retry_stream.started is True
    assert recorder.is_recording() is True
    assert recorder._stream is retry_stream

    block = np.array([[0.3], [0.4]], dtype=np.float32)
    retry_stream.kwargs["callback"](block, frames=2, time_info=None, status=None)
    audio_data = recorder.stop_recording()

    assert audio_data is not None
    np.testing.assert_allclose(audio_data.audio, block.flatten())
    assert retry_stream.stopped is True
    assert retry_stream.closed is True


def test_preferred_open_failure_uses_default_and_keeps_preference(monkeypatch):
    fake_sd = FakeSoundDevice()
    fake_sd.devices.append({"name": "Focusrite", "hostapi": 0, "max_input_channels": 2})
    fake_sd.start_errors.append(OSError("Focusrite disconnected"))
    monkeypatch.setattr("src.audio.sd", fake_sd)
    preference = {"name": "Focusrite", "hostapi": "Windows WASAPI"}
    recorder = AudioRecorder(config=FakeConfig({"microphone": preference}))

    recorder.start_recording()

    assert [stream.kwargs["device"] for stream in fake_sd.streams] == [1, 0]
    assert fake_sd.streams[0].closed is True
    assert recorder.config.get("microphone") == preference
    recorder.stop_recording()


def test_broken_stream_still_returns_captured_audio(
    monkeypatch, isolated_input_backend
):
    recorder, fake_sd = make_fake_recorder(monkeypatch)
    recorder.start_recording()
    block = np.array([[0.1], [0.2]], dtype=np.float32)
    fake_sd.streams[0].kwargs["callback"](block, frames=2, time_info=None, status=None)

    def broken_stop():
        raise OSError("device removed")

    fake_sd.streams[0].stop = broken_stop
    errors = []
    recorder.set_capture_error_callback(errors.append)
    recorder._report_capture_error("Microphone disconnected or stopped")
    audio = recorder.stop_recording()

    assert errors == ["Microphone disconnected or stopped"]
    assert audio is not None
    np.testing.assert_allclose(audio.audio, block.flatten())
    assert isolated_input_backend.generation == 1


def test_failed_refresh_does_not_discard_recorded_audio(
    monkeypatch, isolated_input_backend
):
    recorder, fake_sd = make_fake_recorder(monkeypatch)
    recorder.start_recording()
    fake_sd.streams[0].kwargs["callback"](
        np.array([[0.4]], dtype=np.float32), 1, None, None
    )

    def fail_initialize():
        raise OSError("reconnect pending")

    isolated_input_backend.backend._initialize = fail_initialize
    isolated_input_backend.mark_stale()

    audio = recorder.stop_recording()

    assert audio is not None
    np.testing.assert_allclose(audio.audio, [0.4])
    assert "reconnect pending" in input_status.get_status().error
    input_status.clear_active()


def test_inactive_stream_reports_capture_failure(monkeypatch):
    recorder, _ = make_fake_recorder(monkeypatch)
    recorder.start_recording()
    errors = []
    recorder.set_capture_error_callback(errors.append)
    monkeypatch.setattr("src.audio.time.sleep", lambda _seconds: None)
    stream = recorder._stream
    stream.active = False

    recorder._watch_stream(stream)

    assert errors == ["Microphone disconnected or stopped"]
    recorder.stop_recording()


def test_old_stream_watcher_cannot_stop_new_recording(monkeypatch):
    recorder, _ = make_fake_recorder(monkeypatch)
    errors = []
    recorder.set_capture_error_callback(errors.append)
    recorder.start_recording()
    old_stream = recorder._stream
    recorder.stop_recording()
    recorder.start_recording()

    recorder._report_capture_error("old stream stopped", stream=old_stream)

    assert errors == []
    assert recorder.is_recording() is True
    recorder.stop_recording()


def test_failed_preferred_start_does_not_feed_early_blocks_to_live_callback(
    monkeypatch,
):
    fake_sd = FakeSoundDevice()
    fake_sd.devices.append({"name": "Focusrite", "hostapi": 0, "max_input_channels": 2})
    fake_sd.start_errors.append(OSError("Focusrite disconnected"))
    monkeypatch.setattr("src.audio.sd", fake_sd)
    recorder = AudioRecorder(
        config=FakeConfig(
            {"microphone": {"name": "Focusrite", "hostapi": "Windows WASAPI"}}
        )
    )
    live_blocks = []
    recorder.set_block_callback(live_blocks.append)

    def send_early_block(stream):
        value = 0.1 if stream.kwargs["device"] == 1 else 0.2
        stream.kwargs["callback"](np.array([[value]], dtype=np.float32), 1, None, None)

    fake_sd.on_start = send_early_block
    recorder.start_recording()
    audio = recorder.stop_recording()

    assert audio is not None
    np.testing.assert_allclose(audio.audio, [0.2])
    assert len(live_blocks) == 1
    np.testing.assert_allclose(live_blocks[0], [[0.2]])


def test_audio_callback_caps_audio_at_max_recording_duration(monkeypatch):
    recorder = start_fake_recorder(
        monkeypatch, sample_rate=10, max_recording_duration=0.5
    )
    callback_blocks = []
    recorder.set_block_callback(lambda block: callback_blocks.append(block.copy()))

    first_block = np.array([[0.1], [0.2], [0.3]], dtype=np.float32)
    second_block = np.array([[0.4], [0.5], [0.6]], dtype=np.float32)
    extra_block = np.array([[0.7], [0.8], [0.9]], dtype=np.float32)

    recorder._audio_callback(first_block, frames=3, time_info=None, status=None)
    recorder._audio_callback(second_block, frames=3, time_info=None, status=None)
    recorder._audio_callback(extra_block, frames=3, time_info=None, status=None)

    audio_data = recorder.stop_recording()

    assert audio_data is not None
    assert audio_data.sample_rate == 10
    assert audio_data.duration == pytest.approx(0.5)
    np.testing.assert_allclose(
        audio_data.audio,
        np.array([0.1, 0.2, 0.3, 0.4, 0.5], dtype=np.float32),
    )
    assert len(callback_blocks) == 2
    np.testing.assert_array_equal(callback_blocks[0], first_block)
    np.testing.assert_array_equal(callback_blocks[1], second_block[:2])


def test_audio_callback_notifies_recording_limit_once(monkeypatch):
    recorder = start_fake_recorder(
        monkeypatch, sample_rate=10, max_recording_duration=0.5
    )
    limit_events = []
    recorder.set_recording_limit_callback(limit_events.append)

    capped_block = np.array([[0.1], [0.2], [0.3], [0.4], [0.5]], dtype=np.float32)
    extra_block = np.array([[0.6], [0.7], [0.8]], dtype=np.float32)

    recorder._audio_callback(capped_block, frames=5, time_info=None, status=None)
    recorder._audio_callback(extra_block, frames=3, time_info=None, status=None)

    assert limit_events == [pytest.approx(0.5)]
    assert recorder.is_recording() is False


@pytest.mark.parametrize("invalid_duration", [None, 0, -1, "five"])
def test_recorder_falls_back_to_default_max_recording_duration(invalid_duration):
    recorder = AudioRecorder(
        config=FakeConfig(
            {
                "sample_rate": 16000,
                "max_recording_duration": invalid_duration,
            }
        )
    )

    assert recorder.max_recording_duration == 300
