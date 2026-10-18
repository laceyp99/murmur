import threading

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
        self.active = False

    def start(self):
        if self.on_start is not None:
            self.on_start(self)
        if self.start_error is not None:
            raise self.start_error
        self.started = True
        self.active = True

    def stop(self):
        self.stopped = True
        self.active = False

    def close(self):
        self.closed = True
        self.active = False


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


def make_ready_recorder(monkeypatch):
    recorder, backend = make_fake_recorder(monkeypatch)
    recorder._keep_microphone_ready = True
    backend.on_start = lambda stream: stream.kwargs["callback"](
        np.ones((2, 1), dtype=np.float32), 2, None, None
    )
    return recorder, backend


def test_prepare_discards_idle_audio_and_reuses_stream_across_sessions(monkeypatch):
    recorder, backend = make_ready_recorder(monkeypatch)
    blocks = []
    recorder.set_block_callback(blocks.append)
    assert recorder.prepare()
    stream = recorder._stream
    callback = stream.kwargs["callback"]
    callback(np.ones((2, 1)), 2, None, None)
    assert recorder._audio_data == []
    assert blocks == []
    for _ in range(2):
        recorder.start_recording()
        callback(np.ones((2, 1)), 2, None, None)
        assert len(recorder.stop_recording().audio) == 2
        assert recorder._stream is stream
        assert not stream.closed
        assert recorder._audio_data == []
    assert len(backend.streams) == 1
    recorder.close()
    assert stream.closed
    assert not recorder.prepare()
    with pytest.raises(RuntimeError, match="closed"):
        recorder.start_recording()


def test_warm_start_crops_only_pre_activation_samples(monkeypatch):
    recorder, _ = make_ready_recorder(monkeypatch)
    assert recorder.prepare()
    recorder._stream.time = 10.2
    recorder.start_recording()
    info = type("TimeInfo", (), {"inputBufferAdcTime": 10.0})()
    recorder._stream.kwargs["callback"](np.arange(5).reshape(-1, 1), 5, info, None)
    np.testing.assert_array_equal(recorder.stop_recording().audio, [2, 3, 4])
    recorder.close()


def test_prepare_timeout_releases_stream_and_allows_cold_retry(monkeypatch):
    recorder, backend = make_ready_recorder(monkeypatch)
    backend.on_start = None
    recorder._prepare_timeout = 0.001
    assert not recorder.prepare()
    assert backend.streams[0].closed
    recorder.start_recording()
    assert recorder.is_recording()
    recorder.stop_recording()
    recorder.close()


def test_prepare_failure_cleans_up(monkeypatch):
    recorder, backend = make_ready_recorder(monkeypatch)
    backend.start_errors = [RuntimeError("unplugged")]
    assert not recorder.prepare()
    assert backend.streams[0].closed
    assert recorder._stream is None
    recorder.close()


@pytest.mark.parametrize("change", ["preference", "default", "refresh"])
def test_ready_stream_invalidated_before_next_session(monkeypatch, change):
    recorder, backend = make_ready_recorder(monkeypatch)
    assert recorder.prepare()
    old = recorder._stream
    backend.devices.append({"name": "Other", "hostapi": 0, "max_input_channels": 1})
    if change == "preference":
        recorder.config._values["microphone"] = {
            "name": "Other",
            "hostapi": "Windows WASAPI",
        }
    elif change == "default":
        backend.default.device = (1, -1)
    else:
        from src.audio import input_backend

        input_backend.request_refresh()
    recorder.start_recording()
    assert old.closed
    assert recorder._stream is not old
    assert recorder._stream.kwargs["device"] == (0 if change == "refresh" else 1)
    recorder.stop_recording()
    recorder.close()


def test_idle_device_failure_releases_without_prior_session_error(monkeypatch):
    recorder, _ = make_ready_recorder(monkeypatch)
    errors = []
    recorder.set_capture_error_callback(errors.append)
    assert recorder.prepare()
    stream = recorder._stream
    stream.active = False
    recorder._watch_stream(stream)
    assert stream.closed
    assert recorder._stream is None
    assert errors == []
    recorder.close()


def test_close_interrupts_prepare_wait(monkeypatch):
    recorder, backend = make_ready_recorder(monkeypatch)
    backend.on_start = None
    opened = threading.Event()
    backend.on_start = lambda stream: opened.set()
    result = []
    worker = threading.Thread(target=lambda: result.append(recorder.prepare()))
    worker.start()
    assert opened.wait(1)
    recorder.close()
    worker.join(1)
    assert not worker.is_alive()
    assert result == [False]
    assert backend.streams[0].closed


def test_idle_callback_does_not_copy_input(monkeypatch):
    class Uncopyable(np.ndarray):
        def copy(self, *args, **kwargs):
            raise AssertionError("Idle microphone audio must not be copied")

    recorder, backend = make_ready_recorder(monkeypatch)
    block = np.ones((2, 1)).view(Uncopyable)
    backend.on_start = lambda stream: stream.kwargs["callback"](block, 2, None, None)
    assert recorder.prepare()
    recorder._stream.kwargs["callback"](block, 2, None, None)
    assert recorder._audio_data == []
    recorder.close()


def test_idle_watchdog_releases_pending_refresh(monkeypatch, isolated_input_backend):
    recorder, _ = make_ready_recorder(monkeypatch)
    assert recorder.prepare()
    stream = recorder._stream
    assert not isolated_input_backend.request_refresh()
    recorder._watch_stream(stream)
    assert stream.closed
    assert isolated_input_backend.generation == 1
    assert not isolated_input_backend.refresh_pending
    recorder.close()


def test_warm_duration_limit_and_close_preserve_final_audio(monkeypatch):
    recorder, _ = make_ready_recorder(monkeypatch)
    recorder._max_recording_samples = 3
    assert recorder.prepare()
    recorder.start_recording()
    recorder._stream.kwargs["callback"](np.arange(5).reshape(-1, 1), 5, None, None)
    assert not recorder.is_recording()
    recorder.close()
    np.testing.assert_array_equal(recorder.stop_recording().audio, [0, 1, 2])
    assert recorder.stop_recording() is None


def test_ready_input_status_tracks_open_stream_through_stop_and_close(monkeypatch):
    recorder, _ = make_ready_recorder(monkeypatch)
    assert recorder.prepare()
    selection = input_status.get_status().active
    assert selection is not None
    recorder.start_recording()
    recorder._stream.kwargs["callback"](np.ones((2, 1)), 2, None, None)
    recorder.stop_recording()
    assert input_status.get_status().active == selection
    recorder.close()
    assert input_status.get_status().active is None
    assert input_status.get_status().error is None


def test_close_preserves_capture_failure_status(monkeypatch):
    recorder, _ = make_ready_recorder(monkeypatch)
    assert recorder.prepare()
    recorder.start_recording()
    recorder._report_capture_error("Microphone disconnected or stopped")
    recorder.close()
    assert input_status.get_status().error == "Microphone disconnected or stopped"


@pytest.mark.parametrize("failure", ["timeout", "inactive", "open", "unavailable"])
def test_prepare_failure_reports_fixed_on_demand_fallback(monkeypatch, capsys, failure):
    recorder, backend = make_ready_recorder(monkeypatch)
    if failure == "timeout":
        backend.on_start = None
        recorder._prepare_timeout = 0.001
    elif failure == "inactive":

        def become_inactive(timeout):
            recorder._stream.active = False
            return True

        recorder._first_callback.wait = become_inactive
    elif failure == "open":
        backend.start_errors = [RuntimeError("private device detail")]
    else:
        monkeypatch.setattr("src.audio.sd", None)
    assert not recorder.prepare()
    expected = "Microphone readiness failed; microphone will open on demand."
    assert input_status.get_status().error == expected
    assert capsys.readouterr().out.strip() == expected
    assert input_status.get_status().active is None
    recorder.close()


def test_disabled_and_shutdown_prepare_are_silent(monkeypatch, capsys):
    recorder, _ = make_fake_recorder(monkeypatch)
    assert not recorder.prepare()
    recorder.close()
    assert not recorder.prepare()
    assert capsys.readouterr().out == ""


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


def test_late_callback_replays_buffered_blocks_before_fresh_audio(monkeypatch):
    recorder, fake_sd = make_fake_recorder(monkeypatch)
    recorder.start_recording()
    callback = fake_sd.streams[0].kwargs["callback"]
    blocks = [np.array([[value]], dtype=np.float32) for value in (0.1, 0.2, 0.3)]
    for block in blocks[:2]:
        callback(block, frames=1, time_info=None, status=None)
    delivered = []
    recorder.set_block_callback(delivered.append, replay_buffered=True)
    callback(blocks[2], frames=1, time_info=None, status=None)
    assert len(delivered) == 3
    for actual, expected in zip(delivered, blocks, strict=True):
        np.testing.assert_array_equal(actual, expected)
    audio = recorder.stop_recording()
    np.testing.assert_allclose(audio.audio, [0.1, 0.2, 0.3])


def test_failed_buffered_replay_preserves_full_recording(monkeypatch):
    recorder, fake_sd = make_fake_recorder(monkeypatch)
    recorder.start_recording()
    callback = fake_sd.streams[0].kwargs["callback"]
    block = np.array([[0.1]], dtype=np.float32)
    callback(block, frames=1, time_info=None, status=None)
    errors, attempts = [], []
    recorder.set_block_callback_error_handler(errors.append)

    def fail(block):
        attempts.append(block)
        raise RuntimeError("live setup failed")

    recorder.set_block_callback(fail, replay_buffered=True)
    callback(block, frames=1, time_info=None, status=None)
    assert len(attempts) == 1
    assert len(errors) == 1
    np.testing.assert_allclose(recorder.stop_recording().audio, [0.1, 0.1])


def test_buffered_replay_holds_lock_until_early_blocks_are_delivered(monkeypatch):
    recorder, fake_sd = make_fake_recorder(monkeypatch)
    recorder.start_recording()
    callback = fake_sd.streams[0].kwargs["callback"]
    early = np.array([[0.1]], dtype=np.float32)
    fresh = np.array([[0.2]], dtype=np.float32)
    callback(early, frames=1, time_info=None, status=None)
    entered, release = threading.Event(), threading.Event()
    delivered = []

    def receive(block):
        delivered.append(block.copy())
        if len(delivered) == 1:
            entered.set()
            assert release.wait(2)

    attach = threading.Thread(
        target=lambda: recorder.set_block_callback(receive, replay_buffered=True)
    )
    attach.start()
    try:
        assert entered.wait(2)
        assert not recorder._lock.acquire(blocking=False)
    finally:
        release.set()
        attach.join(2)
    assert not attach.is_alive()
    callback(fresh, frames=1, time_info=None, status=None)
    np.testing.assert_allclose(np.concatenate(delivered).flatten(), [0.1, 0.2])
    recorder.stop_recording()


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


def test_status_query_failure_after_start_preserves_audio(monkeypatch):
    recorder, fake_sd = make_fake_recorder(monkeypatch)
    block = np.array([[0.1], [0.2]], dtype=np.float32)

    class BrokenStatusStream(FakeInputStream):
        @property
        def active(self):
            raise OSError("device status query failed")

        @active.setter
        def active(self, value):
            pass

    monkeypatch.setattr(
        fake_sd,
        "InputStream",
        lambda **kwargs: BrokenStatusStream(
            on_start=lambda stream: stream.kwargs["callback"](block, 2, None, None),
            **kwargs,
        ),
    )
    finalized = threading.Event()
    captured = []
    errors = []

    def capture_error(message):
        errors.append(message)
        captured.append(recorder.stop_recording())
        finalized.set()

    recorder.set_capture_error_callback(capture_error)
    recorder.start_recording()

    assert finalized.wait(2)
    assert errors == ["Microphone disconnected or stopped"]
    assert captured[0] is not None
    np.testing.assert_allclose(captured[0].audio, block.flatten())


def test_stream_close_waits_for_inflight_status_query(isolated_input_backend):
    query_started = threading.Event()
    query_released = threading.Event()
    stop_waiting = threading.Event()
    closed = threading.Event()
    watcher_locked = threading.Event()
    backend_lock = isolated_input_backend.lock

    class ObservedLock:
        def __enter__(self):
            if threading.current_thread().name == "test-stopper":
                stop_waiting.set()
            backend_lock.acquire()
            if threading.current_thread().name == "test-watcher":
                watcher_locked.set()
            return self

        def __exit__(self, *args):
            if threading.current_thread().name == "test-watcher":
                watcher_locked.clear()
            backend_lock.release()

    isolated_input_backend.lock = ObservedLock()

    class Stream:
        @property
        def active(self):
            assert watcher_locked.is_set()
            query_started.set()
            query_released.wait(2)
            return True

        def stop(self):
            pass

        def close(self):
            closed.set()

    recorder = AudioRecorder(config=FakeConfig())
    stream = Stream()
    recorder._recording = True
    recorder._stream = stream
    isolated_input_backend.stream_opened()
    watcher = threading.Thread(
        target=recorder._watch_stream, args=(stream,), name="test-watcher"
    )
    stopper = threading.Thread(target=recorder.stop_recording, name="test-stopper")
    watcher.start()
    try:
        assert query_started.wait(2)
        stopper.start()
        assert stop_waiting.wait(2)
        assert not closed.is_set()
    finally:
        query_released.set()
        watcher.join(2)
        if stopper.ident is not None:
            stopper.join(2)

    assert not watcher.is_alive()
    assert not stopper.is_alive()
    assert closed.is_set()


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
