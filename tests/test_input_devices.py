"""Microphone choice and fallback behavior."""

import pytest

from src.input_devices import (
    InputBackendState,
    InputStatusStore,
    input_candidates,
    list_input_devices,
    open_input_stream,
    resolve_input_device,
)


class FakeBackend:
    def __init__(self, default=0):
        self.default = type("Default", (), {"device": (default, -1)})()
        self.devices = [
            {"name": "Laptop", "hostapi": 0, "max_input_channels": 1},
            {"name": "Focusrite", "hostapi": 1, "max_input_channels": 2},
            {"name": "Speakers", "hostapi": 0, "max_input_channels": 0},
        ]

    def query_devices(self):
        return self.devices

    def query_hostapis(self):
        return [{"name": "WASAPI"}, {"name": "ASIO"}]


class FakeStream:
    def __init__(self, device, fail=False):
        self.device = device
        self.fail = fail
        self.closed = False

    def start(self):
        if self.fail:
            raise OSError("device busy")

    def close(self):
        self.closed = True


PREFERRED = {"name": "Focusrite", "hostapi": "ASIO"}


def test_backend_refresh_waits_for_stream_close_and_reinitializes_once():
    calls = []
    backend = type(
        "Backend",
        (),
        {
            "_terminate": lambda self: calls.append("terminate"),
            "_initialize": lambda self: calls.append("initialize"),
        },
    )()
    state = InputBackendState(backend)

    state.stream_opened()
    assert state.request_refresh() is False
    assert calls == []
    state.stream_closed()
    assert calls == ["terminate", "initialize"]
    assert state.generation == 1
    assert state.refresh_if_pending() is False


def test_backend_retries_initialize_without_terminating_twice():
    calls = []

    class Backend:
        def _terminate(self):
            calls.append("terminate")

        def _initialize(self):
            calls.append("initialize")
            if calls.count("initialize") == 1:
                raise OSError("device still reconnecting")

    state = InputBackendState(Backend())
    with pytest.raises(OSError, match="still reconnecting"):
        state.request_refresh()
    assert state.generation == 0
    assert state.refresh_if_pending() is True
    assert calls == ["terminate", "initialize", "initialize"]
    assert state.generation == 1


def test_list_and_resolve_unique_preferred():
    backend = FakeBackend()
    devices = list_input_devices(backend)
    assert [device.label for device in devices] == [
        "Laptop (WASAPI)",
        "Focusrite (ASIO)",
    ]
    assert devices[1].preference == PREFERRED
    assert resolve_input_device(PREFERRED, backend).device.index == 1
    assert resolve_input_device(None, backend).reason == "system_default"


def test_missing_and_ambiguous_preference_use_default():
    backend = FakeBackend()
    assert resolve_input_device(
        {"name": "Gone", "hostapi": "ASIO"}, backend
    ).reason == ("preferred_missing")
    backend.devices.append(backend.devices[1].copy())
    selection = resolve_input_device(PREFERRED, backend)
    assert selection.reason == "preferred_ambiguous"
    assert selection.device.index == 0


def test_failed_preferred_stream_is_closed_and_default_started():
    backend = FakeBackend()
    streams = []

    def factory(device, **kwargs):
        stream = FakeStream(device, fail=device == 1)
        streams.append(stream)
        return stream

    active, selection = open_input_stream(PREFERRED, factory, backend=backend)
    assert [stream.device for stream in streams] == [1, 0]
    assert streams[0].closed
    assert active is streams[1]
    assert selection.reason == "preferred_open_failed"


def test_default_failure_propagates_and_closes_stream():
    streams = []

    def factory(device):
        stream = FakeStream(device, fail=True)
        streams.append(stream)
        return stream

    with pytest.raises(OSError, match="device busy"):
        open_input_stream(None, factory, backend=FakeBackend())
    assert streams[0].closed


def test_no_default_still_allows_unique_preferred():
    backend = FakeBackend(default=-1)
    assert len(input_candidates(PREFERRED, backend)) == 1
    with pytest.raises(RuntimeError, match="No system default"):
        resolve_input_device(None, backend)


def test_invalid_default_reference_still_allows_preferred():
    backend = FakeBackend()
    backend.default.device = (None, -1)
    assert resolve_input_device(PREFERRED, backend).device.index == 1
    with pytest.raises(RuntimeError, match="No system default"):
        resolve_input_device(None, backend)


def test_invalid_saved_preference_uses_system_default():
    selection = resolve_input_device("invalid config value", FakeBackend())
    assert selection.device.index == 0
    assert selection.reason == "system_default"


def test_status_store_tracks_actual_input_and_error():
    status = InputStatusStore()
    selection = resolve_input_device(PREFERRED, FakeBackend())
    status.set_active(selection)
    assert status.get_status().active == selection
    status.set_error("Microphone disconnected")
    assert status.get_status().active is None
    assert status.get_status().error == "Microphone disconnected"
    status.clear_active()
    assert status.get_status().error is None
