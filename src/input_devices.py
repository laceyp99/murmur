"""Enumerate and select microphone inputs for recording clients."""

from __future__ import annotations

import contextlib
import threading
from collections.abc import Callable
from dataclasses import dataclass
from typing import Any

try:
    import sounddevice as sd
except (ImportError, OSError):
    sd = None


@dataclass(frozen=True)
class InputDevice:
    index: int
    name: str
    hostapi: str

    @property
    def label(self) -> str:
        return f"{self.name} ({self.hostapi})"

    @property
    def preference(self) -> dict[str, str]:
        return {"name": self.name, "hostapi": self.hostapi}


@dataclass(frozen=True)
class InputSelection:
    device: InputDevice
    reason: str


@dataclass(frozen=True)
class InputStatus:
    active: InputSelection | None = None
    error: str | None = None


class InputStatusStore:
    """Share the actual open input and last error with the settings window."""

    def __init__(self) -> None:
        self._lock = threading.Lock()
        self._status = InputStatus()

    def set_active(self, selection: InputSelection) -> None:
        with self._lock:
            self._status = InputStatus(active=selection)

    def set_error(self, message: str) -> None:
        with self._lock:
            self._status = InputStatus(error=message)

    def clear_active(self) -> None:
        with self._lock:
            self._status = InputStatus()

    def get_status(self) -> InputStatus:
        with self._lock:
            return self._status


input_status = InputStatusStore()


def _backend(backend: Any) -> Any:
    result = sd if backend is None else backend
    if result is None:
        raise RuntimeError("sounddevice/PortAudio is unavailable")
    return result


def list_input_devices(backend: Any = None) -> list[InputDevice]:
    """Return input devices with current-run indices and stable descriptions."""
    backend = _backend(backend)
    hostapis = backend.query_hostapis()
    return [
        InputDevice(index, info["name"], hostapis[info["hostapi"]]["name"])
        for index, info in enumerate(backend.query_devices())
        if info["max_input_channels"] > 0
    ]


def _default_input(backend: Any, devices: list[InputDevice]) -> InputDevice:
    try:
        index = backend.default.device[0]
    except (AttributeError, IndexError, TypeError):
        index = None
    if not isinstance(index, int):
        raise RuntimeError("No system default microphone is available")
    match = next((device for device in devices if device.index == index), None)
    if match is None:
        raise RuntimeError("No system default microphone is available")
    return match


def input_candidates(
    preference: dict[str, str] | None, backend: Any = None
) -> list[InputSelection]:
    """Choose a unique preferred input, followed by the current default."""
    backend = _backend(backend)
    devices = list_input_devices(backend)
    try:
        default = _default_input(backend, devices)
    except RuntimeError:
        default = None
    if not preference or not isinstance(preference, dict):
        if default is None:
            raise RuntimeError("No system default microphone is available")
        return [InputSelection(default, "system_default")]

    matches = [
        device
        for device in devices
        if device.name == preference.get("name")
        and device.hostapi == preference.get("hostapi")
    ]
    if len(matches) != 1:
        if default is None:
            raise RuntimeError("No system default microphone is available")
        reason = "preferred_missing" if not matches else "preferred_ambiguous"
        return [InputSelection(default, reason)]

    preferred = matches[0]
    if default is None or preferred.index == default.index:
        return [InputSelection(preferred, "preferred")]
    return [
        InputSelection(preferred, "preferred"),
        InputSelection(default, "preferred_open_failed"),
    ]


def resolve_input_device(
    preference: dict[str, str] | None, backend: Any = None
) -> InputSelection:
    """Preview the device that would be tried first, without opening it."""
    return input_candidates(preference, backend)[0]


def open_input_stream(
    preference: dict[str, str] | None,
    stream_factory: Callable[..., Any],
    *,
    backend: Any = None,
    **kwargs: Any,
) -> tuple[Any, InputSelection]:
    """Open and start an input stream, retrying the default on preferred failure."""
    candidates = input_candidates(preference, backend)
    for position, selection in enumerate(candidates):
        stream = None
        try:
            stream = stream_factory(device=selection.device.index, **kwargs)
            stream.start()
            return stream, selection
        except Exception:
            if stream is not None:
                with contextlib.suppress(Exception):
                    stream.close()
            if position == len(candidates) - 1:
                raise
    raise AssertionError("input_candidates returned no devices")
