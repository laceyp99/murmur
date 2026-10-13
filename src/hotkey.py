"""
Hotkey handling module for murmur.
Manages global hotkey registration and callbacks.
"""

import contextlib
import threading
from collections.abc import Callable
from dataclasses import dataclass
from enum import Enum

import keyboard

from .config import ConfigError, get_config


def is_hotkey_valid(hotkey: str) -> bool:
    """Return whether the keyboard library can parse the supplied hotkey."""
    if not isinstance(hotkey, str):
        return False

    normalized_hotkey = hotkey.strip()
    if not normalized_hotkey:
        return False

    try:
        keyboard.parse_hotkey_combinations(normalized_hotkey)
    except Exception:
        return False

    return True


class HotkeyState(Enum):
    """State of the hotkey system."""

    IDLE = "idle"
    RECORDING = "recording"
    PROCESSING = "processing"


@dataclass
class HotkeyEvent:
    """Event data for hotkey callbacks."""

    hotkey: str
    state: HotkeyState


class RecordingRequest:
    """One hotkey attempt whose cancel/commit boundary is atomic."""

    def __init__(self, identity: int):
        self.identity = identity
        self.cancelled = threading.Event()
        self._lock = threading.Lock()
        self._committed = False
        self._active = True

    def commit_capture(self) -> bool:
        """Commit capture unless an earlier stop or shutdown cancelled it."""
        with self._lock:
            if not self._active or self.cancelled.is_set():
                return False
            self._committed = True
            return True

    def request_stop(self) -> bool:
        """Record a stop and return whether capture had already committed."""
        with self._lock:
            committed = self._committed
            if not committed:
                self.cancelled.set()
            return committed

    def invalidate(self) -> None:
        """Make callbacks from a superseded attempt harmless."""
        with self._lock:
            self._active = False
            if not self._committed:
                self.cancelled.set()

    @property
    def capture_committed(self) -> bool:
        """Whether this attempt crossed the successful capture boundary."""
        with self._lock:
            return self._committed and self._active

    @property
    def is_active(self) -> bool:
        """Whether this request still owns hotkey activation callbacks."""
        with self._lock:
            return self._active


class _ChordGuard:
    """Remember which activation-key presses a single-chord hotkey consumed.

    The key hook calls `allow` for every activation-key event, and the hotkey
    callback calls `claim` when the chord matches.
    """

    def __init__(self) -> None:
        self._held: set[int] = set()
        self._fresh: int | None = None
        self._consumed: set[int] = set()

    def allow(self, event: keyboard.KeyboardEvent) -> bool:
        """Return False for repeats and release of a consumed key."""
        consumed = event.scan_code in self._consumed
        if event.event_type == keyboard.KEY_UP:
            self._held.discard(event.scan_code)
            self._fresh = None
            self._consumed.discard(event.scan_code)
        else:
            self._fresh = None if event.scan_code in self._held else event.scan_code
            self._held.add(event.scan_code)
        # Keep suppressing after modifiers are released, when add_hotkey no
        # longer matches the chord and would leak key-up to a focused player.
        return not consumed

    def claim(self) -> bool:
        """Consume the fresh press that completed the chord, if there is one."""
        # A key held before the chord matched already reached the app.
        # Let its repeats and release through instead of consuming it.
        if self._fresh is None:
            return False
        self._consumed.add(self._fresh)
        self._fresh = None
        return True


class HotkeyManager:
    """
    Manages global hotkey registration and state.

    Provides a toggle-based interface where pressing the hotkey
    starts recording, and pressing it again stops recording.
    """

    def __init__(self):
        self.config = get_config()
        self._state: HotkeyState = HotkeyState.IDLE
        self._on_start: Callable[[RecordingRequest], None] | None = None
        self._on_stop: Callable[[RecordingRequest, bool], None] | None = None
        self._on_state_change: Callable[[HotkeyState], None] | None = None
        self._hotkey_registered: bool = False
        self._last_registration_error: str | None = None
        self._removers: list[Callable[[], None]] = []
        self._guard: _ChordGuard | None = None
        self._lock = threading.Lock()
        self._request: RecordingRequest | None = None
        self._request_id = 0
        self._stopped = False

    def register(
        self,
        on_start: Callable[[RecordingRequest], None] | None = None,
        on_stop: Callable[[RecordingRequest, bool], None] | None = None,
        on_state_change: Callable[[HotkeyState], None] | None = None,
    ) -> bool:
        """
        Register the global hotkey.

        Args:
            on_start: Callback when recording should start.
            on_stop: Callback when recording should stop.
            on_state_change: Callback when state changes.

        Returns:
            True if registration successful, False otherwise.
        """
        if self._hotkey_registered:
            self.unregister()

        self._on_start = on_start
        self._on_stop = on_stop
        self._on_state_change = on_state_change

        try:
            steps = keyboard.parse_hotkey_combinations(self.config.hotkey)
            # Sequence hotkeys need their key-up events to advance library state.
            activation_keys = (
                {
                    scan_code
                    for combination in steps[0]
                    for scan_code in combination
                    if not keyboard.is_modifier(scan_code)
                }
                if len(steps) == 1
                else set()
            )
            # Modifier-only hotkeys have no activation key to guard.
            if activation_keys:
                guard = _ChordGuard()
                self._guard = guard
                for scan_code in sorted(activation_keys):
                    # keyboard tracks hooks by callback, so each key needs its own.
                    self._removers.append(
                        keyboard.hook_key(
                            scan_code,
                            lambda event: guard.allow(event),
                            suppress=True,
                        )
                    )
            self._removers.append(
                keyboard.add_hotkey(
                    self.config.hotkey, self._on_hotkey_pressed, suppress=True
                )
            )
            self._hotkey_registered = True
            self._last_registration_error = None
            print(f"Hotkey '{self.config.hotkey}' registered successfully.")
            return True
        except Exception as e:
            self.unregister()
            self._last_registration_error = str(e)
            print(f"Failed to register hotkey: {e}")
            return False

    def unregister(self) -> None:
        """Unregister the global hotkey."""
        for remove in reversed(self._removers):
            with contextlib.suppress(Exception):
                remove()
        self._removers.clear()
        self._guard = None
        self._hotkey_registered = False
        self._last_registration_error = None

    def get_last_registration_error(self) -> str | None:
        """Return the last registration failure message, if any."""
        return self._last_registration_error

    def _on_hotkey_pressed(self) -> bool:
        """Handle hotkey press event."""
        with self._lock:
            if self._stopped:
                return True
            guard = self._guard
            if guard is not None and not guard.claim():
                return True
            if self._state == HotkeyState.IDLE:
                if self._request is not None:
                    self._request.invalidate()
                self._request_id += 1
                request = RecordingRequest(self._request_id)
                self._request = request
                self._set_state(HotkeyState.RECORDING)
                if self._on_start:
                    # Run callback in separate thread to avoid blocking
                    threading.Thread(target=self._on_start, args=(request,)).start()

            elif self._state == HotkeyState.RECORDING:
                self._set_state(HotkeyState.PROCESSING)
                request = self._request
                if request is not None and self._on_stop:
                    committed = request.request_stop()
                    threading.Thread(
                        target=self._on_stop, args=(request, committed)
                    ).start()
            return False

    def _set_state(self, state: HotkeyState) -> None:
        """Set the current state and notify listeners."""
        self._state = state
        if self._on_state_change:
            self._on_state_change(state)

    def set_idle(self) -> None:
        """Set state back to idle (called after processing complete)."""
        with self._lock:
            if self._request is not None:
                self._request.invalidate()
                self._request = None
            self._set_state(HotkeyState.IDLE)

    def stop_requests(self) -> RecordingRequest | None:
        """Reject future activations and cancel any preparation in progress."""
        with self._lock:
            self._stopped = True
            request = self._request
            if request is not None:
                request.request_stop()
            return request

    def is_current_request(self, request: RecordingRequest) -> bool:
        """Reject a callback delayed past a later hotkey attempt."""
        with self._lock:
            return self._request is request and not self._stopped

    def set_processing(self) -> None:
        """Set state to processing."""
        with self._lock:
            self._set_state(HotkeyState.PROCESSING)

    def get_state(self) -> HotkeyState:
        """Get the current state."""
        return self._state

    def is_registered(self) -> bool:
        """Check if the hotkey is registered."""
        return self._hotkey_registered

    def update_hotkey(self, new_hotkey: str) -> bool:
        """
        Update the registered hotkey.

        Args:
            new_hotkey: The new hotkey combination to register.

        Returns:
            True if successful, False otherwise.
        """
        # Store callbacks
        on_start = self._on_start
        on_stop = self._on_stop
        on_state_change = self._on_state_change
        normalized_hotkey = new_hotkey.strip()

        if not is_hotkey_valid(normalized_hotkey):
            return False

        # Unregister old hotkey
        self.unregister()

        # Update config
        try:
            self.config.set("hotkey", normalized_hotkey)
        except ConfigError as exc:
            print(f"Failed to update hotkey: {exc}")
            return False

        # Re-register with new hotkey
        return self.register(on_start, on_stop, on_state_change)


def wait_for_exit() -> None:
    """
    Block until the user presses Escape or Ctrl+C.

    This is useful for keeping the main thread alive while
    hotkeys are being processed in the background.
    """
    print("\nmurmur is running. Press Escape or Ctrl+C to exit.\n")
    with contextlib.suppress(KeyboardInterrupt):
        keyboard.wait("escape")
