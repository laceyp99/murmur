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


class HotkeyManager:
    """
    Manages global hotkey registration and state.

    Provides a toggle-based interface where pressing the hotkey
    starts recording, and pressing it again stops recording.
    """

    def __init__(self):
        self.config = get_config()
        self._state: HotkeyState = HotkeyState.IDLE
        self._on_start: Callable[[], None] | None = None
        self._on_stop: Callable[[], None] | None = None
        self._on_state_change: Callable[[HotkeyState], None] | None = None
        self._hotkey_registered: bool = False
        self._last_registration_error: str | None = None
        self._remove_hotkey: Callable[[], None] | None = None
        self._key_hooks: list[Callable[[], None]] = []
        self._activation_keys: set[int] = set()
        self._consumed_keys: set[int] = set()
        self._lock = threading.Lock()

    def register(
        self,
        on_start: Callable[[], None] | None = None,
        on_stop: Callable[[], None] | None = None,
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
            if len(steps) == 1:
                self._activation_keys = {
                    scan_code
                    for combination in steps[0]
                    for scan_code in combination
                    if not keyboard.is_modifier(scan_code)
                }
                for scan_code in sorted(self._activation_keys):
                    self._key_hooks.append(
                        keyboard.hook_key(
                            scan_code,
                            lambda event: self._filter_consumed_key(event),
                            suppress=True,
                        )
                    )
            self._remove_hotkey = keyboard.add_hotkey(
                self.config.hotkey, self._on_hotkey_pressed, suppress=True
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
        if self._remove_hotkey is not None:
            with contextlib.suppress(Exception):
                self._remove_hotkey()
            self._remove_hotkey = None
        for remove_hook in self._key_hooks:
            with contextlib.suppress(Exception):
                remove_hook()
        self._key_hooks.clear()
        self._activation_keys.clear()
        self._consumed_keys.clear()
        self._hotkey_registered = False
        self._last_registration_error = None

    def _filter_consumed_key(self, event: keyboard.KeyboardEvent) -> bool:
        """Consume repeats and release of a key used to activate dictation."""
        if event.scan_code not in self._consumed_keys:
            return True
        if event.event_type == keyboard.KEY_UP:
            self._consumed_keys.discard(event.scan_code)
        # Keep suppressing after modifiers are released, when add_hotkey no
        # longer matches the chord and would leak key-up to a focused player.
        return False

    def get_last_registration_error(self) -> str | None:
        """Return the last registration failure message, if any."""
        return self._last_registration_error

    def _on_hotkey_pressed(self) -> None:
        """Handle hotkey press event."""
        with self._lock:
            self._consumed_keys.update(
                scan_code
                for scan_code in self._activation_keys
                if keyboard.is_pressed(scan_code)
            )
            if self._state == HotkeyState.IDLE:
                self._set_state(HotkeyState.RECORDING)
                if self._on_start:
                    # Run callback in separate thread to avoid blocking
                    threading.Thread(target=self._on_start).start()

            elif self._state == HotkeyState.RECORDING:
                self._set_state(HotkeyState.PROCESSING)
                if self._on_stop:
                    threading.Thread(target=self._on_stop).start()

    def _set_state(self, state: HotkeyState) -> None:
        """Set the current state and notify listeners."""
        self._state = state
        if self._on_state_change:
            self._on_state_change(state)

    def set_idle(self) -> None:
        """Set state back to idle (called after processing complete)."""
        with self._lock:
            self._set_state(HotkeyState.IDLE)

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
