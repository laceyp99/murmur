"""
Hotkey handling module for Murmur.
Manages global hotkey registration and callbacks.
"""

import keyboard
import threading
from typing import Optional, Callable
from dataclasses import dataclass
from enum import Enum

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
        self._on_start: Optional[Callable[[], None]] = None
        self._on_stop: Optional[Callable[[], None]] = None
        self._on_state_change: Optional[Callable[[HotkeyState], None]] = None
        self._hotkey_registered: bool = False
        self._lock = threading.Lock()

    def register(
        self,
        on_start: Optional[Callable[[], None]] = None,
        on_stop: Optional[Callable[[], None]] = None,
        on_state_change: Optional[Callable[[HotkeyState], None]] = None,
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
            keyboard.add_hotkey(
                self.config.hotkey, self._on_hotkey_pressed, suppress=True
            )
            self._hotkey_registered = True
            print(f"Hotkey '{self.config.hotkey}' registered successfully.")
            return True
        except Exception as e:
            print(f"Failed to register hotkey: {e}")
            return False

    def unregister(self) -> None:
        """Unregister the global hotkey."""
        if self._hotkey_registered:
            try:
                keyboard.remove_hotkey(self.config.hotkey)
            except Exception:
                pass
            self._hotkey_registered = False

    def _on_hotkey_pressed(self) -> None:
        """Handle hotkey press event."""
        with self._lock:
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
    try:
        keyboard.wait("escape")
    except KeyboardInterrupt:
        pass
