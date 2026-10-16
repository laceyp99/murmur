"""
System tray management for murmur.
"""

import sys
import threading

from PIL import Image

from .assets import get_logo_path
from .config import get_config


class TrayManager:
    """
    Manages the system tray icon and menu for murmur.
    """

    def __init__(
        self,
        on_exit_callback=None,
        on_cancel_callback=None,
        get_cancel_session_callback=None,
    ):
        self.config = get_config()
        self.on_exit_callback = on_exit_callback
        self.on_cancel_callback = on_cancel_callback
        self.get_cancel_session_callback = get_cancel_session_callback
        self.icon = None
        self._status_text = "Ready"
        self._settings_lock = threading.Lock()
        self._stop_requested = threading.Event()

    def _create_image(self):
        """Load and prepare the tray icon image."""
        logo_path = get_logo_path()
        if logo_path:
            try:
                image = Image.open(logo_path)
                # Resize for tray (usually 16x16 or 32x32, but Pillow handles scaling)
                return image.resize((64, 64), Image.Resampling.LANCZOS)
            except Exception as e:
                print(f"Error loading tray icon: {e}")

        # Fallback: Create a simple colored square if no image found
        return Image.new("RGB", (64, 64), color=(73, 109, 137))

    def set_status(self, status: str):
        """Update the status text in the tray menu."""
        self._status_text = status
        if self.icon:
            self.icon.update_menu()

    def _on_settings(self):
        """Open the settings window."""
        with self._settings_lock:
            from .settings_gui import show_settings

            show_settings()

    def _on_exit(self, icon, item):
        """Handle exit from tray menu."""
        if self.on_exit_callback:
            self.on_exit_callback()
        icon.stop()

    def _can_cancel(self, item=None):
        try:
            return bool(
                self.on_cancel_callback
                and self.get_cancel_session_callback
                and self.get_cancel_session_callback() is not None
            )
        except Exception as exc:
            print(
                f"Tray cancellation status unavailable ({type(exc).__name__}).",
                file=sys.stderr,
            )
            return False

    def _on_cancel(self, icon=None, item=None):
        try:
            if not self.on_cancel_callback or not self.get_cancel_session_callback:
                return
            session = self.get_cancel_session_callback()
            if session is not None:
                threading.Thread(
                    target=self._run_cancel, args=(session,), daemon=True
                ).start()
        except Exception as exc:
            print(f"Tray cancellation failed ({type(exc).__name__}).", file=sys.stderr)

    def _run_cancel(self, session):
        try:
            self.on_cancel_callback(session)
        except Exception as exc:
            print(f"Tray cancellation failed ({type(exc).__name__}).", file=sys.stderr)

    def run(self):
        """Start the system tray icon loop."""
        if self._stop_requested.is_set():
            return
        try:
            import pystray
            from pystray import MenuItem
        except Exception as exc:
            print(f"Tray unavailable: {exc}")
            self.icon = None
            return

        try:
            menu = pystray.Menu(
                MenuItem(
                    lambda text: f"Status: {self._status_text}",
                    lambda: None,
                    enabled=False,
                ),
                pystray.Menu.SEPARATOR,
                MenuItem("Cancel dictation", self._on_cancel, enabled=self._can_cancel),
                MenuItem("Settings", self._on_settings),
                MenuItem("Exit", self._on_exit),
            )

            self.icon = pystray.Icon("murmur", self._create_image(), "murmur", menu)
            self.icon.run(setup=self._on_ready)
        except Exception as exc:
            print(f"Tray unavailable: {exc}")
            self.icon = None

    def _on_ready(self, icon):
        """Show the icon and honor shutdown requested before backend readiness."""
        icon.visible = True
        if self._stop_requested.is_set():
            icon.stop()

    def stop(self):
        """Stop the tray icon."""
        self._stop_requested.set()
        if self.icon:
            self.icon.stop()
