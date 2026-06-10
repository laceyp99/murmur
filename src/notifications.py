"""
Notification module for Murmur.
Provides user feedback through Windows toast notifications.
"""

import threading
from typing import Any, Optional

try:
    from win10toast import ToastNotifier

    TOAST_AVAILABLE = True
except ImportError:
    TOAST_AVAILABLE = False

from .config import get_config


class NotificationManager:
    """
    Manages Windows toast notifications for user feedback.

    Falls back to console output if toast notifications are not available.
    """

    def __init__(self):
        self.config = get_config()
        self._toaster: Optional[Any] = None
        self._fallback_reason = "toast unavailable"

        if TOAST_AVAILABLE:
            try:
                self._toaster = ToastNotifier()
                self._fallback_reason = ""
            except Exception:
                self._fallback_reason = "toast initialization failed"

    def notify(
        self, title: str, message: str, duration: int = 3, threaded: bool = True
    ) -> None:
        """
        Show a notification to the user.

        Args:
            title: Notification title.
            message: Notification message.
            duration: Duration in seconds.
            threaded: Whether to show notification in a separate thread.
        """
        if not self.config.enable_notifications:
            return

        if self._toaster is not None:
            try:
                if threaded:
                    thread = threading.Thread(
                        target=self._show_toast, args=(title, message, duration)
                    )
                    thread.start()
                else:
                    self._show_toast(title, message, duration)
            except Exception:
                self._print_fallback(title, "toast delivery failed")
        else:
            self._print_fallback(title, self._fallback_reason)

    def _print_fallback(self, title: str, reason: str) -> None:
        """Emit content-safe console fallback status."""
        if reason == "toast unavailable":
            print("Notification unavailable; message suppressed.")
            return

        print(
            "Notification fallback: "
            f"{reason}; title={title!r}; message suppressed."
        )

    def _show_toast(self, title: str, message: str, duration: int) -> None:
        """Show a toast notification."""
        toaster = self._toaster
        if toaster is None:
            self._print_fallback(title, self._fallback_reason)
            return

        try:
            toaster.show_toast(title, message, duration=duration, threaded=False)
        except Exception:
            self._print_fallback(title, "toast delivery failed")

    def notify_recording_started(self) -> None:
        """Notify that recording has started."""
        self.notify("murmur", "Recording started... Speak now!")

    def notify_recording_limit_reached(self, duration_seconds: float) -> None:
        """Notify that the maximum recording duration stopped capture."""
        self.notify(
            "murmur",
            f"Recording stopped at the {duration_seconds:g}s limit. Finalizing...",
            duration=5,
        )

    def notify_transcription_complete(self, text: str) -> None:
        """Notify that transcription is complete."""
        self.notify_transcription_copied()

    def notify_transcription_copied(self) -> None:
        """Notify that transcription was copied to the clipboard."""
        self.notify("murmur", "Transcription copied to clipboard.")

    def notify_clipboard_failure_retry(self) -> None:
        """Notify that clipboard copy failed and the user should retry recording."""
        self.notify_error("Clipboard copy failed. Please retry the recording.")

    def notify_clipboard_failure_retry_with_training_data(self) -> None:
        """Notify that clipboard copy failed after training data was saved."""
        self.notify_error(
            "Clipboard copy failed. Please retry the recording. "
            "The transcript was saved to training data."
        )

    def notify_error(self, error: str) -> None:
        """Notify about an error."""
        self.notify("murmur Error", error)


# Global notification manager instance
_notification_manager: Optional[NotificationManager] = None


def get_notification_manager() -> NotificationManager:
    """Get the global notification manager instance."""
    global _notification_manager
    if _notification_manager is None:
        _notification_manager = NotificationManager()
    return _notification_manager


def notify(title: str, message: str) -> None:
    """Convenience function to show a notification."""
    get_notification_manager().notify(title, message)
