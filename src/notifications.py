"""
Notification module for Murmur.
Provides user feedback through Windows toast notifications.
"""

import threading
from typing import Optional

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
        self._toaster: Optional[ToastNotifier] = None
        
        if TOAST_AVAILABLE:
            try:
                self._toaster = ToastNotifier()
            except:
                pass
    
    def notify(
        self,
        title: str,
        message: str,
        duration: int = 3,
        threaded: bool = True
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
                        target=self._show_toast,
                        args=(title, message, duration)
                    )
                    thread.start()
                else:
                    self._show_toast(title, message, duration)
            except Exception as e:
                print(f"[{title}] {message}")
        else:
            print(f"[{title}] {message}")
    
    def _show_toast(self, title: str, message: str, duration: int) -> None:
        """Show a toast notification."""
        try:
            self._toaster.show_toast(
                title,
                message,
                duration=duration,
                threaded=False
            )
        except:
            pass
    
    def notify_recording_started(self) -> None:
        """Notify that recording has started."""
        self.notify("Murmur", "Recording started... Speak now!")
    
    def notify_recording_stopped(self) -> None:
        """Notify that recording has stopped."""
        self.notify("Murmur", "Recording stopped. Processing...")
    
    def notify_transcription_complete(self, text: str) -> None:
        """Notify that transcription is complete."""
        preview = text[:50] + "..." if len(text) > 50 else text
        self.notify("Murmur", f"Copied: {preview}")
    
    def notify_error(self, error: str) -> None:
        """Notify about an error."""
        self.notify("Murmur Error", error)


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
