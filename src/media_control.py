"""
Windows media playback control using Global System Media Transport Controls (GSMTC).
Allows pausing/resuming system media (Spotify, YouTube, VLC, etc.) during recording.
"""

import asyncio
import threading
from typing import Optional


class MediaController:
    """
    Controller for Windows Global System Media Transport Controls.

    Provides methods to check playback status and pause/play system media.
    Gracefully handles cases where WinRT is unavailable (older Windows versions).
    """

    def __init__(self):
        self._available: Optional[bool] = None

    def is_available(self) -> bool:
        """
        Check if media control is available on this system.

        Returns:
            True if WinRT media control APIs are available, False otherwise.
        """
        if self._available is not None:
            return self._available

        try:
            # Try importing all required modules
            from winrt.windows.media.control import (
                GlobalSystemMediaTransportControlsSessionManager,
                GlobalSystemMediaTransportControlsSessionPlaybackStatus,
            )

            self._available = True
        except (ImportError, ModuleNotFoundError) as e:
            print(f"[MediaControl] WinRT not available: {e}")
            self._available = False

        return self._available

    def _run_async(self, coro_func):
        """
        Run an async coroutine function synchronously in a dedicated thread.

        This avoids event loop conflicts by always creating a fresh event loop
        in a separate thread.

        Args:
            coro_func: A callable that returns a coroutine (not the coroutine itself)
        """
        result = [None]
        exception = [None]

        def run_in_thread():
            try:
                # Create a completely new event loop for this thread
                loop = asyncio.new_event_loop()
                asyncio.set_event_loop(loop)
                try:
                    result[0] = loop.run_until_complete(coro_func())
                finally:
                    loop.close()
            except Exception as e:
                exception[0] = e

        thread = threading.Thread(target=run_in_thread)
        thread.start()
        thread.join(timeout=5)

        if thread.is_alive():
            print("[MediaControl] Operation timed out")
            return None

        if exception[0] is not None:
            print(f"[MediaControl] Thread exception: {exception[0]}")
            return None

        return result[0]

    async def _is_media_playing_async(self) -> bool:
        """
        Async check if any media is currently playing.

        Returns:
            True if media is playing, False otherwise.
        """
        from winrt.windows.media.control import (
            GlobalSystemMediaTransportControlsSessionManager as MediaManager,
            GlobalSystemMediaTransportControlsSessionPlaybackStatus as PlaybackStatus,
        )

        try:
            # Get a fresh manager each time to avoid stale state
            manager = await MediaManager.request_async()
            if manager is None:
                print("[MediaControl] Manager is None")
                return False

            session = manager.get_current_session()
            if session is None:
                print("[MediaControl] No active media session")
                return False

            playback_info = session.get_playback_info()
            if playback_info is None:
                print("[MediaControl] No playback info available")
                return False

            status = playback_info.playback_status
            is_playing = status == PlaybackStatus.PLAYING
            # print(f"[MediaControl] Status: {status}, is_playing: {is_playing}")
            return is_playing

        except Exception as e:
            print(f"[MediaControl] Error checking playback status: {e}")
            return False

    async def _pause_async(self) -> bool:
        """
        Async pause the current media session.

        Returns:
            True if pause was successful, False otherwise.
        """
        from winrt.windows.media.control import (
            GlobalSystemMediaTransportControlsSessionManager as MediaManager,
        )

        try:
            manager = await MediaManager.request_async()
            if manager is None:
                print("[MediaControl] Manager is None")
                return False

            session = manager.get_current_session()
            if session is None:
                print("[MediaControl] No session to pause")
                return False

            result = await session.try_pause_async()
            # print(f"[MediaControl] Pause result: {result}")
            return result

        except Exception as e:
            print(f"[MediaControl] Error pausing: {e}")
            return False

    async def _play_async(self) -> bool:
        """
        Async play/resume the current media session.

        Returns:
            True if play was successful, False otherwise.
        """
        from winrt.windows.media.control import (
            GlobalSystemMediaTransportControlsSessionManager as MediaManager,
        )

        try:
            manager = await MediaManager.request_async()
            if manager is None:
                print("[MediaControl] Manager is None")
                return False

            session = manager.get_current_session()
            if session is None:
                print("[MediaControl] No session to play")
                return False

            result = await session.try_play_async()
            print(f"[MediaControl] Play result: {result}")
            return result

        except Exception as e:
            print(f"[MediaControl] Error playing: {e}")
            return False

    def is_media_playing(self) -> bool:
        """
        Check if any media is currently playing.

        Returns:
            True if media is playing, False otherwise or if unavailable.
        """
        if not self.is_available():
            return False

        try:
            result = self._run_async(self._is_media_playing_async)
            return result if result is not None else False
        except Exception as e:
            print(f"[MediaControl] Exception in is_media_playing: {e}")
            return False

    def pause(self) -> bool:
        """
        Pause the current media session.

        Returns:
            True if pause was successful, False otherwise.
        """
        if not self.is_available():
            return False

        try:
            result = self._run_async(self._pause_async)
            return result if result is not None else False
        except Exception as e:
            print(f"[MediaControl] Exception in pause: {e}")
            return False

    def play(self) -> bool:
        """
        Play/resume the current media session.

        Returns:
            True if play was successful, False otherwise.
        """
        if not self.is_available():
            return False

        try:
            result = self._run_async(self._play_async)
            return result if result is not None else False
        except Exception as e:
            print(f"[MediaControl] Exception in play: {e}")
            return False


# Global instance for convenience
_media_controller: Optional[MediaController] = None


def get_media_controller() -> MediaController:
    """Get the global MediaController instance."""
    global _media_controller
    if _media_controller is None:
        _media_controller = MediaController()
    return _media_controller
