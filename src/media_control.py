"""
Windows media playback control using Global System Media Transport Controls (GSMTC).
Allows pausing/resuming system media (Spotify, YouTube, VLC, etc.) during recording.
"""

import asyncio
import threading
from importlib.util import find_spec


class MediaController:
    """
    Controller for Windows Global System Media Transport Controls.

    Provides methods to check playback status and pause/play system media.
    Gracefully handles cases where WinRT is unavailable (older Windows versions).
    """

    def __init__(self):
        self._available: bool | None = None

    def is_available(self) -> bool:
        """
        Check if media control is available on this system.

        Returns:
            True if WinRT media control APIs are available, False otherwise.
        """
        if self._available is not None:
            return self._available

        try:
            self._available = find_spec("winrt.windows.media.control") is not None
            if not self._available:
                print("[MediaControl] WinRT not available")
        except Exception as e:
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
        )
        from winrt.windows.media.control import (
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
            return status == PlaybackStatus.PLAYING

        except Exception as e:
            print(f"[MediaControl] Error checking playback status: {e}")
            return False

    async def _pause_async(self) -> str | None:
        """
        Async pause the current media session.

        Returns:
            The paused session's source app id, or None if pause failed.
        """
        from winrt.windows.media.control import (
            GlobalSystemMediaTransportControlsSessionManager as MediaManager,
        )

        try:
            manager = await MediaManager.request_async()
            if manager is None:
                print("[MediaControl] Manager is None")
                return None

            session = manager.get_current_session()
            if session is None:
                print("[MediaControl] No session to pause")
                return None

            source_id = session.source_app_user_model_id
            if not await session.try_pause_async():
                return None
            return source_id

        except Exception as e:
            print(f"[MediaControl] Error pausing: {e}")
            return None

    async def _play_async(self, source_id: str) -> bool:
        """
        Async resume the paused session from the given source app.

        Prefers the current session when several share the source id. Leaves
        other apps alone, and does nothing if that source is gone or no longer
        paused.

        Returns:
            False if the manager is unavailable or the matching resume failed.
        """
        from winrt.windows.media.control import (
            GlobalSystemMediaTransportControlsSessionManager as MediaManager,
        )
        from winrt.windows.media.control import (
            GlobalSystemMediaTransportControlsSessionPlaybackStatus as PlaybackStatus,
        )

        try:
            manager = await MediaManager.request_async()
            if manager is None:
                print("[MediaControl] Manager is None")
                return False

            current = manager.get_current_session()
            sessions = [current] if current is not None else []
            sessions.extend(manager.get_sessions())
            for session in sessions:
                if session.source_app_user_model_id != source_id:
                    continue
                playback_info = session.get_playback_info()
                if (
                    playback_info is None
                    or playback_info.playback_status != PlaybackStatus.PAUSED
                ):
                    continue
                result = await session.try_play_async()
                print(f"[MediaControl] Play result: {result}")
                return result

            print("[MediaControl] Paused session is no longer waiting to resume")
            return True

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

    def pause(self) -> str | None:
        """
        Pause the current media session.

        Returns:
            The paused session's source app id, or None if pause failed.
        """
        if not self.is_available():
            return None

        try:
            return self._run_async(self._pause_async)
        except Exception as e:
            print(f"[MediaControl] Exception in pause: {e}")
            return None

    def play(self, source_id: str) -> bool:
        """
        Resume the session that pause() returned, without starting other apps.

        Returns:
            False if the resume request failed, True otherwise.
        """
        if not self.is_available():
            return False

        try:
            result = self._run_async(lambda: self._play_async(source_id))
            return result if result is not None else False
        except Exception as e:
            print(f"[MediaControl] Exception in play: {e}")
            return False


# Global instance for convenience
_media_controller: MediaController | None = None


def get_media_controller() -> MediaController:
    """Get the global MediaController instance."""
    global _media_controller
    if _media_controller is None:
        _media_controller = MediaController()
    return _media_controller
