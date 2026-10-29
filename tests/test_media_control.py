import sys
from types import ModuleType, SimpleNamespace

import pytest

from src.media_control import MediaController

PLAYING = "playing"
PAUSED = "paused"


class FakeSession:
    def __init__(self, source_id, status=PLAYING):
        self.source_app_user_model_id = source_id
        self.status = status
        self.calls = []

    def get_playback_info(self):
        return SimpleNamespace(playback_status=self.status)

    async def try_pause_async(self):
        self.calls.append("pause")
        self.status = PAUSED
        return True

    async def try_play_async(self):
        self.calls.append("play")
        self.status = PLAYING
        return True


class FakeManager:
    """Stands in for the GSMTC session manager; tests switch `current` freely."""

    def __init__(self, *sessions):
        self.sessions = list(sessions)
        self.current = sessions[0] if sessions else None

    def get_current_session(self):
        return self.current

    def get_sessions(self):
        return list(self.sessions)


@pytest.fixture
def media(monkeypatch):
    """Install a fake WinRT media module and return a holder for its manager."""
    holder = SimpleNamespace(manager=None)

    async def request_async():
        return holder.manager

    module = ModuleType("winrt.windows.media.control")
    module.GlobalSystemMediaTransportControlsSessionManager = SimpleNamespace(
        request_async=request_async
    )
    module.GlobalSystemMediaTransportControlsSessionPlaybackStatus = SimpleNamespace(
        PLAYING=PLAYING, PAUSED=PAUSED
    )
    monkeypatch.setitem(sys.modules, "winrt.windows.media.control", module)
    controller = MediaController()
    controller._available = True
    holder.controller = controller
    return holder


def test_resume_targets_paused_app_after_current_session_switches(media):
    spotify, browser = FakeSession("Spotify"), FakeSession("Chrome")
    media.manager = FakeManager(spotify, browser)

    assert media.controller.pause() == "Spotify"
    media.manager.current = browser

    assert media.controller.play("Spotify") is True
    assert spotify.calls == ["pause", "play"]
    assert browser.calls == []


def test_resume_prefers_current_session_among_same_app_sessions(media):
    first_tab = FakeSession("Chrome", PAUSED)
    second_tab = FakeSession("Chrome", PAUSED)
    media.manager = FakeManager(first_tab, second_tab)
    media.manager.current = second_tab

    assert media.controller.play("Chrome") is True
    assert first_tab.calls == []
    assert second_tab.calls == ["play"]


def test_resume_skips_when_paused_app_closed(media):
    spotify, browser = FakeSession("Spotify"), FakeSession("Chrome", PAUSED)
    media.manager = FakeManager(spotify, browser)
    assert media.controller.pause() == "Spotify"
    media.manager.current = browser
    media.manager.sessions.remove(spotify)

    assert media.controller.play("Spotify") is True
    assert browser.calls == []


def test_resume_plays_app_whose_status_has_not_updated_yet(media):
    spotify, browser = FakeSession("Spotify"), FakeSession("Chrome", PAUSED)
    media.manager = FakeManager(spotify, browser)
    assert media.controller.pause() == "Spotify"
    # Stop right after pause: the app still reports PLAYING.
    spotify.status = PLAYING
    media.manager.current = browser

    assert media.controller.play("Spotify") is True
    assert spotify.calls == ["pause", "play"]
    assert browser.calls == []


def test_unavailable_manager_reports_failure(media):
    media.manager = None

    assert media.controller.pause() is None
    assert media.controller.play("Spotify") is False


def test_failed_pause_returns_no_source(media):
    session = FakeSession("Spotify")

    async def refuse_pause():
        return False

    session.try_pause_async = refuse_pause
    media.manager = FakeManager(session)

    assert media.controller.pause() is None
