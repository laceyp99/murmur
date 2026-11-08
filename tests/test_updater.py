import ast
import inspect
from types import SimpleNamespace

import pytest

from src import updater

REPO_URL = "https://github.com/laceyp99/murmur"


class FakeSource:
    def __init__(self, *args, **kwargs):
        self.args = args
        self.kwargs = kwargs


class FakeManager:
    """Record UpdateManager use without network; the fixture resets its state."""

    def __init__(self, *args, **kwargs):
        self.args = args
        self.kwargs = kwargs
        self.calls = []
        FakeManager.instances.append(self)

    def check_for_updates(self):
        self.calls.append("check")
        return FakeManager.release

    def download_updates(self, info):
        self.calls.append("download")
        if FakeManager.download_error is not None:
            raise FakeManager.download_error

    def wait_exit_then_apply_updates(self, info, **kwargs):
        self.calls.append(("wait_exit_then_apply_updates", info, kwargs))


class InlineThread:
    def __init__(self, target, **kwargs):
        self.target = target

    def start(self):
        self.target()


@pytest.fixture
def velopack_install(monkeypatch):
    FakeManager.instances = []
    FakeManager.release = None
    FakeManager.download_error = None
    monkeypatch.setattr(updater.velopack, "UpdateManager", FakeManager)
    monkeypatch.setattr(updater.velopack, "GithubSource", FakeSource)
    monkeypatch.setattr(updater.threading, "Thread", InlineThread)
    monkeypatch.setattr(updater, "_state", None)
    monkeypatch.setattr(updater, "_pending", None)
    monkeypatch.setattr(updater, "_ready_listener", None)
    return FakeManager


def test_updater_imports_nothing_that_could_reach_user_data():
    tree = ast.parse(inspect.getsource(updater))
    imported = set()
    for node in ast.walk(tree):
        if isinstance(node, ast.Import):
            imported.update(alias.name for alias in node.names)
        elif isinstance(node, ast.ImportFrom):
            assert node.level == 0, "updater must not import other murmur modules"
            imported.add(node.module)

    assert imported == {"enum", "threading", "collections.abc", "velopack"}


def test_updater_never_adds_tokens_or_headers():
    names = {
        node.attr
        for node in ast.walk(ast.parse(inspect.getsource(updater)))
        if isinstance(node, ast.Attribute)
    }

    assert not names & {"HttpSource", "HttpOptions", "HttpHeader", "UpdateOptions"}


def test_update_check_sends_only_the_public_repo_url(velopack_install):
    velopack_install.release = SimpleNamespace()

    assert updater.start_check()

    for manager in velopack_install.instances:
        (source,) = manager.args
        assert manager.kwargs == {}
        assert isinstance(source, FakeSource)
        assert source.args == (REPO_URL,)
        assert source.kwargs == {}


def test_found_release_downloads_and_notifies_listener(velopack_install):
    release = SimpleNamespace()
    velopack_install.release = release
    notified = []
    updater.set_ready_listener(lambda: notified.append(True))

    assert updater.start_check()

    assert updater.get_state() is updater.UpdateState.READY
    assert updater.pending_update() is release
    assert notified == [True]
    assert not updater.start_check()


def test_no_release_reports_up_to_date(velopack_install):
    assert updater.start_check()

    assert updater.get_state() is updater.UpdateState.UP_TO_DATE
    assert updater.pending_update() is None
    assert velopack_install.instances[-1].calls == ["check"]


def test_network_failure_is_treated_as_no_update(velopack_install, capsys):
    velopack_install.release = SimpleNamespace()
    velopack_install.download_error = OSError("offline")

    assert updater.start_check()

    assert updater.get_state() is updater.UpdateState.FAILED
    assert updater.pending_update() is None
    assert "offline" not in capsys.readouterr().out
    assert updater.start_check()


def test_source_checkout_cannot_check_for_updates(velopack_install, monkeypatch):
    def not_installed(*args, **kwargs):
        raise RuntimeError("not installed")

    monkeypatch.setattr(updater.velopack, "UpdateManager", not_installed)

    assert updater.get_state() is updater.UpdateState.UNAVAILABLE
    assert not updater.start_check()


def test_schedule_update_waits_for_normal_shutdown(velopack_install):
    release = SimpleNamespace()

    updater.schedule_update(release)

    assert velopack_install.instances[-1].calls == [
        ("wait_exit_then_apply_updates", release, {"silent": True, "restart": True})
    ]
