from types import SimpleNamespace

import src.notifications as notifications_module


def make_manager(monkeypatch):
    monkeypatch.setattr(notifications_module, "TOAST_AVAILABLE", False)
    monkeypatch.setattr(
        notifications_module,
        "get_config",
        lambda: SimpleNamespace(enable_notifications=True),
    )
    return notifications_module.NotificationManager()


def test_notification_console_fallback_suppresses_message_body(monkeypatch, capsys):
    manager = make_manager(monkeypatch)

    manager.notify("murmur", "secret transcript content", threaded=False)

    stdout = capsys.readouterr().out
    assert "Notification unavailable; message suppressed." in stdout
    assert "secret transcript content" not in stdout
    assert "murmur" not in stdout


def test_transcription_complete_notification_has_no_transcript_preview(
    monkeypatch,
    capsys,
):
    manager = make_manager(monkeypatch)

    manager.notify_transcription_complete("private dictated text")

    stdout = capsys.readouterr().out
    assert "Notification unavailable; message suppressed." in stdout
    assert "private dictated text" not in stdout


def test_toast_failure_stdout_suppresses_transcript_preview(monkeypatch, capsys):
    class FailingToaster:
        def show_toast(self, title, message, duration, threaded):
            raise RuntimeError("private dictated text leaked in toast failure")

    monkeypatch.setattr(notifications_module, "TOAST_AVAILABLE", True)
    monkeypatch.setattr(
        notifications_module,
        "ToastNotifier",
        lambda: FailingToaster(),
        raising=False,
    )
    monkeypatch.setattr(
        notifications_module,
        "get_config",
        lambda: SimpleNamespace(enable_notifications=True),
    )
    manager = notifications_module.NotificationManager()

    manager.notify("murmur", "private dictated text", threaded=False)

    stdout = capsys.readouterr().out
    assert "private dictated text" not in stdout
    assert "leaked in toast failure" not in stdout
