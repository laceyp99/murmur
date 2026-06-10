from types import SimpleNamespace

import src.notifications as notifications_module


def enable_notifications(monkeypatch):
    monkeypatch.setattr(
        notifications_module,
        "get_config",
        lambda: SimpleNamespace(enable_notifications=True),
    )


def make_manager(monkeypatch):
    monkeypatch.setattr(notifications_module, "TOAST_AVAILABLE", False)
    enable_notifications(monkeypatch)
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


def test_transcription_copied_notification_uses_success_message(monkeypatch):
    manager = make_manager(monkeypatch)
    messages = []

    monkeypatch.setattr(
        manager,
        "notify",
        lambda title, message, duration=3, threaded=True: messages.append(
            (title, message, duration, threaded)
        ),
    )

    manager.notify_transcription_copied()

    assert messages == [("murmur", "Transcription copied to clipboard.", 3, True)]


def test_clipboard_failure_notifications_use_retry_wording(monkeypatch):
    manager = make_manager(monkeypatch)
    errors = []

    monkeypatch.setattr(manager, "notify_error", errors.append)

    manager.notify_clipboard_failure_retry()
    manager.notify_clipboard_failure_retry_with_training_data()

    assert errors == [
        "Clipboard copy failed. Please retry the recording.",
        (
            "Clipboard copy failed. Please retry the recording. "
            "The transcript was saved to training data."
        ),
    ]


def test_clipboard_failure_fallback_suppresses_message_body(monkeypatch, capsys):
    manager = make_manager(monkeypatch)

    manager.notify_clipboard_failure_retry_with_training_data()

    stdout = capsys.readouterr().out
    assert "Notification unavailable; message suppressed." in stdout
    assert "Clipboard copy failed" not in stdout
    assert "training data" not in stdout


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
    enable_notifications(monkeypatch)
    manager = notifications_module.NotificationManager()

    manager.notify("murmur Error", "private dictated text", threaded=False)

    stdout = capsys.readouterr().out
    assert "Notification fallback: toast delivery failed" in stdout
    assert "title='murmur Error'" in stdout
    assert "message suppressed" in stdout
    assert "private dictated text" not in stdout
    assert "leaked in toast failure" not in stdout


def test_toast_initialization_failure_uses_safe_stdout_fallback(
    monkeypatch,
    capsys,
):
    def fail_to_initialize():
        raise RuntimeError("private initialization detail")

    monkeypatch.setattr(notifications_module, "TOAST_AVAILABLE", True)
    monkeypatch.setattr(
        notifications_module,
        "ToastNotifier",
        fail_to_initialize,
        raising=False,
    )
    enable_notifications(monkeypatch)
    manager = notifications_module.NotificationManager()

    manager.notify("murmur Error", "secret transcript content", threaded=False)

    stdout = capsys.readouterr().out
    assert "Notification fallback: toast initialization failed" in stdout
    assert "title='murmur Error'" in stdout
    assert "message suppressed" in stdout
    assert "secret transcript content" not in stdout
    assert "private initialization detail" not in stdout


def test_threaded_toast_failure_uses_safe_stdout_fallback(monkeypatch, capsys):
    class FailingToaster:
        def show_toast(self, title, message, duration, threaded):
            raise RuntimeError("private threaded failure detail")

    class ImmediateThread:
        def __init__(self, target, args):
            self._target = target
            self._args = args

        def start(self):
            self._target(*self._args)

    monkeypatch.setattr(notifications_module, "TOAST_AVAILABLE", True)
    monkeypatch.setattr(
        notifications_module,
        "ToastNotifier",
        lambda: FailingToaster(),
        raising=False,
    )
    monkeypatch.setattr(notifications_module.threading, "Thread", ImmediateThread)
    enable_notifications(monkeypatch)
    manager = notifications_module.NotificationManager()

    manager.notify("murmur Error", "private threaded message", threaded=True)

    stdout = capsys.readouterr().out
    assert "Notification fallback: toast delivery failed" in stdout
    assert "title='murmur Error'" in stdout
    assert "message suppressed" in stdout
    assert "private threaded message" not in stdout
    assert "private threaded failure detail" not in stdout
