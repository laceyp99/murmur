from src.clipboard import ClipboardManager


def test_clipboard_copy_failure_stdout_omits_text_and_exception(monkeypatch, capsys):
    def fail_copy(text):
        raise RuntimeError("private dictated text leaked in clipboard exception")

    monkeypatch.setattr("src.clipboard.pyperclip.copy", fail_copy)
    manager = ClipboardManager()

    assert manager.copy("private dictated text") is False

    stdout = capsys.readouterr().out
    assert "Failed to copy to clipboard." in stdout
    assert "private dictated text" not in stdout
    assert "leaked in clipboard exception" not in stdout
