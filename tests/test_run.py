from pathlib import Path

import run


def test_packaging_self_check_returns_success(monkeypatch):
    monkeypatch.setattr(run, "_run_packaging_self_check", lambda: None)

    assert run._packaging_self_check_exit_code() == 0


def test_packaging_self_check_records_failure(monkeypatch, tmp_path: Path):
    error_log = tmp_path / "self-check-error.txt"

    def fail() -> None:
        raise RuntimeError("dependency unavailable")

    monkeypatch.setattr(run, "_run_packaging_self_check", fail)
    monkeypatch.setenv("MURMUR_PACKAGING_SELF_CHECK_LOG", str(error_log))

    assert run._packaging_self_check_exit_code() == 1
    assert "RuntimeError: dependency unavailable" in error_log.read_text(
        encoding="utf-8"
    )
