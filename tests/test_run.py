import hashlib
import io
import os
import runpy
import subprocess
import sys
from pathlib import Path
from types import SimpleNamespace

import pytest

import run


@pytest.mark.parametrize("missing_stream", [None, "stdout", "stderr"])
def test_existing_output_streams_are_preserved(monkeypatch, missing_stream):
    with (
        io.StringIO() as stdout,
        io.StringIO() as stderr,
        monkeypatch.context() as streams,
    ):
        streams.setattr(sys, "stdout", stdout)
        streams.setattr(sys, "stderr", stderr)
        if missing_stream is not None:
            streams.setattr(sys, missing_stream, None)
        try:
            run._ensure_output_streams()
            print("console output")
            print("console error", file=sys.stderr)
            if missing_stream != "stdout":
                assert sys.stdout is stdout
                assert stdout.getvalue() == "console output\n"
            if missing_stream != "stderr":
                assert sys.stderr is stderr
                assert stderr.getvalue() == "console error\n"
        finally:
            if missing_stream is not None:
                replacement = getattr(sys, missing_stream)
                if replacement is not None:
                    replacement.close()


def test_windowless_launch_downloads_model_without_saving_output(tmp_path, monkeypatch):
    import whisper

    payload = b"model download fixture"
    checksum = hashlib.sha256(payload).hexdigest()
    url = f"https://example.invalid/{checksum}/model.pt"

    class DownloadResponse(io.BytesIO):
        def info(self):
            return {"Content-Length": str(len(payload))}

    def urlopen(request_url):
        assert request_url == url
        return DownloadResponse(payload)

    def app_main():
        whisper._download(url, str(tmp_path), in_memory=False)
        print("discarded output")
        print("discarded error output", file=sys.stderr)
        sys.stdout.flush()
        sys.stderr.flush()
        assert sys.stdout.name == os.devnull
        assert sys.stderr.name == os.devnull

    monkeypatch.setattr(whisper.urllib.request, "urlopen", urlopen)
    with monkeypatch.context() as windowless:
        windowless.setattr(sys, "stdout", None)
        windowless.setattr(sys, "stderr", None)
        windowless.setattr(sys, "argv", ["run.py"])
        windowless.setitem(sys.modules, "src.main", SimpleNamespace(main=app_main))
        try:
            runpy.run_path(run.__file__, run_name="__main__")
        finally:
            for stream in (sys.stdout, sys.stderr):
                if stream is not None:
                    stream.close()

    assert (tmp_path / "model.pt").read_bytes() == payload
    assert list(tmp_path.iterdir()) == [tmp_path / "model.pt"]


def test_import_does_not_load_desktop_runtime():
    result = subprocess.run(
        [
            sys.executable,
            "-c",
            "import sys; import run; raise SystemExit('src.main' in sys.modules)",
        ],
        cwd=Path(__file__).resolve().parent.parent,
        check=False,
    )

    assert result.returncode == 0


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
