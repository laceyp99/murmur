import json

import numpy as np

import src.logger as logger_module
from src.audio import AudioData
from src.llm_postprocess import CLEANUP_VERSION
from src.logger import DataLogger


class FakeConfig:
    enable_logging = True


def make_audio_data():
    return AudioData(
        audio=np.zeros(1600, dtype=np.float32),
        sample_rate=16000,
        duration=0.1,
    )


def read_single_entry(log_dir):
    [entry_line] = (
        (log_dir / "transcriptions.jsonl").read_text(encoding="utf-8").splitlines()
    )
    return json.loads(entry_line)


def test_data_logger_writes_ollama_cleanup_pair(monkeypatch, tmp_path):
    monkeypatch.setattr(logger_module, "get_config", lambda: FakeConfig())
    data_logger = DataLogger(log_dir=tmp_path)

    data_logger.log(
        make_audio_data(),
        "Hello, world.",
        cleanup_input="hello world",
        cleanup_status="success",
        cleanup_model="cleanup-model",
        session_id="session-1",
    )

    entry = read_single_entry(tmp_path)
    assert entry == {
        "session_id": "session-1",
        "audio_file": entry["audio_file"],
        "cleanup_input": "hello world",
        "transcription": "Hello, world.",
        "cleanup_status": "success",
        "cleanup_model": "cleanup-model",
        "cleanup_version": CLEANUP_VERSION,
    }
    assert (tmp_path / "audio" / entry["audio_file"]).is_file()


def test_data_logger_omits_cleanup_labels_when_ollama_off(monkeypatch, tmp_path):
    monkeypatch.setattr(logger_module, "get_config", lambda: FakeConfig())
    data_logger = DataLogger(log_dir=tmp_path)

    data_logger.log(
        make_audio_data(),
        "Hello world.",
        cleanup_input="Hello world.",
        cleanup_status="off",
    )

    entry = read_single_entry(tmp_path)
    assert entry["session_id"] is None
    assert entry["cleanup_status"] == "off"
    assert entry["cleanup_model"] is None
    assert entry["cleanup_version"] is None


def test_storage_summary_counts_existing_logged_files_without_creating_dir(
    monkeypatch, tmp_path
):
    monkeypatch.setattr(logger_module, "get_config", lambda: FakeConfig())
    log_dir = tmp_path / "training_data"
    data_logger = DataLogger(log_dir=log_dir)

    empty_summary = data_logger.get_storage_summary()

    assert empty_summary.file_count == 0
    assert empty_summary.total_bytes == 0
    assert not log_dir.exists()

    audio_dir = log_dir / "audio"
    audio_dir.mkdir(parents=True)
    (audio_dir / "sample.wav").write_bytes(b"12345")
    (log_dir / "transcriptions.jsonl").write_text("{}", encoding="utf-8")
    (log_dir / "unrelated.tmp").write_text("ignored", encoding="utf-8")

    summary = data_logger.get_storage_summary()

    assert summary.file_count == 2
    assert summary.total_bytes == 7
