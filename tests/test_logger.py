import json

import numpy as np

import src.logger as logger_module
from src.audio import AudioData
from src.logger import DataLogger
from src.transcription_live import LiveSegmentMetrics


class FakeConfig:
    enable_logging = True

    def get(self, key, default=None):
        if key == "model":
            return "small"
        return default


def make_audio_data():
    return AudioData(
        audio=np.zeros(1600, dtype=np.float32),
        sample_rate=16000,
        duration=0.1,
    )


def test_data_logger_writes_live_segment_metrics(monkeypatch, tmp_path):
    monkeypatch.setattr(logger_module, "get_config", lambda: FakeConfig())
    data_logger = DataLogger(log_dir=tmp_path)

    data_logger.log(
        make_audio_data(),
        "hello",
        1.25,
        LiveSegmentMetrics(
            segment_count=2,
            latency_avg_seconds=0.35,
            latency_max_seconds=0.6,
        ),
    )

    [entry_line] = (
        (tmp_path / "transcriptions.jsonl").read_text(encoding="utf-8").splitlines()
    )
    entry = json.loads(entry_line)

    assert entry["live_segment_count"] == 2
    assert entry["live_segment_latency_avg_seconds"] == 0.35
    assert entry["live_segment_latency_max_seconds"] == 0.6


def test_data_logger_writes_empty_live_segment_metrics_by_default(
    monkeypatch, tmp_path
):
    monkeypatch.setattr(logger_module, "get_config", lambda: FakeConfig())
    data_logger = DataLogger(log_dir=tmp_path)

    data_logger.log(make_audio_data(), "hello", 1.25)

    [entry_line] = (
        (tmp_path / "transcriptions.jsonl").read_text(encoding="utf-8").splitlines()
    )
    entry = json.loads(entry_line)

    assert entry["live_segment_count"] == 0
    assert entry["live_segment_latency_avg_seconds"] is None
    assert entry["live_segment_latency_max_seconds"] is None
