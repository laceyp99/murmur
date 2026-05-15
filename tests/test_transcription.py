import numpy as np
import pytest


pytest.importorskip("torch")
pytest.importorskip("whisper")

from src.audio import AudioData
from src.transcription import Transcriber


class FakeConfig:
    device = "cpu"
    model_name = "tiny"
    language = None


class FakeModel:
    def __init__(self, responses):
        self._responses = list(responses)
        self.calls = []

    def transcribe(self, audio, language, fp16, task):
        self.calls.append(
            {
                "audio": audio.copy(),
                "language": language,
                "fp16": fp16,
                "task": task,
            }
        )
        return {"text": self._responses.pop(0)}


def test_transcribe_segments_returns_raw_segment_text_and_final_document_text():
    fake_model = FakeModel([" hello   ", "world"])
    transcriber = Transcriber(config=FakeConfig(), model=fake_model, device="cpu")
    segments = [
        AudioData(audio=np.array([0.25, -0.5], dtype=np.float32), sample_rate=16000, duration=0.1),
        AudioData(audio=np.array([0.5, 0.25], dtype=np.float32), sample_rate=16000, duration=0.1),
    ]

    result = transcriber.transcribe_segments(segments)

    assert [segment.text for segment in result.segments] == ["hello", "world"]
    assert result.text == "Hello world."
    assert len(fake_model.calls) == 2
    assert all(call["task"] == "transcribe" for call in fake_model.calls)
    assert all(call["fp16"] is False for call in fake_model.calls)
    assert np.isclose(np.max(np.abs(fake_model.calls[0]["audio"])), 1.0)


def test_transcribe_single_clip_applies_final_document_cleanup_once():
    fake_model = FakeModel(["multiple   spaces"])
    transcriber = Transcriber(config=FakeConfig(), model=fake_model, device="cpu")
    audio_data = AudioData(
        audio=np.array([0.1, -0.2, 0.05], dtype=np.float32),
        sample_rate=16000,
        duration=0.1,
    )

    result = transcriber.transcribe(audio_data)

    assert result == "Multiple spaces."
    assert len(fake_model.calls) == 1