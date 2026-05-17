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
    ollama_enabled = False
    ollama_endpoint = "http://localhost:11434"
    ollama_model_name = "llama3.2:1b"
    ollama_timeout_seconds = 15


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


class FakeLLMPostProcessor:
    def __init__(self, response=None, error=None):
        self.response = response
        self.error = error
        self.calls = []
        self.client = self

    def process(self, text):
        self.calls.append(text)
        if self.error is not None:
            raise self.error
        return self.response

    def warm(self):
        if self.error is not None:
            raise self.error
        return True


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


def test_finalize_text_uses_llm_post_processor_when_enabled():
    config = FakeConfig()
    config.ollama_enabled = True
    fake_processor = FakeLLMPostProcessor(response="Hello, world!")
    transcriber = Transcriber(
        config=config,
        model=FakeModel([]),
        device="cpu",
        llm_post_processor=fake_processor,
    )

    result = transcriber.finalize_text("hello world")

    assert result == "Hello, world!"
    assert fake_processor.calls == ["Hello world."]


def test_finalize_text_falls_back_when_llm_post_processor_fails():
    config = FakeConfig()
    config.ollama_enabled = True
    fake_processor = FakeLLMPostProcessor(error=RuntimeError("offline"))
    transcriber = Transcriber(
        config=config,
        model=FakeModel([]),
        device="cpu",
        llm_post_processor=fake_processor,
    )

    result = transcriber.finalize_text("multiple   spaces")

    assert result == "Multiple spaces."
    assert fake_processor.calls == ["Multiple spaces."]


def test_warm_llm_post_processor_returns_false_on_warm_failure():
    config = FakeConfig()
    config.ollama_enabled = True
    fake_processor = FakeLLMPostProcessor(error=RuntimeError("offline"))
    transcriber = Transcriber(
        config=config,
        model=FakeModel([]),
        device="cpu",
        llm_post_processor=fake_processor,
    )

    assert transcriber.warm_llm_post_processor() is False


def test_warm_llm_post_processor_returns_true_when_enabled_and_available():
    config = FakeConfig()
    config.ollama_enabled = True
    fake_processor = FakeLLMPostProcessor(response="unused")
    transcriber = Transcriber(
        config=config,
        model=FakeModel([]),
        device="cpu",
        llm_post_processor=fake_processor,
    )

    assert transcriber.warm_llm_post_processor() is True


def test_get_llm_post_processor_loads_user_vocab(monkeypatch):
    config = FakeConfig()
    config.ollama_enabled = True
    captured = {}

    class StubOllamaClient:
        def __init__(self, endpoint, model_name, timeout):
            captured["client_args"] = {
                "endpoint": endpoint,
                "model_name": model_name,
                "timeout": timeout,
            }

    class StubLLMPostProcessor:
        def __init__(self, client, user_vocab=None):
            captured["client"] = client
            captured["user_vocab"] = user_vocab

    monkeypatch.setattr("src.transcription.OllamaClient", StubOllamaClient)
    monkeypatch.setattr("src.transcription.LLMPostProcessor", StubLLMPostProcessor)
    monkeypatch.setattr(
        "src.transcription.load_user_vocab",
        lambda: {"brew ridge": "Blue Ridge Data"},
    )

    transcriber = Transcriber(config=config, model=FakeModel([]), device="cpu")

    processor = transcriber._get_llm_post_processor()

    assert processor is captured["client"] or processor is not None
    assert captured["client_args"] == {
        "endpoint": "http://localhost:11434",
        "model_name": "llama3.2:1b",
        "timeout": 15.0,
    }
    assert captured["user_vocab"] == {"brew ridge": "Blue Ridge Data"}