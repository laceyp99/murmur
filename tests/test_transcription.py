from threading import Event

import numpy as np
import pytest

pytest.importorskip("torch")
pytest.importorskip("whisper")

from src.audio import AudioData
from src.cancellation import TranscriptionCancelledError
from src.config import DEFAULT_OLLAMA_MODEL_NAME, DEFAULT_OLLAMA_TIMEOUT_SECONDS
from src.llm_postprocess import LLMPostProcessor, OllamaClient
from src.observations import observe_transcription
from src.transcription import (
    LOGPROB_THRESHOLD,
    NO_SPEECH_THRESHOLD,
    Transcriber,
    _should_accept_whisper_segment,
)
from src.vad_types import LiveSpeechSegment


class FakeConfig:
    device = "cpu"
    model_name = "tiny"
    language = None
    ollama_enabled = False
    ollama_endpoint = "http://localhost:11434"
    ollama_model_name = DEFAULT_OLLAMA_MODEL_NAME
    ollama_timeout_seconds = DEFAULT_OLLAMA_TIMEOUT_SECONDS
    speaker_names = ("Pat",)


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
        response = self._responses.pop(0)
        if isinstance(response, dict):
            return response
        return {"text": response}


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


def test_inference_observation_times_only_model_call_and_preserves_loaded_identity(
    monkeypatch,
):
    clock = [0.0]

    class Model:
        def transcribe(self, *args, **kwargs):
            clock[0] += 2.0
            return {"text": "private dictated sentence"}

    config = FakeConfig()
    monkeypatch.setattr("src.transcription.whisper.load_model", lambda *a, **k: Model())
    monkeypatch.setattr("src.transcription.time.perf_counter", lambda: clock[0])
    transcriber = Transcriber(config=config, device="cpu")
    transcriber.load_model()
    config.model_name = "pending-model"

    def prepare(audio, sample_rate):
        clock[0] += 10.0
        return audio

    monkeypatch.setattr(transcriber, "_prepare_audio", prepare)
    events = []
    with observe_transcription(events.append):
        assert transcriber.transcribe_segment(AudioData(np.ones(160), 16000, 0.01))
    assert len(events) == 1
    assert events[0]["seconds"] == 2.0
    assert events[0]["metadata"] == {"whisper_model": "tiny", "device": "cpu"}
    assert "private" not in repr(events)


def test_failed_inference_observation_preserves_exception_and_unknown_identity():
    error = RuntimeError("private dictated sentence")

    class Model:
        def transcribe(self, *args, **kwargs):
            raise error

    events = []
    transcriber = Transcriber(config=FakeConfig(), model=Model(), device="cpu")
    with observe_transcription(events.append), pytest.raises(RuntimeError) as raised:
        transcriber.transcribe_segment(AudioData(np.ones(160), 16000, 0.01))
    assert raised.value is error
    assert len(events) == 1
    assert events[0]["code"] == "whisper_failed"
    assert events[0]["metadata"]["whisper_model"] is None
    assert "private" not in repr(events)


@pytest.mark.parametrize(
    "mode", ["disabled", "empty", "unavailable", "failure", "applied"]
)
def test_finalize_observes_local_and_optional_cleanup_once(mode, monkeypatch):
    config = FakeConfig()
    config.ollama_enabled = mode != "disabled"

    class Client:
        model_name = "cached-model"

        def chat(self, *args, **kwargs):
            if mode == "failure":
                raise RuntimeError("private dictated sentence")
            return "Private dictated sentence."

    transcriber = Transcriber(
        config=config, llm_post_processor=LLMPostProcessor(Client()), device="cpu"
    )
    if mode == "unavailable":
        monkeypatch.setattr(transcriber, "_get_llm_post_processor", lambda: None)
    events = []
    text = "" if mode == "empty" else "Private dictated sentence."
    with observe_transcription(events.append):
        assert transcriber.finalize_text(text) == text
    assert [event["stage"] for event in events] == ["local_cleanup", "llm_cleanup"]
    assert (
        events[-1]["code"]
        == {
            "disabled": "cleanup_disabled",
            "empty": "cleanup_empty",
            "unavailable": "cleanup_unavailable",
            "failure": "cleanup_request_failed",
            "applied": "cleanup_applied",
        }[mode]
    )
    assert events[-1]["outcome"] == {"disabled": "off", "empty": "skipped"}.get(
        mode, events[-1]["outcome"]
    )
    if mode in {"failure", "applied"}:
        assert events[-1]["metadata"]["cleanup_model"] == "cached-model"
    assert "private" not in repr(events).lower()


def test_disabled_cleanup_reports_off_even_for_empty_text():
    config = FakeConfig()
    config.ollama_enabled = False
    events = []
    with observe_transcription(events.append):
        Transcriber(config=config, device="cpu").finalize_text("")
    assert (events[-1]["outcome"], events[-1]["code"]) == ("off", "cleanup_disabled")


def test_transcription_continues_when_observer_raises():
    def broken(event):
        raise RuntimeError("observer failure")

    transcriber = Transcriber(config=FakeConfig(), model=FakeModel(["hello"]))
    with observe_transcription(broken):
        assert transcriber.transcribe(AudioData(np.ones(160), 16000, 0.01)) == "Hello."


@pytest.mark.parametrize(
    "sample_rate", [8000, 16000, 22050, 24000, 32000, 44100, 48000, 96000]
)
@pytest.mark.parametrize("entrypoint", ["transcribe", "transcribe_segments", "live"])
def test_transcription_preserves_waveform_duration_and_pitch(sample_rate, entrypoint):
    # A one-second tone must remain one second at the same frequency for Whisper.
    positions = np.arange(sample_rate) / sample_rate
    audio = (0.5 * np.sin(2 * np.pi * 440 * positions)).astype(np.float32)
    original_audio = audio.copy()
    model = FakeModel(["hello"])
    transcriber = Transcriber(config=FakeConfig(), model=model, device="cpu")
    clip = AudioData(audio=audio, sample_rate=sample_rate, duration=1.0)

    if entrypoint == "transcribe":
        assert transcriber.transcribe(clip) == "Hello."
    elif entrypoint == "transcribe_segments":
        result = transcriber.transcribe_segments([clip])
        assert result.text == "Hello."
        assert result.segments[0].duration == 1.0
    else:
        segment = LiveSpeechSegment(
            start_sample=0,
            end_sample=sample_rate,
            sample_rate=sample_rate,
            audio=audio,
            segment_id=0,
        )
        assert transcriber.transcribe_segment(segment) == "hello"
        assert segment.duration == 1.0

    prepared_audio = model.calls[0]["audio"]
    assert prepared_audio.shape == (16000,)
    assert prepared_audio.dtype == np.float32
    assert np.max(np.abs(prepared_audio)) == pytest.approx(1.0)
    peak_frequency = np.argmax(np.abs(np.fft.rfft(prepared_audio)))
    assert peak_frequency == 440
    np.testing.assert_array_equal(audio, original_audio)
    if sample_rate == 16000:
        np.testing.assert_array_equal(prepared_audio, audio / np.max(np.abs(audio)))


@pytest.mark.parametrize("entrypoint", ["transcribe", "transcribe_segments", "live"])
def test_transcription_rejects_high_frequency_noise_before_normalization(entrypoint):
    sample_rate = 48000
    positions = np.arange(sample_rate) / sample_rate
    audio = (
        0.2 * np.sin(2 * np.pi * 1000 * positions)
        + 0.6 * np.sin(2 * np.pi * 12000 * positions)
    ).astype(np.float32)
    original_audio = audio.copy()
    model = FakeModel(["hello"])
    transcriber = Transcriber(config=FakeConfig(), model=model, device="cpu")
    clip = AudioData(audio=audio, sample_rate=sample_rate, duration=1.0)

    if entrypoint == "transcribe":
        transcriber.transcribe(clip)
    elif entrypoint == "transcribe_segments":
        transcriber.transcribe_segments([clip])
    else:
        transcriber.transcribe_segment(
            LiveSpeechSegment(
                start_sample=0,
                end_sample=sample_rate,
                sample_rate=sample_rate,
                audio=audio,
                segment_id=0,
            )
        )

    prepared_audio = model.calls[0]["audio"]
    assert np.max(np.abs(prepared_audio)) == pytest.approx(1.0)
    # Measure an integer number of periods away from the padded boundaries.
    spectrum = np.abs(np.fft.rfft(prepared_audio[160:-160]))
    retained_tone = spectrum[980]
    folded_noise = spectrum[3920]  # 12 kHz folds to 4 kHz at 16 kHz.
    assert retained_tone > 1000
    assert folded_noise / retained_tone <= 0.03
    np.testing.assert_array_equal(audio, original_audio)


@pytest.mark.parametrize("sample_rate", [8000, 16000, 48000])
def test_transcribe_empty_audio_skips_whisper(sample_rate):
    model = FakeModel([])
    transcriber = Transcriber(config=FakeConfig(), model=model, device="cpu")
    clip = AudioData(np.array([], dtype=np.float32), sample_rate, 0.0)

    assert transcriber.transcribe(clip) == ""
    assert model.calls == []


@pytest.mark.parametrize("sample_rate", [0, -16000])
def test_transcribe_rejects_invalid_source_rate(sample_rate):
    model = FakeModel([])
    transcriber = Transcriber(config=FakeConfig(), model=model, device="cpu")
    clip = AudioData(np.ones(320, dtype=np.float32), sample_rate, 0.02)

    with pytest.raises(ValueError, match="sample_rate must be positive"):
        transcriber.transcribe(clip)

    assert model.calls == []


def test_cancel_during_whisper_discards_result_and_skips_next_segment_and_llm():
    cancelled = Event()

    class CancellingModel(FakeModel):
        def transcribe(self, *args, **kwargs):
            result = super().transcribe(*args, **kwargs)
            cancelled.set()
            return result

    config = FakeConfig()
    config.ollama_enabled = True
    model = CancellingModel(["first", "second"])
    processor = FakeLLMPostProcessor(response="cleaned")
    transcriber = Transcriber(
        config=config, model=model, device="cpu", llm_post_processor=processor
    )
    audio = AudioData(np.ones(320, dtype=np.float32), 16000, 0.02)

    with pytest.raises(TranscriptionCancelledError):
        transcriber.transcribe_segments([audio, audio], cancel_event=cancelled)

    assert len(model.calls) == 1
    assert processor.calls == []


@pytest.mark.parametrize("raises", [False, True])
def test_cancel_during_llm_discards_return_and_suppresses_fallback(raises):
    cancelled = Event()

    class CancellingProcessor(FakeLLMPostProcessor):
        def process(self, text):
            self.calls.append(text)
            cancelled.set()
            if raises:
                raise RuntimeError("failed after cancellation")
            return "late cleanup"

    config = FakeConfig()
    config.ollama_enabled = True
    processor = CancellingProcessor()
    transcriber = Transcriber(
        config=config, model=FakeModel([]), device="cpu", llm_post_processor=processor
    )

    with pytest.raises(TranscriptionCancelledError):
        transcriber.finalize_text("hello", cancel_event=cancelled)

    assert processor.calls == ["Hello."]


def test_whisper_segment_policy_accepts_likely_speech():
    assert (
        _should_accept_whisper_segment(
            {
                "no_speech_prob": NO_SPEECH_THRESHOLD - 0.1,
                "avg_logprob": LOGPROB_THRESHOLD - 0.5,
            }
        )
        is True
    )


def test_whisper_segment_policy_drops_weak_high_no_speech_segment():
    assert (
        _should_accept_whisper_segment(
            {
                "no_speech_prob": NO_SPEECH_THRESHOLD + 0.1,
                "avg_logprob": LOGPROB_THRESHOLD - 0.5,
            }
        )
        is False
    )


def test_whisper_segment_policy_drops_exact_logprob_threshold():
    assert (
        _should_accept_whisper_segment(
            {
                "no_speech_prob": NO_SPEECH_THRESHOLD + 0.1,
                "avg_logprob": LOGPROB_THRESHOLD,
            }
        )
        is False
    )


def test_whisper_segment_policy_keeps_high_confidence_override():
    assert (
        _should_accept_whisper_segment(
            {
                "no_speech_prob": NO_SPEECH_THRESHOLD + 0.1,
                "avg_logprob": LOGPROB_THRESHOLD + 0.5,
            }
        )
        is True
    )


@pytest.mark.parametrize(
    "segment",
    [
        {"text": "hello"},
        {"text": "hello", "no_speech_prob": NO_SPEECH_THRESHOLD + 0.1},
        {"text": "hello", "avg_logprob": LOGPROB_THRESHOLD - 0.5},
    ],
)
def test_whisper_segment_policy_keeps_segments_with_missing_metadata(segment):
    assert _should_accept_whisper_segment(segment) is True


def test_transcribe_segments_returns_raw_segment_text_and_final_document_text():
    fake_model = FakeModel([" hello   ", "world"])
    transcriber = Transcriber(config=FakeConfig(), model=fake_model, device="cpu")
    segments = [
        AudioData(
            audio=np.array(samples, dtype=np.float32),
            sample_rate=16000,
            duration=0.1,
        )
        for samples in ([0.25, -0.5], [0.5, 0.25])
    ]

    result = transcriber.transcribe_segments(segments)

    assert [segment.text for segment in result.segments] == ["hello", "world"]
    assert result.text == "Hello world."
    assert len(fake_model.calls) == 2
    assert all(call["task"] == "transcribe" for call in fake_model.calls)
    assert all(call["fp16"] is False for call in fake_model.calls)
    assert np.isclose(np.max(np.abs(fake_model.calls[0]["audio"])), 1.0)


def test_transcribe_segments_strip_only_trailing_ellipses_from_segment_text():
    fake_model = FakeModel([" hello... ", "world."])
    transcriber = Transcriber(config=FakeConfig(), model=fake_model, device="cpu")
    segments = [
        AudioData(
            audio=np.array(samples, dtype=np.float32),
            sample_rate=16000,
            duration=0.1,
        )
        for samples in ([0.25, -0.5], [0.5, 0.25])
    ]

    result = transcriber.transcribe_segments(segments)

    assert [segment.text for segment in result.segments] == ["hello", "world."]
    assert result.text == "Hello world."


@pytest.mark.parametrize("entrypoint", ["offline", "live"])
@pytest.mark.parametrize(
    ("texts", "expected"),
    [
        (
            ["The launch is ready.", "Microsoft shipped the patch."],
            "The launch is ready Microsoft shipped the patch.",
        ),
        (
            [
                "we should ask",
                "Microsoft about the patch.",
                "NASA has notes.",
                "I think",
                "We should probably",
                "Wait until tomorrow.",
            ],
            "We should ask Microsoft about the patch NASA has notes "
            "I think We should probably Wait until tomorrow.",
        ),
        (
            ["We should go.", "and bring the laptop."],
            "We should go and bring the laptop.",
        ),
        (
            ["this keeps flowing", "without a break"],
            "This keeps flowing without a break.",
        ),
        (
            ["", "  Hello   there . ", "  ", "Next sentence.", ""],
            "Hello there Next sentence.",
        ),
        (
            ["First sentence. Another thought.", "Microsoft agrees.", ""],
            "First sentence. Another thought Microsoft agrees.",
        ),
        (
            ["Is it ready?", "Really?!", "Yes!", "I agree.", ""],
            "Is it ready? Really?! Yes! I agree.",
        ),
        (["", "  "], ""),
    ],
)
def test_segment_finalization_removes_only_boundary_periods_without_ollama(
    entrypoint, texts, expected
):
    fake_model = FakeModel(texts)
    transcriber = Transcriber(config=FakeConfig(), model=fake_model, device="cpu")
    if entrypoint == "live":
        result = transcriber.finalize_segment_texts(texts)
    else:
        segments = [
            AudioData(
                audio=np.array([0.25, -0.5], dtype=np.float32),
                sample_rate=16000,
                duration=0.1,
            )
            for _ in texts
        ]
        result = transcriber.transcribe_segments(segments).text

    assert result == expected


def test_transcribe_segments_skips_filtered_no_speech_segments():
    fake_model = FakeModel(
        [
            {
                "text": "thank you",
                "segments": [
                    {
                        "text": "thank you",
                        "no_speech_prob": NO_SPEECH_THRESHOLD + 0.1,
                        "avg_logprob": LOGPROB_THRESHOLD - 0.5,
                    }
                ],
            },
            {
                "text": "actual speech",
                "segments": [
                    {
                        "text": "actual speech",
                        "no_speech_prob": NO_SPEECH_THRESHOLD - 0.1,
                        "avg_logprob": LOGPROB_THRESHOLD - 0.5,
                    }
                ],
            },
        ]
    )
    transcriber = Transcriber(config=FakeConfig(), model=fake_model, device="cpu")
    segments = [
        AudioData(
            audio=np.array(samples, dtype=np.float32),
            sample_rate=16000,
            duration=0.1,
        )
        for samples in ([0.25, -0.5], [0.5, 0.25])
    ]

    result = transcriber.transcribe_segments(segments)

    assert [segment.text for segment in result.segments] == ["actual speech"]
    assert result.segments[0].index == 1
    assert result.text == "Actual speech."


def test_transcribe_segment_keeps_high_confidence_no_speech_override():
    fake_model = FakeModel(
        [
            {
                "text": "quiet speech",
                "segments": [
                    {
                        "text": "quiet speech",
                        "no_speech_prob": NO_SPEECH_THRESHOLD + 0.1,
                        "avg_logprob": LOGPROB_THRESHOLD + 0.5,
                    }
                ],
            }
        ]
    )
    transcriber = Transcriber(config=FakeConfig(), model=fake_model, device="cpu")
    audio_data = AudioData(
        audio=np.array([0.1, -0.2, 0.05], dtype=np.float32),
        sample_rate=16000,
        duration=0.1,
    )

    assert transcriber.transcribe_segment(audio_data) == "quiet speech"


def test_transcribe_segment_preserves_top_level_text_when_metadata_is_missing():
    fake_model = FakeModel(
        [
            {
                "text": "top level text",
                "segments": [{"text": "segment text"}],
            }
        ]
    )
    transcriber = Transcriber(config=FakeConfig(), model=fake_model, device="cpu")
    audio_data = AudioData(
        audio=np.array([0.1, -0.2, 0.05], dtype=np.float32),
        sample_rate=16000,
        duration=0.1,
    )

    assert transcriber.transcribe_segment(audio_data) == "top level text"


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


def test_finalize_text_does_not_force_mid_sentence_capitalization():
    transcriber = Transcriber(config=FakeConfig(), model=FakeModel([]), device="cpu")

    result = transcriber.finalize_text("hello. there is more")

    assert result == "Hello. there is more."


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


def test_finalize_segment_texts_sends_period_light_text_to_llm():
    config = FakeConfig()
    config.ollama_enabled = True
    fake_processor = FakeLLMPostProcessor(response="Cleaned by LLM.")
    transcriber = Transcriber(
        config=config,
        model=FakeModel([]),
        device="cpu",
        llm_post_processor=fake_processor,
    )

    result = transcriber.finalize_segment_texts(
        ["this should keep flowing.", "Even after whisper split it."]
    )

    assert result == "Cleaned by LLM."
    assert fake_processor.calls == [
        "This should keep flowing Even after whisper split it."
    ]


def test_transcribe_preserves_whisper_text_when_ollama_content_is_null(capsys):
    class NullContentClient:
        def chat(self, **kwargs):
            return {
                "message": {"content": None},
                "done": True,
                "done_reason": "stop",
            }

    config = FakeConfig()
    config.ollama_enabled = True
    processor = LLMPostProcessor(
        OllamaClient(
            config.ollama_endpoint, config.ollama_model_name, client=NullContentClient()
        )
    )
    transcriber = Transcriber(
        config=config,
        model=FakeModel(["invoice is $100, due May 5"]),
        device="cpu",
        llm_post_processor=processor,
    )
    audio = AudioData(
        audio=np.ones(16000, dtype=np.float32), sample_rate=16000, duration=1.0
    )

    assert transcriber.transcribe(audio) == "Invoice is $100, due May 5."
    assert "Invoice" not in capsys.readouterr().out


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
        def __init__(self, client, user_vocab=None, speaker_names=()):
            captured["client"] = client
            captured["user_vocab"] = user_vocab
            captured["speaker_names"] = speaker_names
            captured["processor_instance"] = self

    monkeypatch.setattr("src.transcription.OllamaClient", StubOllamaClient)
    monkeypatch.setattr("src.transcription.LLMPostProcessor", StubLLMPostProcessor)
    monkeypatch.setattr(
        "src.transcription.load_user_vocab",
        lambda: {"brew ridge": "Blue Ridge Data"},
    )

    transcriber = Transcriber(config=config, model=FakeModel([]), device="cpu")

    processor = transcriber._get_llm_post_processor()
    cached_processor = transcriber._get_llm_post_processor()

    assert processor is captured["processor_instance"]
    assert cached_processor is processor
    assert captured["client_args"] == {
        "endpoint": "http://localhost:11434",
        "model_name": DEFAULT_OLLAMA_MODEL_NAME,
        "timeout": float(DEFAULT_OLLAMA_TIMEOUT_SECONDS),
    }
    assert captured["user_vocab"] == {"brew ridge": "Blue Ridge Data"}
    assert captured["speaker_names"] == ("Pat",)
