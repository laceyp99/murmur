from types import SimpleNamespace

import numpy as np
import pytest


main_module = pytest.importorskip("src.main")
AudioData = main_module.AudioData
MurmurApp = main_module.MurmurApp


class FakeSegmenter:
    def __init__(self, segments=None, error=None, sample_rate=16000):
        self.sample_rate = sample_rate
        self._segments = list(segments or [])
        self._error = error
        self.calls = []

    def segment_audio(self, audio):
        self.calls.append(audio.copy())
        if self._error is not None:
            raise self._error
        return list(self._segments)


class FakeTranscriber:
    def __init__(self):
        self.segment_calls = []
        self.audio_calls = []

    def transcribe_segments(self, segments, debug=False):
        self.segment_calls.append({"segments": list(segments), "debug": debug})
        return SimpleNamespace(text="Segmented text.")

    def transcribe(self, audio_data):
        self.audio_calls.append(audio_data)
        return "Fallback text."


def make_app(segmenter, transcriber):
    app = MurmurApp.__new__(MurmurApp)
    app.segmenter = segmenter
    app.transcriber = transcriber
    return app


def make_audio_data():
    audio = np.array([0.1, -0.3, 0.2], dtype=np.float32)
    return AudioData(audio=audio, sample_rate=16000, duration=len(audio) / 16000)


def test_transcribe_audio_uses_vad_segments_when_available():
    segment = SimpleNamespace(duration=0.35)
    segmenter = FakeSegmenter(segments=[segment])
    transcriber = FakeTranscriber()
    app = make_app(segmenter, transcriber)

    text = app._transcribe_audio(make_audio_data())

    assert text == "Segmented text."
    assert len(segmenter.calls) == 1
    assert len(transcriber.segment_calls) == 1
    assert transcriber.segment_calls[0]["segments"] == [segment]
    assert transcriber.segment_calls[0]["debug"] is True
    assert transcriber.audio_calls == []


def test_transcribe_audio_falls_back_when_vad_returns_no_segments():
    segmenter = FakeSegmenter(segments=[])
    transcriber = FakeTranscriber()
    app = make_app(segmenter, transcriber)
    audio_data = make_audio_data()

    text = app._transcribe_audio(audio_data)

    assert text == "Fallback text."
    assert len(transcriber.segment_calls) == 0
    assert transcriber.audio_calls == [audio_data]


def test_transcribe_audio_falls_back_when_vad_raises():
    segmenter = FakeSegmenter(error=RuntimeError("vad unavailable"))
    transcriber = FakeTranscriber()
    app = make_app(segmenter, transcriber)
    audio_data = make_audio_data()

    text = app._transcribe_audio(audio_data)

    assert text == "Fallback text."
    assert len(transcriber.segment_calls) == 0
    assert transcriber.audio_calls == [audio_data]