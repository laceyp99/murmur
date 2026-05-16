from types import SimpleNamespace

from src.transcription_live import LiveTranscriptionWorker, TranscriptAccumulator


class FakeLiveTranscriber:
    def __init__(self, responses):
        self._responses = dict(responses)
        self.calls = []

    def transcribe_segment(self, segment):
        self.calls.append(segment.segment_id)
        return self._responses[segment.segment_id]


def make_segment(segment_id, duration=0.5):
    return SimpleNamespace(segment_id=segment_id, audio=[], sample_rate=16000, duration=duration)


def test_transcript_accumulator_returns_text_in_segment_order():
    accumulator = TranscriptAccumulator()

    accumulator.add_chunk(SimpleNamespace(segment_id=2, text="third", latency_seconds=0.1))
    accumulator.add_chunk(SimpleNamespace(segment_id=0, text="first", latency_seconds=0.1))
    accumulator.add_chunk(SimpleNamespace(segment_id=1, text="second", latency_seconds=0.1))

    assert [chunk.segment_id for chunk in accumulator.ordered_chunks()] == [0, 1, 2]
    assert accumulator.get_text() == "first second third"


def test_live_transcription_worker_processes_segments_serially_and_appends_in_order():
    queued_ids = []
    transcribed_ids = []
    appended_text = []
    accumulator = TranscriptAccumulator()
    worker = LiveTranscriptionWorker(
        transcriber=FakeLiveTranscriber({0: "first", 1: "second"}),
        accumulator=accumulator,
        on_segment_queued=lambda segment: queued_ids.append(segment.segment_id),
        on_segment_transcribed=lambda chunk: transcribed_ids.append(chunk.segment_id),
        on_chunk_appended=lambda chunk, text: appended_text.append((chunk.segment_id, text)),
    )

    worker.start()
    worker.submit_segment(make_segment(0))
    worker.submit_segment(make_segment(1))
    worker.stop()

    assert queued_ids == [0, 1]
    assert transcribed_ids == [0, 1]
    assert accumulator.get_text() == "first second"
    assert appended_text == [(0, "first"), (1, "first second")]


def test_live_transcription_worker_skips_empty_transcript_appends():
    appended_text = []
    accumulator = TranscriptAccumulator()
    worker = LiveTranscriptionWorker(
        transcriber=FakeLiveTranscriber({0: "", 1: "next"}),
        accumulator=accumulator,
        on_chunk_appended=lambda chunk, text: appended_text.append((chunk.segment_id, text)),
    )

    worker.start()
    worker.submit_segment(make_segment(0))
    worker.submit_segment(make_segment(1))
    worker.stop()

    assert accumulator.get_text() == "next"
    assert appended_text == [(1, "next")]