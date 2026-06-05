from types import SimpleNamespace

from src.transcription_live import LiveTranscriptionWorker, TranscriptAccumulator


class FakeLiveTranscriber:
    def __init__(self, responses):
        self._responses = dict(responses)
        self.calls = []

    def transcribe_segment(self, segment):
        self.calls.append(segment.segment_id)
        response = self._responses[segment.segment_id]
        if isinstance(response, list):
            response = response.pop(0)
        if isinstance(response, Exception):
            raise response
        return response


def make_segment(segment_id, duration=0.5):
    return SimpleNamespace(
        segment_id=segment_id, audio=[], sample_rate=16000, duration=duration
    )


def test_transcript_accumulator_returns_text_in_segment_order():
    accumulator = TranscriptAccumulator()

    accumulator.add_chunk(
        SimpleNamespace(segment_id=2, text="third", latency_seconds=0.1)
    )
    accumulator.add_chunk(
        SimpleNamespace(segment_id=0, text="first", latency_seconds=0.1)
    )
    accumulator.add_chunk(
        SimpleNamespace(segment_id=1, text="second", latency_seconds=0.1)
    )

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
        on_chunk_appended=lambda chunk, text: appended_text.append(
            (chunk.segment_id, text)
        ),
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
        on_chunk_appended=lambda chunk, text: appended_text.append(
            (chunk.segment_id, text)
        ),
    )

    worker.start()
    worker.submit_segment(make_segment(0))
    worker.submit_segment(make_segment(1))
    worker.stop()

    assert accumulator.get_text() == "next"
    assert appended_text == [(1, "next")]


def test_live_transcription_worker_retries_failed_segment_and_continues_processing(
    capsys,
):
    failed_segments = []
    degraded_messages = []
    accumulator = TranscriptAccumulator()
    worker = LiveTranscriptionWorker(
        transcriber=FakeLiveTranscriber(
            {
                0: [RuntimeError("boom"), RuntimeError("boom again")],
                1: "next",
            }
        ),
        accumulator=accumulator,
        on_segment_failed=lambda segment, exc, attempts: failed_segments.append(
            (segment.segment_id, str(exc), attempts)
        ),
        on_worker_degraded=degraded_messages.append,
        max_segment_retries=1,
    )

    worker.start()
    worker.submit_segment(make_segment(0))
    worker.submit_segment(make_segment(1))
    worker.stop()

    assert worker.is_degraded() is True
    assert worker.failure_count == 1
    assert worker.get_last_error() == (
        "Live transcription failed for segment 0 after 2 attempts."
    )
    assert accumulator.get_text() == "next"
    assert worker.transcriber.calls == [0, 0, 1]
    assert failed_segments == [(0, "boom again", 2)]
    assert degraded_messages == [
        "Live transcription failed for segment 0 after 2 attempts."
    ]
    stdout = capsys.readouterr().out
    assert "boom" not in stdout
    assert "boom again" not in stdout


def test_live_transcription_worker_recovers_from_temporary_chunk_callback_failure(
    capsys,
):
    appended_chunk_ids = []
    appended_text = []
    accumulator = TranscriptAccumulator()

    def temporarily_failing_chunk_callback(chunk, text):
        appended_chunk_ids.append(chunk.segment_id)
        if chunk.segment_id == 0:
            raise RuntimeError("chunk callback broke")
        appended_text.append(text)

    worker = LiveTranscriptionWorker(
        transcriber=FakeLiveTranscriber({0: "first", 1: "second"}),
        accumulator=accumulator,
        on_chunk_appended=temporarily_failing_chunk_callback,
    )

    worker.start()
    worker.submit_segment(make_segment(0))
    worker.submit_segment(make_segment(1))
    worker.stop()

    assert worker.is_degraded() is False
    assert accumulator.get_text() == "first second"
    assert appended_chunk_ids == [0, 1]
    assert appended_text == ["first second"]
    stdout = capsys.readouterr().out
    assert "chunk callback broke" not in stdout
