import threading

import numpy as np
import pytest

from src.vad import (
    LiveVADSegmentationWorker,
    _SegmentBounds,
    WebRTCVADSegmenter,
    float32_to_pcm16,
    generate_frames,
)


class FakeVad:
    def __init__(self, speech_flags):
        self._speech_flags = list(speech_flags)
        self._index = 0
        self.sample_rates = []

    def is_speech(self, pcm16, sample_rate):
        if self._index >= len(self._speech_flags):
            raise AssertionError("FakeVad ran out of frame decisions")
        self.sample_rates.append(sample_rate)
        decision = self._speech_flags[self._index]
        self._index += 1
        return decision


class RaisingVad:
    def __init__(self, error):
        self.error = error
        self.call_count = 0

    def is_speech(self, pcm16, sample_rate):
        self.call_count += 1
        raise self.error


def test_float32_to_pcm16_clips_and_emits_expected_byte_length():
    audio = np.array([-1.5, -1.0, 0.0, 0.5, 1.5], dtype=np.float32)

    pcm_bytes = float32_to_pcm16(audio)
    pcm_audio = np.frombuffer(pcm_bytes, dtype=np.int16)

    assert len(pcm_bytes) == audio.size * 2
    np.testing.assert_array_equal(
        pcm_audio,
        np.array([-32767, -32767, 0, 16383, 32767], dtype=np.int16),
    )


def test_generate_frames_preserves_offsets_and_pads_last_frame():
    audio = np.linspace(-0.75, 0.75, 750, dtype=np.float32)

    frames = generate_frames(audio, sample_rate=16000, frame_duration_ms=20)

    assert len(frames) == 3
    assert (frames[0].start_sample, frames[0].end_sample) == (0, 320)
    assert (frames[1].start_sample, frames[1].end_sample) == (320, 640)
    assert (frames[2].start_sample, frames[2].end_sample) == (640, 750)
    assert frames[0].audio.size == 320
    assert frames[2].audio.size == 110
    assert len(frames[2].pcm16) == 640


def test_segment_audio_merges_padded_segments_across_short_gap():
    sample_rate = 16000
    frame_samples = 320
    audio = np.ones(frame_samples * 11, dtype=np.float32)
    fake_vad = FakeVad(
        [
            False,
            True,
            True,
            False,
            False,
            False,
            True,
            True,
            False,
            False,
            False,
        ]
    )
    segmenter = WebRTCVADSegmenter(
        sample_rate=sample_rate,
        frame_duration_ms=20,
        aggressiveness=1,
        start_padding_ms=20,
        end_padding_ms=20,
        silence_duration_ms=60,
        min_segment_duration_ms=20,
        merge_gap_ms=40,
        vad=fake_vad,
    )

    segments = segmenter.segment_audio(audio)

    assert len(segments) == 1
    assert segments[0].start_sample == 0
    assert segments[0].end_sample == frame_samples * 9
    np.testing.assert_array_equal(audio[: frame_samples * 9], segments[0].audio)


def test_segment_audio_flushes_trailing_speech_at_end_of_clip():
    sample_rate = 16000
    frame_samples = 320
    audio = np.ones(frame_samples * 3, dtype=np.float32)
    segmenter = WebRTCVADSegmenter(
        sample_rate=sample_rate,
        frame_duration_ms=20,
        start_padding_ms=0,
        end_padding_ms=20,
        silence_duration_ms=40,
        min_segment_duration_ms=20,
        merge_gap_ms=0,
        vad=FakeVad([False, True, True]),
    )

    segments = segmenter.segment_audio(audio)

    assert len(segments) == 1
    assert segments[0].start_sample == frame_samples
    assert segments[0].end_sample == frame_samples * 3


def test_segment_audio_drops_segments_shorter_than_minimum_duration():
    sample_rate = 16000
    frame_samples = 320
    audio = np.ones(frame_samples * 2, dtype=np.float32)
    segmenter = WebRTCVADSegmenter(
        sample_rate=sample_rate,
        frame_duration_ms=20,
        start_padding_ms=0,
        end_padding_ms=0,
        silence_duration_ms=20,
        min_segment_duration_ms=60,
        merge_gap_ms=0,
        vad=FakeVad([True, False]),
    )

    assert segmenter.segment_audio(audio) == []


def test_merge_segments_merges_gap_equal_to_threshold_only():
    sample_rate = 16000
    segmenter = WebRTCVADSegmenter(
        sample_rate=sample_rate,
        frame_duration_ms=20,
        merge_gap_ms=40,
        vad=FakeVad([]),
    )
    merged = segmenter._merge_segments(
        [
            _SegmentBounds(start_sample=0, end_sample=320),
            _SegmentBounds(start_sample=960, end_sample=1280),
            _SegmentBounds(start_sample=1921, end_sample=2240),
        ]
    )

    assert merged == [
        _SegmentBounds(start_sample=0, end_sample=1280),
        _SegmentBounds(start_sample=1921, end_sample=2240),
    ]


def test_segment_audio_resamples_unsupported_source_rates_for_vad():
    sample_rate = 44100
    audio = np.linspace(-1.0, 1.0, 3528, dtype=np.float32)
    fake_vad = FakeVad([False, True, False, False])
    segmenter = WebRTCVADSegmenter(
        sample_rate=sample_rate,
        frame_duration_ms=20,
        start_padding_ms=0,
        end_padding_ms=0,
        silence_duration_ms=20,
        min_segment_duration_ms=20,
        merge_gap_ms=0,
        vad=fake_vad,
    )

    segments = segmenter.segment_audio(audio)

    assert segmenter.vad_sample_rate == 16000
    assert fake_vad.sample_rates == [16000, 16000, 16000, 16000]
    assert len(segments) == 1
    assert segments[0].start_sample == 882
    assert segments[0].end_sample == 1764
    np.testing.assert_array_equal(segments[0].audio, audio[882:1764])


@pytest.mark.parametrize(
    ("kwargs", "message"),
    [
        ({"aggressiveness": 4}, "aggressiveness must be 0, 1, 2, or 3"),
        ({"start_padding_ms": -1}, "start_padding_ms must be non-negative"),
        ({"end_padding_ms": -1}, "end_padding_ms must be non-negative"),
        ({"silence_duration_ms": -1}, "silence_duration_ms must be non-negative"),
    ],
)
def test_segmenter_validates_constructor_parameters(kwargs, message):
    with pytest.raises(ValueError, match=message):
        WebRTCVADSegmenter(sample_rate=16000, vad=FakeVad([]), **kwargs)


def test_live_worker_emits_ordered_segments_for_queued_blocks():
    frame_samples = 320
    worker = LiveVADSegmentationWorker(
        sample_rate=16000,
        frame_duration_ms=20,
        start_padding_ms=20,
        end_padding_ms=20,
        silence_duration_ms=40,
        min_segment_duration_ms=20,
        merge_gap_ms=0,
        vad=FakeVad(
            [
                False,
                True,
                True,
                False,
                False,
                False,
                True,
                True,
                False,
                False,
            ]
        ),
    )
    emitted_segments = []
    worker.on_segment = emitted_segments.append

    worker.start()
    worker.submit_audio_block(np.ones(frame_samples * 5, dtype=np.float32))
    worker.submit_audio_block(np.ones(frame_samples * 5, dtype=np.float32))
    worker.stop()

    assert [segment.segment_id for segment in emitted_segments] == [0, 1]
    assert [segment.start_sample for segment in emitted_segments] == [
        0,
        frame_samples * 5,
    ]
    assert [segment.end_sample for segment in emitted_segments] == [
        frame_samples * 4,
        frame_samples * 9,
    ]
    np.testing.assert_array_equal(
        emitted_segments[0].audio,
        np.ones(frame_samples * 4, dtype=np.float32),
    )
    np.testing.assert_array_equal(
        emitted_segments[1].audio,
        np.ones(frame_samples * 4, dtype=np.float32),
    )


def test_live_worker_retains_partial_frame_between_blocks():
    worker = LiveVADSegmentationWorker(
        sample_rate=16000,
        frame_duration_ms=20,
        start_padding_ms=0,
        end_padding_ms=0,
        silence_duration_ms=20,
        min_segment_duration_ms=20,
        vad=FakeVad([True, False]),
    )
    emitted_segments = []
    worker.on_segment = emitted_segments.append

    worker.start()
    worker.submit_audio_block(np.ones(160, dtype=np.float32))
    worker.submit_audio_block(np.ones(480, dtype=np.float32))
    worker.stop()

    assert len(emitted_segments) == 1
    assert emitted_segments[0].start_sample == 0
    assert emitted_segments[0].end_sample == 320


def test_live_worker_flushes_trailing_speech_on_stop():
    frame_samples = 320
    worker = LiveVADSegmentationWorker(
        sample_rate=16000,
        frame_duration_ms=20,
        start_padding_ms=0,
        end_padding_ms=20,
        silence_duration_ms=40,
        min_segment_duration_ms=20,
        merge_gap_ms=0,
        vad=FakeVad([False, True, True]),
    )
    emitted_segments = []
    worker.on_segment = emitted_segments.append

    worker.start()
    worker.submit_audio_block(np.ones(frame_samples * 3, dtype=np.float32))
    worker.stop()

    assert len(emitted_segments) == 1
    assert emitted_segments[0].segment_id == 0
    assert emitted_segments[0].start_sample == frame_samples
    assert emitted_segments[0].end_sample == frame_samples * 3


def test_live_worker_merges_segments_across_short_gap_before_emitting():
    frame_samples = 320
    worker = LiveVADSegmentationWorker(
        sample_rate=16000,
        frame_duration_ms=20,
        start_padding_ms=20,
        end_padding_ms=20,
        silence_duration_ms=40,
        min_segment_duration_ms=20,
        merge_gap_ms=40,
        vad=FakeVad(
            [
                False,
                True,
                True,
                False,
                False,
                False,
                True,
                True,
                False,
                False,
                False,
            ]
        ),
    )
    emitted_segments = []
    worker.on_segment = emitted_segments.append

    worker.start()
    worker.submit_audio_block(np.ones(frame_samples * 11, dtype=np.float32))
    worker.stop()

    assert len(emitted_segments) == 1
    assert emitted_segments[0].segment_id == 0
    assert emitted_segments[0].start_sample == 0
    assert emitted_segments[0].end_sample == frame_samples * 9
    np.testing.assert_array_equal(
        emitted_segments[0].audio,
        np.ones(frame_samples * 9, dtype=np.float32),
    )


def test_live_worker_degrades_once_after_segment_callback_failure(capsys):
    frame_samples = 320
    fake_vad = FakeVad([True, False, False])
    degraded_event = threading.Event()
    degraded_messages = []

    def fail_segment_callback(segment):
        raise RuntimeError("private dictated text leaked in callback")

    def record_degraded(message):
        degraded_messages.append(message)
        degraded_event.set()

    worker = LiveVADSegmentationWorker(
        sample_rate=16000,
        frame_duration_ms=20,
        start_padding_ms=0,
        end_padding_ms=0,
        silence_duration_ms=20,
        min_segment_duration_ms=20,
        merge_gap_ms=0,
        vad=fake_vad,
        on_segment=fail_segment_callback,
        on_worker_degraded=record_degraded,
        queue_timeout_seconds=0.01,
    )

    worker.start()
    try:
        worker.submit_audio_block(np.ones(frame_samples * 3, dtype=np.float32))
        assert degraded_event.wait(timeout=1.0)
    finally:
        worker.stop()

    processed_frame_count = fake_vad._index
    worker.start()
    worker.submit_audio_block(np.ones(frame_samples, dtype=np.float32))

    assert worker.is_degraded() is True
    assert worker.get_last_error() == (
        "Live VAD segment callback failed with RuntimeError. "
        "Live segmentation disabled."
    )
    assert degraded_messages == [worker.get_last_error()]
    assert worker._running is False
    assert worker._pending_segment is None
    assert worker._pending_gap_frames == []
    assert fake_vad._index == processed_frame_count

    stdout = capsys.readouterr().out
    assert "RuntimeError" in stdout
    assert "private dictated text" not in stdout
    assert "leaked in callback" not in stdout


def test_live_worker_degrades_once_after_internal_worker_failure(capsys):
    frame_samples = 320
    raising_vad = RaisingVad(
        RuntimeError("private dictated text leaked from VAD internals")
    )
    degraded_event = threading.Event()
    degraded_messages = []

    def record_degraded(message):
        degraded_messages.append(message)
        degraded_event.set()

    worker = LiveVADSegmentationWorker(
        sample_rate=16000,
        frame_duration_ms=20,
        start_padding_ms=0,
        end_padding_ms=0,
        silence_duration_ms=20,
        min_segment_duration_ms=20,
        merge_gap_ms=0,
        vad=raising_vad,
        on_worker_degraded=record_degraded,
        queue_timeout_seconds=0.01,
    )

    worker.start()
    try:
        worker.submit_audio_block(np.ones(frame_samples, dtype=np.float32))
        assert degraded_event.wait(timeout=1.0)
    finally:
        worker.stop()

    worker.start()
    worker.submit_audio_block(np.ones(frame_samples, dtype=np.float32))

    assert worker.is_degraded() is True
    assert worker.get_last_error() == (
        "Live VAD worker failed with RuntimeError. Live segmentation disabled."
    )
    assert degraded_messages == [worker.get_last_error()]
    assert worker._running is False
    assert raising_vad.call_count == 1

    stdout = capsys.readouterr().out
    assert "RuntimeError" in stdout
    assert "private dictated text" not in stdout
    assert "leaked from VAD internals" not in stdout
