import numpy as np

from src.vad import WebRTCVADSegmenter, generate_frames


class FakeVad:
    def __init__(self, speech_flags):
        self._speech_flags = list(speech_flags)
        self._index = 0

    def is_speech(self, pcm16, sample_rate):
        if self._index >= len(self._speech_flags):
            raise AssertionError("FakeVad ran out of frame decisions")
        decision = self._speech_flags[self._index]
        self._index += 1
        return decision


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