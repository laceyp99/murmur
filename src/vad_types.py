from __future__ import annotations

from dataclasses import dataclass

import numpy as np


@dataclass(frozen=True)
class AudioFrame:
    """A fixed-duration frame prepared for VAD evaluation."""

    start_sample: int
    end_sample: int
    audio: np.ndarray
    pcm16: bytes

    @property
    def sample_count(self) -> int:
        """Return the number of real samples in the frame before padding."""
        return self.end_sample - self.start_sample


@dataclass(frozen=True)
class SpeechSegment:
    """A speech segment extracted from a larger audio clip."""

    start_sample: int
    end_sample: int
    sample_rate: int
    audio: np.ndarray

    @property
    def duration(self) -> float:
        """Return segment duration in seconds."""
        return (self.end_sample - self.start_sample) / self.sample_rate


@dataclass(frozen=True)
class LiveSpeechSegment(SpeechSegment):
    """A speech segment emitted while recording is still active."""

    segment_id: int


@dataclass(frozen=True)
class _SegmentBounds:
    start_sample: int
    end_sample: int
