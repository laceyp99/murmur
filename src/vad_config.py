from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import numpy as np


SUPPORTED_VAD_SAMPLE_RATES = (8000, 16000, 32000, 48000)
DEFAULT_VAD_SAMPLE_RATE = 16000


@dataclass(frozen=True)
class VADSettings:
    """Shared VAD timing and aggressiveness settings."""

    sample_rate: int = 16000
    frame_duration_ms: int = 20
    aggressiveness: int = 1
    start_padding_ms: int = 132
    end_padding_ms: int = 220
    silence_duration_ms: int = 320
    min_segment_duration_ms: int = 220
    merge_gap_ms: int = 120

    def __post_init__(self) -> None:
        if self.sample_rate <= 0:
            raise ValueError("sample_rate must be positive")
        if self.frame_duration_ms not in (10, 20, 30):
            raise ValueError("frame_duration_ms must be 10, 20, or 30")
        if self.aggressiveness not in (0, 1, 2, 3):
            raise ValueError("aggressiveness must be 0, 1, 2, or 3")

        timing_values = {
            "start_padding_ms": self.start_padding_ms,
            "end_padding_ms": self.end_padding_ms,
            "silence_duration_ms": self.silence_duration_ms,
            "min_segment_duration_ms": self.min_segment_duration_ms,
            "merge_gap_ms": self.merge_gap_ms,
        }
        for name, value in timing_values.items():
            if value < 0:
                raise ValueError(f"{name} must be non-negative")

    @property
    def vad_sample_rate(self) -> int:
        """Return the rate used for WebRTC VAD analysis."""
        if self.sample_rate in SUPPORTED_VAD_SAMPLE_RATES:
            return self.sample_rate
        return DEFAULT_VAD_SAMPLE_RATE

    @property
    def frame_samples(self) -> int:
        return int(self.vad_sample_rate * self.frame_duration_ms / 1000)

    @property
    def start_padding_samples(self) -> int:
        return int(self.vad_sample_rate * self.start_padding_ms / 1000)

    @property
    def end_padding_samples(self) -> int:
        return int(self.vad_sample_rate * self.end_padding_ms / 1000)

    @property
    def silence_samples(self) -> int:
        return int(self.vad_sample_rate * self.silence_duration_ms / 1000)

    @property
    def min_segment_samples(self) -> int:
        return int(self.vad_sample_rate * self.min_segment_duration_ms / 1000)

    @property
    def min_output_segment_samples(self) -> int:
        return int(self.sample_rate * self.min_segment_duration_ms / 1000)

    @property
    def merge_gap_samples(self) -> int:
        return int(self.vad_sample_rate * self.merge_gap_ms / 1000)

    @property
    def start_padding_frames(self) -> int:
        return max(
            0,
            int(np.ceil(self.vad_sample_rate * self.start_padding_ms / 1000 / self.frame_samples)),
        )

    @property
    def end_padding_frames(self) -> int:
        return max(
            0,
            int(np.ceil(self.vad_sample_rate * self.end_padding_ms / 1000 / self.frame_samples)),
        )

    @property
    def silence_close_frames(self) -> int:
        return max(
            1,
            int(np.ceil(self.vad_sample_rate * self.silence_duration_ms / 1000 / self.frame_samples)),
        )

    @classmethod
    def from_app_config(cls, config: Any, sample_rate: int) -> "VADSettings":
        """Build runtime VAD settings from the application config."""
        end_padding_ms = config.vad_padding_ms
        start_padding_ms = round(end_padding_ms * 0.6)

        return cls(
            sample_rate=sample_rate,
            aggressiveness=config.vad_aggressiveness,
            start_padding_ms=start_padding_ms,
            end_padding_ms=end_padding_ms,
            silence_duration_ms=config.vad_silence_duration_ms,
        )