"""
Voice activity detection and speech segmentation helpers for Murmur.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import List, Optional, Sequence

import numpy as np

try:
    import webrtcvad
except ImportError:  # pragma: no cover - exercised when dependency is missing.
    webrtcvad = None


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
class _SegmentBounds:
    start_sample: int
    end_sample: int


def float32_to_pcm16(audio: np.ndarray) -> bytes:
    """Convert normalized float32 audio to 16-bit PCM bytes."""
    mono_audio = np.asarray(audio, dtype=np.float32).reshape(-1)
    clipped_audio = np.clip(mono_audio, -1.0, 1.0)
    pcm_audio = (clipped_audio * 32767.0).astype(np.int16)
    return pcm_audio.tobytes()


def generate_frames(
    audio: np.ndarray,
    sample_rate: int,
    frame_duration_ms: int,
) -> List[AudioFrame]:
    """Split audio into VAD-sized frames, padding the last frame with zeros."""
    mono_audio = np.asarray(audio, dtype=np.float32).reshape(-1)
    if mono_audio.size == 0:
        return []

    frame_samples = int(sample_rate * frame_duration_ms / 1000)
    if frame_samples <= 0:
        raise ValueError("frame_duration_ms must resolve to at least one sample")

    frames: List[AudioFrame] = []
    for start_sample in range(0, mono_audio.size, frame_samples):
        end_sample = min(start_sample + frame_samples, mono_audio.size)
        frame_audio = mono_audio[start_sample:end_sample].copy()
        if frame_audio.size < frame_samples:
            padded_audio = np.pad(
                frame_audio,
                (0, frame_samples - frame_audio.size),
                mode="constant",
            )
        else:
            padded_audio = frame_audio

        frames.append(
            AudioFrame(
                start_sample=start_sample,
                end_sample=end_sample,
                audio=frame_audio,
                pcm16=float32_to_pcm16(padded_audio),
            )
        )

    return frames


class WebRTCVADSegmenter:
    """Split a waveform into speech segments using WebRTC VAD decisions."""

    def __init__(
        self,
        sample_rate: int = 16000,
        frame_duration_ms: int = 20,
        aggressiveness: int = 1,
        start_padding_ms: int = 300,
        end_padding_ms: int = 500,
        silence_duration_ms: int = 400,
        min_segment_duration_ms: int = 500,
        merge_gap_ms: int = 150,
        vad: Optional[object] = None,
    ):
        if frame_duration_ms not in (10, 20, 30):
            raise ValueError("frame_duration_ms must be 10, 20, or 30")
        if sample_rate <= 0:
            raise ValueError("sample_rate must be positive")

        self.sample_rate = sample_rate
        self.frame_duration_ms = frame_duration_ms
        self.aggressiveness = aggressiveness
        self.start_padding_ms = start_padding_ms
        self.end_padding_ms = end_padding_ms
        self.silence_duration_ms = silence_duration_ms
        self.min_segment_duration_ms = min_segment_duration_ms
        self.merge_gap_ms = merge_gap_ms

        self.frame_samples = int(sample_rate * frame_duration_ms / 1000)
        self.start_padding_samples = int(sample_rate * start_padding_ms / 1000)
        self.end_padding_samples = int(sample_rate * end_padding_ms / 1000)
        self.silence_samples = int(sample_rate * silence_duration_ms / 1000)
        self.min_segment_samples = int(sample_rate * min_segment_duration_ms / 1000)
        self.merge_gap_samples = int(sample_rate * merge_gap_ms / 1000)

        if vad is not None:
            self._vad = vad
        else:
            if webrtcvad is None:
                raise RuntimeError(
                    "webrtcvad-wheels is required to use WebRTCVADSegmenter"
                )
            self._vad = webrtcvad.Vad(aggressiveness)

    def segment_audio(self, audio: np.ndarray) -> List[SpeechSegment]:
        """Return merged speech segments extracted from the provided audio."""
        mono_audio = np.asarray(audio, dtype=np.float32).reshape(-1)
        if mono_audio.size == 0:
            return []

        frames = generate_frames(
            mono_audio,
            sample_rate=self.sample_rate,
            frame_duration_ms=self.frame_duration_ms,
        )
        raw_segments = self._detect_segments(frames, mono_audio.size)
        merged_segments = self._merge_segments(raw_segments)

        return [
            SpeechSegment(
                start_sample=segment.start_sample,
                end_sample=segment.end_sample,
                sample_rate=self.sample_rate,
                audio=mono_audio[segment.start_sample:segment.end_sample].copy(),
            )
            for segment in merged_segments
            if segment.end_sample - segment.start_sample >= self.min_segment_samples
        ]

    def _detect_segments(
        self,
        frames: Sequence[AudioFrame],
        total_samples: int,
    ) -> List[_SegmentBounds]:
        """Build raw segment bounds from per-frame VAD decisions."""
        segments: List[_SegmentBounds] = []
        current_start: Optional[int] = None
        last_speech_end: Optional[int] = None
        silence_run_samples = 0

        for frame in frames:
            is_speech = self._vad.is_speech(frame.pcm16, self.sample_rate)

            if is_speech:
                if current_start is None:
                    current_start = max(0, frame.start_sample - self.start_padding_samples)
                last_speech_end = frame.end_sample
                silence_run_samples = 0
                continue

            if current_start is None:
                continue

            silence_run_samples += self.frame_samples
            if silence_run_samples < self.silence_samples:
                continue

            segment_end = min(total_samples, (last_speech_end or frame.end_sample) + self.end_padding_samples)
            segments.append(
                _SegmentBounds(
                    start_sample=current_start,
                    end_sample=segment_end,
                )
            )
            current_start = None
            last_speech_end = None
            silence_run_samples = 0

        if current_start is not None and last_speech_end is not None:
            segments.append(
                _SegmentBounds(
                    start_sample=current_start,
                    end_sample=min(total_samples, last_speech_end + self.end_padding_samples),
                )
            )

        return segments

    def _merge_segments(self, segments: Sequence[_SegmentBounds]) -> List[_SegmentBounds]:
        """Merge padded segments whose remaining gap is still small."""
        if not segments:
            return []

        merged: List[_SegmentBounds] = [segments[0]]
        for segment in segments[1:]:
            previous = merged[-1]
            gap_samples = segment.start_sample - previous.end_sample

            if gap_samples <= self.merge_gap_samples:
                merged[-1] = _SegmentBounds(
                    start_sample=previous.start_sample,
                    end_sample=max(previous.end_sample, segment.end_sample),
                )
                continue

            merged.append(segment)

        return merged
