from __future__ import annotations

from typing import List, Optional, Sequence

import numpy as np

from .vad_audio import generate_frames, resample_audio
from .vad_config import VADSettings
from .vad_types import AudioFrame, SpeechSegment, _SegmentBounds

try:
    import webrtcvad
except ImportError:  # pragma: no cover - exercised when dependency is missing.
    webrtcvad = None


def _create_vad(aggressiveness: int) -> object:
    """Construct a WebRTC VAD instance or fail with a clear message."""
    if webrtcvad is None:
        raise RuntimeError("webrtcvad-wheels is required to use Murmur VAD")
    return webrtcvad.Vad(aggressiveness)


class WebRTCVADSegmenter:
    """Split a waveform into speech segments using WebRTC VAD decisions."""

    def __init__(self, settings: Optional[VADSettings] = None, vad: Optional[object] = None, **kwargs):
        if settings is not None and kwargs:
            raise ValueError("Provide either settings or keyword VAD parameters, not both")

        self.settings = settings or VADSettings(**kwargs)
        self.sample_rate = self.settings.sample_rate
        self.vad_sample_rate = self.settings.vad_sample_rate
        self.frame_duration_ms = self.settings.frame_duration_ms
        self.aggressiveness = self.settings.aggressiveness
        self.start_padding_ms = self.settings.start_padding_ms
        self.end_padding_ms = self.settings.end_padding_ms
        self.silence_duration_ms = self.settings.silence_duration_ms
        self.min_segment_duration_ms = self.settings.min_segment_duration_ms
        self.merge_gap_ms = self.settings.merge_gap_ms
        self.frame_samples = self.settings.frame_samples
        self.start_padding_samples = self.settings.start_padding_samples
        self.end_padding_samples = self.settings.end_padding_samples
        self.silence_samples = self.settings.silence_samples
        self.min_segment_samples = self.settings.min_segment_samples
        self.min_output_segment_samples = self.settings.min_output_segment_samples
        self.merge_gap_samples = self.settings.merge_gap_samples
        self._vad = vad or _create_vad(self.aggressiveness)

    def segment_audio(self, audio: np.ndarray) -> List[SpeechSegment]:
        """Return merged speech segments extracted from the provided audio."""
        mono_audio = np.asarray(audio, dtype=np.float32).reshape(-1)
        if mono_audio.size == 0:
            return []

        vad_audio = self._prepare_vad_audio(mono_audio)
        frames = generate_frames(
            vad_audio,
            sample_rate=self.vad_sample_rate,
            frame_duration_ms=self.frame_duration_ms,
        )
        raw_segments = self._detect_segments(frames, vad_audio.size)
        merged_segments = self._merge_segments(raw_segments)
        output_segments = self._map_segments_to_output_samples(
            merged_segments,
            total_samples=mono_audio.size,
        )

        return [
            SpeechSegment(
                start_sample=segment.start_sample,
                end_sample=segment.end_sample,
                sample_rate=self.sample_rate,
                audio=mono_audio[segment.start_sample:segment.end_sample].copy(),
            )
            for segment in output_segments
            if segment.end_sample - segment.start_sample >= self.min_output_segment_samples
        ]

    def _prepare_vad_audio(self, audio: np.ndarray) -> np.ndarray:
        """Convert input audio to the waveform used for VAD decisions."""
        if self.sample_rate == self.vad_sample_rate:
            return audio.copy()
        return resample_audio(audio, self.sample_rate, self.vad_sample_rate)

    def _map_segments_to_output_samples(
        self,
        segments: Sequence[_SegmentBounds],
        total_samples: int,
    ) -> List[_SegmentBounds]:
        """Map analysis-domain segment bounds back to the original waveform."""
        if self.sample_rate == self.vad_sample_rate:
            return [
                _SegmentBounds(
                    start_sample=max(0, segment.start_sample),
                    end_sample=min(total_samples, segment.end_sample),
                )
                for segment in segments
            ]

        sample_ratio = self.sample_rate / self.vad_sample_rate
        mapped_segments: List[_SegmentBounds] = []
        for segment in segments:
            start_sample = max(0, int(np.floor(segment.start_sample * sample_ratio)))
            end_sample = min(total_samples, int(np.ceil(segment.end_sample * sample_ratio)))
            mapped_segments.append(
                _SegmentBounds(
                    start_sample=start_sample,
                    end_sample=max(start_sample, end_sample),
                )
            )

        return mapped_segments

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
            is_speech = self._vad.is_speech(frame.pcm16, self.vad_sample_rate)

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