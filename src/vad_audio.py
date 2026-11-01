from __future__ import annotations

import math

import numpy as np

from .vad_types import AudioFrame


def float32_to_pcm16(audio: np.ndarray) -> bytes:
    """Convert normalized float32 audio to 16-bit PCM bytes."""
    mono_audio = np.asarray(audio, dtype=np.float32).reshape(-1)
    clipped_audio = np.clip(mono_audio, -1.0, 1.0)
    pcm_audio = (clipped_audio * 32767.0).astype(np.int16)
    return pcm_audio.tobytes()


def resample_audio(audio: np.ndarray, source_rate: int, target_rate: int) -> np.ndarray:
    """Resample mono audio, low-pass filtering before any rate reduction."""
    if source_rate <= 0 or target_rate <= 0:
        raise ValueError("sample rates must be positive")
    mono_audio = np.asarray(audio, dtype=np.float32).reshape(-1)
    if mono_audio.size == 0 or source_rate == target_rate:
        return mono_audio.copy()

    target_length = max(1, round(mono_audio.size * target_rate / source_rate))
    if source_rate > target_rate:
        return _band_limited_downsample(
            mono_audio, source_rate, target_rate, target_length
        )

    source_positions = np.arange(mono_audio.size, dtype=np.float32)
    target_positions = np.linspace(
        0,
        mono_audio.size - 1,
        num=target_length,
        dtype=np.float32,
    )
    resampled_audio = np.interp(target_positions, source_positions, mono_audio)
    return np.asarray(resampled_audio, dtype=np.float32)


def _band_limited_downsample(
    audio: np.ndarray, source_rate: int, target_rate: int, target_length: int
) -> np.ndarray:
    """Evaluate a windowed-sinc low-pass filter at each fractional output position.

    Filtering and sample selection happen together, so non-integer ratios such as
    22.05 kHz to 16 kHz avoid the upper-band loss of linear interpolation.
    """
    ratio = source_rate / target_rate
    # Scale the filter with the rate ratio so its transition stays similar
    # across source rates.
    half_width = int(np.ceil(16 * ratio))
    cutoff = 0.45 / ratio

    # Output n sits at source position n * source_rate / target_rate. Its
    # fractional phase repeats every `phase_count` outputs, while the source
    # position advances by `source_step`, so each phase is one strided pass.
    divisor = math.gcd(source_rate, target_rate)
    phase_count = min(target_rate // divisor, target_length)
    source_step = source_rate // divisor
    numerators = np.arange(phase_count, dtype=np.int64) * source_rate
    offsets = np.arange(-half_width, half_width + 1, dtype=np.float64)
    distance = offsets[None, :] - (numerators % target_rate)[:, None] / target_rate
    angle = np.pi * distance / (half_width + 1)
    kernels = np.sinc(2 * cutoff * distance) * (
        0.42 + 0.5 * np.cos(angle) + 0.08 * np.cos(2 * angle)
    )
    # Give every phase unity gain at DC.
    kernels = (kernels / kernels.sum(axis=1, keepdims=True)).astype(np.float32)

    # Center the filter to avoid shifting segment boundaries. Extending the
    # endpoint values also preserves DC, even for clips shorter than the filter.
    padded_audio = np.pad(
        audio, (half_width, half_width + int(np.ceil(ratio)) + 1), mode="edge"
    )
    windows = np.lib.stride_tricks.sliding_window_view(padded_audio, offsets.size)
    output = np.empty(target_length, dtype=np.float32)
    for phase, kernel in enumerate(kernels):
        count = len(range(phase, target_length, phase_count))
        start = int(numerators[phase] // target_rate)
        output[phase::phase_count] = windows[start::source_step][:count] @ kernel
    return output


def generate_frames(
    audio: np.ndarray,
    sample_rate: int,
    frame_duration_ms: int,
) -> list[AudioFrame]:
    """Split audio into VAD-sized frames, padding the last frame with zeros."""
    mono_audio = np.asarray(audio, dtype=np.float32).reshape(-1)
    if mono_audio.size == 0:
        return []

    frame_samples = int(sample_rate * frame_duration_ms / 1000)
    if frame_samples <= 0:
        raise ValueError("frame_duration_ms must resolve to at least one sample")

    frames: list[AudioFrame] = []
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
