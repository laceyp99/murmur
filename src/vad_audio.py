from __future__ import annotations

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
        # Scale the filter with the rate ratio so its transition stays similar
        # across source rates. At 16 kHz output it preserves speech through
        # 6 kHz and rejects noise from 9 kHz upward by at least 40 dB.
        half_width = int(np.ceil(16 * source_rate / target_rate))
        offsets = np.arange(-half_width, half_width + 1, dtype=np.float64)
        cutoff = 0.45 * target_rate / source_rate
        kernel = 2 * cutoff * np.sinc(2 * cutoff * offsets)
        kernel *= np.blackman(kernel.size)
        kernel /= kernel.sum()

        # Center the filter to avoid shifting segment boundaries. Extending the
        # endpoint values also preserves DC, even for clips shorter than the filter.
        padded_audio = np.pad(mono_audio, (half_width, half_width), mode="edge")
        mono_audio = np.convolve(padded_audio, kernel.astype(np.float32), mode="valid")
        source_positions = np.arange(mono_audio.size, dtype=np.float64)
        target_positions = (
            np.arange(target_length, dtype=np.float64) * source_rate / target_rate
        )
    else:
        source_positions = np.arange(mono_audio.size, dtype=np.float32)
        target_positions = np.linspace(
            0,
            mono_audio.size - 1,
            num=target_length,
            dtype=np.float32,
        )
    resampled_audio = np.interp(target_positions, source_positions, mono_audio)
    return np.asarray(resampled_audio, dtype=np.float32)


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
