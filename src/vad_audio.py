from __future__ import annotations

from typing import List

import numpy as np

from .vad_types import AudioFrame


def float32_to_pcm16(audio: np.ndarray) -> bytes:
    """Convert normalized float32 audio to 16-bit PCM bytes."""
    mono_audio = np.asarray(audio, dtype=np.float32).reshape(-1)
    clipped_audio = np.clip(mono_audio, -1.0, 1.0)
    pcm_audio = (clipped_audio * 32767.0).astype(np.int16)
    return pcm_audio.tobytes()


def resample_audio(audio: np.ndarray, source_rate: int, target_rate: int) -> np.ndarray:
    """Resample mono float32 audio with linear interpolation."""
    mono_audio = np.asarray(audio, dtype=np.float32).reshape(-1)
    if mono_audio.size == 0 or source_rate == target_rate:
        return mono_audio.copy()

    target_length = max(1, int(round(mono_audio.size * target_rate / source_rate)))
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