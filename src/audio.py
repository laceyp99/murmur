"""
Audio recording functionality for Murmur.
"""

import numpy as np
import threading
import time
from typing import Callable, Optional
from dataclasses import dataclass

try:
    import sounddevice as sd
except (ImportError, OSError):
    sd = None

from .config import Config, get_config


DEFAULT_MAX_RECORDING_DURATION = 300


@dataclass
class AudioData:
    """Container for recorded audio data."""

    audio: np.ndarray
    sample_rate: int
    duration: float


class AudioRecorder:
    """Records audio from the microphone."""

    def __init__(self, config: Optional[Config] = None):
        self.config = config or get_config()
        self.sample_rate = self.config.get("sample_rate", 16000)
        self.max_recording_duration = self._parse_max_recording_duration(
            self.config.get("max_recording_duration", DEFAULT_MAX_RECORDING_DURATION)
        )
        self._max_recording_samples = max(
            1, int(self.sample_rate * self.max_recording_duration)
        )

        self._recording = False
        self._audio_data = []
        self._lock = threading.RLock()
        self._stream = None
        self._recording_start: Optional[float] = None
        self._block_callback: Optional[Callable[[np.ndarray], None]] = None
        self._on_block_callback_error: Optional[Callable[[Exception], None]] = None
        self._on_recording_limit: Optional[Callable[[float], None]] = None
        self._block_callback_failed = False
        self._recording_limit_reached = False

    def _parse_max_recording_duration(self, raw_duration) -> float:
        """Return a positive recording limit, falling back to the default."""
        try:
            duration = float(raw_duration)
        except (TypeError, ValueError):
            return DEFAULT_MAX_RECORDING_DURATION

        if duration <= 0:
            return DEFAULT_MAX_RECORDING_DURATION

        return duration

    def set_block_callback(
        self,
        callback: Optional[Callable[[np.ndarray], None]],
    ) -> None:
        """Register a lightweight per-block callback used during recording."""
        with self._lock:
            self._block_callback = callback

    def set_block_callback_error_handler(
        self,
        handler: Optional[Callable[[Exception], None]],
    ) -> None:
        """Register a handler for the first per-block callback failure."""
        with self._lock:
            self._on_block_callback_error = handler

    def set_recording_limit_callback(
        self,
        callback: Optional[Callable[[float], None]],
    ) -> None:
        """Register a callback fired when max recording duration is reached."""
        with self._lock:
            self._on_recording_limit = callback

    def start_recording(self):
        """Start recording audio."""
        if sd is None:
            raise RuntimeError(
                "sounddevice/PortAudio is unavailable; install PortAudio before recording"
            )

        with self._lock:
            if self._recording:
                return

            self._recording = True
            self._audio_data = []
            self._recording_start = time.time()
            self._block_callback_failed = False
            self._recording_limit_reached = False

        self._stream = sd.InputStream(
            samplerate=self.sample_rate,
            channels=1,
            dtype=np.float32,
            callback=self._audio_callback,
            blocksize=int(self.sample_rate * 0.1),  # 100ms blocks
        )
        self._stream.start()

    def stop_recording(self) -> Optional[AudioData]:
        """
        Stop recording and return the audio data.

        Returns:
            AudioData containing the recorded audio, or None if no data recorded
        """
        with self._lock:
            if not self._recording and not self._audio_data:
                return None
            self._recording = False

        if self._stream:
            self._stream.stop()
            self._stream.close()
            self._stream = None

        with self._lock:
            if not self._audio_data:
                return None

            audio = np.concatenate(self._audio_data, axis=0).flatten()
            duration = len(audio) / self.sample_rate
            self._audio_data = []
            return AudioData(
                audio=audio, sample_rate=self.sample_rate, duration=duration
            )

    def is_recording(self) -> bool:
        """Check if currently recording."""
        with self._lock:
            return self._recording

    def _audio_callback(self, indata: np.ndarray, frames: int, time_info, status):
        """Callback for audio stream."""
        if status:
            pass  # Ignore status messages

        audio_block: Optional[np.ndarray] = None
        block_callback: Optional[Callable[[np.ndarray], None]] = None
        error_handler: Optional[Callable[[Exception], None]] = None
        recording_limit_callback: Optional[Callable[[float], None]] = None
        recording_limit_duration: Optional[float] = None
        with self._lock:
            if not self._recording:
                return

            samples_recorded = sum(block.shape[0] for block in self._audio_data)
            max_samples = getattr(self, "_max_recording_samples", None)
            available_samples = (
                indata.shape[0]
                if max_samples is None
                else max_samples - samples_recorded
            )

            if available_samples <= 0:
                self._recording = False
            else:
                samples_to_append = min(indata.shape[0], available_samples)
                audio_block = indata[:samples_to_append].copy()
                self._audio_data.append(audio_block)

                reached_limit = (
                    max_samples is not None
                    and samples_recorded + samples_to_append >= max_samples
                )
                if reached_limit:
                    self._recording = False

            if not self._recording and not getattr(
                self, "_recording_limit_reached", False
            ):
                self._recording_limit_reached = True
                recording_limit_callback = getattr(self, "_on_recording_limit", None)
                recording_limit_duration = self.max_recording_duration

            block_callback = self._block_callback
            error_handler = self._on_block_callback_error

        if block_callback is not None and audio_block is not None:
            try:
                block_callback(audio_block)
            except Exception as exc:
                should_report = False
                with self._lock:
                    if not self._block_callback_failed:
                        self._block_callback_failed = True
                        should_report = True
                    self._block_callback = None

                print(f"⚠️ Audio block callback failed; disabling live callback: {exc}")
                if should_report and error_handler is not None:
                    error_handler(exc)

        if (
            recording_limit_callback is not None
            and recording_limit_duration is not None
        ):
            recording_limit_callback(recording_limit_duration)

    def get_recording_duration(self) -> float:
        """Get current recording duration in seconds."""
        with self._lock:
            if self._recording_start and self._recording:
                return time.time() - self._recording_start
            elif self._audio_data:
                return len(np.concatenate(self._audio_data)) / self.sample_rate
            return 0.0
