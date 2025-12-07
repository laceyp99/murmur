"""
Audio recording functionality for Murmur.
"""

import numpy as np
import sounddevice as sd
import threading
import time
from typing import Optional
from dataclasses import dataclass

from .config import Config, get_config


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
        self.max_recording_duration = self.config.get("max_recording_duration", 300)  # 5 minutes max
        
        self._recording = False
        self._audio_data = []
        self._lock = threading.RLock()
        self._stream: Optional[sd.InputStream] = None
        self._recording_start: Optional[float] = None
    
    def start_recording(self):
        """Start recording audio."""
        with self._lock:
            if self._recording:
                return
            
            self._recording = True
            self._audio_data = []
            self._recording_start = time.time()
        
        self._stream = sd.InputStream(
            samplerate=self.sample_rate,
            channels=1,
            dtype=np.float32,
            callback=self._audio_callback,
            blocksize=int(self.sample_rate * 0.1)  # 100ms blocks
        )
        self._stream.start()
    
    def stop_recording(self) -> Optional[AudioData]:
        """
        Stop recording and return the audio data.
        
        Returns:
            AudioData containing the recorded audio, or None if no data recorded
        """
        with self._lock:
            if not self._recording:
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
            return AudioData(audio=audio, sample_rate=self.sample_rate, duration=duration)
    
    def is_recording(self) -> bool:
        """Check if currently recording."""
        with self._lock:
            return self._recording
    
    def _audio_callback(self, indata: np.ndarray, frames: int, 
                        time_info, status):
        """Callback for audio stream."""
        if status:
            pass  # Ignore status messages
        
        with self._lock:
            if not self._recording:
                return
            
            self._audio_data.append(indata.copy())
    
    def get_recording_duration(self) -> float:
        """Get current recording duration in seconds."""
        with self._lock:
            if self._recording_start and self._recording:
                return time.time() - self._recording_start
            elif self._audio_data:
                return len(np.concatenate(self._audio_data)) / self.sample_rate
            return 0.0
