"""
Data logging module for Murmur.
Logs audio recordings and transcriptions for fine-tuning datasets.
"""

import os
import json
import wave
import numpy as np
from pathlib import Path
from datetime import datetime
from typing import Optional
from dataclasses import dataclass, asdict

from .config import get_config
from .audio import AudioData


@dataclass
class TranscriptionLog:
    """Log entry for a transcription."""
    timestamp: str
    audio_file: str
    transcription: str
    duration: float
    model: str
    processing_time: float


class DataLogger:
    """
    Logs audio recordings and transcriptions for training data collection.
    
    Saves:
    - Audio files as WAV format
    - Transcription metadata as JSON
    """
    
    def __init__(self, log_dir: Optional[Path] = None):
        self.config = get_config()
        
        if log_dir is None:
            # Default to a 'training_data' folder in AppData
            app_data = os.environ.get('APPDATA', os.path.expanduser('~'))
            log_dir = Path(app_data) / 'Murmur' / 'training_data'
        
        self.log_dir = Path(log_dir)
        self.audio_dir = self.log_dir / 'audio'
        self.metadata_file = self.log_dir / 'transcriptions.jsonl'
        
        # Create directories
        self.audio_dir.mkdir(parents=True, exist_ok=True)
        
        self._enabled = self.config.get("enable_logging", True)
    
    def log(
        self,
        audio_data: AudioData,
        transcription: str,
        processing_time: float
    ) -> Optional[TranscriptionLog]:
        """
        Log an audio recording and its transcription.
        
        Args:
            audio_data: The recorded audio data
            transcription: The transcription text
            processing_time: Time taken to transcribe in seconds
        
        Returns:
            TranscriptionLog entry if successful, None otherwise
        """
        if not self._enabled:
            return None
        
        if not transcription.strip():
            return None
        
        try:
            # Generate timestamp-based filename
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S_%f")
            audio_filename = f"{timestamp}.wav"
            audio_path = self.audio_dir / audio_filename
            
            # Save audio as WAV
            self._save_wav(audio_path, audio_data)
            
            # Create log entry
            log_entry = TranscriptionLog(
                timestamp=datetime.now().isoformat(),
                audio_file=audio_filename,
                transcription=transcription,
                duration=audio_data.duration,
                model=self.config.get("model", "small"),
                processing_time=processing_time
            )
            
            # Append to JSONL file
            with open(self.metadata_file, 'a', encoding='utf-8') as f:
                f.write(json.dumps(asdict(log_entry), ensure_ascii=False) + '\n')
            
            print(f"📁 Logged to {audio_filename}")
            return log_entry
        
        except Exception as e:
            print(f"⚠️ Failed to log data: {e}")
            return None
    
    def _save_wav(self, path: Path, audio_data: AudioData) -> None:
        """Save audio data as a WAV file."""
        # Convert float32 [-1, 1] to int16
        audio_int16 = (audio_data.audio * 32767).astype(np.int16)
        
        with wave.open(str(path), 'wb') as wav_file:
            wav_file.setnchannels(1)  # Mono
            wav_file.setsampwidth(2)  # 16-bit
            wav_file.setframerate(audio_data.sample_rate)
            wav_file.writeframes(audio_int16.tobytes())
    
    def is_enabled(self) -> bool:
        """Check if logging is enabled."""
        return self._enabled
    
    def set_enabled(self, enabled: bool) -> None:
        """Enable or disable logging."""
        self._enabled = enabled
    
    def get_log_directory(self) -> Path:
        """Get the logging directory path."""
        return self.log_dir
    
    def get_entry_count(self) -> int:
        """Get the number of logged entries."""
        if not self.metadata_file.exists():
            return 0
        
        with open(self.metadata_file, 'r', encoding='utf-8') as f:
            return sum(1 for _ in f)
    
    def get_total_duration(self) -> float:
        """Get total duration of logged audio in seconds."""
        if not self.metadata_file.exists():
            return 0.0
        
        total = 0.0
        with open(self.metadata_file, 'r', encoding='utf-8') as f:
            for line in f:
                try:
                    entry = json.loads(line)
                    total += entry.get('duration', 0.0)
                except json.JSONDecodeError:
                    continue
        
        return total


# Global logger instance
_logger: Optional[DataLogger] = None


def get_logger() -> DataLogger:
    """Get the global data logger instance."""
    global _logger
    if _logger is None:
        _logger = DataLogger()
    return _logger
