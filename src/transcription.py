"""
Transcription module for Murmur.
Handles Whisper model loading and speech-to-text transcription.
"""

import numpy as np
import torch
import whisper
from typing import Optional
import re

from .config import get_config
from .audio import AudioData


class Transcriber:
    """
    Handles speech-to-text transcription using OpenAI Whisper.
    
    Supports GPU acceleration via CUDA when available.
    """
    
    def __init__(self):
        self.config = get_config()
        self._model: Optional[whisper.Whisper] = None
        self._device: str = self._get_device()
    
    def _get_device(self) -> str:
        """Determine the best device to use for inference."""
        if self.config.device == "cuda" and torch.cuda.is_available():
            return "cuda"
        return "cpu"
    
    def load_model(self) -> None:
        """
        Load the Whisper model.
        
        This may take some time on first run as the model is downloaded.
        """
        if self._model is not None:
            return
        
        print(f"Loading Whisper model '{self.config.model_name}' on {self._device}...")
        self._model = whisper.load_model(
            self.config.model_name,
            device=self._device
        )
        print("Model loaded successfully.")
    
    def transcribe(self, audio_data: AudioData) -> str:
        """
        Transcribe audio to text.
        
        Args:
            audio_data: AudioData containing the recorded audio.
        
        Returns:
            Transcribed text string.
        """
        if self._model is None:
            self.load_model()
        
        # Ensure audio is in the correct format
        audio = audio_data.audio.astype(np.float32)
        
        # Normalize audio
        if np.max(np.abs(audio)) > 0:
            audio = audio / np.max(np.abs(audio))
        
        # Transcribe
        result = self._model.transcribe(
            audio,
            language=self.config.language,
            fp16=(self._device == "cuda"),
            task="transcribe"
        )
        
        text = result["text"].strip()
        
        # Apply post-processing
        text = self._post_process(text)
        
        return text
    
    def _post_process(self, text: str) -> str:
        """
        Apply post-processing to the transcribed text.
        
        Handles punctuation, capitalization, and minor corrections.
        """
        if not text:
            return text
        
        # Capitalize first letter
        text = text[0].upper() + text[1:] if len(text) > 1 else text.upper()
        
        # Ensure ending punctuation
        if text and text[-1] not in '.!?':
            text += '.'
        
        # Fix common issues
        text = self._fix_common_issues(text)
        
        return text
    
    def _fix_common_issues(self, text: str) -> str:
        """Fix common transcription issues."""
        # Fix multiple spaces
        text = re.sub(r'\s+', ' ', text)
        
        # Fix spacing around punctuation
        text = re.sub(r'\s+([.,!?;:])', r'\1', text)
        
        # Fix capitalization after sentence endings
        text = re.sub(r'([.!?])\s+([a-z])', lambda m: m.group(1) + ' ' + m.group(2).upper(), text)
        
        return text.strip()
    
    def is_model_loaded(self) -> bool:
        """Check if the model is loaded."""
        return self._model is not None
    
    def unload_model(self) -> None:
        """Unload the model to free memory."""
        self._model = None
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
    
    def get_device_info(self) -> dict:
        """Get information about the current device."""
        info = {
            "device": self._device,
            "model": self.config.model_name,
            "model_loaded": self.is_model_loaded()
        }
        
        if self._device == "cuda":
            info["cuda_device"] = torch.cuda.get_device_name(0)
            info["cuda_memory_allocated"] = f"{torch.cuda.memory_allocated() / 1024**2:.1f} MB"
        
        return info
