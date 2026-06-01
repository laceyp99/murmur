"""
Transcription module for Murmur.
Handles Whisper model loading and speech-to-text transcription.
"""

from dataclasses import dataclass
import re
import time
from typing import List, Optional, Protocol, Sequence

import numpy as np
import torch
import whisper

from .config import get_config
from .audio import AudioData
from .llm_postprocess import LLMPostProcessor, OllamaClient
from .user_vocab import load_user_vocab


class AudioSegmentLike(Protocol):
    """Structural type for audio segments accepted by the transcriber."""

    audio: np.ndarray
    sample_rate: int


@dataclass(frozen=True)
class SegmentTranscription:
    """A single segment transcription and its runtime metadata."""

    index: int
    text: str
    duration: float
    processing_time: float


@dataclass(frozen=True)
class TranscriptionResult:
    """Combined segment outputs and the final cleaned document text."""

    text: str
    segments: List[SegmentTranscription]


class Transcriber:
    """
    Handles speech-to-text transcription using OpenAI Whisper.

    Supports GPU acceleration via CUDA when available.
    """

    def __init__(
        self,
        config=None,
        model: Optional[whisper.Whisper] = None,
        device: Optional[str] = None,
        llm_post_processor: Optional[LLMPostProcessor] = None,
    ):
        self.config = config or get_config()
        self._model: Optional[whisper.Whisper] = model
        self._device: str = device or self._get_device()
        self._llm_post_processor = llm_post_processor

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
        self._model = whisper.load_model(self.config.model_name, device=self._device)
        print("Model loaded successfully.")

    def transcribe(self, audio_data: AudioData) -> str:
        """
        Transcribe audio to text.

        Args:
            audio_data: AudioData containing the recorded audio.

        Returns:
            Transcribed text string.
        """
        return self.transcribe_segments([audio_data]).text

    def finalize_text(self, text: str) -> str:
        """Apply document-level cleanup to pre-transcribed text."""
        cleaned_text = self._post_process_document(text)
        if not cleaned_text or not getattr(self.config, "ollama_enabled", False):
            return cleaned_text

        processor = self._get_llm_post_processor()
        if processor is None:
            return cleaned_text

        try:
            return processor.process(cleaned_text)
        except Exception as exc:
            print(f"⚠️ Ollama post-processing failed: {exc}")
            return cleaned_text

    def _get_llm_post_processor(self) -> Optional[LLMPostProcessor]:
        """Build the optional final-pass LLM post-processor lazily."""
        if self._llm_post_processor is not None:
            return self._llm_post_processor

        try:
            self._llm_post_processor = LLMPostProcessor(
                client=OllamaClient(
                    endpoint=self.config.ollama_endpoint,
                    model_name=self.config.ollama_model_name,
                    timeout=float(self.config.ollama_timeout_seconds),
                ),
                user_vocab=load_user_vocab(),
            )
        except Exception as exc:
            print(f"⚠️ Ollama post-processor unavailable: {exc}")
            self._llm_post_processor = None

        return self._llm_post_processor

    def warm_llm_post_processor(self) -> bool:
        """Warm the configured Ollama model without affecting transcription fallback."""
        if not getattr(self.config, "ollama_enabled", False):
            return False

        processor = self._get_llm_post_processor()
        if processor is None:
            return False

        client = getattr(processor, "client", None)
        if client is None or not hasattr(client, "warm"):
            return False

        try:
            return bool(client.warm())
        except Exception as exc:
            print(f"⚠️ Ollama model warmup failed: {exc}")
            return False

    def transcribe_segments(
        self,
        segments: Sequence[AudioSegmentLike],
        debug: bool = False,
    ) -> TranscriptionResult:
        """Transcribe ordered audio segments and return raw and final text."""
        if self._model is None:
            self.load_model()

        segment_results: List[SegmentTranscription] = []
        for index, segment in enumerate(segments):
            start_time = time.time()
            text = self._transcribe_segment_audio(segment.audio)
            elapsed = time.time() - start_time

            if not text:
                continue

            segment_result = SegmentTranscription(
                index=index,
                text=text,
                duration=self._get_audio_duration(segment),
                processing_time=elapsed,
            )
            segment_results.append(segment_result)

            if debug:
                print(
                    f"Segment {index}: {segment_result.duration:.2f}s "
                    f"transcribed in {segment_result.processing_time:.2f}s"
                )

        final_text = self.finalize_text(
            " ".join(segment.text for segment in segment_results)
        )

        return TranscriptionResult(text=final_text, segments=segment_results)

    def transcribe_segment(self, segment: AudioSegmentLike) -> str:
        """Transcribe one segment without document-level cleanup."""
        return self._transcribe_segment_audio(segment.audio)

    def _transcribe_segment_audio(self, audio: np.ndarray) -> str:
        """Transcribe one waveform segment without final document cleanup."""
        if self._model is None:
            self.load_model()

        prepared_audio = self._prepare_audio(audio)
        if prepared_audio.size == 0:
            return ""

        result = self._model.transcribe(
            prepared_audio,
            language=self.config.language,
            fp16=(self._device == "cuda"),
            task="transcribe",
        )

        return self._post_process_segment_text(result["text"])

    def _prepare_audio(self, audio: np.ndarray) -> np.ndarray:
        """Convert audio to a normalized float32 mono waveform for Whisper."""
        normalized_audio = np.asarray(audio, dtype=np.float32).reshape(-1)
        if normalized_audio.size == 0:
            return normalized_audio

        peak = np.max(np.abs(normalized_audio))
        if peak > 0:
            normalized_audio = normalized_audio / peak

        return normalized_audio

    def _post_process_segment_text(self, text: str) -> str:
        """Clean segment text without forcing document-level punctuation."""
        if not text:
            return ""

        cleaned_text = self._fix_common_issues(text.strip())
        cleaned_text = re.sub(r"(?:\.\.\.)+$", "", cleaned_text).strip()
        return cleaned_text

    def _post_process_document(self, text: str) -> str:
        """
        Apply post-processing to the final transcribed document.

        Handles punctuation, capitalization, and minor corrections.
        """
        text = self._fix_common_issues(text)
        if not text:
            return text

        # Capitalize first letter
        text = text[0].upper() + text[1:] if len(text) > 1 else text.upper()

        # Ensure ending punctuation
        if text and text[-1] not in ".!?":
            text += "."

        # Fix common issues
        text = self._fix_common_issues(text)

        return text

    def _get_audio_duration(self, segment: AudioSegmentLike) -> float:
        """Compute segment duration from the waveform length and sample rate."""
        if segment.sample_rate <= 0:
            return 0.0

        return len(np.asarray(segment.audio).reshape(-1)) / segment.sample_rate

    def _fix_common_issues(self, text: str) -> str:
        """Fix common transcription issues."""
        # Fix multiple spaces
        text = re.sub(r"\s+", " ", text)

        # Fix spacing around punctuation
        text = re.sub(r"\s+([.,!?;:])", r"\1", text)

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
            "model_loaded": self.is_model_loaded(),
        }

        if self._device == "cuda":
            info["cuda_device"] = torch.cuda.get_device_name(0)
            info["cuda_memory_allocated"] = (
                f"{torch.cuda.memory_allocated() / 1024**2:.1f} MB"
            )

        return info
