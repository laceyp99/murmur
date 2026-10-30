"""
Transcription module for murmur.
Handles Whisper model loading and speech-to-text transcription.
"""

import re
import time
from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from threading import Event, Lock
from typing import Protocol

import numpy as np
import torch
import whisper
from whisper.audio import SAMPLE_RATE as WHISPER_SAMPLE_RATE

from .audio import AudioData
from .cancellation import check_cancelled
from .config import get_config
from .llm_postprocess import CleanupStatus, LLMPostProcessor, OllamaClient
from .observations import emit_observation, failure_category
from .user_vocab import load_user_vocab
from .vad_audio import resample_audio

NO_SPEECH_THRESHOLD = 0.4
LOGPROB_THRESHOLD = -0.6


def _should_accept_whisper_segment(segment: Mapping[str, object]) -> bool:
    """Decide whether Whisper metadata points to speech worth keeping."""
    no_speech_prob = _as_optional_float(segment.get("no_speech_prob"))
    avg_logprob = _as_optional_float(segment.get("avg_logprob"))

    if no_speech_prob is None or avg_logprob is None:
        return True

    return not (
        no_speech_prob > NO_SPEECH_THRESHOLD and avg_logprob <= LOGPROB_THRESHOLD
    )


def _has_whisper_filter_metadata(segment: Mapping[str, object]) -> bool:
    """Return whether a Whisper segment has usable no-speech filter metadata."""
    return (
        _as_optional_float(segment.get("no_speech_prob")) is not None
        and _as_optional_float(segment.get("avg_logprob")) is not None
    )


def _as_optional_float(value: object) -> float | None:
    """Coerce Whisper numeric metadata without rejecting odd result shapes."""
    if value is None:
        return None

    try:
        return float(value)
    except (TypeError, ValueError):
        return None


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
class FinalTranscript:
    """Final document text plus the Ollama cleanup input and outcome."""

    text: str
    cleanup_input: str = ""
    cleanup_status: CleanupStatus = "off"
    cleanup_model: str | None = None


@dataclass(frozen=True)
class TranscriptionResult:
    """Combined segment outputs and the final cleaned document."""

    final: FinalTranscript
    segments: list[SegmentTranscription]


class Transcriber:
    """
    Handles speech-to-text transcription using OpenAI Whisper.

    Supports GPU acceleration via CUDA when available.
    """

    def __init__(
        self,
        config=None,
        model: whisper.Whisper | None = None,
        device: str | None = None,
        llm_post_processor: LLMPostProcessor | None = None,
        *,
        model_name: str | None = None,
    ):
        self.config = config or get_config()
        self._model: whisper.Whisper | None = model
        self._model_name = model_name if model is not None else None
        self._device: str = device or self._get_device()
        self._llm_post_processor = llm_post_processor
        # Startup warmup and finalization can both build the processor.
        self._llm_post_processor_lock = Lock()

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
        model_name = self.config.model_name
        self._model = whisper.load_model(model_name, device=self._device)
        self._model_name = model_name
        print("Model loaded successfully.")

    def transcribe(
        self, audio_data: AudioData, *, cancel_event: Event | None = None
    ) -> FinalTranscript:
        """
        Transcribe audio to text.

        Args:
            audio_data: AudioData containing the recorded audio.

        Returns:
            Final transcript with its cleanup details.
        """
        return self.transcribe_segments([audio_data], cancel_event=cancel_event).final

    def finalize_text(
        self, text: str, *, cancel_event: Event | None = None
    ) -> FinalTranscript:
        """Apply document-level cleanup to pre-transcribed text."""
        check_cancelled(cancel_event)
        started = time.perf_counter()
        try:
            cleaned_text = self._post_process_document(text)
        except Exception:
            emit_observation(
                stage="local_cleanup",
                outcome="failed",
                seconds=time.perf_counter() - started,
                code="local_cleanup_failed",
                category="error",
            )
            raise
        emit_observation(
            stage="local_cleanup",
            outcome="success",
            seconds=time.perf_counter() - started,
            code="local_cleanup_completed",
            category="success",
        )
        enabled = bool(getattr(self.config, "ollama_enabled", False))
        if not enabled or not cleaned_text:
            outcome = "off" if not enabled else "skipped"
            emit_observation(
                stage="llm_cleanup",
                outcome=outcome,
                seconds=0.0,
                code="cleanup_disabled" if not enabled else "cleanup_empty",
                category="skip",
                cleanup_enabled=enabled,
            )
            return FinalTranscript(cleaned_text, cleaned_text, outcome)

        started = time.perf_counter()
        processor = self._get_llm_post_processor()
        check_cancelled(cancel_event)
        if processor is None:
            emit_observation(
                stage="llm_cleanup",
                outcome="unavailable",
                seconds=time.perf_counter() - started,
                code="cleanup_unavailable",
                category="unavailable",
                cleanup_enabled=True,
            )
            return FinalTranscript(cleaned_text, cleaned_text, "unavailable")

        # The cached client's model, which can differ from config until restart.
        cleanup_model = getattr(getattr(processor, "client", None), "model_name", None)
        started = time.perf_counter()
        try:
            result = processor.process(cleaned_text)
        except Exception as exc:
            check_cancelled(cancel_event)
            category = failure_category(exc)
            emit_observation(
                stage="llm_cleanup",
                outcome="failed",
                seconds=time.perf_counter() - started,
                code="cleanup_timeout"
                if category == "timeout"
                else "cleanup_request_failed",
                category="error",
                cleanup_enabled=True,
                cleanup_model=cleanup_model,
            )
            print("Ollama post-processing failed; using local transcript cleanup.")
            return FinalTranscript(cleaned_text, cleaned_text, "failed", cleanup_model)
        check_cancelled(cancel_event)
        return FinalTranscript(result.text, cleaned_text, result.status, cleanup_model)

    def finalize_segment_texts(
        self, segment_texts: Sequence[str], *, cancel_event: Event | None = None
    ) -> FinalTranscript:
        """Join segments with fewer boundary periods, then clean the document."""
        check_cancelled(cancel_event)
        return self.finalize_text(
            self._join_segment_texts(segment_texts), cancel_event=cancel_event
        )

    def _get_llm_post_processor(self) -> LLMPostProcessor | None:
        """Build the optional final-pass LLM post-processor lazily."""
        with self._llm_post_processor_lock:
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
                    speaker_names=self.config.speaker_names,
                )
            except Exception:
                print("Warning: Ollama post-processor unavailable.")
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
            print(f"Warning: Ollama model warmup failed: {exc}")
            return False

    def transcribe_segments(
        self,
        segments: Sequence[AudioSegmentLike],
        debug: bool = False,
        *,
        cancel_event: Event | None = None,
    ) -> TranscriptionResult:
        """Transcribe ordered audio segments and return raw and final text."""
        check_cancelled(cancel_event)
        if self._model is None:
            self.load_model()

        segment_results: list[SegmentTranscription] = []
        for index, segment in enumerate(segments):
            check_cancelled(cancel_event)
            start_time = time.time()
            text = self._transcribe_segment_audio(segment.audio, segment.sample_rate)
            check_cancelled(cancel_event)
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

        final = self.finalize_segment_texts(
            [segment.text for segment in segment_results], cancel_event=cancel_event
        )

        return TranscriptionResult(final=final, segments=segment_results)

    def transcribe_segment(self, segment: AudioSegmentLike) -> str:
        """Transcribe one segment without document-level cleanup."""
        return self._transcribe_segment_audio(segment.audio, segment.sample_rate)

    def _transcribe_segment_audio(self, audio: np.ndarray, sample_rate: int) -> str:
        """Transcribe one waveform segment without final document cleanup."""
        if self._model is None:
            self.load_model()

        prepared_audio = self._prepare_audio(audio, sample_rate)
        if prepared_audio.size == 0:
            return ""

        started = time.perf_counter()
        try:
            result = self._model.transcribe(
                prepared_audio,
                language=self.config.language,
                fp16=(self._device == "cuda"),
                task="transcribe",
            )
        except Exception:
            emit_observation(
                outcome="failed",
                seconds=time.perf_counter() - started,
                code="whisper_failed",
                category="error",
                whisper_model=self._model_name,
                device=self._device,
            )
            raise
        emit_observation(
            outcome="success",
            seconds=time.perf_counter() - started,
            code="whisper_completed",
            category="success",
            whisper_model=self._model_name,
            device=self._device,
        )

        raw_text = self._extract_accepted_whisper_text(result)
        return self._post_process_segment_text(raw_text)

    def _extract_accepted_whisper_text(self, result: Mapping[str, object]) -> str:
        """Use Whisper segment metadata to drop likely non-speech text."""
        segments = result.get("segments")
        if not isinstance(segments, list) or not segments:
            return self._get_whisper_text(result)

        accepted_texts: list[str] = []
        found_filter_metadata = False
        for segment in segments:
            if not isinstance(segment, dict):
                return self._get_whisper_text(result)

            found_filter_metadata = (
                found_filter_metadata or _has_whisper_filter_metadata(segment)
            )
            if _should_accept_whisper_segment(segment):
                accepted_texts.append(self._get_whisper_text(segment))

        if not found_filter_metadata:
            return self._get_whisper_text(result)

        return " ".join(text for text in accepted_texts if text)

    def _get_whisper_text(self, result: Mapping[str, object]) -> str:
        """Read Whisper text fields without turning None into text."""
        text = result.get("text")
        if not isinstance(text, str):
            return ""

        return text

    def _prepare_audio(self, audio: np.ndarray, sample_rate: int) -> np.ndarray:
        """Resample source audio to Whisper's rate and normalize its amplitude."""
        if sample_rate <= 0:
            raise ValueError("sample_rate must be positive")

        normalized_audio = np.asarray(audio, dtype=np.float32).reshape(-1)
        if normalized_audio.size == 0:
            return normalized_audio

        if sample_rate != WHISPER_SAMPLE_RATE:
            normalized_audio = resample_audio(
                normalized_audio, sample_rate, WHISPER_SAMPLE_RATE
            )

        peak = np.max(np.abs(normalized_audio))
        if peak > 0:
            normalized_audio = normalized_audio / peak

        return normalized_audio

    def _post_process_segment_text(self, text: str) -> str:
        """Clean segment text without forcing document-level punctuation."""
        if not text:
            return ""

        cleaned_text = self._fix_common_issues(text.strip())
        return re.sub(r"(?:\.\.\.)+$", "", cleaned_text).strip()

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
        return self._fix_common_issues(text)

    def _join_segment_texts(self, segment_texts: Sequence[str]) -> str:
        """Remove nonfinal boundary periods while preserving casing and other marks."""
        cleaned_segments = [self._fix_common_issues(text) for text in segment_texts]
        cleaned_segments = [text for text in cleaned_segments if text]
        cleaned_segments = [
            text.rstrip(".") if index < len(cleaned_segments) - 1 else text
            for index, text in enumerate(cleaned_segments)
        ]
        return self._fix_common_issues(" ".join(cleaned_segments))

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
        self._model_name = None
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
