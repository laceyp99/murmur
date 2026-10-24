"""Content-free, synchronous observations for transcription callers.

Wrap calls with ``observe_transcription(callback, stage="whisper_offline")``.
The live worker installs its own context with stage ``whisper_live`` because
ContextVars do not propagate to worker threads. Whisper seconds measure only
model inference per attempt, including failed attempts, excluding model loading
and audio preparation. Live inference work overlaps recording, not elapsed wait.

Events have stage, outcome, seconds, code, category and metadata keys. Metadata
contains only actual whisper_model/device or cleanup_model/cleanup_enabled.
Outcomes are success, failed, skipped or unavailable. Codes distinguish applied
cleanup (including unchanged text) from rejected output and request failures.
Categories are success, error, skip or unavailable; timeout has its own code.
Unknown injected model identities are None. No text, audio, or exceptions enter
events. Callbacks run on the calling thread and should be quick and thread safe.
"""

from collections.abc import Callable, Iterator
from contextlib import contextmanager, suppress
from contextvars import ContextVar
from typing import TypedDict

try:
    from httpx import TimeoutException
except ImportError:  # Ollama and its transport are optional.
    TimeoutException = TimeoutError


class TranscriptionObservation(TypedDict):
    stage: str
    outcome: str
    seconds: float
    code: str
    category: str
    metadata: dict[str, str | bool | None]


ObservationCallback = Callable[[TranscriptionObservation], None]
_observer: ContextVar[ObservationCallback | None] = ContextVar("observer", default=None)
_whisper_stage: ContextVar[str] = ContextVar("whisper_stage", default="whisper_offline")


@contextmanager
def observe_transcription(
    callback: ObservationCallback | None, *, stage: str = "whisper_offline"
) -> Iterator[None]:
    """Install an observer for this call scope, restoring any enclosing scope."""
    observer_token = _observer.set(callback)
    stage_token = _whisper_stage.set(stage)
    try:
        yield
    finally:
        _whisper_stage.reset(stage_token)
        _observer.reset(observer_token)


def failure_category(error: Exception) -> str:
    """Classify known timeout types without reading or retaining messages."""
    return "timeout" if isinstance(error, (TimeoutError, TimeoutException)) else "error"


def emit_observation(
    *,
    stage: str | None = None,
    outcome: str,
    seconds: float,
    code: str,
    category: str,
    whisper_model: str | None = None,
    device: str | None = None,
    cleanup_model: str | None = None,
    cleanup_enabled: bool | None = None,
) -> None:
    """Deliver fixed source-defined fields, containing every callback failure."""
    callback = _observer.get()
    if callback is None:
        return
    metadata: dict[str, str | bool | None] = {}
    if stage is None:
        metadata.update(whisper_model=whisper_model, device=device)
    if cleanup_enabled is not None:
        metadata.update(cleanup_model=cleanup_model, cleanup_enabled=cleanup_enabled)
    event: TranscriptionObservation = {
        "stage": stage if stage is not None else _whisper_stage.get(),
        "outcome": outcome,
        "seconds": seconds,
        "code": code,
        "category": category,
        "metadata": metadata,
    }
    # Statistics must never change dictation, including cancellation behavior.
    with suppress(BaseException):
        callback(event)
