"""Cooperative cancellation between dictation processing stages."""

from threading import Event


class TranscriptionCancelledError(Exception):
    """A deliberately discarded session, rather than a transcription failure."""


def check_cancelled(cancel_event: Event | None) -> None:
    """Stop before another processing stage when cancellation was requested."""
    if cancel_event is not None and cancel_event.is_set():
        raise TranscriptionCancelledError
