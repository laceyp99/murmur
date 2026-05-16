from __future__ import annotations

from dataclasses import dataclass
from queue import Empty, Queue
import threading
import time
from typing import Callable, Dict, List, Optional, Protocol


class LiveSegmentLike(Protocol):
    """Structural type for live segments accepted by the worker."""

    segment_id: int
    audio: object
    sample_rate: int
    duration: float


class LiveTranscriberLike(Protocol):
    """Structural type for transcribers used by the live worker."""

    def transcribe_segment(self, segment: LiveSegmentLike) -> str:
        ...


@dataclass(frozen=True)
class TranscriptChunk:
    """One completed live transcription chunk."""

    segment_id: int
    text: str
    latency_seconds: float


class TranscriptAccumulator:
    """Store completed transcript chunks and expose them in segment order."""

    def __init__(self):
        self._chunks: Dict[int, TranscriptChunk] = {}
        self._lock = threading.RLock()

    def add_chunk(self, chunk: TranscriptChunk) -> bool:
        """Store a chunk and report whether it added new transcript text."""
        if not chunk.text:
            return False

        with self._lock:
            self._chunks[chunk.segment_id] = chunk
            return True

    def ordered_chunks(self) -> List[TranscriptChunk]:
        """Return chunks in ascending segment order."""
        with self._lock:
            return [self._chunks[key] for key in sorted(self._chunks)]

    def get_text(self) -> str:
        """Return the concatenated live transcript in segment order."""
        return " ".join(chunk.text for chunk in self.ordered_chunks())


class LiveTranscriptionWorker:
    """Serially transcribe sealed live segments in a background worker."""

    def __init__(
        self,
        transcriber: LiveTranscriberLike,
        accumulator: TranscriptAccumulator,
        on_segment_queued: Optional[Callable[[LiveSegmentLike], None]] = None,
        on_segment_failed: Optional[Callable[[LiveSegmentLike, Exception, int], None]] = None,
        on_segment_transcribed: Optional[Callable[[TranscriptChunk], None]] = None,
        on_chunk_appended: Optional[Callable[[TranscriptChunk, str], None]] = None,
        on_worker_degraded: Optional[Callable[[str], None]] = None,
        queue_timeout_seconds: float = 0.1,
        max_segment_retries: int = 1,
    ):
        self.transcriber = transcriber
        self.accumulator = accumulator
        self.on_segment_queued = on_segment_queued
        self.on_segment_failed = on_segment_failed
        self.on_segment_transcribed = on_segment_transcribed
        self.on_chunk_appended = on_chunk_appended
        self.on_worker_degraded = on_worker_degraded
        self.queue_timeout_seconds = queue_timeout_seconds
        self.max_segment_retries = max_segment_retries
        self._queue: Queue[Optional[LiveSegmentLike]] = Queue()
        self._thread: Optional[threading.Thread] = None
        self._running = False
        self._state_lock = threading.RLock()
        self.has_failures = False
        self.failure_count = 0
        self.last_error: Optional[str] = None
        self._degraded_notified = False

    def start(self) -> None:
        """Start the background transcription worker."""
        if self._running:
            return

        self._running = True
        self._thread = threading.Thread(
            target=self._run,
            name="live-transcription",
            daemon=True,
        )
        self._thread.start()

    def submit_segment(self, segment: LiveSegmentLike) -> None:
        """Queue a sealed segment for serial transcription."""
        if not self._running:
            return

        self._queue.put(segment)
        self._invoke_callback("on_segment_queued", segment)

    def stop(self) -> None:
        """Stop the background worker after finishing already queued segments.

        New segments submitted after shutdown begins are ignored, but any
        segments already queued before the shutdown sentinel may still be
        transcribed before the worker thread exits.
        """
        if not self._running:
            return

        self._running = False
        self._queue.put(None)
        if self._thread is not None:
            self._thread.join()
            self._thread = None

    def _run(self) -> None:
        while True:
            try:
                segment = self._queue.get(timeout=self.queue_timeout_seconds)
            except Empty:
                if self._running:
                    continue
                break

            if segment is None:
                break

            chunk = self._transcribe_segment_with_retry(segment)
            if chunk is None:
                continue

            self._invoke_callback("on_segment_transcribed", chunk)

            if self.accumulator.add_chunk(chunk):
                self._invoke_callback("on_chunk_appended", chunk, self.accumulator.get_text())

    def _transcribe_segment_with_retry(
        self,
        segment: LiveSegmentLike,
    ) -> Optional[TranscriptChunk]:
        last_error: Optional[Exception] = None

        for attempt_count in range(1, self.max_segment_retries + 2):
            start_time = time.time()
            try:
                text = self.transcriber.transcribe_segment(segment)
            except Exception as exc:
                last_error = exc
                if attempt_count <= self.max_segment_retries:
                    continue

                message = (
                    "Live transcription failed for "
                    f"segment {segment.segment_id} after {attempt_count} attempts: {exc}"
                )
                self._record_failure(message)
                self._invoke_callback("on_segment_failed", segment, exc, attempt_count)
                return None

            latency_seconds = time.time() - start_time
            return TranscriptChunk(
                segment_id=segment.segment_id,
                text=text,
                latency_seconds=latency_seconds,
            )

        if last_error is not None:
            self._record_failure(
                "Live transcription failed for "
                f"segment {segment.segment_id}: {last_error}"
            )
        return None

    def _record_failure(self, message: str) -> None:
        with self._state_lock:
            self.has_failures = True
            self.failure_count += 1
            self.last_error = message
            should_notify = not self._degraded_notified
            if should_notify:
                self._degraded_notified = True

        print(f"⚠️ {message}")
        if should_notify:
            self._invoke_callback("on_worker_degraded", message)

    def _invoke_callback(self, attr_name: str, *args: object) -> None:
        callback = getattr(self, attr_name)
        if callback is None:
            return

        try:
            callback(*args)
        except Exception as exc:
            print(f"⚠️ Live transcription callback '{attr_name}' failed: {exc}")
            setattr(self, attr_name, None)

    def is_degraded(self) -> bool:
        with self._state_lock:
            return self.has_failures

    def get_last_error(self) -> Optional[str]:
        with self._state_lock:
            return self.last_error