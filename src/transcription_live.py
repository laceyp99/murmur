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
        on_segment_transcribed: Optional[Callable[[TranscriptChunk], None]] = None,
        on_chunk_appended: Optional[Callable[[TranscriptChunk, str], None]] = None,
        queue_timeout_seconds: float = 0.1,
    ):
        self.transcriber = transcriber
        self.accumulator = accumulator
        self.on_segment_queued = on_segment_queued
        self.on_segment_transcribed = on_segment_transcribed
        self.on_chunk_appended = on_chunk_appended
        self.queue_timeout_seconds = queue_timeout_seconds
        self._queue: Queue[Optional[LiveSegmentLike]] = Queue()
        self._thread: Optional[threading.Thread] = None
        self._running = False

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
        if self.on_segment_queued is not None:
            self.on_segment_queued(segment)

    def stop(self) -> None:
        """Stop the background worker without draining future work."""
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

            start_time = time.time()
            text = self.transcriber.transcribe_segment(segment)
            latency_seconds = time.time() - start_time
            chunk = TranscriptChunk(
                segment_id=segment.segment_id,
                text=text,
                latency_seconds=latency_seconds,
            )

            if self.on_segment_transcribed is not None:
                self.on_segment_transcribed(chunk)

            if self.accumulator.add_chunk(chunk) and self.on_chunk_appended is not None:
                self.on_chunk_appended(chunk, self.accumulator.get_text())