from __future__ import annotations

from collections import deque
from queue import Empty, Queue
import threading
from typing import Callable, Deque, List, Optional, Sequence

import numpy as np

from .vad_audio import float32_to_pcm16
from .vad_config import SUPPORTED_VAD_SAMPLE_RATES, VADSettings
from .vad_segmenter import _create_vad
from .vad_types import AudioFrame, LiveSpeechSegment


class LiveVADSegmentationWorker:
    """Consume recorder blocks in the background and emit sealed speech segments."""

    def __init__(
        self,
        settings: Optional[VADSettings] = None,
        on_segment: Optional[Callable[[LiveSpeechSegment], None]] = None,
        on_worker_degraded: Optional[Callable[[str], None]] = None,
        vad: Optional[object] = None,
        queue_timeout_seconds: float = 0.1,
        **kwargs,
    ):
        if settings is not None and kwargs:
            raise ValueError(
                "Provide either settings or keyword VAD parameters, not both"
            )

        self.settings = settings or VADSettings(**kwargs)
        if self.settings.sample_rate not in SUPPORTED_VAD_SAMPLE_RATES:
            raise ValueError(
                "LiveVADSegmentationWorker requires a WebRTC-supported sample rate"
            )

        self.sample_rate = self.settings.sample_rate
        self.frame_duration_ms = self.settings.frame_duration_ms
        self.frame_samples = self.settings.frame_samples
        self.start_padding_frames = self.settings.start_padding_frames
        self.end_padding_frames = self.settings.end_padding_frames
        self.silence_close_frames = self.settings.silence_close_frames
        self.merge_gap_samples = self.settings.merge_gap_samples
        self.min_segment_samples = int(
            self.sample_rate * self.settings.min_segment_duration_ms / 1000
        )
        self.on_segment = on_segment
        self.queue_timeout_seconds = queue_timeout_seconds
        self.on_worker_degraded = on_worker_degraded
        self._vad = vad or _create_vad(self.settings.aggressiveness)
        self._queue: Queue[Optional[np.ndarray]] = Queue()
        self._thread: Optional[threading.Thread] = None
        self._running = False
        self._state_lock = threading.RLock()
        self._degraded = False
        self._last_error: Optional[str] = None
        self._degraded_notified = False
        self._next_segment_id = 0
        self._processed_samples = 0
        self._pending_block = np.array([], dtype=np.float32)
        self._pre_speech_frames: Deque[AudioFrame] = deque(
            maxlen=self.start_padding_frames
        )
        self._current_frames: List[AudioFrame] = []
        self._pending_segment: Optional[LiveSpeechSegment] = None
        self._pending_gap_frames: List[AudioFrame] = []
        self._last_speech_index: Optional[int] = None
        self._silence_run_frames = 0

    def start(self) -> None:
        """Start the background segmentation worker."""
        if self._running or self.is_degraded():
            return

        self._running = True
        self._thread = threading.Thread(
            target=self._run,
            name="live-vad-segmentation",
            daemon=True,
        )
        self._thread.start()

    def submit_audio_block(self, audio_block: np.ndarray, *, copy: bool = True) -> None:
        """Queue a recorder block for background VAD processing."""
        if not self._running or self.is_degraded():
            return

        prepared_block = np.asarray(audio_block, dtype=np.float32).reshape(-1)
        if copy:
            prepared_block = prepared_block.copy()
        elif not prepared_block.flags.c_contiguous:
            prepared_block = np.ascontiguousarray(prepared_block)

        self._queue.put(prepared_block)

    def stop(self) -> None:
        """Stop the worker and flush any pending speech segment."""
        if not self._running:
            self._join_worker_thread()
            return

        self._running = False
        self._queue.put(None)
        self._join_worker_thread()

    def is_degraded(self) -> bool:
        """Return whether live VAD has degraded for this worker."""
        with self._state_lock:
            return self._degraded

    def get_last_error(self) -> Optional[str]:
        """Return the last content-safe live VAD degradation message."""
        with self._state_lock:
            return self._last_error

    def _run(self) -> None:
        while True:
            try:
                audio_block = self._queue.get(timeout=self.queue_timeout_seconds)
            except Empty:
                if self._running:
                    continue
                break

            if audio_block is None:
                try:
                    self._flush_pending_audio()
                except Exception as exc:
                    self._disable_after_worker_exception(exc)
                break

            try:
                self._process_block(audio_block)
            except Exception as exc:
                self._disable_after_worker_exception(exc)
                break

    def _flush_pending_audio(self) -> None:
        if self._pending_block.size > 0:
            frame_audio = np.zeros(self.frame_samples, dtype=np.float32)
            frame_audio[: self._pending_block.size] = self._pending_block
            frame = AudioFrame(
                start_sample=self._processed_samples,
                end_sample=self._processed_samples + self._pending_block.size,
                audio=frame_audio[: self._pending_block.size].copy(),
                pcm16=float32_to_pcm16(frame_audio),
            )
            self._processed_samples += self._pending_block.size
            self._pending_block = np.array([], dtype=np.float32)
            self._process_frame(frame)

        if self._current_frames and self._last_speech_index is not None:
            self._seal_current_segment()

        self._emit_pending_segment()

    def _process_block(self, audio_block: np.ndarray) -> None:
        if audio_block.size == 0:
            return

        if self._pending_block.size == 0:
            buffered_audio = audio_block
        else:
            buffered_audio = np.concatenate((self._pending_block, audio_block))

        frame_count = buffered_audio.size // self.frame_samples
        consumed_samples = frame_count * self.frame_samples
        self._pending_block = buffered_audio[consumed_samples:].copy()

        for frame_index in range(frame_count):
            if self.is_degraded():
                break

            frame_start = frame_index * self.frame_samples
            frame_audio = buffered_audio[
                frame_start : frame_start + self.frame_samples
            ].copy()
            frame = AudioFrame(
                start_sample=self._processed_samples,
                end_sample=self._processed_samples + self.frame_samples,
                audio=frame_audio,
                pcm16=float32_to_pcm16(frame_audio),
            )
            self._processed_samples += self.frame_samples
            self._process_frame(frame)

    def _process_frame(self, frame: AudioFrame) -> None:
        is_speech = self._vad.is_speech(frame.pcm16, self.sample_rate)

        if not self._current_frames:
            if not is_speech:
                self._track_pending_gap_frame(frame)
                if self.start_padding_frames > 0:
                    self._pre_speech_frames.append(frame)
                return

            leading_frames = list(self._pre_speech_frames)
            self._current_frames = [*leading_frames, frame]
            self._last_speech_index = len(self._current_frames) - 1
            self._silence_run_frames = 0
            self._pre_speech_frames.clear()
            return

        self._current_frames.append(frame)
        if is_speech:
            self._last_speech_index = len(self._current_frames) - 1
            self._silence_run_frames = 0
            return

        self._silence_run_frames += 1
        if self._silence_run_frames >= self.silence_close_frames:
            self._seal_current_segment()

    def _seal_current_segment(self) -> None:
        if not self._current_frames or self._last_speech_index is None:
            self._reset_current_segment([])
            return

        keep_frame_count = min(
            len(self._current_frames),
            self._last_speech_index + 1 + self.end_padding_frames,
        )
        frames_to_emit = self._current_frames[:keep_frame_count]
        overflow_frames = self._current_frames[keep_frame_count:]

        start_sample = frames_to_emit[0].start_sample
        end_sample = frames_to_emit[-1].end_sample
        segment_audio = np.concatenate([frame.audio for frame in frames_to_emit])

        if end_sample - start_sample >= self.min_segment_samples:
            segment = LiveSpeechSegment(
                segment_id=self._next_segment_id,
                start_sample=start_sample,
                end_sample=end_sample,
                sample_rate=self.sample_rate,
                audio=segment_audio,
            )
            self._next_segment_id += 1
            self._queue_segment_for_emit(segment)

        self._reset_current_segment(overflow_frames)

    def _queue_segment_for_emit(self, segment: LiveSpeechSegment) -> None:
        if self._pending_segment is None:
            self._pending_segment = segment
            return

        gap_samples = segment.start_sample - self._pending_segment.end_sample
        if gap_samples <= self.merge_gap_samples:
            bridge_audio = self._get_pending_gap_audio(gap_samples)
            self._pending_segment = LiveSpeechSegment(
                segment_id=self._pending_segment.segment_id,
                start_sample=self._pending_segment.start_sample,
                end_sample=max(self._pending_segment.end_sample, segment.end_sample),
                sample_rate=self.sample_rate,
                audio=np.concatenate(
                    (self._pending_segment.audio, bridge_audio, segment.audio)
                ),
            )
            self._pending_gap_frames = []
            return

        self._emit_pending_segment()
        self._pending_segment = segment

    def _emit_pending_segment(self) -> None:
        if self._pending_segment is None:
            return

        if self.on_segment is not None:
            try:
                self.on_segment(self._pending_segment)
            except Exception as exc:
                message = (
                    "Live VAD segment callback failed with "
                    f"{type(exc).__name__}. Live segmentation disabled."
                )
                self._pending_segment = None
                self._pending_gap_frames = []
                self._disable_after_degradation(message)
                return

        self._pending_segment = None
        self._pending_gap_frames = []

    def _track_pending_gap_frame(self, frame: AudioFrame) -> None:
        if self._pending_segment is None:
            return

        self._pending_gap_frames.append(frame)
        gap_samples = frame.end_sample - self._pending_segment.end_sample
        if gap_samples > self.merge_gap_samples:
            self._emit_pending_segment()

    def _get_pending_gap_audio(self, gap_samples: int) -> np.ndarray:
        if gap_samples <= 0 or not self._pending_gap_frames:
            return np.array([], dtype=np.float32)

        collected_audio = np.concatenate(
            [frame.audio for frame in self._pending_gap_frames]
        )
        return collected_audio[:gap_samples].copy()

    def _reset_current_segment(self, overflow_frames: Sequence[AudioFrame]) -> None:
        self._current_frames = []
        self._last_speech_index = None
        self._silence_run_frames = 0

        for frame in overflow_frames:
            self._track_pending_gap_frame(frame)

        self._pre_speech_frames.clear()
        if self.start_padding_frames <= 0:
            return

        for frame in overflow_frames[-self.start_padding_frames :]:
            self._pre_speech_frames.append(frame)

    def _disable_after_degradation(self, message: str) -> None:
        should_notify = self._record_degradation(message)
        self._running = False
        self._clear_pending_audio_state()
        self._discard_queued_audio_blocks()

        if should_notify:
            self._invoke_worker_degraded(message)

    def _disable_after_worker_exception(self, exc: Exception) -> None:
        message = (
            "Live VAD worker failed with "
            f"{type(exc).__name__}. Live segmentation disabled."
        )
        self._disable_after_degradation(message)

    def _record_degradation(self, message: str) -> bool:
        with self._state_lock:
            self._degraded = True
            self._last_error = message
            should_notify = not self._degraded_notified
            if should_notify:
                self._degraded_notified = True

        if should_notify:
            print(f"Warning: {message}")

        return should_notify

    def _invoke_worker_degraded(self, message: str) -> None:
        if self.on_worker_degraded is None:
            return

        try:
            self.on_worker_degraded(message)
        except Exception:
            print("Live VAD degradation callback failed.")

    def _clear_pending_audio_state(self) -> None:
        self._pending_block = np.array([], dtype=np.float32)
        self._pre_speech_frames.clear()
        self._current_frames = []
        self._pending_segment = None
        self._pending_gap_frames = []
        self._last_speech_index = None
        self._silence_run_frames = 0

    def _discard_queued_audio_blocks(self) -> None:
        while True:
            try:
                self._queue.get_nowait()
            except Empty:
                return

    def _join_worker_thread(self) -> None:
        if self._thread is None:
            return

        if self._thread is threading.current_thread():
            return

        self._thread.join()
        self._thread = None
