"""
Main application module for Murmur.
Orchestrates audio recording, transcription, and clipboard operations.
"""

import sys
import threading
import time
from typing import Optional

from .config import ConfigError, DEFAULT_CONFIG, get_config
from .audio import AudioRecorder, AudioData
from .transcription import Transcriber
from .transcription_live import (
    LiveSegmentMetrics,
    LiveTranscriptionWorker,
    TranscriptAccumulator,
    TranscriptChunk,
)
from .vad import (
    LiveSpeechSegment,
    LiveVADSegmentationWorker,
    VADSettings,
    WebRTCVADSegmenter,
)
from .clipboard import copy_to_clipboard
from .hotkey import HotkeyManager, HotkeyState, is_hotkey_valid
from .notifications import get_notification_manager
from .logger import get_logger
from .tray import TrayManager
from .autostart import set_autostart
from .media_control import get_media_controller


class MurmurApp:
    """
    Main application class that coordinates all components.

    Handles the workflow:
    1. Listen for hotkey
    2. Record audio
    3. Transcribe with Whisper
    4. Copy to clipboard
    5. Notify user
    """

    def __init__(self, preload_model: bool = True):
        """
        Initialize the Murmur application.

        Args:
            preload_model: Whether to preload the Whisper model on startup.
        """
        self.config = get_config()
        self.recorder = AudioRecorder()
        self._set_recording_limit_callback()
        self.transcriber = Transcriber()
        self.segmenter: Optional[WebRTCVADSegmenter] = None
        self.live_segmenter: Optional[LiveVADSegmentationWorker] = None
        self.live_transcription_worker: Optional[LiveTranscriptionWorker] = None
        self.live_transcript_accumulator: Optional[TranscriptAccumulator] = None
        self.hotkey_manager = HotkeyManager()
        self.notifications = get_notification_manager()
        self.logger = get_logger()
        self.tray = TrayManager(on_exit_callback=self.stop)
        self.media_controller = get_media_controller()

        self._running = False
        self._was_media_playing = False
        self._vad_disabled_reason: Optional[str] = None
        self._live_vad_disabled_reason: Optional[str] = None
        self._live_pipeline_degraded = False
        self._live_pipeline_degraded_reason: Optional[str] = None
        self._recording_limit_stop_started = False
        self._recording_limit_stop_lock = threading.Lock()

        if preload_model:
            print("Preloading Whisper model...")
            self.transcriber.load_model()

        if self.config.ollama_enabled and self.config.ollama_preload_model:
            print("Warming Ollama model...")
            self.transcriber.warm_llm_post_processor()

        # Ensure autostart matches config
        if self.config.start_with_windows:
            set_autostart(True)

    def _set_recording_limit_callback(self) -> None:
        """Register app handling for recorder duration-limit events."""
        setter = getattr(self.recorder, "set_recording_limit_callback", None)
        if setter is not None:
            setter(self._on_recording_limit_reached)

    def start(self) -> None:
        """Start the application and begin listening for hotkeys."""
        if self._running:
            return

        self._running = True
        recovery_message = None
        should_retry_registration = False

        # Register hotkey
        success = self.hotkey_manager.register(
            on_start=self._on_recording_start,
            on_stop=self._on_recording_stop,
            on_state_change=self._on_state_change,
        )

        if not success:
            recovery_message, should_retry_registration = (
                self._recover_failed_hotkey_registration()
            )
            if should_retry_registration:
                success = self.hotkey_manager.register(
                    on_start=self._on_recording_start,
                    on_stop=self._on_recording_stop,
                    on_state_change=self._on_state_change,
                )

        if recovery_message:
            print(f"⚠️ {recovery_message}")
            self.notifications.notify("murmur", recovery_message)

        if not success:
            self._running = False
            print("Failed to register hotkey. Exiting.")
            sys.exit(1)

        config_notice = self.config.consume_startup_notice()
        if config_notice:
            print(f"⚠️ {config_notice}")
            self.notifications.notify("murmur", config_notice)

        print(f"\n{'=' * 50}")
        print("murmur - Local Speech-to-Text")
        print(f"{'=' * 50}")
        print(f"Hotkey: {self.config.hotkey}")
        print(f"Model: {self.config.model_name}")
        print(f"Device: {self.transcriber.get_device_info()['device']}")
        logging_status = "enabled" if self.logger.is_enabled() else "disabled"
        print(f"Training data logging: {logging_status}")
        print(f"Training data path: {self.logger.get_log_directory()}")
        if self.logger.is_enabled():
            print(
                "Privacy notice: raw WAV audio and transcription text will be stored locally until you delete them from Settings."
            )
            print(
                f"Logged entries: {self.logger.get_entry_count()} ({self.logger.get_total_duration():.1f}s audio)"
            )
        print(f"{'=' * 50}\n")

        self.notifications.notify("murmur", "Ready! Press hotkey to start recording.")

    def _recover_failed_hotkey_registration(self) -> tuple[Optional[str], bool]:
        """Recover from invalid or unregistrable configured hotkeys."""
        configured_hotkey = self.config.hotkey
        fallback_hotkey = DEFAULT_CONFIG["hotkey"]
        last_error = None
        if hasattr(self.hotkey_manager, "get_last_registration_error"):
            last_error = self.hotkey_manager.get_last_registration_error()
        error_suffix = f" ({last_error})" if last_error else ""

        if configured_hotkey != fallback_hotkey and not is_hotkey_valid(
            configured_hotkey
        ):
            try:
                self.config.set("hotkey", fallback_hotkey)
            except ConfigError as exc:
                print(f"Failed to reset invalid hotkey '{configured_hotkey}': {exc}")
                return None, False

            return (
                f"Configured hotkey '{configured_hotkey}' was invalid and has been reset "
                f"to '{fallback_hotkey}'.",
                True,
            )

        if configured_hotkey == fallback_hotkey:
            return (
                f"Default hotkey '{fallback_hotkey}' could not be registered{error_suffix}. "
                "Check permissions or whether another application is using it.",
                False,
            )

        try:
            self.config.set("hotkey", fallback_hotkey)
        except ConfigError as exc:
            print(
                f"Failed to reset unregistrable hotkey '{configured_hotkey}' to "
                f"'{fallback_hotkey}': {exc}"
            )
            return (
                f"Configured hotkey '{configured_hotkey}' could not be registered"
                f"{error_suffix}.",
                False,
            )

        return (
            f"Configured hotkey '{configured_hotkey}' could not be registered"
            f"{error_suffix} and has been reset to '{fallback_hotkey}'.",
            True,
        )

    def stop(self) -> None:
        """Stop the application."""
        if not self._running:
            return

        self._running = False
        self.hotkey_manager.unregister()
        self.tray.stop()

        if self.recorder.is_recording():
            self.recorder.stop_recording()

        print("\nmurmur stopped.")

    def _on_recording_start(self) -> None:
        """Handle recording start."""
        print("🎤 Recording started...")
        self._live_pipeline_degraded = False
        self._live_pipeline_degraded_reason = None
        self._recording_limit_stop_started = False
        self.notifications.notify_recording_started()
        self.tray.set_status("Recording...")

        # Check and pause media if playing (when enabled)
        if self.config.pause_media_while_recording:
            self._was_media_playing = self.media_controller.is_media_playing()
            if self._was_media_playing:
                if not self.media_controller.pause():
                    print("⚠️ Failed to pause media playback")
                    self.notifications.notify(
                        "Murmur", "Could not pause media playback"
                    )

        try:
            self._start_live_transcription()
            self._start_live_segmentation()
            self.recorder.start_recording()
        except Exception as e:
            self._stop_live_transcription()
            self._stop_live_segmentation()
            print(f"Error starting recording: {e}")
            self.notifications.notify_error(f"Recording failed: {e}")
            self.tray.set_status("Error")
            self.hotkey_manager.set_idle()
            # Resume media if we paused it but recording failed
            if self._was_media_playing:
                self.media_controller.play()
                self._was_media_playing = False

    def _on_recording_stop(self) -> None:
        """Handle recording stop via hotkey."""
        print("⏹️ Recording stopped.")
        self.tray.set_status("Finalizing...")
        finalization_started_at = time.perf_counter()

        audio_data = self.recorder.stop_recording()
        self._stop_live_segmentation()
        self._stop_live_transcription()

        # Resume media if it was playing before recording
        if self._was_media_playing:
            if not self.media_controller.play():
                print("⚠️ Failed to resume media playback")
                self.notifications.notify("Murmur", "Could not resume media playback")
            self._was_media_playing = False

        if audio_data is not None:
            self._finalize_recording(
                audio_data,
                finalization_started_at=finalization_started_at,
            )
        else:
            print("⚠️ No audio data captured.")
            self.tray.set_status("Ready")
            self.hotkey_manager.set_idle()

    def _on_recording_limit_reached(self, duration_seconds: float) -> None:
        """Handle the recorder reaching max duration from the audio callback."""
        with self._recording_limit_stop_lock:
            if self._recording_limit_stop_started:
                return
            self._recording_limit_stop_started = True

        thread = threading.Thread(
            target=self._handle_recording_limit_reached,
            args=(duration_seconds,),
            daemon=True,
        )
        thread.start()

    def _handle_recording_limit_reached(self, duration_seconds: float) -> None:
        """Notify the user and finalize a recording stopped by the duration cap."""
        print(
            "Maximum recording duration reached "
            f"({duration_seconds:g}s). Finalizing recording."
        )
        self.notifications.notify_recording_limit_reached(duration_seconds)
        self.hotkey_manager.set_processing()
        self._on_recording_stop()

    def _finalize_recording(
        self,
        audio_data: AudioData,
        *,
        finalization_started_at: float,
    ) -> None:
        """Finalize stop-time output from the live transcript, or fall back offline."""
        if self._live_pipeline_degraded:
            reason = (
                self._live_pipeline_degraded_reason or "Live transcription degraded"
            )
            print(f"⚠️ {reason}. Recomputing final transcript from the full recording.")
            self._process_audio(
                audio_data,
                finalization_started_at=finalization_started_at,
            )
            return

        live_text = self._get_live_transcript_text()
        if not live_text:
            self._process_audio(
                audio_data,
                finalization_started_at=finalization_started_at,
            )
            return

        self.hotkey_manager.set_processing()
        try:
            self._complete_transcription(
                audio_data,
                live_text,
                finalization_started_at=finalization_started_at,
                live_segment_metrics=self._get_live_segment_metrics(),
            )
        finally:
            self.tray.set_status("Ready")
            self.hotkey_manager.set_idle()

    def _process_audio(
        self,
        audio_data: AudioData,
        *,
        finalization_started_at: float,
    ) -> None:
        """Process recorded audio through transcription."""
        self.hotkey_manager.set_processing()

        print(f"📝 Processing {audio_data.duration:.1f}s of audio...")

        try:
            text = self._transcribe_audio(audio_data)
            self._complete_transcription(
                audio_data,
                text,
                finalization_started_at=finalization_started_at,
                live_segment_metrics=LiveSegmentMetrics.empty(),
            )

        except Exception:
            print("Transcription failed.")
            self.notifications.notify_error("Transcription failed.")

        finally:
            self.tray.set_status("Ready")
            self.hotkey_manager.set_idle()

    def _transcribe_audio(self, audio_data: AudioData) -> str:
        """Transcribe audio through VAD segments with a full-clip fallback."""
        segmenter = self._get_segmenter(audio_data.sample_rate)
        if segmenter is None:
            reason = self._vad_disabled_reason or "VAD unavailable"
            print(f"⚠️ {reason}. Falling back to full clip.")
            return self.transcriber.transcribe(audio_data)

        try:
            segments = segmenter.segment_audio(audio_data.audio)
        except Exception as exc:
            print(f"⚠️ VAD segmentation failed: {exc}. Falling back to full clip.")
            return self.transcriber.transcribe(audio_data)

        if not segments:
            print("⚠️ No VAD speech segments detected. Falling back to full clip.")
            return self.transcriber.transcribe(audio_data)

        print(f"Detected {len(segments)} speech segment(s).")
        for index, segment in enumerate(segments):
            print(f"  Segment {index}: {segment.duration:.2f}s")

        return self.transcriber.transcribe_segments(segments, debug=True).text

    def _get_live_transcript_text(self) -> str:
        """Return the finalized live transcript text accumulated during recording."""
        if self.live_transcript_accumulator is None:
            return ""

        chunks = self.live_transcript_accumulator.ordered_chunks()
        if not chunks:
            return ""

        return self.transcriber.finalize_segment_texts([chunk.text for chunk in chunks])

    def _get_live_segment_metrics(self) -> LiveSegmentMetrics:
        """Return live segment metrics for chunks that contributed text."""
        if self.live_transcript_accumulator is None:
            return LiveSegmentMetrics.empty()

        return self.live_transcript_accumulator.get_metrics()

    def _complete_transcription(
        self,
        audio_data: AudioData,
        text: str,
        *,
        finalization_started_at: float,
        live_segment_metrics: LiveSegmentMetrics,
    ) -> None:
        """Finish clipboard, notification, and logging for a completed transcript."""
        if not text:
            print("⚠️ No speech detected.")
            self.notifications.notify("Murmur", "No speech detected.")
            return

        copied = copy_to_clipboard(text)
        runtime = time.perf_counter() - finalization_started_at
        log_entry = self.logger.log(audio_data, text, runtime, live_segment_metrics)

        if copied:
            print(f"Finalized in {runtime:.1f}s; copied to clipboard.")
            self._print_live_segment_metrics(live_segment_metrics)
            self.notifications.notify_transcription_copied()
            return

        print(f"Finalized in {runtime:.1f}s; failed to copy to clipboard.")
        self._print_live_segment_metrics(live_segment_metrics)
        if log_entry is not None:
            self.notifications.notify_clipboard_failure_retry_with_training_data()
        else:
            self.notifications.notify_clipboard_failure_retry()

    def _print_live_segment_metrics(
        self, live_segment_metrics: LiveSegmentMetrics
    ) -> None:
        """Emit content-safe live segment metric summary to stdout."""
        avg_latency = self._format_optional_seconds(
            live_segment_metrics.latency_avg_seconds
        )
        max_latency = self._format_optional_seconds(
            live_segment_metrics.latency_max_seconds
        )
        print(
            "Live segments: "
            f"count={live_segment_metrics.segment_count} "
            f"avg_latency={avg_latency} "
            f"max_latency={max_latency}"
        )

    def _format_optional_seconds(self, value: Optional[float]) -> str:
        """Format optional second values for content-safe stdout metrics."""
        if value is None:
            return "n/a"

        return f"{value:.2f}s"

    def _get_segmenter(self, sample_rate: int) -> Optional[WebRTCVADSegmenter]:
        """Return a cached segmenter, or disable VAD if initialization fails."""
        if self._vad_disabled_reason is not None:
            return None

        if self.segmenter is not None and self.segmenter.sample_rate == sample_rate:
            return self.segmenter

        try:
            self.segmenter = self._build_segmenter(sample_rate)
        except Exception as exc:
            self.segmenter = None
            self._vad_disabled_reason = f"VAD unavailable: {exc}"
            return None

        return self.segmenter

    def _build_segmenter(self, sample_rate: int) -> WebRTCVADSegmenter:
        """Create a VAD segmenter using the current application config."""
        return WebRTCVADSegmenter(settings=self._build_vad_settings(sample_rate))

    def _start_live_segmentation(self) -> None:
        """Start live VAD segmentation if the runtime supports it."""
        self._stop_live_segmentation()

        if self._live_vad_disabled_reason is not None:
            print(f"⚠️ {self._live_vad_disabled_reason}. Live segmentation disabled.")
            return

        try:
            worker = self._build_live_segmenter(self.recorder.sample_rate)
        except Exception as exc:
            self._live_vad_disabled_reason = f"Live VAD unavailable: {exc}"
            print(f"⚠️ {self._live_vad_disabled_reason}")
            self.recorder.set_block_callback(None)
            self.recorder.set_block_callback_error_handler(None)
            return

        worker.start()
        self.live_segmenter = worker
        self.recorder.set_block_callback_error_handler(
            self._on_live_block_callback_error
        )
        self.recorder.set_block_callback(worker.submit_audio_block)

    def _stop_live_segmentation(self) -> None:
        """Detach and stop the live segmentation worker."""
        self.recorder.set_block_callback(None)
        self.recorder.set_block_callback_error_handler(None)

        if self.live_segmenter is None:
            return

        self.live_segmenter.stop()
        self.live_segmenter = None

    def _start_live_transcription(self) -> None:
        """Start the serial live transcription worker and transcript accumulator."""
        self._stop_live_transcription()

        self.live_transcript_accumulator = TranscriptAccumulator()
        worker = LiveTranscriptionWorker(
            transcriber=self.transcriber,
            accumulator=self.live_transcript_accumulator,
            on_segment_queued=self._on_live_segment_queued,
            on_segment_failed=self._on_live_segment_failed,
            on_segment_transcribed=self._on_live_segment_transcribed,
            on_chunk_appended=self._on_live_chunk_appended,
            on_worker_degraded=self._on_live_pipeline_degraded,
        )
        worker.start()
        self.live_transcription_worker = worker

    def _stop_live_transcription(self) -> None:
        """Stop the background live transcription worker."""
        if self.live_transcription_worker is None:
            return

        worker = self.live_transcription_worker
        if worker.is_degraded():
            self._on_live_pipeline_degraded(
                worker.get_last_error() or "Live transcription degraded"
            )

        worker.stop()
        self.live_transcription_worker = None

    def _build_live_segmenter(self, sample_rate: int) -> LiveVADSegmentationWorker:
        """Create the live VAD worker using the current application config."""
        return LiveVADSegmentationWorker(
            settings=self._build_vad_settings(sample_rate),
            on_segment=self._on_live_segment_ready,
            on_worker_degraded=self._on_live_pipeline_degraded,
        )

    def _build_vad_settings(self, sample_rate: int) -> VADSettings:
        """Return the shared VAD runtime settings for this sample rate."""
        return VADSettings.from_app_config(self.config, sample_rate=sample_rate)

    def _on_live_segment_ready(self, segment: LiveSpeechSegment) -> None:
        """Emit debug logging and queue sealed live segments for transcription."""
        print(
            "Live segment sealed: "
            f"id={segment.segment_id} "
            f"start={segment.start_sample} "
            f"end={segment.end_sample} "
            f"duration={segment.duration:.2f}s"
        )

        if self.live_transcription_worker is not None:
            self.live_transcription_worker.submit_segment(segment)

    def _on_live_segment_queued(self, segment: LiveSpeechSegment) -> None:
        """Emit debug logging when a live segment is queued for transcription."""
        print(
            "Live segment queued: "
            f"id={segment.segment_id} "
            f"duration={segment.duration:.2f}s"
        )

    def _on_live_segment_transcribed(self, chunk: TranscriptChunk) -> None:
        """Emit debug logging when a live segment finishes transcribing."""
        print(
            "Live segment transcribed: "
            f"id={chunk.segment_id} "
            f"latency={chunk.latency_seconds:.2f}s "
            f"text_length={len(chunk.text)}"
        )

    def _on_live_chunk_appended(
        self, chunk: TranscriptChunk, current_text: str
    ) -> None:
        """Emit debug logging when transcript text is appended in segment order."""
        print(
            "Live transcript appended: "
            f"id={chunk.segment_id} "
            f"current_text_length={len(current_text)}"
        )

    def _on_live_segment_failed(
        self,
        segment: LiveSpeechSegment,
        exc: Exception,
        attempt_count: int,
    ) -> None:
        """Emit debug logging when live transcription permanently fails for a segment."""
        print(f"Live segment failed: id={segment.segment_id} attempts={attempt_count}")

    def _on_live_pipeline_degraded(self, message: str) -> None:
        """Mark the live pipeline degraded and record it once per recording."""
        if self._live_pipeline_degraded:
            return

        self._live_pipeline_degraded = True
        self._live_pipeline_degraded_reason = message
        print(f"⚠️ {message}")

    def _on_live_block_callback_error(self, exc: Exception) -> None:
        """Handle a recorder block callback failure during live segmentation."""
        print("Live audio callback failed.")
        self._on_live_pipeline_degraded("Live audio callback failed.")

    def _on_state_change(self, state: HotkeyState) -> None:
        """Handle state changes."""
        pass  # State logging is handled in other methods


def main():
    """Main entry point for the application."""
    print("Starting murmur...")

    try:
        app = MurmurApp(preload_model=True)
        app.start()

        # Run the tray icon (this is the main loop)
        app.tray.run()

        # When tray exits, stop the app
        app.stop()

    except KeyboardInterrupt:
        print("\nInterrupted by user.")
    except Exception as e:
        print(f"Fatal error: {e}")
        sys.exit(1)


if __name__ == "__main__":
    main()
