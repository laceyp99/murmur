"""
Main application module for murmur.
Orchestrates audio recording, transcription, and clipboard operations.
"""

import contextlib
import sys
import threading
import time

from .audio import AudioData, AudioRecorder
from .autostart import set_autostart
from .build_info import capture_build_info_in_background
from .cancellation import TranscriptionCancelledError, check_cancelled
from .clipboard import copy_to_clipboard
from .config import DEFAULT_CONFIG, ConfigError, get_config
from .hotkey import HotkeyManager, HotkeyState, RecordingRequest, is_hotkey_valid
from .logger import get_logger
from .media_control import get_media_controller
from .notifications import get_notification_manager
from .observations import observe_transcription
from .recording_overlay import (
    CANCELLED,
    CANCELLED_DONE,
    ERROR,
    HIDDEN,
    PROCESSING,
    RECORDING,
    SUCCESS,
)
from .transcription import Transcriber
from .transcription_live import (
    LiveSegmentMetrics,
    LiveTranscriptionWorker,
    TranscriptAccumulator,
    TranscriptChunk,
)
from .tray import TrayManager
from .usage_stats import get_usage_stats
from .vad import (
    LiveSpeechSegment,
    LiveVADSegmentationWorker,
    VADSettings,
    WebRTCVADSegmenter,
)
from .windows_identity import configure_windows_app_identity

# Fixed, content-safe overlay messages; transcript text never reaches the overlay.
OVERLAY_START_FAILED = "Couldn't start recording"
OVERLAY_NO_AUDIO = "No audio captured"
OVERLAY_NO_SPEECH = "No speech detected"
OVERLAY_TRANSCRIPTION_FAILED = "Transcription failed"
OVERLAY_CLIPBOARD_FAILED = "Couldn't copy to clipboard"
OVERLAY_AUDIO_INCOMPLETE = "Copied; some audio lost"


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

    # Session number sent with overlay requests so late events cannot replace
    # a newer session's state.
    _overlay_session = 0
    _overlay_shown = False

    def __init__(self, preload_model: bool = True):
        """
        Initialize the murmur application.

        Args:
            preload_model: Whether to preload the Whisper model on startup.
        """
        configure_windows_app_identity()
        self.config = get_config()
        self.recorder = AudioRecorder()
        self._set_recording_limit_callback()
        self.transcriber = Transcriber()
        self.segmenter: WebRTCVADSegmenter | None = None
        self.live_segmenter: LiveVADSegmentationWorker | None = None
        self.live_transcription_worker: LiveTranscriptionWorker | None = None
        self.live_transcript_accumulator: TranscriptAccumulator | None = None
        self.hotkey_manager = HotkeyManager()
        self.notifications = get_notification_manager()
        self.logger = get_logger()
        try:
            self._usage_stats = get_usage_stats()
        except Exception:
            self._usage_stats = None
            print("Warning: Usage statistics are unavailable.")
        self._stats_session = None
        self.tray = TrayManager(
            on_exit_callback=self.stop,
            on_cancel_callback=self._on_recording_cancel,
            get_cancel_session_callback=self._get_cancel_session,
        )
        self.media_controller = get_media_controller()
        self._media_pause_thread: threading.Thread | None = None

        self._running = False
        self._was_media_playing = False
        self._vad_disabled_reason: str | None = None
        self._live_vad_disabled_reason: str | None = None
        self._live_pipeline_degraded = False
        self._live_pipeline_degraded_reason: str | None = None
        self._recording_limit_stop_started = False
        self._recording_limit_stop_lock = threading.Lock()
        self._recording_stop_started = False
        self._recording_stop_lock = threading.RLock()
        self._lifecycle_lock = threading.RLock()
        self._cancel_event = threading.Event()
        self._session_active = False
        self._delivery_started = False
        self._shutdown_event = threading.Event()
        self._current_request: RecordingRequest | None = None

        from .settings_gui import set_overlay_cancel_callback

        set_overlay_cancel_callback(self._on_recording_cancel)

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

    def _stats(self, method, *args, **kwargs):
        """Keep optional metadata collection outside dictation's failure path."""
        collector = getattr(self, "_stats_session", None)
        if collector is not None:
            try:
                getattr(collector, method)(*args, **kwargs)
            except Exception:
                print("Warning: Could not record usage statistics.")

    def _stats_observer(self):
        """Capture this collector so late observations cannot target a new session."""
        collector = getattr(self, "_stats_session", None)
        if collector is None:
            return None

        def observe(event):
            stage = {
                "whisper_live": "live_whisper",
                "whisper_offline": "offline_whisper",
                "llm_cleanup": "ollama_cleanup",
            }.get(event["stage"], event["stage"])
            try:
                metadata = dict(event["metadata"])
                if "device" in metadata:
                    metadata["whisper_device"] = metadata.pop("device")
                collector.update(**metadata)
                outcome = (
                    "failed" if event["outcome"] == "unavailable" else event["outcome"]
                )
                collector.stage(
                    stage,
                    outcome,
                    event["seconds"],
                )
                # A disabled feature is not an issue; the stage outcome says "off".
                if outcome not in {"success", "cancelled", "off"}:
                    category = "skip" if outcome == "skipped" else event["category"]
                    collector.issue(
                        stage,
                        event["code"],
                        category=category
                        if category in {"error", "unavailable", "routing", "skip"}
                        else "error",
                        recovered=stage == "ollama_cleanup" and outcome == "failed",
                    )
            except Exception:
                print("Warning: Could not record stage statistics.")

        return observe

    def start(self) -> None:
        """Start the application and begin listening for hotkeys."""
        if self._running or self._shutdown_event.is_set():
            return

        self._running = True
        recovery_message = None
        should_retry_registration = False

        # Optional warmup finishes before hotkeys can begin a session. Failure
        # leaves ordinary on-demand recording available.
        try:
            self.recorder.prepare()
        except Exception:
            print("Warning: Could not keep microphone ready; using on-demand capture")

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
            print(f"Warning: {recovery_message}")
            self.notifications.notify("murmur", recovery_message)

        if not success:
            self._running = False
            print("Failed to register hotkey. Exiting.")
            sys.exit(1)

        config_notice = self.config.get_startup_notice()
        if config_notice:
            print(f"Warning: {config_notice}")

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
        # Start the shared UI thread now so the first recording shows promptly.
        self._show_overlay(HIDDEN)

    def _recover_failed_hotkey_registration(self) -> tuple[str | None, bool]:
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
        if self._shutdown_event.is_set():
            return
        self._shutdown_event.set()
        self._close_overlay()
        self.hotkey_manager.stop_requests()
        with self._recording_stop_lock:
            if self._session_active and not self._delivery_started:
                self._cancel_event.set()
                self._cancel_live_workers()
                should_stop = not self._recording_stop_started
                self._recording_stop_started = True
            else:
                should_stop = False
        self._running = False
        with contextlib.suppress(Exception):
            self.hotkey_manager.unregister()
        with contextlib.suppress(Exception):
            self.tray.stop()
        # Release even an idle persistent stream before model work can delay
        # shutdown. close() also prevents an in-flight warmup from reopening it.
        try:
            self.recorder.close()
        except Exception:
            print("Warning: Failed to close microphone during shutdown")
        if should_stop:
            self._finish_recording_stop()
        else:
            # Barrier: wait for in-flight startup or finalization before exiting.
            with self._lifecycle_lock:
                try:
                    if self.recorder.is_recording():
                        self.recorder.stop_recording()
                except Exception:
                    print("Warning: Failed to stop recording during shutdown")
                self._resume_owned_media()

        print("\nmurmur stopped.")
        service = getattr(self, "_usage_stats", None)
        if service is not None:
            with contextlib.suppress(Exception):
                service.close()

    def _on_recording_start(self, request: RecordingRequest) -> None:
        """Handle recording start."""
        with self._lifecycle_lock:
            self._start_recording_session(request)

    def _startup_abandoned(self, request: RecordingRequest) -> bool:
        """Whether Exit or a stop cancelled this attempt before capture committed."""
        # Superseding an uncommitted request also cancels it, so no is_active check.
        return self._shutdown_event.is_set() or request.cancelled.is_set()

    def _start_recording_session(self, request: RecordingRequest) -> None:
        """Serialize startup with teardown so a cancelled start cannot leak capture."""
        with self._recording_stop_lock:
            if self._startup_abandoned(request):
                return
            self._current_request = request
            self._recording_stop_started = False
            self._overlay_session += 1
            session = self._overlay_session
            self._cancel_event = request.cancelled
            self._session_active = True
            self._delivery_started = False
        self._stats_session = None
        service = getattr(self, "_usage_stats", None)
        if service is not None:
            try:
                self._stats_session = service.start_session()
            except Exception:
                print("Warning: Could not begin usage statistics.")
        self._stats_capture_error = False
        preparation_started = time.perf_counter()
        self._live_pipeline_degraded = False
        self._live_pipeline_degraded_reason = None
        self._recording_limit_stop_started = False
        self.recorder.set_capture_error_callback(
            lambda message: self._on_capture_error(message, session=session)
        )
        self.recorder.set_recording_limit_callback(
            lambda duration: self._on_recording_limit_reached(duration, session=session)
        )
        self.tray.set_status("Preparing...")

        self._was_media_playing = False
        try:
            if self._startup_abandoned(request):
                self._abort_startup()
                return
            # Capture first. Optional preparation must not delay the microphone.
            self.recorder.start_recording()
            self._stats(
                "stage",
                "preparation",
                "success",
                time.perf_counter() - preparation_started,
            )
            # The atomic commit is the boundary; a stop that won it gets no capture.
            if self._shutdown_event.is_set() or not request.commit_capture():
                with contextlib.suppress(Exception):
                    self.recorder.stop_recording()
                self._abort_startup()
                return
        except Exception as e:
            self._stats("issue", "preparation", "capture_start_failed")
            self._stats(
                "finish", "cancelled" if self._startup_abandoned(request) else "failed"
            )
            self._resume_owned_media()
            self._stop_live_transcription()
            self._stop_live_segmentation()
            if not self._startup_abandoned(request):
                print(f"Error starting recording: {e}")
                self.notifications.notify_error(f"Recording failed: {e}")
                self.tray.set_status("Error")
                self._show_overlay(ERROR, OVERLAY_START_FAILED)
            with self._recording_stop_lock:
                if not self._cancel_event.is_set():
                    self._session_active = False
                    if self._current_request is request:
                        self._current_request = None
                    self.hotkey_manager.set_idle()
            return

        # Show capture before worker setup or media calls. A concurrent cancel
        # must not be replaced by a late recording cue.
        with self._recording_stop_lock:
            if self._startup_abandoned(request) or self._recording_stop_started:
                return
            print("Recording started...")
            self.notifications.notify_recording_started()
            self.tray.set_status("Recording...")
            self._show_overlay(RECORDING)

        try:
            self._start_live_transcription()
            self._start_live_segmentation()
        except Exception:
            # The full recording is already safe. Worker setup is optional.
            self._on_live_pipeline_degraded("Live worker startup failed")
            self._cancel_live_workers()
            self.recorder.set_block_callback(None)
            self.recorder.set_block_callback_error_handler(None)

        if self.config.pause_media_while_recording and self._media_pause_allowed(
            request
        ):
            thread = threading.Thread(
                target=self._pause_media_for_session, args=(request,), daemon=True
            )
            # Track before starting so teardown always joins a running pause.
            self._media_pause_thread = thread
            try:
                thread.start()
            except RuntimeError:
                self._media_pause_thread = None
                print("Warning: Could not start media pause worker")

    def _media_pause_allowed(self, request: RecordingRequest) -> bool:
        """Do not start a media pause after this recording has begun stopping."""
        with self._recording_stop_lock:
            return (
                self._current_request is request
                and self._session_active
                and not self._recording_stop_started
                and not self._startup_abandoned(request)
            )

    def _pause_media_for_session(self, request: RecordingRequest) -> None:
        """Pause in the background; teardown joins before restoring playback."""
        try:
            if not self._media_pause_allowed(request):
                return
            playing = self.media_controller.is_media_playing()
            if playing and self._media_pause_allowed(request):
                self._was_media_playing = self.media_controller.pause()
                if not self._was_media_playing:
                    print("Warning: Failed to pause media playback")
                    self.notifications.notify(
                        "murmur", "Could not pause media playback"
                    )
        except Exception:
            print("Warning: Failed to pause media playback")
            self.notifications.notify("murmur", "Could not pause media playback")

    def _abort_startup(self) -> None:
        """Undo a cancelled startup; the cancel owner releases the hotkey."""
        self._cancel_live_workers()
        self._resume_owned_media()
        self._stop_live_transcription()
        self._stop_live_segmentation()
        self._session_active = False
        self._stats("finish", "cancelled")

    def _resume_owned_media(self) -> None:
        """Resume playback once when this attempt successfully paused it."""
        thread = self._media_pause_thread
        self._media_pause_thread = None
        if thread is not None:
            # Capture is closed before joining. Wait for a late successful pause
            # so it cannot affect the next session or leave playback paused.
            thread.join()
        if not self._was_media_playing:
            return
        self._was_media_playing = False
        try:
            if not self.media_controller.play():
                print("Warning: Failed to resume media playback")
                self.notifications.notify("murmur", "Could not resume media playback")
        except Exception:
            print("Warning: Failed to resume media playback")

    def _can_cancel(self) -> bool:
        """Whether the current session can still be discarded before delivery."""
        return self._get_cancel_session() is not None

    def _get_cancel_session(self) -> int | None:
        """Capture session identity before a tray request leaves the UI thread."""
        with self._recording_stop_lock:
            if (
                self._session_active
                and not self._delivery_started
                and not self._cancel_event.is_set()
            ):
                return self._overlay_session
            return None

    def _on_recording_cancel(
        self,
        session: int | None = None,
        *,
        pending_request: RecordingRequest | None = None,
    ) -> None:
        """Claim cancellation and let the existing stop owner finish teardown.

        `pending_request` is a hotkey stop that arrived before capture committed.
        """
        with self._recording_stop_lock:
            if pending_request is not None and not pending_request.is_active:
                return
            if (
                not self._session_active
                or self._delivery_started
                or (session is not None and session != self._overlay_session)
            ):
                if (
                    pending_request is not None
                    and not self._session_active
                    and not self._shutdown_event.is_set()
                ):
                    # Startup never began or already backed out; release the hotkey.
                    self._current_request = None
                    self.tray.set_status("Ready")
                    self.hotkey_manager.set_idle()
                return
            self._cancel_event.set()
            owns_stop = not self._recording_stop_started
            self._recording_stop_started = True
            self.hotkey_manager.set_processing()
            committed = (
                self._current_request is None or self._current_request.capture_committed
            )
            if committed:
                self.tray.set_status("Cancelled")
                self._show_overlay(CANCELLED)
            # Keep feedback and discard requests ahead of the owner's Ready event.
            self._cancel_live_workers()
        if owns_stop:
            self._finish_recording_stop()

    def _cancel_live_workers(self) -> None:
        """Drop queued work without waiting for model calls on this thread."""
        for worker in (self.live_segmenter, self.live_transcription_worker):
            if worker is not None:
                worker.cancel()

    def _on_recording_stop(self, request: RecordingRequest, committed: bool) -> None:
        """Handle recording stop via hotkey."""
        if not committed:
            self._on_recording_cancel(pending_request=request)
        elif self._claim_recording_stop(request=request):
            self._finish_recording_stop()

    def _claim_recording_stop(
        self, *, session: int | None = None, request: RecordingRequest | None = None
    ) -> bool:
        """Allow only one stop path to finalize the current recording."""
        with self._recording_stop_lock:
            # A stop delayed past this request's session no longer owns it.
            if request is not None and self._current_request is not request:
                return False
            if self._recording_stop_started or (
                session is not None and session != self._overlay_session
            ):
                return False
            self._recording_stop_started = True
        return True

    def _finish_recording_stop(self) -> None:
        """Stop capture and finalize its buffered audio."""
        with self._lifecycle_lock:
            try:
                self._stop_and_finalize_session()
            except TranscriptionCancelledError:
                self._stats("finish", "cancelled")
            except Exception:
                # Any step of finalization may have raised; don't blame capture.
                self._stats("issue", "finalization", "unknown_error")
                self._stats("finish", "failed")
                raise
            finally:
                if self._cancel_event.is_set():
                    self._stats("finish", "cancelled")
                # All worker/model work is done before another session is allowed.
                self.live_transcript_accumulator = None
                with self._recording_stop_lock:
                    if (
                        self._cancel_event.is_set()
                        and not self._shutdown_event.is_set()
                        and (
                            self._current_request is None
                            or self._current_request.capture_committed
                        )
                    ):
                        self._show_overlay(CANCELLED_DONE)
                    self._session_active = False
                    self._current_request = None
                    if not self._shutdown_event.is_set():
                        self.tray.set_status("Ready")
                        self.hotkey_manager.set_idle()

    def _stop_and_finalize_session(self) -> None:
        """Preserve normal stop's drain path and discard deliberately cancelled work."""
        print("Recording stopped.")
        if not self._cancel_event.is_set():
            self.tray.set_status("Finalizing...")
        finalization_started_at = time.perf_counter()

        try:
            audio_data = self.recorder.stop_recording()
            if audio_data is not None:
                self._stats("update", capture_seconds=audio_data.duration)
                # Audio can come back after a mid-recording mic error; keep that visible.
                self._stats(
                    "stage",
                    "capture",
                    "failed"
                    if getattr(self, "_stats_capture_error", False)
                    else "success",
                    audio_data.duration,
                )
            if audio_data is not None and not self._cancel_event.is_set():
                self._show_overlay(PROCESSING)
            # Resume media if it was playing before recording.
            self._resume_owned_media()
        finally:
            if self._cancel_event.is_set():
                self._cancel_live_workers()
            try:
                self._stop_live_segmentation()
            finally:
                self._stop_live_transcription()
        check_cancelled(self._cancel_event)

        if audio_data is not None:
            self._finalize_recording(
                audio_data,
                finalization_started_at=finalization_started_at,
            )
        else:
            self._stats("issue", "capture", "no_audio", category="routing")
            self._stats(
                "finish",
                "failed"
                if getattr(self, "_stats_capture_error", False)
                else "no_speech",
            )
            print("Warning: No audio data captured.")
            self._show_overlay(ERROR, OVERLAY_NO_AUDIO)

    def _on_recording_limit_reached(
        self, duration_seconds: float, *, session: int | None = None
    ) -> None:
        """Handle the recorder reaching max duration from the audio callback."""
        with self._recording_stop_lock:
            if session is not None and session != self._overlay_session:
                return
        with self._recording_limit_stop_lock:
            if self._recording_limit_stop_started:
                return
            self._recording_limit_stop_started = True

        thread = threading.Thread(
            target=self._handle_recording_limit_reached,
            args=(duration_seconds,),
            kwargs={"session": session},
            daemon=True,
        )
        thread.start()

    def _handle_recording_limit_reached(
        self, duration_seconds: float, *, session: int | None = None
    ) -> None:
        """Notify the user and finalize a recording stopped by the duration cap."""
        if not self._claim_recording_stop(session=session):
            return
        print(
            "Maximum recording duration reached "
            f"({duration_seconds:g}s). Finalizing recording."
        )
        self.notifications.notify_recording_limit_reached(duration_seconds)
        self.hotkey_manager.set_processing()
        self._finish_recording_stop()

    def _on_capture_error(self, message: str, *, session: int | None = None) -> None:
        """Finalize buffered audio after a microphone stream stops unexpectedly."""
        if not self._claim_recording_stop(session=session):
            return
        self._stats_capture_error = True
        self._stats("issue", "capture", "capture_failed")
        self.notifications.notify_error(
            f"{message}. Recording ended early; finalizing captured audio."
        )
        self.tray.set_status("Microphone error")
        self.hotkey_manager.set_processing()
        self._finish_recording_stop()

    def _finalize_recording(
        self,
        audio_data: AudioData,
        *,
        finalization_started_at: float,
    ) -> None:
        """Finalize stop-time output from the live transcript, or fall back offline."""
        check_cancelled(self._cancel_event)
        if audio_data.input_overflowed:
            print(
                "Warning: Microphone input overflow; some audio was lost. "
                "Finalizing captured audio; transcript may be incomplete."
            )
        self._stats("update", reached_transcription=True, transcription_route="live")
        if self._live_pipeline_degraded:
            self._stats("update", error_fallback=True)
            self._stats("issue", "live_whisper", "live_degraded")
            reason = (
                self._live_pipeline_degraded_reason or "Live transcription degraded"
            )
            print(
                f"Warning: {reason}. Recomputing final transcript from the full recording."
            )
            self._process_audio(
                audio_data,
                finalization_started_at=finalization_started_at,
            )
            return

        with observe_transcription(self._stats_observer()):
            live_text = self._get_live_transcript_text()
        if not live_text:
            self._stats("issue", "live_whisper", "live_empty", category="routing")
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
        except Exception:
            # Report before set_idle so a new session cannot be replaced.
            self._show_overlay(ERROR, OVERLAY_TRANSCRIPTION_FAILED)
            raise

    def _process_audio(
        self,
        audio_data: AudioData,
        *,
        finalization_started_at: float,
    ) -> None:
        """Process recorded audio through transcription."""
        self.hotkey_manager.set_processing()

        print(f"Processing {audio_data.duration:.1f}s of audio...")

        try:
            self._stats("update", reached_transcription=True)
            with observe_transcription(self._stats_observer()):
                text = self._transcribe_audio(audio_data)
            self._complete_transcription(
                audio_data,
                text,
                finalization_started_at=finalization_started_at,
                live_segment_metrics=LiveSegmentMetrics.empty(),
            )

        except TranscriptionCancelledError:
            raise
        except Exception:
            check_cancelled(self._cancel_event)
            # The transcriber already recorded failures it re-raises; don't double-count.
            self._stats(
                "issue_once",
                "offline_whisper",
                "whisper_failed",
                covered_by=("local_cleanup_failed",),
            )
            self._stats("finish", "failed")
            print("Transcription failed.")
            self.notifications.notify_error("Transcription failed.")
            self._show_overlay(ERROR, OVERLAY_TRANSCRIPTION_FAILED)

    def _transcribe_audio(self, audio_data: AudioData) -> str:
        """Transcribe audio through VAD segments with a full-clip fallback."""
        check_cancelled(self._cancel_event)
        segmenter = self._get_segmenter(audio_data.sample_rate)
        if segmenter is None:
            self._stats("update", transcription_route="full_clip", error_fallback=True)
            self._stats("issue", "offline_segmentation", "vad_unavailable")
            reason = self._vad_disabled_reason or "VAD unavailable"
            print(f"Warning: {reason}. Falling back to full clip.")
            return self.transcriber.transcribe(
                audio_data, cancel_event=self._cancel_event
            )

        segmentation_started = time.perf_counter()
        try:
            segments = segmenter.segment_audio(audio_data.audio)
        except Exception as exc:
            self._stats(
                "stage",
                "offline_segmentation",
                "failed",
                time.perf_counter() - segmentation_started,
            )
            self._stats("issue", "offline_segmentation", "vad_error")
            self._stats("update", transcription_route="full_clip", error_fallback=True)
            print(
                f"Warning: VAD segmentation failed: {exc}. Falling back to full clip."
            )
            check_cancelled(self._cancel_event)
            return self.transcriber.transcribe(
                audio_data, cancel_event=self._cancel_event
            )

        check_cancelled(self._cancel_event)

        self._stats(
            "stage",
            "offline_segmentation",
            "success",
            time.perf_counter() - segmentation_started,
        )

        if not segments:
            self._stats("update", transcription_route="full_clip")
            self._stats(
                "issue", "offline_segmentation", "vad_no_segments", category="routing"
            )
            print(
                "Warning: No VAD speech segments detected. Falling back to full clip."
            )
            return self.transcriber.transcribe(
                audio_data, cancel_event=self._cancel_event
            )

        print(f"Detected {len(segments)} speech segment(s).")
        self._stats("update", transcription_route="offline_segmented")
        for index, segment in enumerate(segments):
            print(f"  Segment {index}: {segment.duration:.2f}s")

        return self.transcriber.transcribe_segments(
            segments, debug=True, cancel_event=self._cancel_event
        ).text

    def _get_live_transcript_text(self) -> str:
        """Return the finalized live transcript text accumulated during recording."""
        if self.live_transcript_accumulator is None:
            return ""

        chunks = self.live_transcript_accumulator.ordered_chunks()
        if not chunks:
            return ""

        return self.transcriber.finalize_segment_texts(
            [chunk.text for chunk in chunks], cancel_event=self._cancel_event
        )

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
        with self._recording_stop_lock:
            check_cancelled(self._cancel_event)
            if self._delivery_started:
                return
            self._delivery_started = True
        if not text:
            self._stats("finish", "no_speech")
            print("Warning: No speech detected.")
            if not self._shutdown_event.is_set():
                self.notifications.notify("murmur", "No speech detected.")
                self._show_overlay(ERROR, OVERLAY_NO_SPEECH)
            return

        clipboard_started = time.perf_counter()
        copied = copy_to_clipboard(text)
        runtime = time.perf_counter() - finalization_started_at
        self._stats(
            "stage",
            "clipboard",
            "success" if copied else "failed",
            time.perf_counter() - clipboard_started,
        )
        if not copied:
            self._stats("issue", "clipboard", "clipboard_failed")
        self._stats(
            "finish",
            "success" if copied else "failed",
            final_word_count=len(text.split()),
            stop_to_clipboard_seconds=runtime if copied else None,
        )
        log_entry = self.logger.log(audio_data, text, runtime, live_segment_metrics)
        if self._shutdown_event.is_set():
            return

        if copied:
            print(f"Finalized in {runtime:.1f}s; copied to clipboard.")
            self._print_live_segment_metrics(live_segment_metrics)
            if audio_data.input_overflowed:
                self.notifications.notify(
                    "murmur", "Some audio was lost. Check the copied text."
                )
                self._show_overlay(ERROR, OVERLAY_AUDIO_INCOMPLETE)
            else:
                self.notifications.notify_transcription_copied()
                self._show_overlay(SUCCESS)
            return

        print(f"Finalized in {runtime:.1f}s; failed to copy to clipboard.")
        self._show_overlay(ERROR, OVERLAY_CLIPBOARD_FAILED)
        self._print_live_segment_metrics(live_segment_metrics)
        if log_entry is not None:
            self.notifications.notify_clipboard_failure_retry_with_training_data()
        else:
            self.notifications.notify_clipboard_failure_retry()

    def _show_overlay(self, state: str, message: str = "") -> None:
        """Mirror session state in the optional overlay without affecting dictation."""
        try:
            if self._shutdown_event.is_set():
                return
            if not self.config.show_recording_overlay:
                # Clear anything shown before the setting was switched off.
                if self._overlay_shown:
                    self._overlay_shown = False
                    self._close_overlay()
                return

            from .settings_gui import request_overlay_state

            if state != HIDDEN:
                self._overlay_shown = True
            request_overlay_state(state, self._overlay_session, message)
        except Exception:
            print("Recording overlay request failed.")

    def _close_overlay(self) -> None:
        """Hide and release the overlay window, if one was started."""
        try:
            from .settings_gui import close_overlay

            close_overlay()
        except Exception:
            print("Recording overlay cleanup failed.")

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

    def _format_optional_seconds(self, value: float | None) -> str:
        """Format optional second values for content-safe stdout metrics."""
        if value is None:
            return "n/a"

        return f"{value:.2f}s"

    def _get_segmenter(self, sample_rate: int) -> WebRTCVADSegmenter | None:
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
            print(
                f"Warning: {self._live_vad_disabled_reason}. Live segmentation disabled."
            )
            return

        try:
            worker = self._build_live_segmenter(self.recorder.sample_rate)
        except Exception as exc:
            self._live_vad_disabled_reason = f"Live VAD unavailable: {exc}"
            print(f"Warning: {self._live_vad_disabled_reason}")
            self.recorder.set_block_callback(None)
            self.recorder.set_block_callback_error_handler(None)
            return

        worker.start()
        self.live_segmenter = worker
        self.recorder.set_block_callback_error_handler(
            self._on_live_block_callback_error
        )
        self.recorder.set_block_callback(
            worker.submit_audio_block, replay_buffered=True
        )

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
            observer=self._stats_observer(),
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
        print(f"Warning: {message}")

    def _on_live_block_callback_error(self, exc: Exception) -> None:
        """Handle a recorder block callback failure during live segmentation."""
        print("Live audio callback failed.")
        self._on_live_pipeline_degraded("Live audio callback failed.")

    def _on_state_change(self, state: HotkeyState) -> None:
        """Handle state changes."""
        # State logging is handled in other methods


def main():
    """Main entry point for the application."""
    # The console script enters here directly; run.py has usually started this.
    capture_build_info_in_background()
    print("Starting murmur...")
    app = None
    tray_thread = None
    tray_error = None
    tray_done = threading.Event()

    def run_tray():
        nonlocal tray_error
        try:
            app.tray.run()
        except BaseException as exc:
            # Forward worker failures to the entrypoint's normal exit handling.
            tray_error = exc
        finally:
            tray_done.set()

    try:
        app = MurmurApp(preload_model=True)
        app.start()

        if sys.platform == "win32":
            # Keep native GetMessage off the thread that handles console signals.
            tray_thread = threading.Thread(target=run_tray, name="murmur-tray")
            tray_thread.start()
            while not tray_done.wait(timeout=0.1):
                pass
            if tray_error is not None:
                raise tray_error
        else:
            # Other tray backends may require the main thread.
            app.tray.run()

    except KeyboardInterrupt:
        print("\nInterrupted by user.")
    except Exception as e:
        print(f"Fatal error: {e}")
        sys.exit(1)
    finally:
        try:
            if app is not None:
                app.stop()
        finally:
            if tray_thread is not None and tray_thread.ident is not None:
                tray_thread.join()


if __name__ == "__main__":
    main()
