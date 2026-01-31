"""
Main application module for Murmur.
Orchestrates audio recording, transcription, and clipboard operations.
"""

import sys
import time
from typing import Optional

from .config import get_config, Config
from .audio import AudioRecorder, AudioData
from .transcription import Transcriber
from .clipboard import copy_to_clipboard
from .hotkey import HotkeyManager, HotkeyState, wait_for_exit
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
        self.transcriber = Transcriber()
        self.hotkey_manager = HotkeyManager()
        self.notifications = get_notification_manager()
        self.logger = get_logger()
        self.tray = TrayManager(on_exit_callback=self.stop)
        self.media_controller = get_media_controller()

        self._running = False
        self._was_media_playing = False

        if preload_model:
            print("Preloading Whisper model...")
            self.transcriber.load_model()

        # Ensure autostart matches config
        if self.config.start_with_windows:
            set_autostart(True)

    def start(self) -> None:
        """Start the application and begin listening for hotkeys."""
        if self._running:
            return

        self._running = True

        # Register hotkey
        success = self.hotkey_manager.register(
            on_start=self._on_recording_start,
            on_stop=self._on_recording_stop,
            on_state_change=self._on_state_change,
        )

        if not success:
            print("Failed to register hotkey. Exiting.")
            sys.exit(1)

        print(f"\n{'=' * 50}")
        print("murmur - Local Speech-to-Text")
        print(f"{'=' * 50}")
        print(f"Hotkey: {self.config.hotkey}")
        print(f"Model: {self.config.model_name}")
        print(f"Device: {self.transcriber.get_device_info()['device']}")
        print(f"Logging: {self.logger.get_log_directory()}")
        print(
            f"Logged entries: {self.logger.get_entry_count()} ({self.logger.get_total_duration():.1f}s audio)"
        )
        print(f"{'=' * 50}\n")

        self.notifications.notify("murmur", "Ready! Press hotkey to start recording.")

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
            self.recorder.start_recording()
        except Exception as e:
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
        self.tray.set_status("Processing...")

        audio_data = self.recorder.stop_recording()

        # Resume media if it was playing before recording
        if self._was_media_playing:
            if not self.media_controller.play():
                print("⚠️ Failed to resume media playback")
                self.notifications.notify("Murmur", "Could not resume media playback")
            self._was_media_playing = False

        if audio_data is not None:
            self._process_audio(audio_data)
        else:
            print("⚠️ No audio data captured.")
            self.tray.set_status("Ready")
            self.hotkey_manager.set_idle()

    def _process_audio(self, audio_data: AudioData) -> None:
        """Process recorded audio through transcription."""
        self.hotkey_manager.set_processing()
        self.notifications.notify_recording_stopped()

        print(f"📝 Processing {audio_data.duration:.1f}s of audio...")

        try:
            # Transcribe
            start_time = time.time()
            text = self.transcriber.transcribe(audio_data)
            elapsed = time.time() - start_time

            if text:
                # Copy to clipboard
                if copy_to_clipboard(text):
                    print(f'✅ Transcribed in {elapsed:.1f}s: "{text}"')
                    self.notifications.notify_transcription_complete(text)

                    # Log for training data
                    self.logger.log(audio_data, text, elapsed)
                else:
                    print(f'⚠️ Transcribed but failed to copy: "{text}"')
                    self.notifications.notify_error("Failed to copy to clipboard")
            else:
                print("⚠️ No speech detected.")
                self.notifications.notify("Murmur", "No speech detected.")

        except Exception as e:
            print(f"❌ Transcription error: {e}")
            self.notifications.notify_error(f"Transcription failed: {e}")

        finally:
            self.tray.set_status("Ready")
            self.hotkey_manager.set_idle()

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
