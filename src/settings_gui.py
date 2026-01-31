"""
Settings GUI for Murmur using tkinter.
"""

import tkinter as tk
from tkinter import ttk, messagebox
from .config import get_config, DEFAULT_CONFIG
from .autostart import set_autostart


class SettingsWindow:
    """A simple tkinter window for editing Murmur configuration."""

    def __init__(self):
        self.config = get_config()
        self.root = tk.Tk()
        self.root.title("Murmur Settings")
        self.root.geometry("400x500")
        self.root.resizable(False, False)

        # Set icon if possible
        # self.root.iconbitmap("path/to/icon.ico")

        self._setup_ui()

    def _setup_ui(self):
        main_frame = ttk.Frame(self.root, padding="20")
        main_frame.pack(fill=tk.BOTH, expand=True)

        # Hotkey
        ttk.Label(main_frame, text="Hotkey:").grid(row=0, column=0, sticky=tk.W, pady=5)
        self.hotkey_var = tk.StringVar(value=self.config.hotkey)
        ttk.Entry(main_frame, textvariable=self.hotkey_var).grid(
            row=0, column=1, sticky=tk.EW, pady=5
        )

        # Model
        ttk.Label(main_frame, text="Whisper Model:").grid(
            row=1, column=0, sticky=tk.W, pady=5
        )
        self.model_var = tk.StringVar(value=self.config.model_name)
        models = ["tiny", "base", "small", "medium", "large"]
        ttk.Combobox(
            main_frame, textvariable=self.model_var, values=models, state="readonly"
        ).grid(row=1, column=1, sticky=tk.EW, pady=5)

        # Device
        ttk.Label(main_frame, text="Device:").grid(row=2, column=0, sticky=tk.W, pady=5)
        self.device_var = tk.StringVar(value=self.config.device)
        devices = ["cuda", "cpu"]
        ttk.Combobox(
            main_frame, textvariable=self.device_var, values=devices, state="readonly"
        ).grid(row=2, column=1, sticky=tk.EW, pady=5)

        # Language
        ttk.Label(main_frame, text="Language (null for auto):").grid(
            row=3, column=0, sticky=tk.W, pady=5
        )
        self.lang_var = tk.StringVar(
            value=str(self.config.language) if self.config.language else ""
        )
        ttk.Entry(main_frame, textvariable=self.lang_var).grid(
            row=3, column=1, sticky=tk.EW, pady=5
        )

        # Notifications
        self.notify_var = tk.BooleanVar(value=self.config.enable_notifications)
        ttk.Checkbutton(
            main_frame, text="Enable Notifications", variable=self.notify_var
        ).grid(row=4, column=0, columnspan=2, sticky=tk.W, pady=5)

        # Logging
        self.logging_var = tk.BooleanVar(value=self.config.get("enable_logging", True))
        ttk.Checkbutton(
            main_frame, text="Enable Training Data Logging", variable=self.logging_var
        ).grid(row=5, column=0, columnspan=2, sticky=tk.W, pady=5)

        # Auto-start
        self.autostart_var = tk.BooleanVar(value=self.config.start_with_windows)
        ttk.Checkbutton(
            main_frame, text="Start with Windows", variable=self.autostart_var
        ).grid(row=6, column=0, columnspan=2, sticky=tk.W, pady=5)

        # Pause media while recording
        self.pause_media_var = tk.BooleanVar(
            value=self.config.pause_media_while_recording
        )
        ttk.Checkbutton(
            main_frame,
            text="Pause media while recording",
            variable=self.pause_media_var,
        ).grid(row=7, column=0, columnspan=2, sticky=tk.W, pady=5)

        # Buttons
        btn_frame = ttk.Frame(main_frame, padding="20")
        btn_frame.grid(row=8, column=0, columnspan=2, sticky=tk.EW)

        ttk.Button(btn_frame, text="Save", command=self._save).pack(
            side=tk.RIGHT, padx=5
        )
        ttk.Button(btn_frame, text="Cancel", command=self.root.destroy).pack(
            side=tk.RIGHT, padx=5
        )

        main_frame.columnconfigure(1, weight=1)

    def _save(self):
        # Update config
        self.config.set("hotkey", self.hotkey_var.get())
        self.config.set("model", self.model_var.get())
        self.config.set("device", self.device_var.get())

        lang = self.lang_var.get().strip()
        self.config.set("language", lang if lang and lang.lower() != "none" else None)

        self.config.set("enable_notifications", self.notify_var.get())
        self.config.set("enable_logging", self.logging_var.get())
        self.config.set("pause_media_while_recording", self.pause_media_var.get())

        old_autostart = self.config.start_with_windows
        new_autostart = self.autostart_var.get()
        self.config.set("start_with_windows", new_autostart)

        # Update registry if autostart changed
        if old_autostart != new_autostart:
            set_autostart(new_autostart)

        messagebox.showinfo(
            "Murmur", "Settings saved! Some changes may require a restart."
        )
        self.root.destroy()

    def show(self):
        self.root.focus_force()
        self.root.mainloop()


def show_settings():
    """Helper function to show the settings window."""
    app = SettingsWindow()
    app.show()
