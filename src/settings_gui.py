"""Settings GUI for Murmur using customtkinter."""

import contextlib
import ctypes
import queue
import threading
import tkinter as tk
import weakref
from ctypes import wintypes
from datetime import datetime
from tkinter import messagebox

import customtkinter as ctk
from PIL import Image

from .assets import get_app_icon_path, get_logo_path
from .autostart import set_autostart
from .config import ConfigError, get_config, get_training_data_dir
from .hotkey import is_hotkey_valid
from .llm_postprocess import check_ollama_connection
from .logger import get_logger
from .settings_schema import (
    SETTINGS_BY_KEY,
    TAB_ORDER,
    NumericSettingError,
    normalize_value,
    parse_numeric_text,
    settings_for_tab,
)


def _format_bytes(size_bytes):
    if size_bytes < 1024:
        return f"{size_bytes} B"
    if size_bytes < 1024 * 1024:
        return f"{size_bytes / 1024:.1f} KB"
    return f"{size_bytes / (1024 * 1024):.1f} MB"


_INVALID_RESTART_COMPARE_VALUE = object()
_APP_DISPLAY_NAME = "murmur"
_SETTINGS_TITLE = "murmur settings"
_SETTINGS_WINDOW_GEOMETRY = "760x700"
_SETTINGS_WINDOW_MIN_SIZE = (700, 640)
_WINDOWS_APP_USER_MODEL_ID = "murmur"
_WINDOWS_APP_ID_SET = False
_IMAGE_ICON = 1
_LR_LOADFROMFILE = 0x00000010
_WM_SETICON = 0x0080
_ICON_SMALL = 0
_ICON_BIG = 1
_GCLP_HICON = -14
_GCLP_HICONSM = -34


def _restart_compare_value(value, setting):
    """Coerce without clamping so restart warnings catch clamped or repaired saves."""
    if setting.control == "number":
        try:
            return round(float(value))
        except (TypeError, ValueError, OverflowError):
            return _INVALID_RESTART_COMPARE_VALUE
    if setting.control == "select":
        return "" if value is None else str(value).strip()
    return normalize_value(value, setting)


def _slider_number_of_steps(setting):
    if (
        setting.step is None
        or setting.min_value is None
        or setting.max_value is None
        or setting.step <= 0
    ):
        return None
    return round((setting.max_value - setting.min_value) / setting.step)


class _SettingsWindowService:
    """Own the single CustomTk root and focus or create one settings window."""

    def __init__(self, master):
        self.master = master
        self.window = None

    def show(self):
        if self.window is not None and self.window.is_open():
            self.window.focus()
            return

        self.window = SettingsWindow(master=self.master, on_close=self._clear_window)
        self.window.focus()

    def _clear_window(self, window):
        if self.window is window:
            self.window = None


_settings_requests = queue.Queue()
_ollama_connection_test_results = queue.Queue()
_settings_thread = None
_settings_thread_lock = threading.Lock()


def _configure_customtkinter():
    ctk.set_appearance_mode("system")
    ctk.set_default_color_theme("blue")


def _configure_windows_app_identity():
    global _WINDOWS_APP_ID_SET

    if _WINDOWS_APP_ID_SET:
        return

    try:
        ctypes.windll.shell32.SetCurrentProcessExplicitAppUserModelID(
            _WINDOWS_APP_USER_MODEL_ID
        )
        _WINDOWS_APP_ID_SET = True
    except (AttributeError, OSError):
        return


def _apply_window_icon(window):
    icon_path = get_app_icon_path()
    logo_path = get_logo_path()
    if icon_path is None and logo_path is None:
        return None

    icon_images = []
    native_icons = None

    def apply_once():
        nonlocal native_icons

        if icon_path is not None:
            with contextlib.suppress(tk.TclError):
                window.iconbitmap(default=str(icon_path))
            native_icons = _apply_native_window_icon(
                window, icon_path, native_icons=native_icons
            )

        if logo_path is not None:
            if not icon_images:
                with contextlib.suppress(tk.TclError):
                    icon_images.append(tk.PhotoImage(file=str(logo_path)))
            if icon_images:
                with contextlib.suppress(tk.TclError):
                    window.iconphoto(True, icon_images[0])

    apply_once()

    try:
        window.after(250, apply_once)
        window.after(500, apply_once)
        window.after(1000, apply_once)
        window.after(1500, apply_once)
    except tk.TclError:
        pass

    return {
        "tk": icon_images,
        "native": native_icons if native_icons is not None else [],
    }


def _apply_native_window_icon(window, icon_path, native_icons=None):
    try:
        tk_hwnd = window.winfo_id()
    except tk.TclError:
        return native_icons

    try:
        user32 = ctypes.windll.user32
        user32.GetParent.argtypes = [wintypes.HWND]
        user32.GetParent.restype = wintypes.HWND
        hwnd = user32.GetParent(tk_hwnd) or tk_hwnd

        if native_icons is None:
            load_image = user32.LoadImageW
            load_image.argtypes = [
                wintypes.HINSTANCE,
                wintypes.LPCWSTR,
                wintypes.UINT,
                ctypes.c_int,
                ctypes.c_int,
                wintypes.UINT,
            ]
            load_image.restype = wintypes.HANDLE

            small_icon = load_image(
                None, str(icon_path), _IMAGE_ICON, 16, 16, _LR_LOADFROMFILE
            )
            big_icon = load_image(
                None, str(icon_path), _IMAGE_ICON, 32, 32, _LR_LOADFROMFILE
            )
            if not small_icon or not big_icon:
                return None

            native_icons = [small_icon, big_icon]

        small_icon, big_icon = native_icons

        user32.SendMessageW.argtypes = [
            wintypes.HWND,
            wintypes.UINT,
            wintypes.WPARAM,
            wintypes.LPARAM,
        ]
        user32.SendMessageW.restype = wintypes.LPARAM
        user32.SendMessageW(hwnd, _WM_SETICON, _ICON_SMALL, small_icon)
        user32.SendMessageW(hwnd, _WM_SETICON, _ICON_BIG, big_icon)

        set_class_long = getattr(user32, "SetClassLongPtrW", user32.SetClassLongW)
        set_class_long.argtypes = [wintypes.HWND, ctypes.c_int, ctypes.c_void_p]
        set_class_long.restype = ctypes.c_void_p
        set_class_long(hwnd, _GCLP_HICONSM, small_icon)
        set_class_long(hwnd, _GCLP_HICON, big_icon)

        return native_icons
    except (AttributeError, OSError, TypeError):
        return native_icons


def _run_settings_ui():
    global _settings_thread

    _configure_windows_app_identity()
    _configure_customtkinter()
    root = ctk.CTk()
    root._murmur_window_icon = _apply_window_icon(root)
    root.withdraw()
    service = _SettingsWindowService(root)

    def process_requests():
        try:
            while True:
                try:
                    request = _settings_requests.get_nowait()
                except queue.Empty:
                    break

                if request == "show":
                    service.show()
        except Exception as exc:
            messagebox.showerror(_APP_DISPLAY_NAME, f"Failed to open settings: {exc}")
        finally:
            root.after(100, process_requests)

    try:
        root.after(0, process_requests)
        root.mainloop()
    finally:
        with _settings_thread_lock:
            if _settings_thread is threading.current_thread():
                _settings_thread = None


def _ensure_settings_ui_thread():
    global _settings_thread

    with _settings_thread_lock:
        if _settings_thread is not None and _settings_thread.is_alive():
            return

        _settings_thread = threading.Thread(
            target=_run_settings_ui,
            name="MurmurSettingsUI",
            daemon=True,
        )
        _settings_thread.start()


class SettingsWindow:
    """A tabbed customtkinter window for editing Murmur configuration."""

    def __init__(self, master=None, on_close=None):
        self.config = get_config()
        self.logger = get_logger()
        self._on_close = on_close
        self._closed = False
        self._owns_root = master is None

        self.root = ctk.CTk() if self._owns_root else ctk.CTkToplevel(master)
        self.root.title(_SETTINGS_TITLE)
        self.root.geometry(_SETTINGS_WINDOW_GEOMETRY)
        self.root.minsize(*_SETTINGS_WINDOW_MIN_SIZE)
        self.root.protocol("WM_DELETE_WINDOW", self._close)
        self.setting_vars = {}
        self.numeric_vars = {}
        self._logo_image = None
        self._window_icon = None
        self._ollama_test_button = None
        self._ollama_test_in_progress = False
        self._ollama_test_token = None

        self._window_icon = _apply_window_icon(self.root)

        self._setup_ui()

    def _setup_ui(self):
        self.root.grid_columnconfigure(0, weight=1)
        self.root.grid_rowconfigure(1, weight=1)

        header = ctk.CTkFrame(self.root, fg_color="transparent")
        header.grid(row=0, column=0, sticky="ew", padx=24, pady=(20, 8))
        header.grid_columnconfigure(1, weight=1)

        logo_path = get_logo_path()
        if logo_path is not None:
            try:
                self._logo_image = ctk.CTkImage(
                    light_image=Image.open(logo_path),
                    dark_image=Image.open(logo_path),
                    size=(36, 36),
                )
                ctk.CTkLabel(header, text="", image=self._logo_image).grid(
                    row=0, column=0, sticky="w", padx=(0, 10)
                )
            except Exception:
                self._logo_image = None

        ctk.CTkLabel(
            header,
            text=_SETTINGS_TITLE,
            font=ctk.CTkFont(size=24, weight="bold"),
        ).grid(row=0, column=1, sticky="w")

        self.tabs = ctk.CTkTabview(self.root)
        self.tabs.grid(row=1, column=0, sticky="nsew", padx=24, pady=8)
        for tab_name in TAB_ORDER:
            tab = self.tabs.add(tab_name)
            tab.grid_columnconfigure(0, weight=1)

        self.hotkey_var = tk.StringVar(value=self.config.hotkey)
        self.model_var = tk.StringVar(
            value=normalize_value(self.config.model_name, SETTINGS_BY_KEY["model"])
        )
        self.device_var = tk.StringVar(
            value=normalize_value(self.config.device, SETTINGS_BY_KEY["device"])
        )
        self.lang_var = tk.StringVar(
            value=str(self.config.language) if self.config.language else ""
        )
        self.notify_var = tk.BooleanVar(value=self.config.enable_notifications)
        self.logging_var = tk.BooleanVar(value=self.config.enable_logging)
        self.autostart_var = tk.BooleanVar(value=self.config.start_with_windows)
        self.pause_media_var = tk.BooleanVar(
            value=self.config.pause_media_while_recording
        )
        self.ollama_enabled_var = tk.BooleanVar(value=self.config.ollama_enabled)
        self.ollama_endpoint_var = tk.StringVar(value=self.config.ollama_endpoint)
        self.ollama_model_name_var = tk.StringVar(value=self.config.ollama_model_name)
        self.ollama_preload_model_var = tk.BooleanVar(
            value=self.config.ollama_preload_model
        )
        self.setting_vars.update(
            {
                "hotkey": self.hotkey_var,
                "model": self.model_var,
                "device": self.device_var,
                "language": self.lang_var,
                "enable_notifications": self.notify_var,
                "enable_logging": self.logging_var,
                "start_with_windows": self.autostart_var,
                "pause_media_while_recording": self.pause_media_var,
                "ollama_enabled": self.ollama_enabled_var,
                "ollama_endpoint": self.ollama_endpoint_var,
                "ollama_model_name": self.ollama_model_name_var,
                "ollama_preload_model": self.ollama_preload_model_var,
            }
        )

        general = self.tabs.tab("General")
        self._add_text_row(general, 0, "Hotkey", self.hotkey_var)
        self._add_switch(general, 1, "Enable notifications", self.notify_var)
        self._add_switch(general, 2, "Start with Windows", self.autostart_var)
        self._add_switch(
            general,
            3,
            "Pause media while recording",
            self.pause_media_var,
        )

        vad = self.tabs.tab("VAD")
        self._add_number_row(vad, 0, SETTINGS_BY_KEY["vad_aggressiveness"])
        self._add_number_row(vad, 1, SETTINGS_BY_KEY["vad_padding_ms"])
        self._add_number_row(vad, 2, SETTINGS_BY_KEY["vad_silence_duration_ms"])

        transcription = self.tabs.tab("Transcription")
        self._add_select_row(
            transcription,
            0,
            "Whisper model",
            self.model_var,
            ["tiny", "base", "small", "medium", "large"],
        )
        self._add_select_row(
            transcription,
            1,
            "Device",
            self.device_var,
            ["cuda", "cpu"],
        )
        self._add_text_row(
            transcription,
            2,
            "Language",
            self.lang_var,
            "Leave blank or enter none for automatic language detection.",
        )
        self._add_number_row(
            transcription,
            3,
            SETTINGS_BY_KEY["max_recording_duration"],
        )

        cleanup = self.tabs.tab("LLM Cleanup")
        self._add_switch(
            cleanup,
            0,
            "Enable Ollama cleanup",
            self.ollama_enabled_var,
        )
        self._add_text_row(
            cleanup,
            1,
            "Ollama endpoint",
            self.ollama_endpoint_var,
        )
        self._add_text_row(
            cleanup,
            2,
            "Ollama model",
            self.ollama_model_name_var,
        )
        self._add_number_row(cleanup, 3, SETTINGS_BY_KEY["ollama_timeout_seconds"])
        self._add_switch(
            cleanup,
            4,
            "Preload Ollama model",
            self.ollama_preload_model_var,
        )
        self._ollama_test_button = ctk.CTkButton(
            cleanup,
            text="Test Ollama Connection",
            command=self._test_ollama_connection,
        )
        self._ollama_test_button.grid(
            row=5, column=0, sticky="w", padx=16, pady=(8, 10)
        )

        privacy = self.tabs.tab("Data Privacy")
        self._add_switch(
            privacy,
            0,
            "Enable training data logging",
            self.logging_var,
            "Stores raw WAV audio and transcript text locally after confirmation.",
        )
        ctk.CTkLabel(
            privacy,
            text=f"Training data location:\n{get_training_data_dir()}",
            anchor="w",
            justify="left",
            wraplength=620,
        ).grid(row=1, column=0, sticky="ew", padx=16, pady=(10, 6))
        ctk.CTkButton(
            privacy,
            text="Delete Logged Data",
            command=self._purge_training_data,
        ).grid(row=2, column=0, sticky="w", padx=16, pady=(4, 10))

        for tab_name in TAB_ORDER:
            self._add_tab_reset_button(self.tabs.tab(tab_name), tab_name)

        button_bar = ctk.CTkFrame(self.root, fg_color="transparent")
        button_bar.grid(row=2, column=0, sticky="ew", padx=24, pady=(8, 20))
        button_bar.grid_columnconfigure(0, weight=1)
        ctk.CTkButton(button_bar, text="Cancel", command=self._close).grid(
            row=0, column=1, padx=(0, 8)
        )
        ctk.CTkButton(button_bar, text="Save", command=self._save).grid(row=0, column=2)

    def _add_tab_reset_button(self, parent, tab_name):
        ctk.CTkButton(
            parent,
            text="Reset This Tab",
            command=lambda: self._reset_tab_to_defaults(tab_name),
        ).grid(row=99, column=0, sticky="w", padx=16, pady=(16, 10))

    def _add_text_row(self, parent, row, label, variable, help_text=""):
        frame = ctk.CTkFrame(parent, fg_color="transparent")
        frame.grid(row=row, column=0, sticky="ew", padx=16, pady=10)
        frame.grid_columnconfigure(1, weight=1)
        ctk.CTkLabel(frame, text=label, anchor="w").grid(row=0, column=0, sticky="w")
        ctk.CTkEntry(frame, textvariable=variable).grid(
            row=0, column=1, sticky="ew", padx=(16, 0)
        )
        if help_text:
            ctk.CTkLabel(
                frame,
                text=help_text,
                anchor="w",
                justify="left",
                text_color=("gray35", "gray70"),
                wraplength=520,
            ).grid(row=1, column=1, sticky="ew", padx=(16, 0), pady=(4, 0))

    def _add_select_row(self, parent, row, label, variable, values):
        frame = ctk.CTkFrame(parent, fg_color="transparent")
        frame.grid(row=row, column=0, sticky="ew", padx=16, pady=10)
        frame.grid_columnconfigure(1, weight=1)
        ctk.CTkLabel(frame, text=label, anchor="w").grid(row=0, column=0, sticky="w")
        ctk.CTkOptionMenu(frame, variable=variable, values=values).grid(
            row=0, column=1, sticky="ew", padx=(16, 0)
        )

    def _add_switch(self, parent, row, text, variable, help_text=""):
        frame = ctk.CTkFrame(parent, fg_color="transparent")
        frame.grid(row=row, column=0, sticky="ew", padx=16, pady=10)
        frame.grid_columnconfigure(0, weight=1)
        ctk.CTkSwitch(frame, text=text, variable=variable).grid(
            row=0, column=0, sticky="w"
        )
        if help_text:
            ctk.CTkLabel(
                frame,
                text=help_text,
                anchor="w",
                justify="left",
                text_color=("gray35", "gray70"),
                wraplength=620,
            ).grid(row=1, column=0, sticky="ew", pady=(4, 0))

    def _add_number_row(self, parent, row, setting):
        current_value = normalize_value(
            self.config.get(setting.key, setting.default),
            setting,
        )
        entry_var = tk.StringVar(value=str(current_value))
        slider_var = tk.DoubleVar(value=current_value)
        self.numeric_vars[setting.key] = entry_var
        self.setting_vars[setting.key] = entry_var

        frame = ctk.CTkFrame(parent, fg_color="transparent")
        frame.grid(row=row, column=0, sticky="ew", padx=16, pady=12)
        frame.grid_columnconfigure(1, weight=1)

        ctk.CTkLabel(frame, text=setting.label, anchor="w").grid(
            row=0, column=0, sticky="w"
        )
        value_label = ctk.CTkLabel(frame, text=f"Current: {current_value}")
        value_label.grid(row=0, column=2, sticky="e", padx=(12, 0))

        def on_slider(value):
            rounded_value = normalize_value(value, setting)
            entry_var.set(str(rounded_value))
            value_label.configure(text=f"Current: {rounded_value}")

        slider = ctk.CTkSlider(
            frame,
            from_=setting.min_value,
            to=setting.max_value,
            number_of_steps=_slider_number_of_steps(setting),
            variable=slider_var,
            command=on_slider,
        )
        slider.grid(row=1, column=1, sticky="ew", padx=(16, 12), pady=(6, 0))

        def on_entry_change(*_args):
            try:
                parsed_value = parse_numeric_text(entry_var.get(), setting)
            except NumericSettingError:
                value_label.configure(text="Current: invalid")
                return
            slider_var.set(parsed_value)
            value_label.configure(text=f"Current: {parsed_value}")

        entry_var.trace_add("write", on_entry_change)
        ctk.CTkEntry(frame, textvariable=entry_var, width=88).grid(
            row=1, column=2, sticky="e", pady=(6, 0)
        )
        ctk.CTkLabel(
            frame,
            text=(
                f"Min {setting.min_value:g} | Max {setting.max_value:g} | "
                f"Default {setting.default}"
            ),
            anchor="w",
            text_color=("gray35", "gray70"),
        ).grid(row=2, column=1, sticky="w", padx=(16, 0), pady=(4, 0))
        if setting.help_text:
            ctk.CTkLabel(
                frame,
                text=setting.help_text,
                anchor="w",
                justify="left",
                text_color=("gray35", "gray70"),
                wraplength=520,
            ).grid(
                row=3, column=1, columnspan=2, sticky="ew", padx=(16, 0), pady=(4, 0)
            )

    def _collect_numeric_values(self):
        numeric_values = {}
        for key, variable in getattr(self, "numeric_vars", {}).items():
            setting = SETTINGS_BY_KEY[key]
            try:
                parsed_value = parse_numeric_text(variable.get(), setting)
            except NumericSettingError:
                persisted_value = normalize_value(
                    self.config.get(setting.key, setting.default),
                    setting,
                )
                variable.set(str(persisted_value))
                messagebox.showerror(
                    _APP_DISPLAY_NAME,
                    f"Please enter a number for {setting.label}.",
                )
                return None
            variable.set(str(parsed_value))
            numeric_values[key] = parsed_value
        return numeric_values

    def _reset_tab_to_defaults(self, tab_name):
        for setting in settings_for_tab(tab_name):
            variable = self.setting_vars.get(setting.key)
            if variable is None:
                continue

            default_value = normalize_value(setting.default, setting)
            if setting.value_type == "optional_str" and default_value is None:
                default_value = ""
            variable.set(
                default_value if setting.value_type == "bool" else str(default_value)
            )

    def _read_config_value(self, key):
        if hasattr(self.config, "get"):
            return self.config.get(key, SETTINGS_BY_KEY[key].default)
        return getattr(self.config, key, SETTINGS_BY_KEY[key].default)

    def _changed_restart_settings(self, updated_values):
        changed_restart_settings = []
        for key, new_value in updated_values.items():
            setting = SETTINGS_BY_KEY.get(key)
            if setting is None or not setting.restart_required:
                continue
            old_value = _restart_compare_value(self._read_config_value(key), setting)
            if old_value != _restart_compare_value(new_value, setting):
                changed_restart_settings.append(setting.label)
        return changed_restart_settings

    def _set_ollama_test_button_busy(self, busy):
        button = getattr(self, "_ollama_test_button", None)
        if button is None:
            return

        button.configure(
            state="disabled" if busy else "normal",
            text="Testing…" if busy else "Test Ollama Connection",
        )

    def _finish_ollama_connection_test(self, result):
        self._set_ollama_test_button_busy(False)
        if result.ok:
            messagebox.showinfo(_APP_DISPLAY_NAME, result.message)
        else:
            messagebox.showerror(_APP_DISPLAY_NAME, result.message)

    def _poll_ollama_connection_test_result(self):
        if not self._ollama_test_in_progress:
            return

        try:
            while True:
                window_ref, token, result = _ollama_connection_test_results.get_nowait()
                if window_ref() is not self or token != self._ollama_test_token:
                    continue

                self._ollama_test_in_progress = False
                self._ollama_test_token = None
                self._finish_ollama_connection_test(result)
                return
        except queue.Empty:
            pass

        with contextlib.suppress(tk.TclError):
            self.root.after(100, self._poll_ollama_connection_test_result)

    def _test_ollama_connection(self):
        timeout_setting = SETTINGS_BY_KEY["ollama_timeout_seconds"]
        timeout_var = self.numeric_vars.get("ollama_timeout_seconds")
        try:
            configured_timeout = parse_numeric_text(timeout_var.get(), timeout_setting)
        except (AttributeError, NumericSettingError):
            configured_timeout = timeout_setting.default

        endpoint = self.ollama_endpoint_var.get()
        model_name = self.ollama_model_name_var.get()
        self._ollama_test_in_progress = True
        self._ollama_test_token = object()
        self._set_ollama_test_button_busy(True)
        self.root.after(100, self._poll_ollama_connection_test_result)

        test_window_ref = weakref.ref(self)
        test_token = self._ollama_test_token

        def run():
            result = check_ollama_connection(
                endpoint=endpoint,
                model_name=model_name,
                timeout=configured_timeout,
            )
            _ollama_connection_test_results.put((test_window_ref, test_token, result))

        threading.Thread(target=run, daemon=True).start()

    def _purge_training_data(self):
        summary = self.logger.get_storage_summary()
        if summary.file_count == 0:
            messagebox.showinfo(_APP_DISPLAY_NAME, "No logged training data was found.")
            return

        if not messagebox.askyesno(
            "Delete logged data",
            (
                "Delete all stored WAV files and transcription logs from the local "
                f"training data folder?\n\nFiles: {summary.file_count}\n"
                f"Approximate size: {_format_bytes(summary.total_bytes)}"
            ),
        ):
            return

        try:
            removed_files = self.logger.purge_all()
        except Exception as exc:
            messagebox.showerror(
                _APP_DISPLAY_NAME, f"Failed to delete logged data: {exc}"
            )
            return

        if removed_files:
            messagebox.showinfo(
                _APP_DISPLAY_NAME,
                f"Deleted {removed_files} logged file(s) from {get_training_data_dir()}.",
            )
        else:
            messagebox.showinfo(_APP_DISPLAY_NAME, "No logged training data was found.")

    def _save(self):
        new_hotkey = self.hotkey_var.get().strip()
        if not is_hotkey_valid(new_hotkey):
            messagebox.showerror(
                _APP_DISPLAY_NAME, "Please enter a valid hotkey before saving settings."
            )
            return

        lang = normalize_value(
            self.lang_var.get(),
            SETTINGS_BY_KEY["language"],
        )
        if lang is not None:
            from whisper.tokenizer import LANGUAGES

            normalized_lang = lang.lower()
            valid_languages = {code.lower() for code in LANGUAGES} | {
                name.lower() for name in LANGUAGES.values()
            }
            if normalized_lang not in valid_languages:
                messagebox.showerror(
                    _APP_DISPLAY_NAME,
                    "Please enter a valid Whisper language code or name for Language.",
                    parent=self.root,
                )
                return

        previous_logging = self.config.enable_logging
        new_logging = self.logging_var.get()

        if new_logging and not previous_logging:
            confirmed = messagebox.askyesno(
                "Enable training data logging",
                "Enabling this stores raw WAV audio and transcript text locally under the training data folder. Do you want to enable it?",
            )
            if not confirmed:
                self.logging_var.set(False)
                return

        numeric_values = self._collect_numeric_values()
        if numeric_values is None:
            return

        new_ollama_enabled = self.ollama_enabled_var.get()
        new_ollama_endpoint = self.ollama_endpoint_var.get().strip()
        new_ollama_model_name = self.ollama_model_name_var.get().strip()
        if new_ollama_enabled and (
            not new_ollama_endpoint or not new_ollama_model_name
        ):
            messagebox.showerror(
                _APP_DISPLAY_NAME,
                "Please enter an Ollama endpoint and model before enabling cleanup.",
            )
            return

        old_autostart = self.config.start_with_windows
        new_autostart = self.autostart_var.get()
        updated_values = {
            "hotkey": new_hotkey,
            "model": normalize_value(self.model_var.get(), SETTINGS_BY_KEY["model"]),
            "device": normalize_value(self.device_var.get(), SETTINGS_BY_KEY["device"]),
            "language": lang,
            "enable_notifications": self.notify_var.get(),
            "enable_logging": new_logging,
            "pause_media_while_recording": self.pause_media_var.get(),
            "ollama_enabled": new_ollama_enabled,
            "ollama_endpoint": new_ollama_endpoint,
            "ollama_model_name": new_ollama_model_name,
            "ollama_preload_model": self.ollama_preload_model_var.get(),
        }
        updated_values.update(numeric_values)

        restart_settings = self._changed_restart_settings(updated_values)

        if previous_logging != new_logging:
            updated_values["logging_consent_updated_at"] = (
                datetime.now().astimezone().isoformat()
            )
            updated_values["logging_consent_source"] = "settings"

        updated_values["start_with_windows"] = new_autostart

        try:
            self.config.update(updated_values)
        except ConfigError as exc:
            messagebox.showerror(_APP_DISPLAY_NAME, f"Failed to save settings: {exc}")
            return

        self.logger.set_enabled(new_logging)

        # Update registry if autostart changed
        if old_autostart != new_autostart:
            set_autostart(new_autostart)

        if restart_settings:
            messagebox.showinfo(
                _APP_DISPLAY_NAME,
                "Settings saved. Restart murmur for these changes to fully apply: "
                + ", ".join(restart_settings),
            )
        else:
            messagebox.showinfo(_APP_DISPLAY_NAME, "Settings saved.")
        self._close()

    def is_open(self):
        try:
            return bool(self.root.winfo_exists()) and not self._closed
        except tk.TclError:
            return False

    def focus(self):
        if not self.is_open():
            return

        self.root.deiconify()
        self.root.lift()
        self.root.focus_force()

    def _close(self):
        if getattr(self, "_closed", False):
            return

        self._closed = True
        try:
            self.root.destroy()
        finally:
            on_close = getattr(self, "_on_close", None)
            if on_close is not None:
                on_close(self)

    def show(self):
        self.focus()
        if self._owns_root:
            self.root.mainloop()


def show_settings():
    """Request the settings window from the persistent CustomTk UI thread."""
    _ensure_settings_ui_thread()
    _settings_requests.put("show")
