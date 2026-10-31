"""Settings GUI for murmur using customtkinter."""

import contextlib
import csv
import ctypes
import math
import queue
import sys
import threading
import tkinter as tk
import weakref
from ctypes import wintypes
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path
from tkinter import filedialog, messagebox

import customtkinter as ctk
from PIL import Image

from .assets import get_app_icon_path, get_logo_path
from .autostart import is_autostart_enabled, set_autostart
from .config import ConfigError, get_config, get_training_data_dir
from .hotkey import is_hotkey_valid
from .input_devices import (
    input_backend,
    input_status,
    list_input_devices,
    resolve_input_device,
)
from .llm_postprocess import check_ollama_connection
from .logger import get_logger
from .notifications import get_notification_manager
from .recording_overlay import RecordingOverlay
from .settings_schema import (
    SETTINGS_BY_KEY,
    TAB_ORDER,
    NumericSettingError,
    normalize_value,
    parse_numeric_text,
    settings_for_tab,
)
from .usage_stats import EXPORT_COLUMNS, StatisticsBusyError, get_usage_stats
from .windows_identity import (
    configure_windows_app_identity as _configure_windows_app_identity,
)


def _write_statistics_csv(path, rows):
    """Write the service's flattened, content-free export outside its transaction."""
    with Path(path).open("w", encoding="utf-8", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=EXPORT_COLUMNS)
        writer.writeheader()
        for row in rows:
            values = {key: row.get(key) for key in EXPORT_COLUMNS}
            for key, value in values.items():
                if isinstance(value, str) and (
                    value.startswith(("\t", "\r", "\n"))
                    or value.lstrip().startswith(("=", "+", "-", "@"))
                ):
                    values[key] = "'" + value
            writer.writerow(values)


Row = tuple[str, str]


@dataclass(frozen=True)
class StatisticsView:
    """Display-ready statistics; the Statistics tab renders this without math."""

    notices: tuple[str, ...]
    empty: tuple[str, str] | None
    tiles: tuple[Row, ...]
    sections: tuple[tuple[str, tuple[Row, ...]], ...]
    footnotes: tuple[str, ...]


_STAGE_LABELS = {
    "capture": "Recording",
    "preparation": "Preparation",
    "live_whisper": "Live Whisper",
    "offline_segmentation": "Offline segmentation",
    "offline_whisper": "Offline Whisper",
    "local_cleanup": "Text cleanup",
    "ollama_cleanup": "Ollama cleanup",
    "clipboard": "Clipboard",
}


def _seconds(value):
    if value is None:
        return "n/a"
    if value < 0.1:
        # Quick steps read as zero in seconds; milliseconds show they ran.
        return "<1 ms" if 0 < value < 0.0005 else f"{value * 1000:.0f} ms"
    return f"{value:.2f} s" if value < 1 else f"{value:.1f} s"


def _percent(value):
    return "n/a" if value is None else f"{value:.0%}"


def _stage_ran(value):
    return bool(value.get("success_count") or value.get("failed_count"))


def _stage_value(value):
    counts = [
        f"{value[key]} {word}"
        for key, word in (
            ("success_count", "ok"),
            ("failed_count", "failed"),
            ("off_count", "off"),
            ("skipped_count", "skipped"),
        )
        if value.get(key)
    ]
    text = ", ".join(counts) or "no attempts"
    if value.get("mean_seconds") is not None:
        text += f" · {_seconds(value['mean_seconds'])} avg"
    return text


def _statistics_view(summary) -> StatisticsView:
    """Turn service aggregates into labelled values without recalculating them."""
    notices = []
    if not summary.get("storage_available", True):
        notices.append("Storage unavailable. History may be incomplete; try Refresh.")
    if summary.get("incomplete") or summary.get("reconciliation_incomplete"):
        notices.append("Some sessions or measurements may be missing.")
    if not summary.get("sessions", 0):
        empty = (
            ("No sessions yet", "Dictate something, then press Refresh.")
            if summary.get("enabled", True)
            else ("Collection is off", "Turn it on and press Save to start.")
        )
        return StatisticsView(tuple(notices), empty, (), (), ())
    if not summary.get("enabled", True):
        notices.append("Collection is off. Existing history is still shown.")

    counts = summary.get("counts", {})
    latency = summary.get("latency", {})
    routing = summary.get("normal_routing_counts", {})
    tiles = (
        (str(summary.get("sessions", 0)), "Sessions"),
        (_percent(summary.get("success_rate")), "Success rate"),
        (_seconds(latency.get("median_seconds")), "Median wait"),
        (str(summary.get("delivered_words") or 0), "Words delivered"),
    )
    reliability = [
        (
            "Succeeded / failed",
            f"{counts.get('success', 0)} / {counts.get('failed', 0)}",
        ),
        (
            "Cancelled / no speech / interrupted",
            f"{counts.get('cancelled', 0)} / {counts.get('no_speech', 0)} / "
            f"{counts.get('interrupted', 0)}",
        ),
        (
            "Needed transcription fallback",
            f"{summary.get('fallback_count', 0)} of "
            f"{summary.get('fallback_denominator', 0)}",
        ),
        ("Problems recovered automatically", str(summary.get("recovered_failures", 0))),
        (
            "Routed offline (empty live text / no speech found)",
            f"{routing.get('live_empty', 0)} / {routing.get('vad_no_segments', 0)}",
        ),
    ]
    if summary.get("active_sessions"):
        reliability.append(("In progress", str(summary["active_sessions"])))
    minutes = summary.get("captured_minutes")
    speed = (
        (
            "Stop to clipboard (median / p95)",
            f"{_seconds(latency.get('median_seconds'))} / "
            f"{_seconds(latency.get('p95_seconds'))}",
        ),
        ("Average recording", _seconds(summary.get("mean_capture_seconds"))),
        ("Total recorded", "n/a" if minutes is None else f"{minutes:.1f} min"),
    )
    sections = [("Reliability", tuple(reliability)), ("Speed", speed)]
    stages = tuple(
        (
            _STAGE_LABELS.get(name, name.replace("_", " ").capitalize()),
            _stage_value(value),
        )
        for name, value in summary.get("stages", {}).items()
        if _stage_ran(value) or value.get("skipped_count") or value.get("off_count")
    )
    if stages:
        sections.append(("Stages", stages))
    models = []
    for group in summary.get("model_groups", []):
        label = (
            f"{group.get('whisper_model') or 'unknown'} · "
            f"{(group.get('whisper_device') or 'unknown').upper()}"
        )
        if group.get("cleanup_enabled"):
            cleanup = group.get("cleanup_model")
            label += f" + {cleanup or 'unknown'} cleanup"
        sessions = group.get("sessions", 0)
        median = _seconds(group.get("latency", {}).get("median_seconds"))
        models.append(
            (label, f"{sessions} session{'s' * (sessions != 1)} · {median} median")
        )
    cleanup_groups = [g for g in summary.get("cleanup_groups", []) if _stage_ran(g)]
    # One cleanup model is already covered by the Ollama cleanup stage row.
    if len(cleanup_groups) > 1:
        models.extend(
            (f"{group.get('model') or 'unknown'} cleanup", _stage_value(group))
            for group in cleanup_groups
        )
    if models:
        sections.append(("Models", tuple(models)))

    footnotes = ["Stage times overlap, so they don't add up to the wait."]
    if latency.get("small_sample"):
        footnotes.insert(
            0,
            f"Fewer than 20 successful sessions so far ({latency.get('count', 0)}), "
            "so speed numbers may swing.",
        )
    return StatisticsView(
        tuple(notices), None, tiles, tuple(sections), tuple(footnotes)
    )


def _format_bytes(size_bytes):
    if size_bytes < 1024:
        return f"{size_bytes} B"
    if size_bytes < 1024 * 1024:
        return f"{size_bytes / 1024:.1f} KB"
    return f"{size_bytes / (1024 * 1024):.1f} MB"


_MUTED_TEXT = ("gray35", "gray70")
_OUTLINE_BUTTON = {
    "fg_color": "transparent",
    "border_width": 1,
    "border_color": ("gray55", "gray40"),
    "text_color": ("gray10", "gray90"),
    "hover_color": ("gray78", "gray25"),
}
# Destructive actions: outlined in red so they stand apart from normal buttons.
_DANGER_BUTTON = {
    **_OUTLINE_BUTTON,
    "border_color": ("#C0392B", "#C94B4B"),
    "text_color": ("#B03A2E", "#FF8A80"),
    "hover_color": ("#F6D5D1", "#4A2323"),
}
_INVALID_RESTART_COMPARE_VALUE = object()
_APP_DISPLAY_NAME = "murmur"
_SETTINGS_TITLE = "murmur settings"
_SETTINGS_WINDOW_GEOMETRY = "760x700"
_SETTINGS_WINDOW_MIN_SIZE = (700, 640)
_IMAGE_ICON = 1
_LR_LOADFROMFILE = 0x00000010
_WM_SETICON = 0x0080
_ICON_SMALL = 0
_ICON_BIG = 1
_GCLP_HICON = -14
_GCLP_HICONSM = -34


class _MonitorInfo(ctypes.Structure):
    """Receive the opening monitor's bounds and taskbar-free work area."""

    _fields_ = [
        ("cbSize", wintypes.DWORD),
        ("rcMonitor", wintypes.RECT),
        ("rcWork", wintypes.RECT),
        ("dwFlags", wintypes.DWORD),
    ]


def _settings_window_dimensions(work_size, decoration_size, scaling):
    """Cap logical client dimensions to the available physical work area."""
    available = tuple(
        max(1, math.floor((work - decoration - 16) / scaling))
        for work, decoration in zip(work_size, decoration_size, strict=True)
    )
    default = tuple(map(int, _SETTINGS_WINDOW_GEOMETRY.split("x")))
    size = tuple(
        min(wanted, limit) for wanted, limit in zip(default, available, strict=True)
    )
    minimum = tuple(
        min(wanted, limit)
        for wanted, limit in zip(_SETTINGS_WINDOW_MIN_SIZE, available, strict=True)
    )
    return size, minimum


def _fit_settings_window(window):
    """Fit a newly laid-out Windows settings window to its monitor's work area."""
    if sys.platform != "win32":
        return

    try:
        window.update_idletasks()
        user32 = ctypes.windll.user32
        user32.GetAncestor.argtypes = [wintypes.HWND, wintypes.UINT]
        user32.GetAncestor.restype = wintypes.HWND
        hwnd = user32.GetAncestor(window.winfo_id(), 2)  # GA_ROOT
        user32.MonitorFromWindow.argtypes = [wintypes.HWND, wintypes.DWORD]
        user32.MonitorFromWindow.restype = wintypes.HANDLE
        monitor = user32.MonitorFromWindow(hwnd, 2)  # MONITOR_DEFAULTTONEAREST
        user32.GetMonitorInfoW.argtypes = [
            wintypes.HANDLE,
            ctypes.POINTER(_MonitorInfo),
        ]
        user32.GetMonitorInfoW.restype = wintypes.BOOL
        info = _MonitorInfo()
        info.cbSize = ctypes.sizeof(info)
        user32.GetWindowRect.argtypes = [wintypes.HWND, ctypes.POINTER(wintypes.RECT)]
        user32.GetWindowRect.restype = wintypes.BOOL
        user32.GetClientRect.argtypes = [wintypes.HWND, ctypes.POINTER(wintypes.RECT)]
        user32.GetClientRect.restype = wintypes.BOOL
        outer, client = wintypes.RECT(), wintypes.RECT()
        if not (
            hwnd
            and monitor
            and user32.GetMonitorInfoW(monitor, ctypes.byref(info))
            and user32.GetWindowRect(hwnd, ctypes.byref(outer))
            and user32.GetClientRect(hwnd, ctypes.byref(client))
        ):
            return

        # CustomTkinter reports logical geometry; Win32 reports physical client size.
        # Measuring the ratio also respects CustomTkinter's custom window scaling.
        logical_width = int(window.geometry().split("x", 1)[0])
        if logical_width <= 0 or client.right <= 0:
            return
        scaling = client.right / logical_width
        work = info.rcWork
        size, minimum = _settings_window_dimensions(
            (work.right - work.left, work.bottom - work.top),
            (
                outer.right - outer.left - client.right,
                outer.bottom - outer.top - client.bottom,
            ),
            scaling,
        )
        window.minsize(*minimum)
        window.geometry(f"{size[0]}x{size[1]}")
        window.update_idletasks()

        # Clamp the outer position too, including monitors with negative coordinates.
        if user32.GetWindowRect(hwnd, ctypes.byref(outer)):
            x = max(work.left, min(outer.left, work.right - (outer.right - outer.left)))
            y = max(work.top, min(outer.top, work.bottom - (outer.bottom - outer.top)))
            user32.SetWindowPos.argtypes = [
                wintypes.HWND,
                wintypes.HWND,
                ctypes.c_int,
                ctypes.c_int,
                ctypes.c_int,
                ctypes.c_int,
                wintypes.UINT,
            ]
            user32.SetWindowPos.restype = wintypes.BOOL
            user32.SetWindowPos(hwnd, None, x, y, 0, 0, 0x0015)
            # SWP_NOSIZE | SWP_NOZORDER | SWP_NOACTIVATE
    except (AttributeError, OSError, tk.TclError):
        return


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


# The UI thread serves both the settings window and the recording overlay.
_SHOW_SETTINGS_REQUEST = "show"
_OVERLAY_REQUEST = "overlay"
_CLOSE_OVERLAY_REQUEST = "overlay_close"
_FOCUS_RECHECK_MS = 250

_settings_requests = queue.Queue()
_ollama_connection_test_results = queue.Queue()
_settings_thread = None
_settings_thread_lock = threading.Lock()
_overlay_unavailable = False
_overlay_cancel_callback = None


def set_overlay_cancel_callback(callback):
    """Register cancellation without creating the UI or capturing an old callback."""
    global _overlay_cancel_callback
    _overlay_cancel_callback = callback


def _dispatch_overlay_cancel(session):
    callback = _overlay_cancel_callback
    if callback is not None:
        threading.Thread(
            target=_run_overlay_cancel, args=(callback, session), daemon=True
        ).start()


def _run_overlay_cancel(callback, session):
    try:
        callback(session)
    except Exception as exc:
        print(f"Overlay cancellation failed ({type(exc).__name__}).", file=sys.stderr)


def _drain_settings_requests():
    drained = []
    while True:
        try:
            drained.append(_settings_requests.get_nowait())
        except queue.Empty:
            return drained


def _report_settings_ui_failure(exc):
    message = f"Failed to open settings: {exc}"
    print(message, file=sys.stderr)
    get_notification_manager().notify_error(message)


def _report_overlay_unavailable(exc):
    """Disable overlay requests after the UI thread cannot start for them."""
    global _overlay_unavailable

    if _overlay_unavailable:
        return
    _overlay_unavailable = True
    print(
        f"Recording overlay unavailable ({type(exc).__name__}).",
        file=sys.stderr,
    )


def _foreground_window():
    if sys.platform != "win32":
        return None
    try:
        user32 = ctypes.windll.user32
        user32.GetForegroundWindow.restype = wintypes.HWND
        return user32.GetForegroundWindow()
    except (AttributeError, OSError):
        return None


def _return_foreground(previous, root):
    """Give focus back if creating the hidden root activated it.

    CustomTkinter briefly maps a new root on Windows. Without this, starting the
    UI thread for the overlay could pull focus from the app being dictated into.
    Only the hidden root is checked, so a real Murmur window such as Settings
    never has its focus taken away.
    """
    if not previous:
        return
    try:
        user32 = ctypes.windll.user32
        user32.IsWindow.argtypes = [wintypes.HWND]
        user32.IsWindow.restype = wintypes.BOOL
        user32.SetForegroundWindow.argtypes = [wintypes.HWND]
        user32.SetForegroundWindow.restype = wintypes.BOOL

        root_window = int(root.wm_frame(), 16)
        if _foreground_window() == root_window and user32.IsWindow(previous):
            user32.SetForegroundWindow(previous)
    except (AttributeError, OSError, ValueError, tk.TclError):
        return


def _process_ui_request(request, service, overlay):
    if request == _SHOW_SETTINGS_REQUEST:
        service.show()
    elif request == _CLOSE_OVERLAY_REQUEST:
        overlay.close()
    elif isinstance(request, tuple) and request[0] == _OVERLAY_REQUEST:
        overlay.request(*request[1:])


def _configure_customtkinter():
    ctk.set_appearance_mode("system")
    ctk.set_default_color_theme("blue")


def _apply_window_icon(window):
    icon_path = get_app_icon_path()
    logo_path = get_logo_path()
    if icon_path is None and logo_path is None:
        return None

    icon_images = []
    native_icons = None
    class_icons_before = []

    def apply_once():
        nonlocal native_icons

        if icon_path is not None:
            with contextlib.suppress(tk.TclError):
                window.iconbitmap(default=str(icon_path))
            native_icons = _apply_native_window_icon(
                window,
                icon_path,
                native_icons=native_icons,
                class_icons_before=class_icons_before,
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
        "class_before": class_icons_before,
    }


def _class_icon_handles(user32, hwnd):
    """Read the shared class icon handles before temporarily replacing them."""
    get_class_long = getattr(user32, "GetClassLongPtrW", None) or getattr(
        user32, "GetClassLongW", None
    )
    if get_class_long is None:
        return None

    get_class_long.argtypes = [wintypes.HWND, ctypes.c_int]
    get_class_long.restype = ctypes.c_void_p
    return [
        get_class_long(hwnd, _GCLP_HICONSM),
        get_class_long(hwnd, _GCLP_HICON),
    ]


def _apply_native_window_icon(
    window, icon_path, native_icons=None, class_icons_before=None
):
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
                _destroy_native_icons([small_icon, big_icon])
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

        if class_icons_before is not None and not class_icons_before:
            previous = _class_icon_handles(user32, hwnd)
            if previous is not None:
                class_icons_before.extend(previous)

        set_class_long = (
            getattr(user32, "SetClassLongPtrW", None) or user32.SetClassLongW
        )
        set_class_long.argtypes = [wintypes.HWND, ctypes.c_int, ctypes.c_void_p]
        set_class_long.restype = ctypes.c_void_p
        set_class_long(hwnd, _GCLP_HICONSM, small_icon)
        set_class_long(hwnd, _GCLP_HICON, big_icon)

        return native_icons
    except (AttributeError, OSError, TypeError):
        return native_icons


def _restore_native_window_class_icons(window, class_icons_before):
    """Restore shared class icons before releasing a settings window's handles."""
    if not class_icons_before:
        return

    try:
        tk_hwnd = window.winfo_id()
        user32 = ctypes.windll.user32
        user32.GetParent.argtypes = [wintypes.HWND]
        user32.GetParent.restype = wintypes.HWND
        hwnd = user32.GetParent(tk_hwnd) or tk_hwnd
        set_class_long = (
            getattr(user32, "SetClassLongPtrW", None) or user32.SetClassLongW
        )
        set_class_long.argtypes = [wintypes.HWND, ctypes.c_int, ctypes.c_void_p]
        set_class_long.restype = ctypes.c_void_p
        set_class_long(hwnd, _GCLP_HICONSM, class_icons_before[0])
        set_class_long(hwnd, _GCLP_HICON, class_icons_before[1])
    except (AttributeError, OSError, TypeError, tk.TclError):
        return


def _destroy_native_icons(native_icons):
    """Release icon handles owned by a closed settings window."""
    if not native_icons:
        return

    try:
        destroy_icon = ctypes.windll.user32.DestroyIcon
        destroy_icon.argtypes = [wintypes.HICON]
        destroy_icon.restype = wintypes.BOOL
        for icon in native_icons:
            if icon:
                destroy_icon(icon)
    except (AttributeError, OSError, TypeError):
        return


def _run_settings_ui(return_focus=False):
    global _settings_thread

    failure = None
    pending = []
    try:
        _configure_windows_app_identity()
        _configure_customtkinter()
        previous_foreground = _foreground_window() if return_focus else None
        root = ctk.CTk()
        root._murmur_window_icon = _apply_window_icon(root)
        root.withdraw()
        if previous_foreground:
            _return_foreground(previous_foreground, root)
            # Check again once the main loop runs, in case activation came late
            # or the first hand-back did not take effect.
            root.after(_FOCUS_RECHECK_MS, _return_foreground, previous_foreground, root)
        service = _SettingsWindowService(root)
        overlay = RecordingOverlay(root, on_cancel=_dispatch_overlay_cancel)

        def process_requests():
            try:
                while True:
                    try:
                        request = _settings_requests.get_nowait()
                    except queue.Empty:
                        break

                    _process_ui_request(request, service, overlay)
            except Exception as exc:
                messagebox.showerror(
                    _APP_DISPLAY_NAME, f"Failed to open settings: {exc}"
                )
            finally:
                root.after(100, process_requests)

        root.after(0, process_requests)
        root.mainloop()
    except Exception as exc:
        failure = exc
    finally:
        with _settings_thread_lock:
            if _settings_thread is threading.current_thread():
                _settings_thread = None
                pending = _drain_settings_requests()

    if failure is not None:
        # Overlay-only startups fail quietly; dictation must not surface a
        # settings error the user never asked for.
        overlay_pending = any(request != _SHOW_SETTINGS_REQUEST for request in pending)
        if _SHOW_SETTINGS_REQUEST in pending or not (return_focus or overlay_pending):
            _report_settings_ui_failure(failure)
        if return_focus or overlay_pending:
            _report_overlay_unavailable(failure)


def _ensure_settings_ui_thread(return_focus=False):
    """Start the shared UI thread if needed.

    ``return_focus`` hands focus back after the hidden root starts, for
    background callers such as the overlay; a settings click keeps focus.
    """
    global _settings_thread

    with _settings_thread_lock:
        if _settings_thread is not None and _settings_thread.is_alive():
            return

        _drain_settings_requests()
        _settings_thread = threading.Thread(
            target=_run_settings_ui,
            args=(return_focus,),
            name="MurmurSettingsUI",
            daemon=True,
        )
        _settings_thread.start()


class SettingsWindow:
    """A tabbed customtkinter window for editing murmur configuration."""

    def __init__(self, master=None, on_close=None):
        self.config = get_config()
        self._startup_notice = self.config.get_startup_notice()
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
        self._ollama_test_button = None
        self._ollama_test_in_progress = False
        self._ollama_test_token = None
        self._statistics_results = queue.Queue()
        self._statistics_token = None

        self._window_icon = _apply_window_icon(self.root)

        self._setup_ui()
        self.root.after_idle(lambda: _fit_settings_window(self.root))

    def _setup_ui(self):
        self.root.grid_columnconfigure(0, weight=1)
        tabs_row = 2 if self._startup_notice else 1
        button_row = tabs_row + 1
        self.root.grid_rowconfigure(tabs_row, weight=1)

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

        if self._startup_notice:
            self._add_startup_notice(row=1)

        self.tabs = ctk.CTkTabview(self.root)
        self.tabs.grid(row=tabs_row, column=0, sticky="nsew", padx=24, pady=8)
        tab_contents = {}
        for tab_name in TAB_ORDER:
            tab = self.tabs.add(tab_name)
            tab.grid_columnconfigure(0, weight=1)
            tab.grid_rowconfigure(0, weight=1)
            content = ctk.CTkScrollableFrame(
                tab,
                fg_color=("gray86", "gray17"),
            )
            if sys.platform == "win32":
                # CTk scrolls 20 units per wheel notch; no public speed option exists.
                content._parent_canvas.configure(yscrollincrement=3)
            content.grid(row=0, column=0, sticky="nsew")
            content.grid_columnconfigure(0, weight=1)
            tab_contents[tab_name] = content

        self.hotkey_var = tk.StringVar(master=self.root, value=self.config.hotkey)
        self.model_var = tk.StringVar(
            master=self.root,
            value=normalize_value(self.config.model_name, SETTINGS_BY_KEY["model"]),
        )
        self.device_var = tk.StringVar(
            master=self.root,
            value=normalize_value(self.config.device, SETTINGS_BY_KEY["device"]),
        )
        self.lang_var = tk.StringVar(
            master=self.root,
            value=str(self.config.language) if self.config.language else "",
        )
        self.notify_var = tk.BooleanVar(
            master=self.root, value=self.config.enable_notifications
        )
        self.overlay_var = tk.BooleanVar(
            master=self.root, value=self.config.show_recording_overlay
        )
        self.logging_var = tk.BooleanVar(
            master=self.root, value=self.config.enable_logging
        )
        self.autostart_var = tk.BooleanVar(
            master=self.root, value=self.config.start_with_windows
        )
        self.pause_media_var = tk.BooleanVar(
            master=self.root, value=self.config.pause_media_while_recording
        )
        self.keep_microphone_ready_var = tk.BooleanVar(
            master=self.root, value=self.config.keep_microphone_ready
        )
        self.ollama_enabled_var = tk.BooleanVar(
            master=self.root, value=self.config.ollama_enabled
        )
        self.ollama_endpoint_var = tk.StringVar(
            master=self.root, value=self.config.ollama_endpoint
        )
        self.ollama_model_name_var = tk.StringVar(
            master=self.root, value=self.config.ollama_model_name
        )
        self.ollama_preload_model_var = tk.BooleanVar(
            master=self.root, value=self.config.ollama_preload_model
        )
        self.statistics_var = tk.BooleanVar(
            master=self.root, value=self.config.enable_statistics
        )
        self.setting_vars.update(
            {
                "hotkey": self.hotkey_var,
                "model": self.model_var,
                "device": self.device_var,
                "language": self.lang_var,
                "enable_notifications": self.notify_var,
                "show_recording_overlay": self.overlay_var,
                "enable_logging": self.logging_var,
                "enable_statistics": self.statistics_var,
                "start_with_windows": self.autostart_var,
                "pause_media_while_recording": self.pause_media_var,
                "keep_microphone_ready": self.keep_microphone_ready_var,
                "ollama_enabled": self.ollama_enabled_var,
                "ollama_endpoint": self.ollama_endpoint_var,
                "ollama_model_name": self.ollama_model_name_var,
                "ollama_preload_model": self.ollama_preload_model_var,
            }
        )

        general = tab_contents["General"]
        self._add_microphone_row(general, 0)
        self._add_text_row(general, 1, "Hotkey", self.hotkey_var, stacked=True)
        self._add_switch(general, 2, "Show recording overlay", self.overlay_var)
        self._add_switch(general, 3, "Enable notifications", self.notify_var)
        self._add_switch(general, 4, "Start with Windows", self.autostart_var)
        self._add_switch(
            general,
            5,
            "Pause media while recording",
            self.pause_media_var,
        )
        self._add_switch(
            general,
            6,
            "Keep microphone ready",
            self.keep_microphone_ready_var,
        )

        vad = tab_contents["VAD"]
        self._add_number_row(vad, 0, SETTINGS_BY_KEY["vad_aggressiveness"])
        self._add_number_row(vad, 1, SETTINGS_BY_KEY["vad_padding_ms"])
        self._add_number_row(vad, 2, SETTINGS_BY_KEY["vad_silence_duration_ms"])

        transcription = tab_contents["Transcription"]
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

        cleanup = tab_contents["LLM Cleanup"]
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

        privacy = tab_contents["Data Privacy"]
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
            **_DANGER_BUTTON,
        ).grid(row=2, column=0, sticky="w", padx=16, pady=(4, 10))

        self._setup_statistics(tab_contents["Statistics"])

        for tab_name in TAB_ORDER:
            self._add_tab_reset_button(tab_contents[tab_name], tab_name)

        button_bar = ctk.CTkFrame(self.root, fg_color="transparent")
        button_bar.grid(row=button_row, column=0, sticky="ew", padx=24, pady=(8, 20))
        button_bar.grid_columnconfigure(0, weight=1)
        ctk.CTkButton(button_bar, text="Cancel", command=self._close).grid(
            row=0, column=1, padx=(0, 8)
        )
        ctk.CTkButton(button_bar, text="Save", command=self._save).grid(row=0, column=2)

    def _add_startup_notice(self, row):
        banner = ctk.CTkFrame(
            self.root,
            fg_color=("#FFF4D6", "#3A2C13"),
            border_color=("#B7791F", "#A87928"),
            border_width=1,
        )
        banner.grid(row=row, column=0, sticky="ew", padx=24, pady=(4, 4))
        banner.grid_columnconfigure(0, weight=1)
        ctk.CTkLabel(
            banner,
            text="⚠ Settings were repaired",
            text_color=("#704400", "#FFD27F"),
            font=ctk.CTkFont(size=15, weight="bold"),
            anchor="w",
        ).grid(row=0, column=0, sticky="ew", padx=(12, 8), pady=(8, 2))
        ctk.CTkButton(
            banner,
            text="Dismiss",
            command=self._dismiss_startup_notice,
            width=76,
            height=26,
            fg_color="transparent",
            text_color=("#704400", "#FFD27F"),
            border_color=("#B7791F", "#A87928"),
            border_width=1,
            hover_color=("#F0DCA9", "#57421E"),
        ).grid(row=0, column=1, sticky="ne", padx=(8, 12), pady=(8, 2))
        notice_text = ctk.CTkTextbox(
            banner,
            height=52,
            wrap="word",
            fg_color="transparent",
            text_color=("#5D431B", "#F1D8AA"),
            border_width=0,
        )
        notice_text.grid(
            row=1, column=0, columnspan=2, sticky="ew", padx=8, pady=(0, 8)
        )
        notice_text.insert("1.0", self._startup_notice)
        notice_text.configure(state="disabled")
        self._startup_notice_banner = banner

    def _dismiss_startup_notice(self):
        self.config.consume_startup_notice()
        banner = getattr(self, "_startup_notice_banner", None)
        if banner is not None:
            banner.destroy()
            self._startup_notice_banner = None
        self._startup_notice = None

    def _add_tab_reset_button(self, parent, tab_name):
        ctk.CTkButton(
            parent,
            text="Reset This Tab",
            command=lambda: self._reset_tab_to_defaults(tab_name),
        ).grid(row=99, column=0, sticky="w", padx=16, pady=(16, 10))

    def _add_text_row(
        self, parent, row, label, variable, help_text="", *, stacked=False
    ):
        frame = ctk.CTkFrame(parent, fg_color="transparent")
        frame.grid(row=row, column=0, sticky="ew", padx=16, pady=10)
        frame.grid_columnconfigure(0 if stacked else 1, weight=1)
        ctk.CTkLabel(frame, text=label, anchor="w").grid(row=0, column=0, sticky="w")
        ctk.CTkEntry(frame, textvariable=variable).grid(
            row=1 if stacked else 0,
            column=0 if stacked else 1,
            sticky="ew",
            padx=0 if stacked else (16, 0),
            pady=(4, 0) if stacked else 0,
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

    def _add_microphone_row(self, parent, row):
        frame = ctk.CTkFrame(parent, fg_color="transparent")
        frame.grid(row=row, column=0, sticky="ew", padx=16, pady=10)
        frame.grid_columnconfigure(0, weight=1)
        ctk.CTkLabel(frame, text="Microphone", anchor="w").grid(
            row=0, column=0, sticky="w"
        )
        controls = ctk.CTkFrame(frame, fg_color="transparent")
        controls.grid(row=1, column=0, sticky="ew", pady=(4, 0))
        controls.grid_columnconfigure(0, weight=1)
        self.microphone_var = tk.StringVar(master=self.root)
        self.microphone_menu = ctk.CTkOptionMenu(
            controls,
            variable=self.microphone_var,
            values=["System default"],
            command=lambda _value: self._update_microphone_status(),
        )
        self.microphone_menu.grid(row=0, column=0, sticky="ew")
        ctk.CTkButton(
            controls,
            text="Refresh",
            width=90,
            command=lambda: self._refresh_microphones(refresh_backend=True),
        ).grid(row=0, column=1, padx=(8, 0))
        self.microphone_status = ctk.CTkLabel(
            frame,
            text="",
            anchor="w",
            justify="left",
            wraplength=580,
            text_color=("gray35", "gray70"),
        )
        self.microphone_status.grid(row=2, column=0, sticky="ew", pady=(5, 0))
        self._refresh_microphones(self.config.microphone, clear_error=False)
        self.root.after(1000, self._poll_microphone_status)

    def _poll_microphone_status(self):
        if self._closed:
            return
        if self._microphone_generation != input_backend.generation:
            self._refresh_microphones(clear_error=False)
        else:
            self._update_microphone_status()
        self.root.after(1000, self._poll_microphone_status)

    def _refresh_microphones(
        self, preference=..., *, clear_error=True, refresh_backend=False
    ):
        if preference is ... and hasattr(self, "_microphone_choices"):
            preference = self._selected_microphone()
        elif preference is ...:
            preference = self.config.microphone
        refresh_failed = False
        if refresh_backend:
            try:
                input_backend.request_refresh()
            except Exception as exc:
                refresh_failed = True
                input_status.set_error(f"Microphone refresh failed: {exc}")
        self._microphone_generation = input_backend.generation
        self._microphone_choices = {"System default": None}
        try:
            devices = list_input_devices()
        except Exception:
            devices = []
        duplicate_keys = {
            (device.name, device.hostapi)
            for device in devices
            if sum(
                other.name == device.name and other.hostapi == device.hostapi
                for other in devices
            )
            > 1
        }
        for device in devices:
            if (device.name, device.hostapi) in duplicate_keys:
                continue
            label = device.label
            self._microphone_choices[label] = device.preference
        preference = normalize_value(preference, SETTINGS_BY_KEY["microphone"])
        selected = next(
            (
                label
                for label, value in self._microphone_choices.items()
                if value == preference
            ),
            None,
        )
        if selected is None and preference is not None:
            reason = (
                "ambiguous"
                if (preference["name"], preference["hostapi"]) in duplicate_keys
                else "unavailable"
            )
            selected = f"{preference['name']} ({preference['hostapi']}) ({reason})"
            self._microphone_choices[selected] = preference
        selected = selected or "System default"
        self.microphone_menu.configure(values=list(self._microphone_choices))
        self.microphone_var.set(selected)
        if (
            clear_error
            and not refresh_failed
            and input_status.get_status().active is None
        ):
            input_status.clear_active()
        self._update_microphone_status()

    def _selected_microphone(self):
        return self._microphone_choices.get(self.microphone_var.get())

    def _update_microphone_status(self):
        preference = normalize_value(
            self.config.microphone, SETTINGS_BY_KEY["microphone"]
        )
        preferred = (
            "System default"
            if preference is None
            else f"{preference['name']} ({preference['hostapi']})"
        )
        pending = ""
        if self._selected_microphone() != preference:
            pending = f"\nSelected: {self.microphone_var.get()} (save to apply)"
        status = input_status.get_status()
        if status.active is not None:
            using = f"Using: {status.active.device.label}"
            if status.active.reason not in ("preferred", "system_default"):
                using += " (fallback)"
        elif status.error:
            using = f"Microphone error: {status.error}"
        else:
            try:
                selection = resolve_input_device(preference)
                using = f"Will use: {selection.device.label}"
                if selection.reason not in ("preferred", "system_default"):
                    using += " (fallback)"
            except Exception:
                using = "No microphone is available."
        self.microphone_status.configure(
            text=f"Preferred: {preferred}\n{using}{pending}"
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
        entry_var = tk.StringVar(master=self.root, value=str(current_value))
        slider_var = tk.DoubleVar(master=self.root, value=current_value)
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
                    parent=self.root,
                )
                return None
            variable.set(str(parsed_value))
            numeric_values[key] = parsed_value
        return numeric_values

    def _reset_tab_to_defaults(self, tab_name):
        for setting in settings_for_tab(tab_name):
            if setting.key == "microphone":
                self._refresh_microphones(None)
                continue
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
            messagebox.showinfo(_APP_DISPLAY_NAME, result.message, parent=self.root)
        else:
            messagebox.showerror(_APP_DISPLAY_NAME, result.message, parent=self.root)

    def _poll_ollama_connection_test_result(self):
        if not self._ollama_test_in_progress:
            return

        try:
            while True:
                window_ref, token, result = _ollama_connection_test_results.get_nowait()
                # The service permits only one settings window. If that changes,
                # use a per-window queue instead of discarding unmatched results.
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

    def _setup_statistics(self, parent):
        self._add_switch(
            parent,
            0,
            "Collect local statistics",
            self.statistics_var,
            "Counts and timings only, stored on this PC. No audio or dictated text.",
        )
        actions = ctk.CTkFrame(parent, fg_color="transparent")
        actions.grid(row=1, column=0, sticky="ew", padx=16, pady=(0, 4))
        actions.grid_columnconfigure(2, weight=1)
        refresh = ctk.CTkButton(
            actions, text="Refresh", width=96, command=self._refresh_statistics
        )
        refresh.grid(row=0, column=0, padx=(0, 8))
        export = ctk.CTkButton(
            actions,
            text="Export CSV",
            width=96,
            command=self._export_statistics,
            **_OUTLINE_BUTTON,
        )
        export.grid(row=0, column=1)
        delete = ctk.CTkButton(
            actions,
            text="Delete Session History",
            command=self._delete_statistics,
            **_DANGER_BUTTON,
        )
        delete.grid(row=0, column=2, sticky="e")
        self._statistics_buttons = [refresh, export, delete]
        self._statistics_status = ctk.CTkLabel(
            parent,
            text="",
            anchor="w",
            justify="left",
            text_color=_MUTED_TEXT,
            wraplength=620,
        )
        self._statistics_status.grid(row=2, column=0, sticky="ew", padx=16)
        self._statistics_body = ctk.CTkFrame(parent, fg_color="transparent")
        self._statistics_body.grid(row=3, column=0, sticky="ew", padx=16, pady=(0, 8))
        self._statistics_body.grid_columnconfigure(0, weight=1)
        self.root.after(0, self._refresh_statistics)

    def _render_statistics(self, view: StatisticsView):
        """Rebuild the Statistics body from a display-ready view."""
        body = self._statistics_body
        for child in body.winfo_children():
            child.destroy()
        row = 0
        for notice in view.notices:
            ctk.CTkLabel(
                body,
                text=notice,
                anchor="w",
                justify="left",
                fg_color=("#FFF4D6", "#3A2C13"),
                text_color=("#704400", "#FFD27F"),
                corner_radius=6,
                wraplength=600,
            ).grid(row=row, column=0, sticky="ew", pady=(0, 6), ipadx=8, ipady=4)
            row += 1
        if view.empty is not None:
            title, hint = view.empty
            ctk.CTkLabel(
                body, text=title, font=ctk.CTkFont(size=18, weight="bold")
            ).grid(row=row, column=0, pady=(32, 4))
            ctk.CTkLabel(body, text=hint, text_color=_MUTED_TEXT).grid(
                row=row + 1, column=0, pady=(0, 32)
            )
            return

        tiles = ctk.CTkFrame(body, fg_color="transparent")
        tiles.grid(row=row, column=0, sticky="ew", pady=(8, 4))
        for column, (value, label) in enumerate(view.tiles):
            tiles.grid_columnconfigure(column, weight=1, uniform="tile")
            tile = ctk.CTkFrame(tiles, fg_color=("gray80", "gray22"), corner_radius=8)
            tile.grid(
                row=0, column=column, sticky="ew", padx=(0 if column == 0 else 8, 0)
            )
            ctk.CTkLabel(
                tile, text=value, anchor="w", font=ctk.CTkFont(size=22, weight="bold")
            ).grid(row=0, column=0, sticky="w", padx=12, pady=(10, 0))
            ctk.CTkLabel(tile, text=label, anchor="w", text_color=_MUTED_TEXT).grid(
                row=1, column=0, sticky="w", padx=12, pady=(0, 10)
            )
        row += 1

        for title, rows in view.sections:
            ctk.CTkLabel(
                body,
                text=title.upper(),
                anchor="w",
                text_color=_MUTED_TEXT,
                font=ctk.CTkFont(size=12, weight="bold"),
            ).grid(row=row, column=0, sticky="w", pady=(14, 2))
            row += 1
            for label, value in rows:
                line = ctk.CTkFrame(body, fg_color="transparent")
                line.grid(row=row, column=0, sticky="ew")
                line.grid_columnconfigure(0, weight=1)
                ctk.CTkLabel(line, text=label, anchor="w").grid(
                    row=0, column=0, sticky="w"
                )
                ctk.CTkLabel(
                    line, text=value, anchor="e", font=ctk.CTkFont(weight="bold")
                ).grid(row=0, column=1, sticky="e", padx=(16, 0))
                row += 1
        for note in view.footnotes:
            ctk.CTkLabel(
                body,
                text=note,
                anchor="w",
                justify="left",
                text_color=_MUTED_TEXT,
                font=ctk.CTkFont(size=12),
                wraplength=600,
            ).grid(row=row, column=0, sticky="w", pady=(8 if row else 0, 0))
            row += 1

    def _statistics_request(self, operation, path=None):
        if getattr(self, "_closed", False) or self._statistics_token is not None:
            return
        token = threading.Event()
        self._statistics_token = token
        results = self._statistics_results
        self._statistics_status.configure(text="Statistics busy. Working...")
        for button in self._statistics_buttons:
            button.configure(state="disabled")

        def work():
            try:
                service = get_usage_stats()
                if operation == "summary":
                    result = service.summary().result()
                elif operation == "clear":
                    result = service.clear().result()
                else:
                    rows = service.export_rows().result()
                    result = len(rows)
                    if rows and not token.is_set():
                        _write_statistics_csv(path, rows)
                results.put((token, operation, result, None))
            except Exception as exc:
                results.put((token, operation, None, exc))

        threading.Thread(target=work, daemon=True).start()
        self.root.after(100, self._poll_statistics_result)

    def _poll_statistics_result(self):
        if getattr(self, "_closed", False):
            return
        while True:
            try:
                token, operation, result, error = self._statistics_results.get_nowait()
            except queue.Empty:
                break
            if token is not self._statistics_token or token.is_set():
                continue
            self._statistics_token = None
            for button in self._statistics_buttons:
                button.configure(state="normal")
            if error is not None:
                text = (
                    "Statistics busy. Try again."
                    if isinstance(error, StatisticsBusyError)
                    else "Statistics storage unavailable or operation failed. Try again."
                )
                self._statistics_status.configure(text=text)
                if operation != "summary":
                    messagebox.showerror(_APP_DISPLAY_NAME, text, parent=self.root)
            elif operation == "summary":
                self._statistics_status.configure(text="")
                self._render_statistics(_statistics_view(result))
            elif operation == "clear":
                messagebox.showinfo(
                    _APP_DISPLAY_NAME, "Session history deleted.", parent=self.root
                )
                self._refresh_statistics()
                return
            else:
                self._statistics_status.configure(
                    text="Export finished. Refresh to view history."
                )
                messagebox.showinfo(
                    _APP_DISPLAY_NAME,
                    f"Exported {result} session(s)."
                    if result
                    else "No statistics history to export. No file was written.",
                    parent=self.root,
                )
        if self._statistics_token is not None:
            self.root.after(100, self._poll_statistics_result)

    def _refresh_statistics(self):
        self._statistics_request("summary")

    def _export_statistics(self):
        if self._statistics_token is not None:
            return
        path = filedialog.asksaveasfilename(
            parent=self.root,
            title="Export Statistics",
            defaultextension=".csv",
            filetypes=[("CSV files", "*.csv")],
            initialfile="murmur-statistics.csv",
        )
        if path:
            self._statistics_request("export", path)

    def _delete_statistics(self):
        if self._statistics_token is not None:
            return
        if messagebox.askyesno(
            "Delete Session History",
            "Permanently delete all dictation session history?\n"
            "This cannot be undone.\n\n"
            "Removes:\n"
            "  •  Every recorded session\n"
            "  •  Stage timings and issue counts\n\n"
            "Keeps:\n"
            "  •  App versions Murmur has run (no times or session details)\n"
            "  •  Exported CSV files\n"
            "  •  Training data\n\n"
            "Sessions still in progress will not bring deleted history back.",
            parent=self.root,
            icon="warning",
            default="no",
        ):
            self._statistics_request("clear")

    def _purge_training_data(self):
        summary = self.logger.get_storage_summary()
        if summary.file_count == 0:
            messagebox.showinfo(
                _APP_DISPLAY_NAME,
                "No logged training data was found.",
                parent=self.root,
            )
            return

        if not messagebox.askyesno(
            "Delete logged data",
            (
                "Delete all stored WAV files and transcription logs from the local "
                f"training data folder?\n\nFiles: {summary.file_count}\n"
                f"Approximate size: {_format_bytes(summary.total_bytes)}"
            ),
            parent=self.root,
        ):
            return

        try:
            removed_files = self.logger.purge_all()
        except Exception as exc:
            messagebox.showerror(
                _APP_DISPLAY_NAME,
                f"Failed to delete logged data: {exc}",
                parent=self.root,
            )
            return

        if removed_files:
            messagebox.showinfo(
                _APP_DISPLAY_NAME,
                f"Deleted {removed_files} logged file(s) from {get_training_data_dir()}.",
                parent=self.root,
            )
        else:
            messagebox.showinfo(
                _APP_DISPLAY_NAME,
                "No logged training data was found.",
                parent=self.root,
            )

    def _save(self):
        new_hotkey = self.hotkey_var.get().strip()
        if not is_hotkey_valid(new_hotkey):
            messagebox.showerror(
                _APP_DISPLAY_NAME,
                "Please enter a valid hotkey before saving settings.",
                parent=self.root,
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
                parent=self.root,
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
                parent=self.root,
            )
            return

        old_overlay_enabled = self.config.show_recording_overlay
        new_autostart = self.autostart_var.get()
        updated_values = {
            "hotkey": new_hotkey,
            "model": normalize_value(self.model_var.get(), SETTINGS_BY_KEY["model"]),
            "device": normalize_value(self.device_var.get(), SETTINGS_BY_KEY["device"]),
            "microphone": self._selected_microphone()
            if hasattr(self, "_microphone_choices")
            else self.config.get("microphone"),
            "language": lang,
            "enable_notifications": self.notify_var.get(),
            "show_recording_overlay": self.overlay_var.get(),
            "enable_logging": new_logging,
            "start_with_windows": new_autostart,
            "pause_media_while_recording": self.pause_media_var.get(),
            "keep_microphone_ready": self.keep_microphone_ready_var.get(),
            "ollama_enabled": new_ollama_enabled,
            "ollama_endpoint": new_ollama_endpoint,
            "ollama_model_name": new_ollama_model_name,
            "ollama_preload_model": self.ollama_preload_model_var.get(),
        }
        updated_values.update(numeric_values)
        statistics_var = getattr(self, "statistics_var", None)
        if statistics_var is not None:
            updated_values["enable_statistics"] = statistics_var.get()

        restart_settings = self._changed_restart_settings(updated_values)

        if previous_logging != new_logging:
            updated_values["logging_consent_updated_at"] = (
                datetime.now().astimezone().isoformat()
            )
            updated_values["logging_consent_source"] = "settings"

        try:
            previous_autostart = is_autostart_enabled()
        except OSError as exc:
            messagebox.showerror(
                _APP_DISPLAY_NAME,
                f"Settings were not saved. Could not check Start with Windows: {exc}",
                parent=self.root,
            )
            return

        # Reconcile even an unchanged choice, including stale disabled entries.
        autostart_needs_write = new_autostart or previous_autostart
        try:
            if autostart_needs_write:
                set_autostart(new_autostart)
        except OSError as exc:
            self.autostart_var.set(previous_autostart)
            messagebox.showerror(
                _APP_DISPLAY_NAME,
                f"Settings were not saved. Could not update Start with Windows: {exc}. "
                "The switch now shows the current Windows startup state.",
                parent=self.root,
            )
            return

        try:
            self.config.update(updated_values)
        except ConfigError as exc:
            rollback_error = ""
            try:
                if autostart_needs_write:
                    set_autostart(previous_autostart)
                self.autostart_var.set(previous_autostart)
            except OSError as rollback_exc:
                rollback_error = (
                    " Start with Windows could not be restored and still reflects "
                    f"your requested choice: {rollback_exc}. Retry saving settings."
                )
            messagebox.showerror(
                _APP_DISPLAY_NAME,
                f"Failed to save settings: {exc}.{rollback_error}",
                parent=self.root,
            )
            return

        if statistics_var is not None:
            try:
                get_usage_stats().set_enabled(updated_values["enable_statistics"])
            except Exception:
                messagebox.showwarning(
                    _APP_DISPLAY_NAME,
                    "Settings saved, but statistics collection could not be updated. "
                    "Restart murmur to apply this setting.",
                    parent=self.root,
                )

        if old_overlay_enabled and not self.overlay_var.get():
            close_overlay()

        self.logger.set_enabled(new_logging)

        if restart_settings:
            messagebox.showinfo(
                _APP_DISPLAY_NAME,
                "Settings saved. Restart murmur for these changes to fully apply: "
                + ", ".join(restart_settings),
                parent=self.root,
            )
        else:
            messagebox.showinfo(
                _APP_DISPLAY_NAME,
                "Settings saved.",
                parent=self.root,
            )
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
        statistics_token = getattr(self, "_statistics_token", None)
        if statistics_token is not None:
            statistics_token.set()
        self._statistics_token = None
        window_icon = getattr(self, "_window_icon", None)
        try:
            if window_icon is not None:
                _restore_native_window_class_icons(
                    self.root, window_icon.get("class_before")
                )
            self.root.destroy()
        finally:
            try:
                if window_icon is not None:
                    _destroy_native_icons(window_icon.get("native"))
                self._window_icon = None
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
    _settings_requests.put(_SHOW_SETTINGS_REQUEST)


def request_overlay_state(state, session=None, message=""):
    """Queue a recording overlay state without blocking the caller.

    Safe to call from any thread. Failures stay inside the overlay so capture
    and finalization continue unaffected.
    """
    if _overlay_unavailable:
        return
    try:
        _ensure_settings_ui_thread(return_focus=True)
        _settings_requests.put((_OVERLAY_REQUEST, state, session, message))
    except Exception as exc:
        _report_overlay_unavailable(exc)


def close_overlay():
    """Hide and release the overlay window if the UI thread is running."""
    with _settings_thread_lock:
        if _settings_thread is None or not _settings_thread.is_alive():
            return
        _settings_requests.put(_CLOSE_OVERLAY_REQUEST)
