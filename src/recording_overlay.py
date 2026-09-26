"""
Passive on-screen recording status overlay for murmur.

Everything here runs on the shared CustomTkinter UI thread owned by
``settings_gui``. The overlay only reflects session state: it never reads audio
levels or transcript text, never takes focus, and passes clicks through.
"""

import contextlib
import ctypes
import math
import sys
import time
import tkinter as tk
from ctypes import wintypes
from tkinter import font as tkfont

HIDDEN = "hidden"
RECORDING = "recording"
PROCESSING = "processing"
SUCCESS = "success"
ERROR = "error"

SUCCESS_MESSAGE = "Copied to clipboard"
SUCCESS_HIDE_MS = 2000
ERROR_HIDE_MS = 4000
ANIMATION_FRAME_MS = 40

# A session's stages only move forward, so a late request cannot rewind it.
_SESSION_STAGES = {RECORDING: 0, PROCESSING: 1, SUCCESS: 2, ERROR: 2}
_ANIMATED_STATES = frozenset({RECORDING, PROCESSING})

_TRANSPARENT_KEY = "#010203"
_PILL_COLOR = "#1c1c1e"
_ICON_COLOR = "#ffffff"
_TRACK_COLOR = "#4a4a4d"
_SUCCESS_COLOR = "#4ade80"
_ERROR_COLOR = "#fbbf24"

# Logical pixels at 96 DPI; scaled by the display DPI at draw time.
_PILL_HEIGHT = 36
_ICON_PILL_WIDTH = 64
_BOTTOM_MARGIN = 16
_TEXT_PADDING = 14
_TEXT_ICON_SIZE = 16
_TEXT_ICON_GAP = 8

_GWL_EXSTYLE = -20
_WS_EX_TRANSPARENT = 0x00000020
_WS_EX_TOOLWINDOW = 0x00000080
_WS_EX_LAYERED = 0x00080000
_WS_EX_NOACTIVATE = 0x08000000
_HWND_TOPMOST = -1
_SWP_NOSIZE = 0x0001
_SWP_NOMOVE = 0x0002
_SWP_NOACTIVATE = 0x0010
_SWP_SHOWWINDOW = 0x0040
_SW_HIDE = 0
_SPI_GETWORKAREA = 0x0030


class RecordingOverlay:
    """Apply overlay state requests in order and own the overlay's timers.

    ``master`` is the Tk root that schedules timers. The window is created on
    the first visible request, so an unused overlay costs nothing. Any UI error
    disables the overlay for the rest of the process after one diagnostic.
    """

    def __init__(self, master, view_factory=None, clock=time.monotonic):
        self._master = master
        self._view_factory = view_factory or OverlayWindow
        self._clock = clock
        self._view = None
        self._failed = False
        self.state = HIDDEN
        self._session = None
        self._stage = -1
        self._message = ""
        self._started_at = 0.0
        self._frame_job = None
        self._hide_job = None

    def request(self, state, session=None, message=""):
        """Show ``state`` for ``session`` unless a newer session or stage owns the overlay.

        ``HIDDEN`` always applies. Returns whether the request was applied.
        """
        if self._failed:
            return False
        try:
            return self._apply(state, session, message)
        except Exception as exc:
            self._fail(exc)
            return False

    def close(self):
        """Hide the overlay and release its window."""
        with contextlib.suppress(Exception):
            self._cancel_jobs()
        view, self._view = self._view, None
        self.state = HIDDEN
        if view is not None:
            with contextlib.suppress(Exception):
                view.destroy()

    def _apply(self, state, session, message):
        if state == HIDDEN:
            self._cancel_jobs()
            self._hide()
            return True

        stage = _SESSION_STAGES.get(state)
        if stage is None:
            return False
        if self._session is not None and session is not None:
            if session < self._session:
                return False
            if session == self._session and stage < self._stage:
                return False

        self._cancel_jobs()
        if self._view is None:
            self._view = self._view_factory(self._master)
        self._session = session
        self._stage = stage
        self.state = state
        self._message = SUCCESS_MESSAGE if state == SUCCESS else message
        self._started_at = self._clock()
        self._render()

        if state in _ANIMATED_STATES:
            self._frame_job = self._master.after(ANIMATION_FRAME_MS, self._on_frame)
        else:
            delay = SUCCESS_HIDE_MS if state == SUCCESS else ERROR_HIDE_MS
            self._hide_job = self._master.after(delay, self._on_hide_timeout)
        return True

    def _render(self):
        self._view.show(self.state, self._message, self._clock() - self._started_at)

    def _hide(self):
        self.state = HIDDEN
        if self._view is not None:
            self._view.hide()

    def _on_frame(self):
        self._frame_job = None
        if self._failed or self.state not in _ANIMATED_STATES:
            return
        try:
            self._render()
            self._frame_job = self._master.after(ANIMATION_FRAME_MS, self._on_frame)
        except Exception as exc:
            self._fail(exc)

    def _on_hide_timeout(self):
        self._hide_job = None
        if self._failed:
            return
        try:
            self._hide()
        except Exception as exc:
            self._fail(exc)

    def _cancel_jobs(self):
        for name in ("_frame_job", "_hide_job"):
            job = getattr(self, name)
            setattr(self, name, None)
            if job is not None:
                self._master.after_cancel(job)

    def _fail(self, exc):
        if self._failed:
            return
        self._failed = True
        # Exception text could echo UI internals; the type is enough to diagnose.
        print(
            f"Recording overlay disabled after a UI error ({type(exc).__name__}).",
            file=sys.stderr,
        )
        self.close()


class OverlayWindow:
    """Borderless, topmost, click-through pill above the primary work area."""

    def __init__(self, master):
        self._window = window = tk.Toplevel(master)
        window.withdraw()
        window.overrideredirect(True)
        window.attributes("-topmost", True)
        window.configure(bg=_TRANSPARENT_KEY)
        if sys.platform == "win32":
            window.attributes("-transparentcolor", _TRANSPARENT_KEY)

        self._scale = window.winfo_fpixels("1i") / 96
        self._font = tkfont.Font(root=window, family="Segoe UI", size=10)
        self._canvas = tk.Canvas(
            window, bg=_TRANSPARENT_KEY, highlightthickness=0, borderwidth=0
        )
        self._canvas.pack(fill="both", expand=True)
        window.update_idletasks()
        self._hwnd = _make_passive(window)
        self._geometry = None
        self._visible = False

    def show(self, state, message, elapsed):
        text_width = self._font.measure(message) if message else 0
        width, height = overlay_size(state, text_width, self._scale)
        self._draw(state, message, elapsed, width, height)

        if not self._visible or self._geometry[2:] != (width, height):
            x, y = overlay_position(
                _primary_work_area(self._window), (width, height), self._scale
            )
            self._geometry = (x, y, width, height)
            self._canvas.configure(width=width, height=height)
            self._window.geometry(f"{width}x{height}+{x}+{y}")

        if not self._visible:
            self._show_window()
            self._visible = True

    def hide(self):
        if not self._visible:
            return
        self._visible = False
        if self._hwnd is not None:
            ctypes.windll.user32.ShowWindow(self._hwnd, _SW_HIDE)
        else:
            self._window.withdraw()

    def destroy(self):
        self._visible = False
        self._window.destroy()

    def _show_window(self):
        if self._hwnd is None:
            self._window.deiconify()
            return
        # Tk's deiconify may activate the window; show it natively without focus.
        self._window.update_idletasks()
        ctypes.windll.user32.SetWindowPos(
            self._hwnd,
            _HWND_TOPMOST,
            0,
            0,
            0,
            0,
            _SWP_NOMOVE | _SWP_NOSIZE | _SWP_NOACTIVATE | _SWP_SHOWWINDOW,
        )

    def _draw(self, state, message, elapsed, width, height):
        canvas, scale = self._canvas, self._scale
        canvas.delete("all")
        _draw_pill(canvas, width, height)
        center_y = height / 2

        if state == RECORDING:
            _draw_bars(canvas, width / 2, center_y, elapsed, scale)
        elif state == PROCESSING:
            _draw_spinner(canvas, width / 2, center_y, elapsed, scale)
        else:
            icon_left = _TEXT_PADDING * scale
            if state == SUCCESS:
                _draw_checkmark(canvas, icon_left, center_y, scale)
            else:
                _draw_warning(canvas, icon_left, center_y, scale)
            canvas.create_text(
                icon_left + (_TEXT_ICON_SIZE + _TEXT_ICON_GAP) * scale,
                center_y,
                text=message,
                anchor="w",
                fill=_ICON_COLOR,
                font=self._font,
            )


def overlay_size(state, text_width, scale):
    """Return the pill's physical size; only text states widen it."""
    height = round(_PILL_HEIGHT * scale)
    if state in _ANIMATED_STATES:
        return round(_ICON_PILL_WIDTH * scale), height
    chrome = (2 * _TEXT_PADDING + _TEXT_ICON_SIZE + _TEXT_ICON_GAP) * scale
    return round(chrome + text_width), height


def overlay_position(work_area, size, scale):
    """Center the pill horizontally just above the work area's bottom edge."""
    left, _top, right, bottom = work_area
    width, height = size
    x = left + (right - left - width) // 2
    y = bottom - height - round(_BOTTOM_MARGIN * scale)
    return x, y


def _primary_work_area(window):
    """Return the primary display's taskbar-free area in physical pixels."""
    if sys.platform == "win32":
        with contextlib.suppress(AttributeError, OSError):
            rect = wintypes.RECT()
            if ctypes.windll.user32.SystemParametersInfoW(
                _SPI_GETWORKAREA, 0, ctypes.byref(rect), 0
            ):
                return rect.left, rect.top, rect.right, rect.bottom
    return 0, 0, window.winfo_screenwidth(), window.winfo_screenheight()


def _make_passive(window):
    """Make the window click-through and non-activating; return its HWND."""
    if sys.platform != "win32":
        return None
    try:
        user32 = ctypes.windll.user32
        user32.GetWindowLongPtrW.argtypes = [wintypes.HWND, ctypes.c_int]
        user32.GetWindowLongPtrW.restype = ctypes.c_ssize_t
        user32.SetWindowLongPtrW.argtypes = [
            wintypes.HWND,
            ctypes.c_int,
            ctypes.c_ssize_t,
        ]
        user32.SetWindowLongPtrW.restype = ctypes.c_ssize_t
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
        user32.ShowWindow.argtypes = [wintypes.HWND, ctypes.c_int]
        user32.ShowWindow.restype = wintypes.BOOL

        # The Tk wrapper frame is the real top-level window.
        hwnd = int(window.wm_frame(), 16)
        style = user32.GetWindowLongPtrW(hwnd, _GWL_EXSTYLE)
        user32.SetWindowLongPtrW(
            hwnd,
            _GWL_EXSTYLE,
            style
            | _WS_EX_LAYERED
            | _WS_EX_TRANSPARENT
            | _WS_EX_NOACTIVATE
            | _WS_EX_TOOLWINDOW,
        )
        return hwnd
    except (AttributeError, OSError, ValueError, tk.TclError):
        return None


def _draw_pill(canvas, width, height):
    options = {"fill": _PILL_COLOR, "outline": ""}
    right = width - 1
    bottom = height - 1
    canvas.create_oval(0, 0, bottom, bottom, **options)
    canvas.create_oval(right - bottom, 0, right, bottom, **options)
    canvas.create_rectangle(height / 2, 0, width - height / 2, height, **options)


def _draw_bars(canvas, center_x, center_y, elapsed, scale):
    """Three rounded bars with a fixed gentle pulse, not driven by audio."""
    bar_width = 4 * scale
    spacing = 9 * scale
    for index in range(3):
        phase = elapsed * 2 * math.pi / 1.2 - index * 0.8
        level = 0.5 + 0.5 * math.sin(phase)
        length = (6 + 12 * level) * scale
        # Round caps add half the bar width at each end.
        half = max(0.0, (length - bar_width) / 2)
        x = center_x + (index - 1) * spacing
        canvas.create_line(
            x,
            center_y - half,
            x,
            center_y + half,
            width=bar_width,
            capstyle=tk.ROUND,
            fill=_ICON_COLOR,
        )


def _draw_spinner(canvas, center_x, center_y, elapsed, scale):
    radius = 8 * scale
    box = (
        center_x - radius,
        center_y - radius,
        center_x + radius,
        center_y + radius,
    )
    line_width = 2.5 * scale
    canvas.create_oval(*box, outline=_TRACK_COLOR, width=line_width)
    canvas.create_arc(
        *box,
        start=-(elapsed * 400) % 360,
        extent=100,
        style=tk.ARC,
        outline=_ICON_COLOR,
        width=line_width,
    )


def _draw_checkmark(canvas, left, center_y, scale):
    canvas.create_line(
        left + 1 * scale,
        center_y,
        left + 6 * scale,
        center_y + 5 * scale,
        left + 15 * scale,
        center_y - 5 * scale,
        width=2.5 * scale,
        capstyle=tk.ROUND,
        joinstyle=tk.ROUND,
        fill=_SUCCESS_COLOR,
    )


def _draw_warning(canvas, left, center_y, scale):
    canvas.create_polygon(
        left + 8 * scale,
        center_y - 7 * scale,
        left + 16 * scale,
        center_y + 7 * scale,
        left,
        center_y + 7 * scale,
        fill=_ERROR_COLOR,
        outline=_ERROR_COLOR,
        joinstyle=tk.ROUND,
        width=1.5 * scale,
    )
    mark_x = left + 8 * scale
    canvas.create_line(
        mark_x,
        center_y - 2.5 * scale,
        mark_x,
        center_y + 2 * scale,
        width=2 * scale,
        capstyle=tk.ROUND,
        fill=_PILL_COLOR,
    )
    dot = 1.1 * scale
    canvas.create_oval(
        mark_x - dot,
        center_y + 4.5 * scale - dot,
        mark_x + dot,
        center_y + 4.5 * scale + dot,
        fill=_PILL_COLOR,
        outline="",
    )
