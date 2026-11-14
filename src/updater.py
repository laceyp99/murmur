"""Check GitHub Releases for a newer packaged murmur and stage it with Velopack.

The settings window starts a check with start_check() and polls get_state().
Once a release is downloaded, the tray offers "Restart to update", which calls
schedule_update() and then quits through the normal shutdown path.

Only installs made by the Velopack Setup can update; source checkouts and
manually copied builds report UNAVAILABLE. The public repo URL is the only
input sent anywhere: no token, headers, config, or user data. Keep this module
free of imports from the rest of murmur so that stays easy to verify.
"""

import enum
import threading
from collections.abc import Callable

import velopack

RELEASES_REPO_URL = "https://github.com/laceyp99/murmur"


class UpdateState(enum.Enum):
    UNAVAILABLE = "unavailable"
    IDLE = "idle"
    CHECKING = "checking"
    UP_TO_DATE = "up_to_date"
    READY = "ready"
    FAILED = "failed"


_lock = threading.Lock()
_state: UpdateState | None = None
_pending: velopack.UpdateInfo | None = None
_ready_listener: Callable[[], None] | None = None


def _manager() -> velopack.UpdateManager | None:
    """Return a manager for a Velopack install, or None when not installed."""
    try:
        # Always pass the source explicitly; omitting it targets Velopack's
        # hosted service instead of GitHub.
        return velopack.UpdateManager(velopack.GithubSource(RELEASES_REPO_URL))
    except Exception:
        return None


def get_state() -> UpdateState:
    """Return the current update state, probing the install on first call."""
    global _state
    with _lock:
        if _state is None:
            _state = (
                UpdateState.IDLE if _manager() is not None else UpdateState.UNAVAILABLE
            )
        return _state


def pending_update() -> velopack.UpdateInfo | None:
    """Return the downloaded release waiting for a restart, if any."""
    with _lock:
        return _pending


def set_ready_listener(listener: Callable[[], None] | None) -> None:
    """Register a callback run on the worker thread once a download finishes."""
    global _ready_listener
    with _lock:
        _ready_listener = listener


def start_check() -> bool:
    """Check and download in the background; return False if one cannot start.

    Network or feed errors end in FAILED, so callers can treat them as
    "no update" without blocking startup or dictation.
    """
    global _state
    if get_state() in (
        UpdateState.UNAVAILABLE,
        UpdateState.CHECKING,
        UpdateState.READY,
    ):
        return False
    with _lock:
        _state = UpdateState.CHECKING
    threading.Thread(
        target=_check_and_download, name="murmur-update", daemon=True
    ).start()
    return True


def _check_and_download() -> None:
    global _state, _pending
    listener = None
    try:
        manager = _manager()
        info = manager.check_for_updates() if manager is not None else None
        if info is not None:
            manager.download_updates(info)
        with _lock:
            _pending = info
            _state = UpdateState.READY if info is not None else UpdateState.UP_TO_DATE
            listener = _ready_listener if info is not None else None
    except Exception as exc:
        print(f"Update check failed ({type(exc).__name__}).")
        with _lock:
            _state = UpdateState.FAILED
    if listener is not None:
        try:
            listener()
        except Exception as exc:
            print(f"Update ready notification failed ({type(exc).__name__}).")


def schedule_update(info: velopack.UpdateInfo) -> None:
    """Have Update.exe swap in the release and relaunch once murmur exits.

    Callers must only invoke this between sessions, then quit through the
    normal shutdown path within 60 seconds, after which Update.exe kills the
    process. The apply_updates_and_* methods are avoided because they call
    exit(0) from Rust, skipping murmur's shutdown cleanup.
    """
    manager = _manager()
    if manager is None:
        raise RuntimeError("murmur is not a Velopack install")
    manager.wait_exit_then_apply_updates(info, silent=True, restart=True)
