"""Windows process identity configuration shared by app entry points."""

import ctypes

_WINDOWS_APP_USER_MODEL_ID = "murmur"
_WINDOWS_APP_ID_SET = False


def configure_windows_app_identity() -> None:
    """Set Murmur's Windows identity once, before it creates any UI."""
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
