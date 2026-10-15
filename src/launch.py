"""Shared initialization for source and installed windowless launchers."""

import os
import sys
from pathlib import Path


def ensure_output_streams() -> None:
    """Discard windowless progress output without writing user data to a log."""
    # Like normal stdio, these streams stay open for the process lifetime.
    if sys.stdout is None:
        sys.stdout = Path(os.devnull).open("w", encoding="utf-8")  # noqa: SIM115
    if sys.stderr is None:
        sys.stderr = Path(os.devnull).open("w", encoding="utf-8")  # noqa: SIM115
