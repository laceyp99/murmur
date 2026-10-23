#!/usr/bin/env python
"""
Murmur - Local Speech-to-Text Hotkey App
Run this script to start Murmur.
"""

import os
import sys
import traceback
from pathlib import Path

# Add the project root to the path
sys.path.insert(0, str(Path(__file__).resolve().parent))

from src.launch import ensure_output_streams as _ensure_output_streams


def _run_app() -> None:
    """Import and start the desktop runtime after handling build-only commands."""
    from src.build_info import get_build_info

    get_build_info()
    from src.main import main

    main()


def _run_packaging_self_check() -> None:
    """Exercise packaged dependencies without starting the desktop runtime."""
    import sqlite3
    from contextlib import closing

    with closing(sqlite3.connect(":memory:")) as connection:
        with connection:
            connection.execute("CREATE TABLE packaging_check (value INTEGER)")
            connection.execute("INSERT INTO packaging_check VALUES (1)")
        if connection.execute("SELECT value FROM packaging_check").fetchone() != (1,):
            raise RuntimeError("SQLite transaction is unavailable")

    import numpy as np
    import torch
    import whisper
    import winrt.windows.media.control  # noqa: F401

    from src.assets import get_logo_path
    from src.main import main as app_main

    if not callable(app_main):
        raise RuntimeError("Murmur desktop runtime is unavailable")

    audio = np.zeros(16000, dtype=np.float32)
    mel = whisper.log_mel_spectrogram(audio)
    if not isinstance(mel, torch.Tensor) or mel.shape[-1] == 0:
        raise RuntimeError("Whisper audio processing is unavailable")

    dimensions = whisper.model.ModelDimensions(
        n_mels=80,
        n_audio_ctx=10,
        n_audio_state=8,
        n_audio_head=2,
        n_audio_layer=1,
        n_vocab=100,
        n_text_ctx=10,
        n_text_state=8,
        n_text_head=2,
        n_text_layer=1,
    )
    model = whisper.model.Whisper(dimensions)
    if next(model.parameters(), None) is None:
        raise RuntimeError("Whisper model initialization is unavailable")
    if get_logo_path() is None:
        raise RuntimeError("Packaged Murmur logo is unavailable")

    # Opt in to a real first-download check without touching the user's cache.
    if model_cache := os.environ.get("MURMUR_PACKAGING_SELF_CHECK_MODEL_CACHE"):
        whisper.load_model("tiny", device="cpu", download_root=model_cache)


def _packaging_self_check_exit_code() -> int:
    try:
        _run_packaging_self_check()
    except Exception:
        if error_path := os.environ.get("MURMUR_PACKAGING_SELF_CHECK_LOG"):
            Path(error_path).write_text(traceback.format_exc(), encoding="utf-8")
        return 1
    return 0


if __name__ == "__main__":
    _ensure_output_streams()
    if "--packaging-self-check" in sys.argv:
        raise SystemExit(_packaging_self_check_exit_code())
    else:
        _run_app()
