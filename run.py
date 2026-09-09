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

from src.main import main


def _run_packaging_self_check() -> None:
    """Exercise packaged dependencies without starting the desktop runtime."""
    import numpy as np
    import torch
    import whisper
    import winrt.windows.media.control  # noqa: F401

    from src.assets import get_logo_path

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


def _packaging_self_check_exit_code() -> int:
    try:
        _run_packaging_self_check()
    except Exception:
        if error_path := os.environ.get("MURMUR_PACKAGING_SELF_CHECK_LOG"):
            Path(error_path).write_text(traceback.format_exc(), encoding="utf-8")
        return 1
    return 0


if __name__ == "__main__":
    if "--packaging-self-check" in sys.argv:
        raise SystemExit(_packaging_self_check_exit_code())
    else:
        main()
