"""Load and save user-specific vocabulary overrides for transcript cleanup."""

from __future__ import annotations

import json
import sys
from collections.abc import Mapping
from pathlib import Path

from .config import get_app_data_dir

DEFAULT_USER_VOCAB_PATH = Path(__file__).resolve().parents[1] / "user_vocab.json"


def _get_default_user_vocab_path() -> Path:
    """Return a writable vocabulary path for the current launch mode."""
    if getattr(sys, "frozen", False):
        return get_app_data_dir() / "user_vocab.json"
    return DEFAULT_USER_VOCAB_PATH


def load_user_vocab(path: Path | None = None) -> dict[str, str]:
    """Return user vocabulary overrides from disk, or an empty mapping."""
    vocab_path = Path(path) if path is not None else _get_default_user_vocab_path()
    if not vocab_path.exists():
        return {}

    try:
        raw_data = json.loads(vocab_path.read_text(encoding="utf-8"))
    except (json.JSONDecodeError, OSError) as exc:
        print(f"⚠️ Failed to load user vocabulary from '{vocab_path}': {exc}")
        return {}

    if not isinstance(raw_data, dict):
        print(f"⚠️ User vocabulary file '{vocab_path}' must contain a JSON object.")
        return {}

    vocab: dict[str, str] = {}
    for source, target in raw_data.items():
        if source is None or target is None:
            continue
        vocab[str(source)] = str(target)

    return vocab


def save_user_vocab(vocab: Mapping[str, str], path: Path | None = None) -> None:
    """Persist user vocabulary overrides to disk."""
    vocab_path = Path(path) if path is not None else _get_default_user_vocab_path()
    vocab_path.parent.mkdir(parents=True, exist_ok=True)
    normalized_vocab = {str(source): str(target) for source, target in vocab.items()}

    try:
        vocab_path.write_text(
            json.dumps(normalized_vocab, indent=2, sort_keys=True) + "\n",
            encoding="utf-8",
        )
    except OSError as exc:
        print(f"⚠️ Failed to save user vocabulary to '{vocab_path}': {exc}")
