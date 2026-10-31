import sys
from pathlib import Path

import pytest

PROJECT_ROOT = Path(__file__).resolve().parents[1]

if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))


@pytest.fixture(autouse=True)
def isolate_autostart_registry(monkeypatch):
    """Require tests to opt into a fake registry rather than changing Windows."""
    monkeypatch.setattr("src.autostart.winreg", None)
