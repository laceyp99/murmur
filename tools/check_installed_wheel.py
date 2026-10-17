"""Check installed runtime files and dispatch; invoked by check_wheel_install.py."""

import os
import runpy
import sys
from importlib.metadata import distribution
from pathlib import Path
from types import ModuleType

from PIL import Image

import src
from src.assets import get_app_icon_path, get_logo_path
from src.autostart import _get_launch_command


def main() -> None:
    """Verify the installed wheel without starting the real desktop runtime."""
    assert Path(src.__file__).is_relative_to(sys.prefix), src.__file__
    assert not Path("run.py").exists()
    logo = get_logo_path()
    assert logo is not None
    assert logo.is_relative_to(sys.prefix), logo
    with Image.open(logo) as image:
        image.verify()
    icon = get_app_icon_path()
    assert icon is not None
    assert icon.is_file(), icon
    with Image.open(icon) as image:
        assert image.format == "ICO"

    executable = Path(sys.executable)
    if executable.name.lower() == "python.exe":
        executable = executable.with_name("pythonw.exe")
    assert executable.is_file(), executable
    assert _get_launch_command() == f'"{executable}" -m src'

    # Substitute the desktop runtime to avoid hotkeys, microphone or registry writes.
    runtime = ModuleType("src.main")
    calls: list[bool] = []

    def record_launch() -> None:
        assert sys.stdout is not None
        assert sys.stderr is not None
        print("windowless output")
        sys.stderr.flush()
        calls.append(True)

    runtime.main = record_launch
    sys.modules["src.main"] = runtime
    stdout, stderr = sys.stdout, sys.stderr
    sys.stdout = sys.stderr = None
    try:
        runpy.run_module("src", run_name="__main__")
        assert sys.stdout.name == os.devnull
        assert sys.stderr.name == os.devnull
    finally:
        for stream in (sys.stdout, sys.stderr):
            if stream is not None:
                stream.close()
        sys.stdout, sys.stderr = stdout, stderr
    assert calls == [True]
    entrypoint = next(
        ep
        for ep in distribution("murmur").entry_points
        if ep.group == "console_scripts" and ep.name == "murmur"
    )
    entrypoint.load()()
    assert calls == [True, True]
    print("Installed module, autostart command, console entrypoint and icons passed")


if __name__ == "__main__":
    main()
