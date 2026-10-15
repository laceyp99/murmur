"""Build and check a non-editable install without launching the desktop app."""

import os
import subprocess
import sys
import tempfile
import venv
from pathlib import Path
from zipfile import ZipFile

INSTALL_CHECK = """
import os
import runpy
import sys
from importlib.metadata import distribution
from pathlib import Path
from types import ModuleType

import src
from PIL import Image
from src.assets import get_app_icon_path, get_logo_path
from src.autostart import _get_launch_command

assert Path(src.__file__).is_relative_to(sys.prefix), src.__file__
assert not Path('run.py').exists()
logo = get_logo_path()
assert logo is not None and logo.is_relative_to(sys.prefix), logo
with Image.open(logo) as image:
    image.verify()
icon = get_app_icon_path()
assert icon is not None and icon.is_file(), icon
with Image.open(icon) as image:
    assert image.format == 'ICO'

executable = Path(sys.executable)
if executable.name.lower() == 'python.exe':
    executable = executable.with_name('pythonw.exe')
assert executable.is_file(), executable
assert _get_launch_command() == f'"{executable}" -m src'

# Exercise installed launch dispatch without hotkeys, microphone or registry writes.
runtime = ModuleType('src.main')
calls = []
def main():
    assert sys.stdout is not None and sys.stderr is not None
    print('windowless output')
    sys.stderr.flush()
    calls.append(True)
runtime.main = main
sys.modules['src.main'] = runtime
stdout, stderr = sys.stdout, sys.stderr
sys.stdout = sys.stderr = None
try:
    runpy.run_module('src', run_name='__main__')
    assert sys.stdout.name == os.devnull and sys.stderr.name == os.devnull
finally:
    for stream in (sys.stdout, sys.stderr):
        if stream is not None:
            stream.close()
    sys.stdout, sys.stderr = stdout, stderr
assert calls == [True]
entrypoint = next(ep for ep in distribution('murmur').entry_points
                  if ep.group == 'console_scripts' and ep.name == 'murmur')
entrypoint.load()()
assert calls == [True, True]
print('Installed module, autostart command, console entrypoint and icons passed')
"""


def main() -> None:
    repo_root = Path(__file__).resolve().parent.parent
    with tempfile.TemporaryDirectory(prefix="murmur-wheel-") as directory:
        work_dir = Path(directory)
        wheel_dir = work_dir / "wheels"
        subprocess.run(
            [
                sys.executable,
                "-m",
                "pip",
                "wheel",
                "--no-deps",
                "--wheel-dir",
                str(wheel_dir),
                str(repo_root),
            ],
            cwd=work_dir,
            check=True,
        )
        (wheel,) = wheel_dir.glob("murmur-*.whl")
        with ZipFile(wheel) as archive:
            required = {"src/__main__.py", "src/launch.py", "src/resources/icon.png"}
            missing = required.difference(archive.namelist())
            if missing:
                raise RuntimeError(f"Wheel is missing runtime files: {sorted(missing)}")

        env_dir = work_dir / "venv"
        venv.EnvBuilder(with_pip=True).create(env_dir)
        python = env_dir / ("Scripts/python.exe" if os.name == "nt" else "bin/python")
        # Pillow is the only dependency needed to check icons and launch dispatch.
        subprocess.run(
            [str(python), "-m", "pip", "install", "--no-deps", str(wheel), "Pillow"],
            cwd=work_dir,
            check=True,
        )
        subprocess.run(
            [str(python), "-I", "-c", INSTALL_CHECK],
            cwd=work_dir,
            env={**os.environ, "APPDATA": str(work_dir / "app-data")},
            check=True,
        )


if __name__ == "__main__":
    main()
