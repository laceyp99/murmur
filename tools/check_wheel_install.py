"""Build and check a non-editable install without launching the desktop app."""

import os
import subprocess
import sys
import tempfile
import venv
from pathlib import Path
from zipfile import ZipFile


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
            [str(python), "-I", str(repo_root / "tools" / "check_installed_wheel.py")],
            cwd=work_dir,
            env={**os.environ, "APPDATA": str(work_dir / "app-data")},
            check=True,
        )


if __name__ == "__main__":
    main()
