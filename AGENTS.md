# Murmur Agent Instructions

## Environment
- Treat the repository-local `venv` as the default Python environment.
- If it does not exist, create it with `python -m venv venv`.
- On Windows, activate it with `venv\Scripts\Activate.ps1`, or call `venv\Scripts\python.exe` and `venv\Scripts\pythonw.exe` directly.
- Do not use the global Python interpreter when a repo venv is available.

## Setup And Validation
- Install dependencies inside the venv with `python -m pip install -e .[dev]`.
- Use the same checks as CI and the pull request template: `ruff format --check .`, `ruff check .`, and `pytest`.
- The main entrypoint is `python run.py`; background launch uses `pythonw run.py` or `run_background.vbs`.

## Project Conventions
- This is a Windows-first desktop app, so prefer PowerShell-friendly commands and paths.
- `user_vocab.json` is a repo-local override file and should be treated as local state.
- Runtime config and local storage live under `%APPDATA%\murmur`; see [README.md](README.md) for setup and configuration details.
- Training data logging is opt-in. Do not reintroduce non-consensual logging behavior or create `training_data/` unless the task requires writing it.
- Keep changes small and aligned with the existing patterns in `src/` and `tests/`.
- For validation expectations and release hygiene, see [.github/pull_request_template.md](.github/pull_request_template.md).
