# murmur: local dictation hotkey app


A lightweight Windows application that enables dictation anywhere on your
system. Press a global hotkey to record your voice, and murmur segments speech
in real time, transcribes with OpenAI's Whisper model running locally, and
copies the cleaned document to your clipboard when you stop.

## Documentation

The documentation site contains the complete setup, usage, configuration,
privacy, pipeline, and troubleshooting guides:

- [Getting Started](docs/getting-started.md)
- [Settings and Privacy](docs/settings-and-privacy.md)
- [Pipeline Documentation](docs/index.md)
- [Troubleshooting](docs/troubleshooting.md)

To preview the full MkDocs site locally from the repository root:

```powershell
venv\Scripts\python.exe -m pip install mkdocs-material pymdown-extensions
venv\Scripts\python.exe -m mkdocs serve
```

Then open [http://127.0.0.1:8000/](http://127.0.0.1:8000/).

## Features

- **Global Hotkey** - Works across all Windows applications
- **Local Inference** - Whisper runs locally; remote Ollama endpoints are opt-in
- **GPU Accelerated** - Fast transcription with CUDA support
- **Live VAD Segmentation** - Detects speech chunks while you are recording
- **Lower Stop Latency** - Starts serial Whisper work before you release the hotkey
- **Clipboard Integration** - Copies the final transcription automatically
- **Auto-Pause Media** - Pauses playing media during recording
- **System Tray** - Runs in the background with a status icon
- **Settings GUI** - Configure hotkey, model, and auto-start
- **Training Data Logging** - Optional local-only audio/transcript capture
- **Final LLM Cleanup** - Optional Ollama cleanup pass for punctuation and light correction
- **Local Vocabulary Overrides** - Use `user_vocab.json` for preferred spellings and names

## Quick Start

Murmur supports Windows 10/11 and Python 3.12. A microphone and FFmpeg on
`PATH` are required; an NVIDIA GPU and Ollama are optional.

```powershell
py -3.12 -m venv venv
venv\Scripts\python.exe -m pip install -e ".[dev]"
venv\Scripts\python.exe run.py
```

See [Getting Started](docs/getting-started.md) for CUDA installation, FFmpeg,
Ollama setup, background launch, first-run behavior, and the recording flow.

To build the standalone Windows application folder with PyInstaller:

```powershell
powershell -ExecutionPolicy Bypass -File .\build_windows.ps1
```

The validated output is written to `dist\Murmur\`. Distribute the complete
folder. See [Build a packaged release](docs/getting-started.md#build-a-packaged-release)
for requirements and repeat-build instructions.

## Development

Run the project checks from the repository virtual environment:

```powershell
venv\Scripts\python.exe -m ruff format --check .
venv\Scripts\python.exe -m ruff check .
venv\Scripts\python.exe -m pytest
```

See the [development and documentation instructions](docs/getting-started.md#development-checks)
for the MkDocs build command.
