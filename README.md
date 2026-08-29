# murmur: local speech-to-text hotkey app

![header]()

A lightweight Windows application that enables dictation anywhere on your system. Press a global hotkey to record your voice, and murmur will segment speech in real time, transcribe sealed chunks with OpenAI's Whisper model running locally on your machine, and finalize the cleaned document to your clipboard when you stop. See the [architecture and process docs](docs/index.md) for the implementation-level flow.

## Features

- 🎤 **Global Hotkey** - Works across all Windows applications
- 🔒 **Local Inference** - Whisper runs locally; Ollama is local by default, with remote endpoints opt-in
- 🚀 **GPU Accelerated** - Fast transcription with CUDA support
- ✂️ **Live VAD Segmentation** - Detects speech chunks while you are still recording
- ⏱️ **Lower Stop Latency** - Starts serial Whisper work before you release the hotkey
- 📋 **Clipboard Integration** - Transcription copied automatically
- 🔇 **Auto-Pause Media** - Automatically pauses playing media during recording
- 🖥️ **System Tray** - Runs in the background with a status icon
- ⚙️ **Settings GUI** - Easily configure hotkey, model, and auto-start
- 📁 **Training Data Logging** - Optional local-only audio/transcript capture for fine-tuning
- 🧠 **Final LLM Cleanup** - Optional-to-configure but enabled-by-default Ollama cleanup pass for punctuation and light transcript correction
- 📝 **Local Vocabulary Overrides** - Repo-local `user_vocab.json` lets you force preferred spellings and names without editing code

## System Requirements

- **OS**: Windows 10/11
- **Python**: 3.12
- **GPU**: Optional NVIDIA GPU with CUDA support
- **CPU**: Supported, but transcription will be slower
- **Audio**: Microphone input and FFmpeg on `PATH`
- **Optional cleanup**: Ollama installed locally with the configured model

## Installation

### 1. Clone the Repository

```bash
git clone https://github.com/laceyp99/murmur.git
cd murmur
```

### 2. Create a Virtual Environment

```powershell
py -3.12 -m venv venv
venv\Scripts\python.exe -m pip install --upgrade pip
```

Use `venv\Scripts\python.exe -m pip ...` for the install commands below. Calling
the venv interpreter directly avoids installing packages into the wrong Python
when PowerShell activation or PATH resolution is inconsistent.

### 3. Install PyTorch with CUDA Support

For NVIDIA GPU acceleration, install the official PyTorch CUDA wheels before
installing murmur:

```powershell
venv\Scripts\python.exe -m pip install torch --index-url https://download.pytorch.org/whl/cu121
```

This project only needs `torch`. If you are setting up a new machine, check the
current selector at
[pytorch.org/get-started/locally](https://pytorch.org/get-started/locally/) and
use the Windows + Pip + CUDA command it recommends.

Verify that the venv has a CUDA-enabled Torch build and can see your GPU:

```powershell
venv\Scripts\python.exe -c "import torch; print('torch', torch.__version__); print('cuda build', torch.version.cuda); print('cuda available', torch.cuda.is_available()); print('gpu', torch.cuda.get_device_name(0) if torch.cuda.is_available() else 'n/a')"
```

Expected for GPU use:

- `torch.__version__` includes a CUDA suffix such as `+cu121`
- `torch.version.cuda` is not `None`
- `torch.cuda.is_available()` prints `True`
- `gpu` prints your NVIDIA GPU name

If `torch.cuda.is_available()` is `False`, update your NVIDIA driver and
reinstall the PyTorch CUDA wheels in the venv:

```powershell
venv\Scripts\python.exe -m pip install --force-reinstall torch --index-url https://download.pytorch.org/whl/cu121
```

For CPU-only installation, skip this step and let the project dependencies
install Torch from PyPI.

### 4. Install murmur and developer dependencies

Use the editable install so local code changes are picked up immediately:

```powershell
venv\Scripts\python.exe -m pip install -e ".[dev]"
```

This installs murmur, the runtime dependencies, Ruff, and pytest. The editable
install should not replace an already-installed CUDA Torch wheel because
`pyproject.toml` only requires `torch>=2.0.0`.

### 5. Install FFmpeg (Required by Whisper)

Whisper requires the FFmpeg executable. Download it from [ffmpeg.org](https://ffmpeg.org/download.html), extract it, and add its `bin` directory to `PATH`. Open a new PowerShell window after changing `PATH`.

### 6. Optional: install Ollama for final cleanup

Ollama is not required for transcription. It is enabled by default for the
final punctuation and light correction pass, but murmur falls back to local
cleanup when Ollama is unavailable. Install Ollama separately, then make the
configured model available:

```powershell
ollama pull granite4.1:3b
```

The default endpoint is `http://localhost:11434`. Start Ollama using its normal
desktop service or `ollama serve` before launching murmur. To keep all transcript
text on this computer, leave the endpoint local; a remote endpoint receives the
final transcript sent for cleanup.

## Usage

### Starting murmur

Run the application with the repository virtual environment:

```powershell
venv\Scripts\python.exe run.py
```

To run in the background without a console window, use:

```powershell
venv\Scripts\pythonw.exe run.py
```

Or double-click `run_background.vbs`; it prefers `venv\Scripts\pythonw.exe` when
the repository virtual environment exists. The editable install also exposes
the `murmur` console entry point, and `python -m src` runs the same application
entry point.

### Startup and recording flow

```mermaid
flowchart LR
    Launch["Launch run.py / python -m src"] --> Config["Load %APPDATA%\\murmur\\config.json"]
    Config --> Resources["Load Whisper and optionally warm Ollama"]
    Resources --> Hotkey["Register global toggle hotkey"]
    Hotkey --> Tray["Run system tray loop"]
    Tray --> Start["Start hotkey"]
    Start --> Capture["Record full audio and segment live"]
    Capture --> Stop["Stop hotkey or max duration"]
    Stop --> Finalize["Drain live work or recompute from full audio"]
    Finalize --> Cleanup["Local cleanup, optional Ollama cleanup"]
    Cleanup --> Clipboard["Copy final text to clipboard"]
```

At startup, a missing config file is created with defaults. An unreadable or
wrong-shaped config is moved to a timestamped `config.corrupt-*.json` backup and
replaced with defaults. If the configured hotkey is invalid or cannot be
registered, murmur attempts to reset it to `ctrl+shift+space`; if registration
still cannot succeed, startup exits with an error.

### Using murmur

1. **System Tray**: Look for the murmur icon in your system tray (bottom right). Right-click it to access **Settings** or **Exit**.
2. **Press `Ctrl+Shift+Space`** to start recording
3. **Speak** your text
4. **Press `Ctrl+Shift+Space`** again to stop recording, or let murmur auto-stop at `max_recording_duration`
5. **Wait briefly while murmur finalizes** any last live segment and document cleanup
6. **Paste** with `Ctrl+V` anywhere

Murmur does not auto-stop because of silence; silence only closes individual VAD
segments. Capture stops when you press the hotkey again or when
`max_recording_duration` is reached.

### How transcription works

- murmur captures audio in lightweight 100 ms recorder blocks.
- A background WebRTC VAD worker reframes those blocks into 20 ms speech frames.
- Completed speech segments are transcribed serially in the background while you are still recording.
- Before accepting Whisper output, murmur filters likely non-speech segments using Whisper metadata. Segments with `no_speech_prob > 0.4` and `avg_logprob <= -0.6` are skipped, while high-confidence speech is kept even when the no-speech probability is elevated. These are private defaults for now; very quiet speech may need future tuning.
- If live VAD initialization is unavailable, murmur continues recording and uses
  the offline path at stop. If live VAD, the recorder callback, or live
  transcription degrades during recording, murmur ignores partial live output
  and falls back to the full recorded audio when you stop.
- When you stop, murmur flushes any pending speech, drains the live transcription queue, applies one final document cleanup pass, and copies the final text to the clipboard.
- If a recording reaches `max_recording_duration`, murmur stops capture automatically, shows a notification, then finalizes the captured audio.

### Final cleanup with Ollama

- After Whisper completes the final document text, murmur runs one optional final-pass cleanup through a local Ollama model.
- The Ollama pass is enabled by default and runs only once per completed recording, not on every live chunk.
- On startup, murmur checks for the configured model and attempts to warm it when
  `ollama_preload_model` is true; warmup does not download a missing model.
- If Ollama is unavailable, times out, or returns invalid-looking output, murmur falls back to the pre-LLM cleaned transcript instead of failing the recording.
- When Ollama is disabled or unavailable, murmur applies only minimal local cleanup, so fallback transcript punctuation and capitalization may be rougher than the final LLM-cleaned output.
- The acceptance gate is intentionally conservative: murmur rejects empty output, obvious assistant preambles, length explosions, and chat/list-shaped responses.

### Personalized vocabulary

Create a repo-root `user_vocab.json` file to provide preferred spellings or corrections for names, products, and phrases:

```json
{
  "brew ridge": "Blue Ridge Data",
  "murmer": "Murmur",
  "q win": "Qwen"
}
```

- This file is gitignored so personal vocabulary stays local.
- murmur loads the file lazily when it builds the Ollama post-processor.
- If the file is missing or invalid JSON, murmur ignores it and continues normally.

### First Run

On the first run, murmur downloads the configured Whisper model if it is not in
Whisper's cache. This needs internet access once; subsequent Whisper inference
runs locally. If Ollama cleanup is enabled, the configured Ollama model must be
installed separately.

### Whisper Model Sizes

| Model | Parameters | VRAM Required | Relative Speed |
|-------|------------|---------------|----------------|
| tiny | 39M | ~1 GB | ~32x |
| base | 74M | ~1 GB | ~16x |
| small | 244M | ~2 GB | ~6x |
| medium | 769M | ~5 GB | ~2x |
| large | 1550M | ~10 GB | 1x |

## Configuration

Configuration is stored in `%APPDATA%\murmur\config.json` and is created on
first launch:

```json
{
  "hotkey": "ctrl+shift+space",
  "model": "small",
  "device": "cuda",
  "language": null,
  "sample_rate": 16000,
  "vad_aggressiveness": 1,
  "vad_padding_ms": 220,
  "vad_silence_duration_ms": 400,
  "max_recording_duration": 300,
  "enable_logging": false,
  "enable_notifications": true,
  "start_with_windows": true,
  "pause_media_while_recording": true,
  "logging_consent_updated_at": null,
  "logging_consent_source": null,
  "ollama_enabled": true,
  "ollama_endpoint": "http://localhost:11434",
  "ollama_model_name": "granite4.1:3b",
  "ollama_timeout_seconds": 60,
  "ollama_preload_model": true
}
```

Open **Settings** from the tray icon to edit the user-facing configuration in a
tabbed control panel:

- **General**: hotkey, notifications, Windows startup, and media pause behavior.
- **VAD**: WebRTC VAD aggressiveness, speech padding, and silence-to-stop timing.
- **Transcription**: Whisper model, device, language, and maximum recording duration.
- **LLM Cleanup**: Ollama enablement, endpoint, model, request timeout, preload behavior, and connection testing.
- **Data Privacy**: training data logging opt-in and logged-data deletion.

Numeric controls use bounded sliders plus editable number fields. Existing
out-of-range config values are clamped when displayed, and saving persists the
nearest allowed value. Non-numeric text in a numeric field blocks Save, resets
that field to its default in the UI, and leaves the config file unchanged.

Each tab has **Reset This Tab**, which resets only that tab's current UI values
to defaults. Reset values are not written until you click **Save**. **Cancel**
and the window close button discard unsaved edits.

Settings that affect cached startup components are marked as restart-required in
the save confirmation: hotkey, Whisper model/device, language, VAD timing,
maximum recording duration, Ollama endpoint/model, and Ollama preload behavior.
Restart murmur after changing those values. Notification, logging, media-pause,
and Windows-startup changes are applied by the settings save path; autostart is
updated in the current user's Windows Run registry key.

`sample_rate` is not exposed in the settings window. If you edit it directly,
restart murmur afterward. Live VAD requires one of 8,000, 16,000, 32,000, or
48,000 Hz; offline VAD can resample other rates for analysis and maps its
segments back to the recorded waveform.

### Configuration Options

| Option | Description | Default |
|--------|-------------|---------|
| `hotkey` | Global hotkey to toggle recording | `ctrl+shift+space` |
| `model` | Whisper model (tiny, base, small, medium, large) | `small` |
| `device` | Compute device (cuda, cpu) | `cuda` |
| `language` | Language code (null for auto-detect) | `null` |
| `sample_rate` | Audio sample rate in Hz | `16000` |
| `vad_aggressiveness` | WebRTC VAD aggressiveness level, bounded to `0-3` in Settings | `1` |
| `vad_padding_ms` | Speech end padding in ms, bounded to `50-800` in Settings; start padding is derived asymmetrically from this value | `220` |
| `vad_silence_duration_ms` | Silence duration in ms required to close a speech segment, bounded to `100-1500` in Settings | `400` |
| `max_recording_duration` | Maximum recording length in seconds before murmur auto-stops, notifies, and finalizes; bounded to `30-1800` in Settings | `300` |
| `enable_logging` | Save non-empty completed recordings as raw audio plus transcript metadata after explicit opt-in | `false` |
| `enable_notifications` | Show Windows toast notifications | `true` |
| `start_with_windows` | Automatically start on login | `true` |
| `pause_media_while_recording` | Pause system media during recording | `true` |
| `logging_consent_updated_at` | Local timestamp of the last logging consent change | `null` |
| `logging_consent_source` | Local source of the last logging consent change | `null` |
| `ollama_enabled` | Enable the final-pass Ollama cleanup step | `true` |
| `ollama_endpoint` | Ollama server endpoint | `http://localhost:11434` |
| `ollama_model_name` | Ollama model used for final cleanup | `granite4.1:3b` |
| `ollama_timeout_seconds` | Request timeout for Ollama calls, bounded to `60-300` in Settings | `60` |
| `ollama_preload_model` | Warm the Ollama model on startup when enabled | `true` |

### Ollama setup

Install Ollama separately, start the server, and make sure the configured model exists locally before launching murmur. Example:

```powershell
ollama pull granite4.1:3b
ollama serve
```

Use **Settings > LLM Cleanup > Test Ollama Connection** to check the currently
entered endpoint and model before saving. If you want to disable the final LLM
pass entirely, turn off **Enable Ollama cleanup** in Settings or set
`ollama_enabled` to `false` in `%APPDATA%\murmur\config.json`.

## Troubleshooting

### Common Issues

**"Failed to register hotkey"**
- Another application may be using the same hotkey
- Run murmur as Administrator
- Try a different hotkey combination

**"No speech detected"**
- Check your microphone is working and selected as default
- Check microphone permissions in Windows Settings

**"Clipboard copy failed"**
- Retry the recording if you still need the transcript on your clipboard
- When training data logging is off, murmur does not store the transcript for recovery
- When training data logging is enabled and the save succeeds, murmur keeps the completed transcript in training data even if clipboard copy fails

**"There is a short pause after I stop before text appears"**
- murmur now transcribes completed speech segments during recording, but it still performs a final drain on stop
- The remaining delay is usually the last queued segment plus final text cleanup and, when enabled, one Ollama post-processing call
- If stop-time latency feels too high, reduce `vad_silence_duration_ms` carefully so segments close sooner

**"Ollama warmup or cleanup failed"**
- Confirm Ollama is running and reachable at the configured `ollama_endpoint`
- Confirm the configured `ollama_model_name` is installed locally
- Increase `ollama_timeout_seconds` if model load or first response is slow on your hardware
- murmur will fall back to the non-LLM cleaned transcript if Ollama is unavailable

**Slow transcription**
- Ensure CUDA is properly installed if you have an NVIDIA GPU
- Use a smaller Whisper model (tiny or base)
- Check that GPU is being used: look for "Device: cuda" on startup

## Training Data

Training data logging is disabled by default. To enable it, open **Settings** from the tray icon, turn on **Enable Training Data Logging**, and confirm the privacy prompt.

When `enable_logging` is true, murmur saves each completed recording that
produces non-empty final text:

**Location:** `%APPDATA%\murmur\training_data\`

```
training_data/
├── audio/                # 16-bit mono WAV at the configured sample rate
│   └── 20241206_143022_123456.wav
└── transcriptions.jsonl  # Metadata
```

Each JSONL entry:
```json
{"timestamp": "2024-12-06T14:30:22", "audio_file": "20241206_143022_123456.wav", "transcription": "Your text", "duration": 3.5, "model": "small", "processing_time": 0.8, "live_segment_count": 2, "live_segment_latency_avg_seconds": 0.34, "live_segment_latency_max_seconds": 0.51}
```

`duration` is the recorded audio length in seconds. `processing_time` is the
finalization latency after recording stops, measured through the clipboard copy
attempt and before training-data logging. `live_segment_count` is the number of
live transcript chunks that contributed text to the final transcript.
`live_segment_latency_avg_seconds` and
`live_segment_latency_max_seconds` are live segment transcription runtimes, not
queue time or finalization time. When no live chunks contributed text, the count
is `0` and the latency fields are `null`.

Privacy notes:

- Raw WAV audio and transcript text are only stored after you opt in and a
  non-empty transcript is produced.
- Normal console status output does not include transcript text.
- You can disable logging at any time from Settings.
- You can delete existing logged data from Settings with **Delete Logged Data**; murmur shows the file count and approximate size before confirmation.
- Ollama defaults to `http://localhost:11434`; if you point it at a remote endpoint, your transcripts leave the local machine for that final cleanup step.
