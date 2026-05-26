# murmur: local speech-to-text hotkey app

![header](murmur_pipeline.png "murmur pipeline")

A lightweight Windows application that enables dictation anywhere on your system. Press a global hotkey to record your voice, and murmur will segment speech in real time, transcribe sealed chunks with OpenAI's Whisper model running locally on your machine, and finalize the cleaned document to your clipboard when you stop.

## Features

- 🎤 **Global Hotkey** - Works across all Windows applications
- 🔒 **100% Local** - No internet required, all processing on your machine
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
- **Python**: 3.10 or higher
- **GPU**: NVIDIA GPU with CUDA support 
- **CPU**: Works without GPU, but transcription will be slower

## Installation

### 1. Clone the Repository

```bash
git clone https://github.com/laceyp99/murmur.git
cd murmur
```

### 2. Create a Virtual Environment (Recommended)

```bash
python -m venv venv
venv\Scripts\activate
```

### 3. Install PyTorch with CUDA Support

For GPU acceleration, install PyTorch with CUDA:

```bash
pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu121
```

For CPU-only installation:

```bash
pip install torch torchvision torchaudio
```

### 4. Install Dependencies

```bash
pip install -r requirements.txt
```

### 5. Install FFmpeg (Required by Whisper)

Whisper requires FFmpeg. You can install it via manual installation. Download from [ffmpeg.org](https://ffmpeg.org/download.html) and add to PATH.

## Usage

### Starting murmur

Run the application:

```bash
python run.py
```

To run in the background without a console window, use:

```bash
pythonw run.py
```

Or double-click `run_background.vbs`.

### Using murmur

1. **System Tray**: Look for the murmur icon in your system tray (bottom right). Right-click it to access **Settings** or **Exit**.
2. **Press `Ctrl+Shift+Space`** to start recording
3. **Speak** your text
4. **Press `Ctrl+Shift+Space`** again to stop recording
5. **Wait briefly while murmur finalizes** any last live segment and document cleanup
6. **Paste** with `Ctrl+V` anywhere

### How transcription works

- murmur captures audio in lightweight 100 ms recorder blocks.
- A background WebRTC VAD worker reframes those blocks into 20 ms speech frames.
- Completed speech segments are transcribed serially in the background while you are still recording.
- If the live VAD or live transcription path fails mid-recording, murmur logs the failure and falls back to finalizing from the full recorded audio when you stop.
- When you stop, murmur flushes any pending speech, drains the live transcription queue, applies one final document cleanup pass, and copies the final text to the clipboard.

### Final cleanup with Ollama

- After Whisper completes the final document text, murmur runs one optional final-pass cleanup through a local Ollama model.
- The Ollama pass is enabled by default and runs only once per completed recording, not on every live chunk.
- On startup, murmur attempts to warm the configured Ollama model when `ollama_preload_model` is true.
- If Ollama is unavailable, times out, or returns invalid-looking output, murmur falls back to the pre-LLM cleaned transcript instead of failing the recording.
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

On the first run, murmur will download the Whisper model (this may take a few minutes depending on the model size).

### Whisper Model Sizes

| Model | Parameters | VRAM Required | Relative Speed |
|-------|------------|---------------|----------------|
| tiny | 39M | ~1 GB | ~32x |
| base | 74M | ~1 GB | ~16x |
| small | 244M | ~2 GB | ~6x |
| medium | 769M | ~5 GB | ~2x |
| large | 1550M | ~10 GB | 1x |

## Configuration

Configuration is stored in `%APPDATA%\murmur\config.json`:

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

### Configuration Options

| Option | Description | Default |
|--------|-------------|---------|
| `hotkey` | Global hotkey to toggle recording | `ctrl+shift+space` |
| `model` | Whisper model (tiny, base, small, medium, large) | `small` |
| `device` | Compute device (cuda, cpu) | `cuda` |
| `language` | Language code (null for auto-detect) | `null` |
| `sample_rate` | Audio sample rate in Hz | `16000` |
| `vad_aggressiveness` | WebRTC VAD aggressiveness level | `1` |
| `vad_padding_ms` | Speech end padding in ms; start padding is derived asymmetrically from this value | `220` |
| `vad_silence_duration_ms` | Silence duration in ms required to close a speech segment | `400` |
| `max_recording_duration` | Maximum recording length in seconds | `300` |
| `enable_logging` | Save raw audio/transcriptions for training after explicit opt-in | `false` |
| `enable_notifications` | Show Windows toast notifications | `true` |
| `start_with_windows` | Automatically start on login | `true` |
| `pause_media_while_recording` | Pause system media during recording | `true` |
| `logging_consent_updated_at` | Local timestamp of the last logging consent change | `null` |
| `logging_consent_source` | Local source of the last logging consent change | `null` |
| `ollama_enabled` | Enable the final-pass Ollama cleanup step | `true` |
| `ollama_endpoint` | Ollama server endpoint | `http://localhost:11434` |
| `ollama_model_name` | Ollama model used for final cleanup | `granite4.1:3b` |
| `ollama_timeout_seconds` | Request timeout for Ollama calls (minimum 60 seconds) | `60` |
| `ollama_preload_model` | Warm the Ollama model on startup when enabled | `true` |

### Ollama setup

Install Ollama separately, start the server, and make sure the configured model exists locally before launching murmur. Example:

```powershell
ollama pull granite4.1:3b
ollama serve
```

If you want to disable the final LLM pass entirely, set `ollama_enabled` to `false` in `%APPDATA%\murmur\config.json`.

## Troubleshooting

### Common Issues

**"Failed to register hotkey"**
- Another application may be using the same hotkey
- Run murmur as Administrator
- Try a different hotkey combination

**"No speech detected"**
- Check your microphone is working and selected as default
- Check microphone permissions in Windows Settings

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

When `enable_logging` is true, murmur saves all recordings for fine-tuning:

**Location:** `%APPDATA%\murmur\training_data\`

```
training_data/
├── audio/                # WAV files (16kHz mono)
│   └── 20241206_143022_123456.wav
└── transcriptions.jsonl  # Metadata
```

Each JSONL entry:
```json
{"timestamp": "2024-12-06T14:30:22", "audio_file": "20241206_143022_123456.wav", "transcription": "Your text", "duration": 3.5, "model": "small", "processing_time": 0.8}
```

Privacy notes:

- Raw WAV audio and transcript text are only stored after you opt in.
- You can disable logging at any time from Settings.
- You can delete existing logged data from Settings with **Delete Logged Data**.
- Ollama defaults to `http://localhost:11434`; if you point it at a remote endpoint, your transcripts leave the local machine for that final cleanup step.
