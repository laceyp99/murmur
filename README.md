# murmur: local speech-to-text hotkey app

![header](murmur pipeline.png "murmur pipeline")

A lightweight Windows application that enables dictation anywhere on your system. Press a global hotkey to record your voice, and murmur will transcribe it using OpenAI's Whisper model running locally on your machine, then copy the text to your clipboard for instant pasting.

## Features

- 🎤 **Global Hotkey** - Works across all Windows applications
- 🔒 **100% Local** - No internet required, all processing on your machine
- 🚀 **GPU Accelerated** - Fast transcription with CUDA support
- 📋 **Clipboard Integration** - Transcription copied automatically
- � **System Tray** - Runs in the background with a status icon
- ⚙️ **Settings GUI** - Easily configure hotkey, model, and auto-start
- 📁 **Training Data Logging** - Saves audio and transcriptions for fine-tuning

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
5. **Paste** with `Ctrl+V` anywhere

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
  "max_recording_duration": 300,
  "enable_logging": true,
  "enable_notifications": true
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
| `max_recording_duration` | Maximum recording length in seconds | `300` |
| `enable_logging` | Save audio/transcriptions for training | `true` |
| `start_with_windows` | Automatically start on login | `true` |

## Troubleshooting

### Common Issues

**"Failed to register hotkey"**
- Another application may be using the same hotkey
- Run murmur as Administrator
- Try a different hotkey combination

**"No speech detected"**
- Check your microphone is working and selected as default
- Check microphone permissions in Windows Settings

**Slow transcription**
- Ensure CUDA is properly installed if you have an NVIDIA GPU
- Use a smaller Whisper model (tiny or base)
- Check that GPU is being used: look for "Device: cuda" on startup

## Training Data

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
