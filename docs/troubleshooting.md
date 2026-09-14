# Troubleshooting

## Common issues

### Failed to register hotkey

- Another application may already use the configured hotkey.
- Try a different hotkey combination from **Settings**.
- If registration continues to fail, try running murmur as Administrator.

### No speech detected

- Confirm that the microphone works and is selected as the Windows default input.
- Check microphone permissions in Windows Settings.
- Speak long enough for a VAD segment to close; silence closes a segment but
  does not stop the overall recording.

### Clipboard copy failed

- Retry the recording if you still need the transcript on your clipboard.
- When training-data logging is disabled, murmur does not store the transcript
  for recovery.
- When logging is enabled and saving succeeds, the transcript remains in the
  local training-data area even if clipboard copy fails.

### There is a short pause after stopping

murmur transcribes completed speech segments during recording, but it still
performs a final flush, transcript cleanup, and optional Ollama request after
the stop hotkey. The remaining delay is usually the last queued segment and
those finalization steps. Lower `vad_silence_duration_ms` carefully if segments
are taking too long to close.

### Ollama warmup or cleanup failed

- Confirm Ollama is running at the configured `ollama_endpoint`.
- Confirm the configured model is installed locally.
- Increase `ollama_timeout_seconds` if model startup is slow.
- murmur falls back to the locally cleaned transcript when Ollama is unavailable.

### Slow transcription

- Verify CUDA is available if you have an NVIDIA GPU.
- Use a smaller Whisper model such as `tiny` or `base`.
- Check startup output for `Device: cuda`.

## Diagnose packaged model loading (maintainers)

Use this check when a packaged release exits before the tray appears and
you suspect model downloading or loading. Run it from the repository root
after [building the executable](getting-started.md#build-a-packaged-release).

The regular build self-check stays offline and does not test model downloads.
This check downloads and loads Whisper's `tiny` model on CPU in a fresh
temporary cache, so an already-cached model cannot hide a download failure.
It requires internet access and leaves your usual cache and settings alone.

```powershell
$modelCheckDir = Join-Path $env:TEMP ("murmur-model-check-" + [guid]::NewGuid())
New-Item -ItemType Directory -Path $modelCheckDir | Out-Null
$previousModelCache = $env:MURMUR_PACKAGING_SELF_CHECK_MODEL_CACHE
$previousCheckLog = $env:MURMUR_PACKAGING_SELF_CHECK_LOG
try {
    $env:MURMUR_PACKAGING_SELF_CHECK_MODEL_CACHE = Join-Path $modelCheckDir "models"
    $env:MURMUR_PACKAGING_SELF_CHECK_LOG = Join-Path $modelCheckDir "error.txt"
    $check = Start-Process -FilePath .\dist\murmur\murmur.exe `
        -ArgumentList "--packaging-self-check" -WindowStyle Hidden -Wait -PassThru
    if ($check.ExitCode -ne 0) {
        throw "Model check failed. Inspect $modelCheckDir\error.txt"
    }
    Write-Host "Model download and loading passed. Temporary files: $modelCheckDir"
}
finally {
    $env:MURMUR_PACKAGING_SELF_CHECK_MODEL_CACHE = $previousModelCache
    $env:MURMUR_PACKAGING_SELF_CHECK_LOG = $previousCheckLog
}
```

A successful result confirms that the packaged app can download and load
a model. It does not test the microphone, tray, clipboard, autostart, or
CUDA. If it fails, use `error.txt` in the printed temporary directory to
investigate; if that file is absent, the process may have failed before
Python could capture the error. The temporary directory can be deleted
afterward.

## Failure and fallback behavior

For implementation details and the full fallback ladder, see
[Fallbacks and Failure Modes](failure-and-fallbacks.md). murmur keeps the full
recording in memory, so live VAD or live transcription failures generally
degrade to offline processing instead of losing the recording.
