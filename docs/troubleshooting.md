# Troubleshooting

## Common issues

### Failed to register hotkey

- Another application may already use the configured hotkey.
- Try a different hotkey combination from **Settings**.
- If registration continues to fail, try running Murmur as Administrator.

### No speech detected

- Confirm that the microphone works and is selected as the Windows default input.
- Check microphone permissions in Windows Settings.
- Speak long enough for a VAD segment to close; silence closes a segment but
  does not stop the overall recording.

### Clipboard copy failed

- Retry the recording if you still need the transcript on your clipboard.
- When training-data logging is disabled, Murmur does not store the transcript
  for recovery.
- When logging is enabled and saving succeeds, the transcript remains in the
  local training-data area even if clipboard copy fails.

### There is a short pause after stopping

Murmur transcribes completed speech segments during recording, but it still
performs a final flush, transcript cleanup, and optional Ollama request after
the stop hotkey. The remaining delay is usually the last queued segment and
those finalization steps. Lower `vad_silence_duration_ms` carefully if segments
are taking too long to close.

### Ollama warmup or cleanup failed

- Confirm Ollama is running at the configured `ollama_endpoint`.
- Confirm the configured model is installed locally.
- Increase `ollama_timeout_seconds` if model startup is slow.
- Murmur falls back to the locally cleaned transcript when Ollama is unavailable.

### Slow transcription

- Verify CUDA is available if you have an NVIDIA GPU.
- Use a smaller Whisper model such as `tiny` or `base`.
- Check startup output for `Device: cuda`.

## Failure and fallback behavior

For implementation details and the full fallback ladder, see
[Fallbacks and Failure Modes](failure-and-fallbacks.md). Murmur keeps the full
recording in memory, so live VAD or live transcription failures generally
degrade to offline processing instead of losing the recording.
