# Fallbacks And Failure Modes

Murmur treats the live pipeline as an optimization, not the only source of
truth. The full recording remains available until finalization, so most live
failures degrade to a slower full-recording path instead of losing the user's
dictation.

## Fallback Overview

```mermaid
flowchart TB
    Recording["Recording active"] --> FullBuffer["Full audio buffer is always kept"]
    Recording --> LivePath["Live VAD and live Whisper path"]

    LivePath --> LiveStatus{"Live path healthy?"}
    LiveStatus -->|yes| LiveText["Use accumulated live transcript"]
    LiveStatus -->|no| Degraded["Mark live pipeline degraded"]

    Stop["Stop hotkey or max duration"] --> Finalize["_finalize_recording"]
    FullBuffer --> Finalize
    LiveText --> Finalize
    Degraded --> Finalize

    Finalize --> Decision{"Use live text?"}
    Decision -->|yes| Cleanup["Final cleanup"]
    Decision -->|no| Recompute["Recompute from full recording"]
    Recompute --> OfflineVad{"Offline VAD available and finds speech?"}
    OfflineVad -->|yes| Segments["Transcribe VAD segments"]
    OfflineVad -->|no| FullClip["Transcribe full clip"]
    Segments --> Cleanup
    FullClip --> Cleanup
    Cleanup --> Clipboard["Copy to clipboard"]
```

## Live Degradation Sources

```mermaid
flowchart LR
    Callback["Recorder block callback failure"] --> Degraded["live_pipeline_degraded"]
    VadInit["Live VAD init failure"] --> Disabled["live VAD disabled"]
    VadWorker["Live VAD worker exception"] --> Degraded
    VadCallback["Live segment callback failure"] --> Degraded
    WhisperFail["Live Whisper segment failure after retry"] --> Degraded
    StopCheck["Worker degraded during stop"] --> Degraded

    Degraded --> FullFallback["Fallback to full recording at finalization"]
    Disabled --> FullFallback
```

A degraded live path does not mean the recording failed. It means Murmur should
ignore partial live output and rebuild the final transcript from the full
recording.

## Offline Fallback Ladder

When the live transcript cannot be used, `_process_audio()` calls
`_transcribe_audio()`:

```mermaid
flowchart TB
    AudioData["AudioData from recorder"] --> Segmenter{"Get WebRTCVADSegmenter"}
    Segmenter -->|unavailable| FullClip["Whisper full clip"]
    Segmenter -->|available| SegmentAudio["segment_audio(audio)"]
    SegmentAudio -->|raises| FullClip
    SegmentAudio -->|no segments| FullClip
    SegmentAudio -->|segments| TranscribeSegments["Whisper each segment serially"]
    TranscribeSegments --> FinalText["finalize_text(joined segment text)"]
    FullClip --> FinalText
```

This gives Murmur three chances to produce useful text:

1. Use the live transcript accumulated during recording.
2. Recompute from offline VAD speech segments.
3. Transcribe the full clip if VAD is unavailable or unhelpful.

## Clipboard And Logging Outcomes

Finalization can still succeed even if clipboard copy fails. In that case,
Murmur reports the copy failure. If training data logging is enabled and the log
write succeeds, the transcript is still saved locally in the opt-in training
data area.

```mermaid
flowchart LR
    FinalText["Final transcript text"] --> Clipboard{"Clipboard copy succeeds?"}
    FinalText --> Logging{"Training data logging enabled?"}
    Clipboard -->|yes| NotifyCopied["Notify copied"]
    Clipboard -->|no| NotifyCopyFailed["Notify copy failed"]
    Logging -->|yes| SaveLog["Save WAV and JSONL metadata"]
    Logging -->|no| SkipLog["Do not persist transcript or audio"]
```

## Content Safety In Logs

Runtime logs should avoid printing dictated transcript contents. The live path
currently logs operational details such as segment IDs, durations, latency, and
text length. That preserves debuggability without exposing private dictated
text in normal console output.

## Implementation Map

| Failure or fallback | Code |
| --- | --- |
| Live pipeline degraded flag | [`src/main.py`](../src/main.py) |
| Live block callback error handling | [`src/audio.py`](../src/audio.py) and [`src/main.py`](../src/main.py) |
| Live VAD worker degradation | [`src/vad_live.py`](../src/vad_live.py) |
| Live transcription retry and degradation | [`src/transcription_live.py`](../src/transcription_live.py) |
| Full recording fallback | [`src/main.py`](../src/main.py) |
| Offline VAD fallback to full clip | [`src/main.py`](../src/main.py) |
| Clipboard result handling | [`src/main.py`](../src/main.py) and [`src/clipboard.py`](../src/clipboard.py) |
| Optional training data logging | [`src/logger.py`](../src/logger.py) |
