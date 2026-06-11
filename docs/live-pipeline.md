# Live Pipeline

The live pipeline is the low-latency path. It begins before the first audio
sample is captured and continues until the stop hotkey drains the pending work.

The important design choice is that live VAD and live Whisper work are
background workers. The audio callback stays lightweight: it appends to the full
recording buffer and submits a block to the live worker. Expensive work happens
off the callback thread.

```mermaid
sequenceDiagram
    autonumber
    actor User
    participant Hotkey as HotkeyManager
    participant App as MurmurApp
    participant Recorder as AudioRecorder
    participant LiveVAD as LiveVADSegmentationWorker
    participant LiveWhisper as LiveTranscriptionWorker
    participant Transcriber as Transcriber
    participant Accumulator as TranscriptAccumulator
    participant Clipboard as Clipboard

    User->>Hotkey: Press start hotkey
    Hotkey->>App: _on_recording_start()
    App->>LiveWhisper: start()
    App->>LiveVAD: start()
    App->>Recorder: start_recording()

    loop Every recorder callback
        Recorder->>Recorder: append block to full buffer
        Recorder->>LiveVAD: submit_audio_block(block)
    end

    loop Live VAD worker
        LiveVAD->>LiveVAD: split blocks into 20 ms frames
        LiveVAD->>LiveVAD: track speech, silence, padding, merge gap
        LiveVAD-->>App: on_segment(LiveSpeechSegment)
        App->>LiveWhisper: submit_segment(segment)
    end

    loop Live transcription worker
        LiveWhisper->>Transcriber: transcribe_segment(segment)
        Transcriber-->>LiveWhisper: raw segment text
        LiveWhisper->>Accumulator: add_chunk(TranscriptChunk)
    end

    User->>Hotkey: Press stop hotkey
    Hotkey->>App: _on_recording_stop()
    App->>Recorder: stop_recording()
    Recorder-->>App: AudioData
    App->>LiveVAD: stop() and flush pending speech
    App->>LiveWhisper: stop() and drain queued segments
    App->>Accumulator: get_text()

    alt Live transcript is usable
        App->>Transcriber: finalize_text(live_text)
        Transcriber-->>App: final text
        App->>Clipboard: copy_to_clipboard(final text)
    else Live path degraded or empty
        App->>App: recompute from full AudioData
    end
```

## Live Worker Responsibilities

### Recorder Callback

The recorder callback is intentionally narrow:

- Append the current block to the full recording buffer.
- Enforce the configured maximum recording duration.
- Forward the block to live VAD if a callback is registered.
- Disable the live callback after the first callback failure.

This keeps capture resilient. Even if the live callback fails, the full buffer
still exists for stop-time fallback.

### Live VAD Worker

The live VAD worker owns:

- Block queueing.
- Carrying partial frames between recorder blocks.
- Converting frame audio to 16-bit PCM for WebRTC VAD.
- Start padding through a pre-speech frame buffer.
- Silence counting to decide when speech is sealed.
- End padding and min segment duration filtering.
- Merge-gap handling before emitting a segment.

### Live Transcription Worker

The live transcription worker owns:

- A serial queue of sealed speech segments.
- Segment-level retry.
- Per-segment transcription latency.
- Callback-safe failure reporting.
- Ordered transcript accumulation.

Serial transcription matters because it avoids multiple Whisper calls competing
for the same local model and keeps output ordering predictable.

## State Summary

```mermaid
stateDiagram-v2
    [*] --> Idle
    Idle --> Recording: start hotkey
    Recording --> Capturing: AudioRecorder stream active
    Capturing --> LiveSegmenting: block callback submits audio
    LiveSegmenting --> LiveTranscribing: sealed segment emitted
    LiveTranscribing --> Capturing: segment queued or transcribed
    Capturing --> Finalizing: stop hotkey or max duration
    LiveSegmenting --> Finalizing: stop flushes pending segment
    LiveTranscribing --> Finalizing: stop drains queue
    Finalizing --> Ready: clipboard copied or no speech
    Ready --> Idle

    Capturing --> Degraded: live callback or worker failure
    LiveSegmenting --> Degraded: VAD worker failure
    LiveTranscribing --> Degraded: Whisper segment failure after retry
    Degraded --> Finalizing: recompute from full recording
```

## What To Watch When Changing This Path

- Keep the audio callback cheap. Do not run Whisper, LLM cleanup, file writes,
  or blocking UI work from the callback.
- Preserve the full recording buffer even when live workers fail.
- Keep live transcription serial unless the Whisper/model ownership changes.
- Keep callback logs content-safe. Current logs print IDs, durations, and text
  lengths rather than dictated text.
- If changing segment IDs or accumulation, verify that final text still follows
  the spoken order.
