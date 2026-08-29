# Murmur Pipeline Docs

This directory documents Murmur's setup, runtime behavior, audio processing
pipeline, and privacy boundaries in more detail than the root README. The
diagrams are written as Mermaid blocks inside Markdown so they are easy to edit,
review, and keep in sync with code changes.

## Reading Order

1. [Getting Started](getting-started.md) covers supported setup, launch modes,
   first-run behavior, and the user recording flow.
2. [Settings And Privacy](settings-and-privacy.md) documents config persistence,
   settings validation, autostart, logging, and the network boundary.
3. [Pipeline Overview](pipeline.md) explains the full path from hotkey capture
   to clipboard output.
4. [Live Pipeline](live-pipeline.md) focuses on the low-latency path that runs
   while recording is still active.
5. [VAD Segmentation](vad-segmentation.md) explains how recorder blocks become
   WebRTC VAD frames and speech segments.
6. [Transcription And Cleanup](transcription-and-cleanup.md) covers Whisper,
   transcript accumulation, local cleanup, and optional Ollama cleanup.
7. [Fallbacks And Failure Modes](failure-and-fallbacks.md) shows how Murmur
   recovers when live processing degrades.

## Diagram Editing

Mermaid diagrams are plain text. Edit the fenced `mermaid` blocks directly in
these Markdown files. GitHub can render Mermaid in Markdown, and most Markdown
editors with Mermaid support can preview these diagrams without committing
generated image files.

Recommended workflow:

1. Edit the Mermaid block in the relevant doc.
2. Preview it locally in an editor or paste it into the Mermaid live editor.
3. Keep node names close to the implementation names in `src/`.
4. Update the surrounding text when changing a diagram so the prose and visual
   stay consistent.

## Source Anchors

The pipeline is coordinated primarily through these modules:

- [`src/main.py`](https://github.com/laceyp99/murmur/blob/main/src/main.py): application orchestration and fallback logic.
- [`src/config.py`](https://github.com/laceyp99/murmur/blob/main/src/config.py): defaults, AppData paths, and config
  recovery.
- [`src/audio.py`](https://github.com/laceyp99/murmur/blob/main/src/audio.py): microphone capture and recorder blocks.
- [`src/hotkey.py`](https://github.com/laceyp99/murmur/blob/main/src/hotkey.py): global toggle hotkey and processing state.
- [`src/tray.py`](https://github.com/laceyp99/murmur/blob/main/src/tray.py): tray menu and application shutdown entry point.
- [`src/settings_gui.py`](https://github.com/laceyp99/murmur/blob/main/src/settings_gui.py): settings UI and persistence
  actions.
- [`src/vad_live.py`](https://github.com/laceyp99/murmur/blob/main/src/vad_live.py): live VAD worker.
- [`src/vad_segmenter.py`](https://github.com/laceyp99/murmur/blob/main/src/vad_segmenter.py): offline VAD segmentation.
- [`src/transcription_live.py`](https://github.com/laceyp99/murmur/blob/main/src/transcription_live.py): serial live
  Whisper worker and transcript accumulation.
- [`src/transcription.py`](https://github.com/laceyp99/murmur/blob/main/src/transcription.py): Whisper model loading,
  segment transcription, and final document cleanup.
- [`src/llm_postprocess.py`](https://github.com/laceyp99/murmur/blob/main/src/llm_postprocess.py): optional Ollama final
  cleanup and output acceptance gate.
- [`src/logger.py`](https://github.com/laceyp99/murmur/blob/main/src/logger.py): opt-in training-data persistence and
  deletion.
