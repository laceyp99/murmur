# Murmur Pipeline Docs

This directory documents Murmur's audio processing pipeline in more detail than
the root README. The diagrams are written as Mermaid blocks inside Markdown so
they are easy to edit, review, and keep in sync with code changes.

## Reading Order

1. [Pipeline Overview](pipeline.md) explains the full path from hotkey capture
   to clipboard output.
2. [Live Pipeline](live-pipeline.md) focuses on the low-latency path that runs
   while recording is still active.
3. [VAD Segmentation](vad-segmentation.md) explains how recorder blocks become
   WebRTC VAD frames and speech segments.
4. [Transcription And Cleanup](transcription-and-cleanup.md) covers Whisper,
   transcript accumulation, local cleanup, and optional Ollama cleanup.
5. [Fallbacks And Failure Modes](failure-and-fallbacks.md) shows how Murmur
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

- [`src/main.py`](../src/main.py): application orchestration and fallback logic.
- [`src/audio.py`](../src/audio.py): microphone capture and recorder blocks.
- [`src/vad_live.py`](../src/vad_live.py): live VAD worker.
- [`src/vad_segmenter.py`](../src/vad_segmenter.py): offline VAD segmentation.
- [`src/transcription_live.py`](../src/transcription_live.py): serial live
  Whisper worker and transcript accumulation.
- [`src/transcription.py`](../src/transcription.py): Whisper model loading,
  segment transcription, and final document cleanup.
- [`src/llm_postprocess.py`](../src/llm_postprocess.py): optional Ollama final
  cleanup and output acceptance gate.
