# Live Segment Metrics Plan

## Goal

Add live transcription segment runtime summary metrics to stdout and opt-in metadata logging.

## Context

`main` now includes PR 19, which changed `processing_time` to represent finalization latency after recording stops. Live transcription already records per-chunk `TranscriptChunk.latency_seconds`, but that value is only printed per segment and is not summarized in final stdout or written to `transcriptions.jsonl`.

For this branch, `live_segment_latency_*` should mean transcription runtime only, matching the current `TranscriptChunk.latency_seconds`. It should not include queue wait time, VAD sealing delay, or stop-time finalization.

The metadata schema should stay explicit and easy to aggregate:

- `live_segment_count`: integer count of live chunks that contributed transcript text.
- `live_segment_latency_avg_seconds`: average runtime for contributing live chunks, or `null` when no contributing live chunks exist.
- `live_segment_latency_max_seconds`: max runtime for contributing live chunks, or `null` when no contributing live chunks exist.

Using explicit default fields is preferable to omitting fields on fallback recordings because downstream JSONL consumers get a stable schema. The `null` latency values still make it clear that no live latency measurement was available; they are not confused with a real zero-second latency.

## Current Branch

`feature/live-segment-latency-metrics`

## Scope

- Add a small structured representation for live segment metric summaries.
- Compute metrics from live transcript chunks that contributed text.
- Thread live metric summaries into finalization and metadata logging.
- Add a final stdout summary line alongside `Finalized in ...`.
- Add metadata fields to `TranscriptionLog` and JSONL output.
- Document the new JSONL fields in `README.md`.
- Cover live and fallback behavior in focused tests.

## Non-Goals

- Do not change the definition of existing per-segment `latency_seconds`.
- Do not include queue wait time, VAD sealing delay, or clipboard/finalization time in live segment latency metrics.
- Do not log transcript text to stdout.
- Do not create `training_data/` during tests or planning.
- Do not add a config toggle for these metrics.

## Commit Plan

### Commit 1

- **Goal:** Add live metric aggregation near the live transcript accumulator.
- **Suggested commit message:**
  - **Subject:** `feat(transcription_live): summarize contributing chunk latency`
  - **Body:** Add a small metrics summary helper on the live transcript accumulator. Count only chunks that are stored and contribute transcript text, and compute average/max runtime from existing chunk latency values.
- **Files / areas:** `src/transcription_live.py`, `tests/test_transcription_live.py`
- **Validation:** `venv\Scripts\python.exe -m pytest tests\test_transcription_live.py`
- **Notes:** Keep the metric definition runtime-only. Empty transcript chunks should remain excluded because they do not affect final text.

### Commit 2

- **Goal:** Thread live metrics into finalization, stdout, and metadata logging.
- **Suggested commit message:**
  - **Subject:** `feat(main): report live segment runtime summary`
  - **Body:** Pass live metric summaries from the accumulator into completion logging, print a final content-safe stdout summary, and write stable live metric fields to training metadata.
- **Files / areas:** `src/main.py`, `src/logger.py`, `tests/test_main.py`
- **Validation:** `venv\Scripts\python.exe -m pytest tests\test_main.py tests\test_transcription_live.py`
- **Notes:** For offline fallback or missing live text, pass a default summary: count `0`, average `None`, max `None`. This keeps logged metadata schema stable.

### Commit 3

- **Goal:** Document the new metadata fields and run full validation.
- **Suggested commit message:**
  - **Subject:** `docs: document live segment runtime metrics`
  - **Body:** Update the training metadata example and field descriptions so `processing_time` remains distinct from live segment runtime metrics.
- **Files / areas:** `README.md`, possibly `.github/pull_request_template.md` only if validation expectations change
- **Validation:** `venv\Scripts\python.exe -m pytest`, `venv\Scripts\python.exe -m ruff format --check .`, `venv\Scripts\python.exe -m ruff check .`
- **Notes:** The README should explicitly distinguish finalization latency from live segment transcription runtime.

## Validation

Before opening the PR, run the repo's standard checks:

- `venv\Scripts\python.exe -m pip install -e .[dev]`
- `venv\Scripts\python.exe -m ruff format --check .`
- `venv\Scripts\python.exe -m ruff check .`
- `venv\Scripts\python.exe -m pytest`

## Risks and Open Questions

- There are no current JSONL consumers, but if there were, they might not expect the extra fields, but adding nullable fields is lower-risk than omitting them conditionally.
- The final stdout summary should stay content-safe and avoid transcript text.

## Definition of Done

- Live transcript completions print a final summary with contributing segment count, average runtime, and max runtime.
- Metadata logs include `live_segment_count`, `live_segment_latency_avg_seconds`, and `live_segment_latency_max_seconds` for every successful logged transcription.
- Offline fallback and no-live-text paths log `0`, `null`, and `null` for the new live metric fields.
- Tests cover accumulator aggregation, live finalization logging, and fallback defaults.
- Full repo validation passes.
