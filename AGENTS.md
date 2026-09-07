# Murmur Agent Instructions

Treat these instructions as good defaults rather than hard rules. Explicit
developer instructions take precedence.

## Scope

Murmur is a Windows-first local dictation application. It owns the path from
global hotkey input through microphone capture, speech segmentation,
transcription, optional cleanup, and final clipboard output.

Keep the core dictation path local-first. Whisper transcription runs locally.
Network-dependent features such as Ollama cleanup are optional and must not
prevent basic transcription from working.

Treat the live VAD and transcription pipeline as a latency optimization rather
than the source of truth. Preserve the full recording until finalization so
Murmur can fall back to offline processing when live processing fails.

## Reliability and privacy

Preserving the user's dictated audio until transcription finalizes is critical. 

A failed optional cleanup step should not discard an otherwise valid local
transcription.

Dictated text and audio are private user data. Do not add transcript contents
to normal runtime logs. Training-data persistence must remain opt-in.

## Environment
- Treat the repository-local `venv` as the default Python environment.
- If it does not exist, create it with `python -m venv venv`.
- On Windows, activate it with `venv\Scripts\Activate.ps1`, or call `venv\Scripts\python.exe` and `venv\Scripts\pythonw.exe` directly.
- Do not use the global Python interpreter when a repo venv is available.
- Navigate through the `docs/` subdirectory to understand the pipeline and trace to relevant code.

## Glossary

| Term | Definition |
| -------------------------- | ------------------------------------------------------------------------------------------------------------------- |
| **Session** | One dictation cycle from hotkey start through final clipboard output or failure. |
| **Live transcript** | Text accumulated incrementally by the live transcription worker during recording. |
| **Final transcript** | The completed text after transcription, document cleanup, and any accepted optional LLM cleanup. |
| **Ollama cleanup** | The optional LLM-based final cleanup pass handled through Ollama. |
| **User vocabulary** | User-defined preferred names, spellings, or terms supplied through `user_vocab.json`. |
| **Training data logging** | The opt-in local persistence of audio and transcript data for later model or prompt evaluation. |

## Working Rules

- Follow “measure twice, cut once” and YAGNI.
- Inspect existing provider and test patterns before editing.
- This is a Windows-first desktop app, so prefer PowerShell-friendly commands and paths.
- Do not log dictated transcript contents in normal runtime logs.
- Keep relevant documentation and Mermaid pipeline diagrams synchronized with meaningful behavior changes.

## Setup And Validation
Install dependencies inside the venv with:

`venv\Scripts\python.exe -m pip install -e ".[dev]"`

While iterating, run the narrowest relevant tests first. Before handing off a change, run the development commands in
[Development Checks](docs/getting-started.md#development-checks).

Report anything that could not be run.

The foreground entrypoint is `python run.py`. Background launch uses
`pythonw run.py` or `run_background.vbs`.

## Pull requests

- For validation expectations and release hygiene, see [.github/pull_request_template.md](.github/pull_request_template.md).
- Never open a pull request unless the developer explicitly asks.
- Use a plain-language Conventional Commit title, such as
  `fix(media): resume playback after cancelled recording`.
- State the problem in one or two sentences, then explain the fix.
- Rebase onto the latest `main` before opening a pull request.
- List validation performed and anything skipped.
- Do not include generated artifacts, credentials, or unrelated cleanup.