# Changelog

All notable changes to murmur are documented in this file.

The format is based on [Keep a Changelog](https://keepachangelog.com/en/1.1.0/).
Release versions are declared in `pyproject.toml`.

## [Unreleased]

## [0.2.0] - 2026-11-07

### Added

- Windows installer published with each GitHub Release. Installed copies check
  for updates from the settings window header and apply them through
  **Restart to update** in the tray, which stays unavailable during dictation.
- Release builds include small delta updates from the previous version.

### Changed

- Uninstalling removes this install's **Start with Windows** entry while keeping
  settings, statistics, and training data.

## [0.1.0] - 2026-11-06

### Added

- Local Whisper dictation with global hotkeys, microphone capture, live speech
  segmentation, and final clipboard output.
- Optional Ollama cleanup, user vocabulary, and opt-in local training data logging.
- Windows tray controls, recording feedback, startup settings, and a standalone
  Windows packaging workflow for local builds.
