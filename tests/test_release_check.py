"""Exercise release metadata rules without accessing GitHub."""

import importlib.util
import subprocess
import sys
import textwrap
from pathlib import Path

import pytest

SCRIPT = Path(__file__).resolve().parents[1] / ".github/scripts/release_check.py"
SPEC = importlib.util.spec_from_file_location("release_check", SCRIPT)
release_check = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(release_check)


@pytest.mark.parametrize(
    ("version", "exit_code"),
    [("0.1.0", 0), ("0.2.0-beta", 1), ("0.2.0+build", 1)],
)
def test_release_workflow_validates_version_before_extracting_notes(version, exit_code):
    root = SCRIPT.parents[2]
    workflow = (root / ".github/workflows/release.yml").read_text(encoding="utf-8")
    # Run the workflow's actual Python notes step without creating a release.
    notes_step = textwrap.dedent(
        workflow.split("<<'PY'\n", 1)[1].split("          PY", 1)[0]
    )
    result = subprocess.run(
        [sys.executable, "-c", notes_step, version],
        cwd=root,
        capture_output=True,
        text=True,
        check=False,
    )
    assert result.returncode == exit_code
    if exit_code:
        assert "Windows exe version" in result.stderr
        assert not result.stdout
    else:
        assert "Local Whisper dictation" in result.stdout


@pytest.fixture
def release_project(tmp_path, monkeypatch):
    monkeypatch.chdir(tmp_path)
    monkeypatch.setenv("PR_TITLE", "feat: improve dictation")
    monkeypatch.setenv("PR_LABELS", "[]")
    monkeypatch.setenv("BASE_REF", "main")
    monkeypatch.setattr(
        release_check, "git", lambda *args: '[project]\nversion = "0.1.0"\n'
    )
    (tmp_path / "CHANGELOG.md").write_text(
        "## [Unreleased]\n\n## [0.2.0] - 2026-11-06\n\n### Added\n\n- Dictation improvement.\n",
        encoding="utf-8",
    )
    return tmp_path


def test_bump_with_matching_notes_passes(release_project, capsys):
    (release_project / "pyproject.toml").write_text(
        '[project]\nversion = "0.2.0"\n', encoding="utf-8"
    )
    release_check.main()
    assert "0.1.0 -> 0.2.0" in capsys.readouterr().out


@pytest.mark.parametrize("version", ["0.2.0-beta", "0.2.0+build", "0.2.65536"])
def test_rejects_versions_incompatible_with_windows(release_project, capsys, version):
    (release_project / "pyproject.toml").write_text(
        f'[project]\nversion = "{version}"\n', encoding="utf-8"
    )
    with pytest.raises(SystemExit, match="1"):
        release_check.main()
    assert "Windows exe version" in capsys.readouterr().err


def test_compares_windows_revision_component():
    assert release_check.greater_than("0.2.0.1", "0.2.0")
    assert not release_check.greater_than("0.2.0.0", "0.2.0")
    assert not release_check.greater_than("0.2.0", "0.2.0.1")
    assert release_check.parse_version("65535.65535.65535.65535") is not None


@pytest.mark.parametrize(
    ("title", "labels"),
    [
        (f"{kind}: maintenance", "[]")
        for kind in ["test", "ci", "docs", "style", "chore"]
    ]
    + [("feat: improve dictation", '["skip-release"]')],
)
def test_skips_maintenance_without_reading_project(
    tmp_path, monkeypatch, title, labels
):
    monkeypatch.chdir(tmp_path)
    monkeypatch.setenv("PR_TITLE", title)
    monkeypatch.setenv("PR_LABELS", labels)
    release_check.main()


@pytest.mark.parametrize(
    "notes",
    [
        "## [0.2.0] - 2026-02-30\n- Invalid date\n",
        "## [0.2.0] - 2026-11-06\n### Added\n<!-- no notes -->\n",
        "## [0.3.0] - 2026-11-06\n- Wrong version\n",
    ],
)
def test_rejects_missing_or_invalid_notes(release_project, notes):
    (release_project / "pyproject.toml").write_text(
        '[project]\nversion = "0.2.0"\n', encoding="utf-8"
    )
    (release_project / "CHANGELOG.md").write_text(notes, encoding="utf-8")
    with pytest.raises(SystemExit, match="1"):
        release_check.main()
