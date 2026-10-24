import json
import sqlite3
import subprocess
import threading
from dataclasses import fields

import pytest

from src.usage_stats import (
    EXPORT_COLUMNS,
    SessionMetadata,
    SessionSnapshot,
    StatisticsBusyError,
    UsageStatsService,
    _owner_dead,
)


@pytest.fixture
def service(tmp_path):
    stats = UsageStatsService(
        tmp_path / "statistics.sqlite3",
        enabled=True,
        provenance={"app_version": "test", "git_commit": "a" * 40, "git_dirty": True},
    )
    stats.summary().result(timeout=5)
    yield stats
    assert stats.close(timeout=5)


def barrier(stats):
    return stats.summary().result(timeout=5)


def test_empty_schema_and_private_contract(service):
    summary = barrier(service)
    assert summary["sessions"] == 0
    assert summary["success_rate"] is None
    assert summary["fallback_rate"] is None
    assert summary["captured_minutes"] is None
    assert summary["latency"]["median_seconds"] is None
    forbidden = {"text", "audio", "prompt", "endpoint", "microphone", "error_message"}
    assert forbidden.isdisjoint(
        f.name for cls in (SessionSnapshot, SessionMetadata) for f in fields(cls)
    )
    collector = service.start_session()
    with pytest.raises(ValueError, match="Unsupported"):
        collector.update(text="private content")
    with sqlite3.connect(service.path) as connection:
        assert connection.execute("PRAGMA user_version").fetchone()[0] == 1


def test_terminal_snapshot_is_idempotent_and_unknown_stays_null(service):
    collector = service.start_session()
    collector.update(whisper_model="small", whisper_device="cpu")
    collector.stage("offline_whisper", "failed", seconds=2)
    collector.issue("offline_whisper", "whisper_failed")
    collector.stage("offline_whisper", "success", seconds=3)
    collector.issue("offline_whisper", "whisper_failed", recovered=True)
    collector.issue("offline_whisper", "whisper_failed", recovered=True)
    collector.finish("success", final_word_count=7, stop_to_clipboard_seconds=5)
    collector.finish("failed", final_word_count=99)
    collector.update(capture_seconds=99)
    collector.stage("clipboard", "failed", seconds=88)
    summary = barrier(service)
    assert summary["counts"]["success"] == 1
    assert summary["delivered_words"] == 7
    assert summary["recording_count"] == 0
    assert summary["recovered_failures"] == 3
    assert summary["issues"] == {"whisper_failed": 3}
    stage = summary["stages"]["offline_whisper"]
    assert stage["attempts"] == 2
    assert stage["success_count"] == stage["failed_count"] == 1
    assert stage["mean_seconds"] == 3
    assert stage["mean_failed_seconds"] == 2
    row = service.export_rows().result(timeout=5)[0]
    assert tuple(row) == EXPORT_COLUMNS
    assert row["capture_seconds"] is None
    assert row["app_version"] == "test"
    assert row["git_commit"] == "a" * 40
    assert row["git_dirty"] is True
    assert row["stage_clipboard_seconds"] is None
    assert row["stage_offline_whisper_seconds"] == 5
    assert row["stage_offline_whisper_attempts"] == 2
    assert row["stage_offline_whisper_failed_count"] == 1


def test_export_rounds_durations_to_milliseconds(service):
    collector = service.start_session()
    collector.update(capture_seconds=7.123456)
    collector.stage("clipboard", "success", seconds=0.0034710000036)
    collector.finish("success", stop_to_clipboard_seconds=0.9475691000116)
    barrier(service)
    row = service.export_rows().result(timeout=5)[0]
    assert row["capture_seconds"] == 7.123
    assert row["stage_clipboard_seconds"] == 0.003
    assert row["stop_to_clipboard_seconds"] == 0.948
    assert len(EXPORT_COLUMNS) == len(set(EXPORT_COLUMNS))


def test_terminal_issue_deduplication_preserves_failures_in_other_stages(service):
    collector = service.start_session()
    collector.issue("live_whisper", "whisper_failed")
    collector.issue_once("offline_whisper", "whisper_failed")
    collector.issue_once("offline_whisper", "whisper_failed")
    collector.finish("failed")
    row = service.export_rows().result(timeout=5)[0]
    issues = json.loads(row["issues"])
    assert {(issue["stage"], issue["code"], issue["count"]) for issue in issues} == {
        ("live_whisper", "whisper_failed", 1),
        ("offline_whisper", "whisper_failed", 1),
    }


def test_exact_metrics_and_context_groups(service):
    fixtures = [
        ("success", 1, True, True, False),
        ("success", 9, True, False, True),
        ("failed", 100, True, True, False),
        ("failed", 100, False, True, False),
        ("no_speech", 100, True, False, False),
        ("cancelled", 100, True, True, False),
    ]
    for outcome, wait, reached, fallback, cleanup in fixtures:
        collector = service.start_session()
        collector.update(
            whisper_model="small",
            whisper_device="cpu",
            cleanup_enabled=cleanup,
            cleanup_model="qwen" if cleanup else None,
            capture_seconds=60,
            reached_transcription=reached,
            error_fallback=fallback,
        )
        collector.finish(outcome, stop_to_clipboard_seconds=wait, final_word_count=10)
        barrier(service)
    active = service.start_session()
    assert active is not None
    summary = barrier(service)
    assert summary["counts"]["running"] == 1
    assert summary["success_rate"] == 0.5
    assert summary["fallback_count"] == 2
    assert summary["fallback_denominator"] == 4
    assert summary["fallback_rate"] == 0.5
    assert summary["latency"] == {
        "count": 2,
        "median_seconds": 5,
        "p95_seconds": 9,
        "small_sample": True,
    }
    assert summary["captured_minutes"] == 6
    assert summary["recording_count"] == 6
    assert summary["delivered_words"] == 20
    assert len(summary["model_groups"]) == 3  # Includes unknown active context.
    assert sorted(g["latency"]["count"] for g in summary["model_groups"]) == [0, 1, 1]


def test_toggle_applies_to_next_session(service):
    current = service.start_session()
    service.set_enabled(False)
    assert service.start_session() is None
    current.finish("success")
    assert barrier(service)["counts"]["success"] == 1
    service.set_enabled(True)
    service.start_session().finish("cancelled")
    assert barrier(service)["sessions"] == 2


def test_disabled_run_creates_no_storage_until_enabled(tmp_path):
    path = tmp_path / "statistics.sqlite3"
    stats = UsageStatsService(path, enabled=False, provenance={})
    try:
        assert barrier(stats)["sessions"] == 0
        assert stats.export_rows().result(timeout=5) == []
        stats.clear().result(timeout=5)
        assert list(tmp_path.iterdir()) == []
        stats.set_enabled(True)
        barrier(stats)
        stats.start_session().finish("success")
        assert barrier(stats)["counts"]["success"] == 1
    finally:
        assert stats.close(timeout=5)


def test_cross_instance_delete_marks_lost_session_incomplete(service):
    other = UsageStatsService(service.path, enabled=True, provenance={})
    try:
        barrier(other)
        service.clear().result(timeout=5)
        assert barrier(service)["incomplete"] is False
        other.start_session().finish("success")
        summary = barrier(other)
        assert summary["sessions"] == 0
        assert summary["incomplete"] is True
        other.start_session().finish("success")
        assert barrier(other)["sessions"] == 1
    finally:
        assert other.close(timeout=5)


def test_cross_instance_generation_fences_every_old_snapshot(service):
    other = UsageStatsService(service.path, enabled=True, provenance={})
    try:
        barrier(other)
        old = other.start_session()
        barrier(other)
        service.clear().result(timeout=5)
        old.update(capture_seconds=12)
        old.stage("capture", "success", seconds=12)
        old.issue("capture", "capture_failed")
        old.finish("success")
        assert barrier(other)["sessions"] == 0
        assert service.export_rows().result(timeout=5) == []
        other.start_session().finish("success")
        assert barrier(other)["sessions"] == 1
    finally:
        assert other.close(timeout=5)


def test_live_owner_preserved_then_abandoned_reconciled(service):
    service.start_session()
    barrier(service)
    other = UsageStatsService(service.path, enabled=True, provenance={})
    try:
        assert barrier(other)["counts"]["running"] == 1
    finally:
        assert other.close(timeout=5)
    assert service.close(timeout=5)
    restarted = UsageStatsService(service.path, enabled=True, provenance={})
    try:
        summary = barrier(restarted)
        assert summary["counts"]["interrupted"] == 1
        row = restarted.export_rows().result(timeout=5)[0]
        assert row["finished_at"] is None
        assert row["capture_seconds"] is None
    finally:
        assert restarted.close(timeout=5)


def lock_names(service):
    return sorted(
        path.stem
        for path in (service.path.parent / "statistics.sqlite3.runs").iterdir()
    )


def test_clean_close_removes_only_finished_run_lock(tmp_path):
    path = tmp_path / "statistics.sqlite3"
    finished = UsageStatsService(path, enabled=True, provenance={})
    barrier(finished)
    finished.start_session().finish("success")
    assert finished.close(timeout=5)
    assert lock_names(finished) == []
    unfinished = UsageStatsService(path, enabled=True, provenance={})
    barrier(unfinished)
    unfinished.start_session()
    assert unfinished.close(timeout=5)
    assert lock_names(unfinished) == [unfinished.run_id]


def test_startup_prunes_exited_locks_but_keeps_live_owner(service):
    service.start_session()
    barrier(service)
    lock_dir = service.path.parent / "statistics.sqlite3.runs"
    (lock_dir / "exited-run.lock").write_bytes(b"\0")
    other = UsageStatsService(service.path, enabled=True, provenance={})
    try:
        barrier(other)
        assert lock_names(other) == [service.run_id]
    finally:
        assert other.close(timeout=5)


def test_reconciled_crash_lock_is_pruned(tmp_path):
    path = tmp_path / "statistics.sqlite3"
    crashed = UsageStatsService(path, enabled=True, provenance={})
    barrier(crashed)
    crashed.start_session()
    assert crashed.close(timeout=5)
    restarted = UsageStatsService(path, enabled=True, provenance={})
    try:
        assert barrier(restarted)["counts"]["interrupted"] == 1
        assert lock_names(restarted) == []
    finally:
        assert restarted.close(timeout=5)


def test_runs_without_sessions_leave_no_run_row_or_lock(tmp_path):
    path = tmp_path / "statistics.sqlite3"
    first = UsageStatsService(path, enabled=True, provenance={})
    barrier(first)
    first.start_session().finish("success")
    assert first.close(timeout=5)
    for enabled in (False, True):
        idle = UsageStatsService(path, enabled=enabled, provenance={})
        try:
            assert barrier(idle)["sessions"] == 1
            assert lock_names(idle) == []
        finally:
            assert idle.close(timeout=5)
    with sqlite3.connect(path) as connection:
        runs = connection.execute("SELECT run_id FROM runs").fetchall()
    assert runs == [(first.run_id,)]


def test_uncertain_owner_preserves_active_history(service, monkeypatch):
    service.start_session()
    barrier(service)
    assert service.close(timeout=5)
    monkeypatch.setattr("src.usage_stats._owner_dead", lambda path: None)
    other = UsageStatsService(service.path, enabled=True, provenance={})
    try:
        summary = barrier(other)
        assert summary["counts"]["running"] == 1
        assert summary["reconciliation_incomplete"] is True
    finally:
        assert other.close(timeout=5)


def test_real_os_lock_released_by_exiting_helper(tmp_path):
    import sys

    path = tmp_path / "owner.lock"
    script = "from pathlib import Path; from src.usage_stats import _RunLock; lock=_RunLock(Path(__import__('sys').argv[1])); print('ready',flush=True); input()"
    helper = subprocess.Popen(
        [sys.executable, "-c", script, str(path)],
        stdin=subprocess.PIPE,
        stdout=subprocess.PIPE,
        text=True,
    )
    try:
        assert helper.stdout.readline().strip() == "ready"
        assert _owner_dead(path) is False
        helper.communicate("exit\n", timeout=5)
        assert helper.returncode == 0
        assert _owner_dead(path) is True
    finally:
        if helper.poll() is None:
            helper.kill()
            helper.wait(timeout=5)


def test_busy_database_failure_recovers_without_replay(service):
    with sqlite3.connect(service.path, timeout=0.1) as connection:
        connection.execute("BEGIN IMMEDIATE")
        service.start_session().finish("success")
        with pytest.raises(RuntimeError, match="storage unavailable"):
            service.clear().result(timeout=5)
    # Failed snapshots are deliberately not replayed.
    assert barrier(service)["sessions"] == 0
    service.start_session().finish("success")
    summary = barrier(service)
    assert summary["sessions"] == 1
    assert summary["incomplete"] is True
    assert summary["storage_available"] is True
    service.clear().result(timeout=5)
    assert barrier(service)["incomplete"] is False


def test_newer_schema_is_preserved(tmp_path):
    path = tmp_path / "future.sqlite3"
    with sqlite3.connect(path) as connection:
        connection.execute("CREATE TABLE precious (value TEXT)")
        connection.execute("INSERT INTO precious VALUES ('keep')")
        connection.execute("PRAGMA user_version=99")
    stats = UsageStatsService(path, enabled=True, provenance={})
    try:
        with pytest.raises(RuntimeError, match="storage unavailable"):
            stats.summary().result(timeout=5)
        assert stats.start_session() is None
    finally:
        assert stats.close(timeout=5)
    with sqlite3.connect(path) as connection:
        assert connection.execute("SELECT value FROM precious").fetchone()[0] == "keep"
        assert connection.execute("PRAGMA user_version").fetchone()[0] == 99


def test_bounded_nonblocking_snapshots_and_controls(service, monkeypatch):
    # Exercise overflow without making correctness depend on 128 disk commits.
    monkeypatch.setattr(service, "_snapshot_slots", threading.BoundedSemaphore(2))
    entered = threading.Event()
    release = threading.Event()
    timed_out = threading.Event()
    original = service._write

    def blocked_write(connection, snapshot):
        entered.set()
        if not release.wait(timeout=5):
            timed_out.set()
        original(connection, snapshot)

    monkeypatch.setattr(service, "_write", blocked_write)
    collector = service.start_session()
    assert entered.wait(timeout=5)
    try:
        # Distinct sessions exceed the test's two slots while writes are blocked.
        for _ in range(3):
            service.start_session()
        collector.finish("success")
        first = service.summary()
        duplicate = service.summary()
        with pytest.raises(StatisticsBusyError):
            duplicate.result(timeout=1)
        exported = service.export_rows()
        cleared = service.clear()
        assert not first.done()
    finally:
        release.set()
    assert first.result(timeout=5)["incomplete"] is True
    assert exported.result(timeout=5)
    cleared.result(timeout=5)
    assert barrier(service)["sessions"] == 0
    assert not timed_out.is_set(), (
        "Snapshot admission blocked until the writer timed out"
    )


def test_long_session_updates_collapse_to_newest_write(service, monkeypatch):
    entered = threading.Event()
    release = threading.Event()
    writes = []
    original = service._write

    def counted_write(connection, snapshot):
        writes.append(snapshot)
        entered.set()
        assert release.wait(timeout=5)
        original(connection, snapshot)

    monkeypatch.setattr(service, "_write", counted_write)
    collector = service.start_session()
    assert entered.wait(timeout=5)
    try:
        # A slow disk while a long dictation sends hundreds of updates.
        for index in range(300):
            collector.stage("live_whisper", "success", seconds=0.01)
            collector.update(capture_seconds=index)
        collector.finish("success", final_word_count=5)
    finally:
        release.set()
    summary = barrier(service)
    assert summary["incomplete"] is False
    assert summary["stages"]["live_whisper"]["success_count"] == 300
    assert summary["delivered_words"] == 5
    assert len(writes) == 2  # First snapshot, then one write with the newest state.


def test_locked_final_write_is_retried_once(service, monkeypatch):
    monkeypatch.setattr("src.usage_stats._FINAL_WRITE_RETRY_SECONDS", 0)
    original = service._write
    attempts = []

    def locked_once(connection, snapshot):
        attempts.append(snapshot.outcome)
        if snapshot.outcome == "success" and attempts.count("success") == 1:
            error = sqlite3.OperationalError("database is locked")
            error.sqlite_errorcode = sqlite3.SQLITE_BUSY
            raise error
        original(connection, snapshot)

    collector = service.start_session()
    barrier(service)
    monkeypatch.setattr(service, "_write", locked_once)
    collector.finish("success")
    summary = barrier(service)
    assert attempts == ["success", "success"]
    assert summary["counts"]["success"] == 1
    assert summary["counts"]["running"] == 0
    assert summary["incomplete"] is False


@pytest.mark.parametrize(
    ("error_code", "expected_attempts"),
    [(sqlite3.SQLITE_BUSY, 2), (sqlite3.SQLITE_READONLY, 1)],
)
def test_final_write_retries_only_contention_and_stays_bounded(
    service, monkeypatch, error_code, expected_attempts
):
    monkeypatch.setattr("src.usage_stats._FINAL_WRITE_RETRY_SECONDS", 0)
    collector = service.start_session()
    barrier(service)
    attempts = []

    def fail(connection, snapshot):
        attempts.append(snapshot.outcome)
        error = sqlite3.OperationalError("synthetic storage failure")
        error.sqlite_errorcode = error_code
        raise error

    monkeypatch.setattr(service, "_write", fail)
    collector.finish("success")
    summary = barrier(service)
    assert attempts == ["success"] * expected_attempts
    assert summary["incomplete"] is True


def test_export_rows_are_chronological(service, monkeypatch):
    from uuid import UUID

    identifiers = iter(UUID(int=value) for value in range(5, 0, -1))
    monkeypatch.setattr("src.usage_stats.uuid.uuid4", lambda: next(identifiers))
    for _ in range(5):
        service.start_session().finish("success")
    rows = service.export_rows().result(timeout=5)
    started = [row["started_at"] for row in rows]
    assert started == sorted(started)


def test_export_rows_stay_raw_and_unknown_codes_are_fixed(service):
    collector = service.start_session()
    collector.update(whisper_model="=1+1", cleanup_model=" @bad")
    collector.issue("capture", "raw secret exception")
    collector.finish("failed")
    row = service.export_rows().result(timeout=5)[0]
    # Formula escaping belongs to the CSV writer, not the export rows.
    assert row["whisper_model"] == "=1+1"
    assert row["cleanup_model"] == " @bad"
    assert json.loads(row["issues"])[0]["code"] == "unknown_error"
    with sqlite3.connect(service.path) as connection:
        payload = connection.execute("SELECT payload FROM sessions").fetchone()[0]
        assert json.loads(payload)["metadata"]["whisper_model"] == "=1+1"
    assert b"raw secret exception" not in service.path.read_bytes()


def test_close_is_bounded_and_rejects_requests(service):
    with pytest.raises(ValueError, match="finite"):
        service.close(float("inf"))
    assert service.close(timeout=5)
    assert service.start_session() is None
    with pytest.raises(StatisticsBusyError):
        service.summary().result(timeout=1)


def test_start_before_generation_is_nonblocking_and_marks_gap(tmp_path, monkeypatch):
    entered = threading.Event()
    release = threading.Event()
    original = UsageStatsService._open

    def delayed_open(self):
        entered.set()
        assert release.wait(timeout=5)
        return original(self)

    monkeypatch.setattr(UsageStatsService, "_open", delayed_open)
    stats = UsageStatsService(tmp_path / "stats.sqlite3", enabled=True, provenance={})
    try:
        assert entered.wait(timeout=5)
        assert stats.start_session() is None
        stats.set_enabled(False)
        release.set()
        assert barrier(stats)["incomplete"] is True
        assert stats.start_session() is None
        stats.set_enabled(True)
        stats.start_session().finish("success", capture_seconds=0, final_word_count=0)
        summary = barrier(stats)
        assert summary["captured_minutes"] == 0
        assert summary["recording_count"] == 1
        assert summary["delivered_words"] == 0
        assert summary["delivered_word_sample_count"] == 1
    finally:
        release.set()
        assert stats.close(timeout=5)


def test_p95_uses_nearest_rank_and_successful_delivery_only(service):
    for seconds in range(1, 21):
        service.start_session().finish("success", stop_to_clipboard_seconds=seconds)
        barrier(service)
    service.start_session().finish("failed", stop_to_clipboard_seconds=1000)
    latency = barrier(service)["latency"]
    assert latency == {
        "count": 20,
        "median_seconds": 10.5,
        "p95_seconds": 19,
        "small_sample": False,
    }


def test_missed_terminal_is_not_rebound_after_clear(service):
    active = service.start_session()
    barrier(service)
    service.clear().result(timeout=5)
    active.finish("success")
    assert barrier(service)["sessions"] == 0
    service.start_session().finish("success")
    assert barrier(service)["sessions"] == 1


def test_failed_attempt_mean_does_not_average_session_totals(service):
    collector = service.start_session()
    collector.stage("live_whisper", "failed", seconds=1)
    collector.stage("live_whisper", "failed", seconds=3)
    collector.stage("live_whisper", "failed", seconds=None)
    collector.stage("live_whisper", "success", seconds=7)
    collector.finish("success")
    stage = barrier(service)["stages"]["live_whisper"]
    assert stage["failed_count"] == 3
    assert stage["failed_seconds_count"] == 2
    assert stage["mean_failed_seconds"] == 2
    assert stage["mean_seconds"] == 7
