"""Content-free local usage statistics. All disk access belongs to one worker.

``summary().result()`` is also a FIFO barrier for previously accepted snapshots.
Unknown measurements remain None. Words are supplied as len(final_text.split()).
Model names are the only caller-supplied free text; never pass transcript content.
"""

from __future__ import annotations

import errno
import json
import math
import queue
import sqlite3
import statistics
import sys
import threading
import uuid
from collections import Counter
from concurrent.futures import Future
from dataclasses import asdict, dataclass, replace
from datetime import UTC, datetime
from pathlib import Path
from typing import Literal

STAGES = (
    "capture",
    "preparation",
    "live_whisper",
    "offline_segmentation",
    "offline_whisper",
    "local_cleanup",
    "ollama_cleanup",
    "clipboard",
)
StageOutcome = Literal["running", "success", "failed", "skipped", "cancelled"]
SessionOutcome = Literal["success", "failed", "no_speech", "cancelled", "interrupted"]
STAGE_OUTCOMES = {"running", "success", "failed", "skipped", "cancelled"}
TERMINAL_OUTCOMES = {"success", "failed", "no_speech", "cancelled", "interrupted"}
ISSUE_CODES = frozenset(
    {
        "unknown_error",
        "capture_start_failed",
        "capture_failed",
        "no_audio",
        "live_start_failed",
        "live_degraded",
        "live_retry_exhausted",
        "live_empty",
        "vad_unavailable",
        "vad_error",
        "vad_no_segments",
        "whisper_failed",
        "local_cleanup_failed",
        "cleanup_unavailable",
        "cleanup_timeout",
        "cleanup_request_failed",
        "cleanup_context_skip",
        "cleanup_output_rejected",
        "cleanup_disabled",
        "cleanup_empty",
        "clipboard_failed",
    }
)
CATEGORIES = {"error", "unavailable", "routing", "skip"}
SCHEMA_VERSION = 1


def _utc() -> str:
    return datetime.now(UTC).isoformat()


@dataclass(frozen=True)
class SessionMetadata:
    whisper_model: str | None = None
    whisper_device: str | None = None
    cleanup_enabled: bool | None = None
    cleanup_model: str | None = None
    cleanup_used_model: str | None = None
    capture_seconds: float | None = None
    final_word_count: int | None = None
    stop_to_clipboard_seconds: float | None = None
    transcription_route: str | None = None
    reached_transcription: bool | None = None
    error_fallback: bool | None = None


def _metadata(current: SessionMetadata, values: dict) -> SessionMetadata:
    for key, value in values.items():
        if key not in SessionMetadata.__dataclass_fields__:
            raise ValueError(f"Unsupported statistics field: {key}")
        if value is None:
            continue
        if key.endswith("seconds"):
            valid = type(value) in (int, float) and math.isfinite(value) and value >= 0
        elif key == "final_word_count":
            valid = type(value) is int and value >= 0
        elif key in {"cleanup_enabled", "reached_transcription", "error_fallback"}:
            valid = type(value) is bool
        elif key == "transcription_route":
            valid = value in {"live", "offline_segmented", "full_clip"}
        elif key == "whisper_device":
            valid = value in {"cpu", "cuda", "mps"}
        else:
            valid = isinstance(value, str) and len(value) <= 256
        if not valid:
            raise ValueError(f"Invalid statistics field: {key}")
    return replace(current, **values)


@dataclass(frozen=True)
class StageSnapshot:
    name: str
    outcome: StageOutcome
    seconds: float | None = None
    success_seconds: float | None = None
    failed_seconds: float | None = None
    attempts: int = 0
    success_count: int = 0
    failed_count: int = 0
    skipped_count: int = 0
    failed_timed_attempts: int = 0


@dataclass(frozen=True)
class IssueSnapshot:
    stage: str
    code: str
    category: str
    recovered: bool
    count: int = 1


@dataclass(frozen=True)
class SessionSnapshot:
    session_id: str
    run_id: str
    generation: int
    started_at: str
    finished_at: str | None
    outcome: str
    metadata: SessionMetadata
    stages: tuple[StageSnapshot, ...]
    issues: tuple[IssueSnapshot, ...]


def _add(left: float | None, right: float | None) -> float | None:
    return left if right is None else (left or 0) + right


class SessionCollector:
    """Accumulate bounded stage/issue totals; terminal ownership is immutable."""

    def __init__(self, service: UsageStatsService, generation: int):
        self._service = service
        self._lock = threading.Lock()
        self._snapshot = SessionSnapshot(
            str(uuid.uuid4()),
            service.run_id,
            generation,
            _utc(),
            None,
            "running",
            SessionMetadata(),
            (),
            (),
        )

    def _publish(self):
        self._service._submit(self._snapshot)

    def update(self, **allowed_metadata):
        with self._lock:
            if self._snapshot.outcome != "running":
                return
            self._snapshot = replace(
                self._snapshot,
                metadata=_metadata(self._snapshot.metadata, allowed_metadata),
            )
            self._publish()

    def stage(self, name, outcome, seconds=None, code=None, category=None, attempts=1):
        if name not in STAGES or outcome not in STAGE_OUTCOMES:
            raise ValueError("Invalid statistics stage/outcome")
        if seconds is not None and (
            type(seconds) not in (int, float)
            or not math.isfinite(seconds)
            or seconds < 0
        ):
            raise ValueError("Invalid stage duration")
        if type(attempts) is not int or attempts < 0:
            raise ValueError("Invalid attempt count")
        with self._lock:
            if self._snapshot.outcome != "running":
                return
            stages = {stage.name: stage for stage in self._snapshot.stages}
            old = stages.get(name, StageSnapshot(name, outcome))
            stages[name] = StageSnapshot(
                name,
                outcome,
                _add(old.seconds, seconds),
                _add(old.success_seconds, seconds if outcome == "success" else None),
                _add(old.failed_seconds, seconds if outcome == "failed" else None),
                old.attempts + attempts,
                old.success_count + (attempts if outcome == "success" else 0),
                old.failed_count + (attempts if outcome == "failed" else 0),
                old.skipped_count + (attempts if outcome == "skipped" else 0),
                old.failed_timed_attempts
                + (attempts if outcome == "failed" and seconds is not None else 0),
            )
            self._snapshot = replace(self._snapshot, stages=tuple(stages.values()))
            if code is not None:
                self._issue(name, code, category or "error", False)
            self._publish()

    def _issue(self, stage, code, category, recovered):
        if (
            stage not in STAGES
            or category not in CATEGORIES
            or type(recovered) is not bool
        ):
            raise ValueError("Invalid statistics issue")
        code = code if code in ISSUE_CODES else "unknown_error"
        issues = {
            (i.stage, i.code, i.category, i.recovered): i for i in self._snapshot.issues
        }
        key = stage, code, category, recovered
        old = issues.get(key)
        issues[key] = IssueSnapshot(*key, count=old.count + 1 if old else 1)
        self._snapshot = replace(self._snapshot, issues=tuple(issues.values()))

    def issue(self, stage, code, category="error", recovered=False):
        with self._lock:
            if self._snapshot.outcome != "running":
                return
            self._issue(stage, code, category, recovered)
            self._publish()

    def issue_once(self, stage, code, category="error"):
        """Record a terminal issue unless an observer already counted it."""
        with self._lock:
            if self._snapshot.outcome != "running" or any(
                issue.stage == stage and issue.code == code
                for issue in self._snapshot.issues
            ):
                return
            self._issue(stage, code, category, False)
            self._publish()

    def finish(self, outcome, **allowed_metadata):
        if outcome not in TERMINAL_OUTCOMES:
            raise ValueError("Invalid terminal outcome")
        with self._lock:
            if self._snapshot.outcome != "running":
                return
            issues = {}
            for issue in self._snapshot.issues:
                recovered = issue.recovered or (
                    outcome == "success" and issue.category in {"error", "unavailable"}
                )
                key = issue.stage, issue.code, issue.category, recovered
                previous = issues.get(key)
                issues[key] = replace(
                    issue,
                    recovered=recovered,
                    count=issue.count + (previous.count if previous else 0),
                )
            self._snapshot = replace(
                self._snapshot,
                outcome=outcome,
                finished_at=_utc(),
                metadata=_metadata(self._snapshot.metadata, allowed_metadata),
                issues=tuple(issues.values()),
            )
            self._publish()


class _RunLock:
    """Hold a one-byte OS file lock; probe errors never establish owner death."""

    def __init__(self, path: Path):
        self.handle = path.open("a+b")
        try:
            self.handle.seek(0, 2)
            if not self.handle.tell():
                self.handle.write(b"\0")
                self.handle.flush()
            self.handle.seek(0)
            if sys.platform == "win32":
                import msvcrt

                msvcrt.locking(self.handle.fileno(), msvcrt.LK_NBLCK, 1)
            else:
                import fcntl

                fcntl.flock(self.handle.fileno(), fcntl.LOCK_EX | fcntl.LOCK_NB)
        except BaseException:
            self.handle.close()
            raise

    def close(self):
        self.handle.close()


def _owner_dead(path: Path) -> bool | None:
    # A missing file is uncertain, since it may be inaccessible or moved.
    if not path.is_file():
        return None
    try:
        probe = _RunLock(path)
    except OSError as exc:
        return (
            False if exc.errno in {errno.EACCES, errno.EAGAIN, errno.EDEADLK} else None
        )
    except ImportError:
        return None
    probe.close()
    return True


class StatisticsBusyError(RuntimeError):
    """The bounded control admission path is busy or closed."""


class UsageStatsService:
    """One worker/connection, with bounded nonblocking admission.

    Production callers use get_usage_stats(); explicit instances support isolated
    tests and independent-instance deletion/reconciliation tests. summary keys:
    counts, success_rate, fallback_count/denominator/rate, captured_minutes,
    recording_count/mean_capture_seconds, delivered_words, latency (count,
    median_seconds, p95_seconds, small_sample), stages, model_groups,
    cleanup_groups, issues, recovered_failures, active_sessions, incomplete,
    reconciliation_incomplete, storage_available, storage_error, enabled, range.
    Rates are fractions; empty numeric samples and zero denominators are None.
    """

    def __init__(self, path=None, enabled=None, provenance=None):
        from .config import get_statistics_db_path

        if enabled is not None and type(enabled) is not bool:
            raise ValueError("enabled must be bool or None")
        self.path = Path(path) if path is not None else get_statistics_db_path()
        self.run_id = str(uuid.uuid4())
        self._enabled = enabled
        self._provenance = dict(provenance) if provenance is not None else None
        self._generation = None
        self._incomplete = False
        self._reconciliation_incomplete = False
        self._storage_error = None
        self._closed = False
        self._state_lock = threading.Lock()
        self._queue = queue.Queue()
        self._snapshot_slots = threading.BoundedSemaphore(128)
        self._control_slots = threading.BoundedSemaphore(8)
        self._pending = set()
        self._admission_lock = threading.Lock()
        self._thread = threading.Thread(
            target=self._work, name="usage-statistics", daemon=True
        )
        self._thread.start()

    def set_enabled(self, enabled: bool):
        if type(enabled) is not bool:
            raise ValueError("enabled must be bool")
        with self._state_lock:
            self._enabled = enabled
            needs_open = enabled and self._generation is None
        if needs_open:
            # Open storage before the next session; busy or closed is harmless.
            self._request("retry")

    def start_session(self) -> SessionCollector | None:
        with self._state_lock:
            if self._closed or self._enabled is False:
                return None
            generation = self._generation
            if generation is None:
                self._incomplete = True
                # Bounded recovery request, with no disk access on this caller.
                self._request("retry")
                return None
        collector = SessionCollector(self, generation)
        collector._publish()
        return collector

    def _submit(self, snapshot):
        with self._state_lock:
            if self._closed or not self._snapshot_slots.acquire(blocking=False):
                self._incomplete = True
                return
            self._queue.put(("snapshot", snapshot, None))

    def _request(self, kind):
        future = Future()
        # Admission is called under the state lock only from start_session.
        with self._admission_lock:
            if (
                self._closed
                or kind in self._pending
                or not self._control_slots.acquire(False)
            ):
                future.set_exception(
                    StatisticsBusyError("Statistics request busy or closed")
                )
                return future
            self._pending.add(kind)
            self._queue.put((kind, None, future))
        return future

    def summary(self) -> Future[dict]:
        """Read aggregates after prior submissions; storage failure fails the Future."""
        return self._request("summary")

    def export_rows(self) -> Future[list[dict]]:
        """CSV-ready dictionaries in EXPORT_COLUMNS order, with sorted JSON issues.

        Unknown values remain None. Model identities are escaped for spreadsheets
        in the export only. Stage columns start with stage_ to avoid collisions.
        """
        return self._request("export")

    def clear(self) -> Future:
        return self._request("clear")

    def close(self, timeout=1.0):
        if not math.isfinite(timeout) or timeout < 0:
            raise ValueError("close requires a finite nonnegative timeout")
        with self._state_lock:
            if not self._closed:
                with self._admission_lock:
                    self._closed = True
                    self._queue.put(("close", None, None))
        if threading.current_thread() is self._thread:
            return False
        self._thread.join(min(timeout, threading.TIMEOUT_MAX))
        return not self._thread.is_alive()

    def _open(self):
        self.path.parent.mkdir(parents=True, exist_ok=True)
        lock_dir = self.path.parent / (self.path.name + ".runs")
        lock_dir.mkdir(exist_ok=True)
        run_lock = _RunLock(lock_dir / (self.run_id + ".lock"))
        connection = None
        try:
            connection = sqlite3.connect(self.path, timeout=0.1)
            version = connection.execute("PRAGMA user_version").fetchone()[0]
            if version not in (0, SCHEMA_VERSION):
                raise RuntimeError("unsupported_schema")
            with connection:
                connection.execute(
                    "CREATE TABLE IF NOT EXISTS metadata (id INTEGER PRIMARY KEY CHECK(id=1), generation INTEGER NOT NULL, incomplete INTEGER NOT NULL)"
                )
                connection.execute("INSERT OR IGNORE INTO metadata VALUES (1,0,0)")
                connection.execute(
                    "CREATE TABLE IF NOT EXISTS runs (run_id TEXT PRIMARY KEY, app_version TEXT, git_commit TEXT, git_dirty INTEGER)"
                )
                connection.execute(
                    "CREATE TABLE IF NOT EXISTS sessions (session_id TEXT PRIMARY KEY, run_id TEXT NOT NULL, outcome TEXT NOT NULL, payload TEXT NOT NULL)"
                )
                connection.execute(
                    "CREATE TABLE IF NOT EXISTS stages (session_id TEXT, name TEXT, payload TEXT NOT NULL, PRIMARY KEY(session_id,name))"
                )
                connection.execute(
                    "CREATE TABLE IF NOT EXISTS issues (session_id TEXT, stage TEXT, code TEXT, category TEXT, recovered INTEGER, count INTEGER, PRIMARY KEY(session_id,stage,code,category,recovered))"
                )
                connection.execute(f"PRAGMA user_version={SCHEMA_VERSION}")
            if self._provenance is None:
                try:
                    from .build_info import get_build_info

                    self._provenance = get_build_info()
                except Exception:
                    self._provenance = {}
            info = self._provenance
            with connection:
                connection.execute("BEGIN IMMEDIATE")
                generation = connection.execute(
                    "SELECT generation FROM metadata"
                ).fetchone()[0]
                for (run_id,) in connection.execute(
                    "SELECT DISTINCT run_id FROM sessions WHERE outcome='running'"
                ).fetchall():
                    if run_id == self.run_id:
                        continue
                    dead = _owner_dead(lock_dir / (run_id + ".lock"))
                    if dead is None:
                        self._reconciliation_incomplete = True
                    if dead:
                        for session_id, payload in connection.execute(
                            "SELECT session_id,payload FROM sessions WHERE run_id=? AND outcome='running'",
                            (run_id,),
                        ).fetchall():
                            data = json.loads(payload)
                            data["outcome"] = "interrupted"
                            # Crash time is unknown; never fabricate a finish time.
                            connection.execute(
                                "UPDATE sessions SET outcome='interrupted',payload=? WHERE session_id=?",
                                (json.dumps(data), session_id),
                            )
                connection.execute(
                    "INSERT OR IGNORE INTO runs VALUES (?,?,?,?)",
                    (
                        self.run_id,
                        info.get("app_version"),
                        info.get("git_commit"),
                        info.get("git_dirty")
                        if type(info.get("git_dirty")) is bool
                        else None,
                    ),
                )
            self._generation = generation
            self._storage_error = None
            return connection, run_lock
        except BaseException:
            if connection is not None:
                connection.close()
            run_lock.close()
            raise

    def _work(self):
        connection = run_lock = None
        try:
            if self._enabled is None:
                from .config import get_config

                try:
                    enabled = get_config().enable_statistics
                except Exception:
                    enabled = False
                with self._state_lock:
                    if self._enabled is None:
                        self._enabled = enabled
            # Opted-out runs leave no database, run row, or lock file behind.
            if self._enabled:
                try:
                    connection, run_lock = self._open()
                except Exception:
                    self._storage_error = "storage_unavailable"
                    self._incomplete = True
            while True:
                kind, snapshot, future = self._queue.get()
                if kind == "close":
                    break
                result = failure = None
                if future is not None:
                    future.set_running_or_notify_cancel()
                try:
                    # Opted out with no history: answer empty without creating it.
                    rows, gap = [], 0
                    if connection is None and (self._enabled or self.path.exists()):
                        connection, run_lock = self._open()
                    if connection is None:
                        pass
                    elif kind == "snapshot":
                        self._write(connection, snapshot)
                    elif kind == "clear":
                        with connection:
                            connection.execute("BEGIN IMMEDIATE")
                            connection.execute(
                                "UPDATE metadata SET generation=generation+1,incomplete=0"
                            )
                            for table in ("issues", "stages", "sessions"):
                                connection.execute(f"DELETE FROM {table}")
                            generation = connection.execute(
                                "SELECT generation FROM metadata"
                            ).fetchone()[0]
                        self._generation = generation
                        self._incomplete = False
                    elif kind in {"summary", "export"}:
                        with connection:
                            connection.execute("BEGIN")
                            rows = self._rows(connection)
                            gap = connection.execute(
                                "SELECT incomplete FROM metadata"
                            ).fetchone()[0]
                    self._storage_error = None
                    if kind == "summary":
                        result = self._summary(rows, gap)
                    elif kind == "export":
                        result = [_export_row(row) for row in rows]
                except Exception:
                    self._incomplete = True
                    self._storage_error = "storage_unavailable"
                    failure = RuntimeError("Statistics storage unavailable")
                finally:
                    if kind == "snapshot":
                        self._snapshot_slots.release()
                    else:
                        with self._admission_lock:
                            self._pending.discard(kind)
                            self._control_slots.release()
                if future is not None and not future.cancelled():
                    if failure is not None:
                        future.set_exception(failure)
                    else:
                        future.set_result(result)
        finally:
            if connection is not None:
                connection.close()
            if run_lock is not None:
                run_lock.close()

    def _write(self, connection, snapshot):
        with connection:
            connection.execute("BEGIN IMMEDIATE")
            generation = connection.execute(
                "SELECT generation FROM metadata"
            ).fetchone()[0]
            if generation != snapshot.generation:
                if self._generation != generation:
                    # Another instance deleted history; this session is lost.
                    self._incomplete = True
                self._generation = generation
                return
            old = connection.execute(
                "SELECT outcome FROM sessions WHERE session_id=?",
                (snapshot.session_id,),
            ).fetchone()
            if old and old[0] != "running":
                return
            payload = asdict(snapshot)
            payload.pop("stages")
            payload.pop("issues")
            connection.execute(
                "INSERT OR REPLACE INTO sessions VALUES (?,?,?,?)",
                (
                    snapshot.session_id,
                    snapshot.run_id,
                    snapshot.outcome,
                    json.dumps(payload),
                ),
            )
            for stage in snapshot.stages:
                connection.execute(
                    "INSERT OR REPLACE INTO stages VALUES (?,?,?)",
                    (snapshot.session_id, stage.name, json.dumps(asdict(stage))),
                )
            connection.execute(
                "DELETE FROM issues WHERE session_id=?", (snapshot.session_id,)
            )
            for issue in snapshot.issues:
                connection.execute(
                    "INSERT OR REPLACE INTO issues VALUES (?,?,?,?,?,?)",
                    (
                        snapshot.session_id,
                        issue.stage,
                        issue.code,
                        issue.category,
                        issue.recovered,
                        issue.count,
                    ),
                )
            if self._incomplete:
                connection.execute("UPDATE metadata SET incomplete=1")

    def _rows(self, connection):
        rows = []
        for session_id, payload, version, commit, dirty in connection.execute(
            "SELECT s.session_id,s.payload,r.app_version,r.git_commit,r.git_dirty FROM sessions s JOIN runs r ON s.run_id=r.run_id ORDER BY s.session_id"
        ):
            data = json.loads(payload)
            metadata = data.pop("metadata")
            data.pop("generation")
            data.update(metadata)
            data.update(
                app_version=version,
                git_commit=commit,
                git_dirty=None if dirty is None else bool(dirty),
            )
            data["stages"] = {
                name: json.loads(value)
                for name, value in connection.execute(
                    "SELECT name,payload FROM stages WHERE session_id=? ORDER BY name",
                    (session_id,),
                )
            }
            data["issues"] = [
                {"stage": s, "code": c, "category": k, "recovered": bool(r), "count": n}
                for s, c, k, r, n in connection.execute(
                    "SELECT stage,code,category,recovered,count FROM issues WHERE session_id=? ORDER BY stage,code,category,recovered",
                    (session_id,),
                )
            ]
            rows.append(data)
        return rows

    def _summary(self, rows, gap):
        counts = dict.fromkeys(("running", *sorted(TERMINAL_OUTCOMES)), 0)
        for row in rows:
            counts[row["outcome"]] += 1
        captures = [
            r["capture_seconds"] for r in rows if r["capture_seconds"] is not None
        ]
        words = [
            r["final_word_count"]
            for r in rows
            if r["outcome"] == "success" and r["final_word_count"] is not None
        ]
        eligible = [
            r
            for r in rows
            if r["outcome"] in {"success", "failed", "no_speech"}
            and r["reached_transcription"] is True
        ]
        fallback_count = sum(r["error_fallback"] is True for r in eligible)
        issues = Counter()
        for row in rows:
            for issue in row["issues"]:
                issues[issue["code"]] += issue["count"]
        groups = {}
        cleanups = {}
        for row in rows:
            key = tuple(
                row[k]
                for k in (
                    "whisper_model",
                    "whisper_device",
                    "cleanup_enabled",
                    "cleanup_model",
                    "cleanup_used_model",
                )
            )
            groups.setdefault(key, []).append(row)
            cleanups.setdefault(
                row["cleanup_used_model"] or row["cleanup_model"], []
            ).append(row)
        return {
            "range": "all retained history",
            "sessions": len(rows),
            "counts": counts,
            "active_sessions": counts["running"],
            "success_rate": _ratio(
                counts["success"], counts["success"] + counts["failed"]
            ),
            "fallback_count": fallback_count,
            "fallback_denominator": len(eligible),
            "fallback_rate": _ratio(fallback_count, len(eligible)),
            "normal_routing_counts": {
                code: sum(any(i["code"] == code for i in r["issues"]) for r in rows)
                for code in ("live_empty", "vad_no_segments")
            },
            "captured_minutes": sum(captures) / 60 if captures else None,
            "recording_count": len(captures),
            "mean_capture_seconds": _mean(captures),
            "delivered_words": sum(words) if words else None,
            "delivered_word_sample_count": len(words),
            "latency": _latency(rows),
            "stages": _stage_summary(rows),
            "model_groups": [
                dict(
                    zip(
                        (
                            "whisper_model",
                            "whisper_device",
                            "cleanup_enabled",
                            "cleanup_model",
                            "cleanup_used_model",
                        ),
                        key,
                        strict=True,
                    ),
                    sessions=len(group),
                    recording_count=sum(
                        r["capture_seconds"] is not None for r in group
                    ),
                    mean_capture_seconds=_mean(
                        [
                            r["capture_seconds"]
                            for r in group
                            if r["capture_seconds"] is not None
                        ]
                    ),
                    inference_work_seconds=_mean(
                        [
                            sum(
                                s["seconds"]
                                for name, s in r["stages"].items()
                                if name in {"live_whisper", "offline_whisper"}
                                and s["seconds"] is not None
                            )
                            for r in group
                            if any(
                                s["seconds"] is not None
                                for name, s in r["stages"].items()
                                if name in {"live_whisper", "offline_whisper"}
                            )
                        ]
                    ),
                    latency=_latency(group),
                )
                for key, group in groups.items()
            ],
            "cleanup_groups": [
                dict(model=model, **_stage_summary(group)["ollama_cleanup"])
                for model, group in cleanups.items()
            ],
            "issues": dict(sorted(issues.items())),
            "recovered_failures": sum(
                i["count"]
                for r in rows
                for i in r["issues"]
                if i["recovered"] and i["category"] in {"error", "unavailable"}
            ),
            "incomplete": bool(gap or self._incomplete),
            "reconciliation_incomplete": self._reconciliation_incomplete,
            "storage_available": self._storage_error is None,
            "storage_error": self._storage_error,
            "enabled": self._enabled,
        }


def _ratio(numerator, denominator):
    return numerator / denominator if denominator else None


def _mean(values):
    return statistics.mean(values) if values else None


def _latency(rows):
    values = sorted(
        r["stop_to_clipboard_seconds"]
        for r in rows
        if r["outcome"] == "success" and r["stop_to_clipboard_seconds"] is not None
    )
    return {
        "count": len(values),
        "median_seconds": statistics.median(values) if values else None,
        "p95_seconds": values[math.ceil(0.95 * len(values)) - 1] if values else None,
        "small_sample": len(values) < 20,
    }


def _stage_summary(rows):
    result = {}
    for name in STAGES:
        stages = [r["stages"][name] for r in rows if name in r["stages"]]
        success = [
            s["success_seconds"] for s in stages if s["success_seconds"] is not None
        ]
        failed = [
            s["failed_seconds"] for s in stages if s["failed_seconds"] is not None
        ]
        failed_attempts = sum(s["failed_timed_attempts"] for s in stages)
        result[name] = {
            "count": len(stages),
            "attempts": sum(s["attempts"] for s in stages),
            "success_count": sum(s["success_count"] for s in stages),
            "failed_count": sum(s["failed_count"] for s in stages),
            "skipped_count": sum(s["skipped_count"] for s in stages),
            "seconds_count": len(success),
            "mean_seconds": _mean(success),
            "failed_seconds_count": failed_attempts,
            "mean_failed_seconds": _ratio(sum(failed), failed_attempts),
        }
    return result


_SESSION_COLUMNS = (
    "session_id",
    "run_id",
    "started_at",
    "finished_at",
    "outcome",
    *SessionMetadata.__dataclass_fields__,
    "app_version",
    "git_commit",
    "git_dirty",
)
_STAGE_COLUMNS = (
    "outcome",
    "seconds",
    "success_seconds",
    "failed_seconds",
    "attempts",
    "success_count",
    "failed_count",
    "skipped_count",
)
EXPORT_COLUMNS = (
    *_SESSION_COLUMNS,
    *(f"stage_{stage}_{field}" for stage in STAGES for field in _STAGE_COLUMNS),
    "issues",
)


def _export_row(row):
    """Stable CSV-ready values. Issues use sorted JSON objects with counted codes.

    None is retained for csv.writer to render as an empty cell. Formula-leading
    model names get an apostrophe only in exported values, never in SQLite.
    """
    result = {key: row[key] for key in _SESSION_COLUMNS}
    for key in ("whisper_model", "cleanup_model", "cleanup_used_model"):
        value = result[key]
        if value and value.lstrip().startswith(("=", "+", "-", "@")):
            result[key] = "'" + value
    for stage in STAGES:
        for field in _STAGE_COLUMNS:
            result[f"stage_{stage}_{field}"] = row["stages"].get(stage, {}).get(field)
    result["issues"] = json.dumps(row["issues"], sort_keys=True, separators=(",", ":"))
    return result


_singleton = None
_singleton_lock = threading.Lock()


def get_usage_stats() -> UsageStatsService:
    """Return the process-wide service without doing disk work on this caller."""
    global _singleton
    with _singleton_lock:
        if _singleton is None:
            _singleton = UsageStatsService()
        return _singleton
