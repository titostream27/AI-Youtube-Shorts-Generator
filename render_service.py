"""Render service for the hybrid miner integration.

The youtube-content-miner already does discovery + scoring, so this service
only renders: download the source video once (cached), cut each clip to its
[start, end] and reframe it vertically (9:16 by default). No LLM calls, no
transcription — pure ffmpeg + OpenCV work.

Run:
    .venv/Scripts/python.exe render_service.py

Endpoints:
    GET  /health
    POST /api/render   {video_url, clips: [{clip_id, title, start_sec, end_sec}], aspect_ratio}
"""
import os
import threading
import time
import datetime
import uuid
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple, Union

from contextlib import asynccontextmanager
from fastapi import FastAPI, HTTPException, Request
from fastapi.responses import FileResponse, JSONResponse
from pydantic import BaseModel, Field

from shorts_generator.local.clipper import crop_highlights_local
from shorts_generator.local.downloader import download_youtube_local

from render_contract import (
    CONTRACT_VERSION,
    CaptionRequest,
    CaptionWord,
    ClipRequest,
    DurableJobSnapshot,
    EffectiveJobSnapshot,
    RenderArtifact,
    RenderArtifactResult,
    RenderJobStatus,
    RenderRequest,
    RenderRequestV2,
    RenderResponse,
    RenderSubmissionResponse,
    RenderJobStatusResponse,
)

RENDER_ROOT = Path(os.getenv("RENDER_OUTPUT_DIR", "rendered")).resolve()
HOST = os.getenv("RENDER_HOST", "127.0.0.1")
PORT = int(os.getenv("RENDER_PORT", "8084"))
FORMAT = os.getenv("RENDER_FORMAT", "2160")
# When exposed through a reverse proxy under a path prefix (e.g.
# hub.aelflab.com/short), strip that prefix so routes like /files/... match.
PATH_PREFIX = os.getenv("RENDER_PATH_PREFIX", "").strip().strip("/")


@asynccontextmanager
async def lifespan(app: FastAPI):
    """Brief v8 B3/R04 — startup correctness independent of __main__.

    Runs on BOTH `python render_service.py` and `uvicorn render_service:app`:
    1. Ensure the output root exists.
    2. Reconcile stale ACTIVE rows from a previous process -> orphaned.
    3. Start the queue worker so /readyz is truthful on an idle service.
    """
    RENDER_ROOT.mkdir(parents=True, exist_ok=True)
    print(f"[render] lifespan: output root {RENDER_ROOT}", flush=True)
    try:
        n = _reconcile_startup_orphans()
        print(f"[render] lifespan reconcile: {n} job(s) orphaned", flush=True)
    except Exception as e:  # noqa: BLE001
        print(f"[render] lifespan reconcile failed: {e}", flush=True)
    ensure_worker_running()
    yield
    # Optional shutdown: nothing durable to flush; worker is daemon.


app = FastAPI(title="Shorts Render Service", version="0.1.0", lifespan=lifespan)


@app.middleware("http")
async def strip_path_prefix(request: Request, call_next):
    if PATH_PREFIX:
        path = request.scope["path"]
        prefix = f"/{PATH_PREFIX}"
        if path == prefix:
            request.scope["path"] = "/"
        elif path.startswith(prefix + "/"):
            request.scope["path"] = path[len(prefix):]
    return await call_next(request)


# NOTE: Request/response models live in render_contract.py (versioned
# contract, Master Task Brief §16). v1 models are imported from there.


def _aspect_ratio_from(request) -> str:
    """Return '9:16' or the aspect_ratio when the request is legacy v1."""
    ratio = getattr(request, "aspect_ratio", "9:16")
    return ratio or "9:16"


def _estimate_upscale(source_path: str, out_w: int, out_h: int) -> float:
    """Approximate the upscale factor from source to output (brief §23)."""
    try:
        from visual_effects import probe_source_resolution
        probe = probe_source_resolution(source_path)
        if not probe:
            return 0.0
        sw, sh, _ = probe
        if sw <= 0 or sh <= 0:
            return 0.0
        # Vertical short: compare along the width (crop keeps source height).
        return round(min(1.0, out_w / sw), 2)
    except Exception:  # noqa: BLE001
        return 0.0


# ── Job persistence (Master Task Brief §19) ────────────────────────────────
# Render jobs are stored in a small SQLite DB (RENDER_JOB_DB, default
# rendered/render_jobs.db) so a service restart does not lose job status.
JOB_DB_PATH = Path(os.getenv("RENDER_JOB_DB", str(RENDER_ROOT / "render_jobs.db"))).resolve()
_job_db_conn = None

# Last sanitized persistence error (Phase 1 §5.6). Exposed via health so a
# degraded DB is visible to ops instead of failing silently.
_last_persist_error = None
_last_persist_error_at = None
_last_db_error = None
_last_db_error_at = None
_last_db_error_stage = None


def _record_db_error(stage: str, error: Exception) -> None:
    """Surface a DB read/write failure in structured health diagnostics.

    Phase-2 correctness (F9): _load_job / _load_job_request /
    _find_job_by_request previously swallowed SQLite errors and returned
    None — indistinguishable from a genuine not-found. Every failure is now
    recorded so /api/render/health can report db.read_errors separately.
    """
    global _last_db_error, _last_db_error_at, _last_db_error_stage
    import datetime
    _last_db_error = f"{type(error).__name__}: {error}"
    _last_db_error_at = datetime.datetime.utcnow().isoformat()
    _last_db_error_stage = stage
    # Keep the legacy write-error field in sync so older consumers still see it.
    global _last_persist_error, _last_persist_error_at
    _last_persist_error = _last_db_error
    _last_persist_error_at = _last_db_error_at

# ── Canonical job state machine (Phase 1 §5.2) ─────────────────────────────
# One vocabulary in memory, SQLite, API responses, logs, and docs.
# Active: queued -> downloading -> analysing -> rendering -> quality_check -> completed
# Terminal: failed | partial_failure | cancelled | orphaned
#
# Phase-2 correctness (F8): ALL database operations run under _db_lock. The
# sqlite3 connection is not thread-safe for interleaved transactions; the
# lock serializes every read/write so _reserve_job's BEGIN IMMEDIATE can
# never interleave with another thread's statements.
_db_lock = threading.Lock()
ATOMIC_JOB_STATES = frozenset(
    {"queued", "downloading", "analysing", "rendering", "quality_check"}
)
TERMINAL_JOB_STATES = frozenset(
    {"completed", "failed", "partial_failure", "cancelled", "orphaned"}
)
ALL_JOB_STATES = ATOMIC_JOB_STATES | TERMINAL_JOB_STATES

# Re-export the historical name for API compatibility; both refer to the same
# active set. Code should prefer the ATOMIC_* name.
ACTIVE_JOB_STATES = ATOMIC_JOB_STATES

# Hardening sprint P1.R1: canonical allowed-transition map. Every memory and
# SQLite state update MUST route through transition_job() which enforces it.
# Terminal states are sinks — no outgoing edges.
ALLOWED_TRANSITIONS: Dict[str, frozenset] = {
    "queued": frozenset({"downloading", "cancelled", "failed", "orphaned"}),
    "downloading": frozenset({"analysing", "failed", "orphaned"}),
    "analysing": frozenset({"rendering", "failed", "orphaned"}),
    "rendering": frozenset({"quality_check", "failed", "orphaned"}),
    "quality_check": frozenset({"completed", "partial_failure", "failed", "orphaned"}),
    "completed": frozenset(),
    "partial_failure": frozenset(),
    "failed": frozenset(),
    "cancelled": frozenset(),
    "orphaned": frozenset(),
}

# Legacy status values -> canonical (Phase 1 §5.2). Existing DB rows and old
# clients may use the old names; we normalize at read/write boundaries.
LEGACY_STATUS_MAP = {
    "running": "rendering",
    "done": "completed",
    "error": "failed",
    "analysing_source": "analysing",
    "rendering_preview": "rendering",
    "rendering_final": "rendering",
}


def canonical_status(status: str) -> str:
    """Map legacy/API status names onto the canonical vocabulary."""
    if status in ALL_JOB_STATES:
        return status
    return LEGACY_STATUS_MAP.get(status, status)


def is_terminal(status: str) -> bool:
    return canonical_status(status) in TERMINAL_JOB_STATES


def is_active(status: str) -> bool:
    return canonical_status(status) in ACTIVE_JOB_STATES


def _job_older_than(created_at: str, threshold_sec: float) -> bool:
    """True when an ISO created_at is older than threshold seconds. Empty /
    unparseable timestamps return True so old pre-timestamp rows can be
    orphaned by the age rule when no boot ownership exists."""
    try:
        import datetime
        iso = created_at.strip().rstrip("Z")
        if "+" in iso:
            dt = datetime.datetime.fromisoformat(iso)
            now = datetime.datetime.now(dt.tzinfo)
        else:
            dt = datetime.datetime.fromisoformat(iso)
            now = datetime.datetime.now(datetime.timezone.utc).replace(tzinfo=None)
            if dt.tzinfo is not None:
                dt = dt.replace(tzinfo=None)
        return (now - dt).total_seconds() > threshold_sec
    except Exception:  # noqa: BLE001
        # Brief v7 R10: an empty/unparseable created_at must count as OLD
        # (return True) so the age rule can orphan legacy rows that predate
        # timestamps — never return False, which would permanently shield
        # them from the orphan policy.
        return True


def _job_db():
    """Open a FRESH connection for the current operation (Phase-2 F8).

    The previous module-global connection (check_same_thread=False) allowed
    API and worker threads to interleave transactions on one handle, which
    surfaced as 'cannot commit - no transaction is active' under concurrency.
    Each call now opens its own short-lived connection; _db_lock serializes
    writers so WAL stays consistent. Opened handles are tracked so tests can
    close them on teardown (Windows keeps DB files locked while open).
    """
    import sqlite3
    conn = sqlite3.connect(str(JOB_DB_PATH), timeout=30)
    with _opened_db_conns_lock:
        _opened_db_conns.add(conn)
    # Phase 1 §5.3: WAL + busy timeout for threaded access. Short
    # transactions: every writer commits immediately.
    conn.execute("PRAGMA journal_mode=WAL")
    conn.execute("PRAGMA busy_timeout=30000")
    conn.execute(
        """CREATE TABLE IF NOT EXISTS render_jobs (
            job_id TEXT PRIMARY KEY,
            status TEXT NOT NULL,
            mode TEXT NOT NULL DEFAULT 'final',
            episode_id TEXT,
            request TEXT,
            response TEXT,
            error TEXT,
            created_at TEXT NOT NULL,
            updated_at TEXT NOT NULL
        )"""
    )
    # Phase 1 §5.3: additive, idempotent migration. Preserves existing
    # records; safe to rerun (each ALTER is guarded by a column check).
    _migrate_render_jobs(conn)
    conn.commit()
    return conn


_opened_db_conns: set = set()
_opened_db_conns_lock = threading.Lock()


def _close_db_conns() -> None:
    """Close every tracked connection (used by tests + shutdown)."""
    with _opened_db_conns_lock:
        conns = list(_opened_db_conns)
        _opened_db_conns.clear()
    for conn in conns:
        try:
            conn.close()
        except Exception:
            pass


def _db_conn():
    """Context manager: open a fresh connection, yield it, always close it.

    Phase-2 correctness (F8): every DB operation opens its own short-lived
    connection and closes it when done, so no handle lingers (Windows keeps
    the DB file locked while a connection is open).
    """
    from contextlib import contextmanager

    @contextmanager
    def _cm():
        conn = _job_db()
        try:
            yield conn
        finally:
            try:
                conn.close()
            except Exception:
                pass
            with _opened_db_conns_lock:
                _opened_db_conns.discard(conn)

    return _cm()


def _migrate_render_jobs(conn) -> None:
    """Add Phase 1 columns + indexes. Additive only — never drops data."""
    import sqlite3
    existing = {row[1] for row in conn.execute("PRAGMA table_info(render_jobs)").fetchall()}
    columns = {
        "request_id": "TEXT",
        "parent_job_id": "TEXT",
        "attempt": "INTEGER NOT NULL DEFAULT 1",
        "started_at": "TEXT",
        "finished_at": "TEXT",
        "last_error_stage": "TEXT",
        # Brief v4 F19: process ownership for orphan detection (idempotent).
        "process_boot_id": "TEXT",
    }
    for name, decl in columns.items():
        if name not in existing:
            conn.execute(f"ALTER TABLE render_jobs ADD COLUMN {name} {decl}")
    conn.execute("CREATE INDEX IF NOT EXISTS idx_render_jobs_request_id ON render_jobs(request_id)")
    conn.execute("CREATE INDEX IF NOT EXISTS idx_render_jobs_status ON render_jobs(status)")
    conn.execute("CREATE INDEX IF NOT EXISTS idx_render_jobs_created_at ON render_jobs(created_at)")
    # Brief v10 C06 (section 6.1): index on parent_job_id for lineage queries.
    conn.execute("CREATE INDEX IF NOT EXISTS idx_render_jobs_parent_job_id ON render_jobs(parent_job_id)")
    # Phase-2 correctness (F7): at most ONE ACTIVE job per request_id. The
    # partial index only covers non-terminal states so a failed/completed job
    # may be resubmitted. This gives idempotency real uniqueness, not just a
    # check-then-act race.
    conn.execute(
        "CREATE UNIQUE INDEX IF NOT EXISTS idx_render_jobs_active_request ON render_jobs(request_id) "
        "WHERE request_id != '' AND status IN "
        "('queued','downloading','analysing','rendering','quality_check')"
    )
    # Brief v10 C06: unique (request_id, attempt) guarantees monotonically
    # allocated attempt numbers. STOP-safe: detect existing duplicates BEFORE
    # creating the unique index; if any exist we must NOT silently create it
    # (Section 15 stop condition). The unique index is intentionally partial on
    # request_id != '' so V1 legacy jobs (empty request_id) are unaffected.
    if _has_duplicate_request_attempt(conn):
        raise PersistenceError(
            "duplicate (request_id, attempt) rows exist; refusing to create "
            "unique index (v10 stop condition). Clean up before retrying."
        )
    conn.execute(
        "CREATE UNIQUE INDEX IF NOT EXISTS uq_render_jobs_request_attempt "
        "ON render_jobs(request_id, attempt) "
        "WHERE request_id IS NOT NULL AND request_id <> ''"
    )


def _has_duplicate_request_attempt(conn) -> bool:
    """Return True when the existing render_jobs rows contain duplicate
    (request_id, attempt) pairs for non-empty request_id — a stop condition
    that must be resolved before the unique index can be created."""
    try:
        row = conn.execute(
            "SELECT COUNT(*) FROM ("
            "  SELECT request_id, attempt FROM render_jobs "
            "  WHERE request_id IS NOT NULL AND request_id <> '' "
            "  GROUP BY request_id, attempt HAVING COUNT(*) > 1"
            ") AS dup"
        ).fetchone()
        return bool(row and row[0] > 0)
    except Exception:  # noqa: BLE001 — column may be absent on legacy schema
        return False


def _persist_job(job_id: str, status: str, *, mode: str = "final",
                 episode_id: str = "", request: str = "", response: str = "",
                 error: str = "", parent_job_id: str = "", attempt: int = 1) -> bool:
    """Persist job state. Brief v5 R-03: named parameters — never positional
    order (the old tuple swapped last_error_stage/process_boot_id).

    Returns True iff the write COMMITTED. Raises PersistenceError on failure —
    callers must NOT silently update memory after a failed commit (R-05).
    """
    import sqlite3
    import datetime
    status = canonical_status(status)
    try:
        with _db_lock, _db_conn() as conn:
            now = datetime.datetime.utcnow().isoformat()
            conn.execute(
                """INSERT INTO render_jobs
                  (job_id, status, mode, episode_id, request, response, error, created_at, updated_at,
                   request_id, parent_job_id, attempt, started_at, finished_at, last_error_stage, process_boot_id)
                  VALUES (:job_id, :status, :mode, :episode_id, :request, :response, :error,
                          :now, :now, :request_id, :parent_job_id, :attempt,
                          :started_at, :finished_at, :error_stage, :process_boot_id)
                  ON CONFLICT(job_id) DO UPDATE SET
                   status=excluded.status, mode=excluded.mode,
                   response=excluded.response, error=excluded.error,
                   parent_job_id=COALESCE(parent_job_id, excluded.parent_job_id),
                   attempt=COALESCE(attempt, excluded.attempt),
                   request_id=COALESCE(request_id, excluded.request_id),
                   process_boot_id=COALESCE(process_boot_id, excluded.process_boot_id),
                   started_at=COALESCE(started_at, excluded.started_at),
                   finished_at=COALESCE(excluded.finished_at, finished_at),
                   last_error_stage=CASE WHEN excluded.error IS NOT NULL AND excluded.error != ''
                       THEN excluded.last_error_stage ELSE last_error_stage END,
                   updated_at=excluded.updated_at""",
                {
                    "job_id": job_id,
                    "status": status,
                    "mode": mode,
                    "episode_id": episode_id,
                    "request": request,
                    "response": response,
                    "error": error,
                    "now": now,
                    "request_id": _extract_request_id(request),
                    "parent_job_id": parent_job_id,
                    "attempt": attempt,
                    "started_at": now if status != "queued" else None,
                    "finished_at": now if status in TERMINAL_JOB_STATES else None,
                    "error_stage": (error.split(":")[0].split(" at ")[-1].strip() if error else None),
                    "process_boot_id": PROCESS_BOOT_ID,
                },
            )
            conn.commit()
        return True
    except Exception as e:  # noqa: BLE001
        # Brief v5 R-05: persistence failure must be observable, never silent.
        global _last_persist_error, _last_persist_error_at
        import datetime
        _last_persist_error = f"{type(e).__name__}: {e}"
        _last_persist_error_at = datetime.datetime.utcnow().isoformat()
        try:
            print(f"[persist] ERROR saving job {job_id} status={status}: {e}", flush=True)
        except Exception:  # noqa: BLE001
            pass
        raise PersistenceError(f"{type(e).__name__}: {e}") from e


def _extract_request_id(request_json: str) -> str:
    """Pull request_id from a persisted request payload without full parsing."""
    if not request_json:
        return ""
    import json
    try:
        parsed = json.loads(request_json)
        if isinstance(parsed, dict):
            return str(parsed.get("request_id") or "")
    except Exception:
        pass
    return ""


def _transition_allowed(expected: str, target: str) -> bool:
    expected = canonical_status(expected)
    target = canonical_status(target)
    return target in ALLOWED_TRANSITIONS.get(expected, frozenset())


def transition_job(job_id: str, expected: str, target: str, *, mode: str = "final",
                   episode_id: str = "", error: str = "", error_stage: str = "",
                   response: str = "") -> bool:
    """Compare-and-swap job state transition (hardening sprint P0.1/P1.R1).

    Atomically verifies that BOTH the in-memory registry and the persisted
    SQLite row are in `expected` and that `expected -> target` is allowed by
    the canonical map; only then updates both stores. Returns True iff the
    swap won. A losing caller observes the state change through the next
    read — no code path checks state and changes it outside this operation.

    Terminal states are sinks (no outgoing edges) and therefore immutable.
    `error` is persisted alongside the target state (used by cancellation).
    `error_stage` records WHERE the failure happened (brief v5 4.3) and
    `response` carries the terminal RenderResponse (brief v5 4.5).
    """
    import sqlite3
    import datetime
    expected = canonical_status(expected)
    target = canonical_status(target)
    if not _transition_allowed(expected, target):
        return False

    with _async_jobs_lock:
        job = _async_jobs.get(job_id)
        mem_state = canonical_status(job.get("state", "")) if job else None
        # Verify persisted state under the same orchestration lock so no
        # interleaved transition can slip between the two reads.
        try:
            with _db_lock, _db_conn() as conn:
                row = conn.execute(
                    "SELECT status FROM render_jobs WHERE job_id = ?", (job_id,)
                ).fetchone()
                db_state = canonical_status(row[0]) if row else None
                if db_state != expected:
                    return False
                if mem_state != expected:
                    return False
                now = datetime.datetime.utcnow().isoformat()
                # Brief v5 4.3: named parameters — never positional order.
                if error or error_stage or response:
                    # Brief v9 C04: set finished_at on terminal with error
                    finished_at_val = now if target in ("completed", "failed", "cancelled", "partial_failure", "orphaned") else None
                    conn.execute(
                        """
                        UPDATE render_jobs
                           SET status = :status,
                               last_error_stage = COALESCE(:error_stage, last_error_stage),
                               response = :response,
                               error = :error,
                               updated_at = :updated_at,
                               finished_at = COALESCE(:finished_at, finished_at)
                         WHERE job_id = :job_id
                        """,
                        {
                            "job_id": job_id,
                            "status": target,
                            "error_stage": error_stage or None,
                            "response": response,
                            "error": error,
                            "updated_at": now,
                            "finished_at": finished_at_val,
                        },
                    )
                else:
                    # Brief v9 C04: set started_at on first active state, finished_at on terminal.
                    started_at_val = now if target == "downloading" else None
                    finished_at_val = now if target in ("completed", "failed", "cancelled", "partial_failure", "orphaned") else None
                    conn.execute(
                        """
                        UPDATE render_jobs
                           SET status = :status,
                               updated_at = :updated_at,
                               started_at = COALESCE(:started_at, started_at),
                               finished_at = COALESCE(:finished_at, finished_at)
                         WHERE job_id = :job_id
                        """,
                        {
                            "job_id": job_id,
                            "status": target,
                            "updated_at": now,
                            "started_at": started_at_val,
                            "finished_at": finished_at_val,
                        },
                    )
                conn.commit()
        except Exception as e:  # noqa: BLE001
            _record_db_error("transition_job", e)
            return False
        if job is not None:
            job["state"] = target
        return True


def require_transition(job_id: str, expected: str, target: str, *, mode: str = "final",
                       episode_id: str = "", error: str = "", error_stage: str = "",
                       response: str = "") -> None:
    """Brief v6 4.1 — CHECKED active-stage transition.

    A failed compare-and-swap is a correctness conflict, not a warning.
    Raises JobTransitionConflict when the expected -> target transition did
    not win; callers MUST NOT continue rendering after this raises.
    """
    ok = transition_job(job_id, expected, target, mode=mode, episode_id=episode_id,
                        error=error, error_stage=error_stage, response=response)
    if not ok:
        current = _load_job(job_id)
        raise JobTransitionConflict(
            f"transition {expected} -> {target} for job {job_id} lost; "
            f"current effective state: {current and current.get('status')}"
        )


def _load_job(job_id: str) -> Optional[Dict]:
    """Load durable job state, distinguishing NOT_FOUND from persistence error.

    Legacy fallback is allowed only when the modern SELECT proves that a
    migration column is absent. Generic SQLite/I/O/corruption failures fail
    closed as PersistenceError and are never returned as ``None``.
    """
    import sqlite3
    import json

    try:
        with _db_lock, _db_conn() as conn:
            row = conn.execute(
                "SELECT status, mode, episode_id, response, error, attempt, parent_job_id, "
                "COALESCE(request_id, ''), COALESCE(process_boot_id, ''), COALESCE(created_at, ''), "
                "started_at, finished_at "
                "FROM render_jobs WHERE job_id = ?",
                (job_id,),
            ).fetchone()
            if not row:
                return None
            return {
                "status": canonical_status(row[0]),
                "mode": row[1],
                "episode_id": row[2],
                "response": json.loads(row[3]) if row[3] else None,
                "error": row[4],
                "attempt": row[5] if row[5] is not None else 1,
                "parent_job_id": row[6],
                "request_id": row[7] or "",
                "process_boot_id": row[8] or None,
                "created_at": row[9] or "",
                "started_at": row[10],
                "finished_at": row[11],
            }
    except sqlite3.OperationalError as exc:
        message = str(exc).lower()
        missing_column = "no such column" in message and any(
            token in message
            for token in ("request_id", "process_boot_id", "created_at", "started_at", "finished_at", "parent_job_id", "attempt")
        )
        if not missing_column:
            _record_db_error("load_job", exc)
            raise PersistenceError(f"load_job failed: {exc}") from exc
        # Explicitly proven legacy schema: use only the old column set.
        try:
            with _db_conn() as conn:
                row = conn.execute(
                    "SELECT status, mode, episode_id, response, error FROM render_jobs WHERE job_id = ?",
                    (job_id,),
                ).fetchone()
                if not row:
                    return None
                return {
                    "status": canonical_status(row[0]),
                    "mode": row[1],
                    "episode_id": row[2],
                    "response": json.loads(row[3]) if row[3] else None,
                    "error": row[4],
                    "attempt": 1,
                    "parent_job_id": None,
                    "request_id": "",
                }
        except Exception as legacy_exc:
            _record_db_error("load_job", legacy_exc)
            raise PersistenceError(f"legacy load_job failed: {legacy_exc}") from legacy_exc
    except sqlite3.Error as exc:
        _record_db_error("load_job", exc)
        raise PersistenceError(f"load_job failed: {exc}") from exc
    except (ValueError, TypeError, json.JSONDecodeError) as exc:
        _record_db_error("load_job", exc)
        raise PersistenceError(f"load_job returned corrupt data: {exc}") from exc
    except Exception as exc:  # noqa: BLE001
        _record_db_error("load_job", exc)
        raise PersistenceError(f"load_job failed: {exc}") from exc


def _load_durable_snapshot(job_id: str) -> Optional[DurableJobSnapshot]:
    """Brief v9 A1 — load canonical durable state from SQLite.

    Returns a typed snapshot or None if the job does not exist.
    """
    row = _load_job(job_id)
    if not row:
        return None
    return DurableJobSnapshot(
        job_id=job_id,
        state=row.get("status", "queued"),
        request_id=row.get("request_id", ""),
        mode=row.get("mode", "final"),
        episode_id=row.get("episode_id", ""),
        attempt=row.get("attempt", 1),
        parent_job_id=row.get("parent_job_id"),
        created_at=row.get("created_at", ""),
        started_at=row.get("started_at"),
        finished_at=row.get("finished_at"),
        response=row.get("response"),
        error=row.get("error"),
        process_boot_id=row.get("process_boot_id"),
    )


def mirror_durable_after_failure(job_id: str, diagnostic: str) -> None:
    """Brief v9 A2 — shared helper for persistence-failure mirroring.

    Called by sync terminal persist failure, worker fatal exception, and
    retry reservation failure. Mirrors durable state into memory with
    runtime_error and persistence_degraded=True, preserving metadata.
    """
    durable = _load_durable_snapshot(job_id)
    with _async_jobs_lock:
        existing = _async_jobs.get(job_id, {})
        if not durable:
            # No durable row at all — fabricate unknown state.
            _async_jobs[job_id] = {
                "state": "unknown",
                "runtime_error": diagnostic,
                "persistence_degraded": True,
                **{k: existing.get(k) for k in ("request_id", "mode", "episode_id", "attempt", "parent_job_id") if existing.get(k) is not None},
            }
            return
        # Merge: state from durable (canonical), metadata from memory if present.
        _async_jobs[job_id] = {
            "state": durable.state,
            "request_id": existing.get("request_id") or durable.request_id,
            "mode": existing.get("mode") or durable.mode,
            "episode_id": existing.get("episode_id") or durable.episode_id,
            "attempt": existing.get("attempt") if existing.get("attempt") is not None else durable.attempt,
            "parent_job_id": existing.get("parent_job_id") or durable.parent_job_id,
            "response": existing.get("response") or durable.response,
            "error": durable.error,
            "runtime_error": diagnostic,
            "persistence_degraded": True,
        }


def _find_job_by_request(request_id: str) -> Optional[str]:
    """Idempotency (brief §20): find a persisted non-failed job that carried
    the same request_id. Returns its job_id or None.

    Phase 1 §5.3: queries the indexed request_id column directly — no JSON
    scan, no dependence on a nonexistent id column.

    Brief v10 C07 (V10-R05): the legacy JSON-scan fallback ONLY runs when the
    caught sqlite3.OperationalError specifically indicates the request_id
    column is missing on an unmigrated legacy schema. ALL other SQLite errors
    become PersistenceError (fail closed) — a DB read error must never be
    interpreted as "no existing job".
    """
    import sqlite3
    if not request_id:
        return None
    try:
        with _db_lock, _db_conn() as conn:
            row = conn.execute(
                "SELECT job_id FROM render_jobs "
                "WHERE request_id = ? AND status IN "
                "('queued','downloading','analysing','rendering','quality_check','completed') "
                "ORDER BY created_at DESC LIMIT 1",
                (request_id,),
            ).fetchone()
            return row[0] if row else None
    except sqlite3.OperationalError as e:
        # Narrow fallback: ONLY when the request_id column is missing.
        if "request_id" in str(e).lower() and ("no such column" in str(e).lower() or "no column" in str(e).lower()):
            try:
                with _db_conn() as conn:
                    rows = conn.execute(
                        "SELECT job_id, status, request FROM render_jobs "
                        "WHERE status IN ('queued','downloading','analysing','rendering',"
                        "'quality_check','completed') ORDER BY created_at DESC LIMIT 50"
                    ).fetchall()
                    import json
                    for job_id, status, request in rows:
                        if not request:
                            continue
                        try:
                            parsed = json.loads(request)
                        except Exception:  # noqa: BLE001
                            continue
                        rid = parsed.get("request_id") if isinstance(parsed, dict) else None
                        if rid == request_id:
                            return job_id
                return None
            except Exception as e2:  # noqa: BLE001
                _record_db_error("find_job_by_request_legacy", e2)
                raise PersistenceError(f"legacy request lookup failed: {e2}") from e2
        # Not a missing-column issue: fail closed.
        _record_db_error("find_job_by_request", e)
        raise PersistenceError(f"find_job_by_request failed: {e}") from e
    except Exception as e:  # noqa: BLE001
        _record_db_error("find_job_by_request", e)
        raise PersistenceError(f"find_job_by_request failed: {e}") from e


# Idempotent request statuses: an existing job in one of these states is
# returned as-is. Terminal 'failed'/'partial_failure'/'cancelled' are NOT
# included — the caller may resubmit those request_ids (Phase-2 F6).
IDEMPOTENT_HIT_STATES = (
    "queued", "downloading", "analysing", "rendering", "quality_check", "completed",
)


class AttemptReservation:
    """Durable result of the single v11 attempt allocator.

    ``created`` is true only after the exact row is committed and re-readable
    from SQLite. When false, ``job_id`` identifies the durable winner/current
    active attempt; it is never a speculative UUID.
    """

    __slots__ = ("job_id", "attempt", "parent_job_id", "created", "existing_winner_job_id", "reason")

    def __init__(self, job_id=None, attempt=1, parent_job_id=None,
                 created=False, existing_winner_job_id=None, reason="retry"):
        self.job_id = job_id
        self.attempt = attempt
        self.parent_job_id = parent_job_id
        self.created = created
        self.existing_winner_job_id = existing_winner_job_id
        self.reason = reason

    def __repr__(self):
        return (
            f"AttemptReservation(job_id={self.job_id!r}, attempt={self.attempt}, "
            f"parent_job_id={self.parent_job_id!r}, created={self.created}, "
            f"reason={self.reason!r}, winner={self.existing_winner_job_id!r})"
        )


def reserve_attempt(*, source_job_id, request_id, request_json, mode,
                    episode_id, reason, preferred_job_id=None) -> AttemptReservation:
    """The ONLY allocator for retry, force, and terminal resubmission.

    Every decision happens in one ``BEGIN IMMEDIATE`` transaction:
    latest attempt, active-attempt policy, parent lineage, next attempt, and
    durable INSERT. A second caller resolves to the actual durable active
    winner. Any database error or an IntegrityError without a re-readable
    winner fails closed; no phantom job ID is returned.
    """
    import datetime
    import sqlite3

    if reason not in ("retry", "force", "resubmit"):
        raise ValueError(f"invalid attempt reason: {reason!r}")
    if not request_id:
        raise HTTPException(
            status_code=409,
            detail=f"{reason} requires a non-empty request_id lineage",
        )

    new_job_id = preferred_job_id or uuid.uuid4().hex[:10]
    try:
        with _db_lock, _db_conn() as conn:
            conn.execute("BEGIN IMMEDIATE")
            rows = conn.execute(
                "SELECT job_id, status, attempt, parent_job_id FROM render_jobs "
                                "WHERE request_id = ? ORDER BY attempt DESC, created_at DESC, rowid DESC",
                (request_id,),
            ).fetchall()
            latest = rows[0] if rows else None
            active = next(
                (row for row in rows if canonical_status(row[1]) in ATOMIC_JOB_STATES),
                None,
            )

            # Retry validates its source before resolving a concurrent child.
            # Thus retrying an active child is rejected, while two retries of
            # the same terminal parent resolve to the one active winner.
            if reason == "retry":
                if not source_job_id:
                    raise HTTPException(status_code=400, detail="retry source job is required")
                source = conn.execute(
                    "SELECT status, attempt FROM render_jobs WHERE job_id = ?",
                    (source_job_id,),
                ).fetchone()
                if not source:
                    raise HTTPException(status_code=404, detail="source job not found for retry")
                source_state = canonical_status(source[0])
                if source_state not in ("failed", "partial_failure"):
                    raise HTTPException(
                        status_code=409,
                        detail=f"job is {source_state}; retry only allowed from failed or partial_failure",
                    )

            # One active attempt per request_id: all allocator reasons resolve
            # to the durable active winner rather than creating attempt N+1.
            if active:
                conn.rollback()
                return AttemptReservation(
                    job_id=active[0], attempt=int(active[2] or 1),
                    parent_job_id=active[3] if len(active) > 3 else source_job_id or None,
                    created=False,
                    existing_winner_job_id=active[0], reason=reason,
                )

            if reason == "resubmit":
                if latest and canonical_status(latest[1]) not in (
                    "failed", "partial_failure", "cancelled", "orphaned",
                ):
                    raise HTTPException(
                        status_code=409,
                        detail=f"request_id has non-resubmittable state {latest[1]}",
                    )

            max_attempt = max((int(row[2] or 1) for row in rows), default=0)
            alloc_attempt = max_attempt + 1
            parent = source_job_id or (latest[0] if latest else None)
            now = datetime.datetime.utcnow().isoformat()
            try:
                conn.execute(
                    "INSERT INTO render_jobs "
                    "(job_id, status, mode, episode_id, request, response, error, created_at, updated_at,"
                    " request_id, parent_job_id, attempt, started_at, finished_at, last_error_stage, process_boot_id)"
                    " VALUES (?, 'queued', ?, ?, ?, '', '', ?, ?, ?, ?, ?, NULL, NULL, NULL, ?)",
                    (new_job_id, mode, episode_id, request_json, now, now,
                     request_id, parent, alloc_attempt, PROCESS_BOOT_ID),
                )
                conn.commit()
            except sqlite3.IntegrityError as exc:
                conn.rollback()
                winner = conn.execute(
                    "SELECT job_id, status, attempt, parent_job_id FROM render_jobs "
                    "WHERE request_id = ? AND attempt = ? ORDER BY created_at ASC, rowid ASC LIMIT 1",
                    (request_id, alloc_attempt),
                ).fetchone()
                if not winner:
                    raise PersistenceError(
                        f"allocator IntegrityError without durable winner for "
                        f"request_id={request_id!r}, attempt={alloc_attempt}"
                    ) from exc
                if canonical_status(winner[1]) not in ALL_JOB_STATES:
                    raise PersistenceError(
                        f"allocator winner has invalid state {winner[1]!r}"
                    ) from exc
                return AttemptReservation(
                    job_id=winner[0], attempt=int(winner[2] or alloc_attempt),
                    parent_job_id=winner[3], created=False,
                    existing_winner_job_id=winner[0], reason=reason,
                )

            # Commit is necessary but not sufficient for the contract: re-read
            # the exact row before reporting created=True.
            persisted = conn.execute(
                "SELECT job_id, attempt, parent_job_id, status FROM render_jobs WHERE job_id = ?",
                (new_job_id,),
            ).fetchone()
            if not persisted or persisted[0] != new_job_id or persisted[3] != "queued":
                raise PersistenceError(
                    f"allocator committed but exact row is not readable: {new_job_id}"
                )
            return AttemptReservation(
                job_id=new_job_id, attempt=int(persisted[1] or alloc_attempt),
                parent_job_id=persisted[2], created=True, reason=reason,
            )
    except Exception as exc:  # noqa: BLE001
        _record_db_error("reserve_attempt", exc)
        if isinstance(exc, (HTTPException, PersistenceError)):
            raise
        raise PersistenceError(f"reserve_attempt failed: {exc}") from exc


def _reserve_job(request_id: str, new_job_id: str, *, mode: str,
                 episode_id: str, request_json: str, force: bool = False) -> str:
    """Deprecated compatibility wrapper around :func:`reserve_attempt`.

    Brief v11 C2: this function is no longer an allocator. Production routes
    call ``reserve_attempt`` directly; legacy tests/callers are kept source
    compatible but delegate to the same transaction and return only its
    durable job_id. force=True remains supported for source compatibility —
    it delegates to reserve_attempt(reason='force').
    """
    if not request_id:
        # Legacy V1 has no lineage. Keep its one-off durable insertion behavior,
        # but never use it for attempt > 1.
        import datetime
        try:
            with _db_lock, _db_conn() as conn:
                conn.execute("BEGIN IMMEDIATE")
                now = datetime.datetime.utcnow().isoformat()
                conn.execute(
                    "INSERT INTO render_jobs "
                    "(job_id, status, mode, episode_id, request, response, error, created_at, updated_at,"
                    " request_id, parent_job_id, attempt, started_at, finished_at, last_error_stage, process_boot_id) "
                    "VALUES (?, 'queued', ?, ?, ?, '', '', ?, ?, '', NULL, 1, NULL, NULL, NULL, ?)",
                    (new_job_id, mode, episode_id, request_json, now, now, PROCESS_BOOT_ID),
                )
                conn.commit()
                return new_job_id
        except Exception as exc:  # noqa: BLE001
            _record_db_error("reserve_job_legacy", exc)
            raise PersistenceError(f"legacy reservation failed: {exc}") from exc

    # Preserve normal idempotent-hit behavior for an existing active/completed
    # row. Any new attempt is still allocated only by reserve_attempt().
    if not force:
        existing = _find_job_by_request(request_id)
        if existing:
            return existing

    source_job_id = None
    if force:
        try:
            with _db_lock, _db_conn() as conn:
                row = conn.execute(
                    "SELECT job_id FROM render_jobs WHERE request_id = ? "
                    "ORDER BY attempt DESC, created_at DESC, rowid DESC LIMIT 1",
                    (request_id,),
                ).fetchone()
                source_job_id = row[0] if row else None
        except Exception as exc:  # noqa: BLE001
            _record_db_error("reserve_job_compat_lookup", exc)
            raise PersistenceError(f"failed to load force parent: {exc}") from exc

    reservation = reserve_attempt(
        source_job_id=source_job_id,
        request_id=request_id,
        request_json=request_json,
        mode=mode,
        episode_id=episode_id,
        reason="force" if force else "resubmit",
        preferred_job_id=new_job_id,
    )
    return reservation.job_id

def _load_job_request(job_id: str) -> Optional[Dict]:
    """Return the original request JSON for a job (for retry)."""
    try:
        with _db_lock, _db_conn() as conn:
            row = conn.execute(
                "SELECT request FROM render_jobs WHERE job_id = ?",
                (job_id,),
            ).fetchone()
            if not row or not row[0]:
                return None
            import json
            return json.loads(row[0])
    except Exception as e:  # noqa: BLE001
        _record_db_error("load_job_request", e)
        raise PersistenceError(f"load_job_request failed: {e}") from e


def _normalize_clips(request) -> List:
    """Return a normalized list of clip dicts from either a v1 RenderRequest
    or a v2 RenderRequestV2 (brief §16 backward compatibility)."""
    clips = []
    for c in request.clips:
        if hasattr(c, "caption_plan") and c.caption_plan is not None:
            # v2: use caption_plan.cues as captions; layout from layout_plan.
            # Hardening v3 D1/D3 (#21/#24): preserve canonical words, language,
            # provenance and clamp each word/cue to the clip — never drop them.
            cues = [
                CaptionRequest(
                    start_sec=cc.start_sec,
                    end_sec=cc.end_sec,
                    text=cc.text,
                    speaker=cc.speaker_id or "",
                    words=[
                        CaptionWord(
                            start_sec=max(float(w.start_sec), float(c.start_sec)),
                            end_sec=min(float(w.end_sec), float(c.end_sec)),
                            text=w.text,
                        )
                        for w in (cc.words or [])
                        if w.end_sec > c.start_sec and w.start_sec < c.end_sec
                    ],
                )
                for cc in c.caption_plan.cues
            ]
            clips.append({
                "clip_id": c.clip_id,
                "title": c.title,
                "start_sec": float(c.start_sec),
                "end_sec": float(c.end_sec),
                "captions": cues,
                "hook": c.hook or "",
                "preferred_layout": c.layout_plan.preferred_layout if c.layout_plan else "auto",
                "expected_speakers": c.layout_plan.expected_speakers if c.layout_plan else None,
                "allow_split": c.layout_plan.allow_split if c.layout_plan else True,
                "allow_blur_background": c.layout_plan.allow_blur_background if c.layout_plan else True,
                "editing_events": [e.model_dump() for e in (c.editing_events or [])],
                "highlight_terms": list(c.caption_plan.highlight_terms) if c.caption_plan else [],
                # Hardening v3 D1 (#21): propagate language + provenance so the
                # STT fallback never hard-codes English.
                "language": c.caption_plan.language or "",
                "provider": c.caption_plan.provider or "unknown",
                "alignment_confidence": float(getattr(c.caption_plan, "alignment_confidence", 0) or 0),
                "transcript_version": c.caption_plan.transcript_version or "",
                "has_word_timing": any(len(cc.words or []) > 0 for cc in c.caption_plan.cues),
            })
        else:
            # v1 legacy
            clips.append({
                "clip_id": c.clip_id,
                "title": c.title,
                "start_sec": float(c.start_sec),
                "end_sec": float(c.end_sec),
                "captions": list(c.captions),
                "hook": c.hook or "",
                "preferred_layout": "auto",
                "expected_speakers": None,
                "allow_split": True,
                "allow_blur_background": True,
                "editing_events": [],
                "highlight_terms": [],
            })
    return clips


@app.get("/health")
def health():
    return {"status": "ok", "service": "shorts-render", "version": "0.1.0"}


@app.get("/readyz")
def readyz():
    """Brief v8 C08/R05/R06 — real readiness.

    - The worker starts during app lifespan, so an IDLE healthy service is
      ready (no lazy first-job dependency).
    - DB readiness is a WRITE probe (temp-table insert/delete in a rolled-back
      transaction), not a SELECT-only check.
    - Reports worker_alive, heartbeat age, queue depth, oldest queued age,
      SQLite journal mode, output writability, free disk, ffmpeg/ffprobe.
    """
    import shutil
    ready = True
    reasons = []
    # Worker must be alive (started during lifespan).
    if _render_queue_worker_thread is None or not _render_queue_worker_thread.is_alive():
        ready = False
        reasons.append("queue_worker_dead")
    # DB must be WRITABLE, not just readable.
    db_ok = False
    db_error = None
    journal_mode = None
    try:
        with _db_lock, _db_conn() as conn:
            conn.execute("CREATE TEMP TABLE IF NOT EXISTS _readyz_probe (v INTEGER)")
            conn.execute("INSERT INTO _readyz_probe (v) VALUES (1)")
            conn.execute("SELECT COUNT(*) FROM _readyz_probe")
            conn.execute("DELETE FROM _readyz_probe")
            conn.commit()
            row = conn.execute("PRAGMA journal_mode").fetchone()
            journal_mode = row[0] if row else None
            db_ok = True
    except Exception as exc:  # noqa: BLE001
        db_error = f"{type(exc).__name__}: {exc}"
        ready = False
        reasons.append(f"db_unwritable:{type(exc).__name__}")
    # Output dir writability + free disk.
    out_ok = False
    out_error = None
    free_bytes = None
    try:
        RENDER_ROOT.mkdir(parents=True, exist_ok=True)
        probe = RENDER_ROOT / f".readyz_{os.getpid()}"
        probe.write_text("ok", encoding="utf-8")
        probe.unlink()
        out_ok = True
        import ctypes
        fb = ctypes.c_ulonglong(0)
        ok = ctypes.windll.kernel32.GetDiskFreeSpaceExW(
            str(RENDER_ROOT.resolve()), None, None, ctypes.byref(fb),
        )
        free_bytes = int(fb.value) if ok else None
    except Exception as exc:  # noqa: BLE001
        out_error = f"{type(exc).__name__}: {exc}"
        ready = False
        reasons.append("output_unwritable")
    try:
        oldest = _oldest_queued_age_sec()
    except Exception:  # noqa: BLE001
        oldest = None
    body = {
        "status": "ready" if ready else "unready",
        "ready": ready,
        "worker_alive": bool(_render_queue_worker_thread and _render_queue_worker_thread.is_alive()),
        "worker_heartbeat_age_sec": _heartbeat_age_sec(),
        "queue_depth": int(_render_queue.qsize()),
        "oldest_queued_age_sec": oldest,
        "sqlite": {"journal_mode": journal_mode or "unknown", "ok": db_ok, "error": db_error},
        "output": {"writable": out_ok, "error": out_error, "free_bytes": free_bytes},
        "ffmpeg": {
            "ffmpeg": shutil.which("ffmpeg") is not None,
            "ffprobe": shutil.which("ffprobe") is not None,
        },
        "checks": None if ready else reasons,
    }
    return JSONResponse(status_code=200 if ready else 503, content=body)


@app.get("/livez")
def livez():
    """Cheap process liveness — always 200 while the API responds."""
    return {"status": "alive"}


@app.get("/files/{job_id}/{filename}")
def serve_file(job_id: str, filename: str):
    """Serve a rendered short so the miner UI can link / play it directly.

    Path traversal guard: both segments must be plain names, and the resolved
    path must stay inside the render root. Media type is derived from the
    extension (mp4 -> video/mp4, jpg/png -> image/*) so browsers render
    thumbnails correctly instead of treating them as video.
    """
    if not job_id or not filename or "/" in filename or "\\" in filename:
        raise HTTPException(status_code=400, detail="invalid file path")
    path = (RENDER_ROOT / job_id / filename).resolve()
    root = RENDER_ROOT.resolve()
    if not path.is_relative_to(root) or not path.is_file():
        raise HTTPException(status_code=404, detail="file not found")
    ext = path.suffix.lower()
    media_type = {
        ".mp4": "video/mp4",
        ".jpg": "image/jpeg",
        ".jpeg": "image/jpeg",
        ".png": "image/png",
    }.get(ext, "application/octet-stream")
    return FileResponse(path, media_type=media_type)


def _srt_timecode(seconds: float) -> str:
    """Format seconds as SRT timecode HH:MM:SS,mmm."""
    ms = int(round(seconds * 1000))
    h, rem = divmod(ms, 3_600_000)
    m, rem = divmod(rem, 60_000)
    s, msec = divmod(rem, 1000)
    return f"{h:02d}:{m:02d}:{s:02d},{msec:03d}"


def _ass_timecode(seconds: float) -> str:
    """Format seconds as ASS timecode H:MM:SS.cc (centiseconds)."""
    cs = int(round(seconds * 100))
    h, rem = divmod(cs, 360_000)
    m, rem = divmod(rem, 6_000)
    s, centi = divmod(rem, 100)
    return f"{h}:{m:02d}:{s:02d}.{centi:02d}"


def _write_srt(captions: List[CaptionRequest], clip_start: float, srt_path: str) -> int:
    """Write captions to an SRT file with timestamps relative to the clip.

    Returns the number of caption lines that fall inside the clip window.
    """
    lines: List[str] = []
    index = 1
    for cap in captions:
        local_start = cap.start_sec - clip_start
        local_end = cap.end_sec - clip_start
        # Skip captions entirely before the clip or after its end.
        if local_end <= 0:
            continue
        if local_start < 0:
            local_start = 0.0
        text = " ".join(cap.text.split()).strip()
        if not text:
            continue
        lines.append(f"{index}\n{_srt_timecode(local_start)} --> {_srt_timecode(local_end)}\n{text}\n")
        index += 1

    with open(srt_path, "w", encoding="utf-8") as f:
        f.write("\n".join(lines))
    return index - 1


# ---------------------------------------------------------------------------
# Karaoke word-highlight captions (viral-shorts style)
#
# Renders each caption line as a full sentence where the currently-spoken
# word is bright white and every other word is dim grey, with a black outline
# for readability. Word timing is estimated by distributing the caption
# duration evenly across its words (the transcript cues are segment-level,
# not word-level).
# ---------------------------------------------------------------------------

ACTIVE_COLOR = (255, 255, 255)      # all visible words
OUTLINE_COLOR = (10, 10, 10)
# True karaoke highlight (brief §1.1): the word currently being spoken pops in
# an accent color while already-spoken / not-yet-spoken visible words stay in
# the base color. Set RENDER_CAPTION_HIGHLIGHT=0 to fall back to plain reveal.
CAPTION_HIGHLIGHT = os.getenv("RENDER_CAPTION_HIGHLIGHT", "1") != "0"
HIGHLIGHT_COLOR = tuple(
    int(x) for x in os.getenv("RENDER_CAPTION_HIGHLIGHT_COLOR", "255,214,10").split(",")
)[:3]

# Reveal style: words appear one-by-one as the audio reaches them, all white.
# LEAD_MS shows each word slightly BEFORE its audio so it is readable in time;
# HOLD_MS keeps the final word on screen briefly after the cue ends.
CAPTION_LEAD_MS = int(os.getenv("RENDER_CAPTION_LEAD_MS", "800"))
CAPTION_HOLD_MS = int(os.getenv("RENDER_CAPTION_HOLD_MS", "400"))
# Vertical position of the caption block as a fraction of frame height from
# the BOTTOM. Lower value = higher on screen. Many source videos carry their
# own burned-in lower-third text/watermarks, so the default sits in the lower-
# middle area rather than hugging the bottom edge.
CAPTION_BOTTOM_MARGIN = float(os.getenv("RENDER_CAPTION_BOTTOM_MARGIN", "0.22"))

# Phase 5: hook intro scene. When a clip carries a hook line we prepend a
# short intro: first frame of the clip, darkened, with the hook rendered large
# and read aloud by an Edge-TTS voice. Duration = the voiceover length (we let
# TTS set it, but clamp to sane bounds).
HOOK_ENABLED = os.getenv("RENDER_HOOK_ENABLED", "1") != "0"
HOOK_TTS_VOICE = os.getenv("RENDER_HOOK_TTS_VOICE", "en-US-AvaNeural")
HOOK_TTS_RATE = os.getenv("RENDER_HOOK_TTS_RATE", "-5%")
HOOK_MAX_SEC = float(os.getenv("RENDER_HOOK_MAX_SEC", "6.0"))
HOOK_MIN_SEC = float(os.getenv("RENDER_HOOK_MIN_SEC", "1.5"))
HOOK_DIM_ALPHA = float(os.getenv("RENDER_HOOK_DIM_ALPHA", "0.55"))  # black overlay
HOOK_FONT_SCALE = float(os.getenv("RENDER_HOOK_FONT_SCALE", "0.055"))  # of frame height


def _word_events(captions: List[CaptionRequest], clip_start: float) -> List[Dict]:
    """Expand segment cues into per-word events with local timestamps."""
    events: List[Dict] = []
    for cap in captions:
        local_start = cap.start_sec - clip_start
        local_end = cap.end_sec - clip_start
        if local_end <= 0:
            continue
        if local_start < 0:
            local_start = 0.0

        text = " ".join(cap.text.split()).strip()
        words = text.split()
        if not words:
            continue

        span = max(local_end - local_start, 0.05)
        per_word = span / len(words)
        for i, word in enumerate(words):
            events.append({
                "word": word,
                "start": local_start + i * per_word,
                "end": local_start + (i + 1) * per_word,
            })
    return events


def _make_word_sprite(word: str, font, color: tuple, outline: int | None = None) -> "Image.Image":
    """Render a single word with outline onto a transparent RGBA sprite.

    Every sprite has the SAME height (font ascent + descent + outline) and the
    word is drawn on a fixed BASELINE (anchor="ls"). Pasting sprites at the
    same y therefore aligns all words on one baseline — words with descenders
    (g, y, p) no longer sit higher than words without them.
    """
    from PIL import Image, ImageDraw

    if outline is None:
        outline = max(2, int(font.size * 0.08))

    ascent, descent = font.getmetrics()
    w = int(font.getlength(word)) + outline * 2 + 2
    h = ascent + descent + outline * 2 + 2
    baseline_y = outline + ascent + 1

    img = Image.new("RGBA", (max(w, 2), max(h, 2)), (0, 0, 0, 0))
    draw = ImageDraw.Draw(img)
    draw.text(
        (outline + 1, baseline_y),
        word,
        font=font,
        fill=color,
        stroke_width=outline,
        stroke_fill=OUTLINE_COLOR,
        anchor="ls",  # left + baseline: fixed baseline across all words
    )
    return img


def _normalize_cues(captions: List[CaptionRequest], clip_start: float) -> List[Dict]:
    """Merge overlapping ASR cues into non-overlapping continuous lines.

    YouTube's ASR transcript is emitted as sliding windows: consecutive cues
    overlap by ~2s (cue A 0-4.5s, cue B 2.4-6.4s). Rendering them verbatim
    shows two caption lines at once. This merges overlapping cues so exactly
    one line is on screen at any moment.

    Each word keeps its OWN timing. When the caption carries whisper word
    timestamps (`words`), those precise timings are used — the karaoke reveal
    then matches the actual speech rhythm. Otherwise the cue's span is
    distributed evenly across its words as a fallback.

    Returns lines with: words (each with start/end in local coords), start, end.
    """
    # Collect cues that fall inside the clip, in absolute coords first.
    cues: List[Dict] = []
    for cap in captions:
        s = cap.start_sec
        e = cap.end_sec
        if e <= clip_start:
            continue
        text = " ".join(cap.text.split()).strip()
        words = text.split()
        if not words:
            continue

        # Prefer whisper word timestamps when present (absolute video coords).
        word_times: Optional[List[Tuple[float, float]]] = None
        if cap.words and len(cap.words) == len(words):
            word_times = [
                (max(w.start_sec, s), min(w.end_sec, e))
                for w in cap.words
            ]

        # If the cue started BEFORE the clip window, drop the words that were
        # spoken before the clip (also drop their timestamps).
        if s < clip_start:
            inside_frac = (e - clip_start) / max(e - s, 0.05)
            keep = max(1, int(round(len(words) * inside_frac)))
            words = words[-keep:] if keep < len(words) else words
            if word_times is not None:
                word_times = word_times[-keep:] if keep < len(word_times) else word_times
            s = clip_start
        cues.append({"s": s, "e": e, "words": words, "word_times": word_times, "speaker": cap.speaker})

    if not cues:
        return []

    cues.sort(key=lambda c: c["s"])

    merged: List[Dict] = []
    for cue in cues:
        if merged and cue["s"] < merged[-1]["e"]:
            # Overlap — extend the previous line instead of starting a new one.
            # Keep the dominant speaker = the cue with the most words so the
            # color doesn't flicker on short interjections.
            merged[-1]["e"] = max(merged[-1]["e"], cue["e"])
            merged[-1]["cues"].append(cue)
            if cue["speaker"]:
                prev_spk = merged[-1].get("speaker", "")
                if prev_spk != cue["speaker"]:
                    prev_words = sum(len(c["words"]) for c in merged[-1]["cues"] if c.get("speaker") == prev_spk)
                    new_words = sum(len(c["words"]) for c in merged[-1]["cues"] if c.get("speaker") == cue["speaker"])
                    if new_words > prev_words:
                        merged[-1]["speaker"] = cue["speaker"]
        else:
            merged.append({"s": cue["s"], "e": cue["e"], "cues": [cue], "speaker": cue.get("speaker", "")})

    lines: List[Dict] = []
    for m in merged:
        # Clip the merged span to the clip window and shift to local coords.
        local_start = max(m["s"] - clip_start, 0.0)
        local_end = m["e"] - clip_start
        if local_end <= 0:
            continue

        # Flatten all words in speech order (the merged line's concatenation).
        # When whisper word timestamps are available they are used verbatim
        # (shifted to local coords) so the reveal matches real speech rhythm;
        # otherwise the merged span is distributed evenly as a fallback.
        word_items: List[Dict] = []
        has_times = all(cue.get("word_times") is not None for cue in m["cues"])
        for cue in m["cues"]:
            times = cue.get("word_times")
            for i, word in enumerate(cue["words"]):
                if has_times and times is not None and i < len(times):
                    ws, we = times[i]
                    word_items.append({
                        "word": word,
                        "start": max(0.0, ws - clip_start),
                        "end": max(0.0, we - clip_start),
                        "timing_source": "canonical",
                    })
                else:
                    # Hardening v3 D3 (#25): a word without real timing is a
                    # SYNTHETIC HINT, not real timing — mark it explicitly so
                    # downstream never mistakes it for canonical timing.
                    word_items.append({"word": word, "timing_source": "synthetic_hint"})

        if not word_items:
            continue

        if not has_times:
            # Monotonic timing fallback: distribute the MERGED span evenly
            # across all words in order. Overlapping ASR cues share time, so
            # per-cue timing would interleave words out of order (random-looking
            # reveal). Flat, ordered timing keeps the reveal left-to-right.
            span = max(local_end - local_start, 0.05)
            per_word = span / len(word_items)
            for i, wi in enumerate(word_items):
                wi["start"] = local_start + i * per_word
                wi["end"] = local_start + (i + 1) * per_word

        lines.append({
            "words": word_items,
            "start": local_start,
            "end": local_end,
            "speaker": m.get("speaker", ""),
        })
    return lines


WHISPER_MODEL = os.getenv("RENDER_WHISPER_MODEL", "base")
WHISPER_DEVICE = os.getenv("RENDER_WHISPER_DEVICE", "cpu")
_whisper_model = None


def _get_whisper_model():
    """Lazily load the faster-whisper model once per process."""
    global _whisper_model
    if _whisper_model is None:
        from faster_whisper import WhisperModel
        _whisper_model = WhisperModel(
            WHISPER_MODEL, device=WHISPER_DEVICE, compute_type="int8"
        )
    return _whisper_model


def _transcribe_with_whisper(
    source_path: str,
    clip_start: float,
    clip_end: float,
    work_dir: str,
    language: str = "",
) -> List[CaptionRequest]:
    """Transcribe the clip window with faster-whisper word timestamps.

    Extracts the clip's audio (ffmpeg), runs faster-whisper with
    word_timestamps=True, and returns one CaptionRequest per whisper segment.
    Each word carries its precise start/end (absolute video coords) so the
    karaoke reveal matches the actual speech rhythm — not an even estimate.

    Hardening v3 D2 (#23): `language` comes from the contract caption_plan —
    the fallback NEVER hard-codes English when a language is available.
    Pass "" to let whisper auto-detect (only used when language is absent).
    """
    import subprocess as sp

    audio_path = os.path.join(work_dir, "clip_audio.wav")
    cmd = [
        "ffmpeg", "-y", "-loglevel", "error",
        "-ss", f"{clip_start:.3f}",
        "-i", source_path,
        "-to", f"{max(clip_end - clip_start, 0.05):.3f}",
        "-vn", "-ac", "1", "-ar", "16000",
        audio_path,
    ]
    sp.run(cmd, check=True)

    model = _get_whisper_model()
    transcribe_kwargs: Dict[str, object] = {
        "word_timestamps": True,
        "vad_filter": True,
        "beam_size": 5,
    }
    if language:
        transcribe_kwargs["language"] = language
    segments, _info = model.transcribe(audio_path, **transcribe_kwargs)

    captions: List[CaptionRequest] = []
    for seg in segments:
        text = " ".join(seg.text.split()).strip()
        if not text:
            continue
        words: List[CaptionWord] = []
        for w in (seg.words or []):
            wt = " ".join(w.word.split()).strip()
            if wt:
                words.append(CaptionWord(
                    start_sec=clip_start + float(w.start),
                    end_sec=clip_start + float(w.end),
                    text=wt,
                ))
        captions.append(CaptionRequest(
            start_sec=clip_start + float(seg.start),
            end_sec=clip_start + float(seg.end),
            text=text,
            words=words,
        ))

    try:
        os.remove(audio_path)
    except OSError:
        pass
    return captions


# Phase 6: speaker diarization (pyannote). Gives each caption line a speaker
# label so each speaker is rendered in their own color. Lazy-loaded per
# process; disabled if pyannote is not installed or RENDER_DIARIZE=0.
DIARIZE_ENABLED = os.getenv("RENDER_DIARIZE", "1") != "0"
DIARIZE_MODEL = os.getenv("RENDER_DIARIZE_MODEL", "pyannote/speaker-diarization-3.1")
_diarize_pipeline = None


def _get_diarize_pipeline():
    """Lazily load the pyannote diarization pipeline once per process."""
    global _diarize_pipeline
    if _diarize_pipeline is None:
        from pyannote.audio import Pipeline
        _diarize_pipeline = Pipeline.from_pretrained(
            DIARIZE_MODEL,
            token=os.getenv("HF_TOKEN"),
        )
    return _diarize_pipeline


def _diarize_clip(source_path: str, clip_start: float, clip_end: float, work_dir: str) -> List[Dict]:
    """Run speaker diarization on the clip window.

    Returns a list of {start, end, speaker} turns (start/end in ABSOLUTE video
    coords). Falls back to [] on any error so caption rendering still works.
    """
    import subprocess as sp
    import soundfile as sf

    audio_path = os.path.join(work_dir, "diar_audio.wav")
    sp.run([
        "ffmpeg", "-y", "-loglevel", "error",
        "-ss", f"{clip_start:.3f}",
        "-i", source_path,
        "-to", f"{max(clip_end - clip_start, 0.05):.3f}",
        "-vn", "-ac", "1", "-ar", "16000",
        audio_path,
    ], check=True)

    try:
        pipeline = _get_diarize_pipeline()
        data, sr = sf.read(audio_path, dtype="float32")
        if data.ndim == 1:
            data = data[None, :]
        import torch
        out = pipeline({"waveform": torch.from_numpy(data), "sample_rate": sr})
        ann = getattr(out, "speaker_diarization", out)
        turns: List[Dict] = []
        for turn, _, speaker in ann.itertracks(yield_label=True):
            turns.append({
                "start": clip_start + turn.start,
                "end": clip_start + turn.end,
                "speaker": speaker,
            })
        return turns
    except Exception as e:  # noqa: BLE001
        print(f"[diarize] failed: {e}", flush=True)
        return []
    finally:
        try:
            os.remove(audio_path)
        except OSError:
            pass


def _assign_speakers(captions: List[CaptionRequest], turns: List[Dict]) -> List[CaptionRequest]:
    """Tag each caption line with its speaker, splitting mixed-speaker lines.

    Whisper segments are utterance boundaries, but two speakers often talk
    over each other, so ONE segment can contain words from both speakers.
    We use the per-word timestamps to decide each word's speaker from the
    diarization turns, then split the caption at speaker changes — each
    resulting line gets its own speaker and color, and overlaps render as
    alternating lines instead of a blended monochrome blob.
    """
    if not turns:
        return captions

    def speaker_at(t: float) -> str:
        """Speaker of the turn containing timestamp t (absolute coords).

        For a point strictly inside a turn this returns that turn's speaker
        directly; for points in overlap/gap we pick the nearest turn. The
        naive min(t,end)-max(t,start) overlap is 0 for interior points, which
        would leave every word unlabeled.
        """
        best, best_dist = "", float("inf")
        for tr in turns:
            if tr["start"] <= t <= tr["end"]:
                return tr["speaker"]
            dist = min(abs(t - tr["start"]), abs(t - tr["end"]))
            if dist < best_dist:
                best, best_dist = tr["speaker"], dist
        return best

    out: List[CaptionRequest] = []
    for cap in captions:
        if not cap.words:
            # No word timestamps: assign by midpoint, keep one line.
            cap.speaker = speaker_at((cap.start_sec + cap.end_sec) / 2)
            out.append(cap)
            continue

        # Tag each word with its speaker.
        tagged: List[Tuple[str, CaptionWord]] = []
        for w in cap.words:
            spk = speaker_at((w.start_sec + w.end_sec) / 2)
            tagged.append((spk, w))

        # Group consecutive words by speaker.
        groups: List[Tuple[str, List[CaptionWord]]] = []
        for spk, w in tagged:
            if groups and groups[-1][0] == spk:
                groups[-1][1].append(w)
            else:
                groups.append((spk, [w]))

        if len(groups) == 1:
            cap.speaker = groups[0][0]
            out.append(cap)
            continue

        # Split into separate caption lines, one per speaker run.
        for spk, words in groups:
            text = " ".join(w.text for w in words).strip()
            if not text:
                continue
            out.append(CaptionRequest(
                start_sec=words[0].start_sec,
                end_sec=words[-1].end_sec,
                text=text,
                words=words,
                speaker=spk,
            ))
    return out


# Speaker -> caption color palette. SPEAKER_00 keeps the classic white; the
# rest cycle through high-contrast colors that read well over video.
SPEAKER_COLORS = {
    "SPEAKER_00": (255, 255, 255),
    "SPEAKER_01": (255, 220, 80),    # amber
    "SPEAKER_02": (120, 220, 255),   # sky
    "SPEAKER_03": (180, 255, 160),   # mint
    "SPEAKER_04": (255, 160, 220),   # pink
}


def _speaker_color(speaker: str) -> tuple:
    if speaker in SPEAKER_COLORS:
        return SPEAKER_COLORS[speaker]
    # Unknown labels: derive a stable color from the label hash.
    import hashlib
    h = int(hashlib.md5(speaker.encode()).hexdigest()[:6], 16)
    return ((h >> 16) & 255, (h >> 8) & 255, h & 255)


# Structured QC (brief §23): caption metrics collected during burn.
_CAPTION_COLLISION_HITS = 0
_CAPTION_OVERFLOW_HITS = 0


def _burn_karaoke_captions(
    video_path: str,
    captions: List[CaptionRequest],
    clip_start: float,
    out_path: str,
    work_dir: str,
    timeline: Optional[object] = None,
) -> int:
    """Burn karaoke word-highlight captions by compositing per-frame overlays.

    For each frame we paste pre-rendered word sprites onto a transparent
    canvas: the active word in white, all others in grey. The overlay PNG
    sequence is then composited over the video with ffmpeg and re-encoded
    to H.264.

    Phase 2 (render timelines): pass the clip's RenderTimeline artifact to
    avoid module-global reads for face/split data. When None, falls back to
    the legacy module getters (kept for old callers).

    Returns the number of caption dialogue lines written.
    """
    import subprocess
    import cv2
    from PIL import Image

    # Merge overlapping ASR cues into non-overlapping continuous lines, so only
    # one caption line is on screen at any moment (fixes duplicate captions).
    lines = _normalize_cues(captions, clip_start)
    print(f"[caption] normalize: {len(captions)} caps -> {len(lines)} lines (clip_start={clip_start:.2f})", flush=True)
    if not lines:
        return 0

    # Video properties. Use ffprobe (not cv2.VideoCapture) so FFV1/mkv
    # lossless intermediates work — OpenCV's VideoCapture cannot decode FFV1
    # and silently returns 0 frames, which erased every caption.
    try:
        import json as _json
        probe = subprocess.run(
            ["ffprobe", "-v", "error", "-select_streams", "v:0",
             "-show_entries", "stream=width,height,r_frame_rate",
             "-of", "json", video_path],
            capture_output=True, text=True, timeout=30,
        )
        probe_data = _json.loads(probe.stdout)["streams"][0]
        width = int(probe_data["width"])
        height = int(probe_data["height"])
        rfr = probe_data.get("r_frame_rate", "25/1").split("/")
        fps = float(rfr[0]) / float(rfr[1]) if len(rfr) == 2 and float(rfr[1]) else 25.0
    except Exception:  # noqa: BLE001
        # Fallback: cv2 (works for plain mp4/h264 sources).
        cap = cv2.VideoCapture(video_path)
        if not cap.isOpened():
            raise RuntimeError(f"could not open {video_path}")
        fps = cap.get(cv2.CAP_PROP_FPS) or 30.0
        width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
        cap.release()
    print(f"[caption] video {os.path.basename(video_path)}: {width}x{height} fps={fps:.3f}", flush=True)

    # ── Phase 3 (brief §44): face collision avoidance needs face boxes ──
    # Phase 2: prefer the explicit timeline artifact when provided.
    # Brief v5 V-01: timeline_ref is used by compose() via state_at(ts) so
    # caption avoidance follows per-frame geometry, not a final snapshot.
    # Brief v6 R06: production REQUIRES an explicit timeline — never read
    # module-global statistics from a previous clip.
    if timeline is None:
        raise RenderTimelineMissingError(
            "caption compositor requires an explicit RenderTimeline; "
            "refusing to fall back to module-global face tracks"
        )
    timeline_ref = timeline
    face_tracks_ref = list(getattr(timeline, "face_tracks", []) or [])
    speaker_track_id_ref = getattr(timeline, "speaker_track_id", None)
    split_ranges_ref = list(getattr(timeline, "split_ranges", []) or [])
    # Time intervals (start_sec, end_sec) where the shot uses the reaction
    # split layout. When split is active the speaker is in the TOP pane and
    # the reactor in the BOTTOM pane, so a bottom-anchored caption would cover
    # the reactor's face. We shift the caption block to the middle of the seam
    # for exactly those frames (a clip can alternate single/split several
    # times; the final frame is usually single view).
    def _is_split_frame(ts: float) -> bool:
        return any(start <= ts <= end for (start, end) in split_ranges_ref)

    # Pre-render sprites per word (active + idle) and lay words into wrapped
    # visual lines (max ~92% of frame width). Each visual line keeps the word
    # ordering so we can compute the active word from elapsed time.
    from PIL import ImageFont
    font_scale = float(os.getenv("RENDER_CAPTION_FONT_SCALE", "0.042"))
    base_size = max(int(height * font_scale), 14)
    font_path = "C:/Windows/Fonts/arialbd.ttf"
    font = ImageFont.truetype(font_path, base_size)
    space = int(base_size * 0.35)
    max_line_w = int(width * 0.92)

    # Build flat word list with per-word timing (each word keeps its own
    # start/end from its source cue — see _normalize_cues).
    flat: List[Dict] = []
    for line in lines:
        color = _speaker_color(line.get("speaker", ""))
        for wi in line["words"]:
            flat.append({
                "word": wi["word"],
                "start": wi["start"],
                "end": wi["end"],
                "cap_start": line["start"],
                "cap_end": line["end"],
                "speaker": line.get("speaker", ""),
                "color": color,
            })

    # Render sprites (one per word, colored by speaker — reveal style).
    for item in flat:
        item["sprite"] = _make_word_sprite(item["word"], font, item["color"])
        # Karaoke highlight: a second sprite in the accent color, shown only
        # while this word is the one being spoken.
        if CAPTION_HIGHLIGHT:
            item["sprite_hi"] = _make_word_sprite(item["word"], font, HIGHLIGHT_COLOR)
        else:
            item["sprite_hi"] = item["sprite"]

    # ── Phase 3 (brief §44): word/line budget ──
    # Max 3-6 words per visual line and max 2 lines per caption display. The
    # wrap breaks on BOTH width and word count so short bursts don't stretch
    # into one long line, and a 3-word sentence never fills 6 slots.
    caption_max_words = int(os.getenv("RENDER_CAPTION_MAX_WORDS", "4"))
    caption_min_words = int(os.getenv("RENDER_CAPTION_MIN_WORDS", "2"))
    caption_max_lines = int(os.getenv("RENDER_CAPTION_MAX_LINES", "1"))

    # Wrap into visual lines by cumulative width AND word budget. Each caption
    # cue is wrapped independently so two cues never share a visual line.
    visual_lines: List[Dict] = []
    for line in lines:
        cur: List[Dict] = []
        cur_w = 0
        # Build this caption's flat items.
        caption_items = [it for it in flat if abs(it["cap_start"] - line["start"]) < 0.01]
        for item in caption_items:
            w = item["sprite"].width
            needed = w + (space if cur else 0)
            # Break if width would overflow, or if adding this word exceeds the
            # max words per line.
            if cur and (cur_w + needed > max_line_w or len(cur) >= caption_max_words):
                # Drop trailing tiny words (e.g. a single "the") onto the next
                # line only if that leaves the current line >= min words.
                if len(cur) >= caption_min_words:
                    visual_lines.append({"items": cur, "width": cur_w})
                    cur = [item]
                    cur_w = item["sprite"].width
                else:
                    cur.append(item)
                    cur_w += needed
            else:
                cur.append(item)
                cur_w += needed
        if cur:
            visual_lines.append({"items": cur, "width": cur_w})

    # ── Phase 3 (brief §44): max 2 visible lines ──
    # NOTE: we do NOT slice the global list here (that would keep only the
    # last two lines of the WHOLE clip — every earlier caption would vanish).
    # The limit is applied per-frame inside compose() instead: at any moment
    # only the most recent caption_max_lines are drawn.

    overlay_dir = os.path.join(work_dir, "overlay")
    os.makedirs(overlay_dir, exist_ok=True)

    # Compose overlays for each frame. Only the visual lines active at this
    # timestamp are drawn, stacked upward from the bottom margin — this keeps
    # the layout tight when a caption wraps to two lines and avoids stacking
    # every caption in the clip into one tall block.
    line_gap = int(base_size * 0.15)
    lead_sec = CAPTION_LEAD_MS / 1000.0
    hold_sec = CAPTION_HOLD_MS / 1000.0

    def compose(ts: float, path: str) -> None:
        global _CAPTION_COLLISION_HITS, _CAPTION_OVERFLOW_HITS
        canvas = Image.new("RGBA", (width, height), (0, 0, 0, 0))
        # A line is active from its first word's reveal (start - lead) until its
        # last word's end + hold.
        active_lines = [
            line for line in visual_lines
            if (min(it["start"] for it in line["items"]) - lead_sec) <= ts <= (max(it["end"] for it in line["items"]) + hold_sec)
        ]
        # Phase 3 (brief §44): max 2 visible lines at any moment — keep the
        # most recent ones (end time latest first), not the whole history.
        if caption_max_lines >= 1 and len(active_lines) > caption_max_lines:
            active_lines = sorted(active_lines, key=lambda l: max(it["end"] for it in l["items"]))[-caption_max_lines:]
        if not active_lines:
            canvas.save(path)
            return

        # ── Phase 3 (brief §44) + Brief v5 V-01: face collision avoidance ──
        # When enabled, drop the bottom-margin anchor down/up around a detected
        # face (mouth zone) so captions never cover the speaker's mouth.
        # Brief v5 7.2: use time-indexed timeline state (state_at) per frame
        # rather than the FINAL face snapshot — captions stay synchronized with
        # camera/face movement across the clip.
        face_avoid = os.getenv("RENDER_CAPTION_FACE_AVOIDANCE", "1") != "0"
        mouth_zone: Optional[Tuple[int, int, int, int]] = None  # (x0,y0,x1,y1)
        if face_avoid:
            frame_state = None
            if timeline_ref is not None:
                try:
                    frame_state = timeline_ref.state_at(ts)
                except Exception:  # noqa: BLE001
                    frame_state = None
            faces_at_ts = None
            if frame_state is not None and frame_state.get("faces"):
                faces_at_ts = frame_state["faces"]
            elif face_tracks_ref:
                faces_at_ts = face_tracks_ref
            if faces_at_ts:
                # Prefer the active speaker track at this timestamp.
                active_sp_id = frame_state.get("active_speaker_id") if frame_state else (speaker_track_id_ref if speaker_track_id_ref is not None else None)
                sp = next((t for t in faces_at_ts if t.get("track_id") == active_sp_id), None)
                f = sp or (max(faces_at_ts, key=lambda t: t.get("area", 0)) if faces_at_ts else None)
                if f is not None and f.get("w"):
                    bx0 = int(max(0, f["cx"] - f["w"] / 2))
                    by0 = int(max(0, f["cy"] - f["h"] / 2))
                    bx1 = int(min(width, f["cx"] + f["w"] / 2))
                    by1 = int(min(height, f["cy"] + f["h"] / 2))
                    mouth_zone = (bx0, by0, bx1, by1)

        # Stack active lines upward from the bottom margin (higher on screen to
        # clear the source video's own lower-third text/watermarks).
        total_h = sum(l["items"][0]["sprite"].height for l in active_lines) + line_gap * (len(active_lines) - 1)
        y = height - total_h - int(height * CAPTION_BOTTOM_MARGIN)
        # In split layout the reactor's face is in the BOTTOM pane, so a
        # bottom-anchored caption would cover it. Move the caption block to the
        # vertical CENTER of the frame (straddling the pane seam) instead.
        if _is_split_frame(ts):
            y = max(int(height * 0.12), int((height - total_h) // 2))
        # If the speaker's face/mouth sits in the lower area where the caption
        # block would go, move the block ABOVE the face instead.
        elif mouth_zone is not None and y < mouth_zone[3] and mouth_zone[3] > height * 0.35:
            y = max(int(height * 0.12), mouth_zone[1] - total_h - line_gap)
            _CAPTION_COLLISION_HITS += 1
        # Overflow: caption block taller than 40% of the frame (bad wrapping).
        if total_h > height * 0.4:
            _CAPTION_OVERFLOW_HITS += 1
        for line in active_lines:
            x = (width - line["width"]) // 2
            for item in line["items"]:
                # Reveal: the word appears (lead_sec early) and stays visible.
                if ts >= item["start"] - lead_sec:
                    # Karaoke: use the accent sprite while this word is the one
                    # currently being spoken (between its own start and end),
                    # otherwise the base-color sprite.
                    if CAPTION_HIGHLIGHT and item["start"] <= ts < item["end"]:
                        spr = item["sprite_hi"]
                    else:
                        spr = item["sprite"]
                    canvas.paste(spr, (x, y), spr)
                x += item["sprite"].width + space
            y += line["items"][0]["sprite"].height + line_gap
        canvas.save(path)

    # Re-open to count frames properly (ffprobe duration * fps; cv2 can't
    # count FFV1 frames).
    frame_count = 0
    try:
        import json as _json2
        probe2 = subprocess.run(
            ["ffprobe", "-v", "error", "-show_entries", "format=duration",
             "-of", "json", video_path],
            capture_output=True, text=True, timeout=30,
        )
        dur2 = float(_json2.loads(probe2.stdout)["format"]["duration"])
        frame_count = int(round(dur2 * fps))
    except Exception:  # noqa: BLE001
        try:
            cap = cv2.VideoCapture(video_path)
            frame_count = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
            cap.release()
        except Exception:  # noqa: BLE001
            frame_count = 0
    if frame_count <= 0:
        frame_count = 1
    print(f"[caption] frame_count={frame_count} (dur*fps)", flush=True)

    overlay_paths: List[str] = []
    non_empty_overlays = 0
    for i in range(frame_count):
        ts = i / fps
        p = os.path.join(overlay_dir, f"ov_{i:05d}.png")
        compose(ts, p)
        # Quick check: count overlays with actual content (non-transparent).
        try:
            from PIL import Image as _Img
            _ov = _Img.open(p).convert("RGBA")
            if _ov.getextrema()[3][1] > 0:
                non_empty_overlays += 1
        except Exception:  # noqa: BLE001
            pass
        overlay_paths.append(p)
    print(f"[caption] overlay: {non_empty_overlays}/{frame_count} frames with content", flush=True)

    # Composite overlays over the video with ffmpeg.
    # ── Phase 3 (brief §39): keep this intermediate LOSSLESS (FFV1). The
    # single lossy H.264 encode happens once at the very end of the pipeline,
    # after captions AND hook are composited — never per-stage.
    tmp_out = out_path + ".captioned.mkv"
    seq = os.path.join(overlay_dir, "ov_%05d.png")
    cmd = [
        "ffmpeg", "-y", "-loglevel", "error",
        "-i", video_path,
        "-framerate", f"{fps:.3f}", "-i", seq,
        "-filter_complex", "[0:v][1:v]overlay=0:0[out]",
        "-map", "[out]", "-map", "0:a?",
        "-c:v", "ffv1",
        "-c:a", "copy",
        "-shortest",
        tmp_out,
    ]
    subprocess.run(cmd, check=True)
    os.replace(tmp_out, out_path)

    # Cleanup overlay PNGs.
    for p in overlay_paths:
        try:
            os.remove(p)
        except OSError:
            pass
    try:
        os.rmdir(overlay_dir)
    except OSError:
        pass

    return len(lines)


def _build_hook_intro(video_path: str, hook: str, work_dir: str) -> Optional[str]:
    """Build the hook intro video (frame + dim + hook text + TTS voiceover).

    Returns the path to an mp4 (video + voiceover audio) or None if the hook
    is empty / disabled / a step fails. The intro is prepended to the rendered
    short; its duration is the voiceover length clamped to [HOOK_MIN_SEC,
    HOOK_MAX_SEC].

    Pipeline:
      1. ffmpeg: extract the FIRST frame of the clip at output resolution.
      2. Pillow: darken the frame, wrap the hook text large and centered.
      3. Edge-TTS: synthesize the voiceover (mp3).
      4. ffmpeg: loop the still image for the voiceover duration, mux audio.
    """
    hook = " ".join(hook.split()).strip()
    if not HOOK_ENABLED or not hook:
        return None
    import subprocess as sp
    from PIL import Image, ImageDraw, ImageFont

    # 1. First frame at native size.
    frame_path = os.path.join(work_dir, "hook_frame.jpg")
    sp.run([
        "ffmpeg", "-y", "-loglevel", "error",
        "-i", video_path, "-frames:v", "1",
        frame_path,
    ], check=True)

    # 2. Darkened frame + wrapped hook text.
    img = Image.open(frame_path).convert("RGB")
    w, h = img.size
    dark = Image.new("RGB", (w, h), (0, 0, 0))
    img = Image.blend(img, dark, HOOK_DIM_ALPHA)
    draw = ImageDraw.Draw(img)

    base_size = max(int(h * HOOK_FONT_SCALE), 18)
    # Eye-catching display font: Anton (Google Fonts, shipped locally) with
    # fallbacks to Impact / Arial Black / Arial Bold.
    font_candidates = [
        "C:/Windows/Fonts/impact.ttf",
        os.path.join(os.path.dirname(os.path.abspath(__file__)), "fonts", "Anton-Regular.ttf"),
        "C:/Windows/Fonts/ariblk.ttf",
        "C:/Windows/Fonts/arialbd.ttf",
    ]
    font = None
    for fp in font_candidates:
        try:
            font = ImageFont.truetype(fp, base_size)
            break
        except Exception:  # noqa: BLE001
            continue
    if font is None:
        font = ImageFont.load_default()

    # Wrap text: max ~70% width per line, center vertically.
    max_w = int(w * 0.70)
    lines: List[str] = []
    for word in hook.split():
        if not lines:
            lines.append(word)
            continue
        trial = lines[-1] + " " + word
        if draw.textlength(trial, font=font) <= max_w:
            lines[-1] = trial
        else:
            lines.append(word)
    line_h = base_size * 1.25
    total_h = line_h * len(lines)
    y = (h - total_h) // 2
    outline_w = max(3, base_size // 10)  # thick outline for punch
    for line in lines:
        lw = draw.textlength(line, font=font)
        x = (w - lw) // 2
        # Thick black outline (offset by outline_w in 8 directions) for a bold
        # shorts-style look that stays readable over any background.
        for dx in range(-outline_w, outline_w + 1):
            for dy in range(-outline_w, outline_w + 1):
                if dx * dx + dy * dy <= outline_w * outline_w:
                    draw.text((x + dx, y + dy), line, font=font, fill=(0, 0, 0))
        draw.text((x, y), line, font=font, fill=(255, 255, 255))
        y += line_h
    img.save(frame_path, quality=92)

    # 3. Edge-TTS voiceover.
    audio_path = os.path.join(work_dir, "hook_voice.mp3")
    try:
        import asyncio
        import edge_tts
        communicate = edge_tts.Communicate(hook, HOOK_TTS_VOICE, rate=HOOK_TTS_RATE)
        asyncio.run(communicate.save(audio_path))
    except Exception as e:  # noqa: BLE001
        print(f"[hook] TTS failed: {e}", flush=True)
        return None

    # Measure voiceover duration.
    try:
        probe = sp.run(
            ["ffprobe", "-v", "error", "-show_entries", "format=duration",
             "-of", "csv=p=0", audio_path],
            capture_output=True, text=True, check=True,
        )
        dur = float(probe.stdout.strip())
    except Exception:  # noqa: BLE001
        dur = 2.5
    dur = max(HOOK_MIN_SEC, min(dur, HOOK_MAX_SEC))

    # 4. Loop still + mux voiceover. `-t` on both inputs bounds the output;
    # `-shortest` additionally stops at the shorter stream (edge-tts mp3 can
    # carry a bloated duration in its header, so never trust its stream length).
    intro_path = os.path.join(work_dir, "hook_intro.mp4")
    sp.run([
        "ffmpeg", "-y", "-loglevel", "error",
        "-loop", "1", "-t", f"{dur:.3f}", "-i", frame_path,
        "-i", audio_path,
        "-vf", "format=yuv420p",
        "-c:v", "libx264", "-preset", "fast", "-crf", "20",
        "-c:a", "aac", "-b:a", "128k", "-shortest",
        intro_path,
    ], check=True)
    return intro_path


def _pick_best_frame(video_path: str, work_dir: str) -> str:
    """Find the most engaging frame (largest detected face) and save it.

    Scans the video in steps, runs YuNet face detection (same model as the
    clipper), and returns the path of the frame with the biggest face area —
    faces are the strongest thumbnail hook. Falls back to the first frame.
    """
    import cv2

    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        return ""
    fps = cap.get(cv2.CAP_PROP_FPS) or 30.0
    total = int(cap.get(cv2.CAP_PROP_FRAME_COUNT)) or 0
    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))

    model_path = os.path.join(
        os.path.dirname(os.path.abspath(__file__)),
        "shorts_generator", "local", "models", "face_detection_yunet_2023mar.onnx",
    )
    detector = None
    try:
        detector = cv2.FaceDetectorYN_create(model_path, "", (width, height))
    except Exception:  # noqa: BLE001
        detector = None

    best_path, best_area = "", 0.0
    step = max(1, int(fps * 0.5))  # sample every 0.5s
    frame_idx = 0
    saved = 0
    out = os.path.join(work_dir, "thumb_frames")
    os.makedirs(out, exist_ok=True)
    while True:
        ret, frame = cap.read()
        if not ret:
            break
        if frame_idx % step == 0 and detector is not None:
            try:
                _, faces = detector.detect(frame)
                if faces is not None and len(faces) > 0:
                    # Score by usable area: prefer a MEDIUM SHOT (face ~10-15%
                    # of frame = head-and-shoulders / half body, like popular
                    # Shorts thumbnails). Too close (face fills frame) or too
                    # far (tiny face) both score lower.
                    for f in faces:
                        fx, fy, fw, fh = float(f[0]), float(f[1]), float(f[2]), float(f[3])
                        area = fw * fh
                        frame_area = float(width * height)
                        ratio = area / frame_area if frame_area > 0 else 0
                        fit = 1.0 - abs(ratio - 0.12) / 0.12  # peak at ~12%
                        score = area * max(fit, 0.2)
                        if score > best_area:
                            best_area = score
                            p = os.path.join(out, f"best_{saved}.jpg")
                            cv2.imwrite(p, frame)
                            best_path = p
                            saved += 1
            except Exception:  # noqa: BLE001
                pass
        frame_idx += 1
    cap.release()

    if not best_path:
        # Fallback: first frame.
        best_path = os.path.join(out, "first.jpg")
        cap = cv2.VideoCapture(video_path)
        ok, frame = cap.read()
        cap.release()
        if ok:
            cv2.imwrite(best_path, frame)
        else:
            return ""
    return best_path


def _build_thumbnail(video_path: str, hook: str, work_dir: str) -> Optional[str]:
    """Generate a Shorts-style thumbnail: best frame + big Impact hook text.

    Returns the path to a 9:16 (portrait) JPG matching the video's native
    resolution, or None on failure. The frame with the best-placed face is
    used as-is (no cover-crop — cropping a portrait frame to landscape pushes
    the subject out of frame). A dark gradient at the bottom keeps the hook
    text readable; the hook is rendered in Impact with a thick outline.
    """
    hook = " ".join(hook.split()).strip()
    from PIL import Image, ImageDraw, ImageFont

    frame_path = _pick_best_frame(video_path, work_dir)
    if not frame_path:
        return None

    img = Image.open(frame_path).convert("RGB")
    TARGET_W, TARGET_H = img.size  # keep native portrait resolution
    draw = ImageDraw.Draw(img)
    # No dark gradient / strip: text sits directly on the photo, only the
    # thick outline keeps it readable. Transparent background as requested.

    if hook:
        # Shorten the hook for thumbnail readability: keep the last ~6 words
        # (the payoff) unless the whole line is already short. A wall of text
        # on a thumbnail gets skipped; a punchy fragment gets clicked.
        words = hook.split()
        if len(words) > 7:
            hook = " ".join(words[-6:])
        # Font scales with frame width (portrait ~606px vs landscape 1280px).
        base_size = max(int(TARGET_W * 0.16), 56)
        font_candidates = [
            "C:/Windows/Fonts/impact.ttf",
            os.path.join(os.path.dirname(os.path.abspath(__file__)), "fonts", "Anton-Regular.ttf"),
            "C:/Windows/Fonts/ariblk.ttf",
        ]
        font = None
        for fp in font_candidates:
            try:
                font = ImageFont.truetype(fp, base_size)
                break
            except Exception:  # noqa: BLE001
                continue
        if font is None:
            font = ImageFont.load_default()

        # Wrap to max ~92% width.
        max_w = int(TARGET_W * 0.92)
        lines: List[str] = []
        for word in hook.split():
            if not lines:
                lines.append(word)
                continue
            trial = lines[-1] + " " + word
            if draw.textlength(trial, font=font) <= max_w:
                lines[-1] = trial
            else:
                lines.append(word)

        # Auto-shrink font if the text is still too wide (>=3 lines).
        while len(lines) >= 3 and base_size > 48:
            base_size = int(base_size * 0.85)
            font = ImageFont.truetype(font_candidates[0] if font_candidates else "", base_size)
            lines = []
            for word in hook.split():
                if not lines:
                    lines.append(word)
                    continue
                trial = lines[-1] + " " + word
                if draw.textlength(trial, font=font) <= max_w:
                    lines[-1] = trial
                else:
                    lines.append(word)

        line_h = base_size * 1.18
        total_h = line_h * len(lines)
        # Text at the TOP as a headline (leaves the bottom clear — the same
        # zone where subtitles will appear in the video).
        y = int(TARGET_H * 0.06)
        outline_w = max(4, base_size // 16)

        # Keyword highlight: render each line with its LAST word in amber
        # (the payoff word) and the rest in white — matches the punchy
        # white/yellow contrast seen on high-CTR Shorts thumbnails. No strip
        # behind the text (transparent background); the thick outline keeps it
        # readable over any frame.
        for line in lines:
            parts = line.rsplit(" ", 1)
            if len(parts) == 2:
                main_text, accent_text = parts
                main_w = draw.textlength(main_text, font=font)
                space_w = draw.textlength(" ", font=font)
                accent_w = draw.textlength(accent_text, font=font)
                total_w = main_w + space_w + accent_w
                x = (TARGET_W - total_w) // 2
                for dx in range(-outline_w, outline_w + 1):
                    for dy in range(-outline_w, outline_w + 1):
                        if dx * dx + dy * dy <= outline_w * outline_w:
                            draw.text((x + dx, y + dy), main_text, font=font, fill=(0, 0, 0))
                            draw.text((x + main_w + space_w + dx, y + dy), accent_text, font=font, fill=(0, 0, 0))
                draw.text((x, y), main_text, font=font, fill=(255, 255, 255))
                draw.text((x + main_w + space_w, y), accent_text, font=font, fill=(255, 220, 80))
            else:
                lw = draw.textlength(line, font=font)
                x = (TARGET_W - lw) // 2
                for dx in range(-outline_w, outline_w + 1):
                    for dy in range(-outline_w, outline_w + 1):
                        if dx * dx + dy * dy <= outline_w * outline_w:
                            draw.text((x + dx, y + dy), line, font=font, fill=(0, 0, 0))
                draw.text((x, y), line, font=font, fill=(255, 255, 255))
            y += line_h

    thumb_path = os.path.join(work_dir, "thumbnail.jpg")
    img.save(thumb_path, quality=92)
    return thumb_path


class JobTransitionConflict(RuntimeError):
    """Raised when a compare-and-swap job transition loses (brief v5 R-01).
    The caller must NOT start rendering; the state belongs to someone else."""


class RenderTimelineMissingError(RuntimeError):
    """Brief v6 4.6 — production cropper returned a bare path when an
    explicit per-job RenderTimeline was required. Never fall back to module
    globals."""


def _require_explicit_timeline(crop_result, job_id: str = ""):
    """Brief v6 4.6 — validate that crop_clip_local(return_timeline=True)
    actually returned (path, RenderTimeline). Raises RenderTimelineMissingError
    on a bare path so production never reads stale module-global stats."""
    from shorts_generator.local.clipper import RenderTimeline
    if (
        isinstance(crop_result, tuple)
        and len(crop_result) == 2
        and isinstance(crop_result[1], RenderTimeline)
    ):
        return crop_result
    raise RenderTimelineMissingError(
        f"cropper did not return an explicit timeline for job {job_id or '(unknown)'} "
        "(return_timeline=True must return (path, RenderTimeline))"
    )


class PersistenceError(RuntimeError):
    """Raised when a SQLite persistence write fails (brief v5 R-05). Callers
    must not update memory after a failed commit."""


class RenderOutcome:
    """Phase-2 correctness: _render returns BOTH the response and the final
    status it computed. The job orchestration layer (async/retry/sync
    wrappers) is the ONLY place allowed to persist terminal status, so
    partial_failure is never overwritten by a wrapper's blanket completed.
    """

    __slots__ = ("response", "final_status")

    def __init__(self, response: RenderResponse, final_status: str) -> None:
        self.response = response
        self.final_status = final_status


def _render(request, job_id: str) -> RenderOutcome:
    # job_id is supplied by the job service (Phase 1 §5.1). This function must
    # NEVER generate a replacement identity: the same id must appear in the
    # response, output directory, SQLite row, and artifact URLs.
    #
    # Phase-2 correctness: _render performs NON-terminal transitions and
    # returns RenderOutcome(response, final_status). It must NOT persist the
    # terminal status — the caller does that, exactly once.
    job_dir = RENDER_ROOT / job_id
    job_dir.mkdir(parents=True, exist_ok=True)

    # Normalize v1/v2 request (brief §16-17). v2 carries mode, narrative,
    # layout plan, caption plan, editing events; v1 is upgraded internally.
    clips = _normalize_clips(request)
    mode = getattr(request, "mode", "final") or "final"
    episode_id = getattr(request, "episode_id", "") or ""
    output_w = int(getattr(getattr(request, "output", None), "width", 1080) or 1080)
    output_h = int(getattr(getattr(request, "output", None), "height", 1920) or 1920)
    preview = mode == "preview"
    if preview:
        # Preview rendering (brief §21): cheaper, faster, smaller.
        output_w = min(output_w, int(os.getenv("RENDER_PREVIEW_WIDTH", "540")))
        output_h = min(output_h, int(os.getenv("RENDER_PREVIEW_HEIGHT", "960")))

    # The caller (async worker / sync / retry worker) owns the queued ->
    # downloading CAS. _render assumes it already won and proceeds.
    try:
        source = download_youtube_local(
            request.video_url,
            fmt=FORMAT,
            out_dir=str(RENDER_ROOT / "source"),
        )
        # Brief v6 R01: checked active transition — a lost CAS stops work.
        require_transition(job_id, "downloading", "analysing", mode=mode, episode_id=episode_id)
    except HTTPException:
        raise
    except Exception as e:  # noqa: BLE001
        raise HTTPException(status_code=502, detail=f"download failed: {e}") from e

    # 2. Render each clip as a vertical short, burning captions if provided.
    from shorts_generator.local.clipper import crop_clip_local
    import subprocess as sp

    # ── Phase 3/4 (brief §42-43): crop quality score + layout selection ──
    # Decide once per job from the source; per-clip face count refines it.
    layout_mode = "face_crop"
    crop_score = None
    try:
        from visual_effects import probe_source_resolution, crop_quality_score, choose_layout
        src_probe = probe_source_resolution(source)
        if src_probe:
            sw, sh, sratio = src_probe
            # Crop window at 9:16 from this source.
            if sratio < 9.0 / 16.0:
                cw, ch = int(sh * 9 / 16), sh
            else:
                cw, ch = sw, int(sw * 16 / 9)
            crop_score = crop_quality_score(sw, sh, cw, ch, face_count=0)
            layout_mode = choose_layout(crop_score, 0, sratio)
            print(
                f"[render] source {sw}x{sh} ratio={sratio:.2f} crop_score={crop_score} "
                f"layout={layout_mode}",
                flush=True,
            )
    except Exception as e:  # noqa: BLE001
        print(f"[render] layout selection failed ({e}), using face_crop", flush=True)

    start = time.time()
    rendered = []
    artifacts = []
    # Brief v6 R01: checked active transition — a lost CAS stops work.
    require_transition(job_id, "analysing", "rendering", mode=mode, episode_id=episode_id)
    for i, c in enumerate(clips, 1):
        out_path = os.path.join(job_dir, f"short_{i:02d}.mp4")
        item = {
            "clip_id": c["clip_id"],
            "title": c["title"],
            "start_sec": c["start_sec"],
            "end_sec": c["end_sec"],
            "status": "error",
            "duration_sec": round(float(c["end_sec"]) - float(c["start_sec"]), 2),
        }
        artifact = RenderArtifact(
            clip_id=c["clip_id"],
            status="error",
            requested_layout=c["preferred_layout"],
            duration_sec=round(float(c["end_sec"]) - float(c["start_sec"]), 2),
        )
        try:
            print(f"[render] clip {i}/{len(clips)}: {c['title'] or c['clip_id']} mode={mode}", flush=True)

            # ── Phase 4 (brief §47): derive emphasis events from captions ──
            emphasis_events = None
            if c["captions"]:
                try:
                    from visual_effects import build_emphasis_events
                    raw_events = build_emphasis_events(c["captions"], float(c["end_sec"]) - float(c["start_sec"]))
                    emphasis_events = [
                        {
                            "time": round(float(ev["time"]) - float(c["start_sec"]), 2),
                            "type": ev["type"],
                            "intensity": ev["intensity"],
                        }
                        for ev in raw_events
                    ]
                    if emphasis_events:
                        print(f"[render] clip {i}: emphasis events -> {len(emphasis_events)}", flush=True)
                except Exception as e:  # noqa: BLE001
                    print(f"[render] clip {i}: emphasis events failed ({e}), continuing", flush=True)

            # Per-clip layout decision honoring the miner's preferred layout
            # when technical quality allows (brief §17); report fallback.
            clip_layout = layout_mode
            fallback_reason = None
            preferred = c["preferred_layout"]
            try:
                if preferred and preferred != "auto":
                    # Miner asked for a specific layout; only fall back when the
                    # source cannot technically support it.
                    from visual_effects import probe_source_resolution, crop_quality_score
                    src_probe = probe_source_resolution(source)
                    if src_probe:
                        sw, sh, sratio = src_probe
                        if sratio < 9.0 / 16.0:
                            cw, ch = int(sh * 9 / 16), sh
                        else:
                            cw, ch = sw, int(sw * 16 / 9)
                        score = crop_quality_score(sw, sh, cw, ch, face_count=c["expected_speakers"] or 0)
                        min_score = int(os.getenv("RENDER_CROP_QUALITY_MIN_SCORE", "60"))
                        if preferred == "face_crop" and score < min_score:
                            clip_layout = "blur_background" if c["allow_blur_background"] else "face_crop"
                            fallback_reason = "effective vertical crop below minimum quality"
                        else:
                            clip_layout = preferred
                        print(f"[render] clip {i}: requested={preferred} score={score} actual={clip_layout}"
                              f"{' (' + fallback_reason + ')' if fallback_reason else ''}", flush=True)
            except Exception as e:  # noqa: BLE001
                print(f"[render] clip {i}: layout fallback check failed ({e}), using {clip_layout}", flush=True)

            # Phase 2 (render timelines): request the explicit artifact so we
            # never read module globals for QC/caption data.
            crop_result = crop_clip_local(
                source,
                float(c["start_sec"]),
                float(c["end_sec"]),
                "9:16",
                out_path,
                cache_dir=str(RENDER_ROOT / "cache"),
                final_encode=False,  # Phase 3: keep lossless until final H.264 pass
                emphasis_events=emphasis_events,
                layout_mode=clip_layout,
                output_size=(output_w, output_h) if preview else None,
                return_timeline=True,
            )
            # Brief v6 4.6/R06: production REQUIRES an explicit timeline.
            # A bare path is a contract violation — never fall back to globals.
            _, timeline = _require_explicit_timeline(crop_result, job_id)
            # The crop succeeded; final status may still flip to error later if
            # the quality gate fails.
            item["status"] = "ok"
            artifact.status = "ok"

            # Phase 4 (brief §23): structured tracking QC from the clipper.
            try:
                # Brief v6 R06: timeline is REQUIRED — never module globals.
                stats = timeline.stats
                artifact.qc.focus_switch_count = int(stats.get("focus_switch_count", 0) or 0)
                artifact.qc.focus_ping_pong_detected = bool(stats.get("focus_ping_pong_detected", False))
                artifact.qc.random_crop_detected = bool(stats.get("random_crop_detected", False))
                artifact.qc.face_cutoff_ratio = float(stats.get("face_cutoff_ratio", 0) or 0)
                item["tracking_stats"] = {
                    "focus_switch_count": artifact.qc.focus_switch_count,
                    "focus_ping_pong_detected": artifact.qc.focus_ping_pong_detected,
                    "face_cutoff_ratio": artifact.qc.face_cutoff_ratio,
                    "frames": int(stats.get("frames", 0) or 0),
                }
            except Exception as e:  # noqa: BLE001
                print(f"[render] clip {i}: tracking stats failed ({e}), continuing", flush=True)

            # ── Phase 3 (brief §46): long-pause trim ──
            # Optional: cut long silences inside the clip window. Disabled by
            # default (RENDER_TRIM_REMOVE_LONG_PAUSES=1 to enable) because it
            # changes the timeline — captions are re-aligned below by re-running
            # whisper against the TRIMMED file, not the source window.
            trimmed_path = None
            trimmed_media = False
            if os.getenv("RENDER_TRIM_REMOVE_LONG_PAUSES", "0") == "1":
                try:
                    from audio_master import trim_pauses
                    trimmed_path = trim_pauses(out_path, 0.0, float(c["end_sec"]) - float(c["start_sec"]))
                    if trimmed_path and os.path.exists(trimmed_path):
                        os.replace(trimmed_path, out_path)
                        trimmed_media = True
                        print(f"[render] clip {i}: long pauses trimmed", flush=True)
                        # Invalidate caption cache so whisper re-transcribes.
                        _transcribe_with_whisper.cache_clear() if hasattr(_transcribe_with_whisper, "cache_clear") else None
                except Exception as e:  # noqa: BLE001
                    print(f"[render] clip {i}: pause trim failed ({e}), continuing", flush=True)

            if c["captions"]:
                # Hardening v3 D2 (#22/#25): use TRUSTED canonical word timing
                # directly — skip full Whisper. Cue-only captions stay as the
                # canonical cues (forced-alignment/Whisper is only a fallback
                # when there is no trusted transcript at all).
                # Brief v5 R-06: trusted canonical word timing describes the
                # SOURCE media timeline. After pause trimming the media
                # timeline changed — trusted words MUST NOT be used as-is.
                # Invalidate trusted timing and force alignment/transcription
                # against the FINAL trimmed media.
                trusted_word_timing = (not trimmed_media) and bool(c.get("has_word_timing")) and float(c.get("alignment_confidence") or 0) >= float(os.getenv("RENDER_CAPTION_CONFIDENCE_THRESHOLD", "0.5"))
                if trusted_word_timing:
                    transcript_captions = c["captions"]
                    print(f"[render] clip {i}: using canonical word timing ({c.get('provider', 'unknown')} v{c.get('transcript_version', '')})", flush=True)
                else:
                    try:
                        # Phase 4: transcribe the clip with faster-whisper for
                        # precise word-level timing. Fall back to the miner's
                        # ASR cues if transcription fails.
                        if os.getenv("RENDER_TRIM_REMOVE_LONG_PAUSES", "0") == "1" and os.path.exists(out_path):
                            transcript_captions = _transcribe_with_whisper(
                                out_path, 0.0, float(c["end_sec"]) - float(c["start_sec"]), job_dir,
                                language=c.get("language") or "",
                            )
                        else:
                            transcript_captions = _transcribe_with_whisper(
                                source,
                                float(c["start_sec"]),
                                float(c["end_sec"]),
                                job_dir,
                                language=c.get("language") or "",
                            )
                        print(f"[render] clip {i}: whisper -> {len(transcript_captions)} segments", flush=True)
                    except Exception as e:  # noqa: BLE001
                        print(f"[render] clip {i}: whisper failed ({e}), using ASR cues", flush=True)
                        transcript_captions = c["captions"]

                # Phase 6: speaker diarization — tag each caption line with a
                # speaker so colors differ per speaker. Optional; on failure
                # lines just stay uncolored (white).
                if transcript_captions and DIARIZE_ENABLED:
                    try:
                        turns = _diarize_clip(
                            source,
                            float(c["start_sec"]),
                            float(c["end_sec"]),
                            job_dir,
                        )
                        if turns:
                            transcript_captions = _assign_speakers(transcript_captions, turns)
                            speakers = sorted({t["speaker"] for t in turns})
                            print(f"[render] clip {i}: diarize -> {len(speakers)} speakers {speakers}", flush=True)
                    except Exception as e:  # noqa: BLE001
                        print(f"[render] clip {i}: diarize failed ({e}), captions uncolored", flush=True)

                if transcript_captions:
                    burned = _burn_karaoke_captions(
                        out_path,
                        transcript_captions,
                        float(c["start_sec"]),
                        out_path,
                        job_dir,
                        # Phase 2 (render timelines): pass the explicit
                        # artifact instead of reading module globals.
                        timeline=timeline,
                    )
                    if burned > 0:
                        item["caption_lines"] = burned
                    else:
                        print(f"[render] clip {i}: no captions inside window, skipping burn", flush=True)
                    # Structured QC (brief §23): caption overflow / face collision.
                    artifact.qc.caption_overflow = _CAPTION_OVERFLOW_HITS > 0
                    artifact.qc.caption_face_collision = _CAPTION_COLLISION_HITS > 0

            # Phase 5: prepend the hook intro (frame + dim + hook text + TTS).
            if c["hook"]:
                try:
                    intro_path = _build_hook_intro(out_path, c["hook"], job_dir)
                    if intro_path:
                        final_path = os.path.join(job_dir, f"short_{i:02d}_final.mkv")
                        # Concat intro + content with filter_complex. The plain
                        # concat demuxer + stream copy produces bloated durations
                        # (edge-tts AAC metadata + source timestamps), so we
                        # re-encode both segments onto a clean timeline. Phase 3:
                        # intermediate stays LOSSLESS (ffv1); the single H.264
                        # encode happens in the final pass below.
                        sp.run([
                            "ffmpeg", "-y", "-loglevel", "error",
                            "-i", intro_path, "-i", out_path,
                            "-filter_complex",
                            "[0:v]setpts=PTS-STARTPTS,format=yuv420p[v0];"
                            "[0:a]aresample=44100,asetpts=PTS-STARTPTS[a0];"
                            "[1:v]setpts=PTS-STARTPTS,format=yuv420p[v1];"
                            "[1:a]aresample=44100,asetpts=PTS-STARTPTS[a1];"
                            "[v0][a0][v1][a1]concat=n=2:v=1:a=1[v][a]",
                            "-map", "[v]", "-map", "[a]",
                            "-c:v", "ffv1",
                            "-c:a", "aac", "-b:a", "192k",
                            final_path,
                        ], check=True)
                        os.replace(final_path, out_path)
                        item["hook"] = c["hook"]
                        print(f"[render] clip {i}: hook intro prepended ({c['hook'][:50]}...)", flush=True)
                except Exception as e:  # noqa: BLE001
                    print(f"[render] clip {i}: hook intro failed ({e}), continuing without it", flush=True)

            # ── Phase 3: single final H.264 encode (brief §39-40) ──
            # All stages above (crop, captions, hook) are lossless intermediates.
            # THIS is the one and only lossy encode: CRF 17, preset slow,
            # High profile, yuv420p, faststart, AAC 192k. Phase 4 color
            # correction (brief §49) rides on this same pass.
            try:
                final_h264 = os.path.join(job_dir, f"short_{i:02d}_h264.mp4")
                if preview:
                    # Preview (brief §21): faster preset + higher CRF, still
                    # 540x960 (or the smaller requested output).
                    crf = os.getenv("RENDER_PREVIEW_CRF", "26")
                    preset = os.getenv("RENDER_PREVIEW_PRESET", "veryfast")
                else:
                    crf = os.getenv("RENDER_VIDEO_CRF", "17")
                    preset = os.getenv("RENDER_VIDEO_PRESET", "slow")
                color_filter = None
                try:
                    from visual_effects import build_color_filter
                    color_filter = build_color_filter()
                except Exception:  # noqa: BLE001
                    color_filter = None
                vf_parts = ["format=yuv420p"]
                if color_filter and not preview:
                    vf_parts.insert(0, color_filter)
                # Brand watermark (brief §3.3): channel handle in a corner.
                # Opt-in via RENDER_WATERMARK_TEXT; skipped in preview.
                if not preview:
                    try:
                        from visual_effects import build_watermark_filter
                        wm = build_watermark_filter(output_w, output_h)
                        if wm:
                            vf_parts.append(wm)
                    except Exception:  # noqa: BLE001
                        pass
                cmd = [
                    "ffmpeg", "-y", "-loglevel", "error",
                    "-i", out_path,
                    "-vf", ",".join(vf_parts),
                    "-c:v", "libx264", "-preset", preset, "-crf", crf,
                    "-profile:v", "high", "-level", "4.0",
                    "-pix_fmt", "yuv420p", "-movflags", "+faststart",
                    "-c:a", "copy",
                    final_h264,
                ]
                sp.run(cmd, check=True)
                os.replace(final_h264, out_path)
            except Exception as e:  # noqa: BLE001
                # The styled pass (color + watermark) can crash on some ffmpeg
                # builds when drawtext/fontconfig is broken behind a segfault.
                # Never store the lossless intermediate as the FINAL artifact
                # (YouTube rejects FFV1) — retry a clean H.264 encode without
                # any filter to guarantee a publishable file.
                print(f"[render] clip {i}: styled final encode failed ({e}); retrying clean H.264 encode", flush=True)
                try:
                    cmd_clean = [
                        "ffmpeg", "-y", "-loglevel", "error",
                        "-i", out_path,
                        "-c:v", "libx264", "-preset", preset, "-crf", crf,
                        "-profile:v", "high", "-level", "4.0",
                        "-pix_fmt", "yuv420p", "-movflags", "+faststart",
                        "-c:a", "copy",
                        final_h264,
                    ]
                    sp.run(cmd_clean, check=True)
                    os.replace(final_h264, out_path)
                    print(f"[render] clip {i}: clean H.264 fallback encode succeeded", flush=True)
                except Exception as e2:  # noqa: BLE001
                    print(f"[render] clip {i}: clean retry failed ({e2}), keeping lossless intermediate", flush=True)
                    # Brief v4 F10: a lossless FFV1 intermediate is NOT a
                    # publishable artifact (YouTube rejects FFV1). Mark the
                    # clip failed so no video_url is exposed; the final encode
                    # failure must fail CLOSED.
                    item["status"] = "error"
                    item["error"] = f"final encode failed: styled+clean ({e}; {e2})"
                    artifact.status = "error"
                    artifact.error = item["error"]
                    _RENDER_STATS["final_encode_failed"] = True

            # ── Phase 3 (brief §45): audio mastering chain ──
            # Applied AFTER the final video encode: re-encodes audio only
            # (video stream copied), so no extra video generation. Preview
            # mode skips full mastering (brief §21).
            if not preview:
                try:
                    from audio_master import master_audio
                    mastered = os.path.join(job_dir, f"short_{i:02d}_mastered.mp4")
                    result = master_audio(out_path, mastered)
                    if result and result != out_path:
                        os.replace(mastered, out_path)
                except Exception as e:  # noqa: BLE001
                    print(f"[render] clip {i}: audio mastering failed ({e}), keeping original", flush=True)

            # ── Phase 4 (brief §50): automated quality gate ──
            # Run QC on the FINAL file (after hook + final encode + mastering).
            # Brief v4 F3: this is PER-ARTIFACT QC — it must NOT change the
            # JOB state (the job enters quality_check ONCE after all clips are
            # rendered, below). When QC_BLOCK_UPLOAD=1 and the video fails,
            # mark the clip failed so the publisher refuses to upload.
            try:
                from quality_gate import quality_gate
                qc = quality_gate(out_path)
                _apply_qc_to_artifact(artifact, item, qc, mode)
                artifact.qc.upscale_factor = _estimate_upscale(source, output_w, output_h)
            except Exception as e:  # noqa: BLE001
                # Brief v5 4.5: FINAL mode must fail closed when QC is
                # unavailable — a clip without QC cannot be published. Preview
                # mode may continue with a warning (publishable=false).
                print(f"[render] clip {i}: quality gate error ({e}), continuing", flush=True)
                _apply_qc_to_artifact(artifact, item, None, mode)

            if item["status"] == "ok":
                item["clip_path"] = os.path.abspath(out_path)
                item["clip_url"] = f"{job_id}/{os.path.basename(out_path)}"
                item["video_url"] = f"{job_id}/{os.path.basename(out_path)}"
                artifact.status = "ok"
                artifact.video_url = f"{job_id}/{os.path.basename(out_path)}"
                artifact.actual_layout = clip_layout
                artifact.fallback_reason = fallback_reason

            # Phase 7: auto thumbnail (best face frame + hook text).
            try:
                thumb_path = _build_thumbnail(
                    out_path,
                    c["hook"],
                    job_dir,
                )
                if thumb_path:
                    item["thumbnail_url"] = f"{job_id}/thumbnail.jpg"
                    artifact.thumbnail_url = f"{job_id}/thumbnail.jpg"
                    print(f"[render] clip {i}: thumbnail generated", flush=True)
            except Exception as e:  # noqa: BLE001
                print(f"[render] clip {i}: thumbnail failed ({e})", flush=True)
        except Exception as e:  # noqa: BLE001
            print(f"[render] clip {i} failed: {e}", flush=True)
            item["error"] = str(e)
            artifact.status = "error"
            artifact.error = str(e)
        rendered.append(item)
        artifacts.append(artifact)

    print(f"[render] job {job_id} finished in {time.time() - start:.1f}s", flush=True)
    # Brief v4 F3: the job enters quality_check ONCE, after ALL clips have
    # rendered — never inside the per-clip loop. Observers always see the
    # true current stage (rendering while clips render, quality_check only
    # after the last clip).
    # Brief v6 4.4 Option A (job-level phases) + R01: checked transition.
    require_transition(job_id, "rendering", "quality_check", mode=mode, episode_id=episode_id)
    print(f"[render] job {job_id}: quality_check (all clips rendered)", flush=True)
    # Brief v7 Q01 (real two-pass): the quality_check phase must DO work, not
    # just hop state. Aggregate QC verifies the full artifact set — every ok
    # artifact must have reached QC, produced a video_url, and passed the
    # gate; any missing/regressed ok artifact is demoted to error so the
    # final publishability decision is made on real QC results.
    for _a in artifacts:
        if _a.status == "ok":
            _qc_st = (getattr(_a.qc, "status", "") or "").strip().lower()
            if not _a.video_url or _qc_st in ("", "unavailable", "fail", "failed"):
                            _a.status = "error"
                            if not _a.error:
                                _a.error = f"quality_check aggregate gate rejected qc_status={_qc_st!r}"
    # Recompute statuses after the aggregate QC pass.
    # Brief v10 C04: use the canonical helper (0/N -> failed, not partial).
    final_status = terminal_status_from_artifacts(artifacts)
    src_info = None
    try:
        from visual_effects import probe_source_resolution
        src_probe = probe_source_resolution(source)
        if src_probe:
            src_info = {"width": src_probe[0], "height": src_probe[1]}
    except Exception:  # noqa: BLE001
        pass
    # Brief v8 R02/C04: build the final response from ONE canonical artifact
    # list derived from the RenderArtifact models (already QC-mutated). The
    # legacy `rendered` key is an alias/serialization of the SAME list — never
    # constructed from a second data structure aggregate QC cannot touch.
    resp = _build_render_response(
        job_id=job_id,
        source_video=source,
        mode=mode,
        canonical_results=_canonicalize(artifacts, mode),
        final_status=final_status,
        src_info=src_info or {},
    )
    return RenderOutcome(resp, final_status)


def _apply_qc_to_artifact(artifact, item, qc, mode: str) -> bool:
    """Apply one quality_gate result to an artifact (brief v11 C12).

    Returns True when the artifact stays publishable-eligible (ok + passed).
    The canonical qc.status vocabulary is 'passed' | 'failed' | 'unavailable'
    (QCDetail contract). quality_gate returns 'pass'/'fail' — the mapping to
    the canonical value happens HERE, so the aggregate-QC gate and
    _canonicalize() always see the same vocabulary.

    Before v11 this mapping was missing: a passing gate ('pass', score 100)
    left artifact.qc.status at its default 'unavailable', the aggregate gate
    demoted the artifact to error (fail-closed on unavailable), and every
    real render died with an empty error.
    """
    if qc is None:
        artifact.qc.status = "unavailable"
        if mode == "final":
            item["status"] = "error"
            item["error"] = "quality gate unavailable"
            artifact.status = "error"
            artifact.error = item["error"]
            print(f"[render] clip {artifact.clip_id}: quality gate unavailable (final)", flush=True)
            return False
        item.setdefault("quality", {}).update({
            "status": "unavailable",
            "warnings": ["quality gate unavailable"],
        })
        return True

    item["quality"] = {
        "status": qc["status"],
        "score": qc["quality_score"],
        "warnings": qc["warnings"][:6],
    }
    # Structured QC detail (brief §23).
    artifact.qc.score = float(qc["quality_score"])
    try:
        artifact.qc.output_width = int(qc.get("checks", {}).get("resolution", "1080x1920").split("x")[0] or 1080)
    except Exception:  # noqa: BLE001
        pass
    try:
        artifact.qc.output_height = int(qc["checks"]["resolution"].split("x")[1])
    except Exception:  # noqa: BLE001
        pass
    artifact.qc.codec = qc.get("checks", {}).get("codec", "h264")
    artifact.qc.pixel_format = qc.get("checks", {}).get("pix_fmt", "yuv420p")
    artifact.qc.audio_lufs = qc.get("checks", {}).get("audio_lufs")
    artifact.qc.audio_true_peak = qc.get("checks", {}).get("audio_true_peak")
    artifact.qc.audio_sync_ms = qc.get("checks", {}).get("audio_sync_ms")
    artifact.qc.black_frame_ratio = qc.get("checks", {}).get("black_frame_ratio", 0) or 0
    artifact.qc.frozen_frame_ratio = qc.get("checks", {}).get("frozen_frame_ratio", 0) or 0
    artifact.qc.upscale_factor = artifact.qc.upscale_factor  # set by caller
    artifact.qc.warnings = qc["warnings"][:6]

    if qc["status"] != "pass":
        item["status"] = "error"
        item["error"] = f"quality gate failed: {qc['warnings'][:3]}"
        artifact.status = "error"
        artifact.error = item["error"]
        artifact.qc.status = "failed"
        print(f"[render] clip {artifact.clip_id}: QC FAILED ({qc['warnings'][:3]})", flush=True)
        return False

    artifact.qc.status = "passed"
    return True


def _canonicalize(artifact_models, mode: str) -> "list[RenderArtifactResult]":
    """Convert RenderArtifact models (already QC-mutated) into the ONE
    canonical RenderArtifactResult list used by every representation."""
    out = []
    for a in artifact_models:
        out.append(RenderArtifactResult(
            clip_id=str(a.clip_id),
            status=a.status,
            video_url=a.video_url,
            thumbnail_url=a.thumbnail_url,
            publishable=(
                a.status == "ok"
                and bool(a.video_url)
                and (getattr(a.qc, "status", "unavailable") or "unavailable") == "passed"
                and mode == "final"
            ),
            qc_status=getattr(a.qc, "status", "unavailable") or "unavailable",
            error=({"message": a.error} if getattr(a, "error", None) else None),
        ))
    return out


def terminal_status_from_artifacts(results) -> str:
    """Brief v10 C04 — canonical terminal status from a list of artifact
    results (section 5.1). Single source of truth for both the render worker's
    final_status and the response builder.

    Rules:
      N == 0                      -> "failed"
      N > 0 and ok_count == N     -> "completed"
      N > 0 and ok_count == 0     -> "failed"
      N > 0 and 0 < ok_count < N  -> "partial_failure"
    """
    artifacts = list(results or [])
    if not artifacts:
        return "failed"
    ok = sum(1 for a in artifacts if a.status == "ok")
    if ok == len(artifacts):
        return "completed"
    if ok == 0:
        return "failed"
    return "partial_failure"


def _build_render_response(*, job_id: str, source_video: str, mode: str,
                           canonical_results, final_status: str = "completed",
                           src_info: dict = None) -> RenderResponse:
    """Build a RenderResponse from ONE canonical list of RenderArtifactResult.

    Aggregate QC has already demoted any invalid ok artifact. Deriving
    `rendered` and `artifacts` from this ONE list guarantees they can never
    diverge, and final_status/publishability are consistent. Brief v10 C4:
    effective terminal status comes from the shared
    terminal_status_from_artifacts helper (V10-R01 0/N -> failed).
    """
    canonical = list(canonical_results)
    # Recompute final status from the SAME canonical list (paranoia guard)
    # using the canonical helper — never duplicate the mapping logic.
    effective_status = terminal_status_from_artifacts(canonical)
    return RenderResponse(
        job_id=job_id,
        source_video=source_video,
        status=effective_status or final_status,
        rendered=canonical,
        artifacts=canonical,
        mode=mode,
        source=src_info or {},
    )


# Serial render queue: only ONE render job runs at a time. Concurrent
# requests (e.g. batch render-all + a manual re-render) would otherwise start
# parallel downloads/encodes and overload the CPU/disk — the exact failure we
# saw. The lock serializes them; waiters simply block until their turn.
_render_lock = threading.Lock()
_render_busy = False
# Async job registry: job_id -> {state, response, error}
_async_jobs: Dict[str, Dict] = {}
_async_jobs_lock = threading.Lock()

# Brief v4 F18: bounded queue + ONE persistent worker (no thread-per-job).
# Submission enqueues a job_id; the worker dequeues and processes serially.
import queue as _queue_module
RENDER_QUEUE_MAX = int(os.getenv("RENDER_QUEUE_MAX", "100"))
_render_queue: "_queue_module.Queue[str]" = _queue_module.Queue(maxsize=RENDER_QUEUE_MAX)
_render_queue_worker_started = False
_render_queue_worker_lock = threading.Lock()
# Brief v5 9.2: keep the ACTUAL thread reference + heartbeat + last exception
# so health reports thread.is_alive() (not a boolean flag that stays True
# after a crash).
_render_queue_worker_thread: "threading.Thread | None" = None
_render_worker_heartbeat_at: str = ""
_render_worker_last_exception: str = ""

# Brief v4 F19: process boot identity for orphan ownership. A job reserved by
# this process carries this boot_id; orphan detection only marks rows whose
# boot_id != current (or rows from before the column existed AND older than
# the age threshold) — never a freshly reserved job from THIS process.
PROCESS_BOOT_ID = uuid.uuid4().hex[:12]
ORPHAN_AGE_THRESHOLD_SEC = float(os.getenv("RENDER_ORPHAN_AGE_THRESHOLD", "300"))


@app.get("/api/render/status")
def render_status():
    """Report whether a render job is currently running (queue status)."""
    return {"busy": _render_busy}


def _heartbeat_age_sec() -> float:
    """Seconds since the worker last heartbeat. -1 if never heartbeated."""
    if not _render_worker_heartbeat_at:
        return -1.0
    try:
        hb = datetime.datetime.fromisoformat(_render_worker_heartbeat_at)
        return max(0.0, (datetime.datetime.utcnow() - hb).total_seconds())
    except Exception:  # noqa: BLE001
        return -1.0


def _oldest_queued_age_sec() -> float:
    """Brief v6 10.3 — age of the oldest job still in an active stage, from
    the SQLite created_at. Returns 0 when nothing is active."""
    try:
        with _db_lock, _db_conn() as conn:
            row = conn.execute(
                "SELECT created_at FROM render_jobs "
                "WHERE status IN ('queued','downloading','analysing','rendering','quality_check') "
                "ORDER BY created_at ASC LIMIT 1"
            ).fetchone()
        if not row or not row[0]:
            return 0.0
        created = datetime.datetime.fromisoformat(row[0])
        return max(0.0, (datetime.datetime.utcnow() - created).total_seconds())
    except Exception:  # noqa: BLE001
        return 0.0


@app.get("/api/render/health")
def render_health():
    """Operational readiness without downloading a video or loading models
    (Phase 1 §5.6)."""
    import shutil
    import sqlite3
    # 1. SQLite read/write status (short transaction, no data mutation).
    db_ok = False
    db_error = None
    try:
        with _db_lock, _db_conn() as conn:
            conn.execute("SELECT COUNT(*) FROM render_jobs").fetchone()
            # Write probe: insert into a temp table in a transaction we roll back.
            conn.execute("CREATE TEMP TABLE IF NOT EXISTS _health_probe (v INTEGER)")
            conn.execute("INSERT INTO _health_probe (v) VALUES (1)")
            conn.execute("DELETE FROM _health_probe")
            conn.commit()
            db_ok = True
    except Exception as e:  # noqa: BLE001
        db_error = f"{type(e).__name__}: {e}"
    # 2. Queue depth + active job.
    with _async_jobs_lock:
        active_jobs = [jid for jid, j in _async_jobs.items() if is_active(j.get("state", ""))]
        active_job_id = active_jobs[0] if active_jobs else None
        queue_depth = len(active_jobs)
    # 3. FFmpeg / FFprobe availability.
    ffmpeg = shutil.which("ffmpeg") is not None
    ffprobe = shutil.which("ffprobe") is not None
    # 4. Output dir writability + free disk.
    out_ok = False
    out_error = None
    free_bytes = None
    try:
        RENDER_ROOT.mkdir(parents=True, exist_ok=True)
        probe = RENDER_ROOT / f".health_{os.getpid()}"
        probe.write_text("ok", encoding="utf-8")
        probe.unlink()
        out_ok = True
        import ctypes
        free_bytes = ctypes.c_ulonglong(0)
        ok = ctypes.windll.kernel32.GetDiskFreeSpaceExW(
            str(RENDER_ROOT.resolve()),
            None, None, ctypes.byref(free_bytes),
        )
        free_bytes = int(free_bytes.value) if ok else None
    except Exception as e:  # noqa: BLE001
        out_error = f"{type(e).__name__}: {e}"
    # 5. Contract version + build id.
    from render_contract import CONTRACT_VERSION
    build = os.getenv("RENDER_BUILD_ID", "0.1.0")
    return {
        "status": "ok" if (db_ok and out_ok) else "degraded",
        "service": "shorts-render-service",
        "build": build,
        "db": {"ok": db_ok, "error": db_error,
               "last_error": _last_db_error, "last_error_at": _last_db_error_at,
               "last_error_stage": _last_db_error_stage},
        "queue": {"depth": _render_queue.qsize(), "active_job_id": active_job_id,
                  "max": RENDER_QUEUE_MAX,
                  # Brief v5 9.2: real thread reference + is_alive(); the
                  # boolean flag alone stays True after a crash.
                  "worker_started": _render_queue_worker_started,
                  "worker_alive": bool(_render_queue_worker_thread and _render_queue_worker_thread.is_alive()),
                  "worker_heartbeat": _render_worker_heartbeat_at,
                  "worker_last_exception": _render_worker_last_exception,
                  # Brief v6 10.3: oldest queued job age.
                  "oldest_queued_age_sec": _oldest_queued_age_sec(),
                  "process_boot_id": PROCESS_BOOT_ID},
        "ffmpeg": {"available": ffmpeg, "ffprobe": ffprobe},
        "output": {"writable": out_ok, "error": out_error, "free_bytes": free_bytes},
        "contract_version": CONTRACT_VERSION,
        "last_persist_error": _last_persist_error,
        "last_persist_error_at": _last_persist_error_at,
        "rendering_available_when_persistence_degraded": not db_ok,
    }


def _reconcile_startup_orphans() -> int:
    """Brief v5 4.6: at process startup, mark stale ACTIVE jobs whose
    process_boot_id differs from this process (or pre-ownership rows older
    than the threshold) as 'orphaned' in a controlled transaction.

    Runs BEFORE any client GET can repair state. Returns count orphaned.
    Fresh rows from THIS boot (including the reservation-to-memory window)
    are never touched — boot id + minimum age threshold protect them.
    """
    orphaned_count = 0
    try:
        with _db_lock, _db_conn() as conn:
            rows = conn.execute(
                "SELECT job_id, status, process_boot_id, created_at FROM render_jobs "
                "WHERE status IN ('queued','downloading','analysing','rendering','quality_check')"
            ).fetchall()
        for job_id, status, boot_id, created_at in rows:
            is_foreign = boot_id is not None and boot_id != PROCESS_BOOT_ID
            is_ancient = (boot_id is None) and _job_older_than(created_at or "", ORPHAN_AGE_THRESHOLD_SEC)
            if is_foreign or is_ancient:
                # After a restart the in-memory registry is EMPTY for these
                # rows — transition_job (which verifies memory == expected)
                # cannot apply. This is the one sanctioned direct write: a
                # controlled, verified DB-only transition at startup.
                try:
                    with _db_lock, _db_conn() as conn:
                        now = datetime.datetime.utcnow().isoformat()
                        cur = conn.execute(
                            "UPDATE render_jobs SET status='orphaned', error=?, finished_at=?, updated_at=? "
                            "WHERE job_id=? AND status=?",
                            ("job orphaned by render service restart", now, now, job_id, status),
                        )
                        conn.commit()
                        if cur.rowcount > 0:
                            orphaned_count += 1
                except Exception as e:  # noqa: BLE001
                    _record_db_error("startup_reconcile_row", e)
    except Exception as e:  # noqa: BLE001
        _record_db_error("startup_reconcile", e)
    return orphaned_count


@app.get("/api/render/status/{job_id}", response_model=RenderJobStatusResponse)
def render_job_status(job_id: str):
    """Return the canonical state of a render job (brief v9 A1: durable-first).

    SQLite is canonical. Memory provides runtime diagnostics only and never
    overrides durable state. GET is side-effect free; orphan detection is
    owned by startup reconciliation.
    """
    # 1. Load canonical durable state from SQLite.
    durable = _load_durable_snapshot(job_id)
    if not durable:
        raise HTTPException(status_code=404, detail="job not found")

    # 2. Merge in-memory diagnostics if present.
    with _async_jobs_lock:
        mem = _async_jobs.get(job_id)

    state = durable.state
    error = durable.error
    response_model = None
    persistence_degraded = False
    runtime_error = None
    worker_attached = False

    # Orphan detection: only for foreign-boot active rows.
    if is_active(state):
        boot_id = durable.process_boot_id
        created_at = durable.created_at or ""
        is_foreign = boot_id is not None and boot_id != PROCESS_BOOT_ID
        is_ancient = (boot_id is None) and _job_older_than(created_at, ORPHAN_AGE_THRESHOLD_SEC)
        if is_foreign or is_ancient:
            # Brief v9: orphaned via startup reconciliation, not GET mutation.
            # If we got here, startup already ran; just report orphaned.
            state = "orphaned"
            error = error or "job orphaned by render service restart"
        elif mem:
            # Current-process active job: worker may be attached.
            worker_attached = True

    # Memory diagnostics overlay (never override canonical state).
    if mem:
        runtime_error = mem.get("runtime_error")
        persistence_degraded = bool(mem.get("persistence_degraded", False))
        # If memory state differs from durable, mark as degraded.
        if mem.get("state") != state:
            persistence_degraded = True
        # If memory has a response object for terminal states, use it.
        if state in ("completed", "partial_failure", "failed") and isinstance(mem.get("response"), RenderResponse):
            response_model = mem["response"]

    # Deserialize stored response if present.
    if not response_model and durable.response:
        try:
            import json as _json
            raw = durable.response
            if isinstance(raw, str):
                raw = _json.loads(raw)
            if isinstance(raw, dict):
                response_model = RenderResponse(**raw)
        except Exception:  # noqa: BLE001
            response_model = None

    if state == "orphaned" and not error:
        error = "job orphaned by render service restart"

    return RenderJobStatusResponse(
        job_id=job_id,
        request_id=durable.request_id,
        state=state,
        mode=durable.mode,
        attempt=durable.attempt,
        parent_job_id=durable.parent_job_id,
        error=error,
        response=response_model,
        persistence_degraded=persistence_degraded,
        runtime_error=runtime_error,
        worker_attached=worker_attached,
    )


def parse_render_request(request):
    """Parse a raw dict into the correct request model (V1 or V2).

    Phase-2 correctness:
    - V2 bodies (contract_version == '2.0') parse as RenderRequestV2.
    - A PRESENT but unsupported contract_version (e.g. '2.1') is rejected
      explicitly instead of falling through to the V1 parser.
    - Legacy v1 bodies (no contract_version) parse as RenderRequest.
    Used by sync, async and retry so all three paths behave identically.
    """
    if not isinstance(request, dict):
        return request
    if "contract_version" in request:
        if request["contract_version"] != CONTRACT_VERSION:
            raise HTTPException(
                status_code=400,
                detail=f"unsupported contract_version {request['contract_version']!r}; "
                       f"supported: {CONTRACT_VERSION!r}",
            )
        return RenderRequestV2(**request)
    return RenderRequest(**request)


def get_existing_request_result(request_id: str):
    """Brief v10 C07 — durable-first idempotent-retrieval helper for sync and
    async (V10-R04).

    SQLite chooses the canonical job_id — never memory. Memory only overlays
    live diagnostics for THAT chosen job. Canonical attempt policy: prefer an
    ACTIVE attempt; otherwise newest terminal attempt by attempt DESC then
    created_at DESC.
    """
    if not request_id:
        return None
    # 1. Choose identity from DURABLE state (never iterate memory to pick one).
    job_id = _find_job_by_request(request_id)
    if job_id is None:
        return None
    stored = _load_job(job_id)
    # 2. Memory overlays diagnostics for the chosen job only.
    mem = None
    with _async_jobs_lock:
        if job_id in _async_jobs:
            mem = _async_jobs[job_id]
    state = canonical_status((stored or {}).get("status") or (mem or {}).get("state") or "queued")
    return {
        "job_id": job_id,
        "request_id": request_id,
        "state": state,
        "mode": (stored or {}).get("mode") or (mem or {}).get("mode", "final"),
        "attempt": (stored or {}).get("attempt") or (mem or {}).get("attempt", 1),
        "parent_job_id": (stored or {}).get("parent_job_id") or (mem or {}).get("parent_job_id"),
        "response": (stored or {}).get("response"),
        "idempotent_hit": True,
    }


@app.post("/api/render/async", response_model=RenderSubmissionResponse)
def render_async(request: Dict[str, Any]):
    """Queue a render job (v1 or v2 contract) and return immediately.
    The request is parsed MANUALLY (not via Union) so that a v2 body with
    contract_version="2.0" is never mis-parsed as v1. FastAPI's Union tries
    v1 first, and v1 ignores unknown fields — which silently dropped
    mode=preview (brief §21) and made previews render as finals.

    The job runs in a background thread (still serialized by the process-wide
    lock). Clients poll GET /api/render/status/{job_id} for completion. This
    avoids the ~5min client timeout that killed long batch renders.

    Idempotency (brief §20): when the request carries a request_id that already
    exists in a non-failed job, the EXISTING job id is returned instead of
    starting a duplicate render.
    """
    # Parse manually via the shared parse_render_request(): v2 -> V2 model,
    # unsupported versions fail explicitly, legacy v1 -> V1 model.
    request = parse_render_request(request)
    request_id = getattr(request, "request_id", "") or ""
    force_rerender = bool(getattr(request, "force_rerender", False))
    if request_id and not force_rerender:
        # Brief v8 A3/R07: single idempotent lookup reports the ACTUAL
        # persisted state — never a hardcoded 'queued'.
        existing = get_existing_request_result(request_id)
        if existing is not None and existing["state"] not in ("failed", "cancelled"):
            print(f"[render] idempotent hit: {request_id} -> {existing['job_id']} state={existing['state']}", flush=True)
            return RenderSubmissionResponse(
                job_id=existing["job_id"],
                request_id=request_id,
                state=existing["state"],
                idempotent_hit=True,
                attempt=existing["attempt"],
                parent_job_id=existing["parent_job_id"],
            )

    mode = getattr(request, "mode", "final") or "final"
    episode_id = getattr(request, "episode_id", "") or ""
    request_json = request.model_dump_json() if hasattr(request, "model_dump_json") else ""
    new_job_id = uuid.uuid4().hex[:10]

    if request_id:
        # C3: every attempt-producing production submission uses the one
        # allocator. Normal terminal resubmit is reason=resubmit; force is
        # reason=force. The allocator itself enforces one active attempt.
        source_job_id = None
        if force_rerender:
            latest = get_existing_request_result(request_id)
            source_job_id = latest["job_id"] if latest else None
        reservation = reserve_attempt(
            source_job_id=source_job_id,
            request_id=request_id,
            request_json=request_json,
            mode=mode,
            episode_id=episode_id,
            reason="force" if force_rerender else "resubmit",
        )
    else:
        # Legacy V1 initial submission has no request lineage and remains a
        # one-off attempt=1 durable insert. It cannot create retries.
        reserved_id = _reserve_job(
            request_id, new_job_id, mode=mode, episode_id=episode_id,
            request_json=request_json, force=False,
        )
        reservation = AttemptReservation(
            job_id=reserved_id, attempt=1, parent_job_id=None,
            created=(reserved_id == new_job_id), existing_winner_job_id=(None if reserved_id == new_job_id else reserved_id),
            reason="resubmit",
        )

    job_id = reservation.job_id
    if not reservation.created:
        ex = get_existing_request_result(request_id) if request_id else None
        stored = _load_job(job_id)
        return RenderSubmissionResponse(
            job_id=job_id,
            request_id=request_id,
            state=(stored or {}).get("status") or (ex or {}).get("state") or "queued",
            idempotent_hit=True,
            attempt=(stored or {}).get("attempt") or reservation.attempt,
            parent_job_id=(stored or {}).get("parent_job_id") or reservation.parent_job_id,
        )

    with _async_jobs_lock:
        _async_jobs[job_id] = {
            "state": "queued", "response": None, "error": None,
            "request_id": request_id, "mode": mode, "episode_id": episode_id,
            "attempt": reservation.attempt, "parent_job_id": reservation.parent_job_id,
        }

    try:
        _enqueue_job(job_id)
    except QueueAdmissionError as exc:
        raise HTTPException(status_code=503, detail="Render queue is full") from exc
    return RenderSubmissionResponse(
        job_id=job_id, request_id=request_id, state="queued",
        attempt=reservation.attempt, parent_job_id=reservation.parent_job_id,
    )


def _enqueue_job(job_id: str) -> None:
    """Push a reserved job onto the bounded queue and ensure the single
    persistent worker is running (brief v4 F18). Submission enqueues IDs, it
    never creates daemon threads. Brief v5 R-04: queue admission failures must
    be compensated by a transition to failed — never leave a stranded queued
    row. Returns None on success, raises QueueAdmissionError on full."""
    global _render_queue_worker_started
    try:
        _render_queue.put(job_id, timeout=5.0)
    except _queue_module.Full:
        # Brief v5 4.4 / v7 R08: no durable queued job may exist without a
        # queue item OR an explicit COMMITTED terminal failure describing
        # queue admission. If the queued->failed compensation does not
        # commit, the stranded queued row is a correctness bug.
        try:
            ok = transition_job(job_id, "queued", "failed",
                                error=f"queue_admission: render queue is full ({RENDER_QUEUE_MAX} jobs)",
                                error_stage="queue_admission")
        except Exception as e:  # noqa: BLE001
            _record_db_error("queue_full_compensation", e)
            raise QueueAdmissionError(f"render queue is full ({RENDER_QUEUE_MAX} jobs); try later") from e
        if not ok:
            _record_db_error("queue_full_compensation",
                             RuntimeError(f"compensation queued->failed lost for {job_id}"))
            # Drop the stranded in-memory phantom so it cannot be polled.
            with _async_jobs_lock:
                _async_jobs.pop(job_id, None)
        raise QueueAdmissionError(f"render queue is full ({RENDER_QUEUE_MAX} jobs); try later")
    # Brief v6 4.5/R05: ensure the worker is ALIVE (restart if crashed).
    ensure_worker_running()


def _next_state(state: str) -> str:
    """The canonical successor in the active chain (brief v5 4.5)."""
    return {
        "queued": "downloading",
        "downloading": "analysing",
        "analysing": "rendering",
        "rendering": "quality_check",
    }.get(state, state)


class QueueAdmissionError(RuntimeError):
    """Raised when the bounded render queue rejected a submission (brief v5 4.4)."""


def _register_job_memory(job_id: str, request_id: str, mode: str, episode_id: str = "") -> None:
    """Register a reserved job in the shared in-memory registry (brief v5 4.1).

    Only called for a NEWLY reserved job — never for an idempotency hit.
    """
    with _async_jobs_lock:
        _async_jobs[job_id] = {"state": "queued", "response": None, "error": None,
                               "request_id": request_id, "mode": mode, "episode_id": episode_id}


def _persist_terminal_via_transition(job_id: str, status: str, *, mode: str = "final",
                                     episode_id: str = "", response: str = "",
                                     error: str = "") -> bool:
    """Persist a terminal status through the canonical state machine (brief v5 4.2).

    Only terminal targets are accepted here; the worker should transition
    rendering -> quality_check -> completed/partial_failure via transition_job.
    Returns True iff the terminal transition was applied AND committed.
    """
    status = canonical_status(status)
    if status not in TERMINAL_JOB_STATES:
        raise ValueError(f"_persist_terminal_via_transition requires terminal status, got {status}")
    # Find the current state to know the expected source for the CAS.
    current = _load_job(job_id)
    if current is None:
        return False
    src = canonical_status(current["status"])
    # quality_check -> completed/partial_failure/failed is the legal terminal hop.
    if src == "quality_check":
        ok = transition_job(job_id, src, status, mode=mode, episode_id=episode_id, error=error,
                            response=response)
        return ok
    # Brief v5 4.5: a renderer that returns an outcome without having advanced
    # through every stage (e.g. mocked E2E, or early-exit paths) must still
    # reach a LEGAL terminal state. Auto-advance along the canonical path
    # downloading -> analysing -> rendering -> quality_check -> terminal, then
    # apply the final hop. Every hop is a validated CAS; a lost hop aborts.
    if status == "failed":
        return transition_job(job_id, src, "failed", mode=mode, episode_id=episode_id, error=error)
    # Brief v7 R07: never auto-advance FROM queued. A queued job that reports
    # a render outcome has not actually worked (the worker must first win the
    # queued->downloading CAS). Auto-advance is only allowed from an
    # in-flight stage the worker demonstrably reached.
    if src in ("queued", "cancelled"):
        return False
    chain = ["downloading", "analysing", "rendering", "quality_check"]
    if src in chain:
        try:
            idx = chain.index(src)
        except ValueError:
            return False
        current_state = src
        for nxt in chain[idx + 1:]:
            # expected = the CURRENT state; target = the next canonical stage.
            if not transition_job(job_id, current_state, nxt, mode=mode, episode_id=episode_id):
                return False
            current_state = nxt
        # Now at quality_check; apply the terminal hop.
        return transition_job(job_id, "quality_check", status, mode=mode, episode_id=episode_id,
                              error=error, response=response)
    return False


def ensure_worker_running() -> None:
    """Brief v6 4.5/R05 — start the queue worker if it is not ALIVE.

    The boolean flag alone is insufficient: a crashed thread leaves the flag
    True. Restart exactly when the thread reference is missing or dead.
    Guarded by _render_queue_worker_lock so two workers never start.
    """
    global _render_queue_worker_started, _render_queue_worker_thread
    with _render_queue_worker_lock:
        if (
            not _render_queue_worker_started
            or _render_queue_worker_thread is None
            or not _render_queue_worker_thread.is_alive()
        ):
            _render_queue_worker_started = True
            t = threading.Thread(target=_queue_worker_loop, daemon=True)
            t.start()
            _render_queue_worker_thread = t
            print(f"[render] queue worker started (boot={PROCESS_BOOT_ID}, max={RENDER_QUEUE_MAX})", flush=True)


def _queue_worker_loop() -> None:
    """Persistent FIFO worker: dequeues one job_id at a time and processes it
    under the serial render lock. One thread for the whole process.

    Brief v6 R05: the OUTERMOST handler resets the started flag before exit so
    a crashed worker (even on queue.get()) is restarted on the next enqueue.
    """
    try:
        while True:
            job_id = _render_queue.get()
            try:
                _process_queued_job(job_id)
            except Exception as e:  # noqa: BLE001
                global _render_worker_last_exception
                _render_worker_last_exception = f"{type(e).__name__}: {e}"
                print(f"[render] queue worker error on {job_id}: {e}", flush=True)
                try:
                    # Brief v9 A2: mirror durable state, never fabricate.
                    # Brief v5 R-02: canonical transition to failed — never a raw
                    # _persist_job bypassing the state machine.
                    cur = _load_job(job_id)
                    src = canonical_status(cur["status"]) if cur else "queued"
                    won_failed = transition_job(job_id, src, "failed", error=str(e), error_stage="worker")
                    if won_failed:
                        # Transition won: update memory to reflect durable failed.
                        mirror_durable_after_failure(job_id, f"worker error: {e}")
                    else:
                        # Transition lost: mirror the winning durable state.
                        mirror_durable_after_failure(job_id, f"worker error (transition lost): {e}")
                except Exception:  # noqa: BLE001
                    # Even transition failure must mirror durable state.
                    mirror_durable_after_failure(job_id, f"worker error + transition failure: {e}")
            finally:
                global _render_worker_heartbeat_at
                _render_worker_heartbeat_at = datetime.datetime.utcnow().isoformat()
                _render_queue.task_done()
    finally:
        # Brief v6 R05: reset the flag so the NEXT enqueue restarts the worker.
        global _render_queue_worker_started
        with _render_queue_worker_lock:
            _render_queue_worker_started = False
            _render_queue_worker_thread = None


def _process_queued_job(job_id: str) -> None:
    """Process one reserved job: load its request, advance state, render,
    persist the terminal outcome exactly once (brief v4 F1/F18)."""
    global _render_busy
    raw = _load_job_request(job_id) or {}
    if not raw:
        raise RuntimeError(f"job {job_id}: request payload missing")
    request = parse_render_request(raw)
    with _async_jobs_lock:
        mode = _async_jobs.get(job_id, {}).get("mode") or getattr(request, "mode", "final") or "final"
        episode_id = _async_jobs.get(job_id, {}).get("episode_id") or getattr(request, "episode_id", "") or ""
    _render_lock.acquire()
    _render_busy = True
    outcome = None
    won = False
    try:
        # P0.1: the worker wins ONLY through queued -> downloading CAS.
        # If a cancel already won (queued -> cancelled), this fails and the
        # worker exits without rendering. No check-then-act.
        won = transition_job(job_id, "queued", "downloading", mode=mode, episode_id=episode_id)
        if won:
            outcome = _render(request, job_id)
    finally:
        _render_busy = False
        _render_lock.release()
    if not won:
        return
    # Brief v5 4.2/4.5: terminal status through the canonical transition
    # (rendering -> quality_check -> completed/partial_failure), committed
    # BEFORE memory update. If SQLite fails, memory must NOT claim terminal.
    try:
        terminal_ok = _persist_terminal_via_transition(
            job_id,
            outcome.final_status,
            mode=mode,
            episode_id=episode_id,
            response=outcome.response.model_dump_json() if hasattr(outcome.response, "model_dump_json") else "",
        )
    except PersistenceError as e:
        # Brief v9 A2: mirror durable state, never fabricate.
        _record_db_error("worker_terminal_persist", e)
        mirror_durable_after_failure(job_id, f"persist: {e}")
        return
    if not terminal_ok:
        # Brief v9 A2: terminal transition lost — mirror the DURABLE winner.
        mirror_durable_after_failure(job_id, "terminal transition lost")
        return
    # Success: update in place so request_id/mode/episode/attempt/parent survive
    # (STATE-03/R09).
    with _async_jobs_lock:
        job = _async_jobs.get(job_id)
        if job is not None:
            job.update({
                "state": outcome.final_status,
                "response": outcome.response,
                "error": None,
                "runtime_error": None,
                "persistence_degraded": False,
            })
        else:
            _async_jobs[job_id] = {"state": outcome.final_status,
                                   "response": outcome.response, "error": None}


@app.post("/api/render/jobs/{job_id}/cancel")
def render_job_cancel(job_id: str):
    """Cancel a QUEUED job (Phase 1 §5.4, hardening sprint P0.1).

    - queued (waiting for the render lock): cancelled IF the queued ->
      cancelled CAS wins. A worker that already won queued -> downloading
      makes this cancel return 409 (first transition wins).
    - rendering (already running): NOT supported in Phase 1 — no active
      FFmpeg cancellation. Return 409 conflict so the caller knows the job
      will finish.
    - terminal: return current state unchanged.
    """
    won = transition_job(job_id, "queued", "cancelled", error="cancelled by user")
    if won:
        return {"job_id": job_id, "state": "cancelled"}

    # Brief v10 C07 (V10-R06): the cancel conflict response reads DURABLE
    # state first; memory is diagnostic only. A stale memory snapshot must
    # never report the wrong current state (e.g. after process restart).
    stored = _load_job(job_id)
    if not stored:
        raise HTTPException(status_code=404, detail="job not found")
    state = canonical_status(stored["status"])
    if is_active(state):
        return JSONResponse(
            status_code=409,
            content={"job_id": job_id, "state": state,
                     "error": "active render cancellation not supported in Phase 1"},
        )
    return {"job_id": job_id, "state": state}


@app.post("/api/render/jobs/{job_id}/retry")
def render_job_retry(job_id: str):
    """Re-queue a FAILED or PARTIAL_FAILURE job using its original request.
    Returns the NEW job id (the old one is kept for history).

    Hardening sprint P1.R2: only failed and partial_failure may be retried.
    Completed, queued, rendering and cancelled require a different action and
    are rejected with 409 / 400.
    """
    stored = _load_job(job_id)
    if not stored:
        raise HTTPException(status_code=404, detail="job not found")
    src_status = canonical_status(stored["status"])
    if src_status not in ("failed", "partial_failure"):
        # Not retryable. Report the actual state for the caller.
        raise HTTPException(status_code=409, detail=f"job is {src_status}; retry only allowed from failed or partial_failure")
    original = _load_job_request(job_id)
    if not original:
        raise HTTPException(status_code=400, detail="original request not available for retry")

    # Rebuild the request object from its JSON (works for v1 and v2).
    # Phase-2 correctness: use the same parse_render_request() so unsupported
    # versions stored in old rows are rejected explicitly.
    try:
        request = parse_render_request(original)
    except HTTPException:
        raise
    except Exception as e:  # noqa: BLE001
        raise HTTPException(status_code=400, detail=f"stored request invalid: {e}") from e

    mode = getattr(request, "mode", "final") or "final"
    episode_id = getattr(request, "episode_id", "") or ""
    request_id = getattr(request, "request_id", "") or stored.get("request_id") or ""
    req_json = request.model_dump_json() if hasattr(request, "model_dump_json") else ""
    # C3: /retry is a production caller of the single atomic allocator.
    reservation = reserve_attempt(
        source_job_id=job_id,
        request_id=request_id,
        request_json=req_json,
        mode=mode,
        episode_id=episode_id,
        reason="retry",
    )
    if not reservation.created:
        return {
            "job_id": reservation.job_id,
            "original_job_id": job_id,
            "state": (_load_job(reservation.job_id) or {}).get("status", "queued"),
            "attempt": reservation.attempt,
            "idempotent_hit": True,
        }
    child_id = reservation.job_id
    with _async_jobs_lock:
        _async_jobs[child_id] = {
            "state": "queued", "response": None, "error": None,
            "request_id": request_id, "episode_id": episode_id, "mode": mode,
            "parent_job_id": reservation.parent_job_id,
            "attempt": reservation.attempt,
        }

    try:
        _enqueue_job(child_id)
    except QueueAdmissionError as exc:
        raise HTTPException(status_code=503, detail="Render queue is full") from exc
    return {
        "job_id": child_id, "original_job_id": job_id,
        "state": "queued", "attempt": reservation.attempt,
    }


@app.post("/api/render", response_model=RenderResponse)
def render(request: Dict[str, Any]):
    """Render clips synchronously (v1 or v2 contract). Long videos download
    first — poll client-side.

    Serialized by a process-wide lock so concurrent render requests never run
    in parallel (downloads of multi-hundred-MB sources and OpenCV encodes are
    CPU/disk heavy; parallelism just slows everything down and crashes).

    Phase-2 correctness: parsed through parse_render_request() like async and
    retry — V2 is never mis-parsed as V1 and unsupported contract versions
    fail explicitly. Terminal status persisted here once from RenderOutcome.
    """
    global _render_busy
    request = parse_render_request(request)
    # Phase 1 §5.1: synchronous rendering follows the same identity rule —
    # the API creates job_id exactly once, _render never generates its own.
    job_id = uuid.uuid4().hex[:10]
    mode = getattr(request, "mode", "final") or "final"
    episode_id = getattr(request, "episode_id", "") or ""
    request_json = request.model_dump_json() if hasattr(request, "model_dump_json") else ""
    request_id = getattr(request, "request_id", "") or ""
    force_rerender = bool(getattr(request, "force_rerender", False))
    # Normal sync submission is idempotent for completed/active requests.
    # Terminal failed/cancelled/partial states proceed to the allocator as a
    # documented new resubmission attempt.
    if request_id and not force_rerender:
        existing = get_existing_request_result(request_id)
        if existing and existing.get("state") not in ("failed", "cancelled", "partial_failure"):
            existing_job = existing["job_id"]
            with _async_jobs_lock:
                existing_mem = _async_jobs.get(existing_job, {})
                existing_response = existing_mem.get("response")
            if isinstance(existing_response, RenderResponse):
                return existing_response
            durable_existing = _load_job(existing_job)
            if durable_existing and durable_existing.get("response"):
                import json as _json
                raw_existing = durable_existing["response"]
                return RenderResponse(**(raw_existing if isinstance(raw_existing, dict) else _json.loads(raw_existing)))
            raise HTTPException(
                status_code=409,
                detail=f"Job {existing_job} is still active; use GET /api/render/status/{existing_job}",
            )
    # C3: synchronous force/resubmit also uses the one atomic allocator.
    reservation = None
    if request_id:
        source_job_id = None
        if force_rerender:
            latest = get_existing_request_result(request_id)
            source_job_id = latest["job_id"] if latest else None
        reservation = reserve_attempt(
            source_job_id=source_job_id,
            request_id=request_id,
            request_json=request_json,
            mode=mode,
            episode_id=episode_id,
            reason="force" if force_rerender else "resubmit",
            preferred_job_id=job_id,
        )
    else:
        reserved = _reserve_job(request_id, job_id, mode=mode, episode_id=episode_id,
                                request_json=request_json, force=False)
        reservation = AttemptReservation(
            job_id=reserved, attempt=1, parent_job_id=None,
            created=(reserved == job_id),
            existing_winner_job_id=(None if reserved == job_id else reserved),
            reason="resubmit",
        )
    job_id = reservation.job_id
    if not reservation.created:
        # Existing completed result survives process-memory loss.
        with _async_jobs_lock:
            existing = _async_jobs.get(job_id, {})
            stored_resp = existing.get("response")
        if stored_resp and isinstance(stored_resp, RenderResponse):
            return stored_resp
        durable = _load_job(job_id)
        if durable and durable.get("response"):
            try:
                import json as _json
                raw = durable["response"]
                return RenderResponse(**(raw if isinstance(raw, dict) else _json.loads(raw)))
            except Exception as exc:  # noqa: BLE001
                raise PersistenceError(f"stored response is invalid for {job_id}: {exc}") from exc
        raise HTTPException(
            status_code=409,
            detail=f"Job {job_id} is still active; use GET /api/render/status/{job_id}",
        )
    with _async_jobs_lock:
        _async_jobs[job_id] = {
            "state": "queued", "response": None, "error": None,
            "request_id": request_id, "mode": mode, "episode_id": episode_id,
            "attempt": reservation.attempt, "parent_job_id": reservation.parent_job_id,
        }
    # Block until the current job finishes (true FIFO queue) — a 503 timeout
    # would just push the error back to the client, not serialize the work.
    _render_lock.acquire()
    _render_busy = True
    outcome = None
    sync_error = None
    try:
        # P0.1: sync renderer also wins through queued -> downloading CAS.
        # Brief v5 R-01: check the return value; a lost CAS must NOT render.
        won = transition_job(job_id, "queued", "downloading", mode=mode, episode_id=episode_id)
        if not won:
            raise JobTransitionConflict(
                f"sync job {job_id}: queued->downloading CAS lost "
                f"(expected queued, got {_load_job(job_id) and _load_job(job_id).get('status')})"
            )
        outcome = _render(request, job_id)
    except Exception as e:  # noqa: BLE001
        # Brief v5 R-01/R-05: persist terminal 'failed' THROUGH the canonical
        # transition when possible; never silently claim failure in memory.
        sync_error = e
        print(f"[render] sync job {job_id} failed: {e}", flush=True)
        try:
            # queued/active -> failed via canonical transition.
            cur = _load_job(job_id)
            src = canonical_status(cur["status"]) if cur else "queued"
            transition_job(job_id, src, "failed", mode=mode, episode_id=episode_id, error=str(e),
                           error_stage="render")
        except Exception as e2:  # noqa: BLE001
            _record_db_error("sync_persist_failed", e2)
        raise
    finally:
        _render_busy = False
        _render_lock.release()
    # Persist terminal THROUGH the state machine (rendering -> quality_check ->
    # completed/partial_failure); only then update memory (R-05 order).
    try:
        # Brief v6 4.3/R03: a False terminal commit is a correctness conflict —
        # no success response, no memory completed.
        persisted = _persist_terminal_via_transition(
            job_id,
            outcome.final_status,
            mode=mode,
            episode_id=episode_id,
            response=outcome.response.model_dump_json() if hasattr(outcome.response, "model_dump_json") else "",
        )
        if not persisted:
            raise JobTransitionConflict(
                f"sync job {job_id}: terminal transition to {outcome.final_status} was not committed"
            )
    except Exception as e:  # noqa: BLE001
        # C4: sync and worker paths share durable-first mirroring. Memory must
        # retain lineage/metadata and must never invent a terminal state.
        _record_db_error("sync_terminal_persist", e)
        mirror_durable_after_failure(job_id, f"persist: {e}")
        raise
    with _async_jobs_lock:
        _async_jobs[job_id] = {"state": outcome.final_status, "response": outcome.response,
                               "error": None, "request_id": request_id, "mode": mode}
    return outcome.response


if __name__ == "__main__":
    import uvicorn

    RENDER_ROOT.mkdir(parents=True, exist_ok=True)
    print(f"[render] output root: {RENDER_ROOT}", flush=True)
    # Brief v5 4.6: reconcile stale jobs from a previous process BEFORE the
    # server accepts requests — never wait for a client GET.
    try:
        n = _reconcile_startup_orphans()
        print(f"[render] startup reconcile: {n} job(s) orphaned", flush=True)
    except Exception as e:  # noqa: BLE001
        print(f"[render] startup reconcile failed: {e}", flush=True)
    uvicorn.run(app, host=HOST, port=PORT, log_level="info")
